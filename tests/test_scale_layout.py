"""Packed-scale expert layout ("mxfp4_g32_ps4"): CPU-only tests (no GPU, no model, no real store).

  * layout arithmetic and the store descriptors (raw descriptors unchanged; packed one distinct),
  * ExpertLayout <-> store selection (layout_for_store) and the strict open() identity check,
  * mxfp4_pool_views: where the raw / packed scale regions sit in a slot; the Triton launchers' layout inference,
  * ExpertArena chunk size follows the slot size,
  * cpu_prefill.bind_scale_layout and the required-layout module helpers (a packed store is never read as raw by omission),
  * tools/pack_store.py on a tiny synthetic store: round trip, refusal of a span-16 row, source untouched,
  * the tracked CPU kernels (gptoss_cpu_cap2.dll decode + capture, gptoss_cpu_multi.dll prefill): packed == raw bit for bit.
    The DLLs are build artifacts checked into the repo: a missing or unloadable one FAILS these tests.

    python -m pytest tests\\test_scale_layout.py
"""
import ctypes
import dataclasses
import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import paths as NP  # noqa: E402
import scale_pack as sp  # noqa: E402
from neural.moe.layout import (GPTOSS_LAYOUT, GPTOSS_LAYOUT_PS4, Q80_LAYOUT, gptoss_layout_for_repr,  # noqa: E402
                               mxfp4_pool_views, prepacked_descriptor)

VP = ctypes.c_void_p
H, GU = sp.H, sp.GU

# the descriptor of a raw gpt-oss store (tools/build_store.py writes it), minus the informational `row` label
RAW_STORE_DESCRIPTOR = {"slot_bytes": 13219200, "n_layers": 36, "n_experts": 128, "hidden": 2880, "gate_up_n": 5760,
                        "down_n": 2880, "down_k": 2880, "int4_group": 32, "inner_k_tiles": 2,
                        "kernel": "neural.q80.mxfp4_kernels (Triton, fp32-accum)", "weight_repr": "mxfp4_g32"}


def test_layout_arithmetic():
    assert GPTOSS_LAYOUT.slot_bytes == 13_219_200 == sp.SLOT_RAW
    assert GPTOSS_LAYOUT_PS4.slot_bytes == 12_839_040 == sp.SLOT_PACKED
    assert GPTOSS_LAYOUT_PS4.gate_saz_bytes == 5760 * 46 and GPTOSS_LAYOUT_PS4.down_saz_bytes == 2880 * 46
    assert GPTOSS_LAYOUT.gate_pack_bytes == GPTOSS_LAYOUT_PS4.gate_pack_bytes == 5760 * 1440
    assert (GPTOSS_LAYOUT.scale_layout_mode, GPTOSS_LAYOUT_PS4.scale_layout_mode) == (0, 1)
    assert GPTOSS_LAYOUT.is_mxfp4 and GPTOSS_LAYOUT_PS4.is_mxfp4 and not Q80_LAYOUT.is_mxfp4
    assert gptoss_layout_for_repr("mxfp4_g32") is GPTOSS_LAYOUT and gptoss_layout_for_repr("mxfp4_g32_ps4") is GPTOSS_LAYOUT_PS4
    assert round(100 * (1 - 12_839_040 / 13_219_200), 2) == 2.88
    with pytest.raises(KeyError):
        gptoss_layout_for_repr("int4_g32")


def test_descriptors():
    raw = prepacked_descriptor(GPTOSS_LAYOUT)
    assert {k: v for k, v in raw.items() if k != "row"} == RAW_STORE_DESCRIPTOR            # raw store: byte-compatible
    ps4 = prepacked_descriptor(GPTOSS_LAYOUT_PS4)
    assert ps4["weight_repr"] == "mxfp4_g32_ps4" and ps4["slot_bytes"] == 12_839_040
    assert ps4["kernel"] != raw["kernel"]
    diff = {k for k in set(raw) | set(ps4) if raw.get(k) != ps4.get(k)}
    assert {"slot_bytes", "weight_repr", "kernel"} <= diff                                  # what makes open() refuse the wrong one
    assert prepacked_descriptor(Q80_LAYOUT).get("weight_repr") is None                      # int4 stores carry no weight_repr key


def _pools(cap=3, seed=1):
    import torch

    rng = np.random.default_rng(seed)
    raws = [sp.synthetic_slot(rng) for _ in range(cap)]
    pks = [sp.pack_slot(r) for r in raws]
    return torch.from_numpy(np.stack(raws)), torch.from_numpy(np.stack(pks))


def test_pool_views_place_the_scales():
    import torch

    raw_pool, pk_pool = _pools()
    gb, db, gs, ds = mxfp4_pool_views(raw_pool, GPTOSS_LAYOUT)
    gb2, db2, gs2, ds2 = mxfp4_pool_views(pk_pool, GPTOSS_LAYOUT_PS4)
    assert gs.shape == (3, 5760, 90) and ds.shape == (3, 2880, 90)
    assert gs2.shape == (3, 5760, 46) and ds2.shape == (3, 2880, 46)
    assert gb.shape == gb2.shape == (3, 5760, 90, 16) and db.shape == db2.shape == (3, 2880, 90, 16)
    assert gs.stride(0) == 13_219_200 and gs2.stride(0) == 12_839_040 and gs2.stride(1) == 46 and gs2.stride(2) == 1
    for i in range(3):
        assert torch.equal(gb[i], gb2[i]) and torch.equal(db[i], db2[i])                   # codes are byte-identical
        assert np.array_equal(sp.unpack_scale_rows(gs2[i].numpy()), gs[i].numpy())
        assert np.array_equal(sp.unpack_scale_rows(ds2[i].numpy()), ds[i].numpy())
    with pytest.raises(AssertionError):                                                     # a raw pool is not a packed pool
        mxfp4_pool_views(raw_pool, GPTOSS_LAYOUT_PS4)


def test_triton_launchers_infer_the_layout_from_the_view():
    """_scale_layout is the pure-Python gate in front of the Triton launches (no GPU, no compile)."""
    import torch

    pytest.importorskip("triton")
    from neural.q80.mxfp4_kernels import _scale_layout

    raw_pool, pk_pool = _pools(cap=2)
    _, _, gs, ds = mxfp4_pool_views(raw_pool, GPTOSS_LAYOUT)
    _, _, gs2, ds2 = mxfp4_pool_views(pk_pool, GPTOSS_LAYOUT_PS4)
    GK = 90
    assert _scale_layout(gs, GK, None) is False and _scale_layout(ds, GK, None) is False
    assert _scale_layout(gs2, GK, None) is True and _scale_layout(ds2, GK, None) is True
    assert _scale_layout(gs, GK, False) is False and _scale_layout(gs2, GK, True) is True
    with pytest.raises(ValueError):                                                         # an explicit flag never overrides the view
        _scale_layout(gs, GK, True)
    with pytest.raises(ValueError):
        _scale_layout(gs2, GK, False)
    with pytest.raises(ValueError):                                                         # neither 90 nor 46 wide
        _scale_layout(torch.zeros(2, 8, 47, dtype=torch.uint8), GK, None)
    with pytest.raises(ValueError):                                                         # rows must be contiguous
        _scale_layout(torch.zeros(2, 8, 180, dtype=torch.uint8)[:, :, ::2], GK, None)


def test_arena_chunk_follows_the_slot_size():
    import arena

    for slot, rows in ((13_219_200, 162), (12_839_040, 167)):
        a = arena.ExpertArena("pinned", slot)
        assert a.rows_per_chunk == rows and a.slot_bytes == slot
        assert rows * slot < 2**31 <= (rows + 1) * slot                                     # the most rows that stay under 2 GiB
    assert arena.ExpertArena("pinned", 12_839_040, rows_per_chunk=5).rows_per_chunk == 5    # an explicit value still wins


def _write_meta(root, layout, model_id="gpt-oss-120b-b5c939de", hashes=None):
    os.makedirs(root, exist_ok=True)
    meta = {"magic": "Q80-PREPACKED-SLOTS", "version": 1, "model_id": model_id, "layout": prepacked_descriptor(layout),
            "sha256_per_layer": hashes or {}, "built_unix": 1, "build_seconds": 1.0}
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f)


def test_layout_for_store(tmp_path):
    from neural.moe.gptoss_adapter import layout_for_store

    _write_meta(str(tmp_path / "raw"), GPTOSS_LAYOUT)
    _write_meta(str(tmp_path / "ps4"), GPTOSS_LAYOUT_PS4)
    assert layout_for_store(str(tmp_path / "raw")) is GPTOSS_LAYOUT
    assert layout_for_store(str(tmp_path / "ps4")) is GPTOSS_LAYOUT_PS4
    with pytest.raises(RuntimeError):
        layout_for_store(str(tmp_path / "missing"))
    bad = tmp_path / "bad"
    _write_meta(str(bad), Q80_LAYOUT)                                                       # not a gpt-oss store
    with pytest.raises(RuntimeError):
        layout_for_store(str(bad))
    lie = tmp_path / "lie"
    _write_meta(str(lie), GPTOSS_LAYOUT_PS4)
    meta = json.load(open(lie / "metadata.json"))
    meta["layout"]["slot_bytes"] = 13_219_200                                               # descriptor contradicts itself
    json.dump(meta, open(lie / "metadata.json", "w"))
    with pytest.raises(RuntimeError):
        layout_for_store(str(lie))


# ------------------------------------------------------------------ the kernel DLLs (tracked build artifacts)
def _load(name):
    p = os.path.join(ROOT, name)
    assert os.path.exists(p), f"{name} is missing: it is a tracked build artifact (rebuild with tools\\build_kernels.bat)"
    NP.add_devkit_dll_dir()
    try:
        return ctypes.CDLL(p)
    except OSError as e:
        pytest.fail(f"{name} cannot be loaded: {e} (needs the MinGW runtime on PATH or NEURAL_DEVKIT; tools\\build_kernels.bat)")


def test_bind_scale_layout():
    import cpu_prefill as cp

    old = types.SimpleNamespace()                                                           # a kernel build without the switch
    assert cp.bind_scale_layout(old, 0, sp.SLOT_RAW, "old") is False                        # raw store: fine, it reads raw
    with pytest.raises(RuntimeError, match="predates gptoss_set_scale_layout"):             # packed store: refused, never misread
        cp.bind_scale_layout(old, 1, sp.SLOT_PACKED, "old")
    with pytest.raises(RuntimeError, match="slots only"):
        cp.bind_scale_layout(old, 0, sp.SLOT_PACKED, "old")
    for dll in ("gptoss_cpu_cap2.dll", "gptoss_cpu_multi.dll"):
        lib = _load(dll)
        try:
            assert cp.bind_scale_layout(lib, 1, sp.SLOT_PACKED, dll) is True and lib.gptoss_get_scale_layout() == 1
            assert cp.bind_scale_layout(lib, 0, sp.SLOT_RAW, dll) is True and lib.gptoss_get_scale_layout() == 0
            with pytest.raises(RuntimeError, match="slot size"):                            # right mode, wrong store size
                cp.bind_scale_layout(lib, 1, sp.SLOT_RAW, dll)
            with pytest.raises(RuntimeError):                                               # unsupported mode
                cp.bind_scale_layout(lib, 2, None, dll)
        finally:
            lib.gptoss_set_scale_layout(0)


def test_module_helpers_require_the_layout(monkeypatch):
    import cpu_prefill as cp

    monkeypatch.setattr(cp, "_DEFAULT", None)
    with pytest.raises(TypeError):                                                          # no default: cannot be forgotten
        cp.default()
    with pytest.raises(TypeError):
        cp.experts_multi([], [0], None, None, [], [])
    dll = os.path.join(ROOT, "gptoss_cpu_multi.dll")
    _load("gptoss_cpu_multi.dll")
    M = cp.default(1, dll=dll, scale_layout=1)
    try:
        assert M.scale_layout() == 1 and M.slot_bytes() == sp.SLOT_PACKED
        assert cp.default(1, scale_layout=0) is M and M.scale_layout() == 0                  # every call restates the layout
    finally:
        M.set_scale_layout(0)


def _al(a):
    return (a.ctypes.data + 63) // 64 * 64


def test_cpu_decode_kernel_packed_equals_raw_including_capture():
    """gptoss_experts_cap, 1 thread: packed output == raw output bit for bit for every prefetch / pair / fuse setting,
    and the copy-on-compute capture row (46-byte scale rows) equals the packed slot; the raw capture equals the raw slot."""
    import cpu_prefill as cp

    lib = _load("gptoss_cpu_cap2.dll")
    lib.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
    lib.gptoss_set_tuning.argtypes = [ctypes.c_int] * 3
    lib.gptoss_set_fuse.argtypes = [ctypes.c_int]
    rng = np.random.default_rng(3)
    NEXP = 4
    raw = np.stack([sp.synthetic_slot(rng) for _ in range(NEXP)])
    pk = np.stack([sp.pack_slot(r) for r in raw])
    bgu = (rng.standard_normal((NEXP, GU)) * 0.05).astype(np.float32)
    bdn = (rng.standard_normal((NEXP, H)) * 0.05).astype(np.float32)
    x = (rng.standard_normal(H) * 0.8).astype(np.float32)
    scr = np.zeros(4 * (GU + 3 * H) + H + 64, np.float32)
    pools = {0: raw, 1: pk}

    def run(mode, sel, capture):
        cp.bind_scale_layout(lib, mode, pools[mode].shape[1], "gptoss_cpu_cap2.dll")
        E = len(sel)
        w = np.full(E, 1.0 / E, np.float32)
        out = np.zeros(H, np.float32)
        bufs = [np.zeros(pools[mode].shape[1] + 64, np.uint8) for _ in range(E)] if capture else None
        cap = (VP * E)(*([_al(b) for b in bufs] if capture else [None] * E))
        lib.gptoss_experts_cap(E, (VP * E)(*[pools[mode][i].ctypes.data for i in sel]), x.ctypes.data,
                               (VP * E)(*[bgu[i].ctypes.data for i in sel]), (VP * E)(*[bdn[i].ctypes.data for i in sel]),
                               w.ctypes.data, out.ctypes.data, scr.ctypes.data, 1, cap)
        rows = [b[_al(b) - b.ctypes.data:][:pools[mode].shape[1]] for b in bufs] if capture else None
        return out, rows

    checks = 0
    try:
        for sel in ([2], [3, 1, 0]):
            for pf, pair, fuse in ((0, 0, 0), (1024, 0, 0), (0, 1, 0), (1024, 1, 0), (0, 0, 1), (1024, 0, 1)):
                lib.gptoss_set_tuning(pf, pair, 0)
                lib.gptoss_set_fuse(fuse)
                o_raw, rows_raw = run(0, sel, True)
                o_pk, rows_pk = run(1, sel, True)
                assert np.isfinite(o_raw).all()
                assert np.array_equal(o_raw.view(np.uint32), o_pk.view(np.uint32)), (sel, pf, pair, fuse)
                for j, i in enumerate(sel):
                    assert np.array_equal(rows_raw[j], raw[i]) and np.array_equal(rows_pk[j], pk[i]), (sel, j)
                checks += 1
        assert checks == 12
        # negative control: one flipped delta nibble changes the packed result (the comparison can fail)
        lib.gptoss_set_tuning(0, 0, 0)
        lib.gptoss_set_fuse(0)
        o_raw, _ = run(0, [1], False)
        pk[1, sp.OFF_GS + 46 * 3000 + 20] ^= 0x04
        o_bad, _ = run(1, [1], False)
        assert not np.array_equal(o_raw.view(np.uint32), o_bad.view(np.uint32))
    finally:
        lib.gptoss_set_tuning(0, 0, 0)
        lib.gptoss_set_fuse(0)
        lib.gptoss_set_scale_layout(0)


def test_cpu_multi_kernel_packed_equals_raw():
    import cpu_prefill as cp

    _load("gptoss_cpu_multi.dll")
    M = cp.CpuMultiExperts(threads=1)
    assert M.has_scale_layout, "gptoss_cpu_multi.dll predates the scale-layout switch: rebuild with tools\\build_kernels.bat"
    rng = np.random.default_rng(2)
    raw = np.stack([sp.synthetic_slot(rng) for _ in range(3)])
    pk = np.stack([sp.pack_slot(r) for r in raw])
    assert (M.slot_bytes(0), M.slot_bytes(1)) == (sp.SLOT_RAW, sp.SLOT_PACKED)
    bgu = (rng.standard_normal((3, GU)) * 0.05).astype(np.float32)
    bdn = (rng.standard_normal((3, H)) * 0.05).astype(np.float32)
    off = np.array([0, 3, 3, 12], np.int32)                                                # 3 / 0 / 9 tokens: incl. an empty expert
    X = (rng.standard_normal((12, H)) * 0.8).astype(np.float32)
    W = rng.uniform(0.1, 0.9, 12).astype(np.float32)
    out = {}
    try:
        for mode, pool in ((0, raw), (1, pk)):
            M.set_scale_layout(mode)
            out[mode] = M.experts([pool[i].ctypes.data for i in range(3)], off, X, W,
                                  [bgu[i].ctypes.data for i in range(3)], [bdn[i].ctypes.data for i in range(3)])
    finally:
        M.set_scale_layout(0)
    assert np.array_equal(out[0].view(np.uint32), out[1].view(np.uint32))
    with pytest.raises(ValueError):
        M.set_scale_layout(2)


# ------------------------------------------------------------------ the converter
def _mini_raw_store(root, bad_span=False, layers=1, experts=2):
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(11)
    hashes = {}
    for li in range(layers):
        h = hashlib.sha256()
        with open(os.path.join(root, f"layer_{li}.slots"), "wb") as f:
            for e in range(experts):
                s = sp.synthetic_slot(rng)
                if bad_span and e == experts - 1:
                    sc = s[sp.OFF_GS:].reshape(sp.NROWS, sp.GK)
                    sc[9, :] = 100
                    sc[9, 3] = 116
                f.write(s.tobytes())
                h.update(s.tobytes())
        hashes[str(li)] = h.hexdigest()
    with open(os.path.join(root, "expert_biases.pt"), "wb") as f:
        f.write(b"bias-placeholder")
    lay = prepacked_descriptor(GPTOSS_LAYOUT)
    lay["n_layers"], lay["n_experts"] = layers, experts
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump({"magic": "Q80-PREPACKED-SLOTS", "version": 1, "model_id": "gpt-oss-120b-b5c939de", "layout": lay,
                   "sha256_per_layer": hashes, "built_unix": 1, "build_seconds": 1.0}, f)


def _tree(root):
    return {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest() for f in sorted(os.listdir(root))}


def _pack_main(*argv):
    import pack_store as ps

    old = sys.argv
    sys.argv = ["pack_store.py", *argv]
    try:
        return ps.main()
    finally:
        sys.argv = old


def test_pack_store_round_trip_and_strict_open(tmp_path):
    import pack_store as ps
    from neural.q80.prepacked_store import FORMAT_MAGIC, FORMAT_VERSION, Q80PrepackedStore

    assert (ps.FORMAT_MAGIC, ps.FORMAT_VERSION) == (FORMAT_MAGIC, FORMAT_VERSION)        # duplicated constants stay in sync
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    _mini_raw_store(src)
    before = _tree(src)
    assert _pack_main("--src", src, "--dst", dst, "--verify") == 0
    assert _tree(src) == before                                                            # the source is never modified
    assert not os.path.exists(os.path.join(dst, "pack_progress.json"))                     # bookkeeping removed once finished
    assert sorted(os.listdir(dst)) == ["expert_biases.pt", "layer_0.slots", "metadata.json"]
    meta = json.load(open(os.path.join(dst, "metadata.json")))
    assert meta["layout"]["weight_repr"] == "mxfp4_g32_ps4" and meta["layout"]["slot_bytes"] == 12_839_040
    a = np.fromfile(os.path.join(src, "layer_0.slots"), np.uint8).reshape(2, sp.SLOT_RAW)
    b = np.fromfile(os.path.join(dst, "layer_0.slots"), np.uint8).reshape(2, sp.SLOT_PACKED)
    for e in range(2):
        assert np.array_equal(sp.unpack_slot(b[e]), a[e]) and np.array_equal(sp.pack_slot(a[e]), b[e])
    # the runtime's store class: accepted with the packed layout, refused with the raw one, never coerced
    pk = dataclasses.replace(GPTOSS_LAYOUT_PS4, n_layers=1, n_experts=2)
    raw = dataclasses.replace(GPTOSS_LAYOUT, n_layers=1, n_experts=2)
    store = Q80PrepackedStore(dst, layout=pk)
    assert store.open(expect_model_id="gpt-oss-120b-b5c939de", shared=False)["status"] == "ok"
    assert (store.slot_bytes, store.weight_repr) == (12_839_040, "mxfp4_g32_ps4")
    st = Q80PrepackedStore(dst, layout=raw).open(expect_model_id="gpt-oss-120b-b5c939de", shared=False)
    assert st["status"] == "fail" and st["store_weight_repr"] == "mxfp4_g32_ps4" and st["runtime_weight_repr"] == "mxfp4_g32"
    st = Q80PrepackedStore(src, layout=pk).open(expect_model_id="gpt-oss-120b-b5c939de", shared=False)
    assert st["status"] == "fail" and st["store_weight_repr"] == "mxfp4_g32"                # and the raw store with the packed layout


def test_pack_store_refuses_span_16(tmp_path):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    _mini_raw_store(src, bad_span=True)
    assert _pack_main("--src", src, "--check-only") == 3
    assert _pack_main("--src", src, "--dst", dst) == 3
    assert not os.path.exists(os.path.join(dst, "metadata.json"))                          # an unfinished store cannot be opened


def test_pack_store_refuses_dst_inside_or_equal_to_src(tmp_path):
    src = str(tmp_path / "src")
    _mini_raw_store(src)
    before = _tree(src)
    assert _pack_main("--src", src, "--dst", src) == 2
    assert _pack_main("--src", src, "--dst", os.path.join(src, "inner")) == 2
    if os.name == "nt":                                                                    # Windows paths ignore case
        assert _pack_main("--src", src, "--dst", src.upper()) == 2
        assert _pack_main("--src", src, "--dst", os.path.join(src.lower(), "Inner")) == 2
    assert _tree(src) == before
