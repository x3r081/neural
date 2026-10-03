#!/usr/bin/env python3
"""Bitwise synthetic parity gate for the portable GPT-OSS CPU DLL.

No model/checkpoint is loaded. `--quick` checks one expert and one prefill pair
in raw layout; `--full` covers varied expert counts, raw/PS4 slots, capture,
multi-token dispatch, and masked pair combining against the shipped reference DLLs.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
H, GU, GK, RB = 2880, 5760, 90, 1440
OFF_DC = GU * RB
OFF_GS = OFF_DC + H * RB
RAW_BYTES = OFF_GS + (GU + H) * GK
PACKED_SR = 46
PACKED_BYTES = OFF_GS + (GU + H) * PACKED_SR
VP = ctypes.c_void_p
CASES: list[dict] = []


def ptr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def configure(lib):
    lib.gptoss_experts.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]
    lib.gptoss_experts.restype = None
    lib.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
    lib.gptoss_experts_cap.restype = None
    if hasattr(lib, "gptoss_experts_multi"):
        lib.gptoss_experts_multi.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]
        lib.gptoss_experts_multi.restype = None
    if hasattr(lib, "gptoss_multi_scratch_floats"):
        lib.gptoss_multi_scratch_floats.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.gptoss_multi_scratch_floats.restype = ctypes.c_int
    if hasattr(lib, "gptoss_combine_pairs"):
        lib.gptoss_combine_pairs.argtypes = [ctypes.c_int, ctypes.c_int, VP, VP, VP, VP, ctypes.c_int]
        lib.gptoss_combine_pairs.restype = None
    if hasattr(lib, "gptoss_multi_activation"):
        lib.gptoss_multi_activation.argtypes = [ctypes.c_int, VP, VP, ctypes.c_int]
        lib.gptoss_multi_activation.restype = None
    if hasattr(lib, "gptoss_multi_check_exp"):
        lib.gptoss_multi_check_exp.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_longlong)]
        lib.gptoss_multi_check_exp.restype = ctypes.c_longlong
    if hasattr(lib, "gptoss_multi_get_config"):
        lib.gptoss_multi_get_config.argtypes = [VP]
        lib.gptoss_multi_get_config.restype = None
    lib.gptoss_set_scale_layout.argtypes = [ctypes.c_int]
    lib.gptoss_set_scale_layout.restype = ctypes.c_int
    lib.gptoss_get_scale_layout.restype = ctypes.c_int


def pack_slot(raw: np.ndarray) -> np.ndarray:
    out = np.empty(PACKED_BYTES, np.uint8)
    out[:OFF_GS] = raw[:OFF_GS]
    src = raw[OFF_GS:].reshape(GU + H, GK)
    dst = out[OFF_GS:].reshape(GU + H, PACKED_SR)
    for i, row in enumerate(src):
        base = int(row.min())
        delta = row.astype(np.int16) - base
        if int(delta.max()) > 15:
            raise AssertionError("bad synthetic scale span")
        dst[i, 0] = base
        dst[i, 1:] = delta[0::2].astype(np.uint8) | (delta[1::2].astype(np.uint8) << 4)
    return out


def make_experts(count: int, seed: int, layout: int):
    rng = np.random.default_rng(seed)
    raw = []
    slots = []
    for _ in range(count):
        b = np.empty(RAW_BYTES, np.uint8)
        b[:OFF_GS] = rng.integers(0, 256, OFF_GS, dtype=np.uint8)
        scales = rng.integers(116, 125, (GU + H, GK), dtype=np.uint8)
        b[OFF_GS:] = scales.reshape(-1)
        raw.append(b)
        slots.append(pack_slot(b) if layout else b)
    bgu = [rng.normal(0, .02, GU).astype(np.float32) for _ in range(count)]
    bdn = [rng.normal(0, .02, H).astype(np.float32) for _ in range(count)]
    return raw, slots, bgu, bdn


def same_bits(a: np.ndarray, b: np.ndarray, label: str):
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a.view(np.uint8), b.view(np.uint8)):
        bad = np.flatnonzero(a.view(np.uint32).reshape(-1) != b.view(np.uint32).reshape(-1))[:8]
        details = [(int(i), float(a.reshape(-1)[i]), float(b.reshape(-1)[i]),
                    hex(int(a.view(np.uint32).reshape(-1)[i])), hex(int(b.view(np.uint32).reshape(-1)[i]))) for i in bad]
        raise AssertionError(f"{label}: bit mismatch {details}")


def run_single(ref, cand, ecount: int, layout: int, seed: int, capture: bool):
    raw, slots, bg, bd = make_experts(ecount, seed, layout)
    bgp = (VP * ecount)(*[ptr(x) for x in bg])
    bdp = (VP * ecount)(*[ptr(x) for x in bd])
    slp = (VP * ecount)(*[ptr(x) for x in slots])
    x = np.random.default_rng(seed + 70).normal(0, .1, H).astype(np.float32)
    w = np.random.default_rng(seed + 71).normal(0, .2, ecount).astype(np.float32)
    # Native single-token scratch contract; deliberately overallocate modestly.
    n_scratch = ecount * (GU + 3*H) + H + 128
    scr0 = np.zeros(n_scratch, np.float32)
    scr1 = np.zeros(n_scratch, np.float32)
    scr8 = np.zeros(n_scratch, np.float32)
    y0 = np.zeros(H, np.float32); y1 = np.zeros(H, np.float32)
    y8 = np.zeros(H, np.float32)
    cap0 = [np.zeros_like(slot) if capture and i % 2 == 0 else None for i, slot in enumerate(slots)]
    cap1 = [np.zeros_like(slot) if capture and i % 2 == 0 else None for i, slot in enumerate(slots)]
    cap8 = [np.zeros_like(slot) if capture and i % 2 == 0 else None for i, slot in enumerate(slots)]
    cp0 = (VP * ecount)(*[ptr(x) if x is not None else None for x in cap0])
    cp1 = (VP * ecount)(*[ptr(x) if x is not None else None for x in cap1])
    cp8 = (VP * ecount)(*[ptr(x) if x is not None else None for x in cap8])
    for lib in (ref, cand):
        if lib.gptoss_set_scale_layout(layout) != 0 or lib.gptoss_get_scale_layout() != layout:
            raise AssertionError("scale-layout API failed")
    ref.gptoss_experts_cap(ecount, slp, ptr(x), bgp, bdp, ptr(w), ptr(y0), ptr(scr0), 1, cp0)
    cand.gptoss_experts_cap(ecount, slp, ptr(x), bgp, bdp, ptr(w), ptr(y1), ptr(scr1), 1, cp1)
    cand.gptoss_experts_cap(ecount, slp, ptr(x), bgp, bdp, ptr(w), ptr(y8), ptr(scr8), 8, cp8)
    same_bits(scr0[:H], scr1[:H], f"deinterleaved input E={ecount} layout={layout}")
    same_bits(scr0[H:H+ecount*GU], scr1[H:H+ecount*GU], f"gate/up E={ecount} layout={layout}")
    h0 = H + ecount*GU
    same_bits(scr0[h0:h0+ecount*H], scr1[h0:h0+ecount*H], f"activation E={ecount} layout={layout}")
    # experts_impl scratch stores even and odd activation halves contiguously,
    # together occupying E*H floats (not two separate E*H regions).
    yoff = h0 + ecount*H
    same_bits(scr0[yoff:yoff+ecount*H], scr1[yoff:yoff+ecount*H], f"down E={ecount} layout={layout}")
    same_bits(scr0[:H], scr8[:H], f"deinterleaved input E={ecount} layout={layout} threads=8")
    same_bits(scr0[H:H+ecount*GU], scr8[H:H+ecount*GU], f"gate/up E={ecount} layout={layout} threads=8")
    same_bits(scr0[h0:h0+ecount*H], scr8[h0:h0+ecount*H], f"activation E={ecount} layout={layout} threads=8")
    same_bits(scr0[yoff:yoff+ecount*H], scr8[yoff:yoff+ecount*H], f"down E={ecount} layout={layout} threads=8")
    same_bits(y0, y1, f"single E={ecount} layout={layout}")
    same_bits(y0, y8, f"single E={ecount} layout={layout} threads=8")
    if capture:
        for i in range(ecount):
            if cap0[i] is None: continue
            same_bits(cap0[i], cap1[i], f"capture E={ecount} expert={i} layout={layout}")
            same_bits(cap0[i], cap8[i], f"capture E={ecount} expert={i} layout={layout} threads=8")
            same_bits(cap1[i], slots[i], f"capture preserves source bytes expert={i}")
    CASES.append({"operation": "single", "layout": "ps4" if layout else "raw",
                  "experts": ecount, "capture_experts": [i for i in range(ecount) if capture and i % 2 == 0],
                  "threads_tested": [1, 8], "status": "PASS", "bitwise_mismatches": 0})
    print(f"PASS single E={ecount} layout={layout} capture={capture}", flush=True)


def run_multi(ref, cand, ecount: int, layout: int, seed: int):
    _, slots, bg, bd = make_experts(ecount, seed, layout)
    counts = [1 + (i % 2) for i in range(ecount)]
    off = np.zeros(ecount + 1, np.int32)
    for i, c in enumerate(counts): off[i+1] = off[i] + c
    pcount = int(off[-1])
    rng = np.random.default_rng(seed + 80)
    x = rng.normal(0, .1, pcount*H).astype(np.float32)
    w = rng.normal(0, .2, pcount).astype(np.float32)
    y0 = np.zeros(pcount*H, np.float32); y1 = np.zeros_like(y0)
    y8 = np.zeros_like(y0)
    ns = int(cand.gptoss_multi_scratch_floats(ecount, pcount))
    if ns != pcount*(H+GU)+16: raise AssertionError(f"wrong scratch size {ns}")
    s0 = np.zeros(ns, np.float32); s1 = np.zeros(ns, np.float32)
    slp = (VP * ecount)(*[ptr(z) for z in slots])
    bgp = (VP * ecount)(*[ptr(z) for z in bg]); bdp = (VP * ecount)(*[ptr(z) for z in bd])
    for lib in (ref, cand):
        if lib.gptoss_set_scale_layout(layout) != 0: raise AssertionError("scale-layout set failed")
    ref.gptoss_experts_multi(ecount, slp, ptr(off), ptr(x), bgp, bdp, ptr(w), ptr(y0), ptr(s0), 1)
    cand.gptoss_experts_multi(ecount, slp, ptr(off), ptr(x), bgp, bdp, ptr(w), ptr(y1), ptr(s1), 1)
    cand.gptoss_experts_multi(ecount, slp, ptr(off), ptr(x), bgp, bdp, ptr(w), ptr(y8), ptr(s1), 8)
    same_bits(y0, y1, f"multi E={ecount} P={pcount} layout={layout}")
    same_bits(y0, y8, f"multi E={ecount} P={pcount} layout={layout} threads=8")
    CASES.append({"operation": "multi", "layout": "ps4" if layout else "raw",
                  "experts": ecount, "pairs": pcount, "threads_tested": [1, 8],
                  "status": "PASS", "bitwise_mismatches": 0})
    print(f"PASS multi E={ecount} P={pcount} layout={layout}", flush=True)


def run_combine(ref, cand):
    rng = np.random.default_rng(993)
    n, k, p = 3, 9, 9
    idx = np.full((n, k), -1, np.int32)
    for t, active in enumerate((0, 3, 8)):
        idx[t, :active] = np.arange(p, p+active, dtype=np.int32)
    w = rng.normal(0, .2, n*k).astype(np.float32)
    yr = rng.normal(0, .1, (p+n*k, H)).astype(np.float32)
    out0 = np.zeros(n*H, np.float32)
    out1 = np.zeros_like(out0)
    ref.gptoss_combine_pairs(n, k, ptr(idx), ptr(w), ptr(yr), ptr(out0), 1)
    cand.gptoss_combine_pairs(n, k, ptr(idx), ptr(w), ptr(yr), ptr(out1), 1)
    same_bits(out0, out1, "masked combine")
    CASES.append({"operation": "combine", "layout": "n/a", "tokens": n,
                  "experts_per_token": [0, 3, 8], "status": "PASS", "bitwise_mismatches": 0})
    print("PASS masked combine", flush=True)


def run_aux_abi(ref, cand):
    cfg = (ctypes.c_int * 4)()
    cand.gptoss_multi_get_config(cfg)
    if list(cfg) != [16, 8, 1, 64]: raise AssertionError(f"unexpected default multi config {list(cfg)}")
    bits, fallback = ctypes.c_uint(), ctypes.c_longlong()
    mismatches = cand.gptoss_multi_check_exp(0, 256, 1, ctypes.byref(bits), ctypes.byref(fallback))
    if mismatches != 0 or bits.value != 0xFFFFFFFF or fallback.value != 256:
        raise AssertionError("portable scalar exp check contract failed")
    gu = np.random.default_rng(199).normal(0, .4, GU).astype(np.float32)
    a = np.empty(H, np.float32); b = np.empty(H, np.float32)
    ref.gptoss_multi_activation(1, ptr(gu), ptr(a), 0)
    cand.gptoss_multi_activation(1, ptr(gu), ptr(b), 1)
    same_bits(a, b, "activation ABI")
    CASES.append({"operation": "auxiliary_abi", "status": "PASS", "bitwise_mismatches": 0,
                  "default_multi_config": list(cfg), "exp_check_lanes": 256, "scalar_exp_lanes": fallback.value})
    print("PASS auxiliary prefill ABI", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--portable", default=str(ROOT / ".tools" / "gptoss_cpu_portable_avx2.dll"))
    ap.add_argument("--reference-single", default=str(ROOT / "gptoss_cpu_cap2.dll"))
    ap.add_argument("--reference-multi", default=str(ROOT / "gptoss_cpu_multi.dll"))
    ap.add_argument("--quick", action="store_true", help="one expert, one pair, raw layout only")
    ap.add_argument("--full", action="store_true", help="all planned ABI cases (CPU intensive)")
    ap.add_argument("--compiler", help="the GCC executable used for the candidate build (reported, not invoked)")
    ap.add_argument("--build-target", choices=("avx2-fma", "scalar"), default="avx2-fma",
                    help="target selection used by build_portable_cpu.py")
    ap.add_argument("--report", default=str(ROOT / "artifacts" / "gptoss_portable_validation.json"),
                    help="JSON evidence report (created only when the validator runs)")
    args = ap.parse_args()
    if args.quick == args.full:
        ap.error("choose exactly one of --quick or --full")
    report_path = Path(args.report).resolve()
    report = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator": str(Path(__file__).resolve()),
        "mode": "quick" if args.quick else "full",
        "result": "FAIL",
        "full_validation": bool(args.full),
        "full_validation_passed": False,
        "portable_dll_sha256": None,
        "target_isa": ("scalar row-dot" if args.build_target == "scalar"
                       else "AVX2+FMA row-dot with scalar fallback"),
        "build_target": "avx2-fma" if args.build_target == "avx2-fma" else "x86-64-baseline",
        "build_flags": ["-O3", "-shared", "-ffp-contract=off", "-fopenmp", "-static"] +
                       (["-mavx2", "-mfma"] if args.build_target == "avx2-fma" else []),
        "openmp_transient_regions": True,
        "persistent_worker_pool": False,
        "thread_counts_tested": [1, 8],
        "minimum_cpu_features": ["AVX2", "FMA3"] if args.build_target == "avx2-fma" else [],
        "cpu_signature_informational": "|".join((platform.system(), platform.machine(), platform.processor().strip())),
        "cpu_features_informational": {},
        "python": sys.version,
        "numpy": np.__version__,
        "dlls": {},
        "reference_provenance": "The existing Neural GPT-OSS CPU cap2 and multi DLLs in this repository; no model weights loaded.",
        "symbols_required": ["gptoss_set_scale_layout", "gptoss_get_scale_layout", "gptoss_slot_bytes",
                             "gptoss_experts", "gptoss_experts_cap", "gptoss_experts_multi",
                             "gptoss_multi_scratch_floats", "gptoss_combine_pairs", "gptoss_multi_set_tiling",
                             "gptoss_multi_set_schedule", "gptoss_multi_get_config", "gptoss_multi_activation",
                             "gptoss_multi_check_exp"],
        "abi_contracts": {"raw_layout": RAW_BYTES, "ps4_layout": PACKED_BYTES,
                           "scratch_formula_floats": "P*(H+GU)+16", "persistent_worker_pool": False},
        "cases": CASES,
        "error": None,
    }
    try:
        try:
            from numpy._core._multiarray_umath import __cpu_features__
            report["cpu_features_informational"] = {k: bool(v) for k, v in __cpu_features__.items()
                                                     if k in ("AVX2", "FMA3", "AVX512F", "AVX512BW", "AVX512VL", "AVX512DQ")}
        except (ImportError, AttributeError):
            pass
        required_reference = ("AVX2", "FMA3", "AVX512F", "AVX512BW", "AVX512VL", "AVX512DQ")
        if not all(report["cpu_features_informational"].get(flag, False) for flag in required_reference):
            raise RuntimeError("parity validation loads the AVX-512 reference kernels; run it on a compatible validation host, "
                               "not an AVX2-only deployment machine")
        known = json.loads((ROOT / "artifacts/gptoss_known_build.json").read_text(encoding="utf-8"))
        if known["cpu_signature"] != report["cpu_signature_informational"]:
            raise RuntimeError("the bundled reference kernels are validated for a different CPU signature")
        portable = ctypes.CDLL(str(Path(args.portable).resolve()))
        ref_single = ctypes.CDLL(str(Path(args.reference_single).resolve()))
        ref_multi = ctypes.CDLL(str(Path(args.reference_multi).resolve()))
        for lib in (portable, ref_single, ref_multi): configure(lib)
        missing_symbols = [name for name in report["symbols_required"] if not hasattr(portable, name)]
        if missing_symbols:
            raise AssertionError("portable DLL lacks ABI symbols: " + ", ".join(missing_symbols))
        slot_bytes = [int(portable.gptoss_slot_bytes(i)) for i in (0, 1)]
        if slot_bytes != [RAW_BYTES, PACKED_BYTES]:
            raise AssertionError(f"slot size API mismatch: {slot_bytes}")
        report["abi_contracts"]["slot_bytes_api"] = slot_bytes
        cc = Path(args.compiler).resolve() if args.compiler else Path(
            os.environ.get("NEURAL_DEVKIT", r"F:\AI\Neural\third_party\tools\w64devkit\bin")) / "gcc.exe"
        if cc.is_file():
            env = os.environ.copy(); env["PATH"] = str(cc.parent) + os.pathsep + env.get("PATH", "")
            version = subprocess.run([str(cc), "-dumpfullversion"], capture_output=True, text=True, env=env, check=True).stdout.strip()
            report["compiler"] = {"path": str(cc), "version": version}
        source = ROOT / "kernels" / "gptoss_cpu_portable.c"
        report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        dll_paths = {"portable": Path(args.portable).resolve(), "reference_single": Path(args.reference_single).resolve(),
                     "reference_multi": Path(args.reference_multi).resolve()}
        for name, path in dll_paths.items():
            report["dlls"][name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report["portable_dll_sha256"] = report["dlls"]["portable"]["sha256"]
        if args.quick:
            run_single(ref_single, portable, 1, 0, 40, True)
            run_multi(ref_multi, portable, 1, 0, 41)
            run_aux_abi(ref_multi, portable)
        else:
            for layout in (0, 1):
                for ecount in (1, 2, 3, 4, 5, 8):
                    run_single(ref_single, portable, ecount, layout, 100 + ecount + 10*layout, ecount in (1, 4))
                for ecount in (1, 2, 4):
                    run_multi(ref_multi, portable, ecount, layout, 200 + ecount + 10*layout)
            run_combine(ref_multi, portable)
            run_aux_abi(ref_multi, portable)
        report["result"] = "PASS"
        report["full_validation_passed"] = bool(args.full)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Validation report: {report_path}", flush=True)
    print("PORTABLE CPU PARITY PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
