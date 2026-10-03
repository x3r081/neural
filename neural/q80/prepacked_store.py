"""Q80-20 — versioned prepacked GEMM-ready expert store.

Layout: one file per layer, ``layer_<li>.slots``, shape [n_experts, SLOT_BYTES]
uint8 — each row is EXACTLY the bytes the VRAM slot pool consumes
(gate_pack | down_pack | gate_saz | down_saz, tinygemm-prepacked via
``aten::_convert_weight_to_int4pack``). A retrieval is therefore ONE
contiguous read/memcpy; no runtime conversion, no reconstruction, no layout
work. Rows are deterministic (the convert is deterministic — Q80-9) and the
store is immutable after construction.

``metadata.json`` carries: format version, model identity, tensor-layout
descriptor (all constants the runtime asserts against), per-layer SHA-256 for
corruption detection, kernel-coupling note. ``open()`` validates identity,
version, sizes; ``verify_layer()`` re-hashes on demand. Incompatible stores
are rejected, never coerced.

Kernel coupling (disclosed): rows embed the tinygemm INNER_K_TILES=2 packing
for sm_86-class GPUs. The store format is generic (rows of GEMM-ready bytes
+ a layout descriptor); the ROW CONTENT is kernel-specific by design — a
different GEMM backend needs a re-pack pass, guarded by the descriptor.

Scale layout (gpt-oss): the descriptor's ``weight_repr`` is part of the store's
identity. "mxfp4_g32" rows hold one raw E8M0 byte per group; "mxfp4_g32_ps4"
rows (tools/pack_store.py) hold the same values as base byte + 4-bit deltas and
are 2.88% smaller. open() compares slot_bytes, kernel and weight_repr strictly,
so a raw store is never opened with the packed layout or vice versa; use
``Q80PrepackedStore.peek_layout()`` / ``neural.moe.gptoss_adapter.layout_for_store``
to learn which one a directory holds BEFORE choosing the layout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neural.q80.expert_runtime import (
    DOWN_K,
    DOWN_N,
    GATE_UP_N,
    HIDDEN,
    INNER_K_TILES,
    INT4_GROUP,
    N_LAYERS,
    SLOT_BYTES,
)

FORMAT_MAGIC = "Q80-PREPACKED-SLOTS"
FORMAT_VERSION = 1
N_EXPERTS = 512


def _layout_descriptor(layout=None) -> dict[str, Any]:
    if layout is None:
        return {"slot_bytes": SLOT_BYTES, "n_layers": N_LAYERS, "n_experts": N_EXPERTS,
                "hidden": HIDDEN, "gate_up_n": GATE_UP_N, "down_n": DOWN_N,
                "down_k": DOWN_K, "int4_group": INT4_GROUP,
                "inner_k_tiles": INNER_K_TILES,
                "row": "gate_pack|down_pack|gate_saz|down_saz (tinygemm prepacked)",
                "kernel": "aten::_weight_int4pack_mm (sm_86-class tinygemm)"}
    from neural.moe.layout import prepacked_descriptor   # single source of truth, torch-free
    return prepacked_descriptor(layout)


class Q80PrepackedStore:
    def __init__(self, root: Path | str, layout=None) -> None:
        self.root = Path(root)
        self.layout = layout                 # None => module-global Q80 defaults
        self.meta: dict[str, Any] | None = None
        self._maps: dict[int, Any] = {}

    def _n_layers(self):
        return self.layout.n_layers if self.layout else N_LAYERS

    def _n_experts(self):
        return self.layout.n_experts if self.layout else N_EXPERTS

    def _slot_bytes(self):
        return self.layout.slot_bytes if self.layout else SLOT_BYTES

    @property
    def slot_bytes(self) -> int:
        """Bytes per row of THIS store (from the layout it was opened with)."""
        return self._slot_bytes()

    @property
    def weight_repr(self) -> str:
        """Representation key of the layout this store was opened with (e.g. "mxfp4_g32", "mxfp4_g32_ps4")."""
        return getattr(self.layout, "weight_repr", "int4_g32") if self.layout else "int4_g32"

    @staticmethod
    def peek_layout(root: Path | str) -> dict[str, Any] | None:
        """The layout descriptor recorded in ``<root>/metadata.json`` (None if there is no readable
        metadata). Pure JSON: no torch, no mapping. Use it to pick the ExpertLayout BEFORE open(); open()
        still validates everything strictly."""
        mp = Path(root) / "metadata.json"
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        lay = meta.get("layout")
        return lay if isinstance(lay, dict) else None

    # ---- construction ------------------------------------------------------
    @staticmethod
    def build(root: Path | str, builder, *, model_id: str,
              layers: range | None = None, progress=None,
              layout=None) -> dict[str, Any]:
        """Construct from a per-expert builder(layer, expert) -> uint8[SLOT_BYTES]
        (typically PrepackedHostPool._build over the existing INT4 store).
        Streams row-by-row: temp disk requirement is ZERO beyond the store
        itself; peak extra RAM is one slot buffer."""
        import time

        import torch

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        n_layers = layout.n_layers if layout else N_LAYERS
        n_experts = layout.n_experts if layout else N_EXPERTS
        slot_bytes = layout.slot_bytes if layout else SLOT_BYTES
        layers = layers or range(n_layers)
        hashes: dict[str, str] = {}
        t0 = time.time()
        for li in layers:
            path = root / f"layer_{li}.slots"
            h = hashlib.sha256()
            with open(path, "wb") as f:
                for e in range(n_experts):
                    buf = builder(li, e)
                    assert buf.dtype == torch.uint8 and buf.numel() == slot_bytes
                    b = buf.cpu().numpy().tobytes()
                    f.write(b)
                    h.update(b)
            hashes[str(li)] = h.hexdigest()
            if progress:
                progress(li)
        meta = {"magic": FORMAT_MAGIC, "version": FORMAT_VERSION,
                "model_id": model_id, "layout": _layout_descriptor(layout),
                "sha256_per_layer": hashes,
                "built_unix": int(time.time()),
                "build_seconds": round(time.time() - t0, 1)}
        (root / "metadata.json").write_text(json.dumps(meta, indent=1),
                                            encoding="utf-8")
        return meta

    # ---- access ------------------------------------------------------------
    def open(self, *, expect_model_id: str | None = None,
             shared: bool = True) -> dict[str, Any]:
        import torch

        mp = self.root / "metadata.json"
        if not mp.is_file():
            return {"status": "fail", "reason": "no metadata.json"}
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("magic") != FORMAT_MAGIC:
            return {"status": "fail", "reason": "bad magic"}
        if meta.get("version") != FORMAT_VERSION:
            return {"status": "fail", "reason": f"version {meta.get('version')} != {FORMAT_VERSION}"}
        # GPTOSS-TRANSFER-1: the descriptor carries STRUCTURAL fields (which
        # define the byte layout) plus a human-readable `row` label. Comparing
        # the whole dict made a COSMETIC RENAME of that label reject a 56.7 GiB
        # store whose three sampled layers hash-match their recorded SHA-256
        # exactly. Structure, kernel and weight_repr stay strict; `row` is
        # informational and a difference is reported, not fatal.
        want = _layout_descriptor(self.layout)
        have = meta.get("layout") or {}
        row_note = None
        if isinstance(have, dict):
            w = {k: v for k, v in want.items() if k != "row"}
            h = {k: v for k, v in have.items() if k != "row"}
            if w != h:
                out_ = {"status": "fail",
                        "reason": "layout descriptor mismatch (kernel/layout change)",
                        "differs": sorted(k for k in set(w) | set(h)
                                          if w.get(k) != h.get(k))}
                if w.get("weight_repr") != h.get("weight_repr"):
                    # e.g. a packed-scale store opened with the raw layout: never coerced, but say which is which
                    out_["store_weight_repr"] = h.get("weight_repr")
                    out_["runtime_weight_repr"] = w.get("weight_repr")
                return out_
            if want.get("row") != have.get("row"):
                row_note = (f"row label differs (store {have.get('row')!r} vs "
                            f"current {want.get('row')!r}); structural fields, "
                            "kernel and weight_repr all match")
        elif have != want:
            return {"status": "fail", "reason": "layout descriptor mismatch (kernel/layout change)"}
        if expect_model_id is not None and meta.get("model_id") != expect_model_id:
            return {"status": "fail", "reason": "model identity mismatch"}
        expected = self._n_experts() * self._slot_bytes()
        for li in range(self._n_layers()):
            p = self.root / f"layer_{li}.slots"
            if not p.is_file() or p.stat().st_size != expected:
                return {"status": "fail", "reason": f"layer {li} missing/size"}
            storage = torch.UntypedStorage.from_file(str(p), shared=shared,
                                                     nbytes=expected)
            t = torch.empty(0, dtype=torch.uint8)
            t.set_(storage)
            self._maps[li] = t.view(self._n_experts(), self._slot_bytes())
        self.meta = meta
        self.mapping_shared = shared
        out = {"status": "ok", "layers": self._n_layers(),
               "size_gib": round(self._n_layers() * expected / 1024**3, 2)}
        if row_note:
            out["descriptor_note"] = row_note
        return out

    def read_into(self, layer: int, expert: int, out_buf) -> None:
        """The entire retrieval: one contiguous memcpy from the mmap row."""
        out_buf.copy_(self._maps[layer][expert])

    def row(self, layer: int, expert: int):
        return self._maps[layer][expert]

    def verify_layer(self, layer: int) -> bool:
        """On-demand corruption check against the stored SHA-256."""
        h = hashlib.sha256()
        p = self.root / f"layer_{layer}.slots"
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 24)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest() == self.meta["sha256_per_layer"][str(layer)]


def attach_prepacked(host, pstore: Q80PrepackedStore) -> None:
    """Route the host pool's build path through the prepacked store.

    Flag-gated and reversible: ``host._prepacked_active`` toggles per call;
    when False (or on a store miss) the original build runs. Counters still
    account the bytes as store reads. Same bytes, same destinations — the
    LRU/tier semantics of the pool are untouched."""
    import torch

    orig_build = host._build
    host._prepacked_store = pstore
    host._prepacked_active = False
    host._prepacked_reads = 0

    slot_bytes = host.L.slot_bytes if hasattr(host, "L") else SLOT_BYTES

    host._prepacked_direct = False   # Q80-INT6-PROBE zero-copy prototype flag

    def build_prepacked(layer: int, expert: int):
        if not host._prepacked_active:
            return orig_build(layer, expert)
        if host.c is not None:
            host.c.ssd_fetches += 1
            host.c.ssd_bytes += slot_bytes
        if host._prepacked_direct:
            # ZERO-COPY prototype: hand the mmap row VIEW to the admission
            # path as a (pageable) H2D source - no pinned buffer, no host
            # memcpy; residency is delegated to the OS page cache
            host._prepacked_reads += 1
            return pstore.row(layer, expert)
        buf = host._free.pop() if host._free else torch.empty(
            slot_bytes, dtype=torch.uint8, pin_memory=True)
        pstore.read_into(layer, expert, buf)
        host._prepacked_reads += 1
        return buf

    host._build = build_prepacked
    host._orig_build = orig_build
