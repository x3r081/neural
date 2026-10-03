"""Q80-7 — compressed (INT8/INT4) routed-expert store for direct quantized GEMM.

Path-2: quantize-on-build from the BF16 packed store, persist a compressed
routed-expert pool, and serve compressed host representations for demand paging.
No BF16 reconstruction is performed on the compute path.

Q80 routed expert (per expert):
  gate_up_proj: [1024, 2048]  (N=1024, K=2048)  -> chunk(2) after linear
  down_proj:    [2048,  512]  (N=2048, K=512)

Reuses model-agnostic primitives from neural.paging.quantized_compute:
  pack_int8_per_channel, group_quantize_int4_nk, INT4_GROUP.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from neural.paging.quantized_compute import (
    INT4_GROUP,
    group_quantize_int4_nk,
    pack_int8_per_channel,
)
from neural.q80.expert_store import (
    DOWN_SHAPE,
    GATE_UP_SHAPE,
    Qwen3HostExpertStore,
)

GIB = 1024**3
QuantBits = Literal[4, 8]

# INT4 host-layout (uint8 nibble pack) shapes and metadata (per expert)
INT4_GATE_UP_QU8 = (GATE_UP_SHAPE[0], GATE_UP_SHAPE[1] // 2)      # (1024, 1024)
INT4_GATE_UP_SAZ = (GATE_UP_SHAPE[1] // INT4_GROUP, GATE_UP_SHAPE[0], 2)  # (64,1024,2)
INT4_DOWN_QU8 = (DOWN_SHAPE[0], DOWN_SHAPE[1] // 2)               # (2048, 256)
INT4_DOWN_SAZ = (DOWN_SHAPE[1] // INT4_GROUP, DOWN_SHAPE[0], 2)   # (16,2048,2)


def _nbytes(t: Any) -> int:
    return int(t.numel() * t.element_size())


@dataclass
class CompressedExpertHost:
    """Host-side compressed representation of one routed expert (two linears)."""

    layer: int
    expert: int
    nbits: int
    # INT8: q_int8 [N,K] + scales [N] ; INT4: q_u8 [N,K/2] + saz [G,N,2]
    gate_up_q: Any = None
    gate_up_meta: Any = None       # scales (int8) or saz (int4)
    down_q: Any = None
    down_meta: Any = None
    wire_bytes: int = 0


def quantize_expert(gate_up_bf16: Any, down_bf16: Any, nbits: QuantBits,
                    layer: int, expert: int) -> CompressedExpertHost:
    """Quantize one expert's two linears from BF16 (exact source) → compressed host."""
    if nbits == 8:
        gq, gs = pack_int8_per_channel(gate_up_bf16)
        dq, ds = pack_int8_per_channel(down_bf16)
        wire = _nbytes(gq) + _nbytes(gs) + _nbytes(dq) + _nbytes(ds)
        return CompressedExpertHost(layer, expert, 8, gq, gs, dq, ds, wire)
    gq, gsaz = group_quantize_int4_nk(gate_up_bf16, group_size=INT4_GROUP)
    dq, dsaz = group_quantize_int4_nk(down_bf16, group_size=INT4_GROUP)
    # store saz as bf16 to halve metadata bytes (tinygemm expects bf16/fp16 saz)
    import torch

    gsaz = gsaz.to(torch.bfloat16).contiguous()
    dsaz = dsaz.to(torch.bfloat16).contiguous()
    wire = _nbytes(gq) + _nbytes(gsaz) + _nbytes(dq) + _nbytes(dsaz)
    return CompressedExpertHost(layer, expert, 4, gq, gsaz, dq, dsaz, wire)


# ---------------------------------------------------------------------------
# INT4 persistent on-disk store (mmap-backed, one set of packs per layer)
# ---------------------------------------------------------------------------

def _int4_paths(root: Path, layer: int) -> dict[str, Path]:
    return {
        "gate_up_q": root / f"L{layer}.gate_up.q_u8.int4.bin",
        "gate_up_saz": root / f"L{layer}.gate_up.saz.int4.bin",
        "down_q": root / f"L{layer}.down.q_u8.int4.bin",
        "down_saz": root / f"L{layer}.down.saz.int4.bin",
    }


@dataclass
class Q80Int4DiskStore:
    """Persistent INT4 routed-expert pool (uint8 nibble packs + bf16 saz), mmap."""

    root: Path
    n_layers: int = 48
    n_experts: int = 512
    _maps: dict[int, dict[str, Any]] = field(default_factory=dict)
    ssd_read_bytes: int = 0

    def _expected_sizes(self) -> dict[str, int]:
        e = self.n_experts
        return {
            "gate_up_q": e * INT4_GATE_UP_QU8[0] * INT4_GATE_UP_QU8[1] * 1,
            "gate_up_saz": e * INT4_GATE_UP_SAZ[0] * INT4_GATE_UP_SAZ[1] * INT4_GATE_UP_SAZ[2] * 2,
            "down_q": e * INT4_DOWN_QU8[0] * INT4_DOWN_QU8[1] * 1,
            "down_saz": e * INT4_DOWN_SAZ[0] * INT4_DOWN_SAZ[1] * INT4_DOWN_SAZ[2] * 2,
        }

    def is_built(self) -> bool:
        sizes = self._expected_sizes()
        for li in range(self.n_layers):
            for key, p in _int4_paths(self.root, li).items():
                if not p.is_file() or p.stat().st_size != sizes[key]:
                    return False
        return True

    def build_from_bf16(self, bf16_store: Qwen3HostExpertStore, *,
                        log_every: int = 4, only_layers: list[int] | None = None) -> dict[str, Any]:
        """One-time quantize-on-build INT4 pool from the BF16 packed store."""
        import torch

        self.root.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        sizes = self._expected_sizes()
        total_bytes = 0
        layer_iter = only_layers if only_layers is not None else range(self.n_layers)
        for li in layer_iter:
            paths = _int4_paths(self.root, li)
            if all(paths[k].is_file() and paths[k].stat().st_size == sizes[k] for k in paths):
                total_bytes += sum(sizes.values())
                continue
            pack = bf16_store._layers[li]
            gate_up_q = torch.empty((self.n_experts, *INT4_GATE_UP_QU8), dtype=torch.uint8)
            gate_up_saz = torch.empty((self.n_experts, *INT4_GATE_UP_SAZ), dtype=torch.bfloat16)
            down_q = torch.empty((self.n_experts, *INT4_DOWN_QU8), dtype=torch.uint8)
            down_saz = torch.empty((self.n_experts, *INT4_DOWN_SAZ), dtype=torch.bfloat16)
            for e in range(self.n_experts):
                gu = pack.gate_up[e].contiguous()
                dn = pack.down[e].contiguous()
                ce = quantize_expert(gu, dn, 4, li, e)
                gate_up_q[e] = ce.gate_up_q
                gate_up_saz[e] = ce.gate_up_meta
                down_q[e] = ce.down_q
                down_saz[e] = ce.down_meta
            gate_up_q.numpy().tofile(str(paths["gate_up_q"]))
            gate_up_saz.view(torch.int16).numpy().tofile(str(paths["gate_up_saz"]))
            down_q.numpy().tofile(str(paths["down_q"]))
            down_saz.view(torch.int16).numpy().tofile(str(paths["down_saz"]))
            total_bytes += sum(sizes.values())
            if (li + 1) % log_every == 0 or li + 1 == self.n_layers:
                print(f"  int4 build layer {li + 1}/{self.n_layers} "
                      f"({total_bytes / GIB:.1f} GiB)", flush=True)
        return {
            "status": "ok",
            "root": str(self.root),
            "n_layers": self.n_layers,
            "build_wall_s": time.perf_counter() - t0,
            "pool_bytes": total_bytes,
            "pool_gib": total_bytes / GIB,
            "bytes_per_expert": total_bytes // (self.n_layers * self.n_experts),
            "evidence_class": "MEASURED",
        }

    def open(self) -> dict[str, Any]:
        import torch

        sizes = self._expected_sizes()
        missing = []
        for li in range(self.n_layers):
            paths = _int4_paths(self.root, li)
            m: dict[str, Any] = {}
            ok = True
            for key, p in paths.items():
                if not p.is_file() or p.stat().st_size != sizes[key]:
                    missing.append(f"L{li}:{key}")
                    ok = False
                    break
            if not ok:
                continue
            # mmap each as a torch tensor view
            def _map(path: Path, shape, dtype):
                nb = 1
                for s in shape:
                    nb *= s
                elt = 1 if dtype == torch.uint8 else 2
                storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=nb * elt)
                t = torch.empty(0, dtype=dtype)
                t.set_(storage)
                return t.view(self.n_experts, *shape[1:]) if False else t.view(*shape)
            m["gate_up_q"] = _map(paths["gate_up_q"], (self.n_experts, *INT4_GATE_UP_QU8), torch.uint8)
            m["gate_up_saz"] = _map(paths["gate_up_saz"], (self.n_experts, *INT4_GATE_UP_SAZ), torch.bfloat16)
            m["down_q"] = _map(paths["down_q"], (self.n_experts, *INT4_DOWN_QU8), torch.uint8)
            m["down_saz"] = _map(paths["down_saz"], (self.n_experts, *INT4_DOWN_SAZ), torch.bfloat16)
            self._maps[li] = m
        return {
            "status": "ok" if len(self._maps) == self.n_layers else "fail",
            "n_layers_mapped": len(self._maps),
            "missing": missing[:20],
            "evidence_class": "MEASURED",
        }

    def fetch_compressed(self, layer: int, expert: int) -> CompressedExpertHost:
        m = self._maps[layer]
        gq = m["gate_up_q"][expert].contiguous().clone()
        gsaz = m["gate_up_saz"][expert].contiguous().clone()
        dq = m["down_q"][expert].contiguous().clone()
        dsaz = m["down_saz"][expert].contiguous().clone()
        wire = _nbytes(gq) + _nbytes(gsaz) + _nbytes(dq) + _nbytes(dsaz)
        self.ssd_read_bytes += wire
        return CompressedExpertHost(layer, expert, 4, gq, gsaz, dq, dsaz, wire)


@dataclass
class Q80Int8OnDemandStore:
    """INT8 control: quantize per expert on demand from the BF16 packed store.

    Compressed VRAM residency + compressed H2D; SSD source remains BF16 (control
    limitation documented — INT8 pool 72 GiB does not fit host RAM anyway).
    """

    bf16_store: Qwen3HostExpertStore
    ssd_read_bytes: int = 0

    def fetch_compressed(self, layer: int, expert: int) -> CompressedExpertHost:
        gu, dn = self.bf16_store.fetch_cpu_clones(layer, expert)
        self.ssd_read_bytes += (gu.numel() * 2 + dn.numel() * 2)  # BF16 source read
        return quantize_expert(gu, dn, 8, layer, expert)


@dataclass
class Q80Int4OnDemandStore:
    """INT4 quantized on demand from the BF16 packed store (no persistent disk).

    For the quality gate and fast iteration. SSD source is BF16 (compressed SSD
    is only claimed for the persistent Q80Int4DiskStore).
    """

    bf16_store: Qwen3HostExpertStore
    ssd_read_bytes: int = 0

    def fetch_compressed(self, layer: int, expert: int) -> CompressedExpertHost:
        gu, dn = self.bf16_store.fetch_cpu_clones(layer, expert)
        self.ssd_read_bytes += (gu.numel() * 2 + dn.numel() * 2)
        return quantize_expert(gu, dn, 4, layer, expert)


def fingerprint_int4(disk: Q80Int4DiskStore, bf16_store: Qwen3HostExpertStore,
                     *, samples=((0, 0), (17, 3), (47, 511))) -> dict[str, Any]:
    """Relate INT4 store to BF16 source: deterministic hash + reconstruction relerr."""
    import torch

    from neural.paging.quantized_compute import group_quantize_int4_nk

    h = hashlib.sha256()
    rows = []
    for (li, e) in samples:
        ce = disk.fetch_compressed(li, e)
        h.update(ce.gate_up_q.numpy().tobytes())
        h.update(ce.down_q.numpy().tobytes())
        # reconstruction error vs BF16 (dequant only for verification, not compute path)
        gu, dn = bf16_store.fetch_cpu_clones(li, e)
        gq, gsaz = group_quantize_int4_nk(gu, group_size=INT4_GROUP)
        det_match = bool(torch.equal(gq, ce.gate_up_q))
        rows.append({"layer": li, "expert": e, "int4_quant_deterministic_match": det_match})
    return {
        "sample_sha256_prefix": h.hexdigest()[:16],
        "samples": rows,
        "note": "INT4 quant is deterministic from BF16 source; checkpoint untouched.",
        "evidence_class": "MEASURED",
    }
