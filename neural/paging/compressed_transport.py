"""Runtime-only compressed expert transport (path 1: pack→H2D→GPU dequant→BF16).

Does not modify checkpoint files. Packs are derived in-memory (or optional
cache dir outside model trees) from current BF16 module parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

TransportName = Literal["bf16", "int8", "int4"]


@dataclass
class TensorPack:
    name: str
    shape: tuple[int, ...]
    nbits: int
    scale: torch.Tensor  # CPU float32 scalar
    payload: torch.Tensor  # CPU int8 (INT8) or packed uint8 (INT4)
    orig_dtype: torch.dtype


@dataclass
class ExpertPack:
    layer: int
    expert: int
    tensors: list[TensorPack]
    wire_bytes: int
    bf16_bytes: int
    transport: TransportName


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def pack_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor INT8. Returns (q_int8_cpu, scale_f32_cpu)."""
    w = weight.detach().float().cpu()
    amax = w.abs().max().clamp_min(1e-12)
    scale = (amax / 127.0).to(torch.float32)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q.contiguous(), scale.reshape(())


def pack_int4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor INT4 packed 2×nibbles/byte. Returns (packed_u8, scale)."""
    w = weight.detach().float().cpu()
    amax = w.abs().max().clamp_min(1e-12)
    scale = (amax / 7.0).to(torch.float32)
    q = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8).view(-1)
    # pack pairs; pad odd length
    n = q.numel()
    if n % 2 == 1:
        q = torch.cat([q, q.new_zeros(1)])
    lo = (q[0::2] & 0x0F).to(torch.uint8)
    hi = (q[1::2] & 0x0F).to(torch.uint8)
    packed = (lo | (hi << 4)).contiguous()
    return packed, scale.reshape(())


@torch.no_grad()
def dequant_int8_to_bf16(
    q: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, float]:
    q_g = q.to(device=device, non_blocking=True)
    s_g = scale.to(device=device, non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = (q_g.float() * s_g).to(torch.bfloat16).view(shape)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out, dt


@torch.no_grad()
def dequant_int4_to_bf16(
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    p_g = packed.to(device=device, non_blocking=True)
    s_g = scale.to(device=device, non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    lo = (p_g & 0x0F).to(torch.int8)
    hi = ((p_g >> 4) & 0x0F).to(torch.int8)
    # sign-extend 4-bit two's complement stored in low nibble 0..15 with values -8..7
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    q = torch.empty(p_g.numel() * 2, device=device, dtype=torch.int8)
    q[0::2] = lo
    q[1::2] = hi
    n = int(torch.tensor(shape).prod().item())
    q = q[:n]
    out = (q.float() * s_g).to(torch.bfloat16).view(shape)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out, dt


def pack_expert_bf16(mod: Any, layer: int = 0, expert: int = 0) -> ExpertPack:
    return pack_expert_module(mod, layer, expert, "bf16")


def pack_expert_int8(mod: Any, layer: int = 0, expert: int = 0) -> ExpertPack:
    return pack_expert_module(mod, layer, expert, "int8")


def pack_expert_int4(mod: Any, layer: int = 0, expert: int = 0) -> ExpertPack:
    return pack_expert_module(mod, layer, expert, "int4")


def pack_expert_module(mod: Any, layer: int, expert: int, transport: TransportName) -> ExpertPack:
    tensors: list[TensorPack] = []
    wire = 0
    bf16 = 0
    for name, p in mod.named_parameters(recurse=True):
        bf16 += _tensor_nbytes(p.data)
        if transport == "bf16":
            # no pack; wire = bf16
            continue
        if transport == "int8":
            q, scale = pack_int8(p.data)
            tp = TensorPack(
                name=name,
                shape=tuple(p.shape),
                nbits=8,
                scale=scale,
                payload=q,
                orig_dtype=p.dtype,
            )
            wire += _tensor_nbytes(q) + _tensor_nbytes(scale)
        elif transport == "int4":
            packed, scale = pack_int4(p.data)
            tp = TensorPack(
                name=name,
                shape=tuple(p.shape),
                nbits=4,
                scale=scale,
                payload=packed,
                orig_dtype=p.dtype,
            )
            wire += _tensor_nbytes(packed) + _tensor_nbytes(scale)
        else:
            raise ValueError(transport)
        tensors.append(tp)
    if transport == "bf16":
        wire = bf16
    return ExpertPack(
        layer=layer,
        expert=expert,
        tensors=tensors,
        wire_bytes=wire,
        bf16_bytes=bf16,
        transport=transport,
    )


@dataclass
class TransportCounters:
    ensure_calls: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    h2d_bytes: int = 0  # wire bytes
    d2h_bytes: int = 0
    h2d_transfer_s: float = 0.0
    dequant_s: float = 0.0
    stall_s: float = 0.0  # transfer + dequant (sync admit)
    compute_s: float = 0.0
    bf16_resident_bytes_peak: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ensure_calls": self.ensure_calls,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "evictions": self.evictions,
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "h2d_gib": self.h2d_bytes / 1024**3,
            "h2d_transfer_s": self.h2d_transfer_s,
            "dequant_s": self.dequant_s,
            "stall_s": self.stall_s,
            "expert_compute_s": self.compute_s,
            "bf16_resident_bytes_peak": self.bf16_resident_bytes_peak,
            "effective_h2d_gib_s": (
                (self.h2d_bytes / 1024**3) / self.h2d_transfer_s
                if self.h2d_transfer_s > 0
                else None
            ),
        }


@dataclass
class CompressedExpertResidencyManager:
    """Expert residency with optional compressed H2D + GPU dequant to BF16."""

    access: Any
    budget_bytes: int  # BF16-resident capacity (same as expansion curve)
    policy: Literal["demand_sync", "lru", "lfu"]
    device: Any
    expert_bytes_bf16: int
    transport: TransportName = "bf16"
    counters: TransportCounters = field(default_factory=TransportCounters)

    def __post_init__(self) -> None:
        from collections import OrderedDict, defaultdict

        self._resident: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._freq: dict[tuple[int, int], int] = defaultdict(int)
        self._capacity = max(1, self.budget_bytes // max(self.expert_bytes_bf16, 1))
        self._packs: dict[tuple[int, int], ExpertPack] = {}
        self.enabled = True
        self._wire_bytes_per_expert: int | None = None

    @property
    def capacity_experts(self) -> int:
        return self._capacity

    def prepare_packs(self) -> dict[str, Any]:
        """Build in-memory packs from current CPU BF16 params. Model files untouched."""
        t0 = time.perf_counter()
        wires = []
        n_total = self.access.n_layers * self.access.n_experts
        for i, (layer, expert, mod) in enumerate(self.access.iter_expert_modules()):
            # Ensure on CPU BF16 source
            mod.to("cpu")
            pack = pack_expert_module(mod, layer, expert, self.transport)
            self._packs[(layer, expert)] = pack
            wires.append(pack.wire_bytes)
            if (i + 1) % 64 == 0 or (i + 1) == n_total:
                print(
                    f"  pack {self.transport}: {i + 1}/{n_total}",
                    flush=True,
                )
        self._wire_bytes_per_expert = int(sum(wires) / max(1, len(wires)))
        return {
            "transport": self.transport,
            "n_experts_packed": len(self._packs),
            "mean_wire_bytes": self._wire_bytes_per_expert,
            "mean_bf16_bytes": self.expert_bytes_bf16,
            "compression_ratio_vs_bf16": (
                self._wire_bytes_per_expert / self.expert_bytes_bf16
                if self.expert_bytes_bf16
                else None
            ),
            "pack_wall_s": time.perf_counter() - t0,
            "provenance": "runtime_derived_from_bf16_module_params_in_memory",
            "checkpoint_files_modified": False,
            "evidence": "MEASURED",
        }

    def relocate_all_experts_to_cpu(self) -> dict[str, Any]:
        n = 0
        for layer, expert, mod in self.access.iter_expert_modules():
            mod.to("cpu")
            n += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._resident.clear()
        return {"n_experts_on_cpu": n, "evidence": "MEASURED"}

    def _evict_one(self) -> None:
        if not self._resident:
            return
        if self.policy == "lfu":
            victim = min(self._resident.keys(), key=lambda k: (self._freq[k], k))
            self._resident.pop(victim, None)
        else:
            victim, _ = self._resident.popitem(last=False)
        layer, expert = victim
        mod = self.access.expert_module(layer, expert)
        nb = self.expert_bytes_bf16
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mod.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.counters.d2h_bytes += nb  # BF16 d2h for eviction path
        self.counters.evictions += 1

    def _admit_bf16(self, layer: int, expert: int) -> None:
        mod = self.access.expert_module(layer, expert)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mod.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.counters.h2d_bytes += self.expert_bytes_bf16
        self.counters.h2d_transfer_s += dt
        self.counters.stall_s += dt

    def _admit_compressed(self, layer: int, expert: int) -> None:
        mod = self.access.expert_module(layer, expert)
        pack = self._packs[(layer, expert)]
        # Allocate BF16 param storage on GPU without H2D of full BF16 source.
        for p in mod.parameters():
            if p.device != self.device:
                p.data = torch.empty(p.shape, dtype=torch.bfloat16, device=self.device)
        name_to_param = dict(mod.named_parameters(recurse=True))
        wire = sum(
            _tensor_nbytes(tp.payload) + _tensor_nbytes(tp.scale) for tp in pack.tensors
        )

        # Phase 1: compressed H2D (one sync barrier; match BF16 mod.to style)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_payloads: list[tuple[TensorPack, torch.Tensor, torch.Tensor]] = []
        for tp in pack.tensors:
            payload_g = tp.payload.to(self.device, non_blocking=True)
            scale_g = tp.scale.to(self.device, non_blocking=True)
            gpu_payloads.append((tp, payload_g, scale_g))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        transfer_s = time.perf_counter() - t0

        # Phase 2: GPU dequant → BF16 param (separate from H2D)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for tp, payload_g, scale_g in gpu_payloads:
            if tp.nbits == 8:
                out = (payload_g.float() * scale_g).to(torch.bfloat16).view(tp.shape)
            else:
                lo = (payload_g & 0x0F).to(torch.int8)
                hi = ((payload_g >> 4) & 0x0F).to(torch.int8)
                lo = torch.where(lo >= 8, lo - 16, lo)
                hi = torch.where(hi >= 8, hi - 16, hi)
                q = torch.empty(
                    payload_g.numel() * 2, device=self.device, dtype=torch.int8
                )
                q[0::2] = lo
                q[1::2] = hi
                n = 1
                for d in tp.shape:
                    n *= d
                out = (q[:n].float() * scale_g).to(torch.bfloat16).view(tp.shape)
            name_to_param[tp.name].data.copy_(out)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dequant_s = time.perf_counter() - t1

        self.counters.h2d_bytes += wire
        self.counters.h2d_transfer_s += transfer_s
        self.counters.dequant_s += dequant_s
        self.counters.stall_s += transfer_s + dequant_s

    def _admit(self, layer: int, expert: int) -> None:
        key = (layer, expert)
        if key in self._resident:
            self._resident.move_to_end(key)
            return
        while len(self._resident) >= self._capacity:
            self._evict_one()
        if self.transport == "bf16":
            self._admit_bf16(layer, expert)
        else:
            self._admit_compressed(layer, expert)
        self._resident[key] = None
        peak = len(self._resident) * self.expert_bytes_bf16
        if peak > self.counters.bf16_resident_bytes_peak:
            self.counters.bf16_resident_bytes_peak = peak

    def ensure(self, layer: int, expert: int) -> None:
        if not self.enabled:
            return
        self.counters.ensure_calls += 1
        key = (layer, expert)
        if key in self._resident:
            self.counters.hits += 1
            self._resident.move_to_end(key)
            self._freq[key] += 1
            return
        self.counters.misses += 1
        self._admit(layer, expert)
        self._freq[key] += 1


# Alias expected by neural.paging package exports / older call sites.
CompressedResidencyManager = CompressedExpertResidencyManager
