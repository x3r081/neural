"""Path-2: compressed VRAM residency + quantized expert GEMM (no BF16 reconstruct).

Backends (MEASURED availability on torch 2.10+cu130):
  INT8: aten::_weight_int8pack_mm  (W8A16/BF16, per-channel scales)
  INT4: aten::_convert_weight_to_int4pack + aten::_weight_int4pack_mm
        (tinygemm W4A16/BF16, group-wise scales_and_zeros)

Does not modify checkpoint files. Packs derived in-memory from BF16 params.
"""

from __future__ import annotations

import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F

QuantBits = Literal[4, 8]
PolicyName = Literal["demand_sync", "lru", "lfu"]
INT4_GROUP = 32
INT4_INNER_K_TILES = 2


def _nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def group_quantize_int4_nk(
    weight_nk: torch.Tensor, *, group_size: int = INT4_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric group INT4 for tinygemm. weight [N,K] -> q_u8 [N,K/2], saz [G,N,2].

    Port of torch.testing._internal.common_quantization._group_quantize_tensor
    applied directly to Linear [out, in] layout.
    """
    w = weight_nk.detach().float().contiguous()
    assert w.dim() == 2
    n, k = w.shape
    if k % group_size != 0:
        raise ValueError(f"in_features {k} not divisible by group_size {group_size}")
    to_quant = w.reshape(-1, group_size)
    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    scales = (max_val - min_val).clamp(min=1e-6) / 15.0
    zeros = min_val + scales * 8.0
    out = to_quant.sub(min_val).div(scales).round().clamp_(0, 15)
    out = out.to(torch.int32).reshape(n, k)
    q_u8 = (out[:, 0::2] << 4 | out[:, 1::2]).to(torch.uint8).contiguous()
    scales = scales.view(n, -1)
    zeros = zeros.view(n, -1)
    saz = (
        torch.cat([scales.unsqueeze(-1), zeros.unsqueeze(-1)], dim=2)
        .transpose(0, 1)
        .contiguous()
    )
    return q_u8, saz


def group_quantize_int4_nk_lsq(
    weight_nk: torch.Tensor, *, group_size: int = INT4_GROUP, iters: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """CAPACITY-4: least-squares refinement of the min-max affine INT4 group
    quantizer. Same tinygemm code semantics (w' = (q-8)*s + z, q in [0,15]);
    alternates closed-form (s, z) regression on the current assignments with
    re-assignment. Never worse than min-max in group MSE by construction
    (falls back per group when the refit regresses).
    """
    w = weight_nk.detach().float().contiguous()
    n, k = w.shape
    assert k % group_size == 0
    v = w.reshape(-1, group_size)                       # [G32, 32]
    max_val = v.amax(dim=1, keepdim=True)
    min_val = v.amin(dim=1, keepdim=True)
    s = (max_val - min_val).clamp(min=1e-6) / 15.0
    z = min_val + s * 8.0
    q = v.sub(min_val).div(s).round().clamp_(0, 15)

    def mse(s_, z_, q_):
        return (v - ((q_ - 8.0) * s_ + z_)).pow(2).mean(dim=1, keepdim=True)

    best_s, best_z, best_q = s, z, q
    best_e = mse(s, z, q)
    for _ in range(iters):
        x = q - 8.0                                     # regressor
        xm = x.mean(dim=1, keepdim=True)
        vm = v.mean(dim=1, keepdim=True)
        var = (x - xm).pow(2).sum(dim=1, keepdim=True)
        cov = ((x - xm) * (v - vm)).sum(dim=1, keepdim=True)
        s = torch.where(var > 0, cov / var.clamp(min=1e-12), s).clamp(min=1e-6)
        z = vm - s * xm
        q = (v - z).div(s).add(8.0).round().clamp_(0, 15)
        e = mse(s, z, q)
        better = e < best_e
        best_s = torch.where(better, s, best_s)
        best_z = torch.where(better, z, best_z)
        best_q = torch.where(better.expand_as(q), q, best_q)
        best_e = torch.minimum(e, best_e)
    q = best_q.to(torch.int32).reshape(n, k)
    q_u8 = (q[:, 0::2] << 4 | q[:, 1::2]).to(torch.uint8).contiguous()
    scales = best_s.view(n, -1)
    zeros = best_z.view(n, -1)
    saz = (
        torch.cat([scales.unsqueeze(-1), zeros.unsqueeze(-1)], dim=2)
        .transpose(0, 1)
        .contiguous()
    )
    return q_u8, saz


def pack_int8_per_channel(weight_nk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-out-channel INT8. weight [N,K] -> q[N,K] int8, scales[N] f32."""
    w = weight_nk.detach().float().contiguous()
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    scales = (amax / 127.0).to(torch.float32)
    q = torch.clamp(torch.round(w / scales[:, None]), -127, 127).to(torch.int8)
    return q.contiguous(), scales.contiguous()


@dataclass
class QuantLinearHost:
    name: str
    nbits: int
    out_features: int
    in_features: int
    # INT8 host
    q_int8: torch.Tensor | None = None
    scales_f32: torch.Tensor | None = None
    # INT4 host (pre-convert uint8 + saz)
    q_u8: torch.Tensor | None = None
    saz: torch.Tensor | None = None
    group_size: int = INT4_GROUP
    wire_bytes: int = 0
    bf16_bytes: int = 0


@dataclass
class QuantLinearDevice:
    name: str
    nbits: int
    out_features: int
    in_features: int
    # INT8
    q_int8: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    # INT4 (converted tinygemm pack)
    packed: torch.Tensor | None = None
    saz: torch.Tensor | None = None
    group_size: int = INT4_GROUP
    resident_bytes: int = 0


@dataclass
class QuantExpertHost:
    layer: int
    expert: int
    kind: Literal["phi_w123", "olmoe_swiglu"]
    linears: list[QuantLinearHost]
    wire_bytes: int
    bf16_bytes: int
    resident_bytes_est: int  # after GPU convert (INT4 pack size ≈ wire)


@dataclass
class QuantExpertDevice:
    layer: int
    expert: int
    kind: Literal["phi_w123", "olmoe_swiglu"]
    linears: dict[str, QuantLinearDevice]
    resident_bytes: int


def _detect_kind(mod: Any) -> Literal["phi_w123", "olmoe_swiglu"]:
    names = {n for n, _ in mod.named_parameters()}
    if {"w1.weight", "w2.weight", "w3.weight"} <= names:
        return "phi_w123"
    if {"gate_proj.weight", "up_proj.weight", "down_proj.weight"} <= names:
        return "olmoe_swiglu"
    raise ValueError(f"Unsupported expert param set: {sorted(names)}")


def pack_expert_host(mod: Any, layer: int, expert: int, nbits: QuantBits) -> QuantExpertHost:
    kind = _detect_kind(mod)
    linears: list[QuantLinearHost] = []
    wire = 0
    bf16 = 0
    for name, p in mod.named_parameters():
        if not name.endswith(".weight"):
            continue
        w = p.data
        bf = _nbytes(w)
        bf16 += bf
        short = name.split(".")[0]  # w1 / gate_proj / ...
        if nbits == 8:
            q, scales = pack_int8_per_channel(w)
            wb = _nbytes(q) + _nbytes(scales)
            linears.append(
                QuantLinearHost(
                    name=short,
                    nbits=8,
                    out_features=int(w.shape[0]),
                    in_features=int(w.shape[1]),
                    q_int8=q,
                    scales_f32=scales,
                    wire_bytes=wb,
                    bf16_bytes=bf,
                )
            )
            wire += wb
        else:
            q_u8, saz = group_quantize_int4_nk(w, group_size=INT4_GROUP)
            wb = _nbytes(q_u8) + _nbytes(saz)
            linears.append(
                QuantLinearHost(
                    name=short,
                    nbits=4,
                    out_features=int(w.shape[0]),
                    in_features=int(w.shape[1]),
                    q_u8=q_u8,
                    saz=saz,
                    group_size=INT4_GROUP,
                    wire_bytes=wb,
                    bf16_bytes=bf,
                )
            )
            wire += wb
    # INT4 convert keeps similar payload size; scales stay
    return QuantExpertHost(
        layer=layer,
        expert=expert,
        kind=kind,
        linears=linears,
        wire_bytes=wire,
        bf16_bytes=bf16,
        resident_bytes_est=wire,
    )


@torch.no_grad()
def quant_linear_forward(x: torch.Tensor, lin: QuantLinearDevice) -> torch.Tensor:
    """x [M,K] bf16/fp16 -> y [M,N] same dtype. Weight-only quantized GEMM."""
    if lin.nbits == 8:
        assert lin.q_int8 is not None and lin.scales is not None
        return torch.ops.aten._weight_int8pack_mm(x, lin.q_int8, lin.scales)
    assert lin.packed is not None and lin.saz is not None
    return torch.ops.aten._weight_int4pack_mm(
        x, lin.packed, int(lin.group_size), lin.saz
    )


@torch.no_grad()
def quant_expert_forward(
    x: torch.Tensor, exp: QuantExpertDevice
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run SwiGLU-style expert; returns (y, timing breakdown seconds)."""
    times = {
        "quant_gemm_s": 0.0,
        "activation_s": 0.0,
        "sync_s": 0.0,
    }

    def gemm(name: str, inp: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = quant_linear_forward(inp, exp.linears[name])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times["quant_gemm_s"] += time.perf_counter() - t0
        return out

    if exp.kind == "phi_w123":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = gemm("w1", x)
        b = gemm("w3", x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        h = F.silu(a) * b
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times["activation_s"] += time.perf_counter() - t1
        y = gemm("w2", h)
        times["sync_s"] += 0.0
        return y, times

    # olmoe
    g = gemm("gate_proj", x)
    u = gemm("up_proj", x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    h = F.silu(g) * u
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times["activation_s"] += time.perf_counter() - t1
    y = gemm("down_proj", h)
    return y, times


@dataclass
class QuantComputeCounters:
    ensure_calls: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    h2d_bytes: int = 0
    h2d_transfer_s: float = 0.0
    convert_s: float = 0.0  # INT4 pack convert / scale move
    quant_gemm_s: float = 0.0
    activation_s: float = 0.0
    stall_s: float = 0.0  # admit stalls (h2d+convert)
    compute_s: float = 0.0  # gemm+activation (expert invoke)
    resident_bytes_peak: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ensure_calls": self.ensure_calls,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "evictions": self.evictions,
            "h2d_bytes": self.h2d_bytes,
            "h2d_gib": self.h2d_bytes / 1024**3,
            "h2d_transfer_s": self.h2d_transfer_s,
            "convert_or_scale_overhead_s": self.convert_s,
            "quant_gemm_s": self.quant_gemm_s,
            "activation_s": self.activation_s,
            "stall_s": self.stall_s,
            "expert_compute_s": self.compute_s,
            "resident_bytes_peak": self.resident_bytes_peak,
            "cache_hit_rate": (
                self.hits / self.ensure_calls if self.ensure_calls else 0.0
            ),
            "dequant_s": 0.0,  # path-2: no BF16 reconstruction
            "path": "quantized_compute_residency",
        }


@dataclass
class QuantizedExpertResidencyManager:
    """Compressed-resident experts + quantized GEMM (path-2)."""

    access: Any
    budget_bytes: int  # VRAM budget for compressed residents
    policy: PolicyName
    device: Any
    nbits: QuantBits
    expert_bytes_bf16: int
    counters: QuantComputeCounters = field(default_factory=QuantComputeCounters)

    def __post_init__(self) -> None:
        self._hosts: dict[tuple[int, int], QuantExpertHost] = {}
        self._device: dict[tuple[int, int], QuantExpertDevice] = {}
        self._resident: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._freq: dict[tuple[int, int], int] = defaultdict(int)
        self._resident_bytes_each: int | None = None
        self.enabled = True
        self._capacity = 1

    @property
    def capacity_experts(self) -> int:
        return self._capacity

    @property
    def resident_bytes_per_expert(self) -> int:
        return int(self._resident_bytes_each or 0)

    def prepare_packs(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        wires = []
        res_est = []
        n_total = self.access.n_layers * self.access.n_experts
        for i, (layer, expert, mod) in enumerate(self.access.iter_expert_modules()):
            mod.to("cpu")
            host = pack_expert_host(mod, layer, expert, self.nbits)
            self._hosts[(layer, expert)] = host
            wires.append(host.wire_bytes)
            res_est.append(host.resident_bytes_est)
            if (i + 1) % 64 == 0 or (i + 1) == n_total:
                print(f"  quant-pack int{self.nbits}: {i + 1}/{n_total}", flush=True)
        self._resident_bytes_each = int(sum(res_est) / max(1, len(res_est)))
        self._capacity = max(
            1, self.budget_bytes // max(self._resident_bytes_each, 1)
        )
        return {
            "nbits": self.nbits,
            "n_experts_packed": len(self._hosts),
            "mean_wire_bytes": int(sum(wires) / max(1, len(wires))),
            "mean_resident_bytes_est": self._resident_bytes_each,
            "mean_bf16_bytes": self.expert_bytes_bf16,
            "compression_ratio_vs_bf16": (
                self._resident_bytes_each / self.expert_bytes_bf16
                if self.expert_bytes_bf16
                else None
            ),
            "capacity_experts": self._capacity,
            "budget_bytes": self.budget_bytes,
            "bf16_capacity_experts": max(
                1, self.budget_bytes // max(self.expert_bytes_bf16, 1)
            ),
            "capacity_multiplier_vs_bf16_slots": (
                self._capacity
                / max(1, self.budget_bytes // max(self.expert_bytes_bf16, 1))
            ),
            "pack_wall_s": time.perf_counter() - t0,
            "backends": {
                8: "aten::_weight_int8pack_mm",
                4: "aten::_convert_weight_to_int4pack + aten::_weight_int4pack_mm",
            }[self.nbits],
            "int4_group_size": INT4_GROUP if self.nbits == 4 else None,
            "provenance": "runtime_derived_from_bf16_module_params_in_memory",
            "checkpoint_files_modified": False,
            "evidence": "MEASURED",
        }

    def relocate_all_experts_to_cpu(self) -> dict[str, Any]:
        # Path-2: free GPU quantized residents; leave original modules on CPU
        self._device.clear()
        self._resident.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n = 0
        for layer, expert, mod in self.access.iter_expert_modules():
            mod.to("cpu")
            n += 1
        return {"n_experts_on_cpu": n, "quant_residents_cleared": True}

    def _evict_one(self) -> None:
        if not self._resident:
            return
        if self.policy == "lfu":
            victim = min(self._resident.keys(), key=lambda k: (self._freq[k], k))
            self._resident.pop(victim, None)
        else:
            victim, _ = self._resident.popitem(last=False)
        self._device.pop(victim, None)
        self.counters.evictions += 1

    def _admit(self, layer: int, expert: int) -> None:
        key = (layer, expert)
        host = self._hosts[key]
        while len(self._resident) >= self._capacity:
            self._evict_one()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        linears: dict[str, QuantLinearDevice] = {}
        resident = 0
        convert_s = 0.0
        for hl in host.linears:
            if hl.nbits == 8:
                assert hl.q_int8 is not None and hl.scales_f32 is not None
                q = hl.q_int8.to(self.device, non_blocking=True)
                s = hl.scales_f32.to(self.device, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                rb = _nbytes(q) + _nbytes(s)
                linears[hl.name] = QuantLinearDevice(
                    name=hl.name,
                    nbits=8,
                    out_features=hl.out_features,
                    in_features=hl.in_features,
                    q_int8=q,
                    scales=s,
                    resident_bytes=rb,
                )
                resident += rb
            else:
                assert hl.q_u8 is not None and hl.saz is not None
                q_u8 = hl.q_u8.to(self.device, non_blocking=True)
                saz = hl.saz.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_c0 = time.perf_counter()
                packed = torch.ops.aten._convert_weight_to_int4pack(
                    q_u8, INT4_INNER_K_TILES
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                convert_s += time.perf_counter() - t_c0
                # drop host-layout q_u8 after convert; keep packed + saz
                del q_u8
                rb = _nbytes(packed) + _nbytes(saz)
                linears[hl.name] = QuantLinearDevice(
                    name=hl.name,
                    nbits=4,
                    out_features=hl.out_features,
                    in_features=hl.in_features,
                    packed=packed,
                    saz=saz,
                    group_size=hl.group_size,
                    resident_bytes=rb,
                )
                resident += rb
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        transfer_s = time.perf_counter() - t0 - convert_s

        # Update measured mean resident bytes from first admits
        if self._resident_bytes_each is None or len(self._resident) == 0:
            self._resident_bytes_each = resident
            self._capacity = max(1, self.budget_bytes // max(resident, 1))

        self.counters.h2d_bytes += host.wire_bytes
        self.counters.h2d_transfer_s += max(transfer_s, 0.0)
        self.counters.convert_s += convert_s
        self.counters.stall_s += max(transfer_s, 0.0) + convert_s
        self._device[key] = QuantExpertDevice(
            layer=layer,
            expert=expert,
            kind=host.kind,
            linears=linears,
            resident_bytes=resident,
        )
        self._resident[key] = None
        peak = sum(self._device[k].resident_bytes for k in self._resident)
        if peak > self.counters.resident_bytes_peak:
            self.counters.resident_bytes_peak = peak

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

    def run_expert(self, layer: int, expert: int, x: torch.Tensor) -> torch.Tensor:
        self.ensure(layer, expert)
        exp = self._device[(layer, expert)]
        y, times = quant_expert_forward(x, exp)
        self.counters.quant_gemm_s += times["quant_gemm_s"]
        self.counters.activation_s += times["activation_s"]
        self.counters.compute_s += times["quant_gemm_s"] + times["activation_s"]
        return y


def install_quantized_paging_forward(
    mgr: QuantizedExpertResidencyManager,
    access: Any,
    *,
    model_kind: Literal["phi", "olmoe"],
) -> list[tuple[Any, Any]]:
    """Patch MoE block forward: native routing, quantized expert compute."""
    installed: list[tuple[Any, Any]] = []
    if model_kind == "phi":
        from neural.models.phi_runtime import _sparsemixer_fn

        for li in range(access.n_layers):
            block = access.moe_block(li)
            orig = block.forward
            sparsemixer = _sparsemixer_fn(block)

            def make(layer_id: int, blk: Any, sm=sparsemixer):
                def forward(hidden_states):
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    hs = hidden_states
                    if blk.training and blk.input_jitter_noise > 0:
                        hs = hs * torch.empty_like(hs).uniform_(
                            1.0 - blk.input_jitter_noise,
                            1.0 + blk.input_jitter_noise,
                        )
                    hs = hs.view(-1, hidden_dim)
                    router_logits = blk.gate(hs)
                    routing_weights, selected_experts = sm(
                        router_logits,
                        top_k=2,
                        jitter_eps=blk.router_jitter_noise,
                        training=blk.training,
                    )
                    final_hidden_states = torch.zeros(
                        (batch_size * sequence_length, hidden_dim),
                        dtype=hs.dtype,
                        device=hs.device,
                    )
                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=blk.num_experts
                    ).permute(2, 1, 0)
                    for expert_idx in range(blk.num_experts):
                        idx, top_x = torch.where(expert_mask[expert_idx])
                        if top_x.shape[0] == 0:
                            continue
                        current_state = hs[None, top_x.tolist()].reshape(-1, hidden_dim)
                        y = mgr.run_expert(layer_id, expert_idx, current_state)
                        final_hidden_states.index_add_(
                            0,
                            top_x,
                            (
                                y
                                * routing_weights[
                                    top_x.tolist(), idx.tolist(), None
                                ]
                            ).to(hs.dtype),
                        )
                    return (
                        final_hidden_states.reshape(
                            batch_size, sequence_length, hidden_dim
                        ),
                        router_logits,
                    )

                return forward

            block.forward = make(li, block)
            installed.append((block, orig))
        return installed

    # OLMoE
    for li in range(access.n_layers):
        block = access.moe_block(li)
        orig = block.forward

        def make(layer_id: int, blk: Any):
            def forward(hidden_states):
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                hs = hidden_states.view(-1, hidden_dim)
                router_logits = blk.gate(hs)
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(
                    routing_weights, blk.top_k, dim=-1
                )
                if getattr(blk, "norm_topk_prob", False):
                    routing_weights = routing_weights / routing_weights.sum(
                        dim=-1, keepdim=True
                    )
                routing_weights = routing_weights.to(hs.dtype)
                final_hidden_states = torch.zeros(
                    (batch_size * sequence_length, hidden_dim),
                    dtype=hs.dtype,
                    device=hs.device,
                )
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=blk.num_experts
                ).permute(2, 1, 0)
                for expert_idx in range(blk.num_experts):
                    idx, top_x = torch.where(expert_mask[expert_idx])
                    if top_x.numel() == 0:
                        continue
                    current_state = hs[None, top_x].reshape(-1, hidden_dim)
                    y = mgr.run_expert(layer_id, expert_idx, current_state)
                    final_hidden_states.index_add_(
                        0, top_x, (y * routing_weights[top_x, idx, None]).to(hs.dtype)
                    )
                return (
                    final_hidden_states.reshape(
                        batch_size, sequence_length, hidden_dim
                    ),
                    router_logits,
                )

            return forward

        block.forward = make(li, block)
        installed.append((block, orig))
    return installed


def uninstall_quantized_paging_forward(installed: list[tuple[Any, Any]]) -> None:
    for block, orig in installed:
        block.forward = orig
    installed.clear()
