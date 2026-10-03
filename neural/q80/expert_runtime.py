"""Q80-10 — production Variant-D expert runtime (slot-pool INT4 expert path).

Integrates the Q80-9-proven techniques into the real Q80 decode path:

  A. sync-free routing  — ONE bounded routing-index materialization
     (``top_k_index.tolist()``) per experts.forward call; no ``.item()``,
     no ``torch.where``/``nonzero`` on the hot path
  B. stacked gate_up    — the 10 selected gate_up packs run as ONE
     ``aten::_weight_int4pack_mm`` (bit-exact vs separate calls; Q80-9)
  C. prepacked weights  — ``_convert_weight_to_int4pack`` runs once per
     expert at (lazy) build time, never during decode
  D. fixed VRAM slot pool — one uint8 pool tensor + typed views; miss =
     ONE contiguous pinned->slot async H2D copy; LRU at expert granularity

Semantics preserved exactly: native routing/top-k, LRU @ 4 GiB, direct INT4
GEMM, no BF16 reconstruction. Aggregation iterates experts in ascending id
order with sequential adds — the same floating-point accumulation order as the
old path's sorted ``index_add_`` loop, so outputs are bit-identical.

NOT included (by milestone scope): CUDA graphs, Triton, prefetch/prediction,
GDN changes, core quantization.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from neural.q80.quant_store import Q80Int4DiskStore

GIB = 1024**3
N_LAYERS = 48
TOP_K = 10
HIDDEN = 2048
GATE_UP_N = 1024
DOWN_N = 2048
DOWN_K = 512
INT4_GROUP = 32
INNER_K_TILES = 2

GATE_PACK_BYTES = 1_048_576
DOWN_PACK_BYTES = 524_288
GATE_SAZ_BYTES = 262_144
DOWN_SAZ_BYTES = 131_072
SLOT_BYTES = GATE_PACK_BYTES + DOWN_PACK_BYTES + GATE_SAZ_BYTES + DOWN_SAZ_BYTES


# ---------------------------------------------------------------------------
# CAPACITY-4: expert activation registry. Resolved from ExpertLayout.act
# (a FEATURE key, never a model name). "swiglu" reproduces the historical
# inline math exactly, op for op — all pre-existing layouts default to it.
# ---------------------------------------------------------------------------

def _act_swiglu(gu):
    import torch.nn.functional as F

    gate, up = gu.chunk(2, dim=-1)
    return F.silu(gate) * up


def _make_clamped_swiglu(alpha: float, limit: float):
    def act(gu):
        import torch

        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        return (up + 1) * glu
    return act


def resolve_activation(layout):
    name = getattr(layout, "act", "swiglu")
    if name == "swiglu":
        return _act_swiglu
    if name == "clamped_swiglu":
        return _make_clamped_swiglu(layout.act_alpha, layout.act_limit)
    raise ValueError(f"unknown expert activation feature: {name!r}")


# Q80-GRAPH-PREAMBLE A/B switch. Default True = the reduced host preamble.
# Measurement harnesses flip it to compare against the pre-existing path in the
# SAME session; a control measured in another session is not a control.
USE_V4_GRAPH = True


class GraphFallbackError(RuntimeError):
    """An intended-graphed decode fell back to eager execution.

    Q80-INT6-PRODUCTION Part 1: a silent eager fallback (the Q80-INT6-PROBE
    dispatch bug) produced two milestones of invalid physics before review
    caught it. When ``runtime.require_graph`` is set, that failure mode raises
    instead of degrading quietly, so a fallback can never again yield a number
    that looks like a valid measurement.
    """


@dataclass
class RuntimeCounters:
    ensure_calls: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    h2d_ops: int = 0
    h2d_bytes: int = 0
    forced_syncs: int = 0
    ssd_fetches: int = 0
    ssd_bytes: int = 0
    prepack_converts: int = 0  # build-time only, never per-decode
    host_pinned_evictions: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = {**self.__dict__}
        d["vram_hit_rate"] = self.hits / max(self.ensure_calls, 1)
        d["h2d_gib"] = self.h2d_bytes / GIB
        d["ssd_gib"] = self.ssd_bytes / GIB
        return d

    def reset(self) -> None:
        for k in list(self.__dict__):
            setattr(self, k, 0)


class PrepackedHostPool:
    """Lazy prepacked host tier: INT4 store -> GEMM-ready pinned bytes.

    Build path (once per expert, off the steady-state hot path): SSD/mmap fetch
    -> GPU ``_convert_weight_to_int4pack`` (deterministic; bit-exact per Q80-9)
    -> ONE contiguous pinned host buffer laid out slot-compatible. LRU-bounded
    pinned budget; per-buffer CUDA events guard against reuse of a buffer whose
    H2D is still in flight. No new persistent disk store is created (decision
    recorded in the Q80-10 report; F: free space and warm-primary mission).
    """

    # Q80-HOSTTIER-SCAN-RESISTANCE Part 2: every host-tier event is attributed
    # to the ROLE that caused it. The VRAM pool isolates prefill through a
    # scratch ring; the host tiers do not, so prefill behaves as a cache scan
    # over exactly the residents decode depends on. Nothing here changes
    # behaviour — it makes the scan visible.
    ROLES = ("decode", "prefill", "other")
    EVENTS = ("pinned_hit", "pinned_refresh", "pageable_hit_promote",
              "build", "insert_pinned", "demote", "evict_pinned",
              "evict_pageable")

    def _new_tier_stats(self):
        return {r: dict.fromkeys(self.EVENTS, 0) for r in self.ROLES}

    def note(self, event: str) -> None:
        s = self.tier_stats.get(self.access_role)
        if s is not None:
            s[event] += 1

    def __init__(self, store: Q80Int4DiskStore, device: Any,
                 pinned_budget_gib: float = 12.0, layout=None) -> None:
        import torch

        from neural.moe.layout import Q80_LAYOUT

        self.L = layout or Q80_LAYOUT
        self.store = store            # may be None when a prepacked store serves builds
        self.device = device
        self.capacity = max(64, int(pinned_budget_gib * GIB) // self.L.slot_bytes)
        self._resident: "OrderedDict[tuple[int,int], torch.Tensor]" = OrderedDict()
        self._events: dict[tuple[int, int], Any] = {}
        self._free: list = []
        self.access_role = "other"    # set by the runtime around each forward
        self.tier_stats = self._new_tier_stats()
        self.c: RuntimeCounters | None = None  # attached by runtime

    def _build(self, layer: int, expert: int):
        import torch

        if self.store is None:
            raise RuntimeError("no raw store attached; prepacked store must serve builds")
        ce = self.store.fetch_compressed(layer, expert)  # SSD/page-cache read
        if self.c is not None:
            self.c.ssd_fetches += 1
            self.c.ssd_bytes += ce.wire_bytes
        gq = ce.gate_up_q.to(self.device)
        dq = ce.down_q.to(self.device)
        gp = torch.ops.aten._convert_weight_to_int4pack(gq, self.L.inner_k_tiles)
        dp = torch.ops.aten._convert_weight_to_int4pack(dq, self.L.inner_k_tiles)
        if self.c is not None:
            self.c.prepack_converts += 2
        buf = self._free.pop() if self._free else torch.empty(
            self.L.slot_bytes, dtype=torch.uint8, pin_memory=True)
        o = 0
        for t in (gp, dp):
            b = t.contiguous().view(torch.uint8).flatten().cpu()
            buf[o:o + b.numel()].copy_(b)
            o += b.numel()
        for t in (ce.gate_up_meta, ce.down_meta):
            b = t.contiguous().view(torch.uint8).flatten()
            buf[o:o + b.numel()].copy_(b)
            o += b.numel()
        return buf

    def get(self, layer: int, expert: int):
        key = (layer, expert)
        buf = self._resident.get(key)
        if buf is not None:
            self._resident.move_to_end(key)
            return buf
        while len(self._resident) >= self.capacity:
            vk, vbuf = self._resident.popitem(last=False)
            ev = self._events.pop(vk, None)
            if ev is not None and not ev.query():
                ev.synchronize()  # buffer's H2D still in flight (rare)
            self._free.append(vbuf)
            if self.c is not None:
                self.c.host_pinned_evictions += 1
        buf = self._build(layer, expert)
        self._resident[key] = buf
        return buf

    def note_h2d(self, key, event) -> None:
        self._events[key] = event

    def prewarm(self, keys) -> int:
        n = 0
        for (l, e) in keys:
            self.get(int(l), int(e))
            n += 1
        return n


class Q80SlotPoolRuntime:
    """Fixed VRAM slot pool + Variant-D expert forward (production path).

    SPARSE-2: parameterized by ExpertLayout; defaults to Q80_LAYOUT so all
    existing Q80 behavior is bit-for-bit unchanged."""

    def __init__(self, host: PrepackedHostPool, device: Any,
                 budget_gib: float = 4.0, layout=None) -> None:
        import torch

        from neural.moe.layout import Q80_LAYOUT

        self.L = layout or Q80_LAYOUT
        L = self.L
        self._act = resolve_activation(L)
        # optional per-expert biases (CAPACITY-4): flat [(n_layers*n_experts), N]
        # bf16 GPU tensors, attached via attach_expert_biases; None = no biases
        # (all pre-existing layouts), which leaves every kernel sequence and
        # captured graph byte-identical to before.
        self.bias_gu = None
        self.bias_dn = None
        self.host = host
        self.device = device
        self.capacity = max(L.top_k + 1, int(budget_gib * GIB) // L.slot_bytes)
        self.c = RuntimeCounters()
        host.c = self.c
        self.pool = torch.empty((self.capacity, L.slot_bytes), dtype=torch.uint8,
                                device=device)
        self.slot_of: "OrderedDict[tuple[int,int], int]" = OrderedDict()
        self.free = list(range(self.capacity))
        g_end = L.gate_pack_bytes
        d_end = g_end + L.down_pack_bytes
        gs_end = d_end + L.gate_saz_bytes
        cap = self.capacity
        kt = L.inner_k_tiles
        # prefill dequant scratch (Part 2): allocated on first use, reused by
        # every expert; bounded and transient, never a store expansion
        self._deq_scratch: dict[str, Any] = {}
        repr_ = getattr(L, "weight_repr", "int4_g32")
        self._mxfp4 = L.is_mxfp4        # fp4 codes + E8M0 scales, either scale layout (layout.MXFP4_REPRS)
        # True: scale rows are packed (base + 4-bit deltas, [cap, N, GK//2 + 1] views); the Triton launchers
        # infer that from the view, the CPU kernels must be told (cpu_prefill.bind_scale_layout, server.py)
        self._scale_packed = self._mxfp4 and L.scale_layout_mode == 1
        self._int8pc = repr_ == "int8_pc"
        self._int6 = repr_ == "int6_g32"
        if self._int6:
            g_end_ = L.gate_pack_bytes
            d_end_ = g_end_ + L.down_pack_bytes
            gs_end_ = d_end_ + L.gate_saz_bytes
            self.gate_codes = self.pool[:, :g_end_].view(
                cap, L.gate_up_n, L.hidden // 32, 24)
            self.down_codes = self.pool[:, g_end_:d_end_].view(
                cap, L.down_n, L.down_k // 32, 24)
            self.gate_saz6 = self.pool[:, d_end_:gs_end_].view(
                torch.bfloat16).view(cap, L.gate_up_n, L.hidden // 32, 2)
            self.down_saz6 = self.pool[:, gs_end_:].view(
                torch.bfloat16).view(cap, L.down_n, L.down_k // 32, 2)
            self.gate_pack = self.down_pack = None
            self.gate_saz = self.down_saz = None
            return
        if self._int8pc:
            # int8 per-channel views: codes as int8, one bf16 scale per row
            g_end_ = L.gate_pack_bytes
            d_end_ = g_end_ + L.down_pack_bytes
            gs_end_ = d_end_ + L.gate_saz_bytes
            self.gate_codes = self.pool[:, :g_end_].view(torch.int8).view(
                cap, L.gate_up_n, L.hidden)
            self.down_codes = self.pool[:, g_end_:d_end_].view(torch.int8).view(
                cap, L.down_n, L.down_k)
            self.gate_scales = self.pool[:, d_end_:gs_end_].view(
                torch.bfloat16).view(cap, L.gate_up_n)
            self.down_scales = self.pool[:, gs_end_:].view(
                torch.bfloat16).view(cap, L.down_n)
            self.gate_pack = self.down_pack = None
            self.gate_saz = self.down_saz = None
            return
        if self._mxfp4:
            # source-code views (u8): blocks [cap, N, K/32, 16], scales [cap, N, K/32] (raw) or
            # [cap, N, K/64 + 1] (packed); one shared builder so the runtime and its tests cannot diverge
            from neural.moe.layout import mxfp4_pool_views

            (self.gate_blocks, self.down_blocks,
             self.gate_scales, self.down_scales) = mxfp4_pool_views(self.pool, L)
            self.gate_pack = self.down_pack = None
            self.gate_saz = self.down_saz = None
            return
        self.gate_pack = self.pool[:, :g_end].view(torch.int32).view(
            cap, L.gate_up_n // 8, L.hidden // (kt * 16), 32, kt // 2)
        self.down_pack = self.pool[:, g_end:d_end].view(torch.int32).view(
            cap, L.down_n // 8, L.down_k // (kt * 16), 32, kt // 2)
        self.gate_saz = self.pool[:, d_end:gs_end].view(torch.bfloat16).view(
            cap, L.hidden // L.int4_group, L.gate_up_n, 2)
        self.down_saz = self.pool[:, gs_end:].view(torch.bfloat16).view(
            cap, L.down_k // L.int4_group, L.down_n, 2)

    # ---- residency (conventional LRU, expert granularity) ----
    def _ensure(self, layer: int, ids: list[int]) -> list[int]:
        misses = []
        for e in ids:
            k = (layer, e)
            self.c.ensure_calls += 1
            if k in self.slot_of:
                self.slot_of.move_to_end(k)
                self.c.hits += 1
            else:
                self.c.misses += 1
                misses.append(e)
        while len(self.slot_of) + len(misses) > self.capacity and self.slot_of:
            _, s = self.slot_of.popitem(last=False)
            self.free.append(s)
            self.c.evictions += 1
        return misses

    def _admit(self, layer: int, e: int) -> None:
        import torch

        buf = self.host.get(layer, e)
        s = self.free.pop()
        self.pool[s].copy_(buf, non_blocking=True)  # ONE contiguous H2D
        ev = torch.cuda.Event()
        ev.record()
        self.host.note_h2d((layer, e), ev)
        self.c.h2d_ops += 1
        self.c.h2d_bytes += self.L.slot_bytes
        self.slot_of[(layer, e)] = s

    def relocate_all_to_cpu(self) -> None:
        import torch

        self.slot_of.clear()
        self.free = list(range(self.capacity))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def attach_expert_biases(self, bias_gu, bias_dn) -> None:
        """Attach per-expert biases (layouts with has_expert_bias). Shapes:
        bias_gu [(n_layers*n_experts), gate_up_n], bias_dn [..., down_n], bf16,
        moved to the pool device and held resident (tiny vs the slot pool)."""
        import torch

        L = self.L
        n = L.n_layers * L.n_experts
        assert tuple(bias_gu.shape) == (n, L.gate_up_n)
        assert tuple(bias_dn.shape) == (n, L.down_n)
        self.bias_gu = bias_gu.to(self.device, torch.bfloat16).contiguous()
        self.bias_dn = bias_dn.to(self.device, torch.bfloat16).contiguous()

    # ---- forward paths ----
    def forward_decode(self, layer: int, x, top_k_index, top_k_weights):
        """T==1 fast path: stacked gate_up + sorted sequential aggregation."""
        import torch
        import torch.nn.functional as F

        L = self.L
        ids = top_k_index[0].tolist()          # the one bounded materialization
        self.c.forced_syncs += 1
        for e in self._ensure(layer, ids):
            self._admit(layer, e)
        order = sorted(range(L.top_k), key=lambda j: ids[j])  # ascending id = old path order
        sl = [self.slot_of[(layer, ids[j])] for j in order]
        slt = torch.tensor(sl, dtype=torch.long, device=x.device)
        if self._int6:
            from neural.q80.int6_kernels import int6_gemv_slots

            gu = int6_gemv_slots(x, self.gate_codes, self.gate_saz6,
                                 slt).view(L.top_k, L.gate_up_n)
        elif self._int8pc:
            from neural.q80.int8_kernels import int8_gemv_slots

            gu = int8_gemv_slots(x, self.gate_codes, self.gate_scales,
                                 slt).view(L.top_k, L.gate_up_n)
        elif self._mxfp4:
            from neural.q80.mxfp4_kernels import mxfp4_gemm

            gu = mxfp4_gemm(x, self.gate_blocks, self.gate_scales,
                            slt).view(L.top_k, L.gate_up_n)
        else:
            kt = L.inner_k_tiles
            gp = self.gate_pack.index_select(0, slt).view(
                -1, L.hidden // (kt * 16), 32, kt // 2)
            gm = self.gate_saz.index_select(0, slt).permute(1, 0, 2, 3).reshape(
                L.hidden // L.int4_group, -1, 2).contiguous()
            gate_up = torch.ops.aten._weight_int4pack_mm(x, gp, L.int4_group, gm)
            gu = gate_up.view(L.top_k, L.gate_up_n)
        if self.bias_gu is not None:
            gid = torch.tensor([layer * L.n_experts + ids[j] for j in order],
                               dtype=torch.long, device=x.device)
            gu = gu + self.bias_gu.index_select(0, gid)
        final = torch.zeros_like(x)
        for r, j in enumerate(order):
            h = self._act(gu[r:r + 1])
            if self._int6:
                from neural.q80.int6_kernels import int6_gemv_slots

                out = int6_gemv_slots(h, self.down_codes, self.down_saz6,
                                      slt[r:r + 1]).view(1, L.down_n)
            elif self._int8pc:
                from neural.q80.int8_kernels import int8_gemv_slots

                out = int8_gemv_slots(h, self.down_codes, self.down_scales,
                                      slt[r:r + 1]).view(1, L.down_n)
            elif self._mxfp4:
                from neural.q80.mxfp4_kernels import mxfp4_gemm

                out = mxfp4_gemm(h, self.down_blocks, self.down_scales,
                                 slt[r:r + 1]).view(1, L.down_n)
            else:
                out = torch.ops.aten._weight_int4pack_mm(
                    h, self.down_pack[sl[r]], L.int4_group, self.down_saz[sl[r]])
            if self.bias_dn is not None:
                # index_select (not scalar indexing): stays device-side, no
                # host sync — required for CUDA-graph capture legality
                out = out + self.bias_dn.index_select(0, gid[r:r + 1])
            w = top_k_weights[0:1, j:j + 1].to(out.dtype)
            final = final + out * w                     # sequential, sorted order
        return final

    def forward_prefill(self, layer: int, x, top_k_index, top_k_weights):
        """T>1 path: per-unique-expert (ascending) GEMMs + index_add_ (no syncs
        beyond the single tolist)."""
        import torch
        import torch.nn.functional as F

        rows = top_k_index.tolist()             # [T][K] — one sync
        self.c.forced_syncs += 1
        per_expert: dict[int, list[tuple[int, int]]] = {}
        for t, ids in enumerate(rows):
            for j, e in enumerate(ids):
                per_expert.setdefault(e, []).append((t, j))
        for e in self._ensure(layer, sorted(per_expert.keys())):
            self._admit(layer, e)
        final = torch.zeros_like(x)
        for e in sorted(per_expert.keys()):
            tj = per_expert[e]
            t_idx = torch.tensor([t for t, _ in tj], dtype=torch.long, device=x.device)
            j_idx = torch.tensor([j for _, j in tj], dtype=torch.long, device=x.device)
            state = x.index_select(0, t_idx)
            s = self.slot_of[(layer, e)]
            gate_up = self._one_expert_gemm(state, s, "gate")
            if self.bias_gu is not None:
                gate_up = gate_up + self.bias_gu[layer * self.L.n_experts + e]
            h = self._act(gate_up)
            out = self._one_expert_gemm(h, s, "down")
            if self.bias_dn is not None:
                out = out + self.bias_dn[layer * self.L.n_experts + e]
            w = top_k_weights[t_idx, j_idx, None].to(out.dtype)
            final.index_add_(0, t_idx, (out * w).to(final.dtype))
        return final

    def _scratch(self, key: str, n: int, k: int, device):
        import torch

        buf = self._deq_scratch.get(key)
        if buf is None or buf.shape != (n, k):
            buf = torch.empty(n, k, dtype=torch.bfloat16, device=device)
            self._deq_scratch[key] = buf
        return buf

    def scratch_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self._deq_scratch.values())

    def _one_expert_gemm(self, x, s: int, which: str):
        """Single-expert GEMM, representation-dispatched (prefill paths)."""
        import torch

        L = self.L
        if self._int6:
            from neural.q80.int6_kernels import dequant_int6

            # ONE bf16 scratch per stage, reused by every expert of every
            # layer (gate 4 MiB + down 2 MiB total). This is a transient
            # working buffer, NOT a copy of the store: the resident INT6
            # representation and its memory economics are unchanged.
            if which == "gate":
                w = dequant_int6(self.gate_codes[s], self.gate_saz6[s],
                                 out=self._scratch("gate", L.gate_up_n, L.hidden,
                                                   x.device))
            else:
                w = dequant_int6(self.down_codes[s], self.down_saz6[s],
                                 out=self._scratch("down", L.down_n, L.down_k,
                                                   x.device))
            return x @ w.t()
        if self._int8pc:
            # prefill: transient dequant + cuBLAS (equivalence-bounded vs the
            # GEMV decode path, cf. q80_decode_path_equivalence methodology)
            if which == "gate":
                w = (self.gate_codes[s].to(torch.bfloat16)
                     * self.gate_scales[s][:, None])
            else:
                w = (self.down_codes[s].to(torch.bfloat16)
                     * self.down_scales[s][:, None])
            return x @ w.t()
        if self._mxfp4:
            from neural.q80.mxfp4_kernels import mxfp4_gemm

            slt = torch.tensor([s], dtype=torch.long, device=x.device)
            if which == "gate":
                return mxfp4_gemm(x, self.gate_blocks, self.gate_scales,
                                  slt).view(x.shape[0], L.gate_up_n)
            return mxfp4_gemm(x, self.down_blocks, self.down_scales,
                              slt).view(x.shape[0], L.down_n)
        if which == "gate":
            return torch.ops.aten._weight_int4pack_mm(
                x, self.gate_pack[s], L.int4_group, self.gate_saz[s])
        return torch.ops.aten._weight_int4pack_mm(
            x, self.down_pack[s], L.int4_group, self.down_saz[s])


def install_q80_expert_runtime(runtime: Q80SlotPoolRuntime, model: Any,
                               *, timers: Any = None) -> tuple[list, dict[str, Any]]:
    """Patch every layer's experts.forward with the slot-pool Variant-D path."""
    installed: list[tuple[Any, Any]] = []
    layers = model.model.layers

    def make(layer_id: int):
        def forward(hidden_states, top_k_index, top_k_weights):
            if timers is not None:
                timers.gpu_start("expert_path")
            if hidden_states.shape[0] == 1:
                out = runtime.forward_decode(layer_id, hidden_states,
                                             top_k_index, top_k_weights)
            else:
                out = runtime.forward_prefill(layer_id, hidden_states,
                                              top_k_index, top_k_weights)
            if timers is not None:
                timers.gpu_stop("expert_path")
            return out
        return forward

    for li in range(N_LAYERS):
        experts = layers[li].mlp.experts
        installed.append((experts, experts.forward))
        experts.forward = make(li)
    backend = {
        "runtime": "Q80-10 slot-pool Variant-D",
        "path2_active": True,
        "int4_backend": "aten::_weight_int4pack_mm (stacked gate_up + per-expert down)",
        "bf16_reconstruction_on_compute_path": False,
        "runtime_int4_converts_per_decode": 0,
        "cache_policy": "lru_expert_granularity",
    }
    return installed, backend


def uninstall_q80_expert_runtime(installed: list) -> None:
    for mod, orig in installed:
        mod.forward = orig
    installed.clear()


# ===========================================================================
# Q80-11 V2 runtime — additive classes; the V1 classes above are unchanged and
# remain the OLD baseline for same-harness comparison.
# ===========================================================================

class PrepackedHostPoolV2(PrepackedHostPool):
    """Two-tier host pool: pinned LRU + pageable overflow LRU.

    Pinned evictions DEMOTE the prepacked bytes to the pageable tier (one host
    memcpy) instead of discarding them; pageable hits PROMOTE back to pinned.
    SSD + GPU convert are paid only on true first touch (or after pageable
    eviction). This is what makes 'the 45 GiB pool fits RAM' actually true for
    conversation-scale working sets: default 10 GiB pinned + 26 GiB pageable.
    """

    def __init__(self, store: Q80Int4DiskStore, device: Any,
                 pinned_budget_gib: float = 10.0,
                 pageable_budget_gib: float = 26.0, layout=None) -> None:
        super().__init__(store, device, pinned_budget_gib=pinned_budget_gib,
                         layout=layout)
        self.pageable_capacity = max(0, int(pageable_budget_gib * GIB) // self.L.slot_bytes)
        self._pageable: "OrderedDict[tuple[int,int], Any]" = OrderedDict()
        self._pageable_free: list = []
        # Deferred demotion (opt-in). The pinned->pageable memcpy on eviction
        # was measured at ~15 ms per decode token in the fast regime - about a
        # sixth of the whole token - and NOTHING reads the demoted copy until
        # that row is requested again. Moving the memcpy off the critical path
        # is therefore a pure latency win, not a policy change: the same bytes
        # end up in the same tier in the same order.
        self.deferred_demote = False
        self._demote_pending: dict = {}      # key -> pinned buf being copied
        self._demote_q: Any = None
        self._done_q: Any = None
        self._demote_thread: Any = None
        self._demote_error: Any = None
        # Engagement counter. With deferred demotion ON, host_demote_ms goes to
        # ~0 by design, so a zero there proves nothing - it looks identical to
        # "no evictions happened". This counts demotions that actually LANDED,
        # which is what a correctness run must assert.
        self.deferred_demotes_completed = 0

    def enable_deferred_demote(self) -> None:
        import queue
        import threading

        import torch

        if self._demote_thread is not None:
            self.deferred_demote = True
            return
        self._demote_q = queue.Queue()
        self._done_q = queue.Queue()

        # The worker does ONE thing: the memcpy. It never touches _pageable,
        # _pageable_free or _free. An earlier version had it insert into
        # _pageable under a lock while the main thread evicted from the same
        # OrderedDict WITHOUT that lock - a real race that would have produced
        # wrong bytes under a flag I was about to measure with. All tier
        # mutation is now single-threaded: the main thread drains _done_q.
        def worker():
            # The pool's buffers are allocated under torch.inference_mode(), so
            # they are INFERENCE TENSORS: an in-place write to one from a thread
            # that is not itself in inference mode raises. The decode loop runs
            # under inference_mode, which is why the synchronous version never
            # hit this - the failure is created purely by moving the copy to
            # another thread, and it killed the worker on the first eviction.
            with torch.inference_mode():
                while True:
                    item = self._demote_q.get()
                    if item is None:
                        return
                    key, buf, dst = item
                    try:
                        dst.copy_(buf)       # torch CPU copy releases the GIL
                    except BaseException as exc:   # never die silently
                        self._demote_error = exc
                        self._done_q.put((key, buf, None))
                        continue
                    self._done_q.put((key, buf, dst))

        self._demote_thread = threading.Thread(target=worker, daemon=True,
                                               name="q80-demote")
        self._demote_thread.start()
        self.deferred_demote = True

    def _drain_demotes(self) -> None:
        """Main thread only. Land completed demotions into the pageable tier."""
        q = self._done_q
        while q is not None and not q.empty():
            try:
                key, buf, dst = q.get_nowait()
            except Exception:
                return
            self._demote_pending.pop(key, None)
            if dst is None:                  # worker failed on this row
                raise RuntimeError(
                    "deferred demotion worker failed; the measurement is "
                    f"invalid: {self._demote_error!r}")
            if key not in self._resident:
                self._pageable[key] = dst
            else:                            # re-admitted while in flight
                self._pageable_free.append(dst)
            self._free.append(buf)
            self.deferred_demotes_completed += 1

    def _demote(self, key, buf) -> None:
        import torch

        if self.pageable_capacity <= 0:
            self._free.append(buf)
            return
        self.note("demote")
        while len(self._pageable) >= self.pageable_capacity:
            _, old = self._pageable.popitem(last=False)
            self._pageable_free.append(old)
            self.note("evict_pageable")
        dst = self._pageable_free.pop() if self._pageable_free else torch.empty(
            self.L.slot_bytes, dtype=torch.uint8)
        if self.deferred_demote:
            # The buffer stays OURS until the copy lands: it is parked in
            # _demote_pending (where get() finds it and can serve from it -
            # both uses are reads) and returns to _free only when the MAIN
            # thread drains the completion queue.
            self._demote_pending[key] = buf
            self._demote_q.put((key, buf, dst))
            return
        dst.copy_(buf)
        self._pageable[key] = dst
        self._free.append(buf)

    def get(self, layer: int, expert: int):
        key = (layer, expert)
        buf = self._resident.get(key)
        if buf is not None:
            self._resident.move_to_end(key)
            return buf
        # pinned miss -> make room (demote victim), then fill from pageable or build
        while len(self._resident) >= self.capacity:
            vk, vbuf = self._resident.popitem(last=False)
            ev = self._events.pop(vk, None)
            if ev is not None and not ev.query():
                ev.synchronize()
            self._demote(vk, vbuf)
            if self.c is not None:
                self.c.host_pinned_evictions += 1
        pg = self._pageable.pop(key, None)
        if pg is not None:
            import torch

            buf = self._free.pop() if self._free else torch.empty(
                self.L.slot_bytes, dtype=torch.uint8, pin_memory=True)
            buf.copy_(pg)          # pageable -> pinned promote (host memcpy)
            self._pageable_free.append(pg)
        else:
            buf = self._build(layer, expert)  # SSD + one-time GPU convert
        self._resident[key] = buf
        return buf

    def peek_tier(self, layer: int, expert: int) -> str:
        key = (layer, expert)
        if key in self._resident:
            return "pinned"
        if key in self._pageable:
            return "pageable"
        return "none"


class _ExpertGraph:
    """ONE CUDA graph for the decode-path expert compute of ANY layer.

    Layer identity is fully carried by tensor VALUES (slot indices + routing
    weights + input hidden state), so a single captured graph serves all 48
    layers: gather packs from the slot pool -> stacked gate_up int4 GEMM ->
    SiLU*up -> 10 down int4 GEMMs -> weighted sum (ascending-id sequential
    order preserved OUTSIDE the graph by pre-sorting the inputs).
    """

    def __init__(self, rt: "Q80SlotPoolRuntimeV2") -> None:
        import torch

        self.rt = rt
        L = rt.L
        dev = rt.device
        self.x = torch.zeros(1, L.hidden, dtype=torch.bfloat16, device=dev)
        self.slots = torch.zeros(L.top_k, dtype=torch.long, device=dev)
        self.w = torch.zeros(L.top_k, dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros(1, L.hidden, dtype=torch.bfloat16, device=dev)
        self._slots_pin = torch.zeros(L.top_k, dtype=torch.long, pin_memory=True)
        self._order_pin = torch.zeros(L.top_k, dtype=torch.long, pin_memory=True)
        self._order_dev = torch.zeros(L.top_k, dtype=torch.long, device=dev)
        # global-expert-id buffer, consumed inside the capture ONLY when the
        # runtime has biases attached (bias-free layouts capture the exact
        # pre-existing kernel sequence)
        self._gids_pin = torch.zeros(L.top_k, dtype=torch.long, pin_memory=True)
        self.gids = torch.zeros(L.top_k, dtype=torch.long, device=dev)
        self.graph = None

    def _compute(self) -> None:
        import torch
        import torch.nn.functional as F

        rt = self.rt
        L = rt.L
        if rt._int6:
            from neural.q80.int6_kernels import int6_gemv_slots

            gu = int6_gemv_slots(self.x, rt.gate_codes, rt.gate_saz6,
                                 self.slots).view(L.top_k, L.gate_up_n)
        elif rt._int8pc:
            from neural.q80.int8_kernels import int8_gemv_slots

            gu = int8_gemv_slots(self.x, rt.gate_codes, rt.gate_scales,
                                 self.slots).view(L.top_k, L.gate_up_n)
        elif rt._mxfp4:
            from neural.q80.mxfp4_kernels import mxfp4_gemv

            gu = mxfp4_gemv(self.x, rt.gate_blocks, rt.gate_scales,
                            self.slots, block_n=32, block_g=4, num_warps=8
                            ).view(L.top_k, L.gate_up_n)
        else:
            kt = L.inner_k_tiles
            gp = rt.gate_pack.index_select(0, self.slots).view(
                -1, L.hidden // (kt * 16), 32, kt // 2)
            gm = rt.gate_saz.index_select(0, self.slots).permute(1, 0, 2, 3).reshape(
                L.hidden // L.int4_group, -1, 2).contiguous()
            gate_up = torch.ops.aten._weight_int4pack_mm(self.x, gp, L.int4_group, gm)
            gu = gate_up.view(L.top_k, L.gate_up_n)
        if rt.bias_gu is not None:
            gu = gu + rt.bias_gu.index_select(0, self.gids)
        if not (rt._mxfp4 or rt._int8pc or rt._int6):
            dp = rt.down_pack.index_select(0, self.slots)
            dm = rt.down_saz.index_select(0, self.slots)
        acc = torch.zeros_like(self.x)
        _fast = rt._mxfp4 or rt._int8pc or rt._int6
        h_all = rt._act(gu) if _fast else None  # rowwise == per-row (elementwise)
        y_all = None
        if rt._int6:
            from neural.q80.int6_kernels import int6_gemv_slots

            y_all = int6_gemv_slots(h_all, rt.down_codes, rt.down_saz6,
                                    self.slots, per_expert_x=True)
        elif rt._int8pc:
            from neural.q80.int8_kernels import int8_gemv_slots

            y_all = int8_gemv_slots(h_all, rt.down_codes, rt.down_scales,
                                    self.slots, per_expert_x=True)
        elif rt._mxfp4:
            from neural.q80.mxfp4_kernels import mxfp4_gemv

            y_all = mxfp4_gemv(h_all, rt.down_blocks, rt.down_scales,
                               self.slots, per_expert_x=True, block_n=32,
                               block_g=4, num_warps=8)     # ONE launch, 4 GEMVs
        for r in range(L.top_k):
            if _fast:
                y = y_all[r:r + 1]
            else:
                h = rt._act(gu[r:r + 1])
                y = torch.ops.aten._weight_int4pack_mm(
                    h, dp[r], L.int4_group, dm[r].contiguous())
            if rt.bias_dn is not None:
                y = y + rt.bias_dn.index_select(0, self.gids[r:r + 1])
            acc = acc + y * self.w[r]
        self.out.copy_(acc)

    def capture(self) -> None:
        import torch

        self._compute()          # allocator warm-up outside capture
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._compute()

    def run(self, x, slot_list: list[int], order: list[int], top_k_weights,
            gid_list: list[int] | None = None) -> Any:
        import torch

        self.x.copy_(x[0:1])
        for i, s in enumerate(slot_list):
            self._slots_pin[i] = s
            self._order_pin[i] = order[i]
        self.slots.copy_(self._slots_pin, non_blocking=True)
        self._order_dev.copy_(self._order_pin, non_blocking=True)
        if gid_list is not None:
            for i, g in enumerate(gid_list):
                self._gids_pin[i] = g
            self.gids.copy_(self._gids_pin, non_blocking=True)
        self.w.copy_(top_k_weights[0].index_select(0, self._order_dev).to(torch.bfloat16))
        self.graph.replay()
        return self.out


class Q80SlotPoolRuntimeV2(Q80SlotPoolRuntime):
    """V2: prefill streaming (scratch slots, decode cache preserved) + optional
    single CUDA graph for decode expert compute. LRU decode semantics, expert
    representation, routing and INT4 math are identical to V1."""

    N_SCRATCH = 16  # reserved scratch slots for prefill streaming

    def __init__(self, host: PrepackedHostPoolV2, device: Any,
                 budget_gib: float = 4.0, use_graph: bool = True,
                 prefill_streaming: bool = True, overlap_h2d: bool = True,
                 layout=None) -> None:
        import torch

        super().__init__(host, device, budget_gib=budget_gib, layout=layout)
        # carve scratch slots out of the tail of the pool (bounded for tiny pools)
        self.N_SCRATCH = min(self.N_SCRATCH, max(4, (self.capacity - TOP_K - 1) // 4))
        self.scratch_slots = list(range(self.capacity - self.N_SCRATCH, self.capacity))
        # Q80-12: admission copies ride a dedicated copy stream; compute waits
        # GPU-side (no host sync) via an event before the expert graph replay.
        self.overlap_h2d = overlap_h2d
        self._copy_stream = torch.cuda.Stream() if overlap_h2d else None
        self._copy_done = torch.cuda.Event() if overlap_h2d else None
        # pre-built row views: pool[s] fancy-indexing cost ~20-30us/miss in Python
        self._rows = [self.pool[i] for i in range(self.pool.shape[0])]
        self.free = [s for s in self.free if s < self.capacity - self.N_SCRATCH]
        self.capacity -= self.N_SCRATCH
        self._scratch_i = 0
        self.use_graph = use_graph
        self.prefill_streaming = prefill_streaming
        self._graph: _ExpertGraph | None = None
        self.graph_replays = 0
        self.eager_decode_calls = 0
        self.graph_capture_failures = 0
        self.meta_fill_numpy = 0
        self.meta_fill_setitem = 0
        self.last_capture_error: str | None = None
        # Part 1: when True, an eager-served decode raises GraphFallbackError
        # instead of silently degrading. Benchmarks set this; research controls
        # (use_graph=False by choice) leave it off.
        self.require_graph = False
        self.prefill_stream_loads = 0
        # Q80-HOSTTIER-SCAN-RESISTANCE: authentic (role, layer, experts) access
        # trace for the oracle replay. None = disabled (zero overhead).
        self.access_trace = None

    # ---- Part 1: dispatch-path observability ----
    KERNEL_NAMES = {"int6_g32": "int6_gemv_slots (Triton)",
                    "int8_pc": "int8_gemv_slots (Triton)",
                    "mxfp4_g32": "mxfp4_gemv (Triton)",
                    "mxfp4_g32_ps4": "mxfp4_gemv (Triton, packed scales)",
                    "int4_g32": "aten::_weight_int4pack_mm"}

    def dispatch_snapshot(self) -> dict[str, int]:
        """Counter snapshot for WINDOWED admissibility checks.

        Cumulative counters cannot gate a measurement: warm-up replays make
        ``graph_replays > 0`` true regardless of what the measured window did,
        and the compiled-GDN wrapper legitimately takes its fallback during
        prefill (T>1). Only deltas across the decode window mean anything.
        """
        g = getattr(self, "_gdn_stats", None) or {}
        a = getattr(self, "_attn_stats", None)
        a = a.as_dict() if a is not None else {}
        return {
            "expert_replays": self.graph_replays,
            "expert_eager": self.eager_decode_calls,
            "expert_capture_failures": self.graph_capture_failures,
            "gdn_compiled": g.get("compiled_calls", 0),
            "gdn_fallback": g.get("fallback_calls", 0),
            "attn_replays": a.get("replays", 0),
            "attn_module_fallbacks": a.get("module_fallbacks", 0),
            "attn_capture_failures": a.get("capture_failures", 0),
            "host_pageable_serves": getattr(self.host, "pageable_serves", 0),
            # which meta-fill branch actually executed, counted per call
            "meta_fill_numpy": getattr(self, "meta_fill_numpy", 0),
            "meta_fill_setitem": getattr(self, "meta_fill_setitem", 0),
            # Evicting a pinned host buffer whose H2D is still in flight blocks
            # the CPU on a GPU event. The code called that "rare"; nothing
            # counted it, and the conversation regime evicts ~13 pinned rows
            # per decode token. Surfaced so "rare" can be checked.
            "host_evict_sync_waits": getattr(self.host, "evict_sync_waits", 0),
            "host_evict_sync_ms": round(getattr(self.host, "evict_sync_ms", 0.0), 1),
            "host_build_ms": round(getattr(self.host, "build_ms", 0.0), 1),
            "host_demote_ms": round(getattr(self.host, "demote_ms", 0.0), 1),
            "deferred_demote": bool(getattr(self.host, "deferred_demote", False)),
            "deferred_demotes_completed": getattr(
                self.host, "deferred_demotes_completed", 0),
        }

    def dispatch_report(self) -> dict[str, Any]:
        """Everything needed to decide whether a measurement is admissible.

        Every benchmark result in this project must carry these fields: the
        Q80-INT6-PROBE bug was invisible precisely because throughput numbers
        did not say which path produced them.

        Adversarial review of the first version of this fix found three
        SIBLING degradation channels of the same class — compiled GDN layers,
        attention CUDA graphs and pageable-tier H2D — each able to cost
        10-20% while the expert-path counters stayed perfectly clean. They are
        all reported here, and gated by assert_graphed().
        """
        repr_ = getattr(self.L, "weight_repr", "int4_g32")
        # Q80-GRAPH-PREAMBLE: derive the variant from the ACTUAL graph class.
        # The previous hasattr(_meta_pin) test was true for _ExpertGraph (after
        # slim_expert_graph) AND for _ExpertGraphV4, so both reported "v3_slim"
        # and no artifact could say which host preamble produced a number - the
        # same blind spot adversarial review found in the meta-fill A/B.
        # enable_admission_v4 installs its own forward_decode and parks the
        # capture on rt._graph_v4, leaving rt._graph None - so reading only
        # _graph reported "none" for every admission_v4 runtime (all of
        # gpt-oss) while 36 graphs per token were replaying. Third instance of
        # this blind-spot class, so resolve the graph the SAME way the decode
        # path does rather than from one attribute.
        graph = self._graph if self._graph is not None else getattr(
            self, "_graph_v4", None)
        if graph is None:
            variant = "none"
        else:
            cls = type(graph).__name__
            if cls == "_ExpertGraphV4":
                variant = ("v4_fused_meta+ring" if getattr(self, "admission_v4", False)
                           else "v4_fused_meta")
            elif hasattr(graph, "_meta_pin"):
                variant = "v3_slim"
            else:
                variant = "v2"
        c = self.c.as_dict()
        return {
            # STRING, so it lives here and NOT in dispatch_snapshot(), whose
            # values are differenced across the decode window.
            "meta_fill_variant": (
                "v3_slim_numpy" if getattr(self, "meta_fill_numpy", 0)
                and not getattr(self, "meta_fill_setitem", 0)
                else "v3_slim_setitem" if getattr(self, "meta_fill_setitem", 0)
                and not getattr(self, "meta_fill_numpy", 0)
                else "mixed_or_unused"),
            "graph_replays": self.graph_replays,
            "eager_decode_calls": self.eager_decode_calls,
            "graph_capture_failures": self.graph_capture_failures,
            "last_capture_error": self.last_capture_error,
            "graph_variant": variant,
            "graph_class": type(graph).__name__ if graph is not None else None,
            "use_graph": self.use_graph,
            "require_graph": self.require_graph,
            "expert_kernel": self.KERNEL_NAMES.get(repr_, repr_),
            "weight_repr": repr_,
            "layout": self.L.name,
            "slot_bytes": self.L.slot_bytes,
            "pool_slots": self.capacity,
            "hit_rate": c["vram_hit_rate"],
            "h2d_gib": c["h2d_gib"],
            "prefill_kernel": self.prefill_kernel_name(),
            "cold_io": self._cold_io_report(),
            **self.dispatch_snapshot(),
            "gdn_stats": dict(getattr(self, "_gdn_stats", None) or {}),
            "attn_stats": (self._attn_stats.as_dict()
                           if getattr(self, "_attn_stats", None) is not None
                           else None),
            "staged_pageable_through_ring": getattr(self, "c_staged", 0),
        }

    def _cold_io_report(self) -> dict[str, Any]:
        pool = getattr(self, "cold_io", None)
        if pool is None:
            return {"attached": False}
        return {"attached": True,
                "overlap": bool(getattr(self, "cold_io_overlap", False)),
                "decode_hook": bool(getattr(self, "cold_io_decode", False)),
                "stats": dict(pool.stats)}

    def prefill_kernel_name(self) -> str:
        if self._int6:
            from neural.q80.int6_kernels import triton_dequant_available

            return ("dequant_int6_triton + cuBLAS" if triton_dequant_available()
                    else "dequant_int6_torch_ops + cuBLAS")
        if self._int8pc:
            return "transient int8 dequant + cuBLAS"
        if self._mxfp4:
            return "mxfp4_gemm (Triton, packed scales)" if self._scale_packed else "mxfp4_gemm (Triton)"
        return "aten::_weight_int4pack_mm"

    def assert_graphed(self, before: dict[str, int] | None = None,
                       decode_tokens: int | None = None,
                       tolerance: float = 0.05) -> dict[str, Any]:
        """Admissibility gate for an accepted measurement (raises otherwise).

        With ``before`` (a dispatch_snapshot taken at the start of the decode
        window) and ``decode_tokens``, this checks the three optimized paths
        actually ran for the WHOLE window:

          * expert graph  — one replay per layer per token, zero eager calls
          * compiled GDN  — one compiled call per linear-attention layer per
            token, zero fallbacks (the T>1 prefill fallback is excluded by
            the windowing)
          * attention graphs — zero module fallbacks, zero capture failures

        Without ``before`` it degrades to the weak cumulative check, which is
        explicitly NOT sufficient to bless a measurement.
        """
        d = self.dispatch_report()
        problems = []
        if d["eager_decode_calls"] or d["graph_capture_failures"]:
            problems.append(f"expert path: eager={d['eager_decode_calls']} "
                            f"capture_failures={d['graph_capture_failures']} "
                            f"last_error={d['last_capture_error']}")
        if not d["graph_replays"]:
            problems.append("expert path: zero graph replays")
        if not self.require_graph:
            problems.append("require_graph was OFF: a fallback would not have "
                            "aborted this run")
        if before is not None:
            now = self.dispatch_snapshot()
            delta = {k: now[k] - before.get(k, 0) for k in now}
            for k in ("expert_eager", "expert_capture_failures", "gdn_fallback",
                      "attn_module_fallbacks", "attn_capture_failures"):
                if delta[k]:
                    problems.append(f"{k} fired {delta[k]}x inside the "
                                    "measured decode window")
            if decode_tokens:
                exp = decode_tokens * self.L.n_layers
                if delta["expert_replays"] < exp * (1 - tolerance):
                    problems.append(
                        f"expert replays {delta['expert_replays']} < expected "
                        f"~{exp} for {decode_tokens} tokens: part of the window "
                        "did not run on the graphed path")
                gdn = getattr(self, "_gdn_layers", None)
                if gdn:
                    expg = decode_tokens * gdn
                    if delta["gdn_compiled"] < expg * (1 - tolerance):
                        problems.append(
                            f"compiled-GDN calls {delta['gdn_compiled']} < "
                            f"expected ~{expg}")
            d["window_delta"] = delta
        # AN OPT-IN MECHANISM THAT SILENTLY DID NOT RUN is the failure class
        # that has cost this project the most: Q80-18's admission_v4 verdict,
        # Q80-21's cold_io_decode verdict, a quality gate that passed with the
        # staging ring at zero, and a whole pre-registered arm whose flag had
        # no live consumer on the shipped decode path.
        #
        # STRUCTURAL impossibility is a hard error - it is decidable without
        # any counter, so producing a number for it is never defensible.
        if getattr(self, "cold_io_decode", False) and not getattr(
                self, "admission_v4", False):
            problems.append(
                "cold_io_decode is ON but admission_v4 is OFF: the decode hook "
                "is read only inside forward_decode_v4, so the flag is INERT "
                "and this run is a control, not a cold_io_decode arm")
        # ENGAGEMENT is reported, not enforced: a window with nothing to stage
        # (short prompt, working set already pinned) is a legitimate zero. The
        # caller asserts engagement when its claim depends on it.
        cio = d.get("cold_io") or {}
        d["mechanism_engagement"] = {
            "admission_v4": {
                "enabled": bool(getattr(self, "admission_v4", False)),
                "staged_pageable_through_ring":
                    d.get("staged_pageable_through_ring", 0)},
            "cold_io_decode": {
                "enabled": bool(getattr(self, "cold_io_decode", False)),
                "decode_batches": (cio.get("stats") or {}).get("batches", 0)},
        }
        if problems:
            raise GraphFallbackError("measurement is NOT admissible: "
                                     + "; ".join(problems))
        return d

    def _admit_batch_overlapped(self, layer: int, misses: list[int]) -> None:
        """Q80-12 lever: lean bookkeeping + all miss copies on the copy stream.

        The copy stream waits on current compute (so evicted-slot reuse is
        ordered), copies every missed expert pinned->slot, records one event;
        the caller makes the compute stream wait GPU-side before the graph
        replay. No host synchronization anywhere."""
        import torch

        c = self.c
        slot_of = self.slot_of
        free = self.free
        need = len(misses)
        while len(slot_of) + need > self.capacity and slot_of:
            _, s = slot_of.popitem(last=False)
            free.append(s)
            c.evictions += 1
        cs = self._copy_stream
        rows = self._rows
        host_get = self.host.get
        note = self.host.note_h2d
        done = self._copy_done
        cs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cs):
            for e in misses:
                buf = host_get(layer, e)
                s = free.pop()
                rows[s].copy_(buf, non_blocking=True)
                slot_of[(layer, e)] = s
                note((layer, e), done)   # one shared completion event per batch
            done.record()
        c.h2d_ops += need
        c.h2d_bytes += need * self.L.slot_bytes

    # ---- decode: graph fast path with eager fallback ----
    def forward_decode(self, layer: int, x, top_k_index, top_k_weights):
        import torch

        if not self.use_graph:
            self.eager_decode_calls += 1
            if self.require_graph:
                raise GraphFallbackError(
                    "decode served EAGER while require_graph is set "
                    f"(capture failures={self.graph_capture_failures}, "
                    f"last error: {self.last_capture_error})")
            return super().forward_decode(layer, x, top_k_index, top_k_weights)
        ids = top_k_index[0].tolist()
        self.c.forced_syncs += 1
        self.host.access_role = "decode"
        if self.access_trace is not None:
            self.access_trace.append((1, layer, tuple(ids)))
        if self.overlap_h2d:
            # lean ensure (inline, list-based)
            slot_of = self.slot_of
            misses = []
            c = self.c
            for e in ids:
                k = (layer, e)
                c.ensure_calls += 1
                if k in slot_of:
                    slot_of.move_to_end(k)
                    c.hits += 1
                else:
                    c.misses += 1
                    misses.append(e)
            if misses:
                self._admit_batch_overlapped(layer, misses)
                torch.cuda.current_stream().wait_event(self._copy_done)
        else:
            for e in self._ensure(layer, ids):
                self._admit(layer, e)
        order = sorted(range(self.L.top_k), key=lambda j: ids[j])
        sl = [self.slot_of[(layer, ids[j])] for j in order]
        if self._graph is None:
            # Q80-GRAPH-PREAMBLE: _ExpertGraphV4 resolves slot/order/gid from
            # VIEWS into one device meta buffer and does the weight gather
            # INSIDE the capture, so the host path per expert call drops from
            # {meta H2D, slots copy, order copy, [gids copy], x copy, gather,
            # w copy, replay} to {meta H2D, x copy, w_raw copy, replay}. The
            # kernel sequence, dtypes and summation order are unchanged, so it
            # is bit-exact. It existed only behind enable_admission_v4, which
            # bundled it with the staging ring - two unrelated mechanisms
            # behind one flag. USE_V4_GRAPH unbundles them.
            self._graph = (_ExpertGraphV4(self) if USE_V4_GRAPH
                           else _ExpertGraph(self))
            try:
                self._graph.capture()
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                self._graph = None
                self.use_graph = False
                self.graph_capture_failures += 1
                self.last_capture_error = f"{type(exc).__name__}: {exc}"
                if self.require_graph:
                    raise GraphFallbackError(
                        "expert-graph capture FAILED and require_graph is set: "
                        f"{self.last_capture_error}") from exc
                self.eager_decode_calls += 1
                return super().forward_decode(layer, x, top_k_index, top_k_weights)
        # bias-free layouts keep the original 5-arg call: external graph
        # overrides (hot_path.slim_expert_graph run_v3) predate gid_list
        gl = ([layer * self.L.n_experts + ids[j] for j in order]
              if self.bias_gu is not None else None)
        if isinstance(self._graph, _ExpertGraphV4):
            out = self._graph.run_v4(x, sl, order, top_k_weights, gl)
        elif gl is not None:
            out = self._graph.run(x, sl, order, top_k_weights, gl)
        else:
            out = self._graph.run(x, sl, order, top_k_weights)
        self.graph_replays += 1
        # clone: detach result from the static graph buffer before the next
        # layer's replay overwrites it (cheap [1,2048] copy; insurance against
        # any caller retaining the tensor)
        return out.clone()

    # ---- prefill: stream misses through scratch slots (no LRU pollution) ----
    def forward_prefill(self, layer: int, x, top_k_index, top_k_weights):
        import torch
        import torch.nn.functional as F

        if not self.prefill_streaming:
            return super().forward_prefill(layer, x, top_k_index, top_k_weights)
        rows = top_k_index.tolist()
        self.c.forced_syncs += 1
        self.host.access_role = "prefill"
        per_expert: dict[int, list[tuple[int, int]]] = {}
        for t, ids in enumerate(rows):
            for j, e in enumerate(ids):
                per_expert.setdefault(e, []).append((t, j))
        if self.access_trace is not None:
            self.access_trace.append((0, layer, tuple(sorted(per_expert))))
        # Q80-21: parallel cold-row prefill of the host tier (flag-gated; the
        # whole per-layer prefill demand is known up front — big batches).
        # Q80-22 (cold_io_overlap): the same reads submitted NON-blocking and
        # delivered in the loop's own sorted order — compute on earlier
        # experts hides later reads; accumulation order unchanged.
        cold = getattr(self, "cold_io", None)
        plan = None
        if cold is not None and getattr(self.host, "_prepacked_active", False):
            need = [e for e in sorted(per_expert)
                    if (layer, e) not in self.slot_of
                    and (layer, e) not in self.host._resident
                    and (layer, e) not in getattr(self.host, "_pageable", {})]
            if getattr(self, "cold_io_overlap", False):
                if len(need) >= 2:
                    plan = cold.plan(layer, need)
            else:
                cold.prefill_host(self.host, layer, need)
        final = torch.zeros_like(x)
        for e in sorted(per_expert.keys()):
            key = (layer, e)
            if plan is not None:
                plan.deliver(self.host, e)
            self.c.ensure_calls += 1
            if key in self.slot_of:                 # decode-cache hit: reuse, refresh
                self.slot_of.move_to_end(key)
                self.c.hits += 1
                s = self.slot_of[key]
            else:                                    # stream through scratch
                self.c.misses += 1
                s = self.scratch_slots[self._scratch_i]
                self._scratch_i = (self._scratch_i + 1) % self.N_SCRATCH
                buf = self.host.get(layer, e)
                self.pool[s].copy_(buf, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                self.host.note_h2d((layer, e), ev)  # guard pinned-buffer reuse
                self.c.h2d_ops += 1
                self.c.h2d_bytes += self.L.slot_bytes
                self.prefill_stream_loads += 1
            tj = per_expert[e]
            t_idx = torch.tensor([t for t, _ in tj], dtype=torch.long, device=x.device)
            j_idx = torch.tensor([j for _, j in tj], dtype=torch.long, device=x.device)
            state = x.index_select(0, t_idx)
            gate_up = self._one_expert_gemm(state, s, "gate")
            if self.bias_gu is not None:
                gate_up = gate_up + self.bias_gu[layer * self.L.n_experts + e]
            h = self._act(gate_up)
            out = self._one_expert_gemm(h, s, "down")
            if self.bias_dn is not None:
                out = out + self.bias_dn[layer * self.L.n_experts + e]
            w = top_k_weights[t_idx, j_idx, None].to(out.dtype)
            final.index_add_(0, t_idx, (out * w).to(final.dtype))
        return final


# ===========================================================================
# Q80-18 V4 admission — additive, flag-gated (enable_admission_v4). Semantics
# identical to the V2 overlapped path: same expert selection, same LRU order
# updates/evictions, same INT4 bytes to the same slot destinations, same
# sorted accumulation order. Three measured collapses:
#   S. pageable-tier sources are staged through a pinned ring (host memcpy)
#      so every H2D is a fast non-stalling pinned async copy (the Q80-18
#      profile showed ~30% pageable misses dragging PCIe to 13.5 GB/s and
#      stalling submission for ~12-15 ms/token)
#   B. pinned copies submit via ONE torch._foreach_copy_ batch per layer
#   G. _ExpertGraphV4 folds weight ordering/cast into the graph; slots/order
#      read straight from the fused meta buffer -> 3 copies + replay per
#      layer, and the per-layer defensive out.clone() is dropped (consumer
#      op is enqueued on the same stream before the next replay).
# ===========================================================================


class _ExpertGraphV4(_ExpertGraph):
    """Graph variant: order/weights resolved INSIDE the capture from static
    meta buffers. Kernel sequence of the GEMM/accumulation path is unchanged
    (same ops, same sorted order, same dtypes) — the w gather/cast simply
    moves inside the graph."""

    def __init__(self, rt: "Q80SlotPoolRuntimeV2") -> None:
        import torch

        super().__init__(rt)
        dev = rt.device
        K = rt.L.top_k
        self._meta_pin = torch.zeros(3 * K, dtype=torch.long, pin_memory=True)
        self._meta_dev = torch.zeros(3 * K, dtype=torch.long, device=dev)
        self.slots = self._meta_dev[:K]              # views: values updated by
        self._order_dev = self._meta_dev[K:2 * K]    # the single meta H2D
        self.gids = self._meta_dev[2 * K:]           # bias row ids (bias layouts)
        self.w_raw = torch.zeros(K, dtype=torch.bfloat16, device=dev)
        self._meta_np = self._meta_pin.numpy()   # view, no copy

    def _compute(self) -> None:
        import torch
        import torch.nn.functional as F

        rt = self.rt
        L = rt.L
        if rt._int6:
            from neural.q80.int6_kernels import int6_gemv_slots

            gu = int6_gemv_slots(self.x, rt.gate_codes, rt.gate_saz6,
                                 self.slots).view(L.top_k, L.gate_up_n)
        elif rt._int8pc:
            from neural.q80.int8_kernels import int8_gemv_slots

            gu = int8_gemv_slots(self.x, rt.gate_codes, rt.gate_scales,
                                 self.slots).view(L.top_k, L.gate_up_n)
        elif rt._mxfp4:
            from neural.q80.mxfp4_kernels import mxfp4_gemv

            gu = mxfp4_gemv(self.x, rt.gate_blocks, rt.gate_scales,
                            self.slots, block_n=32, block_g=4, num_warps=8
                            ).view(L.top_k, L.gate_up_n)
        else:
            kt = L.inner_k_tiles
            gp = rt.gate_pack.index_select(0, self.slots).view(
                -1, L.hidden // (kt * 16), 32, kt // 2)
            gm = rt.gate_saz.index_select(0, self.slots).permute(1, 0, 2, 3).reshape(
                L.hidden // L.int4_group, -1, 2).contiguous()
            gate_up = torch.ops.aten._weight_int4pack_mm(self.x, gp, L.int4_group, gm)
            gu = gate_up.view(L.top_k, L.gate_up_n)
        if rt.bias_gu is not None:
            gu = gu + rt.bias_gu.index_select(0, self.gids)
        if not (rt._mxfp4 or rt._int8pc or rt._int6):
            dp = rt.down_pack.index_select(0, self.slots)
            dm = rt.down_saz.index_select(0, self.slots)
        w = self.w_raw.index_select(0, self._order_dev)  # same gather, now in-graph
        acc = torch.zeros_like(self.x)
        _fast = rt._mxfp4 or rt._int8pc or rt._int6
        h_all = rt._act(gu) if _fast else None  # rowwise == per-row (elementwise)
        y_all = None
        if rt._int6:
            from neural.q80.int6_kernels import int6_gemv_slots

            y_all = int6_gemv_slots(h_all, rt.down_codes, rt.down_saz6,
                                    self.slots, per_expert_x=True)
        elif rt._int8pc:
            from neural.q80.int8_kernels import int8_gemv_slots

            y_all = int8_gemv_slots(h_all, rt.down_codes, rt.down_scales,
                                    self.slots, per_expert_x=True)
        elif rt._mxfp4:
            from neural.q80.mxfp4_kernels import mxfp4_gemv

            y_all = mxfp4_gemv(h_all, rt.down_blocks, rt.down_scales,
                               self.slots, per_expert_x=True, block_n=32,
                               block_g=4, num_warps=8)     # ONE launch, 4 GEMVs
        for r in range(L.top_k):
            if _fast:
                y = y_all[r:r + 1]
            else:
                h = rt._act(gu[r:r + 1])
                y = torch.ops.aten._weight_int4pack_mm(
                    h, dp[r], L.int4_group, dm[r].contiguous())
            if rt.bias_dn is not None:
                y = y + rt.bias_dn.index_select(0, self.gids[r:r + 1])
            acc = acc + y * w[r]
        self.out.copy_(acc)

    def run_v4(self, x, slot_list: list[int], order: list[int], top_k_weights,
               gid_list: list[int] | None = None):
        K = self.rt.L.top_k
        # same numpy-view fill the v3 path uses: 0.8 us against 73.2 us for the
        # element loop this replaced, identical bytes into the same buffer.
        # Counted, not inferred: without this the artifact reads
        # "mixed_or_unused" and cannot distinguish "v4 ran the numpy fill" from
        # "no meta fill happened at all".
        self.rt.meta_fill_numpy = getattr(self.rt, "meta_fill_numpy", 0) + 1
        mn = self._meta_np
        mn[:K] = slot_list
        mn[K:2 * K] = order
        if gid_list is not None:
            mn[2 * K:3 * K] = gid_list
        self._meta_dev.copy_(self._meta_pin, non_blocking=True)  # ONE meta H2D
        self.x.copy_(x[0:1])                          # D2D
        self.w_raw.copy_(top_k_weights[0])            # D2D (unordered; graph orders)
        self.graph.replay()
        return self.out                               # no clone (see class doc)


def enable_admission_v4(rt, n_stage: int = 32) -> None:
    """Install the V4 admission + graph path on a Q80SlotPoolRuntimeV2.

    Additive and reversible: rt.admission_v4 toggles between the production V2
    path and V4 at every call, enabling same-process interleaved A/B."""
    import torch

    rt._stage_ring = [torch.empty(rt.L.slot_bytes, dtype=torch.uint8, pin_memory=True)
                      for _ in range(n_stage)]
    rt._stage_events = [torch.cuda.Event() for _ in range(n_stage)]
    for ev in rt._stage_events:
        ev.record()                     # mark all ring buffers immediately free
    rt._stage_i = 0
    rt._n_stage = n_stage
    rt._graph_v4 = None
    rt.admission_v4 = False             # off until explicitly toggled
    rt.c_staged = 0                     # pageable sources staged through the ring

    orig_forward_decode = rt.forward_decode

    def forward_decode_v4(layer, x, top_k_index, top_k_weights):
        if not rt.admission_v4:
            return orig_forward_decode(layer, x, top_k_index, top_k_weights)
        c = rt.c
        ids = top_k_index[0].tolist()
        c.forced_syncs += 1
        # These two lines are NOT optional bookkeeping. Without them the role
        # tag stays stuck on "prefill" from the last prefill, so every
        # host-tier event decode causes is attributed to prefill, and any
        # trace captured under V4 contains PREFILL EVENTS ONLY - which would
        # silently feed the Part-3 oracle a decode-free trace. Their absence
        # was invisible in the numbers (the totals still add up) and was found
        # only by an adversarial reader diffing tier_stats between arms.
        rt.host.access_role = "decode"
        if rt.access_trace is not None:
            rt.access_trace.append((1, layer, tuple(ids)))
        slot_of = rt.slot_of
        misses = []
        for e in ids:
            k = (layer, e)
            c.ensure_calls += 1
            if k in slot_of:
                slot_of.move_to_end(k)
                c.hits += 1
            else:
                c.misses += 1
                misses.append(e)
        admit_override = getattr(rt, '_admit_v4_override', None)
        if misses and admit_override is not None:
            admit_override(layer, misses)
        elif misses:
            # Q80-21: parallel cold-row prefill of the host tier (flag-gated;
            # only rows whose demand is ALREADY known — this layer's misses).
            # Decode-side engagement measured as a regression on page-cache-
            # warm rows (batch overhead > serial mmap) => off by default;
            # the PREFILL hook is where the pipeline pays.
            cold = getattr(rt, "cold_io", None)
            if (cold is not None and getattr(rt, "cold_io_decode", False)
                    and getattr(rt.host, "_prepacked_active", False)):
                cold.prefill_host(rt.host, layer, misses)
            need = len(misses)
            free = rt.free
            while len(slot_of) + need > rt.capacity and slot_of:
                _, s = slot_of.popitem(last=False)
                free.append(s)
                c.evictions += 1
            cs = rt._copy_stream
            rows = rt._rows
            host_get = rt.host.get
            note = rt.host.note_h2d
            done = rt._copy_done
            dsts, srcs = [], []
            used_ring = []
            for e in misses:
                buf = host_get(layer, e)
                if not buf.is_pinned():          # pageable source -> stage pinned
                    i = rt._stage_i
                    rt._stage_i = (i + 1) % rt._n_stage
                    ev = rt._stage_events[i]
                    if not ev.query():           # ring slot still in flight (rare)
                        ev.synchronize()
                    stage = rt._stage_ring[i]
                    stage.copy_(buf)             # host memcpy pageable -> pinned
                    buf = stage
                    used_ring.append(i)
                    rt.c_staged += 1
                s = free.pop()
                slot_of[(layer, e)] = s
                dsts.append(rows[s])
                srcs.append(buf)
                note((layer, e), done)
            cs.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cs):
                torch._foreach_copy_(dsts, srcs, non_blocking=True)  # ONE batch
                done.record()
                for i in used_ring:              # each ring slot freed only after
                    rt._stage_events[i].record()  # this batch's copies complete
            c.h2d_ops += need
            c.h2d_bytes += need * rt.L.slot_bytes
            torch.cuda.current_stream().wait_event(done)
        order = sorted(range(rt.L.top_k), key=lambda j: ids[j])
        sl = [slot_of[(layer, ids[j])] for j in order]
        g = rt._graph_v4
        if g is None:
            g = _ExpertGraphV4(rt)
            try:
                g.capture()
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                rt.graph_capture_failures += 1
                rt.last_capture_error = f"{type(exc).__name__}: {exc}"
                if rt.require_graph:
                    raise GraphFallbackError(
                        "V4 expert-graph capture FAILED and require_graph is "
                        f"set: {rt.last_capture_error}") from exc
                raise
            rt._graph_v4 = g
        gl = ([layer * rt.L.n_experts + ids[j] for j in order]
              if rt.bias_gu is not None else None)
        out = g.run_v4(x, sl, order, top_k_weights, gl)
        rt.graph_replays += 1
        return out

    rt.forward_decode = forward_decode_v4
