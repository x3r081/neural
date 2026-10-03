"""Generic expert residency scheduler.

Operates on (LayerID, ExpertID) keys and an ExpertModuleAccess backend.
Architecture-specific MoE forward math lives in adapters — not here.
"""

from __future__ import annotations

import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

PolicyName = Literal["demand_sync", "lru", "lfu"]


@dataclass
class PagingCounters:
    ensure_calls: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    h2d_transfer_s: float = 0.0
    stall_s: float = 0.0
    compute_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ensure_calls": self.ensure_calls,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "evictions": self.evictions,
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "h2d_gib": self.h2d_bytes / 1024**3,
            "d2h_gib": self.d2h_bytes / 1024**3,
            "h2d_transfer_s": self.h2d_transfer_s,
            "stall_s": self.stall_s,
            "expert_compute_s": self.compute_s,
        }


@runtime_checkable
class ExpertModuleAccess(Protocol):
    """Adapter-supplied view of expert modules (no OLMoE/Phi names in scheduler)."""

    n_layers: int
    n_experts: int

    def expert_module(self, layer: int, expert: int) -> Any: ...

    def iter_expert_modules(self) -> Any: ...


# Optional adapter hooks (duck-typed; not required by Protocol):
#   materialize_to_device(layer, expert, device) -> int   # nbytes H2D
#   dematerialize_to_host(layer, expert) -> int           # nbytes D2H / release
#   relocate_all_to_host() -> dict
# Used when experts are not plain nn.Modules movable via .to().


@dataclass
class ExpertResidencyManager:
    """CUDA expert cache with CPU/host backing; policy ∈ {demand_sync, lru, lfu}.

    Optional access hooks materialize_to_device / dematerialize_to_host allow
    packed/mmap expert stores (e.g. Qwen3-Next) without a separate cache engine.
    """

    access: ExpertModuleAccess
    budget_bytes: int
    policy: PolicyName
    device: Any
    expert_bytes: int
    counters: PagingCounters = field(default_factory=PagingCounters)

    def __post_init__(self) -> None:
        self._resident: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._freq: dict[tuple[int, int], int] = defaultdict(int)
        self._capacity = max(1, self.budget_bytes // max(self.expert_bytes, 1))
        self.enabled = True
        self._decode_step = 0
        # When True, begin_token() flushes VRAM residency (Q80-5 "demand" =
        # no temporal carry-over). Default False preserves OLMoE demand_sync.
        self.flush_each_token = False

    @property
    def capacity_experts(self) -> int:
        return self._capacity

    def _nbytes(self, layer: int, expert: int) -> int:
        mod = self.access.expert_module(layer, expert)
        return sum(int(p.numel() * p.element_size()) for p in mod.parameters())

    def _has_materialize_hooks(self) -> bool:
        return hasattr(self.access, "materialize_to_device") and hasattr(
            self.access, "dematerialize_to_host"
        )

    def relocate_all_experts_to_cpu(self) -> dict[str, Any]:
        import torch

        if hasattr(self.access, "relocate_all_to_host"):
            out = dict(self.access.relocate_all_to_host())
            self._resident.clear()
            self._freq.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            out.setdefault("capacity_experts", self._capacity)
            out.setdefault("budget_bytes", self.budget_bytes)
            out.setdefault("evidence", "MEASURED_relocation")
            return out

        n = 0
        bytes_moved = 0
        for layer, expert, mod in self.access.iter_expert_modules():
            for p in mod.parameters():
                if p.device.type != "cpu":
                    bytes_moved += int(p.numel() * p.element_size())
            mod.to("cpu")
            n += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._resident.clear()
        return {
            "n_experts_moved_to_cpu": n,
            "bytes_from_cuda_est": bytes_moved,
            "capacity_experts": self._capacity,
            "budget_bytes": self.budget_bytes,
            "evidence": "MEASURED_relocation",
        }

    def begin_token(self) -> None:
        """Token boundary hook. Optional flush for no-carryover demand experiments."""
        self._decode_step += 1
        if self.flush_each_token and self._resident:
            self.relocate_all_experts_to_cpu()

    def _evict_one(self) -> None:
        import torch

        if not self._resident:
            return
        if self.policy == "lfu":
            victim = min(self._resident.keys(), key=lambda k: (self._freq[k], k))
            self._resident.pop(victim, None)
        else:
            victim, _ = self._resident.popitem(last=False)
        layer, expert = victim
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._has_materialize_hooks():
            nb = int(self.access.dematerialize_to_host(layer, expert))  # type: ignore[attr-defined]
        else:
            mod = self.access.expert_module(layer, expert)
            nb = self._nbytes(layer, expert)
            mod.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self.counters.d2h_bytes += nb
        self.counters.evictions += 1

    def _admit(self, layer: int, expert: int) -> None:
        import torch

        key = (layer, expert)
        if key in self._resident:
            self._resident.move_to_end(key)
            return
        while len(self._resident) >= self._capacity:
            self._evict_one()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if self._has_materialize_hooks():
            nb = int(self.access.materialize_to_device(layer, expert, self.device))  # type: ignore[attr-defined]
        else:
            mod = self.access.expert_module(layer, expert)
            nb = self._nbytes(layer, expert)
            mod.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.counters.h2d_bytes += nb
        self.counters.h2d_transfer_s += dt
        self.counters.stall_s += dt
        self._resident[key] = None

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

    def reset_counters(self) -> None:
        self.counters = PagingCounters()

    # ---- Q80-6 batched-admission API (additive; no per-expert sync) ----
    def _evict_one_async(self) -> None:
        """Evict one victim without a per-eviction global synchronize.

        Same-stream ordering keeps freed VRAM safe to reallocate for the next
        admission; the token/layer boundary provides the bounded sync.
        """
        if not self._resident:
            return
        if self.policy == "lfu":
            victim = min(self._resident.keys(), key=lambda k: (self._freq[k], k))
            self._resident.pop(victim, None)
        else:
            victim, _ = self._resident.popitem(last=False)
        layer, expert = victim
        if hasattr(self.access, "dematerialize_to_host_async"):
            nb = int(self.access.dematerialize_to_host_async(layer, expert))  # type: ignore[attr-defined]
        elif self._has_materialize_hooks():
            nb = int(self.access.dematerialize_to_host(layer, expert))  # type: ignore[attr-defined]
        else:
            mod = self.access.expert_module(layer, expert)
            nb = self._nbytes(layer, expert)
            mod.to("cpu")
        self.counters.d2h_bytes += nb
        self.counters.evictions += 1

    def begin_layer(self, layer: int, hit_ids: list[int]) -> list[int]:
        """Record hits/misses for a layer's routed experts; evict to fit misses.

        Returns the list of miss expert ids the caller must materialize. Does NOT
        materialize (the optimized forward batches the transfers itself).
        """
        misses: list[int] = []
        for e in hit_ids:
            key = (layer, int(e))
            self.counters.ensure_calls += 1
            if key in self._resident:
                self.counters.hits += 1
                self._resident.move_to_end(key)
                self._freq[key] += 1
            else:
                self.counters.misses += 1
                misses.append(int(e))
        need = len(misses)
        if need > self._capacity:
            raise RuntimeError(
                f"layer {layer}: {need} misses exceed cache capacity "
                f"{self._capacity}; raise VRAM budget for this configuration"
            )
        # LRU/LFU eviction of currently-resident experts NOT hit this layer.
        while len(self._resident) + need > self._capacity and self._resident:
            self._evict_one_async()
        return misses

    def mark_admitted(self, layer: int, expert: int) -> None:
        self._resident[(layer, int(expert))] = None
        self._freq[(layer, int(expert))] += 1
