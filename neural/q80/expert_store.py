"""Exact BF16 Qwen3-Next routed-expert backing store (SSD mmap → host).

Pageable tensors only:
  - model.layers.{L}.mlp.experts.gate_up_proj[E]  shape (1024, 2048)
  - model.layers.{L}.mlp.experts.down_proj[E]     shape (2048, 512)

Not pageable (conventional residency):
  router (gate), shared_expert*, embeddings, attention / Gated DeltaNet,
  norms, lm_head, non-expert core.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neural.q80.intake import Qwen3NextTopology, calculate_expert_anatomy

GATE_UP_SHAPE = (1024, 2048)  # (2 * intermediate, hidden)
DOWN_SHAPE = (2048, 512)  # (hidden, intermediate)
PACKED_GATE_UP = (512, 1024, 2048)
PACKED_DOWN = (512, 2048, 512)


def expert_bf16_nbytes() -> int:
    anatomy = calculate_expert_anatomy(Qwen3NextTopology())
    return int(anatomy["bf16"]["bytes_per_routed_expert"])


def _group_key(layer: int, kind: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{kind}"


def _pack_path(resume_dir: Path, layer: int, kind: str) -> Path:
    return resume_dir / f"{_group_key(layer, kind)}.bf16.bin"


@dataclass
class LayerMmaps:
    gate_up: Any  # torch.Tensor view (512, 1024, 2048) bf16
    down: Any  # torch.Tensor view (512, 2048, 512) bf16
    gate_up_path: Path
    down_path: Path


@dataclass
class Qwen3HostExpertStore:
    """Exact BF16 experts backed by FIX6 resume packs (mmap)."""

    resume_dir: Path
    n_layers: int = 48
    n_experts: int = 512
    expert_bytes: int = field(default_factory=expert_bf16_nbytes)
    _layers: dict[int, LayerMmaps] = field(default_factory=dict)
    ssd_read_bytes: int = 0

    def open(self) -> dict[str, Any]:
        import torch

        missing: list[str] = []
        for li in range(self.n_layers):
            gp = _pack_path(self.resume_dir, li, "gate_up_proj")
            dp = _pack_path(self.resume_dir, li, "down_proj")
            if not gp.is_file() or not dp.is_file():
                missing.append(f"L{li}")
                continue
            g_nbytes = self.n_experts * GATE_UP_SHAPE[0] * GATE_UP_SHAPE[1] * 2
            d_nbytes = self.n_experts * DOWN_SHAPE[0] * DOWN_SHAPE[1] * 2
            if gp.stat().st_size != g_nbytes or dp.stat().st_size != d_nbytes:
                missing.append(f"L{li}_size")
                continue
            g_storage = torch.UntypedStorage.from_file(
                str(gp), shared=True, nbytes=g_nbytes
            )
            d_storage = torch.UntypedStorage.from_file(
                str(dp), shared=True, nbytes=d_nbytes
            )
            gate = torch.tensor([], dtype=torch.bfloat16)
            gate.set_(g_storage)
            gate = gate.view(*PACKED_GATE_UP)
            down = torch.tensor([], dtype=torch.bfloat16)
            down.set_(d_storage)
            down = down.view(*PACKED_DOWN)
            self._layers[li] = LayerMmaps(
                gate_up=gate, down=down, gate_up_path=gp, down_path=dp
            )
        return {
            "status": "ok" if not missing and len(self._layers) == self.n_layers else "fail",
            "n_layers_mapped": len(self._layers),
            "missing_or_bad": missing[:20],
            "resume_dir": str(self.resume_dir),
            "expert_bytes": self.expert_bytes,
            "evidence_class": "MEASURED",
        }

    def fetch_cpu_clones(self, layer: int, expert: int) -> tuple[Any, Any]:
        """SSD/page-cache → private CPU tensors (exact BF16)."""
        import torch

        pack = self._layers[layer]
        # Contiguous clone forces a host materialization of the mmap pages.
        gate = pack.gate_up[expert].contiguous().clone()
        down = pack.down[expert].contiguous().clone()
        nb = int(gate.numel() * gate.element_size() + down.numel() * down.element_size())
        self.ssd_read_bytes += nb
        assert gate.shape == GATE_UP_SHAPE
        assert down.shape == DOWN_SHAPE
        assert gate.dtype == torch.bfloat16 and down.dtype == torch.bfloat16
        return gate, down

    def verify_against_safetensors(
        self,
        model_dir: Path,
        *,
        layer: int = 0,
        expert: int = 0,
    ) -> dict[str, Any]:
        """Equality check vs checkpoint tensors (MEASURED)."""
        import torch
        from safetensors import safe_open

        gate_cpu, down_cpu = self.fetch_cpu_clones(layer, expert)
        # On-disk: separate gate_proj / up_proj / down_proj
        g_key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
        u_key = f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
        d_key = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
        shard = None
        index = model_dir / "model.safetensors.index.json"
        if index.is_file():
            import json

            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            shard = model_dir / weight_map[g_key]
        else:
            # single file fallback
            cands = list(model_dir.glob("*.safetensors"))
            shard = cands[0] if cands else None
        if shard is None or not shard.is_file():
            return {"ok": False, "error": "safetensors_missing", "evidence_class": "MEASURED"}

        with safe_open(str(shard), framework="pt", device="cpu") as f:
            # May need other shards for up/down
            def _load(key: str) -> Any:
                nonlocal f, shard
                try:
                    return f.get_tensor(key)
                except Exception:
                    import json

                    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                    other = model_dir / weight_map[key]
                    with safe_open(str(other), framework="pt", device="cpu") as f2:
                        return f2.get_tensor(key)

            gate_proj = _load(g_key)
            up_proj = _load(u_key)
            down_proj = _load(d_key)

        expected_gate_up = torch.cat([gate_proj, up_proj], dim=0)
        gate_ok = bool(torch.equal(gate_cpu, expected_gate_up.to(torch.bfloat16)))
        down_ok = bool(torch.equal(down_cpu, down_proj.to(torch.bfloat16)))
        return {
            "ok": gate_ok and down_ok,
            "layer": layer,
            "expert": expert,
            "gate_up_equal": gate_ok,
            "down_equal": down_ok,
            "gate_up_shape": list(gate_cpu.shape),
            "down_shape": list(down_cpu.shape),
            "shard": str(shard),
            "evidence_class": "MEASURED",
        }


@dataclass
class HostRamExpertCache:
    """Optional host RAM cache of exact BF16 expert clones (above SSD mmap)."""

    budget_bytes: int
    expert_bytes: int
    _resident: dict[tuple[int, int], tuple[Any, Any]] = field(default_factory=dict)
    _order: list[tuple[int, int]] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def capacity(self) -> int:
        return max(1, self.budget_bytes // max(self.expert_bytes, 1))

    def get(self, layer: int, expert: int) -> tuple[Any, Any] | None:
        key = (layer, expert)
        if key in self._resident:
            self.hits += 1
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return self._resident[key]
        self.misses += 1
        return None

    def put(self, layer: int, expert: int, gate: Any, down: Any) -> None:
        key = (layer, expert)
        while len(self._resident) >= self.capacity and self._order:
            vic = self._order.pop(0)
            self._resident.pop(vic, None)
            self.evictions += 1
        self._resident[key] = (gate, down)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "capacity_experts": self.capacity,
            "resident": len(self._resident),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "evidence_class": "MEASURED",
        }


def paging_boundary_doc() -> dict[str, Any]:
    return {
        "pageable_routed_experts_only": [
            "mlp.experts.gate_up_proj[expert]  # merged gate+up, BF16",
            "mlp.experts.down_proj[expert]",
        ],
        "permanently_resident_initially": [
            "mlp.gate (Qwen3NextTopKRouter)",
            "mlp.shared_expert*",
            "mlp.shared_expert_gate",
            "embeddings / lm_head",
            "self_attn / linear_attn (Gated DeltaNet state)",
            "layer norms / non-expert core",
        ],
        "bytes_per_routed_expert_bf16": expert_bf16_nbytes(),
        "evidence_class": "CALCULATED_topology__MEASURED_pack_layout",
    }
