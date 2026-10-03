"""Read one original BF16 MoE expert at a time from a sharded checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BF16ExpertStore:
    """Lazy, read-only access to packed Qwen3.5/3.6 MoE expert tensors.

    The checkpoint stores each projection as a single expert-major tensor. This
    class slices an individual expert from safetensors' mmap-backed view, so it
    never materializes the whole expert pool in host RAM.
    """

    def __init__(self, model_dir: str | Path, *, index_file: str = "model.safetensors.index.json"):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / index_file
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index.get("weight_map", {})
        if not self.weight_map:
            raise ValueError(f"No weight_map in {index_path}")
        self._handles: dict[str, Any] = {}
        self._expert_keys: dict[tuple[int, str], str] = {}
        self._key_prefix: str | None = None
        self.bytes_read = 0
        self.expert_reads = 0

    def _open(self, shard: str):
        if shard not in self._handles:
            from safetensors import safe_open

            self._handles[shard] = safe_open(
                str(self.model_dir / shard), framework="pt", device="cpu"
            )
        return self._handles[shard]

    def _expert_key(self, layer: int, projection: str) -> str:
        if projection not in {"gate_up_proj", "down_proj"}:
            raise ValueError(f"Unknown expert projection: {projection}")
        cache_key = (int(layer), projection)
        key = self._expert_keys.get(cache_key)
        if key is not None:
            return key
        stems = (
            f"model.language_model.layers.{layer}.mlp.experts.{projection}",
            f"model.layers.{layer}.mlp.experts.{projection}",
        )
        for candidate in stems:
            if candidate in self.weight_map:
                self._expert_keys[cache_key] = candidate
                return candidate
        raise KeyError(f"No checkpoint tensor found for layer {layer} {projection}")

    def fetch_projection(self, layer: int, expert: int, projection: str):
        """Return a contiguous CPU tensor for one exact source expert tensor."""
        key = self._expert_key(layer, projection)
        shard = self.weight_map[key]
        tensor = self._open(shard).get_slice(key)[int(expert)].contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        self.bytes_read += nbytes
        return tensor

    def fetch_expert(self, layer: int, expert: int):
        """Return ``(gate_up, down)`` CPU tensors for one BF16 expert."""
        gate_up = self.fetch_projection(layer, expert, "gate_up_proj")
        down = self.fetch_projection(layer, expert, "down_proj")
        if str(gate_up.dtype) != "torch.bfloat16" or str(down.dtype) != "torch.bfloat16":
            raise TypeError(
                f"Expected original BF16 expert weights, got {gate_up.dtype} and {down.dtype}"
            )
        self.expert_reads += 1
        return gate_up, down

    def report(self) -> dict[str, Any]:
        return {
            "source": str(self.model_dir),
            "expert_reads": self.expert_reads,
            "projection_bytes_sliced": self.bytes_read,
            "host_expert_cache_bytes": 0,
            "access": "safetensors mmap slice; transient per-expert CPU tensors",
            "evidence_class": "MEASURED_RUNTIME_COUNTERS",
        }

    def reset_counters(self) -> None:
        self.bytes_read = 0
        self.expert_reads = 0

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self) -> "BF16ExpertStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
