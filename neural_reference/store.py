"""Lazy exact-dtype MoE weights from single-file or indexed safetensors checkpoints."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .adapters import _expert_key_kind


_SUPPORTED_WEIGHT_DTYPES = {"torch.bfloat16", "torch.float16", "torch.float32"}


class ExpertStore:
    """Read individual experts from mmap-backed safetensors and bound retained views.

    `fetch_expert` returns CPU `(gate_up, down)` tensors. Split gate/up source
    matrices are concatenated losslessly in their source dtype. Cache accounting
    charges the complete logical tensor size even when a slice happens to share
    mmap storage; it is a conservative bound, not an OS-resident-memory claim.
    """

    def __init__(self, model_dir: str | Path, *, host_cache_bytes: int = 0,
                 packed_projection_cache: bool = True):
        if not isinstance(host_cache_bytes, int) or host_cache_bytes < 0:
            raise ValueError("host_cache_bytes must be a nonnegative integer")
        if not isinstance(packed_projection_cache, bool):
            raise TypeError("packed_projection_cache must be a boolean")
        self.model_dir = Path(model_dir).resolve()
        index_path = self.model_dir / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = {}
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = {str(k): str(v) for k, v in index.get("weight_map", {}).items()}
            if not self.weight_map:
                raise ValueError(f"No weight_map in {index_path}")
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if len(files) != 1:
                if not files:
                    raise FileNotFoundError(f"No safetensors checkpoint found under {self.model_dir}")
                raise ValueError("Multiple safetensors shards require model.safetensors.index.json")
            self.weight_map = {}
            self._single_file = files[0].name
        self._single_file = getattr(self, "_single_file", None)
        self._handles: dict[str, Any] = {}
        self.packed_projection_cache = packed_projection_cache
        # At most one tensor wrapper per packed projection. These are CPU
        # safetensors mmap views, not materialized expert-weight copies.
        self._projection_views: dict[str, Any] = {}
        self._packed: dict[tuple[int, str], str] = {}
        self._split: dict[tuple[int, int, str], str] = {}
        self._keys_by_shard: dict[str, list[str]] = {}
        self._host_cache_bytes = host_cache_bytes
        self._cache: OrderedDict[tuple[int, int], tuple[Any, Any, int]] = OrderedDict()
        self._cache_bytes = 0
        self.bytes_read = 0
        self.expert_reads = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        if self.weight_map:
            for key, shard in self.weight_map.items():
                self._add_key(key, shard)
        else:
            self._open(self._single_file)
            for key in self._handles[self._single_file].keys():
                self.weight_map[key] = self._single_file
                self._add_key(key, self._single_file)
        if not self._packed and not self._split:
            raise ValueError("No recognized supported MoE expert tensors in checkpoint")

    def validate_layout(self, *, hidden_size: int, intermediate_size: int, expert_count: int) -> None:
        """Validate all supported expert headers before any model forward call."""
        from collections import defaultdict

        all_dtypes: set[str] = set()
        layers = {layer for layer, _ in self._packed} | {layer for layer, _, _ in self._split}
        for layer in sorted(layers):
            has_packed = (layer, "gate_up_proj") in self._packed or (layer, "down_proj") in self._packed
            split_experts = {expert for li, expert, _ in self._split if li == layer}
            if has_packed and split_experts:
                raise ValueError(f"Layer {layer} mixes packed and split expert storage")
            if has_packed:
                gu_key = self._packed.get((layer, "gate_up_proj"))
                dn_key = self._packed.get((layer, "down_proj"))
                if gu_key is None or dn_key is None:
                    raise ValueError(f"Layer {layer} has incomplete packed expert projections")
                gu_meta = self._metadata(gu_key)
                dn_meta = self._metadata(dn_key)
                if gu_meta[0] not in _SUPPORTED_WEIGHT_DTYPES or dn_meta[0] != gu_meta[0]:
                    raise TypeError(f"Unsupported/inconsistent expert dtypes in layer {layer}: {gu_meta[0]}, {dn_meta[0]}")
                all_dtypes.add(gu_meta[0])
                if gu_meta[1] != (expert_count, 2 * intermediate_size, hidden_size):
                    raise ValueError(f"Bad packed gate_up shape for layer {layer}: {gu_meta[1]}")
                if dn_meta[1] != (expert_count, hidden_size, intermediate_size):
                    raise ValueError(f"Bad packed down shape for layer {layer}: {dn_meta[1]}")
                continue
            if split_experts != set(range(expert_count)):
                raise ValueError(
                    f"Layer {layer} has expert IDs {sorted(split_experts)}, expected 0..{expert_count - 1}"
                )
            per_expert: dict[int, dict[str, str]] = defaultdict(dict)
            for (li, expert, projection), key in self._split.items():
                if li == layer:
                    per_expert[expert][projection] = key
            expected_shapes = {
                "gate_proj": (intermediate_size, hidden_size),
                "up_proj": (intermediate_size, hidden_size),
                "down_proj": (hidden_size, intermediate_size),
            }
            for expert in range(expert_count):
                entries = per_expert.get(expert, {})
                if set(entries) != set(expected_shapes):
                    raise ValueError(f"Incomplete projections in layer {layer} expert {expert}")
                dtypes = set()
                for projection, shape in expected_shapes.items():
                    dtype, actual_shape = self._metadata(entries[projection])
                    dtypes.add(dtype)
                    if actual_shape != shape:
                        raise ValueError(
                            f"Bad {projection} shape for layer {layer} expert {expert}: {actual_shape}; expected {shape}"
                        )
                if len(dtypes) != 1 or next(iter(dtypes)) not in _SUPPORTED_WEIGHT_DTYPES:
                    raise TypeError(f"Unsupported/inconsistent source dtypes for layer {layer} expert {expert}: {sorted(dtypes)}")
                all_dtypes.update(dtypes)
        if len(all_dtypes) != 1:
            raise TypeError(f"Mixed expert dtypes are unsupported for a single activation path: {sorted(all_dtypes)}")
        return next(iter(all_dtypes))

    def _metadata(self, key: str) -> tuple[str, tuple[int, ...]]:
        shard = self.weight_map[key]
        view = self._open(shard).get_slice(key)
        dtype = str(view.get_dtype()).upper()
        dtype_name = {
            "BF16": "torch.bfloat16", "F16": "torch.float16", "F32": "torch.float32",
        }.get(dtype, dtype)
        return dtype_name, tuple(int(x) for x in view.get_shape())

    def _add_key(self, key: str, shard: str) -> None:
        item = _expert_key_kind(key)
        if item is None:
            return
        layer, kind, projection, expert = item
        if kind == "packed":
            if (layer, projection) in self._packed:
                raise ValueError(f"Duplicate packed expert alias for layer {layer} {projection}: {key}")
            self._packed[(layer, projection)] = key
        else:
            if (layer, expert, projection) in self._split:
                raise ValueError(f"Duplicate split expert alias for layer {layer} expert {expert} {projection}: {key}")
            self._split[(layer, expert, projection)] = key
        self._keys_by_shard.setdefault(shard, []).append(key)

    def _safe_shard_path(self, shard: str) -> Path:
        target = (self.model_dir / shard).resolve()
        try:
            target.relative_to(self.model_dir)
        except ValueError as exc:
            raise ValueError(f"Safetensors shard path escapes checkpoint directory: {shard}") from exc
        if not target.is_file():
            raise FileNotFoundError(target)
        return target

    def _open(self, shard: str):
        if shard not in self._handles:
            from safetensors import safe_open

            self._handles[shard] = safe_open(
                str(self._safe_shard_path(shard)), framework="pt", device="cpu"
            )
        return self._handles[shard]

    def _tensor(self, key: str):
        shard = self.weight_map[key]
        tensor = self._open(shard).get_slice(key)
        # Header/slice access; packed expert lookup can separately retain an
        # mmap-backed full-projection tensor wrapper without copying weights.
        return tensor

    def fetch_parameter(self, key: str):
        """Materialize one named non-expert parameter on CPU, preserving dtype."""
        if key not in self.weight_map:
            raise KeyError(key)
        tensor = self._open(self.weight_map[key]).get_tensor(key)
        if str(tensor.dtype) not in _SUPPORTED_WEIGHT_DTYPES:
            raise TypeError(f"Unsupported source dtype {tensor.dtype} for {key}")
        return tensor

    def fetch_expert(self, layer: int, expert: int):
        import torch

        cache_key = (int(layer), int(expert))
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(cache_key)
            return cached[0], cached[1]
        self.cache_misses += 1
        li, ei = cache_key
        packed_gate = self._packed.get((li, "gate_up_proj"))
        packed_down = self._packed.get((li, "down_proj"))
        if packed_gate is not None and packed_down is not None:
            gu = self._packed_expert(packed_gate, ei)
            dn = self._packed_expert(packed_down, ei)
        else:
            gate_key = self._split.get((li, ei, "gate_proj"))
            up_key = self._split.get((li, ei, "up_proj"))
            down_key = self._split.get((li, ei, "down_proj"))
            if None in (gate_key, up_key, down_key):
                raise KeyError(f"No complete expert {ei} in sparse layer {li}")
            gate = self._read_slice(gate_key)
            up = self._read_slice(up_key)
            if gate.dtype != up.dtype or tuple(gate.shape) != tuple(up.shape):
                raise ValueError(f"Incompatible gate/up source tensors for layer {li} expert {ei}")
            # HF fused layout is [gate rows, then up rows], same as Qwen/ Mixtral.
            gu = torch.cat((gate, up), dim=0)
            dn = self._read_slice(down_key)
        if str(gu.dtype) not in _SUPPORTED_WEIGHT_DTYPES or dn.dtype != gu.dtype:
            raise TypeError(f"Unsupported source dtype or inconsistent expert dtypes for layer {li} expert {ei}: {gu.dtype}, {dn.dtype}")
        if gu.ndim != 2 or dn.ndim != 2 or gu.shape[0] % 2:
            raise ValueError(f"Invalid fused expert shapes for layer {li} expert {ei}: {tuple(gu.shape)}, {tuple(dn.shape)}")
        if tuple(dn.shape) != (gu.shape[1], gu.shape[0] // 2):
            raise ValueError(f"Incompatible expert projections for layer {li} expert {ei}: {tuple(gu.shape)}, {tuple(dn.shape)}")
        # Native pointer ABI requires contiguous matrices; torch paths also
        # benefit from the canonical layout. contiguous() may return a view.
        gu, dn = gu.contiguous(), dn.contiguous()
        logical_bytes = (gu.numel() + dn.numel()) * gu.element_size()
        self.bytes_read += logical_bytes
        self.expert_reads += 1
        if self._host_cache_bytes and logical_bytes <= self._host_cache_bytes:
            while self._cache and self._cache_bytes + logical_bytes > self._host_cache_bytes:
                _, (_, _, old_bytes) = self._cache.popitem(last=False)
                self._cache_bytes -= old_bytes
                self.cache_evictions += 1
            self._cache[cache_key] = (gu, dn, logical_bytes)
            self._cache_bytes += logical_bytes
        return gu, dn

    def _packed_expert(self, key: str, expert: int):
        if not self.packed_projection_cache:
            return self._open(self.weight_map[key]).get_slice(key)[expert]
        projection = self._projection_views.get(key)
        if projection is None:
            projection = self._open(self.weight_map[key]).get_tensor(key)
            if not projection.is_contiguous():
                raise ValueError(f"Packed source projection is not contiguous: {key}")
            self._projection_views[key] = projection
        return projection[expert]

    def _read_slice(self, key: str):
        tensor = self._open(self.weight_map[key]).get_slice(key)[...]
        if str(tensor.dtype) not in _SUPPORTED_WEIGHT_DTYPES:
            raise TypeError(f"Unsupported source dtype {tensor.dtype} for {key}")
        return tensor

    def report(self) -> dict[str, Any]:
        # /health can run while inference adds a previously unseen projection.
        # Snapshot the references before iterating or calling tensor methods.
        projection_views = tuple(self._projection_views.values())
        return {
            "source": str(self.model_dir),
            "expert_reads": self.expert_reads,
            "projection_bytes_sliced": self.bytes_read,
            "host_expert_cache_bytes": self._cache_bytes,
            "host_expert_cache_limit_bytes": self._host_cache_bytes,
            "host_expert_cache_entries": len(self._cache),
            "host_expert_cache_hits": self.cache_hits,
            "host_expert_cache_misses": self.cache_misses,
            "host_expert_cache_evictions": self.cache_evictions,
            "packed_projection_cache": self.packed_projection_cache,
            "packed_projection_view_count": len(projection_views),
            "packed_projection_view_limit": len(self._packed),
            "mapped_projection_logical_bytes": sum(t.numel() * t.element_size() for t in projection_views),
            "projection_view_accounting": "one mmap tensor wrapper per packed projection; logical mapped bytes are not copied, pinned, or a resident-RAM measurement",
            "cache_accounting": "logical tensor bytes; conservative bound, not OS-resident pages",
        }

    def close(self) -> None:
        self._cache.clear()
        self._cache_bytes = 0
        self._projection_views.clear()
        self._handles.clear()

    def __enter__(self) -> "ExpertStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
