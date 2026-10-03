"""Explicit Transformers architecture adapters and metadata-only checkpoint inspection."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PACKED_RE = re.compile(
    r"(?:^|\.)model\.(?:language_model\.)?layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(gate_up_proj|down_proj)$"
)
_SPLIT_RE = re.compile(
    r"(?:^|\.)model\.(?:language_model\.)?layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)(?:\.weight)?$"
)


@dataclass(frozen=True)
class ModelAdapter:
    model_type: str
    module_path: str
    class_name: str
    expert_owner: str
    has_qwen_gdn: bool = False

    @property
    def modeling_module(self):
        return __import__(self.module_path, fromlist=[self.class_name])

    def text_config(self, config):
        nested = getattr(config, "text_config", None)
        if nested is not None and hasattr(nested, "num_hidden_layers"):
            return nested
        return config

    def model_class(self):
        try:
            return getattr(self.modeling_module, self.class_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Transformers {self.module_path} has no {self.class_name}"
            ) from exc

    def sparse_modules(self, model):
        """Yield (layer index, expert module, sparse block) for actual MoE layers."""
        base = getattr(model, "model", None)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise RuntimeError(f"{self.model_type} model has no model.layers collection")
        result = []
        for index, layer in enumerate(layers):
            block = getattr(layer, self.expert_owner, None)
            if block is None:
                continue
            experts = getattr(block, "experts", None)
            if experts is not None:
                result.append((index, experts, block))
        if not result:
            raise RuntimeError(f"No sparse expert modules found for {self.model_type}")
        return result

    @staticmethod
    def is_expert_parameter(name: str) -> bool:
        return ".experts." in name


_ADAPTERS = {
    "qwen3_5_moe": ModelAdapter(
        "qwen3_5_moe", "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeForCausalLM", "mlp", has_qwen_gdn=True,
    ),
    "qwen3_5_moe_text": ModelAdapter(
        "qwen3_5_moe_text", "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeForCausalLM", "mlp", has_qwen_gdn=True,
    ),
    "qwen3_moe": ModelAdapter(
        "qwen3_moe", "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "Qwen3MoeForCausalLM", "mlp",
    ),
    "mixtral": ModelAdapter(
        "mixtral", "transformers.models.mixtral.modeling_mixtral",
        "MixtralForCausalLM", "mlp",
    ),
}


def adapter_for_type(model_type: str) -> ModelAdapter:
    try:
        return _ADAPTERS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unsupported MoE architecture {model_type!r}; supported model types: {supported}"
        ) from exc


def adapter_for_config(config) -> ModelAdapter:
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_5_moe_text":
        return _ADAPTERS[model_type]
    if model_type not in _ADAPTERS and getattr(getattr(config, "text_config", None), "model_type", None) in _ADAPTERS:
        model_type = config.text_config.model_type
    return adapter_for_type(model_type)


def _read_index(root: Path) -> tuple[dict[str, str], list[str]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty safetensors index: {index_path}")
        shards = sorted(set(str(v) for v in weight_map.values()))
        for shard in shards:
            resolved = (root / shard).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"Safetensors shard path escapes checkpoint directory: {shard}") from exc
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
        return {str(k): str(v) for k, v in weight_map.items()}, shards
    files = sorted(root.glob("*.safetensors"))
    if len(files) != 1:
        if not files:
            raise FileNotFoundError(f"No safetensors checkpoint found under {root}")
        raise ValueError("Multiple safetensors shards require model.safetensors.index.json")
    # Discover the one-file checkpoint's tensor names from headers later, but
    # still refuse a symlink escaping the selected local checkpoint directory.
    resolved = files[0].resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Safetensors file escapes checkpoint directory: {files[0].name}") from exc
    return {}, [files[0].name]


def _expert_key_kind(name: str):
    packed = _PACKED_RE.search(name)
    if packed:
        return int(packed.group(1)), "packed", packed.group(2), None
    split = _SPLIT_RE.search(name)
    if split:
        projection = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}.get(split.group(3), split.group(3))
        return int(split.group(1)), "split", projection, int(split.group(2))
    return None


def inspect_checkpoint(model_dir: str | Path) -> dict[str, Any]:
    """Inspect local config/safetensors headers without constructing a model or reading tensors."""
    from safetensors import safe_open

    root = Path(model_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if raw_config.get("quantization_config"):
        raise ValueError("Quantized checkpoints are not supported; provide source floating-point safetensors")
    raw_text = raw_config.get("text_config") or raw_config
    if raw_text.get("quantization_config"):
        raise ValueError("Quantized text checkpoints are not supported; provide source floating-point safetensors")
    model_type = str(raw_config.get("model_type", raw_text.get("model_type", "")))
    text_model_type = str(raw_text.get("model_type", model_type))
    adapter = adapter_for_type(text_model_type)
    weight_map, shards = _read_index(root)
    entries: list[tuple[str, str, tuple[int, ...], int]] = []
    dtypes: set[str] = set()
    expert_dtypes: set[str] = set()
    if weight_map:
        grouped: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            grouped.setdefault(shard, []).append(key)
        for shard, keys in grouped.items():
            with safe_open(str(root / shard), framework="pt", device="cpu") as f:
                for key in keys:
                    view = f.get_slice(key)
                    dtype = str(view.get_dtype()).lower().removeprefix("torch.")
                    shape = tuple(int(x) for x in view.get_shape())
                    nbytes = 1
                    for dim in shape:
                        nbytes *= dim
                    # Safetensors dtype spellings match torch except BF16/F16.
                    bpe = {"bool": 1, "u8": 1, "i8": 1, "f8_e4m3": 1, "f8_e5m2": 1,
                           "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4,
                           "i64": 8, "f64": 8}.get(dtype)
                    if bpe is None:
                        raise TypeError(f"Unsupported safetensors dtype {dtype!r} for {key}")
                    entries.append((key, dtype, shape, nbytes * bpe))
                    dtypes.add(dtype)
    else:
        shard = shards[0]
        with safe_open(str(root / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                view = f.get_slice(key)
                dtype = str(view.get_dtype()).lower().removeprefix("torch.")
                shape = tuple(int(x) for x in view.get_shape())
                bpe = {"bool": 1, "u8": 1, "i8": 1, "f8_e4m3": 1, "f8_e5m2": 1,
                       "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4,
                       "i64": 8, "f64": 8}.get(dtype)
                if bpe is None:
                    raise TypeError(f"Unsupported safetensors dtype {dtype!r} for {key}")
                entries.append((key, dtype, shape, math_prod(shape) * bpe))
                dtypes.add(dtype)

    core_bytes = expert_bytes = 0
    layer_bytes: dict[int, int] = {}
    individual_expert_bytes: dict[tuple[int, int], int] = {}
    sparse_layers: set[int] = set()
    packed_keys: set[tuple[int, str]] = set()
    canonical_expert_keys: set[tuple[int, str, str, int | None]] = set()
    for key, _dtype, shape, nbytes in entries:
        kind = _expert_key_kind(key)
        if kind is None:
            # Auxiliary prediction modules and vision tensors are not part of
            # the supported causal text model instantiated by these adapters.
            if not key.startswith(("mtp.", "model.visual.")) and ".mtp." not in key:
                core_bytes += nbytes
            continue
        expert_dtypes.add(_dtype)
        layer, storage, projection, expert = kind
        canonical = (layer, storage, projection, expert)
        if canonical in canonical_expert_keys:
            raise ValueError(f"Ambiguous duplicate expert tensor alias in checkpoint: {key}")
        canonical_expert_keys.add(canonical)
        expert_bytes += nbytes
        layer_bytes[layer] = layer_bytes.get(layer, 0) + nbytes
        sparse_layers.add(layer)
        if storage == "packed":
            packed_keys.add((layer, projection))
            expert_total = nbytes // max(1, int(shape[0]))
            for expert_id in range(int(shape[0])):
                individual_expert_bytes[(layer, expert_id)] = individual_expert_bytes.get((layer, expert_id), 0) + expert_total
        else:
            individual_expert_bytes[(layer, int(expert))] = individual_expert_bytes.get((layer, int(expert)), 0) + nbytes
    if not sparse_layers:
        raise ValueError("Checkpoint contains no recognized supported MoE expert tensors")
    if packed_keys and expert_bytes:
        formats = {"fused" if packed_keys else "split"}
        if packed_keys and any(_expert_key_kind(k)[1] == "split" for k, *_ in entries if _expert_key_kind(k)):
            formats.add("split")
        fmt = "+".join(sorted(formats)) + " safetensors"
    else:
        fmt = ("fused safetensors" if packed_keys else "split safetensors")
    text_cfg = dict(raw_text)
    expert_count = text_cfg.get("num_experts", text_cfg.get("num_local_experts"))
    top_k = text_cfg.get("num_experts_per_tok")
    layer_count = text_cfg.get("num_hidden_layers")
    layer_bytes_dict = {str(k): int(v) for k, v in sorted(layer_bytes.items())}
    return {
        "model_type": model_type,
        "text_model_type": text_model_type,
        "hidden_size": text_cfg.get("hidden_size"),
        "num_hidden_layers": layer_count,
        "expert_count": expert_count,
        "top_k": top_k,
        "dtypes": sorted(dtypes),
        "dtype": sorted(dtypes)[0] if len(dtypes) == 1 else "mixed",
        "expert_dtypes": sorted(expert_dtypes),
        "core_bytes": int(core_bytes),
        "expert_bytes": int(expert_bytes),
        "layer_expert_bytes": layer_bytes_dict,
        "sparse_layers": sorted(sparse_layers),
        "largest_expert_bytes": max(individual_expert_bytes.values(), default=0),
        "largest_layer_expert_bytes": max(layer_bytes.values(), default=0),
        "text_config": text_cfg,
        "format": fmt,
        "adapter": adapter.model_type,
        "checkpoint_files": shards,
    }


def math_prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out
