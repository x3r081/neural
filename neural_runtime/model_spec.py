"""Metadata-only model identification and backend capability selection.

This module deliberately uses only the Python standard library.  It reads
configuration, store descriptors, file sizes, and Safetensors headers; it
never imports torch/Transformers or touches tensor payloads.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ModelSpecError(ValueError):
    """Metadata is incomplete or incompatible with the requested runtime."""


GPTOSS_MODEL_ID = "gpt-oss-120b-b5c939de"
GPTOSS_EXPECTED_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
_GPTOSS_REPRS = {"mxfp4_g32", "mxfp4_g32_ps4"}
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # These families are discoverable and can use a future/reference HF adapter.
    # They are not advertised as compatible with the GPT-OSS optimized kernels.
    "mixtral": {"backend": "hf-reference", "capabilities": ("hf_model_metadata", "reference_expert_hook")},
    "qwen3_moe": {"backend": "hf-reference", "capabilities": ("hf_model_metadata", "reference_expert_hook")},
    "qwen3_5_moe": {"backend": "hf-reference", "capabilities": ("hf_model_metadata", "reference_expert_hook")},
    "qwen3_5_moe_text": {"backend": "hf-reference", "capabilities": ("hf_model_metadata", "reference_expert_hook")},
}


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    backend: str
    mode: str
    model_id: str | None
    geometry: dict[str, Any]
    source_format: dict[str, Any]
    paths: dict[str, str | None]
    context_limits: dict[str, int | None]
    capabilities: tuple[str, ...]
    identity: dict[str, Any]
    memory: dict[str, Any]
    store: dict[str, Any] | None = None

    @property
    def core_bytes(self) -> int | None:
        """GPU-resident core and bias bytes (GPT-OSS embeddings stay on CPU)."""
        return self.memory.get("gpu_core_bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "backend": self.backend,
            "mode": self.mode,
            "model_id": self.model_id,
            "geometry": dict(self.geometry),
            "source_format": dict(self.source_format),
            "paths": dict(self.paths),
            "context_limits": dict(self.context_limits),
            "capabilities": list(self.capabilities),
            "identity": dict(self.identity),
            "memory": dict(self.memory),
            "core_bytes": self.core_bytes,
            "store": None if self.store is None else dict(self.store),
        }


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ModelSpecError(f"cannot read {label} {path}: {exc}") from exc
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSpecError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ModelSpecError(f"{label} must contain a JSON object: {path}")
    return obj, hashlib.sha256(raw).hexdigest()


def _safe_child(root: Path, name: str, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ModelSpecError(f"invalid {label} path {name!r}")
    root_real = root.resolve()
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root_real)
    except ValueError as exc:
        raise ModelSpecError(f"{label} path escapes its directory: {name!r}") from exc
    return candidate


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    experts = config.get("num_local_experts", config.get("num_experts"))
    top_k = config.get("num_experts_per_tok", config.get("experts_per_token"))
    intermediate = config.get("moe_intermediate_size", config.get("intermediate_size"))
    attention = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads", attention)
    head_dim = config.get("head_dim")
    if head_dim is None and attention and config.get("hidden_size"):
        head_dim = int(config["hidden_size"]) // int(attention)
    window = config.get("sliding_window")
    layer_types = config.get("layer_types")
    return {
        "layers": config.get("num_hidden_layers"),
        "experts": experts,
        "top_k": top_k,
        "hidden": config.get("hidden_size"),
        "intermediate": intermediate,
        "num_heads": attention,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "layer_types": list(layer_types) if isinstance(layer_types, list) else layer_types,
        "window": window,
    }


def _context_limits(config: Mapping[str, Any]) -> dict[str, int | None]:
    def integer(*names: str) -> int | None:
        for name in names:
            value = config.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    rope = config.get("rope_scaling")
    rope_original = (rope.get("original_max_position_embeddings")
                     if isinstance(rope, dict) else None)
    return {
        "max_position_embeddings": integer("max_position_embeddings"),
        "original_max_position_embeddings": (rope_original if isinstance(rope_original, int)
                                             else integer("original_max_position_embeddings")),
        "initial_context_length": integer("initial_context_length"),
        "sliding_window": integer("sliding_window"),
    }


def _safetensors_header(path: Path, *, max_header_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Read a Safetensors JSON header only; no tensor payload pages are read."""
    try:
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ModelSpecError(f"short Safetensors header: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if length == 0 or length > max_header_bytes:
                raise ModelSpecError(f"invalid/oversized Safetensors header ({length} bytes): {path}")
            raw = f.read(length)
    except OSError as exc:
        raise ModelSpecError(f"cannot read Safetensors header {path}: {exc}") from exc
    if len(raw) != length:
        raise ModelSpecError(f"truncated Safetensors header: {path}")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSpecError(f"invalid Safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ModelSpecError(f"Safetensors header is not an object: {path}")
    data_end = max((int(v["data_offsets"][1]) for k, v in header.items()
                    if k != "__metadata__" and isinstance(v, dict)
                    and isinstance(v.get("data_offsets"), list)
                    and len(v["data_offsets"]) == 2), default=0)
    if _file_size(path) < 8 + length + data_end:
        raise ModelSpecError(f"Safetensors payload is shorter than its header declares: {path}")
    return header


def _checkpoint_memory(model_dir: Path) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Calculate source tensor bytes from Safetensors offsets without mmap/load."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return ({"source_total_bytes": None, "source_expert_bytes": None,
                 "source_expert_bias_bytes": None, "source_core_bytes": None,
                 "gpu_expert_bias_bytes": None, "gpu_core_bytes": None,
                 "core_dtypes": [], "expert_weight_dtypes": [],
                 "expert_bias_dtypes": [], "source_dtypes": []},
                {"safetensors_index": None})
    index, _ = _read_json(index_path, "Safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelSpecError(f"Safetensors index has no weight_map: {index_path}")
    shards = sorted(set(weight_map.values()))
    total = expert = expert_bias = core = gpu_expert_bias = gpu_core = 0
    core_dtypes: set[str] = set()
    expert_dtypes: set[str] = set()
    expert_bias_dtypes: set[str] = set()
    for shard in shards:
        shard_path = _safe_child(model_dir, shard, "Safetensors shard")
        header = _safetensors_header(shard_path)
        # The shard index and Safetensors header must agree; otherwise byte
        # totals could omit or double-count tensors.
        indexed_names = {n for n, file_name in weight_map.items() if file_name == shard}
        header_names = {n for n in header if n != "__metadata__"}
        if indexed_names != header_names:
            raise ModelSpecError(f"Safetensors index/header names disagree for {shard}")
        for name in header_names:
            entry = header[name]
            offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ModelSpecError(f"invalid data offsets for {name!r} in {shard}")
            size = int(offsets[1]) - int(offsets[0])
            if size < 0:
                raise ModelSpecError(f"invalid data offsets for {name!r} in {shard}")
            total += size
            tensor_dtype = entry.get("dtype")
            if not isinstance(tensor_dtype, str) or not tensor_dtype:
                raise ModelSpecError(f"missing Safetensors dtype for {name!r} in {shard}")
            if ".mlp.experts." in name and "bias" in name:
                expert_bias += size
                gpu_expert_bias += size
                expert_bias_dtypes.add(tensor_dtype)
            elif ".mlp.experts." in name:
                expert += size
                expert_dtypes.add(tensor_dtype)
            else:
                core += size
                core_dtypes.add(tensor_dtype)
                # GPT-OSS keeps token embeddings on CPU in the production path.
                if not name.endswith("embed_tokens.weight"):
                    gpu_core += size
    # GPT-OSS expert biases are kept in the GPU runtime separately from slots.
    gpu_core += gpu_expert_bias
    source_dtypes = core_dtypes | expert_dtypes | expert_bias_dtypes
    return ({"source_total_bytes": total, "source_expert_bytes": expert,
             "source_expert_bias_bytes": expert_bias,
             "source_core_bytes": core, "gpu_expert_bias_bytes": gpu_expert_bias,
             "gpu_core_bytes": gpu_core,
             "core_dtypes": sorted(core_dtypes),
             "expert_weight_dtypes": sorted(expert_dtypes),
             "expert_bias_dtypes": sorted(expert_bias_dtypes),
             "source_dtypes": sorted(source_dtypes)},
            {"safetensors_index": str(index_path)})


def _verify_download_revision(model_dir: Path, weight_map: Mapping[str, Any]) -> dict[str, str]:
    """Require consistent HF snapshot metadata; this does not hash payload bytes."""
    names = {"config.json", "model.safetensors.index.json"}
    for shard in weight_map.values():
        if not isinstance(shard, str) or not shard:
            raise ModelSpecError("Safetensors weight_map contains an invalid shard name")
        names.add(shard)
    metadata_root = _safe_child(model_dir, ".cache/huggingface/download", "Hugging Face metadata directory")
    revisions: dict[str, str] = {}
    for name in sorted(names):
        _safe_child(model_dir, name, "checkpoint file")
        metadata_path = _safe_child(metadata_root, f"{name}.metadata", "Hugging Face download metadata")
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                revision = f.readline().strip()
        except OSError as exc:
            raise ModelSpecError(
                f"missing Hugging Face download metadata for {name}; source revision cannot be verified") from exc
        if not revision:
            raise ModelSpecError(f"empty Hugging Face revision metadata for {name}")
        if revision != GPTOSS_EXPECTED_REVISION:
            raise ModelSpecError(
                f"GPT-OSS fast profile requires source revision {GPTOSS_EXPECTED_REVISION}, "
                f"but {name} metadata reports {revision}")
        revisions[name] = revision
    return revisions


def _verified_source_dtype(config: Mapping[str, Any], memory: Mapping[str, Any]) -> str | None:
    """Return the unanimous non-expert dtype proven by Safetensors headers."""
    declared = config.get("torch_dtype", config.get("dtype"))
    dtypes = memory.get("core_dtypes", [])
    if isinstance(dtypes, list) and len(dtypes) == 1 and isinstance(dtypes[0], str):
        header_dtype = dtypes[0]
        declared_aliases = {"bfloat16": "BF16", "float16": "F16", "float32": "F32",
                            "float64": "F64", "bool": "BOOL", "int8": "I8",
                            "uint8": "U8", "int16": "I16", "int32": "I32",
                            "int64": "I64"}
        if isinstance(declared, str) and declared:
            normalized = declared_aliases.get(declared.lower(), declared.upper())
            if normalized != header_dtype.upper():
                raise ModelSpecError(
                    f"config declares source dtype {declared!r}, but Safetensors core headers prove {header_dtype!r}")
        return header_dtype
    return None


def _slot_bytes(layout: Mapping[str, Any], weight_repr: str) -> int:
    hidden = int(layout["hidden"])
    gu_n = int(layout["gate_up_n"])
    down_n = int(layout["down_n"])
    down_k = int(layout["down_k"])
    group = int(layout["int4_group"])
    code_bytes = (gu_n * hidden + down_n * down_k) // 2
    if weight_repr == "mxfp4_g32":
        scale_bytes = gu_n * (hidden // group) + down_n * (down_k // group)
    elif weight_repr == "mxfp4_g32_ps4":
        # Each row is one base scale byte followed by packed 4-bit deltas.
        scale_bytes = gu_n * (1 + hidden // group // 2) + down_n * (1 + down_k // group // 2)
    else:
        raise ModelSpecError(f"unsupported GPT-OSS fast-store format: {weight_repr!r}")
    return code_bytes + scale_bytes


def _validate_gptoss(config: Mapping[str, Any], config_sha: str,
                     model_dir: Path, store_dir: Path | None, mode: str) -> ModelSpec:
    if mode == "reference":
        # GPT-OSS currently has no registered generic reference adapter in this
        # launcher contract; require its descriptor even when fast mode is not
        # selected so a quantized checkpoint is never silently treated as BF16.
        raise ModelSpecError("GPT-OSS reference mode is not registered; select its validated fast profile")
    if mode not in ("auto", "fast"):
        raise ModelSpecError(f"unsupported GPT-OSS mode: {mode!r}")
    qcfg = config.get("quantization_config")
    if not isinstance(qcfg, dict) or str(qcfg.get("quant_method", "")).lower() != "mxfp4":
        raise ModelSpecError("GPT-OSS fast path requires config quantization_config.quant_method='mxfp4'")
    if store_dir is None:
        raise ModelSpecError("GPT-OSS requires a validated prepacked expert store descriptor")
    store_root = Path(store_dir)
    metadata_path = store_root / "metadata.json"
    metadata, store_sha = _read_json(metadata_path, "prepacked-store descriptor")
    layout = metadata.get("layout")
    if metadata.get("magic") != "Q80-PREPACKED-SLOTS" or not isinstance(layout, dict):
        raise ModelSpecError("unrecognized prepacked store descriptor")
    model_id = metadata.get("model_id")
    if model_id != GPTOSS_MODEL_ID:
        raise ModelSpecError(f"store model_id must be {GPTOSS_MODEL_ID!r}, got {model_id!r}")
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("model_id") != GPTOSS_MODEL_ID:
        raise ModelSpecError("store descriptor must identify the matching GPT-OSS source model")

    geometry = _geometry(config)
    # This is a capability check for today's measured native path, not a claim
    # that the underlying generic slot runtime supports arbitrary geometries.
    expected = {"layers": 36, "experts": 128, "top_k": 4, "hidden": 2880,
                "intermediate": 2880, "num_heads": 64, "num_kv_heads": 8,
                "head_dim": 64, "window": 128}
    for key, value in expected.items():
        if geometry.get(key) != value:
            raise ModelSpecError(f"GPT-OSS fast profile requires geometry.{key}={value}, got {geometry.get(key)!r}")
    expected_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(36)]
    if geometry.get("layer_types") != expected_types:
        raise ModelSpecError("GPT-OSS fast profile requires alternating sliding/full attention layers")
    if str(config.get("hidden_act", "")).lower() != "silu" or float(config.get("swiglu_limit", 0)) != 7.0:
        raise ModelSpecError("GPT-OSS fast profile requires SiLU with swiglu_limit=7")
    if int(config.get("head_dim", 0)) != 64:
        raise ModelSpecError("GPT-OSS fast profile requires head_dim=64")
    rope = config.get("rope_scaling")
    if (config.get("attention_bias") is not True
            or float(config.get("attention_dropout", -1)) != 0.0
            or not isinstance(rope, dict)
            or rope.get("rope_type", rope.get("type")) != "yarn"
            or float(rope.get("factor", 0)) != 32.0):
        raise ModelSpecError("GPT-OSS fast profile requires its validated biased, zero-dropout YaRN attention config")

    repr_ = layout.get("weight_repr")
    if repr_ not in _GPTOSS_REPRS:
        raise ModelSpecError(f"unsupported GPT-OSS store weight_repr: {repr_!r}")
    # Slot rows are independent of routing fanout; historical store metadata
    # does not include top_k, so the model config remains its authority.
    expected_layout = {"n_layers": 36, "n_experts": 128,
                       "hidden": 2880, "gate_up_n": 5760, "down_n": 2880,
                       "down_k": 2880, "int4_group": 32, "inner_k_tiles": 2}
    for key, value in expected_layout.items():
        if layout.get(key) != value:
            raise ModelSpecError(f"GPT-OSS store layout requires {key}={value}, got {layout.get(key)!r}")
    if layout.get("top_k", 4) != 4:
        raise ModelSpecError(f"GPT-OSS store layout top_k must be 4 when recorded, got {layout.get('top_k')!r}")
    expected_slot = _slot_bytes(layout, repr_)
    if layout.get("slot_bytes") != expected_slot:
        raise ModelSpecError(f"GPT-OSS store slot_bytes must be {expected_slot}, got {layout.get('slot_bytes')!r}")
    layer_hashes = metadata.get("sha256_per_layer")
    expected_hash_keys = {str(i) for i in range(36)}
    if not isinstance(layer_hashes, dict) or set(layer_hashes) != expected_hash_keys:
        raise ModelSpecError("store descriptor sha256_per_layer must map string layer IDs 0..35")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
           for value in layer_hashes.values()):
        raise ModelSpecError("store descriptor layer SHA-256 values must be 64 hexadecimal characters")
    bias_hash = metadata.get("expert_biases_sha256")
    if not isinstance(bias_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", bias_hash) is None:
        raise ModelSpecError("store descriptor expert_biases_sha256 must be 64 hexadecimal characters")
    bias_path = _safe_child(store_root, "expert_biases.pt", "expert bias file")
    if not bias_path.is_file() or _file_size(bias_path) == 0:
        raise ModelSpecError("GPT-OSS store is missing non-empty expert_biases.pt")
    layer_files: list[str] = []
    expected_file_bytes = 128 * expected_slot
    for i in range(36):
        layer_path = _safe_child(store_root, f"layer_{i}.slots", "expert layer file")
        if not layer_path.is_file() or _file_size(layer_path) != expected_file_bytes:
            raise ModelSpecError(f"GPT-OSS store layer_{i}.slots must be {expected_file_bytes} bytes")
        layer_files.append(str(layer_path))

    memory, checkpoint_paths = _checkpoint_memory(model_dir)
    safetensors_index = checkpoint_paths["safetensors_index"]
    if safetensors_index is None:
        raise ModelSpecError("GPT-OSS fast profile requires model.safetensors.index.json for source-byte planning")
    index, _ = _read_json(Path(safetensors_index), "Safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelSpecError("GPT-OSS fast profile requires a non-empty Safetensors weight_map")
    _verify_download_revision(model_dir, weight_map)
    dtype = _verified_source_dtype(config, memory)
    if (dtype != "BF16" or memory.get("expert_weight_dtypes") != ["U8"]
            or memory.get("expert_bias_dtypes") != ["BF16"]):
        raise ModelSpecError("GPT-OSS native kernels require original BF16 core/bias and U8 MXFP4 tensor storage; no dtype conversion is permitted")
    source_format = {"quant_method": "mxfp4", "source_dtype": dtype,
                     "source_dtypes": memory.get("source_dtypes", []),
                     "dtype_verified_from_headers": dtype is not None,
                     "store_format": repr_, "weight_preserving": True}
    store = {"slot_bytes": expected_slot, "format": repr_, "metadata": metadata,
             "metadata_path": str(metadata_path),
             "layer_files": layer_files, "bias_path": str(bias_path),
             "layer_count": 36, "experts_per_layer": 128}
    return ModelSpec(
        model_type="gpt_oss", backend="gptoss-native-mxfp4", mode="fast",
        model_id=model_id, geometry=geometry, source_format=source_format,
        paths={"model_dir": str(model_dir.resolve()), "store_dir": str(store_root.resolve()),
               "config": str((model_dir / "config.json").resolve()), "metadata": str(metadata_path.resolve()),
               **checkpoint_paths},
        context_limits=_context_limits(config),
        capabilities=("native_gptoss_fastpath", "mxfp4_lossless_store_format", "expert_biases",
                      "source_header_memory_estimate", "cpu_embedding"),
        identity={"config_sha256": config_sha, "store_metadata_sha256": store_sha,
                  "source_revision": GPTOSS_EXPECTED_REVISION,
                  "source_revision_evidence": "huggingface-download-metadata",
                  "source_payload_hash_verified": False,
                  "store_payload_hash_verified": False,
                  "store_payload_verification_scope": "metadata_schema_and_file_sizes_only"},
        memory=memory, store=store)


def inspect_model(model_dir: str | Path, store_dir: str | Path | None = None,
                  *, mode: str = "auto") -> ModelSpec:
    """Identify a supported MoE model using local metadata only.

    `auto` selects the validated GPT-OSS fast profile or an explicitly
    reference-only HF profile. `fast` is fail-closed for all other models.
    """
    root = Path(model_dir)
    config_path = root / "config.json"
    config, config_sha = _read_json(config_path, "model config")
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        raise ModelSpecError("model config has no string model_type")
    if mode not in ("auto", "fast", "reference"):
        raise ModelSpecError(f"unknown mode {mode!r}")
    if model_type == "gpt_oss":
        return _validate_gptoss(config, config_sha, root,
                                None if store_dir is None else Path(store_dir), mode)
    registration = _MODEL_REGISTRY.get(model_type)
    if registration is None:
        raise ModelSpecError(f"unsupported model_type {model_type!r}")
    if mode == "fast":
        raise ModelSpecError(f"no optimized fast profile is registered for {model_type!r}")
    geometry = _geometry(config)
    memory, checkpoint_paths = _checkpoint_memory(root)
    quant = config.get("quantization_config")
    source_format = {"quant_method": quant.get("quant_method") if isinstance(quant, dict) else None,
                     "source_dtype": _verified_source_dtype(config, memory),
                     "source_dtypes": memory.get("source_dtypes", []),
                     "dtype_verified_from_headers": _verified_source_dtype(config, memory) is not None,
                     "weight_preserving": True}
    arch = (config.get("architectures") or [model_type])[0]
    return ModelSpec(
        model_type=model_type, backend=registration["backend"], mode="reference",
        model_id=config.get("_name_or_path") or arch,
        geometry=geometry, source_format=source_format,
        paths={"model_dir": str(root.resolve()), "config": str(config_path.resolve()), **checkpoint_paths},
        context_limits=_context_limits(config), capabilities=tuple(registration["capabilities"]),
        identity={"config_sha256": config_sha, "store_metadata_sha256": None},
        memory=memory, store=None)


__all__ = ["ModelSpec", "ModelSpecError", "inspect_model"]
