import hashlib
import json
import struct
from pathlib import Path

import pytest

from neural_runtime import model_spec


def test_native_profile_refuses_implicitly_downcasting_a_float32_core(tmp_path, monkeypatch):
    checkpoint, store = tmp_path / "checkpoint", tmp_path / "store"
    config = _gptoss_config()
    config["torch_dtype"] = "float32"
    _write_json(checkpoint / "config.json", config)
    _safetensors_fixture(checkpoint)
    _gptoss_store(store)
    original_memory = model_spec._checkpoint_memory
    original_size = model_spec._file_size

    def changed_core(root):
        memory, paths = original_memory(root)
        memory["core_dtypes"] = ["F32"]
        return memory, paths

    monkeypatch.setattr(model_spec, "_checkpoint_memory", changed_core)
    monkeypatch.setattr(model_spec, "_file_size", lambda p: 128 * 12_839_040
                        if p.name.startswith("layer_") and p.name.endswith(".slots") else original_size(p))
    with pytest.raises(model_spec.ModelSpecError, match="no dtype conversion"):
        model_spec.inspect_model(checkpoint, store)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_download_metadata(model_dir: Path, names):
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (metadata_dir / f"{name}.metadata").write_text(
            model_spec.GPTOSS_EXPECTED_REVISION + "\nfixture\n", encoding="utf-8")


def _safetensors_fixture(model_dir: Path):
    shard = model_dir / "model-00001-of-00001.safetensors"
    entries = {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": {
            "dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "model.layers.0.mlp.experts.gate_up_proj_bias": {
            "dtype": "BF16", "shape": [2], "data_offsets": [4, 8]},
        "model.embed_tokens.weight": {
            "dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 16]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "BF16", "shape": [4, 2], "data_offsets": [16, 32]},
    }
    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(struct.pack("<Q", len(header)) + header + bytes(32))
    _write_json(model_dir / "model.safetensors.index.json", {
        "metadata": {"total_size": 32},
        "weight_map": {name: shard.name for name in entries},
    })
    _write_download_metadata(model_dir, ["config.json", "model.safetensors.index.json", shard.name])


def _gptoss_config():
    return {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 36,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "layer_types": ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(36)],
        "hidden_act": "silu",
        "swiglu_limit": 7.0,
        "max_position_embeddings": 131072,
        "initial_context_length": 4096,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "rope_scaling": {"rope_type": "yarn", "factor": 32.0},
        "quantization_config": {"quant_method": "mxfp4"},
        "torch_dtype": "bfloat16",
    }


def _gptoss_store(store_dir: Path, weight_repr="mxfp4_g32_ps4"):
    layout = {
        "n_layers": 36, "n_experts": 128, "top_k": 4,
        "hidden": 2880, "gate_up_n": 5760, "down_n": 2880,
        "down_k": 2880, "int4_group": 32, "inner_k_tiles": 2,
        "weight_repr": weight_repr,
        "slot_bytes": 12839040 if weight_repr == "mxfp4_g32_ps4" else 13219200,
    }
    descriptor = {
        "magic": "Q80-PREPACKED-SLOTS", "version": 1,
        "model_id": model_spec.GPTOSS_MODEL_ID,
        "source": {"model_id": model_spec.GPTOSS_MODEL_ID},
        "layout": layout,
        "sha256_per_layer": {str(i): hashlib.sha256(str(i).encode()).hexdigest() for i in range(36)},
        "expert_biases_sha256": hashlib.sha256(b"biases").hexdigest(),
    }
    _write_json(store_dir / "metadata.json", descriptor)
    (store_dir / "expert_biases.pt").write_bytes(b"x")
    for i in range(36):
        (store_dir / f"layer_{i}.slots").touch()


def test_layout_slot_sizes_for_both_supported_source_representations():
    layout = {
        "hidden": 2880, "gate_up_n": 5760, "down_n": 2880,
        "down_k": 2880, "int4_group": 32,
    }
    assert model_spec._slot_bytes(layout, "mxfp4_g32") == 13_219_200
    assert model_spec._slot_bytes(layout, "mxfp4_g32_ps4") == 12_839_040


def test_inspect_gptoss_validates_profile_and_reports_metadata_only_memory(tmp_path, monkeypatch):
    model_dir = tmp_path / "checkpoint"
    store_dir = tmp_path / "store"
    _write_json(model_dir / "config.json", _gptoss_config())
    _safetensors_fixture(model_dir)
    _gptoss_store(store_dir)
    real_size = model_spec._file_size

    def size(path):
        if path.name.startswith("layer_") and path.name.endswith(".slots"):
            return 128 * 12_839_040
        return real_size(path)

    monkeypatch.setattr(model_spec, "_file_size", size)
    spec = model_spec.inspect_model(model_dir, store_dir).to_dict()
    assert spec["backend"] == "gptoss-native-mxfp4"
    assert spec["mode"] == "fast"
    assert spec["model_id"] == model_spec.GPTOSS_MODEL_ID
    assert spec["geometry"]["layers"] == 36
    assert spec["geometry"]["experts"] == 128
    assert spec["geometry"]["top_k"] == 4
    assert spec["store"]["format"] == "mxfp4_g32_ps4"
    assert spec["memory"] == {
        "source_total_bytes": 32,
        "source_expert_bytes": 4,
        "source_expert_bias_bytes": 4,
        "source_core_bytes": 24,
        "gpu_expert_bias_bytes": 4,
        "gpu_core_bytes": 20,
        "core_dtypes": ["BF16"],
        "expert_weight_dtypes": ["U8"],
        "expert_bias_dtypes": ["BF16"],
        "source_dtypes": ["BF16", "U8"],
    }
    assert spec["source_format"]["source_dtype"] == "BF16"
    assert spec["source_format"]["dtype_verified_from_headers"] is True
    assert len(spec["store"]["layer_files"]) == 36
    assert len(spec["identity"]["config_sha256"]) == 64
    assert len(spec["identity"]["store_metadata_sha256"]) == 64
    assert spec["identity"]["source_revision"] == model_spec.GPTOSS_EXPECTED_REVISION
    assert spec["identity"]["source_revision_evidence"] == "huggingface-download-metadata"
    assert spec["identity"]["source_payload_hash_verified"] is False
    assert spec["identity"]["store_payload_hash_verified"] is False
    assert spec["identity"]["store_payload_verification_scope"] == "metadata_schema_and_file_sizes_only"
    assert "mxfp4_lossless_store_format" in spec["capabilities"]
    assert "mxfp4_exact_store" not in spec["capabilities"]


@pytest.mark.parametrize("problem", ["missing_layer", "bad_layer_hash", "bad_bias_hash"])
def test_gptoss_store_hash_metadata_must_be_well_formed(tmp_path, monkeypatch, problem):
    model_dir = tmp_path / "checkpoint"
    store_dir = tmp_path / "store"
    _write_json(model_dir / "config.json", _gptoss_config())
    _safetensors_fixture(model_dir)
    _gptoss_store(store_dir)
    metadata_path = store_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if problem == "missing_layer":
        del metadata["sha256_per_layer"]["17"]
    elif problem == "bad_layer_hash":
        metadata["sha256_per_layer"]["17"] = "not-a-sha256"
    else:
        metadata["expert_biases_sha256"] = "0" * 63
    _write_json(metadata_path, metadata)
    real_size = model_spec._file_size

    def size(path):
        if path.name.startswith("layer_") and path.name.endswith(".slots"):
            return 128 * 12_839_040
        return real_size(path)

    monkeypatch.setattr(model_spec, "_file_size", size)
    with pytest.raises(model_spec.ModelSpecError, match="sha256|SHA-256"):
        model_spec.inspect_model(model_dir, store_dir)


@pytest.mark.parametrize("metadata_state", ["unknown", "mixed", "missing"])
def test_gptoss_requires_consistent_snapshot_revision_metadata(tmp_path, monkeypatch, metadata_state):
    model_dir = tmp_path / "checkpoint"
    store_dir = tmp_path / "store"
    _write_json(model_dir / "config.json", _gptoss_config())
    _safetensors_fixture(model_dir)
    _gptoss_store(store_dir)
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    shard_meta = metadata_dir / "model-00001-of-00001.safetensors.metadata"
    if metadata_state == "unknown":
        (metadata_dir / "config.json.metadata").write_text("unknown-revision\n", encoding="utf-8")
    elif metadata_state == "mixed":
        shard_meta.write_text("different-revision\n", encoding="utf-8")
    else:
        shard_meta.unlink()
    real_size = model_spec._file_size

    def size(path):
        if path.name.startswith("layer_") and path.name.endswith(".slots"):
            return 128 * 12_839_040
        return real_size(path)

    monkeypatch.setattr(model_spec, "_file_size", size)
    with pytest.raises(model_spec.ModelSpecError, match="revision|metadata"):
        model_spec.inspect_model(model_dir, store_dir)


def test_non_gptoss_profiles_are_reference_only(tmp_path):
    config = {
        "model_type": "mixtral", "architectures": ["MixtralForCausalLM"],
        "num_hidden_layers": 2, "num_local_experts": 4,
        "num_experts_per_tok": 2, "hidden_size": 16,
        "intermediate_size": 32, "num_attention_heads": 4,
        "num_key_value_heads": 2, "max_position_embeddings": 512,
    }
    _write_json(tmp_path / "config.json", config)
    spec = model_spec.inspect_model(tmp_path).to_dict()
    assert spec["model_type"] == "mixtral"
    assert spec["backend"] == "hf-reference"
    assert spec["mode"] == "reference"
    assert "reference_expert_hook" in spec["capabilities"]
    assert spec["geometry"]["top_k"] == 2
    assert spec["store"] is None


def test_source_dtype_is_derived_from_safetensors_headers_when_config_omits_it(tmp_path):
    config = {
        "model_type": "mixtral", "architectures": ["MixtralForCausalLM"],
        "num_hidden_layers": 1, "num_local_experts": 2,
        "num_experts_per_tok": 1, "hidden_size": 4,
    }
    _write_json(tmp_path / "config.json", config)
    _safetensors_fixture(tmp_path)
    spec = model_spec.inspect_model(tmp_path).to_dict()
    assert spec["source_format"]["source_dtype"] == "BF16"
    assert spec["source_format"]["dtype_verified_from_headers"] is True


def test_fast_mode_fails_closed_for_reference_only_architectures(tmp_path):
    _write_json(tmp_path / "config.json", {"model_type": "qwen3_moe"})
    with pytest.raises(model_spec.ModelSpecError, match="no optimized fast profile"):
        model_spec.inspect_model(tmp_path, mode="fast")


def test_gptoss_rejects_non_mxfp4_or_missing_store_before_loading(tmp_path):
    config = _gptoss_config()
    config["quantization_config"] = {"quant_method": "gptq"}
    _write_json(tmp_path / "config.json", config)
    with pytest.raises(model_spec.ModelSpecError, match="quant_method='mxfp4'"):
        model_spec.inspect_model(tmp_path, tmp_path / "store")

    config["quantization_config"] = {"quant_method": "mxfp4"}
    _write_json(tmp_path / "config.json", config)
    with pytest.raises(model_spec.ModelSpecError, match="requires a validated prepacked"):
        model_spec.inspect_model(tmp_path)


def test_gptoss_rejects_geometry_outside_frozen_kernel_profile(tmp_path):
    config = _gptoss_config()
    config["hidden_size"] = 4096
    _write_json(tmp_path / "config.json", config)
    store_dir = tmp_path / "store"
    _gptoss_store(store_dir)
    with pytest.raises(model_spec.ModelSpecError, match="geometry.hidden"):
        model_spec.inspect_model(tmp_path, store_dir)


def test_safetensors_header_parser_rejects_truncated_payload(tmp_path):
    path = tmp_path / "bad.safetensors"
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [8],
                                     "data_offsets": [0, 16]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(model_spec.ModelSpecError, match="payload is shorter"):
        model_spec._safetensors_header(path)


def test_store_shard_paths_cannot_escape_directory(tmp_path):
    with pytest.raises(model_spec.ModelSpecError, match="escapes"):
        model_spec._safe_child(tmp_path, "../outside.safetensors", "shard")
