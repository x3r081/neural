import json

import pytest

from neural_reference.adapters import adapter_for_type, inspect_checkpoint


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text", "qwen3_moe", "mixtral"])
def test_supported_architectures_have_explicit_adapters(model_type):
    adapter = adapter_for_type(model_type)
    assert adapter.model_type == model_type
    assert adapter.class_name.endswith("ForCausalLM")


def test_unknown_architecture_fails_closed():
    with pytest.raises(ValueError, match="Unsupported MoE architecture"):
        adapter_for_type("unrecognized_moe")


def test_inspect_checkpoint_reads_headers_without_loading_model(tmp_path):
    from safetensors.torch import save_file
    import torch

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "mixtral",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
    }), encoding="utf-8")
    save_file({
        "model.embed_tokens.weight": torch.zeros((4, 4), dtype=torch.float32),
        "model.layers.0.block_sparse_moe.experts.gate_up_proj": torch.zeros((2, 8, 4), dtype=torch.bfloat16),
        "model.layers.0.block_sparse_moe.experts.down_proj": torch.zeros((2, 4, 4), dtype=torch.bfloat16),
    }, str(tmp_path / "model.safetensors"))

    profile = inspect_checkpoint(tmp_path)
    assert profile["model_type"] == "mixtral"
    assert profile["hidden_size"] == 4
    assert profile["sparse_layers"] == [0]
    assert profile["expert_count"] == 2
    assert profile["dtypes"] == ["bf16", "f32"]
    assert profile["core_bytes"] == 64
    assert profile["expert_bytes"] == 2 * 2 * (8 * 4 + 4 * 4)
    assert profile["layer_expert_bytes"] == {"0": profile["expert_bytes"]}
    assert profile["largest_expert_bytes"] == profile["expert_bytes"] // 2
    assert "safetensors" in profile["format"]


def test_quantized_config_is_rejected_before_tensor_loading(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "mixtral", "quantization_config": {"load_in_4bit": True},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Quantized checkpoints are not supported"):
        inspect_checkpoint(tmp_path)


def test_inspect_checkpoint_rejects_shard_path_escape(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mixtral"}), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.layers.0.mlp.experts.gate_up_proj": "../outside.safetensors"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes checkpoint directory"):
        inspect_checkpoint(tmp_path)
