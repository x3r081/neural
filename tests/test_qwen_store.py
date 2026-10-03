import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from qwen_neural.backend import QwenBackend, _set_module_tensor_from_source
from qwen_neural.store import BF16ExpertStore


def test_store_slices_exact_bf16_expert_without_loading_pool(tmp_path):
    gate = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3).to(torch.bfloat16)
    down = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).to(torch.bfloat16)
    shard = "weights.safetensors"
    safetensors.save_file(
        {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        },
        str(tmp_path / shard),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": shard,
                    "model.language_model.layers.0.mlp.experts.down_proj": shard,
                }
            }
        ),
        encoding="utf-8",
    )

    with BF16ExpertStore(tmp_path) as store:
        gate_row, down_row = store.fetch_expert(0, 1)
        assert gate_row.dtype == torch.bfloat16
        assert down_row.dtype == torch.bfloat16
        assert gate_row.is_contiguous() and down_row.is_contiguous()
        torch.testing.assert_close(gate_row, gate[1])
        torch.testing.assert_close(down_row, down[1])
        assert store.report()["expert_reads"] == 1
        assert store.report()["projection_bytes_sliced"] == gate[1].numel() * 2 + down[1].numel() * 2


def test_store_rejects_non_bf16_experts(tmp_path):
    shard = "weights.safetensors"
    tensors = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros((2, 4, 3), dtype=torch.float16),
        "model.layers.0.mlp.experts.down_proj": torch.zeros((2, 3, 4), dtype=torch.float16),
    }
    safetensors.save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}}),
        encoding="utf-8",
    )

    with BF16ExpertStore(tmp_path) as store, pytest.raises(TypeError, match="original BF16"):
        store.fetch_expert(0, 0)


def test_meta_parameter_replacement_preserves_checkpoint_dtype():
    module = torch.nn.Module()
    module.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty((2, 3), device="meta", dtype=torch.float32)),
    )
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)

    _set_module_tensor_from_source(module, "weight", source, torch.device("cpu"))

    assert not module.weight.is_meta
    assert module.weight.device.type == "cpu"
    assert module.weight.dtype == torch.bfloat16
    torch.testing.assert_close(module.weight, source)


def test_forward_allows_caller_to_override_use_cache():
    class TinyModel(torch.nn.Module):
        def forward(self, *, input_ids, past_key_values=None, use_cache=True):
            return input_ids.device.type, past_key_values, use_cache

    backend = object.__new__(QwenBackend)
    backend.device = torch.device("cpu")
    backend.model = TinyModel()
    ids = torch.tensor([[5, 7]])

    assert backend.forward(ids, past_key_values="cache", use_cache=False) == (
        "cpu", "cache", False
    )
    assert backend.forward(ids)[2] is True
