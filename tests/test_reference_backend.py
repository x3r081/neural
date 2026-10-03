import json
import copy

import pytest


def _make_family(family):
    if family == "qwen3_moe":
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

        config = Qwen3MoeConfig(
            vocab_size=29, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=8, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1,
            num_experts=3, num_experts_per_tok=2,
        )
        return config, Qwen3MoeForCausalLM
    if family.startswith("qwen3_5_moe"):
        from transformers import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

        config = Qwen3_5MoeTextConfig(
            vocab_size=29, hidden_size=16, moe_intermediate_size=8,
            shared_expert_intermediate_size=8, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            linear_key_head_dim=4, linear_value_head_dim=4,
            linear_num_key_heads=2, linear_num_value_heads=2,
            num_experts=3, num_experts_per_tok=2,
            layer_types=["full_attention", "linear_attention"],
        )
        return config, Qwen3_5MoeForCausalLM
    if family == "qwen3_5_moe":
        from transformers import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

        config = Qwen3_5MoeTextConfig(
            vocab_size=29, hidden_size=16, moe_intermediate_size=8,
            shared_expert_intermediate_size=8, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            linear_key_head_dim=4, linear_value_head_dim=4,
            linear_num_key_heads=2, linear_num_value_heads=2,
            num_experts=3, num_experts_per_tok=2,
            layer_types=["full_attention", "linear_attention"],
        )
        return config, Qwen3_5MoeForCausalLM
    if family == "mixtral":
        from transformers import MixtralConfig
        from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

        config = MixtralConfig(
            vocab_size=29, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            num_local_experts=3, num_experts_per_tok=2,
        )
        return config, MixtralForCausalLM
    raise AssertionError(family)


def _split_mixtral_experts(state):
    split = {}
    for key, value in state.items():
        prefix = ".mlp.experts."
        if key.endswith(prefix + "gate_up_proj"):
            base = key[:-len("gate_up_proj")]
            gate, up = value.chunk(2, dim=1)
            for expert in range(value.shape[0]):
                split[f"{base}{expert}.w1.weight"] = gate[expert].clone()
                split[f"{base}{expert}.w3.weight"] = up[expert].clone()
        elif key.endswith(prefix + "down_proj"):
            base = key[:-len("down_proj")]
            for expert in range(value.shape[0]):
                split[f"{base}{expert}.w2.weight"] = value[expert].clone()
        else:
            split[key] = value.clone().contiguous()
    return split


@pytest.mark.parametrize(
    "family,split_source",
    [("qwen3_moe", False), ("qwen3_5_moe_text", False), ("qwen3_5_moe", False),
     ("mixtral", False), ("mixtral", True)],
)
def test_backend_preserves_upstream_full_forward(tmp_path, family, split_source):
    import torch
    from safetensors.torch import save_file
    from neural_reference.backend import NeuralBackend

    config, model_class = _make_family(family)
    if family.startswith("qwen3_5_moe"):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
        saved_norm = modeling_qwen3_5_moe.FusedRMSNormGated
        try:
            modeling_qwen3_5_moe.FusedRMSNormGated = None
            reference = model_class(config).eval()
        finally:
            modeling_qwen3_5_moe.FusedRMSNormGated = saved_norm
    else:
        reference = model_class(config).eval()
    if family == "qwen3_5_moe":
        from transformers import Qwen3_5MoeConfig
        root_config = Qwen3_5MoeConfig(text_config=copy.deepcopy(config))
        root_config.save_pretrained(tmp_path)
    else:
        config.save_pretrained(tmp_path)
    state = {key: value.detach().cpu().contiguous().clone() for key, value in reference.state_dict().items()}
    if family == "qwen3_5_moe":
        state = {
            ("model.language_model." + key.removeprefix("model.")) if key.startswith("model.") else key: value
            for key, value in state.items()
        }
    if split_source:
        state = _split_mixtral_experts(state)
    save_file(state, str(tmp_path / "model.safetensors"))

    if family.startswith("qwen3_5_moe"):
        from qwen_neural.fast_delta import bind_torch_gdn
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
        ref_gdn = bind_torch_gdn(reference, modeling_module=modeling_qwen3_5_moe)
    else:
        ref_gdn = None
    try:
        with NeuralBackend(
            tmp_path, device="cpu", expert_backend="torch-cpu", load_tokenizer=False,
            host_expert_cache_gib=0.01,
        ) as backend:
            input_ids = torch.tensor([[1, 7, 4, 2]], dtype=torch.long)
            with torch.inference_mode():
                expected = reference(input_ids=input_ids, use_cache=False).logits
                actual = backend.forward(input_ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert backend.model_type == family
            assert backend.memory_report()["store"]["host_expert_cache_limit_bytes"] > 0
    finally:
        if ref_gdn is not None:
            ref_gdn.close()


def test_recurrent_graph_rejects_non_qwen35_before_forward(tmp_path):
    from neural_reference.backend import NeuralBackend

    config, model_class = _make_family("mixtral")
    config.save_pretrained(tmp_path)
    # The architecture rejection is intentionally checked before source loading.
    from safetensors.torch import save_file
    import torch
    model = model_class(config)
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
              str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="only for Qwen3.5"):
        NeuralBackend(tmp_path, device="cpu", expert_backend="torch-cpu", recurrent_graph=True,
                     load_tokenizer=False)


def test_close_releases_sparse_modules_and_saved_state(tmp_path):
    import gc
    import weakref
    from safetensors.torch import save_file
    from neural_reference.backend import NeuralBackend
    config, cls = _make_family('qwen3_moe')
    reference = cls(config)
    config.save_pretrained(tmp_path)
    save_file({k: v.detach().clone().contiguous() for k, v in reference.state_dict().items()},
              str(tmp_path / 'model.safetensors'))
    backend = NeuralBackend(tmp_path, device='cpu', expert_backend='torch-cpu', load_tokenizer=False)
    block = weakref.ref(backend._sparse_modules[0][2])
    backend.close()
    gc.collect()
    assert block() is None
