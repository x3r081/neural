from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from qwen_neural.backend import _construct_model_with_torch_norm
from qwen_neural.fast_delta import bind_fla_gdn, bind_torch_gdn


def _model(*modules):
    return SimpleNamespace(modules=lambda: list(modules))


def _gdn_class(*, recurrent=True):
    cls = type("Qwen3_5MoeGatedDeltaNet", (), {})
    module = cls()
    module.chunk_gated_delta_rule = lambda *a, **k: "old-chunk"
    if recurrent:
        module.recurrent_gated_delta_rule = lambda *a, **k: "old-recurrent"
    return module


def test_torch_binding_explicitly_selects_hf_reference_ops_and_restores():
    module = _gdn_class()
    original_chunk = module.chunk_gated_delta_rule
    original_recurrent = module.recurrent_gated_delta_rule
    torch_chunk = lambda *a, **k: "torch-chunk"
    torch_recurrent = lambda *a, **k: "torch-recurrent"
    hf = SimpleNamespace(
        torch_chunk_gated_delta_rule=torch_chunk,
        torch_recurrent_gated_delta_rule=torch_recurrent,
    )

    binding = bind_torch_gdn(_model(module), modeling_module=hf)
    assert binding.module_count == 1
    assert module.chunk_gated_delta_rule is torch_chunk
    assert module.recurrent_gated_delta_rule is torch_recurrent
    binding.close()
    assert module.chunk_gated_delta_rule is original_chunk
    assert module.recurrent_gated_delta_rule is original_recurrent


def test_fla_binding_is_explicit_and_rolls_back_partial_failure():
    first, incomplete = _gdn_class(), _gdn_class(recurrent=False)
    first_chunk = first.chunk_gated_delta_rule
    fla_chunk = lambda *a, **k: "fla-chunk"
    fla_recurrent = lambda *a, **k: "fla-recurrent"
    ops = SimpleNamespace(
        chunk_gated_delta_rule=fla_chunk,
        fused_recurrent_gated_delta_rule=fla_recurrent,
    )

    with pytest.raises(TypeError, match="callable HF kernel slots"):
        bind_fla_gdn(_model(first, incomplete), ops_module=ops)
    assert first.chunk_gated_delta_rule is first_chunk
    assert first.chunk_gated_delta_rule is not fla_chunk

    binding = bind_fla_gdn(_model(first), ops_module=ops)
    assert first.chunk_gated_delta_rule is fla_chunk
    assert first.recurrent_gated_delta_rule is fla_recurrent
    binding.close()
    assert first.chunk_gated_delta_rule is first_chunk


def test_model_construction_uses_torch_norm_then_restores_import_default():
    fused = object()
    seen = []

    class Model:
        def __init__(self, config):
            seen.append((config, modeling.FusedRMSNormGated))

    modeling = SimpleNamespace(FusedRMSNormGated=fused, Qwen3_5MoeForCausalLM=Model)

    @contextmanager
    def init_empty_weights(*, include_buffers):
        assert include_buffers is False
        yield

    result = _construct_model_with_torch_norm(modeling, "tiny-config", init_empty_weights)
    assert isinstance(result, Model)
    assert seen == [("tiny-config", None)]
    assert modeling.FusedRMSNormGated is fused
