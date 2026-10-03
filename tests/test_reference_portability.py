import pytest


@pytest.mark.parametrize("source_dtype", ["bfloat16", "float16"])
def test_tied_embedding_missing_lm_head_preserves_source_dtype_and_logits(tmp_path, source_dtype):
    import torch
    from safetensors.torch import save_file
    from test_reference_backend import _make_family
    from neural_reference.backend import NeuralBackend

    dtype = getattr(torch, source_dtype)
    torch.manual_seed(128)
    config, model_class = _make_family("qwen3_moe")
    config.tie_word_embeddings = True
    config.save_pretrained(tmp_path)
    reference = model_class(config).eval().to(dtype=dtype)
    state = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in reference.state_dict().items()}
    assert "lm_head.weight" in state
    state.pop("lm_head.weight")
    save_file(state, str(tmp_path / "model.safetensors"))

    with NeuralBackend(tmp_path, device="cpu", expert_backend="torch-cpu",
                       load_tokenizer=False) as backend:
        embedding = backend.model.model.embed_tokens.weight
        head = backend.model.lm_head.weight
        assert embedding.dtype is dtype
        assert head.dtype is dtype
        assert embedding.data_ptr() == head.data_ptr()
        input_ids = torch.tensor([[1, 7, 4, 2]], dtype=torch.long)
        with torch.inference_mode():
            expected = reference(input_ids=input_ids, use_cache=False).logits
            actual = backend.forward(input_ids, use_cache=False).logits
        # A BF16 CPU expert reduction can differ by one representable unit even
        # when source tensors and routing are unchanged; FP16 in this fixture is
        # exact. This numerical tolerance is not a general answer-quality test.
        atol = 1e-3 if dtype is torch.bfloat16 else 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=atol)
