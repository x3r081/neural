"""Opt-in real CUDA checks using tiny random architecture fixtures, not pretrained quality tests."""
import os
from types import SimpleNamespace
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("NEURAL_TEST_CUDA") != "1",
                                reason="set NEURAL_TEST_CUDA=1 with the GPU exclusively available")


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe_text", "mixtral"])
def test_cuda_upstream_and_full_cache_prefix_parity(tmp_path, family):
    import torch
    from safetensors.torch import save_file
    from test_reference_backend import _make_family
    from neural_reference.backend import NeuralBackend
    from neural_reference.generation import GenerationEngine
    config, cls = _make_family(family)
    torch.manual_seed(128)
    if family == "qwen3_5_moe_text":
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
        saved_norm = modeling_qwen3_5_moe.FusedRMSNormGated
        try:
            modeling_qwen3_5_moe.FusedRMSNormGated = None
            reference = cls(config).eval()
        finally:
            modeling_qwen3_5_moe.FusedRMSNormGated = saved_norm
    else:
        reference = cls(config).eval()
    config.save_pretrained(tmp_path)
    save_file({k: v.detach().contiguous().clone() for k, v in reference.state_dict().items()},
              str(tmp_path / "model.safetensors"))
    reference = reference.cuda()
    ref_binding = None
    if family == "qwen3_5_moe_text":
        from qwen_neural.fast_delta import bind_torch_gdn
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
        ref_binding = bind_torch_gdn(reference, modeling_module=modeling_qwen3_5_moe)
    class Tokenizer:
        eos_token_id = None
        def __len__(self): return 29
        def decode(self, ids, **kwargs): return " ".join(map(str, ids))
    try:
        with NeuralBackend(tmp_path, device="cuda:0", expert_backend="staged",
                           gpu_expert_budget_gib=.05, native_gpu_layers=1,
                           load_tokenizer=False) as backend:
            ids = torch.tensor([[1, 7, 4, 2]], device="cuda:0")
            with torch.inference_mode():
                expected = reference(input_ids=ids, use_cache=False).logits
                actual = backend.forward(ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            backend.tokenizer = Tokenizer()
            backend.generation_config = SimpleNamespace(eos_token_id=None)
            engine = GenerationEngine(backend, context=256, prefill_chunk=16,
                                      prefix_cache_max_bytes=16*1024**2)
            prompt = [1, 7, 4, 2]*24
            cold = engine.generate(prompt, max_tokens=8, ignore_eos=True)
            warm = engine.generate(prompt, max_tokens=8, ignore_eos=True)
            assert cold['tokens'] == warm['tokens']
            assert warm['prefix_cache']['cached_tokens'] == 80
            engine.prefix_cache.clear()
    finally:
        if ref_binding is not None: ref_binding.close()
        del reference
        torch.cuda.empty_cache()
