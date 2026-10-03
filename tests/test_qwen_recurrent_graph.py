from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from qwen_neural.recurrent_graph import bind_recurrent_graph


def _eager_stub(*args, **kwargs):
    return "eager", args[5]


class Qwen3_5MoeGatedDeltaNet:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.recurrent_gated_delta_rule = _eager_stub


class FakeModel:
    def __init__(self, *modules):
        self._modules = modules

    def modules(self):
        return iter((self, *self._modules))


class FakeCpuTensor:
    def __init__(self, shape=(1, 1, 1, 1)):
        self.shape = shape
        self.dtype = "fake-f32"
        self.device = SimpleNamespace(type="cpu", index=None)

    def stride(self):
        total = 1
        strides = []
        for dim in reversed(self.shape):
            strides.append(total)
            total *= dim
        return tuple(reversed(strides))


def test_binding_falls_back_on_cpu_and_reports_reason():
    module = Qwen3_5MoeGatedDeltaNet(0)
    binding = bind_recurrent_graph(FakeModel(module))
    tensor = FakeCpuTensor()

    result = module.recurrent_gated_delta_rule(
        tensor, tensor, tensor, tensor, tensor, tensor,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )

    assert result == ("eager", tensor)
    report = binding.report()
    assert report["status"] == "not_captured"
    assert report["capture_count"] == report["replay_count"] == 0
    assert report["fallback_count"] == 1
    assert report["fallback_reasons"] == {"non_cuda_tensor": 1}
    assert report["layers_seen"] == [0]
    binding.close()
    assert module.recurrent_gated_delta_rule is _eager_stub


def test_close_restores_only_wrappers_still_owned_by_binding():
    first = Qwen3_5MoeGatedDeltaNet(0)
    second = Qwen3_5MoeGatedDeltaNet(1)
    binding = bind_recurrent_graph(FakeModel(first, second))
    externally_replaced = lambda *args, **kwargs: "external"
    first.recurrent_gated_delta_rule = externally_replaced

    binding.close()

    assert first.recurrent_gated_delta_rule is externally_replaced
    assert second.recurrent_gated_delta_rule is _eager_stub
    assert binding.report()["status"] == "not_captured"


def test_binding_rejects_models_without_qwen_gdn_modules():
    with pytest.raises(ValueError, match="No Qwen3_5MoeGatedDeltaNet"):
        bind_recurrent_graph(FakeModel())


def test_graph_matches_eager_for_layer_reset_and_fallback_signatures():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph parity needs a CUDA device")
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_recurrent_gated_delta_rule,
    )

    class GdnModule(Qwen3_5MoeGatedDeltaNet):
        pass

    # The binder intentionally identifies modules by HF's stable class name.
    GdnModule.__name__ = "Qwen3_5MoeGatedDeltaNet"
    first, second = GdnModule(2), GdnModule(3)
    first.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    second.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    binding = bind_recurrent_graph(FakeModel(first, second))
    device = torch.device("cuda:0")

    def inputs(dtype=torch.bfloat16, state_fill=0.25, seq_len=1, state=None):
        # Match Qwen3.6: 16 K heads repeated to 32 V heads, head dims 128,
        # beta in activation dtype, g produced in FP32, and FP32 SSM state.
        q = torch.randn(1, seq_len, 16, 128, device=device, dtype=dtype).repeat_interleave(2, dim=2)
        k = torch.randn(1, seq_len, 16, 128, device=device, dtype=dtype).repeat_interleave(2, dim=2)
        if seq_len == 1:
            # The cached token path starts from transposed QKV projections;
            # value keeps the projection's wide batch stride when seq_len=1.
            v = torch.empty_strided(
                (1, 1, 32, 128), (8192, 1, 128, 1),
                device=device, dtype=dtype,
            ).normal_()
        else:
            v = torch.randn(1, seq_len, 32, 128, device=device, dtype=dtype)
        g = torch.randn(1, seq_len, 32, device=device, dtype=torch.float32)
        beta = torch.sigmoid(torch.randn(1, seq_len, 32, device=device, dtype=dtype))
        if state is None:
            state = torch.full((1, 32, 128, 128), state_fill, device=device, dtype=torch.float32)
        return q, k, v, g, beta, state

    def call(module, values, final=True):
        return module.recurrent_gated_delta_rule(
            *values, output_final_state=final, use_qk_l2norm_in_kernel=True,
        )

    try:
        # First capture is layer 2; the same immutable signature must replay for
        # layer 3. Then feed the returned state into the next token and reset
        # the state for layer 3, as a separate request would after prefill.
        for module, values in ((first, inputs(state_fill=0.25)),
                               (second, inputs(state_fill=0.0))):
            original_state = values[-1].clone()
            actual_out, actual_state = call(module, values)
            expected_out, expected_state = torch_recurrent_gated_delta_rule(
                *values, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize(device)
            assert torch.equal(actual_out, expected_out)
            assert torch.equal(actual_state, expected_state)
            assert torch.equal(values[-1], original_state)

            continued = list(inputs(state=actual_state))
            continued[-1] = actual_state
            continued_state_before = actual_state.clone()
            continued_out, continued_state = call(module, tuple(continued))
            continued_expected_out, continued_expected_state = torch_recurrent_gated_delta_rule(
                *continued, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize(device)
            assert torch.equal(continued_out, continued_expected_out)
            assert torch.equal(continued_state, continued_expected_state)
            assert torch.equal(actual_state, continued_state_before)

        # Changed sequence length and dtype use eager fallback without changing
        # the captured signature or applying casts to model tensors.
        long_values = inputs(seq_len=2)
        long_actual = call(first, long_values)
        long_expected = torch_recurrent_gated_delta_rule(
            *long_values, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert torch.equal(long_actual[0], long_expected[0])
        assert torch.equal(long_actual[1], long_expected[1])

        float_values = inputs(dtype=torch.float32)
        float_actual = call(second, float_values)
        float_expected = torch_recurrent_gated_delta_rule(
            *float_values, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert torch.equal(float_actual[0], float_expected[0])
        assert torch.equal(float_actual[1], float_expected[1])

        # Calls from distinct caller streams must serialize shared staging and
        # hand each caller its own completed output/state copies.
        streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
        concurrent_values = [None, None]

        def run(index):
            with torch.cuda.stream(streams[index]):
                values = inputs(state_fill=float(index))
                concurrent_values[index] = values
                return call((first, second)[index], concurrent_values[index])

        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(run, range(2)))
        for index, actual in enumerate(concurrent):
            torch.cuda.current_stream(device).wait_stream(streams[index])
            expected = torch_recurrent_gated_delta_rule(
                *concurrent_values[index], output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize(device)
            assert torch.equal(actual[0], expected[0])
            assert torch.equal(actual[1], expected[1])

        report = binding.report()
        assert report["capture_count"] == 1
        assert report["replay_count"] == 6
        assert report["fallback_reasons"] == {
            "not_single_token_decode": 1,
            "signature_mismatch": 1,
        }
        assert report["static_tensor_bytes"] > 0
        assert report["layers_seen"] == [2, 3]
    finally:
        binding.close()
