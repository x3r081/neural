"""Small correctness checks for the original-BF16 Qwen CPU expert kernel."""
import pytest

torch = pytest.importorskip("torch")

from qwen_neural import cpu_experts


@pytest.fixture(scope="module", autouse=True)
def _native_dll_available():
    if not cpu_experts.available():
        pytest.skip("build qwen_bf16_experts.dll with tools/build_qwen_cpu.ps1")
    cpu_experts.load_library()


def _reference(x, gate_up, down):
    import torch.nn.functional as F

    outputs = []
    for gu, dn in zip(gate_up, down):
        gate, up = F.linear(x, gu).chunk(2, dim=-1)
        activated = F.silu(gate) * up
        outputs.append(F.linear(activated, dn))
    return torch.stack(outputs, dim=1)


def _error_stats(actual, expected):
    delta = actual.float() - expected.float()
    rel_l2 = delta.norm() / expected.float().norm().clamp_min(1e-12)
    max_abs_normalized = delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)
    mismatch_count = int(torch.count_nonzero(actual != expected))
    return float(rel_l2), float(max_abs_normalized), mismatch_count


def test_bf16_experts_match_torch_rounding_and_route_layout():
    torch.manual_seed(41)
    hidden, intermediate, routes, tokens = 64, 32, 3, 4
    x = (torch.randn(tokens, hidden) * 0.3).to(torch.bfloat16).contiguous()
    gate_up = [
        (torch.randn(2 * intermediate, hidden) * 0.08).to(torch.bfloat16).contiguous()
        for _ in range(routes)
    ]
    down = [
        (torch.randn(hidden, intermediate) * 0.08).to(torch.bfloat16).contiguous()
        for _ in range(routes)
    ]

    actual = cpu_experts.run_experts(x, gate_up, down, threads=2)
    expected = _reference(x, gate_up, down)
    assert actual.shape == (tokens, routes, hidden)
    assert actual.dtype is torch.bfloat16
    # BF16 activation boundaries are reproduced; only FP32 dot-product
    # reduction order differs from PyTorch's CPU linear backend.
    rel_l2, max_abs_normalized, mismatches = _error_stats(actual, expected)
    assert rel_l2 < 0.02 and max_abs_normalized < 0.06, (
        f"rel_l2={rel_l2:.6g}, max_abs_normalized={max_abs_normalized:.6g}, "
        f"BF16 mismatch_count={mismatches}/{actual.numel()}"
    )


def test_one_expert_accepts_a_single_input_vector_and_stacked_weights():
    torch.manual_seed(3)
    hidden, intermediate = 48, 24
    x = torch.zeros(hidden, dtype=torch.bfloat16)
    gu = torch.zeros(2 * intermediate, hidden, dtype=torch.bfloat16)
    dn = torch.zeros(hidden, intermediate, dtype=torch.bfloat16)

    actual = cpu_experts.run_expert(x, gu, dn, threads=1)
    stacked = cpu_experts.run_experts(
        x.unsqueeze(0), gu.unsqueeze(0), dn.unsqueeze(0), threads=1
    )
    assert actual.shape == (1, hidden)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert torch.equal(actual[:, None], stacked)


def test_bf16_silu_handles_large_finite_gate_values():
    hidden, intermediate = 32, 16
    x = torch.ones(1, hidden, dtype=torch.bfloat16)
    gu = torch.zeros(2 * intermediate, hidden, dtype=torch.bfloat16)
    gu[:intermediate:2].fill_(12)
    gu[1:intermediate:2].fill_(-12)
    gu[intermediate:].fill_(0.25)
    dn = torch.ones(hidden, intermediate, dtype=torch.bfloat16) * 0.01

    actual = cpu_experts.run_expert(x, gu, dn, threads=1)
    expected = _reference(x, [gu], [dn])[:, 0]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.01)


def test_known_cancellation_tie_silu_rounding_and_route_order():
    hidden, intermediate = 32, 1
    x = torch.zeros(hidden, dtype=torch.bfloat16)
    x[0] = 1.0
    x[1] = 1.0 / 256.0

    # Route 0 gate dot is exactly 1 + 1/256, halfway between BF16 values;
    # ties-to-even must round it to 1.0 before SiLU. Its up is 2. Route 1
    # has gate=up=2. All down weights are one, making the expected outputs
    # independently known and different by route.
    gu0 = torch.zeros(2, hidden, dtype=torch.bfloat16)
    gu0[0, 0] = 1.0
    gu0[0, 1] = 1.0
    gu0[1, 0] = 2.0
    gu1 = torch.zeros_like(gu0)
    gu1[0, 0] = 2.0
    gu1[1, 0] = 2.0
    dn = torch.ones(hidden, intermediate, dtype=torch.bfloat16)

    actual = cpu_experts.run_experts(x, [gu0, gu1], [dn, dn], threads=1)
    expected = torch.stack(
        [
            torch.full((hidden,), 1.4609375, dtype=torch.bfloat16),
            torch.full((hidden,), 3.515625, dtype=torch.bfloat16),
        ]
    )
    assert torch.equal(actual[0], expected), (
        f"known-value route/order mismatch: got {actual[0, :, 0].tolist()}, "
        f"expected {expected[:, 0].tolist()}"
    )


def test_seeded_model_shape_bf16_error_report():
    torch.manual_seed(2026)
    hidden, intermediate, routes = 2048, 512, 8
    x = (torch.randn(1, hidden) * 0.5).to(torch.bfloat16).contiguous()
    gate_up = [
        (torch.randn(2 * intermediate, hidden) * 0.02).to(torch.bfloat16).contiguous()
        for _ in range(routes)
    ]
    down = [
        (torch.randn(hidden, intermediate) * 0.02).to(torch.bfloat16).contiguous()
        for _ in range(routes)
    ]

    actual = cpu_experts.run_experts(x, gate_up, down, threads=8)
    expected = _reference(x, gate_up, down)
    rel_l2, max_abs_normalized, mismatches = _error_stats(actual, expected)
    report = (
        f"model-shape BF16: rel_l2={rel_l2:.6g}, "
        f"max_abs_normalized={max_abs_normalized:.6g}, "
        f"mismatch_count={mismatches}/{actual.numel()}"
    )
    print(report)
    assert rel_l2 < 0.015 and max_abs_normalized < 0.04, report
