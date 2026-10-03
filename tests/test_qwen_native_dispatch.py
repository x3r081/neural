"""CPU-only ordering checks for native expert dispatch and aggregation."""
import pytest
import torch

from qwen_neural.native_dispatch import dispatch_native_experts
from qwen_neural import cpu_experts


class _Store:
    def __init__(self, hidden=4):
        self.hidden = hidden
        self.fetches = []

    def fetch_expert(self, layer, expert):
        self.fetches.append((layer, expert))
        tag = torch.tensor(float(expert + 1), dtype=torch.bfloat16)
        return tag.expand(2, self.hidden).contiguous(), tag.expand(self.hidden, 1).contiguous()


def _constant_results(experts, hidden):
    return torch.stack([
        torch.full((hidden,), float(int(gu[0, 0]) ), dtype=torch.bfloat16)
        for gu in experts
    ]).unsqueeze(0)


def test_single_token_batches_routes_sorted_once_and_accumulates_bf16():
    store = _Store()
    calls = []

    def run_many(x, gate_up, down, *, threads):
        route_ids = [int(gu[0, 0]) - 1 for gu in gate_up]
        calls.append((route_ids, threads, tuple(x.shape)))
        return _constant_results(gate_up, x.shape[-1])

    hidden = torch.zeros(1, 4, dtype=torch.bfloat16)
    ids = torch.tensor([[2, 1, 2, 0]], dtype=torch.long)
    weights = torch.tensor([[0.5, 0.25, 0.125, 0.25]], dtype=torch.bfloat16)
    actual = dispatch_native_experts(
        hidden, ids, weights, layer=7, store=store, threads=3,
        run_experts_fn=run_many,
    )

    assert actual.shape == hidden.shape and actual.dtype == torch.bfloat16
    assert torch.equal(actual, torch.full_like(hidden, 2.625))
    assert store.fetches == [(7, 0), (7, 1), (7, 2)]  # duplicate route reuses fetched weights
    assert calls == [([0, 1, 2, 2], 3, (4,))]


def test_prefill_groups_sorted_experts_and_preserves_hit_iteration_order():
    store = _Store(hidden=1)
    calls = []

    def run_one(states, gate_up, down, *, threads):
        expert_id = int(gate_up[0, 0]) - 1
        calls.append((expert_id, states[:, 0].tolist(), threads))
        return torch.full((states.shape[0], 1), float(expert_id + 1), dtype=torch.bfloat16)

    hidden = torch.tensor([[10.0], [20.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)
    weights = torch.tensor([[0.125, 0.25], [0.5, 0.25]], dtype=torch.bfloat16)
    actual = dispatch_native_experts(
        hidden, ids, weights, layer=3, store=store, threads=2,
        run_expert_fn=run_one,
    )

    expected = torch.tensor([[0.625], [1.75]], dtype=torch.bfloat16)
    assert torch.equal(actual, expected)
    assert calls == [(0, [10.0], 2), (1, [20.0], 2), (2, [10.0, 20.0], 2)]
    assert store.fetches == [(3, 0), (3, 1), (3, 2)]


def test_single_token_dispatch_calls_real_native_route_batch():
    if not cpu_experts.available():
        pytest.skip("build qwen_bf16_experts.dll with tools/build_qwen_cpu.ps1")

    torch.manual_seed(17)
    hidden_size, intermediate = 32, 8
    pairs = [
        (
            (torch.randn(2 * intermediate, hidden_size) * 0.05).to(torch.bfloat16).contiguous(),
            (torch.randn(hidden_size, intermediate) * 0.05).to(torch.bfloat16).contiguous(),
        )
        for _ in range(2)
    ]

    class Store:
        def fetch_expert(self, layer, expert):
            return pairs[expert]

    x = (torch.randn(1, hidden_size) * 0.2).to(torch.bfloat16).contiguous()
    ids = torch.tensor([[1, 0, 1]], dtype=torch.long)
    weights = torch.tensor([[0.25, 0.5, 0.125]], dtype=torch.bfloat16)
    actual = dispatch_native_experts(x, ids, weights, layer=0, store=Store(), threads=2)

    import torch.nn.functional as F

    expected = torch.zeros_like(x)
    for route_pos in sorted(range(ids.shape[1]), key=lambda pos: int(ids[0, pos])):
        expert_id = int(ids[0, route_pos])
        gate, up = F.linear(x[0], pairs[expert_id][0]).chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, pairs[expert_id][1])
        expected[0].add_((expert_output * weights[0, route_pos]).to(torch.bfloat16))
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.003)
