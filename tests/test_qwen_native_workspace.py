"""CPU-only lifecycle/order tests for the reusable native dispatch workspace."""

from __future__ import annotations

import pytest
import torch

from qwen_neural import cpu_experts
from qwen_neural.native_dispatch import NativeExpertWorkspace, dispatch_native_experts


class _ViewStore:
    def __init__(self, hidden=4, intermediate=2, experts=4):
        self.pairs = {}
        self.fetches = []
        for expert in range(experts):
            tag = torch.tensor(float(expert + 1), dtype=torch.bfloat16)
            gate_up = tag.expand(2 * intermediate, hidden).contiguous()
            down = torch.ones(hidden, intermediate, dtype=torch.bfloat16)
            self.pairs[expert] = (gate_up, down)

    def fetch_expert(self, layer, expert):
        self.fetches.append((layer, expert))
        return self.pairs[expert]


def _run_routes(x, gate_up, down, *, threads):
    values = torch.tensor([float(pair[0, 0]) for pair in gate_up], dtype=torch.bfloat16)
    tokens = x.shape[0] if x.ndim == 2 else 1
    hidden = x.shape[-1] if x.ndim == 2 else x.shape[0]
    return values.view(1, -1, 1).expand(tokens, -1, hidden).clone()


def _run_one(x, gate_up, down, *, threads):
    value = float(gate_up[0, 0])
    return torch.full((x.shape[0], x.shape[-1]), value, dtype=torch.bfloat16)


def test_workspace_reuses_expert_views_and_dispatch_scratch_without_output_alias():
    store = _ViewStore()
    workspace = NativeExpertWorkspace(max_expert_cache_entries=8)
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    ids = torch.tensor([[2, 1, 2, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.25, 0.125, 0.25]], dtype=torch.bfloat16)

    first = workspace.dispatch(
        hidden, ids, weights, layer=5, store=store, threads=2,
        run_experts_fn=_run_routes,
    )
    assert torch.equal(first, torch.full_like(first, 2.625))
    assert store.fetches == [(5, 0), (5, 1), (5, 2)]
    assert workspace._expert_views[(5, 0)][0] is store.pairs[0][0]

    first_saved = first.clone()
    weights2 = torch.tensor([[0.125, 0.5, 0.25, 0.125]], dtype=torch.bfloat16)
    second = workspace.dispatch(
        hidden, ids, weights2, layer=5, store=store, threads=2,
        run_experts_fn=_run_routes,
    )
    assert torch.equal(first, first_saved)
    assert first.data_ptr() != second.data_ptr()
    assert store.fetches == [(5, 0), (5, 1), (5, 2)]
    report = workspace.report()
    assert report["expert_cache_hits"] == 3
    assert report["cached_expert_view_entries"] == 3
    assert report["cached_expert_view_workspace_copy_bytes"] == 0
    assert report["dispatch_scratch_reallocations"] == 1
    workspace.close()
    assert workspace.report()["cached_expert_view_entries"] == 0
    assert torch.equal(first, first_saved)


def test_workspace_lru_and_scratch_budget_are_bounded():
    store = _ViewStore()
    pair_bytes = sum(t.numel() * t.element_size() for t in store.pairs[0])
    workspace = NativeExpertWorkspace(
        max_expert_cache_entries=1,
        max_dispatch_scratch_bytes=64,
    )
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    weights = torch.ones((1, 1), dtype=torch.bfloat16)

    for expert in (0, 1):
        workspace.dispatch(
            hidden,
            torch.tensor([[expert]], dtype=torch.int64),
            weights,
            layer=0,
            store=store,
            run_experts_fn=_run_routes,
        )
    report = workspace.report()
    assert report["cached_expert_view_entries"] == 1
    assert report["expert_view_entry_limit"] == 1
    assert report["cached_expert_view_logical_bytes"] == pair_bytes
    assert report["expert_cache_evictions"] == 1
    assert report["dispatch_scratch_bytes"] <= report["dispatch_scratch_limit_bytes"]

    too_small = NativeExpertWorkspace(max_dispatch_scratch_bytes=1)
    with pytest.raises(MemoryError, match="dispatch scratch"):
        too_small.dispatch(
            hidden,
            torch.tensor([[0]], dtype=torch.int64),
            weights,
            layer=0,
            store=store,
            run_experts_fn=_run_routes,
        )
    assert store.fetches == [(0, 0), (0, 1)]
    with pytest.raises(RuntimeError, match="closed"):
        too_small.close()
        too_small.dispatch(
            hidden,
            torch.tensor([[0]], dtype=torch.int64),
            weights,
            layer=0,
            store=store,
            run_experts_fn=_run_routes,
        )
    workspace.close()


def test_workspace_rejects_a_different_store_and_handles_shape_dtype_changes():
    store = _ViewStore()
    workspace = NativeExpertWorkspace(max_expert_cache_entries=8)
    ids = torch.tensor([[1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
    first = workspace.dispatch(
        torch.zeros((1, 4), dtype=torch.bfloat16), ids, weights,
        layer=0, store=store, run_expert_fn=_run_one, run_experts_fn=_run_routes,
    )
    first_saved = first.clone()

    # A larger token shape expands reusable capacity, and the route-weight
    # dtype change is reflected in the correctly typed scratch allocation.
    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    weights_fp32 = torch.tensor([[0.5, 0.25], [0.25, 0.5]], dtype=torch.float32)
    second = workspace.dispatch(
        x, ids.expand(2, -1), weights_fp32,
        layer=0, store=store, run_expert_fn=_run_one, run_experts_fn=_run_routes,
    )
    assert second.shape == (2, 4)
    assert second.dtype is torch.bfloat16
    assert torch.equal(
        second,
        torch.tensor([[1.25] * 4, [1.0] * 4], dtype=torch.bfloat16),
    )
    assert torch.equal(first, first_saved)
    assert workspace.report()["dispatch_scratch_reallocations"] == 2

    other_store = _ViewStore()
    with pytest.raises(ValueError, match="different expert store"):
        workspace.dispatch(
            x[:1], ids, weights, layer=0, store=other_store,
            run_expert_fn=_run_one, run_experts_fn=_run_routes,
        )
    assert other_store.fetches == []
    workspace.close()


def test_cpu_kernel_workspace_keeps_prior_results_and_reuses_scratch():
    if not cpu_experts.available():
        pytest.skip("build qwen_bf16_experts.dll with tools/build_qwen_cpu.ps1")
    torch.manual_seed(2026)
    hidden, intermediate, routes, tokens = 16, 8, 2, 3
    x = torch.randn(tokens, hidden).to(torch.bfloat16).contiguous()
    gate_up = [torch.randn(2 * intermediate, hidden).to(torch.bfloat16).contiguous() for _ in range(routes)]
    down = [torch.randn(hidden, intermediate).to(torch.bfloat16).contiguous() for _ in range(routes)]
    workspace = cpu_experts.ExpertWorkspace(max_scratch_bytes=1024 * 1024)

    reference = cpu_experts.run_experts(x, gate_up, down, threads=2)
    first = cpu_experts.run_experts(x, gate_up, down, threads=2, workspace=workspace)
    first_saved = first.clone()
    second = cpu_experts.run_experts(x + 1, gate_up, down, threads=2, workspace=workspace)
    assert torch.equal(first, reference)
    assert torch.equal(first, first_saved)
    assert first.data_ptr() != second.data_ptr()
    assert workspace.report()["scratch_reallocations"] == 1
    assert workspace.report()["scratch_bytes"] <= workspace.report()["scratch_limit_bytes"]
    workspace.close()
    with pytest.raises(RuntimeError, match="closed"):
        cpu_experts.run_experts(x, gate_up, down, threads=2, workspace=workspace)
