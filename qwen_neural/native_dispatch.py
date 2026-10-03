"""CPU dispatch/aggregation for the native original-BF16 expert kernel.

The model's routed hidden states, expert IDs, and router weights are copied to
CPU once per MoE layer. Expert weights are fetched only for selected experts;
the native kernel returns BF16 expert outputs before router weighting. Results
are accumulated on CPU in ascending expert order, then copied to the model
device once.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any


class NativeExpertWorkspace:
    """Per-backend synchronous scratch and bounded expert-view cache.

    Expert tensors returned by ``BF16ExpertStore.fetch_expert`` are contiguous
    views into its read-only safetensors mmap. This object retains references
    to selected views only; it never materializes an eager expert pool. Cached
    tensor storage stays owned by the store's mmap handles. The
    dispatcher and CPU compute workspace are single-caller and must not be
    shared between concurrently executing backends.
    """

    def __init__(
        self,
        *,
        max_expert_cache_entries: int = 4096,
        max_dispatch_scratch_bytes: int = 128 * 1024 * 1024,
        max_kernel_scratch_bytes: int = 64 * 1024 * 1024,
    ):
        if not isinstance(max_expert_cache_entries, int) or max_expert_cache_entries < 0:
            raise ValueError("max_expert_cache_entries must be a nonnegative integer")
        if not isinstance(max_dispatch_scratch_bytes, int) or max_dispatch_scratch_bytes < 0:
            raise ValueError("max_dispatch_scratch_bytes must be a nonnegative integer")
        if not isinstance(max_kernel_scratch_bytes, int) or max_kernel_scratch_bytes < 0:
            raise ValueError("max_kernel_scratch_bytes must be a nonnegative integer")
        self.max_expert_cache_entries = max_expert_cache_entries
        self.max_dispatch_scratch_bytes = max_dispatch_scratch_bytes
        self.max_kernel_scratch_bytes = max_kernel_scratch_bytes
        self._expert_views: OrderedDict[tuple[int, int], tuple[Any, Any, int]] = OrderedDict()
        self._bound_store = None
        self._expert_view_logical_bytes = 0
        self.expert_cache_hits = 0
        self.expert_cache_misses = 0
        self.expert_cache_evictions = 0
        self._tokens = self._routes = self._hidden = 0
        self._hidden_dtype = self._weights_dtype = None
        self._states = self._route_ids = self._route_weights = self._output = None
        self._dispatch_scratch_bytes = 0
        self._dispatch_reallocations = 0
        self._compute_workspace = None
        self._closed = False

    def fetch_expert(self, layer: int, expert: int, store):
        if self._closed:
            raise RuntimeError("NativeExpertWorkspace is closed")
        if self._bound_store is None:
            self._bound_store = store
        elif self._bound_store is not store:
            raise ValueError(
                "NativeExpertWorkspace is bound to a different expert store; "
                "close it before reusing it with another store"
            )
        key = (int(layer), int(expert))
        pair = self._expert_views.get(key)
        if pair is not None:
            self.expert_cache_hits += 1
            self._expert_views.move_to_end(key)
            return pair[0], pair[1]
        self.expert_cache_misses += 1
        gate_up, down = store.fetch_expert(*key)
        logical_bytes = (
            int(gate_up.numel() * gate_up.element_size())
            + int(down.numel() * down.element_size())
        )
        if self.max_expert_cache_entries > 0:
            while len(self._expert_views) >= self.max_expert_cache_entries:
                _, (_, _, old_bytes) = self._expert_views.popitem(last=False)
                self._expert_view_logical_bytes -= old_bytes
                self.expert_cache_evictions += 1
            self._expert_views[key] = (gate_up, down, logical_bytes)
            self._expert_view_logical_bytes += logical_bytes
        return gate_up, down

    def _prepare_dispatch(self, hidden_states, top_k_index, top_k_weights):
        import torch

        if self._closed:
            raise RuntimeError("NativeExpertWorkspace is closed")
        tokens, hidden = hidden_states.shape
        routes = top_k_index.shape[1]
        t_cap = max(self._tokens, int(tokens))
        r_cap = max(self._routes, int(routes))
        h_cap = max(self._hidden, int(hidden))
        hidden_dtype = hidden_states.dtype
        weights_dtype = top_k_weights.dtype
        needs_alloc = (
            (t_cap, r_cap, h_cap) != (self._tokens, self._routes, self._hidden)
            or hidden_dtype != self._hidden_dtype
            or weights_dtype != self._weights_dtype
        )
        needed = (
            t_cap * h_cap * hidden_states.element_size() * 2
            + t_cap * r_cap * 8
            + t_cap * r_cap * top_k_weights.element_size()
        )
        if needed > self.max_dispatch_scratch_bytes:
            raise MemoryError(
                f"native dispatch scratch needs {needed} bytes, limit is {self.max_dispatch_scratch_bytes}"
            )
        if needs_alloc:
            self._states = torch.empty((t_cap * h_cap,), dtype=hidden_dtype, device="cpu")
            self._route_ids = torch.empty((t_cap * r_cap,), dtype=torch.int64, device="cpu")
            self._route_weights = torch.empty((t_cap * r_cap,), dtype=weights_dtype, device="cpu")
            self._output = torch.empty((t_cap * h_cap,), dtype=hidden_dtype, device="cpu")
            self._tokens, self._routes, self._hidden = t_cap, r_cap, h_cap
            self._hidden_dtype, self._weights_dtype = hidden_dtype, weights_dtype
            self._dispatch_scratch_bytes = needed
            self._dispatch_reallocations += 1
        states = self._states.view(-1)[: tokens * hidden].view(tokens, hidden)
        route_ids = self._route_ids.view(-1)[: tokens * routes].view(tokens, routes)
        route_weights = self._route_weights.view(-1)[: tokens * routes].view(tokens, routes)
        output = self._output.view(-1)[: tokens * hidden].view(tokens, hidden)
        # Pageable host copies are synchronous by design. Nothing continues
        # reading these buffers after dispatch returns.
        states.copy_(hidden_states.detach(), non_blocking=False)
        route_ids.copy_(top_k_index.detach(), non_blocking=False)
        route_weights.copy_(top_k_weights.detach(), non_blocking=False)
        output.zero_()
        return states, route_ids, route_weights, output

    def dispatch(
        self, hidden_states, top_k_index, top_k_weights, *, layer: int,
        store, threads: int = 8, run_expert_fn=None, run_experts_fn=None,
    ):
        return dispatch_native_experts(
            hidden_states,
            top_k_index,
            top_k_weights,
            layer=layer,
            store=store,
            threads=threads,
            run_expert_fn=run_expert_fn,
            run_experts_fn=run_experts_fn,
            workspace=self,
        )

    def report(self) -> dict[str, Any]:
        kernel = self._compute_workspace.report() if self._compute_workspace is not None else {
            "scratch_bytes": 0,
            "scratch_limit_bytes": self.max_kernel_scratch_bytes,
            "scratch_reallocations": 0,
            "closed": self._closed,
        }
        return {
            "cached_expert_view_entries": len(self._expert_views),
            "expert_view_entry_limit": self.max_expert_cache_entries,
            "cached_expert_view_logical_bytes": self._expert_view_logical_bytes,
            "cached_expert_view_workspace_copy_bytes": 0,
            "cached_expert_view_source": (
                f"{type(self._bound_store).__module__}.{type(self._bound_store).__qualname__}"
                if self._bound_store is not None else None
            ),
            "cached_expert_view_provenance": (
                "fetch_expert tensors retained by reference; workspace does not copy or mutate weights"
            ),
            "expert_cache_hits": self.expert_cache_hits,
            "expert_cache_misses": self.expert_cache_misses,
            "expert_cache_evictions": self.expert_cache_evictions,
            "dispatch_scratch_bytes": self._dispatch_scratch_bytes,
            "dispatch_scratch_limit_bytes": self.max_dispatch_scratch_bytes,
            "dispatch_scratch_reallocations": self._dispatch_reallocations,
            "kernel_workspace": kernel,
            "closed": self._closed,
        }

    def close(self) -> None:
        if self._compute_workspace is not None:
            self._compute_workspace.close()
            self._compute_workspace = None
        self._expert_views.clear()
        self._bound_store = None
        self._expert_view_logical_bytes = 0
        self._states = self._route_ids = self._route_weights = self._output = None
        self._dispatch_scratch_bytes = 0
        self._tokens = self._routes = self._hidden = 0
        self._closed = True


def dispatch_native_experts(
    hidden_states,
    top_k_index,
    top_k_weights,
    *,
    layer: int,
    store,
    threads: int = 8,
    run_expert_fn=None,
    run_experts_fn=None,
    workspace: NativeExpertWorkspace | None = None,
):
    """Compute routed experts and aggregate on CPU.

    Args are the tensors passed to Qwen's expert module: hidden states
    ``[tokens, hidden]``, expert IDs ``[tokens, routes]``, and BF16 route
    weights with the same leading dimensions. `store.fetch_expert(layer, id)`
    must return original contiguous CPU BF16 ``(gate_up, down)`` matrices.
    `run_expert_fn` and `run_experts_fn` are injectable for small CPU tests;
    normal callers should omit them to use the native kernel wrappers.

    For a single token, all routes are sent through one native batched call.
    For prefill, tokens are grouped by expert to reuse each selected expert
    matrix for all of its hits. Each weighted contribution is rounded to the
    hidden-state dtype before ordered accumulation, matching the existing
    native backend path's BF16 weighting boundary.
    """
    import torch

    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if top_k_index.ndim != 2 or top_k_weights.ndim != 2:
        raise ValueError("top_k_index and top_k_weights must have shape [tokens, routes]")
    if tuple(top_k_index.shape) != tuple(top_k_weights.shape):
        raise ValueError("top_k_index and top_k_weights must have the same shape")
    if hidden_states.shape[0] != top_k_index.shape[0]:
        raise ValueError("hidden_states and routing tensors must have the same token count")
    if hidden_states.shape[0] <= 0 or hidden_states.shape[1] <= 0 or top_k_index.shape[1] <= 0:
        raise ValueError("hidden and routing dimensions must be non-empty")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    use_default_run_expert = run_expert_fn is None
    use_default_run_experts = hidden_states.shape[0] == 1 and run_experts_fn is None
    if run_expert_fn is None or use_default_run_experts:
        from . import cpu_experts

        if run_expert_fn is None:
            run_expert_fn = cpu_experts.run_expert
        if run_experts_fn is None:
            run_experts_fn = cpu_experts.run_experts

    # Do exactly one device transfer for each of the three layer inputs. The
    # detached contiguous tensors are read-only in this dispatch operation.
    if workspace is None:
        states_cpu = hidden_states.detach().to(device="cpu").contiguous()
        ids_cpu = top_k_index.detach().to(device="cpu", dtype=torch.int64).contiguous()
        weights_cpu = top_k_weights.detach().to(device="cpu").contiguous()
        output_cpu = torch.zeros_like(states_cpu)
    else:
        states_cpu, ids_cpu, weights_cpu, output_cpu = workspace._prepare_dispatch(
            hidden_states, top_k_index, top_k_weights
        )
    if workspace is not None and (use_default_run_expert or use_default_run_experts):
        from .cpu_experts import ExpertWorkspace

        if workspace._compute_workspace is None:
            workspace._compute_workspace = ExpertWorkspace(
                max_scratch_bytes=workspace.max_kernel_scratch_bytes
            )

    def fetched(expert_id: int, cache: dict[int, tuple]):
        pair = cache.get(expert_id)
        if pair is None:
            pair = (
                workspace.fetch_expert(layer, expert_id, store)
                if workspace is not None
                else store.fetch_expert(layer, expert_id)
            )
            cache[expert_id] = pair
        return pair

    if states_cpu.shape[0] == 1:
        # The old expert loop visits hits in ascending expert id and positions
        # within an expert in ascending route order. Stable sort preserves that
        # same order while allowing all routes to share one native OpenMP team.
        route_order = sorted(
            range(ids_cpu.shape[1]), key=lambda pos: int(ids_cpu[0, pos])
        )
        pairs: dict[int, tuple] = {}
        route_pairs = [fetched(int(ids_cpu[0, pos]), pairs) for pos in route_order]
        gate_up = [pair[0] for pair in route_pairs]
        down = [pair[1] for pair in route_pairs]
        if workspace is not None and use_default_run_experts:
            computed = run_experts_fn(
                states_cpu[0], gate_up, down, threads=threads,
                workspace=workspace._compute_workspace,
            )
        else:
            computed = run_experts_fn(states_cpu[0], gate_up, down, threads=threads)
        if computed.ndim == 2:
            computed = computed.unsqueeze(0)
        if tuple(computed.shape[:2]) != (1, len(route_order)):
            raise ValueError("run_experts_fn returned an unexpected route shape")
        for sorted_pos, route_pos in enumerate(route_order):
            weighted = (
                computed[0, sorted_pos] * weights_cpu[0, route_pos]
            ).to(dtype=output_cpu.dtype)
            output_cpu[0].add_(weighted)
    else:
        # Shape [routes,tokens] reproduces torch.where(mask[expert_id]) order
        # from the backend: route position first, token index second.
        unique_ids = torch.unique(ids_cpu, sorted=True).tolist()
        pairs = {}
        for expert_id in unique_ids:
            expert_id = int(expert_id)
            route_pos, token_idx = torch.where(ids_cpu.transpose(0, 1) == expert_id)
            if route_pos.numel() == 0:
                continue
            gate_up, down = fetched(expert_id, pairs)
            states = states_cpu[token_idx]
            if workspace is not None and use_default_run_expert:
                computed = run_expert_fn(
                    states, gate_up, down, threads=threads,
                    workspace=workspace._compute_workspace,
                )
            else:
                computed = run_expert_fn(states, gate_up, down, threads=threads)
            if tuple(computed.shape) != (token_idx.numel(), hidden_states.shape[1]):
                raise ValueError("run_expert_fn returned an unexpected output shape")
            weighted = (
                computed * weights_cpu[token_idx, route_pos, None]
            ).to(dtype=output_cpu.dtype)
            output_cpu.index_add_(0, token_idx, weighted)

    result = output_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=False)
    # On CPU, .to may return the workspace's reusable storage itself.
    return result.clone() if workspace is not None and result.device.type == "cpu" else result
