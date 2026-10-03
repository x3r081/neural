"""Exact-dtype expert math shared by the supported architecture adapters."""
from __future__ import annotations


def torch_experts(hidden_states, top_k_index, top_k_weights, *, module, fetch_expert, device):
    """Execute upstream HF expert semantics with weights on `device`."""
    import torch
    import torch.nn.functional as F

    original_device = hidden_states.device
    x = hidden_states.to(device=device)
    ids = top_k_index.to(device=device, dtype=torch.long)
    weights = top_k_weights.to(device=device)
    output = torch.zeros_like(x)
    with torch.no_grad():
        mask = F.one_hot(ids, num_classes=int(module.num_experts)).permute(2, 1, 0)
        hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_row in hits:
        expert_id = int(expert_row[0].item())
        route_pos, token_idx = torch.where(mask[expert_id])
        states = x[token_idx]
        gate_up, down = fetch_expert(expert_id, device)
        gate, up = F.linear(states, gate_up).chunk(2, dim=-1)
        value = module.act_fn(gate) * up
        value = F.linear(value, down)
        value = value * weights[token_idx, route_pos, None].to(value.device)
        output.index_add_(0, token_idx, value.to(device=output.device, dtype=output.dtype))
    return output.to(device=original_device)


def native_bf16_experts(
    hidden_states, top_k_index, top_k_weights, *, module, fetch_expert, threads: int,
):
    """Run the approved native BF16 kernel on CPU and preserve HF route accumulation."""
    import torch
    import torch.nn.functional as F
    from qwen_neural.cpu_experts import run_expert

    if hidden_states.dtype is not torch.bfloat16:
        raise TypeError("native expert backend supports only source BF16 hidden states")
    original_device = hidden_states.device
    x = hidden_states.detach().to(device="cpu").contiguous()
    ids = top_k_index.detach().to(device="cpu", dtype=torch.long).contiguous()
    weights = top_k_weights.detach().to(device="cpu").contiguous()
    result = torch.zeros_like(x)
    with torch.no_grad():
        mask = F.one_hot(ids, num_classes=int(module.num_experts)).permute(2, 1, 0)
        hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_row in hits:
        expert_id = int(expert_row[0].item())
        route_pos, token_idx = torch.where(mask[expert_id])
        gate_up, down = fetch_expert(expert_id, torch.device("cpu"))
        states = x[token_idx].contiguous()
        expert_output = run_expert(states, gate_up, down, threads=threads)
        route_weights = weights[token_idx, route_pos, None]
        contribution = (expert_output * route_weights).to(dtype=result.dtype)
        result.index_add_(0, token_idx, contribution)
    return result.to(device=original_device)
