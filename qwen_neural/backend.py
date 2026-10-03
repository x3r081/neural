"""Minimal exact-BF16 Qwen3.6 MoE CUDA-core runtime with expert staging."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from .store import BF16ExpertStore

GIB = 1024**3


def _source_candidates(parameter_name: str) -> tuple[str, ...]:
    """Checkpoint text weights live under model.language_model in the VLM."""
    if parameter_name.startswith("model."):
        text_key = "model.language_model." + parameter_name.removeprefix("model.")
        return text_key, parameter_name
    return (parameter_name,)


def _set_module_tensor_from_source(module, name: str, value, device) -> None:
    """Materialize one meta parameter/buffer without changing source dtype."""
    from accelerate.utils import set_module_tensor_to_device

    set_module_tensor_to_device(
        module, name, device, value=value, dtype=value.dtype
    )


def _move_model_buffers(model, device) -> None:
    """Move initialized deterministic buffers; refuse uninitialized meta buffers."""
    for name, buffer in list(model.named_buffers()):
        if buffer.is_meta:
            raise RuntimeError(f"Model buffer {name} was left on meta without a checkpoint value")
        if buffer.device != device:
            _set_module_tensor_from_source(model, name, buffer, device)


def _construct_model_with_torch_norm(modeling_module, config, init_empty_weights):
    """Build the HF model while forcing its non-fused Torch gated norm.

    Transformers selects FusedRMSNormGated at module-import time when FLA is
    available. Temporarily suppress that choice during construction so the
    explicit GDN backend controls only GDN kernels, not a second Triton path.
    """
    saved_fused_norm = modeling_module.FusedRMSNormGated
    try:
        modeling_module.FusedRMSNormGated = None
        with init_empty_weights(include_buffers=False):
            return modeling_module.Qwen3_5MoeForCausalLM(config)
    finally:
        modeling_module.FusedRMSNormGated = saved_fused_norm


class QwenBackend:
    """Load text core to CUDA and keep the full routed expert pool mmap-backed.

    No whole-model CPU state dict is created. Expert host tensors are mmap
    views, optionally retained in a bounded per-backend lookup cache. A separate
    bounded LRU holds exact BF16 copies on CUDA. `model` and `tokenizer` are
    exposed for integrations that need full Transformers APIs.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda:0",
        expert_backend: str = "staged",
        gdn_backend: str = "torch",
        gpu_expert_budget_gib: float = 0.0,
        gpu_headroom_gib: float = 2.0,
        expert_compute: Any | None = None,
        native_threads: int = 8,
        native_dispatch: str = "grouped",
        native_gpu_layers: int = 0,
        native_workspace: bool = True,
        recurrent_graph: bool = False,
    ) -> None:
        if expert_backend not in {"staged", "native", "provider"}:
            raise ValueError("expert_backend must be 'staged', 'native', or 'provider'")
        if expert_backend == "provider" and expert_compute is None:
            raise ValueError("provider mode requires expert_compute")
        if gdn_backend not in {"torch", "fla"}:
            raise ValueError("gdn_backend must be 'torch' or 'fla'")
        self.model_dir = Path(model_dir)
        self.device_name = device
        self.expert_backend = expert_backend
        self.gdn_backend = gdn_backend
        self.gpu_expert_budget_bytes = max(0, int(gpu_expert_budget_gib * GIB))
        self.expert_compute = expert_compute
        self.native_threads = max(1, int(native_threads))
        if native_dispatch not in {"legacy", "grouped"}:
            raise ValueError("native_dispatch must be legacy or grouped")
        self.native_dispatch = native_dispatch
        if not isinstance(native_workspace, bool) or not isinstance(recurrent_graph, bool):
            raise TypeError("native_workspace and recurrent_graph must be booleans")
        if recurrent_graph and gdn_backend != "torch":
            raise ValueError("recurrent_graph requires gdn_backend='torch'")
        self.native_workspace_enabled = native_workspace
        self.recurrent_graph_enabled = recurrent_graph
        self._native_workspace = None
        self._recurrent_graph_binding = None
        if not isinstance(native_gpu_layers, int) or isinstance(native_gpu_layers, bool):
            raise TypeError("native_gpu_layers must be an integer")
        if native_gpu_layers < 0:
            raise ValueError("native_gpu_layers cannot be negative")
        self.native_gpu_layers = native_gpu_layers
        self.native_gpu_layer_start = 0
        self.native_gpu_required_bytes = 0
        self.native_gpu_layer_bytes: dict[int, int] = {}
        self._gpu_experts: OrderedDict[tuple[int, int], tuple[Any, Any, int]] = OrderedDict()
        self._gpu_expert_bytes = 0
        self._past_key_values = None
        self._expert_modules = []
        self.forward_calls = 0
        self.input_tokens = 0
        self._gpu_expert_hits = 0
        self._gpu_expert_misses = 0
        self._forward_hook = None
        self._gdn_binding = None
        self.store = BF16ExpertStore(self.model_dir)

        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen_modeling
        Qwen3_5MoeForCausalLM = qwen_modeling.Qwen3_5MoeForCausalLM

        if not torch.cuda.is_available() or not device.startswith("cuda"):
            self.store.close()
            raise RuntimeError("QwenBackend requires an available CUDA device")
        config = AutoConfig.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False
        )
        text_config = Qwen3_5MoeTextConfig.from_dict(config.text_config.to_dict())
        text_config.use_cache = True
        self.config = text_config
        if native_workspace and expert_backend == "native" and native_dispatch == "grouped":
            from .native_dispatch import NativeExpertWorkspace
            self._native_workspace = NativeExpertWorkspace(
                max_expert_cache_entries=int(text_config.num_hidden_layers) * int(text_config.num_experts)
            )
        layer_count = int(text_config.num_hidden_layers)
        if native_gpu_layers > layer_count:
            self.store.close()
            raise ValueError(
                f"native_gpu_layers={native_gpu_layers} exceeds model layer count {layer_count}"
            )
        self.native_gpu_layer_start = layer_count - native_gpu_layers
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False
        )

        # Only parameters are meta; small deterministic buffers are initialized
        # normally. Model weights are subsequently assigned one tensor at a time.
        # Optional fla-core availability is detected at Transformers import
        # time and also selects FusedRMSNormGated in GDN construction. Keep
        # that unrelated fused norm disabled so both backend modes retain the
        # HF Torch norm implementation; only the two GDN operators vary.
        self.model = _construct_model_with_torch_norm(
            qwen_modeling, text_config, init_empty_weights
        )
        self.model.eval()
        from .fast_delta import bind_fla_gdn, bind_torch_gdn
        self._gdn_binding = (
            bind_fla_gdn(self.model)
            if self.gdn_backend == "fla"
            else bind_torch_gdn(self.model, modeling_module=qwen_modeling)
        )

        expected = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if ".mlp.experts." not in name
        ]
        try:
            expected_bytes, source_names = self._estimate_core_bytes(expected)
            if self.native_gpu_layers:
                self.native_gpu_layer_bytes = {
                    layer_index: self._estimate_expert_layer_bytes(layer_index)
                    for layer_index in range(self.native_gpu_layer_start, layer_count)
                }
                self.native_gpu_required_bytes = sum(self.native_gpu_layer_bytes.values())
        except Exception:
            self.store.close()
            raise
        if self.native_gpu_required_bytes > self.gpu_expert_budget_bytes:
            self.store.close()
            raise MemoryError(
                "native last-layer GPU placement exceeds the configured expert cache budget: "
                f"layers={self.native_gpu_layers}, required={self.native_gpu_required_bytes / GIB:.2f} GiB, "
                f"budget={self.gpu_expert_budget_bytes / GIB:.2f} GiB"
            )
        free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
        requested_expert_bytes = self.gpu_expert_budget_bytes
        if expected_bytes + requested_expert_bytes + int(gpu_headroom_gib * GIB) > free_bytes:
            self.store.close()
            raise MemoryError(
                "Qwen text core does not fit the current CUDA free-memory budget: "
                f"core={expected_bytes / GIB:.2f} GiB, staged_experts={requested_expert_bytes / GIB:.2f} GiB, "
                f"free={free_bytes / GIB:.2f} GiB, "
                f"headroom={gpu_headroom_gib:.2f} GiB"
            )
        self.device = torch.device(device)
        try:
            self.core_bytes_loaded = self._load_core_parameters(expected, source_names)
            _move_model_buffers(self.model, self.device)
            self._expert_modules = self._patch_experts()
            if recurrent_graph:
                from .recurrent_graph import bind_recurrent_graph
                self._recurrent_graph_binding = bind_recurrent_graph(self.model)
            self.model.requires_grad_(False)
            self._forward_hook = self.model.register_forward_pre_hook(
                self._record_forward_input, with_kwargs=True
            )
        except Exception:
            self.close()
            torch.cuda.empty_cache()
            raise

        self.model.generation_config.use_cache = True

    def _record_forward_input(self, module, args, kwargs=None):
        input_ids = kwargs.get("input_ids") if kwargs else None
        if input_ids is None and args:
            input_ids = args[0]
        self.forward_calls += 1
        if input_ids is not None and hasattr(input_ids, "numel"):
            self.input_tokens += int(input_ids.numel())

    def _patch_experts(self):
        modules = []
        for layer_index, layer in enumerate(self.model.model.layers):
            experts = layer.mlp.experts
            original = experts.forward

            def make_forward(li: int, module: Any, original_forward: Any):
                def forward(hidden_states, top_k_index, top_k_weights):
                    import torch
                    import torch.nn.functional as F

                    if self.expert_backend == "provider":
                        return self.expert_compute(
                            layer=li,
                            module=module,
                            hidden_states=hidden_states,
                            top_k_index=top_k_index,
                            top_k_weights=top_k_weights,
                            store=self.store,
                        )
                    if self.expert_backend == "native":
                        if li < self.native_gpu_layer_start:
                            if self.native_dispatch == "grouped":
                                from .native_dispatch import dispatch_native_experts
                                dispatch = (
                                    self._native_workspace.dispatch
                                    if self._native_workspace is not None else dispatch_native_experts
                                )
                                return dispatch(
                                    hidden_states, top_k_index, top_k_weights,
                                    layer=li, store=self.store, threads=self.native_threads,
                                )
                            from .cpu_experts import run_expert

                            return self._native_experts_forward(
                                li, module, hidden_states, top_k_index,
                                top_k_weights, run_expert,
                            )
                        # The final N MoE layers stay on CUDA using the same
                        # exact-BF16, budgeted LRU used by staged mode.
                    final = torch.zeros_like(hidden_states)
                    # Match upstream's expert-hit loop and scatter-add order.
                    with torch.no_grad():
                        mask = F.one_hot(
                            top_k_index, num_classes=module.num_experts
                        ).permute(2, 1, 0)
                        hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
                    for expert_tensor in hits:
                        expert_id = int(expert_tensor[0].item())
                        token_pos, token_idx = torch.where(mask[expert_id])
                        state = hidden_states[token_idx]
                        gate_up, down = self._expert_on_device(li, expert_id)
                        gate, up = F.linear(state, gate_up).chunk(2, dim=-1)
                        value = module.act_fn(gate) * up
                        value = F.linear(value, down)
                        value = value * top_k_weights[token_idx, token_pos, None].to(value.device)
                        final.index_add_(
                            0, token_idx, value.to(device=final.device, dtype=final.dtype)
                        )
                    return final

                return forward

            experts.forward = make_forward(layer_index, experts, original)
            modules.append((experts, original))
        return modules

    def _native_experts_forward(
        self, layer: int, module: Any, hidden_states, top_k_index,
        top_k_weights, run_expert,
    ):
        """CPU native BF16 compute, preserving native router selections."""
        import torch
        import torch.nn.functional as F

        output = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(
                top_k_index, num_classes=module.num_experts
            ).permute(2, 1, 0)
            hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_tensor in hits:
            expert_id = int(expert_tensor[0].item())
            route_pos, token_idx = torch.where(mask[expert_id])
            states = hidden_states[token_idx].to("cpu").contiguous()
            gate_up, down = self.store.fetch_expert(layer, expert_id)
            computed = run_expert(
                states, gate_up, down, threads=self.native_threads
            )
            weights = top_k_weights[token_idx, route_pos, None].to(device="cpu")
            output.index_add_(
                0,
                token_idx,
                (computed * weights).to(device=output.device, dtype=output.dtype),
            )
        return output

    def _estimate_core_bytes(self, expected) -> tuple[int, dict[str, str]]:
        """Read tensor headers only to get exact original core residency bytes."""
        dtype_bytes = {
            "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4,
            "I64": 8, "F64": 8,
        }
        total = 0
        sources: dict[str, str] = {}
        for target_name, parameter in expected:
            source_name = next(
                (name for name in _source_candidates(target_name) if name in self.store.weight_map),
                None,
            )
            if source_name is None:
                continue
            shard = self.store.weight_map[source_name]
            metadata = self.store._open(shard).get_slice(source_name)
            dtype_name = metadata.get_dtype().upper()
            if dtype_name not in dtype_bytes:
                raise TypeError(f"Unsupported checkpoint dtype {dtype_name} for {source_name}")
            shape = tuple(metadata.get_shape())
            if shape != tuple(parameter.shape):
                raise ValueError(
                    f"Shape mismatch for {target_name} <- {source_name}: "
                    f"checkpoint {shape} vs model {tuple(parameter.shape)}"
                )
            total += parameter.numel() * dtype_bytes[dtype_name]
            sources[target_name] = source_name
        missing = [name for name, _ in expected if name not in sources]
        if missing:
            raise KeyError("Missing text checkpoint weights: " + ", ".join(missing[:8]))
        return total, sources

    def _estimate_expert_layer_bytes(self, layer: int) -> int:
        """Estimate the full expert pool of a layer from safetensors headers."""
        dtype_bytes = {
            "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4,
            "I64": 8, "F64": 8,
        }
        total = 0
        for projection in ("gate_up_proj", "down_proj"):
            key = self.store._expert_key(layer, projection)
            metadata = self.store._open(self.store.weight_map[key]).get_slice(key)
            dtype_name = metadata.get_dtype().upper()
            if dtype_name not in dtype_bytes:
                raise TypeError(f"Unsupported expert dtype {dtype_name} for {key}")
            numel = 1
            for dimension in metadata.get_shape():
                numel *= int(dimension)
            total += numel * dtype_bytes[dtype_name]
        return total

    def _load_core_parameters(self, expected, source_names) -> int:
        import torch

        bytes_loaded = 0
        for target_name, parameter in expected:
            source_name = source_names[target_name]
            shard = self.store.weight_map[source_name]
            source = self.store._open(shard).get_tensor(source_name)
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Shape mismatch for {target_name} <- {source_name}: "
                    f"checkpoint {tuple(source.shape)} vs model {tuple(parameter.shape)}"
                )
            _set_module_tensor_from_source(
                self.model, target_name, source, self.device
            )
            bytes_loaded += source.numel() * source.element_size()
            del source
        return bytes_loaded

    def _expert_on_device(self, layer: int, expert: int):
        import torch

        key = (int(layer), int(expert))
        cached = self._gpu_experts.get(key)
        if cached is not None:
            self._gpu_expert_hits += 1
            self._gpu_experts.move_to_end(key)
            return cached[0], cached[1]
        self._gpu_expert_misses += 1
        gate_up_cpu, down_cpu = self.store.fetch_expert(*key)
        nbytes = gate_up_cpu.numel() * gate_up_cpu.element_size() + down_cpu.numel() * down_cpu.element_size()
        gate_up = gate_up_cpu.to(self.device, non_blocking=False)
        down = down_cpu.to(self.device, non_blocking=False)
        if nbytes <= self.gpu_expert_budget_bytes:
            while self._gpu_experts and self._gpu_expert_bytes + nbytes > self.gpu_expert_budget_bytes:
                _, (_, _, old_bytes) = self._gpu_experts.popitem(last=False)
                self._gpu_expert_bytes -= old_bytes
            self._gpu_experts[key] = (gate_up, down, nbytes)
            self._gpu_expert_bytes += nbytes
        return gate_up, down

    def forward(self, input_ids, *, past_key_values=None, **kwargs):
        """Forward on CUDA; callers can use native Transformers cache objects."""
        ids = input_ids.to(self.device)
        kwargs.setdefault("use_cache", True)
        return self.model(
            input_ids=ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    def prefill(self, input_ids, **kwargs):
        result = self.forward(input_ids, past_key_values=None, **kwargs)
        self._past_key_values = result.past_key_values
        return result

    def decode(self, next_ids, **kwargs):
        if self._past_key_values is None:
            raise RuntimeError("Call prefill before decode")
        result = self.forward(next_ids, past_key_values=self._past_key_values, **kwargs)
        self._past_key_values = result.past_key_values
        return result

    def reset(self, *, clear_gpu_experts: bool = True) -> None:
        self._past_key_values = None
        self.forward_calls = 0
        self.input_tokens = 0
        self._gpu_expert_hits = 0
        self._gpu_expert_misses = 0
        self.store.reset_counters()
        if clear_gpu_experts:
            self._gpu_experts.clear()
            self._gpu_expert_bytes = 0

    def memory_report(self) -> dict[str, Any]:
        import torch

        free = total = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(self.device)
        return {
            "device": str(self.device),
            "core_bytes_loaded": self.core_bytes_loaded,
            "gpu_expert_cache_bytes": self._gpu_expert_bytes,
            "gpu_expert_cache_entries": len(self._gpu_experts),
            "gpu_expert_cache_hits": self._gpu_expert_hits,
            "gpu_expert_cache_misses": self._gpu_expert_misses,
            "native_gpu_layers": self.native_gpu_layers,
            "native_workspace": self._native_workspace.report() if self._native_workspace is not None else None,
            "recurrent_graph": self._recurrent_graph_binding.report() if self._recurrent_graph_binding is not None else None,
            "native_gpu_layer_start": self.native_gpu_layer_start,
            "native_gpu_layer_bytes": {str(k): v for k, v in self.native_gpu_layer_bytes.items()},
            "native_gpu_required_bytes": self.native_gpu_required_bytes,
            "native_gpu_budget_sufficient": self.native_gpu_required_bytes <= self.gpu_expert_budget_bytes,
            "forward_calls": self.forward_calls,
            "input_tokens": self.input_tokens,
            "gpu_free_bytes": free,
            "gpu_total_bytes": total,
            "store": self.store.report(),
        }

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.reset(clear_gpu_experts=True)
        if getattr(self, "_forward_hook", None) is not None:
            self._forward_hook.remove()
            self._forward_hook = None
        for module, original in self._expert_modules:
            module.forward = original
        self._expert_modules.clear()
        if self._recurrent_graph_binding is not None:
            self._recurrent_graph_binding.close()
            self._recurrent_graph_binding = None
        if self._gdn_binding is not None:
            self._gdn_binding.close()
            self._gdn_binding = None
        if self._native_workspace is not None:
            self._native_workspace.close()
            self._native_workspace = None
        self.store.close()
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "QwenBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
