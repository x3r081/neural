"""Model-agnostic Transformers-backed MoE runtime with exact source weights."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from .adapters import adapter_for_config, inspect_checkpoint
from .dispatch import torch_experts
from .store import ExpertStore

GIB = 1024**3
_SUPPORTED_DTYPES = {"torch.bfloat16", "torch.float16", "torch.float32"}


def _source_candidates(name: str) -> tuple[str, ...]:
    candidates = [name]
    if name.startswith("model."):
        candidates.insert(0, "model.language_model." + name.removeprefix("model."))
    # A few HF text-only checkpoints prefix every decoder key with language_model.
    if name.startswith("model."):
        candidates.append("language_model." + name.removeprefix("model."))
    return tuple(dict.fromkeys(candidates))


def _set_tensor(model, name: str, value, device) -> None:
    from accelerate.utils import set_module_tensor_to_device

    set_module_tensor_to_device(model, name, device, value=value, dtype=value.dtype)


def _construct_model(adapter, config, init_empty_weights):
    module = adapter.modeling_module
    model_class = adapter.model_class()
    if adapter.has_qwen_gdn:
        # Match the proven Qwen runtime: optional FLA import must not select its
        # fused gated norm when this backend promises the Torch GDN path.
        saved_norm = getattr(module, "FusedRMSNormGated", None)
        try:
            if hasattr(module, "FusedRMSNormGated"):
                module.FusedRMSNormGated = None
            with init_empty_weights(include_buffers=False):
                return model_class(config)
        finally:
            if hasattr(module, "FusedRMSNormGated"):
                module.FusedRMSNormGated = saved_norm
    with init_empty_weights(include_buffers=False):
        return model_class(config)


class NeuralBackend:
    """Run supported Hugging Face MoE models with the original expert tensors.

    Attention, positional encoding, cache objects, router, and tokenizer stay
    in upstream Transformers. Only each sparse expert module's `forward` is
    replaced. The constructor is local-only and never downloads model files.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda:0",
        expert_backend: str = "staged",
        gpu_expert_budget_gib: float = 0.0,
        native_gpu_layers: int = 0,
        native_threads: int = 8,
        recurrent_graph: bool = False,
        host_expert_cache_gib: float = 0.0,
        load_tokenizer: bool = True,
        packed_projection_cache: bool = True,
    ) -> None:
        if expert_backend not in {"torch-cpu", "native", "staged"}:
            raise ValueError("expert_backend must be 'torch-cpu', 'native', or 'staged'")
        if not isinstance(native_gpu_layers, int) or isinstance(native_gpu_layers, bool) or native_gpu_layers < 0:
            raise ValueError("native_gpu_layers must be a nonnegative integer")
        if not isinstance(native_threads, int) or native_threads <= 0:
            raise ValueError("native_threads must be a positive integer")
        if not isinstance(gpu_expert_budget_gib, (int, float)) or gpu_expert_budget_gib < 0:
            raise ValueError("gpu_expert_budget_gib must be nonnegative")
        if not isinstance(host_expert_cache_gib, (int, float)) or host_expert_cache_gib < 0:
            raise ValueError("host_expert_cache_gib must be nonnegative")
        self.model_dir = Path(model_dir).resolve()
        self.device_name = str(device)
        self.expert_backend = expert_backend
        self.gpu_expert_budget_bytes = int(gpu_expert_budget_gib * GIB)
        self.native_threads = native_threads
        self.native_gpu_layers = native_gpu_layers
        self.host_expert_cache_bytes = int(host_expert_cache_gib * GIB)
        self.recurrent_graph_enabled = bool(recurrent_graph)
        self.native_workspace = None
        self._native_workspace = None
        self._gpu_experts: OrderedDict[tuple[int, int], tuple[Any, Any, int]] = OrderedDict()
        self._gpu_expert_bytes = 0
        self._gpu_fixed_bytes = 0
        self._fixed_gpu_keys: set[tuple[int, int]] = set()
        self._fixed_gpu_layer_ids: list[int] = []
        self._expert_modules: list[tuple[Any, Any]] = []
        self._gdn_binding = None
        self._recurrent_graph_binding = None
        self.forward_calls = 0
        self.input_tokens = 0
        self._gpu_expert_hits = 0
        self._gpu_expert_misses = 0
        self._closed = False
        self.store = ExpertStore(
            self.model_dir,
            host_cache_bytes=self.host_expert_cache_bytes,
            packed_projection_cache=packed_projection_cache,
        )

        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoTokenizer

        self.device = torch.device(self.device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.store.close()
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            self.store.close()
            raise ValueError("device must be a CPU or CUDA device")
        self.raw_profile = inspect_checkpoint(self.model_dir)
        try:
            self.config_root = AutoConfig.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=False
            )
            self.adapter = adapter_for_config(self.config_root)
            self.model_type = self.adapter.model_type
            source_text_config = self.adapter.text_config(self.config_root)
            self.config = type(source_text_config).from_dict(source_text_config.to_dict())
            if recurrent_graph and not self.adapter.has_qwen_gdn:
                raise ValueError("recurrent_graph is supported only for Qwen3.5-MoE text models")
            if recurrent_graph and self.device.type != "cuda":
                raise ValueError("recurrent_graph requires a CUDA device")
            if self.device.type == "cpu" and native_gpu_layers:
                raise ValueError("native_gpu_layers requires a CUDA device")
            if self.device.type == "cpu" and expert_backend == "staged":
                raise ValueError("staged expert backend requires a CUDA device")
            if native_gpu_layers > len(self.raw_profile["sparse_layers"]):
                raise ValueError(
                    f"native_gpu_layers={native_gpu_layers} exceeds sparse layer count "
                    f"{len(self.raw_profile['sparse_layers'])}"
                )
            if expert_backend == "native":
                self._validate_native_host()
                from qwen_neural.native_dispatch import NativeExpertWorkspace
                # Expert tensor retention is bounded by ExpertStore's byte
                # limit; the dispatch workspace owns scratch only.
                self._native_workspace = NativeExpertWorkspace(max_expert_cache_entries=0)

            self.model = _construct_model(self.adapter, self.config, init_empty_weights)
            self.model.eval()
            if self.adapter.has_qwen_gdn:
                from qwen_neural.fast_delta import bind_torch_gdn
                self._gdn_binding = bind_torch_gdn(
                    self.model, modeling_module=self.adapter.modeling_module
                )

            if load_tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_dir), local_files_only=True, trust_remote_code=False
                )
            else:
                self.tokenizer = None

            self._sparse_modules = self.adapter.sparse_modules(self.model)
            for layer_index, experts_module, _ in self._sparse_modules:
                if expert_backend == "native" and any("bias" in name for name, _ in experts_module.named_parameters()):
                    raise ValueError(f"Native BF16 backend does not support expert biases (layer {layer_index})")
            if set(i for i, _, _ in self._sparse_modules) != set(self.raw_profile["sparse_layers"]):
                raise ValueError(
                    "Checkpoint sparse layer indices do not match the Transformers architecture: "
                    f"checkpoint={self.raw_profile['sparse_layers']}, "
                    f"model={[i for i, _, _ in self._sparse_modules]}"
                )
            intermediate_size = getattr(self.config, "moe_intermediate_size", None)
            if intermediate_size is None:
                intermediate_size = getattr(self.config, "intermediate_size", None)
            self.expert_dtype = self.store.validate_layout(
                hidden_size=int(self.config.hidden_size),
                intermediate_size=int(intermediate_size),
                expert_count=int(self.raw_profile["expert_count"]),
            )
            self.native_gpu_layer_start = len(self._sparse_modules) - native_gpu_layers
            selected_fixed_layers = self._sparse_modules[self.native_gpu_layer_start:] if native_gpu_layers else []
            self._fixed_gpu_layer_ids = [int(i) for i, _, _ in selected_fixed_layers]
            self.native_gpu_required_bytes = sum(
                int(self.raw_profile["layer_expert_bytes"].get(str(i), 0))
                for i in self._fixed_gpu_layer_ids
            )
            if self.native_gpu_required_bytes > self.gpu_expert_budget_bytes:
                raise MemoryError(
                    "Complete GPU expert layers exceed configured budget: "
                    f"required={self.native_gpu_required_bytes / GIB:.3f} GiB, "
                    f"budget={self.gpu_expert_budget_bytes / GIB:.3f} GiB"
                )

            expected = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if not self.adapter.is_expert_parameter(name)
            ]
            source_names: dict[str, str] = {}
            missing: list[str] = []
            tied_lm_head = bool(getattr(self.config, "tie_word_embeddings", False))
            for name, _parameter in expected:
                found = [candidate for candidate in _source_candidates(name)
                         if candidate in self.store.weight_map]
                if len(found) > 1:
                    raise ValueError(f"Ambiguous source aliases for {name}: {found}")
                source = found[0] if found else None
                if source is None and tied_lm_head and name.endswith("lm_head.weight"):
                    continue
                if source is None:
                    missing.append(name)
                else:
                    source_names[name] = source
            if missing:
                raise KeyError("Missing non-expert checkpoint parameters: " + ", ".join(missing[:12]))

            self.core_bytes_loaded = self._load_core_parameters(expected, source_names)
            self._move_model_buffers(self.model, self.device)
            if tied_lm_head:
                self.model.tie_weights()
            self._validate_activation_dtypes()
            unmaterialized = [
                name for name, parameter in self.model.named_parameters()
                if not self.adapter.is_expert_parameter(name) and parameter.is_meta
            ]
            if unmaterialized:
                raise RuntimeError("Unmaterialized non-expert parameters: " + ", ".join(unmaterialized[:8]))
            self.model.requires_grad_(False)
            self._patch_experts()
            if self.device.type == "cuda" and self._fixed_gpu_layer_ids:
                self._load_fixed_gpu_layers()
            if recurrent_graph:
                from qwen_neural.recurrent_graph import bind_recurrent_graph
                self._recurrent_graph_binding = bind_recurrent_graph(self.model)

            from transformers import GenerationConfig
            generation_path = self.model_dir / "generation_config.json"
            if generation_path.is_file():
                self.model.generation_config = GenerationConfig.from_pretrained(
                    str(self.model_dir), local_files_only=True
                )
            self.generation_config = self.model.generation_config
        except Exception:
            self.close()
            raise

    def _validate_native_host(self) -> None:
        from pathlib import Path
        from qwen_neural import cpu_experts
        from .hardware import native_kernel_status

        status = native_kernel_status()
        if not status.get("available"):
            raise RuntimeError(f"Verified native expert kernel unavailable: {status.get('reason')}")
        if sorted(self.raw_profile.get("expert_dtypes", [])) != ["bf16"]:
            raise TypeError(
                "Native expert backend requires every source expert tensor to be BF16; "
                f"found {self.raw_profile.get('expert_dtypes')}"
            )
        if getattr(self.config, "hidden_act", None) != "silu":
            raise ValueError("Native expert backend supports only the exact gated SiLU expert activation")
        selected = cpu_experts._dll_path().resolve()
        verified = Path(status["library"]).resolve()
        if selected != verified:
            raise RuntimeError(
                f"Configured native library {selected} does not match verified build {verified}; "
                "set QWEN_BF16_EXPERTS_DLL to the manifest-approved library"
            )

    @staticmethod
    def _move_model_buffers(model, device) -> None:
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta:
                raise RuntimeError(f"Model buffer {name} was left on meta")
            if buffer.device != device:
                module_name, _, leaf = name.rpartition(".")
                module = model.get_submodule(module_name) if module_name else model
                module._buffers[leaf] = buffer.to(device=device)

    def _validate_activation_dtypes(self) -> None:
        import torch

        embed = self.model.get_input_embeddings()
        activation_dtype = embed.weight.dtype
        if str(activation_dtype) != self.expert_dtype:
            raise TypeError(
                f"Expert dtype {self.expert_dtype} does not match embedding/activation dtype {activation_dtype}; "
                "mixed-precision conversion is disabled"
            )
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                if module.weight.dtype != activation_dtype:
                    raise TypeError(
                        f"Core activation dtype {activation_dtype} is incompatible with "
                        f"{name}.weight source dtype {module.weight.dtype}"
                    )
                if module.bias is not None and module.bias.dtype != activation_dtype:
                    raise TypeError(
                        f"Core activation dtype {activation_dtype} is incompatible with "
                        f"{name}.bias source dtype {module.bias.dtype}"
                    )

    def _load_core_parameters(self, expected, source_names) -> int:
        bytes_loaded = 0
        for target_name, parameter in expected:
            source_name = source_names.get(target_name)
            if source_name is None:  # tied lm_head; tie_weights reconnects it below.
                continue
            source = self.store.fetch_parameter(source_name)
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Shape mismatch for {target_name} <- {source_name}: "
                    f"checkpoint {tuple(source.shape)} vs model {tuple(parameter.shape)}"
                )
            if str(source.dtype) not in _SUPPORTED_DTYPES:
                raise TypeError(f"Unsupported source dtype {source.dtype} for {source_name}")
            _set_tensor(self.model, target_name, source, self.device)
            bytes_loaded += source.numel() * source.element_size()
            del source
        return bytes_loaded

    def _patch_experts(self) -> None:
        for layer_index, module, _block in self._sparse_modules:
            original = module.forward

            def make_forward(li: int, experts_module, original_forward):
                def forward(hidden_states, top_k_index, top_k_weights):
                    import torch

                    if self._closed:
                        raise RuntimeError("NeuralBackend is closed")
                    if self.expert_backend == "native" and li not in self._fixed_gpu_layer_ids:
                        return self._native_workspace.dispatch(
                            hidden_states,
                            top_k_index,
                            top_k_weights,
                            layer=li,
                            store=self.store,
                            threads=self.native_threads,
                        )
                    if li in self._fixed_gpu_layer_ids:
                        target = self.device
                    else:
                        target = torch.device("cpu") if self.expert_backend == "torch-cpu" else self.device
                    return torch_experts(
                        hidden_states, top_k_index, top_k_weights,
                        module=experts_module,
                        fetch_expert=lambda expert, dev: self._expert_on_device(li, expert, dev),
                        device=target,
                    )
                return forward

            module.forward = make_forward(layer_index, module, original)
            self._expert_modules.append((module, original))

    def _expert_on_device(self, layer: int, expert: int, device):
        import torch

        key = (int(layer), int(expert))
        target = torch.device(device)
        if target.type == "cuda":
            cached = self._gpu_experts.get(key)
            if cached is not None:
                self._gpu_expert_hits += 1
                self._gpu_experts.move_to_end(key)
                return cached[0], cached[1]
            self._gpu_expert_misses += 1
        gate_up_cpu, down_cpu = self.store.fetch_expert(*key)
        nbytes = (gate_up_cpu.numel() + down_cpu.numel()) * gate_up_cpu.element_size()
        if target.type == "cuda" and key not in self._fixed_gpu_keys and nbytes <= self.gpu_expert_budget_bytes:
            while self._gpu_expert_bytes + nbytes > self.gpu_expert_budget_bytes:
                old_key = next((candidate for candidate in self._gpu_experts
                                if candidate not in self._fixed_gpu_keys), None)
                if old_key is None:
                    break
                _, _, old_bytes = self._gpu_experts.pop(old_key)
                self._gpu_expert_bytes -= old_bytes
        # Evict before the H2D copies so the previous entries are actually
        # released before allocating the replacement expert.
        gate_up = gate_up_cpu.to(device=target) if target.type != "cpu" else gate_up_cpu
        down = down_cpu.to(device=target) if target.type != "cpu" else down_cpu
        if (target.type == "cuda" and key not in self._fixed_gpu_keys
                and self._gpu_expert_bytes + nbytes <= self.gpu_expert_budget_bytes):
            self._gpu_experts[key] = (gate_up, down, nbytes)
            self._gpu_expert_bytes += nbytes
        return gate_up, down

    def _load_fixed_gpu_layers(self) -> None:
        layer_to_module = {int(i): module for i, module, _ in self._sparse_modules}
        for layer in self._fixed_gpu_layer_ids:
            module = layer_to_module[layer]
            for expert in range(int(module.num_experts)):
                gate_up, down = self.store.fetch_expert(layer, expert)
                gate_up = gate_up.to(self.device)
                down = down.to(self.device)
                size = (gate_up.numel() + down.numel()) * gate_up.element_size()
                key = (layer, expert)
                self._gpu_experts[key] = (gate_up, down, size)
                self._fixed_gpu_keys.add(key)
                self._gpu_expert_bytes += size
                self._gpu_fixed_bytes += size

    def forward(self, input_ids, *, past_key_values=None, **kwargs):
        import torch

        if self._closed:
            raise RuntimeError("NeuralBackend is closed")
        ids = input_ids.to(device=self.device)
        kwargs.setdefault("use_cache", True)
        self.forward_calls += 1
        self.input_tokens += int(ids.numel())
        return self.model(input_ids=ids, past_key_values=past_key_values, **kwargs)

    def prefill(self, input_ids, **kwargs):
        result = self.forward(input_ids, past_key_values=None, **kwargs)
        self._past_key_values = result.past_key_values
        return result

    def decode(self, next_ids, **kwargs):
        past = getattr(self, "_past_key_values", None)
        if past is None:
            raise RuntimeError("Call prefill before decode")
        result = self.forward(next_ids, past_key_values=past, **kwargs)
        self._past_key_values = result.past_key_values
        return result

    def memory_report(self) -> dict[str, Any]:
        import torch

        free = total = None
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(self.device)
        return {
            "model_type": getattr(self.adapter, "model_type", None),
            "device": str(self.device) if self.device is not None else None,
            "expert_backend": self.expert_backend,
            "core_bytes_loaded": getattr(self, "core_bytes_loaded", 0),
            "gpu_expert_cache_bytes": self._gpu_expert_bytes,
            "gpu_fixed_expert_bytes": self._gpu_fixed_bytes,
            "gpu_expert_cache_limit_bytes": self.gpu_expert_budget_bytes,
            "gpu_expert_cache_entries": len(self._gpu_experts),
            "gpu_expert_cache_hits": self._gpu_expert_hits,
            "gpu_expert_cache_misses": self._gpu_expert_misses,
            "native_gpu_layers": self.native_gpu_layers,
            "native_gpu_layer_ids": list(self._fixed_gpu_layer_ids),
            "native_gpu_required_bytes": getattr(self, "native_gpu_required_bytes", 0),
            "forward_calls": self.forward_calls,
            "input_tokens": self.input_tokens,
            "gpu_free_bytes": free,
            "gpu_total_bytes": total,
            "store": self.store.report() if self.store is not None else None,
        }

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for module, original in reversed(self._expert_modules):
            module.forward = original
        self._expert_modules.clear()
        if self._recurrent_graph_binding is not None:
            self._recurrent_graph_binding.close()
            self._recurrent_graph_binding = None
        if self._gdn_binding is not None:
            self._gdn_binding.close()
            self._gdn_binding = None
        self._gpu_experts.clear()
        self._fixed_gpu_keys.clear()
        self._gpu_expert_bytes = 0
        self._gpu_fixed_bytes = 0
        if self._native_workspace is not None:
            self._native_workspace.close()
            self._native_workspace = None
        if getattr(self, "store", None) is not None:
            self.store.close()
        self._sparse_modules = []
        self._past_key_values = None
        self.model = None
        self.tokenizer = None
        self.config = None
        self.config_root = None
        self.model_type = getattr(getattr(self, "adapter", None), "model_type", None)
        self.device = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
