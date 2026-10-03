"""Experimental CUDA-graph wrapper for the cached one-token Torch GDN op.

This deliberately captures only Transformers' existing recurrent operator.
Projection, convolution-cache updates, layer/cache mutation, expert dispatch,
sampling and tokenizer work stay on their original paths.  The graph receives
private staging buffers and returns owned copies, so it never captures a
request's live recurrent-state address.
"""

from __future__ import annotations

from collections import Counter
import threading
from typing import Any


_MODULE_CLASS_NAME = "Qwen3_5MoeGatedDeltaNet"
_INPUT_NAMES = ("query", "key", "value", "g", "beta", "initial_state")


def _nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _device_type(tensor: Any) -> str | None:
    return getattr(getattr(tensor, "device", None), "type", None)


class _SharedRecurrentGraph:
    """One serialized static-buffer graph shared by same-signature GDN layers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._graph = None
        self._stream = None
        self._static_inputs = None
        self._static_output = None
        self._static_state = None
        self._signature = None
        self._torch = None
        self._device = None
        self._capture_disabled = False
        self._capture_error = None
        self._counter = Counter()
        self._fallback_reasons = Counter()
        self._layers_seen: set[int] = set()
        self._static_tensor_bytes = 0
        self._capture_pool_reserved_delta_bytes = None

    @staticmethod
    def _metadata(original, query, key, value, g, beta, initial_state,
                  output_final_state, use_qk_l2norm_in_kernel, kwargs):
        tensors = (query, key, value, g, beta, initial_state)
        return (
            id(original),
            tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
                  for t in tensors),
            bool(output_final_state), bool(use_qk_l2norm_in_kernel),
            tuple(sorted(kwargs)),
        )

    def _count_fallback(self, reason: str, layer_idx: int | None = None) -> None:
        with self._lock:
            self._counter["fallbacks"] += 1
            self._fallback_reasons[reason] += 1
            if layer_idx is not None:
                self._layers_seen.add(int(layer_idx))

    @staticmethod
    def _call(original, query, key, value, g, beta, initial_state,
              output_final_state, use_qk_l2norm_in_kernel, kwargs):
        return original(
            query, key, value, g, beta, initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )

    def _eligible_reason(self, query, key, value, g, beta, initial_state,
                         output_final_state, use_qk_l2norm_in_kernel, kwargs):
        tensors = (query, key, value, g, beta, initial_state)
        if kwargs:
            return "extra_operator_kwargs"
        if any(_device_type(t) != "cuda" for t in tensors):
            return "non_cuda_tensor"
        if initial_state is None:
            return "missing_recurrent_state"
        if not output_final_state:
            return "final_state_not_requested"
        if not isinstance(use_qk_l2norm_in_kernel, bool):
            return "non_boolean_qk_norm_flag"
        if query.ndim != 4 or query.shape[1] != 1:
            return "not_single_token_decode"
        if any(t.device != query.device for t in tensors):
            return "mixed_devices"
        if initial_state.dtype != self._torch.float32:
            return "recurrent_state_not_fp32"
        return None

    def _allocate_staging(self, tensors):
        torch = self._torch
        return tuple(
            torch.empty_strided(
                tuple(t.shape), tuple(t.stride()), dtype=t.dtype, device=t.device
            )
            for t in tensors
        )

    @staticmethod
    def _copy_inputs(staging, inputs):
        for dst, src in zip(staging, inputs):
            dst.copy_(src)

    def _wait_for_inputs(self, current_stream):
        """Order the shared graph stream after inputs made on the caller stream."""
        torch = self._torch
        ready = torch.cuda.Event()
        ready.record(current_stream)
        self._stream.wait_event(ready)

    def _capture(self, original, inputs, output_final_state,
                 use_qk_l2norm_in_kernel, signature):
        torch = self._torch
        device = inputs[0].device
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._static_inputs = self._allocate_staging(inputs)
        self._signature = signature
        self._wait_for_inputs(torch.cuda.current_stream(device))

        # Warm up only private inputs. Reset every staging value before each
        # pass, including recurrent state, in case a future HF op mutates it.
        with torch.cuda.stream(self._stream):
            for _ in range(3):
                self._copy_inputs(self._static_inputs, inputs)
                self._call(original, *self._static_inputs, output_final_state,
                           use_qk_l2norm_in_kernel, {})
        self._stream.synchronize()

        reserved_before = int(torch.cuda.memory_reserved(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._stream):
            self._copy_inputs(self._static_inputs, inputs)
            try:
                with torch.cuda.graph(graph, stream=self._stream):
                    self._static_output, self._static_state = self._call(
                        original, *self._static_inputs, output_final_state,
                        use_qk_l2norm_in_kernel, {},
                    )
            except Exception:
                self._graph = None
                self._static_output = None
                self._static_state = None
                # An aborted CUDA capture can leave work queued on its private
                # stream even though no graph was installed. Drain it before
                # the eager fallback or eventual teardown.
                try:
                    self._stream.synchronize()
                except Exception:
                    pass
                raise
            self._graph = graph
            # Capture records kernels; replay executes them. The capture-time
            # staging state is isolated from the caller's cache.
            graph.replay()
        reserved_after = int(torch.cuda.memory_reserved(device))
        self._capture_pool_reserved_delta_bytes = max(0, reserved_after - reserved_before)
        self._static_tensor_bytes = sum(_nbytes(t) for t in self._static_inputs)
        self._static_tensor_bytes += _nbytes(self._static_output) + _nbytes(self._static_state)
        self._counter["captures"] += 1

    def _replay_and_clone(self, current_stream, inputs):
        torch = self._torch
        self._wait_for_inputs(current_stream)
        with torch.cuda.stream(self._stream):
            # Copy this invocation's values, not the first call's capture-time
            # sample. The event above makes writes on the caller stream visible
            # before the shared graph stream reads these sources.
            for dst, src in zip(self._static_inputs, inputs):
                dst.copy_(src)
                src.record_stream(self._stream)
            self._graph.replay()
            # The shared graph overwrites its static outputs on every replay.
            # Clone on the shared stream before releasing the host lock; the
            # next caller's work is ordered after these copies on this stream.
            output = self._static_output.clone()
            state = self._static_state.clone()
            finished = torch.cuda.Event()
            finished.record(self._stream)
        current_stream.wait_event(finished)
        # Clones are allocated on the graph stream and consumed on the caller
        # stream; tell the allocator that their lifetime spans that use.
        output.record_stream(current_stream)
        state.record_stream(current_stream)
        self._counter["replays"] += 1
        return output, state

    def run(self, layer_idx, original, query, key, value, g, beta,
            initial_state, output_final_state,
            use_qk_l2norm_in_kernel=False, **kwargs):
        tensors = (query, key, value, g, beta, initial_state)
        if any(_device_type(t) != "cuda" for t in tensors if t is not None):
            self._count_fallback("non_cuda_tensor", layer_idx)
            return self._call(original, query, key, value, g, beta,
                              initial_state, output_final_state,
                              use_qk_l2norm_in_kernel, kwargs)

        try:
            import torch
        except Exception:
            self._count_fallback("torch_unavailable", layer_idx)
            return self._call(original, query, key, value, g, beta,
                              initial_state, output_final_state,
                              use_qk_l2norm_in_kernel, kwargs)
        self._torch = torch
        reason = self._eligible_reason(
            query, key, value, g, beta, initial_state, output_final_state,
            use_qk_l2norm_in_kernel, kwargs,
        )
        if reason:
            self._count_fallback(reason, layer_idx)
            return self._call(original, query, key, value, g, beta,
                              initial_state, output_final_state,
                              use_qk_l2norm_in_kernel, kwargs)
        if not torch.cuda.is_available():
            self._count_fallback("cuda_unavailable", layer_idx)
            return self._call(original, query, key, value, g, beta,
                              initial_state, output_final_state,
                              use_qk_l2norm_in_kernel, kwargs)

        signature = self._metadata(
            original, query, key, value, g, beta, initial_state,
            output_final_state, use_qk_l2norm_in_kernel, kwargs,
        )
        with self._lock:
            self._layers_seen.add(int(layer_idx))
            if self._capture_disabled:
                self._counter["fallbacks"] += 1
                self._fallback_reasons["capture_disabled"] += 1
                fallback = True
            elif self._graph is not None and signature != self._signature:
                self._counter["fallbacks"] += 1
                self._fallback_reasons["signature_mismatch"] += 1
                fallback = True
            else:
                fallback = False
                current_stream = torch.cuda.current_stream(query.device)
                if self._graph is None:
                    try:
                        self._capture(
                            original, tensors, output_final_state,
                            use_qk_l2norm_in_kernel, signature,
                        )
                    except Exception as exc:
                        self._capture_disabled = True
                        self._capture_error = f"{type(exc).__name__}: {exc}"
                        self._counter["fallbacks"] += 1
                        self._fallback_reasons["capture_failed"] += 1
                        fallback = True
                if not fallback:
                    output, state = self._replay_and_clone(current_stream, tensors)
        if fallback:
            return self._call(original, query, key, value, g, beta,
                              initial_state, output_final_state,
                              use_qk_l2norm_in_kernel, kwargs)
        return output, state

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "captured" if self._graph is not None else (
                    "capture_disabled" if self._capture_disabled else "not_captured"
                ),
                "capture_count": int(self._counter["captures"]),
                "replay_count": int(self._counter["replays"]),
                "fallback_count": int(self._counter["fallbacks"]),
                "fallback_reasons": dict(self._fallback_reasons),
                "layers_seen": sorted(self._layers_seen),
                "signature": self._signature,
                "static_tensor_bytes": self._static_tensor_bytes,
                "capture_pool_reserved_delta_bytes": self._capture_pool_reserved_delta_bytes,
                "capture_error": self._capture_error,
            }

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.synchronize()
            self._graph = None
            self._static_inputs = None
            self._static_output = None
            self._static_state = None
            self._stream = None


class RecurrentGraphBinding:
    """Reversible per-model binding for the shared recurrent-op graph."""

    def __init__(self, bindings, runner: _SharedRecurrentGraph):
        self._bindings = bindings
        self._runner = runner
        self._closed = False

    @property
    def module_count(self) -> int:
        return len(self._bindings)

    def report(self) -> dict[str, Any]:
        result = self._runner.report()
        result["module_count"] = self.module_count
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._runner.close()
        for module, original, wrapper in reversed(self._bindings):
            if getattr(module, "recurrent_gated_delta_rule", None) is wrapper:
                module.recurrent_gated_delta_rule = original
        self._closed = True

    def __enter__(self) -> "RecurrentGraphBinding":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def bind_recurrent_graph(model: Any) -> RecurrentGraphBinding:
    """Bind the captured one-token recurrence after ``bind_torch_gdn``.

    Unsupported calls and signatures use the module's original Torch function
    and are counted. This does not capture FLA kernels or mutate cache tensors.
    """
    modules = [m for m in model.modules() if m.__class__.__name__ == _MODULE_CLASS_NAME]
    if not modules:
        raise ValueError(f"No {_MODULE_CLASS_NAME} modules found in model")

    runner = _SharedRecurrentGraph()
    bindings = []
    try:
        for module in modules:
            original = getattr(module, "recurrent_gated_delta_rule", None)
            if not callable(original):
                raise TypeError("Qwen GDN module has no callable recurrent Torch op")

            def make_wrapper(layer_idx, eager):
                def wrapper(query, key, value, g, beta, initial_state,
                            output_final_state, use_qk_l2norm_in_kernel=False,
                            **kwargs):
                    return runner.run(
                        layer_idx, eager, query, key, value, g, beta,
                        initial_state, output_final_state,
                        use_qk_l2norm_in_kernel, **kwargs,
                    )
                return wrapper

            wrapper = make_wrapper(int(getattr(module, "layer_idx", len(bindings))), original)
            bindings.append((module, original, wrapper))
            module.recurrent_gated_delta_rule = wrapper
    except Exception:
        for module, original, wrapper in reversed(bindings):
            if getattr(module, "recurrent_gated_delta_rule", None) is wrapper:
                module.recurrent_gated_delta_rule = original
        runner.close()
        raise
    return RecurrentGraphBinding(bindings, runner)
