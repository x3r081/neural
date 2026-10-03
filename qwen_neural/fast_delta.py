"""Explicit, reversible binding of FLA gated-delta kernels to Qwen modules.

This module never changes Transformers defaults. Call :func:`bind_fla_gdn`
only when the optional FLA dependency stack has been deliberately installed
and validated on the current platform. It replaces the two callable
attributes on existing ``Qwen3_5MoeGatedDeltaNet`` instances and leaves
convolution, projections, cache/state handling, and tensor dtypes untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_MODULE_CLASS_NAME = "Qwen3_5MoeGatedDeltaNet"


@dataclass
class _SavedBinding:
    module: Any
    chunk: Any
    recurrent: Any


class FlaGDNBinding:
    """Handle for restoring the original per-module HF kernel callables."""

    def __init__(self, saved: list[_SavedBinding], chunk_op: Any, recurrent_op: Any):
        self._saved = saved
        self.chunk_op = chunk_op
        self.recurrent_op = recurrent_op
        self._closed = False

    @property
    def module_count(self) -> int:
        return len(self._saved)

    def close(self) -> None:
        if self._closed:
            return
        for binding in reversed(self._saved):
            binding.module.chunk_gated_delta_rule = binding.chunk
            binding.module.recurrent_gated_delta_rule = binding.recurrent
        self._closed = True

    def __enter__(self) -> "FlaGDNBinding":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _bind_gdn(model: Any, chunk_op: Any, recurrent_op: Any) -> FlaGDNBinding:
    modules = [m for m in model.modules() if m.__class__.__name__ == _MODULE_CLASS_NAME]
    if not modules:
        raise ValueError(f"No {_MODULE_CLASS_NAME} modules found in model")

    saved: list[_SavedBinding] = []
    try:
        for module in modules:
            chunk = getattr(module, "chunk_gated_delta_rule", None)
            recurrent = getattr(module, "recurrent_gated_delta_rule", None)
            if not callable(chunk) or not callable(recurrent):
                raise TypeError("Qwen GDN module does not expose callable HF kernel slots")
            saved.append(_SavedBinding(module, chunk, recurrent))
            # Functions saved on instances are not method-bound; the call
            # signature remains exactly what the HF layer expects.
            module.chunk_gated_delta_rule = chunk_op
            module.recurrent_gated_delta_rule = recurrent_op
    except Exception:
        for binding in reversed(saved):
            binding.module.chunk_gated_delta_rule = binding.chunk
            binding.module.recurrent_gated_delta_rule = binding.recurrent
        raise
    return FlaGDNBinding(saved, chunk_op, recurrent_op)


def bind_torch_gdn(model: Any, *, modeling_module: Any | None = None) -> FlaGDNBinding:
    """Force Transformers' reference Torch GDN callables on every layer.

    Transformers chooses its default functions when the module is imported.
    If optional FLA is installed before that import, the defaults are FLA;
    this explicit binding guarantees ``gdn_backend='torch'`` stays the
    reference path regardless of import timing.
    """
    if modeling_module is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling_module
    chunk_op = getattr(modeling_module, "torch_chunk_gated_delta_rule", None)
    recurrent_op = getattr(modeling_module, "torch_recurrent_gated_delta_rule", None)
    if not callable(chunk_op) or not callable(recurrent_op):
        raise RuntimeError("Installed Transformers does not expose Qwen Torch GDN functions")
    return _bind_gdn(model, chunk_op, recurrent_op)


def bind_fla_gdn(model: Any, *, ops_module: Any | None = None) -> FlaGDNBinding:
    """Bind official FLA GDN ops to all Qwen3.5-MoE GDN modules.

    Import failure is intentionally fatal: callers asked for the FLA path, so
    silently falling back would make the selected backend misleading. FLA's
    signatures accept the same ``q, k, v, g, beta, initial_state,
    output_final_state, use_qk_l2norm_in_kernel`` arguments passed by current
    Transformers Qwen3.5-MoE. ``g`` and ``beta`` are already transformed by HF.

    This does not cast checkpoint weights, recurrent state, or outputs. In
    particular the FP32 recurrent cache remains owned and typed by HF.
    """
    if ops_module is None:
        try:
            from fla.ops import gated_delta_rule as ops_module
        except Exception as exc:
            raise RuntimeError(
                "FLA GDN was explicitly requested, but fla-core kernels are not "
                "importable; install and validate the optional requirements before "
                "selecting gdn_backend='fla'."
            ) from exc
    chunk_op = getattr(ops_module, "chunk_gated_delta_rule", None)
    recurrent_op = getattr(ops_module, "fused_recurrent_gated_delta_rule", None)
    if not callable(chunk_op) or not callable(recurrent_op):
        raise RuntimeError("fla-core lacks the required Qwen GDN kernels")
    return _bind_gdn(model, chunk_op, recurrent_op)
