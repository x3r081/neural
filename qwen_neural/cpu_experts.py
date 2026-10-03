"""Original-BF16 CPU expert MLP kernel for Qwen3.6.

The native kernel reads the BF16 payload directly from contiguous CPU tensors,
including read-only mmap-backed weights. It returns the expert MLP result
before router weighting or top-k accumulation. No weight conversion or
quantization is performed here.

Build the library with ``tools/build_qwen_cpu.ps1``. The backend deliberately
does not fall back silently when the DLL is missing or a call fails.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DLL = _ROOT / "qwen_bf16_experts.dll"
_DLL_ENV = "QWEN_BF16_EXPERTS_DLL"
_LIBRARIES: dict[str, ctypes.CDLL] = {}
_F32_PTR = ctypes.c_void_p


class ExpertWorkspace:
    """Reusable native-kernel pointer tables and bounded BF16 scratch.

    Owns only output/intermediate scratch, never expert weights. A workspace
    is single-caller and synchronous; keep it per backend, not process-global.
    Returned tensors from public wrappers are independent copies, so a later
    invocation cannot mutate an earlier result.
    """

    def __init__(self, *, max_scratch_bytes: int = 64 * 1024 * 1024):
        if not isinstance(max_scratch_bytes, int) or max_scratch_bytes < 0:
            raise ValueError("max_scratch_bytes must be a nonnegative integer")
        self.max_scratch_bytes = max_scratch_bytes
        self._tokens = self._routes = self._hidden = self._intermediate = 0
        self._output = self._gate_tmp = self._act_tmp = None
        self._gate_ptrs = self._down_ptrs = None
        self._scratch_bytes = 0
        self._reallocations = 0
        self._closed = False

    def _reserve(self, tokens: int, routes: int, hidden: int, intermediate: int):
        import torch

        if self._closed:
            raise RuntimeError("ExpertWorkspace is closed")
        sizes = (
            max(self._tokens, tokens),
            max(self._routes, routes),
            max(self._hidden, hidden),
            max(self._intermediate, intermediate),
        )
        t_cap, r_cap, h_cap, i_cap = sizes
        needed = 2 * (t_cap * r_cap * h_cap + t_cap * r_cap * 2 * i_cap + t_cap * r_cap * i_cap)
        if needed > self.max_scratch_bytes:
            raise MemoryError(
                f"native expert scratch needs {needed} bytes, limit is {self.max_scratch_bytes}"
            )
        if sizes != (self._tokens, self._routes, self._hidden, self._intermediate):
            self._output = torch.empty((t_cap * r_cap * h_cap,), dtype=torch.bfloat16, device="cpu")
            self._gate_tmp = torch.empty((t_cap * r_cap * 2 * i_cap,), dtype=torch.bfloat16, device="cpu")
            self._act_tmp = torch.empty((t_cap * r_cap * i_cap,), dtype=torch.bfloat16, device="cpu")
            self._tokens, self._routes, self._hidden, self._intermediate = sizes
            self._scratch_bytes = needed
            self._reallocations += 1
        if self._gate_ptrs is None or len(self._gate_ptrs) < routes:
            self._gate_ptrs = (ctypes.c_void_p * r_cap)()
            self._down_ptrs = (ctypes.c_void_p * r_cap)()
        return self._output, self._gate_tmp, self._act_tmp, self._gate_ptrs, self._down_ptrs

    def report(self) -> dict[str, int | bool]:
        return {
            "scratch_bytes": self._scratch_bytes,
            "scratch_limit_bytes": self.max_scratch_bytes,
            "scratch_reallocations": self._reallocations,
            "closed": self._closed,
        }

    def close(self) -> None:
        self._output = self._gate_tmp = self._act_tmp = None
        self._gate_ptrs = self._down_ptrs = None
        self._scratch_bytes = 0
        self._tokens = self._routes = self._hidden = self._intermediate = 0
        self._closed = True


def _dll_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.environ.get(_DLL_ENV)
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_DLL


def available(path: str | os.PathLike[str] | None = None) -> bool:
    """Return whether the configured native library file exists."""
    return _dll_path(path).is_file()


def load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    """Load and bind the native expert DLL, raising if it is unavailable."""
    dll = _dll_path(path)
    key = str(dll)
    if key not in _LIBRARIES:
        if not dll.is_file():
            raise FileNotFoundError(
                f"Qwen BF16 expert DLL not found at {dll}; build it with "
                "tools/build_qwen_cpu.ps1 or set QWEN_BF16_EXPERTS_DLL"
            )
        lib = ctypes.CDLL(key)
        fn = lib.qwen_bf16_experts
        fn.argtypes = [
            _F32_PTR, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p), _F32_PTR, _F32_PTR, _F32_PTR,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        fn.restype = ctypes.c_int
        _LIBRARIES[key] = lib
    return _LIBRARIES[key]


def _require_bf16_cpu(tensor, name: str, ndim: int | None = None) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if tensor.dtype is not torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {tensor.ndim}")


def _expert_matrices(experts, name: str) -> list:
    """Normalize K individual tensors or a stacked [K,N,K] tensor to views."""
    import torch

    if isinstance(experts, torch.Tensor):
        _require_bf16_cpu(experts, name, ndim=3)
        matrices = list(experts.unbind(0))
    elif isinstance(experts, (tuple, list)):
        matrices = list(experts)
    else:
        raise TypeError(f"{name} must be a stacked tensor or a sequence of tensors")
    if not matrices:
        raise ValueError(f"{name} cannot be empty")
    for i, matrix in enumerate(matrices):
        _require_bf16_cpu(matrix, f"{name}[{i}]", ndim=2)
    return matrices


def run_experts(
    x, gate_up_experts, down_experts, *, threads: int = 8,
    workspace: ExpertWorkspace | None = None,
):
    """Compute K selected BF16 expert MLPs over T inputs.

    ``x`` is ``[hidden]`` or ``[tokens, hidden]``. The expert arguments are
    either K contiguous matrices or contiguous stacked tensors:
    ``gate_up[K, 2*intermediate, hidden]`` and
    ``down[K, hidden, intermediate]``. The same K experts are applied to each
    input token. Results have shape ``[tokens, K, hidden]`` and are unweighted;
    callers apply router probabilities and sum routes in their required order.
    ``threads`` bounds this call's OpenMP team.
    """
    import torch

    _require_bf16_cpu(x, "x")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError("x must have shape [hidden] or [tokens, hidden]")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("x dimensions must be non-empty")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    gate_up = _expert_matrices(gate_up_experts, "gate_up_experts")
    down = _expert_matrices(down_experts, "down_experts")
    routes = len(gate_up)
    if len(down) != routes:
        raise ValueError("gate_up_experts and down_experts must have equal route counts")
    hidden = x.shape[1]
    gu_shape = gate_up[0].shape
    dn_shape = down[0].shape
    if gu_shape[1] != hidden or gu_shape[0] <= 0 or gu_shape[0] % 2:
        raise ValueError("gate_up expert shape must be [2*intermediate, hidden]")
    intermediate = gu_shape[0] // 2
    if dn_shape != (hidden, intermediate):
        raise ValueError("down expert shape must be [hidden, intermediate]")
    for i, (gu, dn) in enumerate(zip(gate_up, down)):
        if tuple(gu.shape) != tuple(gu_shape) or tuple(dn.shape) != tuple(dn_shape):
            raise ValueError(f"expert {i} has a shape inconsistent with expert 0")

    lib = load_library()
    # c_void_p arrays point directly into the tensors' uint16 BF16 storage. Keep
    # all views alive through the call so their mmap / storage owners stay valid.
    tokens = x.shape[0]
    if workspace is None:
        gu_ptrs = (ctypes.c_void_p * routes)()
        dn_ptrs = (ctypes.c_void_p * routes)()
        out = torch.empty((tokens, routes, hidden), dtype=torch.bfloat16)
        gu_tmp = torch.empty((tokens, routes, 2 * intermediate), dtype=torch.bfloat16)
        act_tmp = torch.empty((tokens, routes, intermediate), dtype=torch.bfloat16)
    else:
        out, gu_tmp, act_tmp, gu_ptrs, dn_ptrs = workspace._reserve(
            tokens, routes, hidden, intermediate
        )
    for i, matrix in enumerate(gate_up):
        gu_ptrs[i] = matrix.data_ptr()
    for i, matrix in enumerate(down):
        dn_ptrs[i] = matrix.data_ptr()
    try:
        status = lib.qwen_bf16_experts(
            ctypes.c_void_p(x.data_ptr()),
            gu_ptrs,
            dn_ptrs,
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(gu_tmp.data_ptr()),
            ctypes.c_void_p(act_tmp.data_ptr()),
            tokens,
            routes,
            hidden,
            intermediate,
            threads,
        )
    finally:
        # The synchronous native call is finished before references are
        # dropped. Avoid stale raw addresses surviving cache eviction/close.
        if workspace is not None:
            for i in range(routes):
                gu_ptrs[i] = None
                dn_ptrs[i] = None
    if status:
        raise RuntimeError(f"qwen_bf16_experts failed with status {status}")
    if workspace is None:
        return out
    # The C ABI writes a compact shape-specific prefix even when the reusable
    # buffers were reserved for a larger shape. Clone it so future calls do
    # not mutate tensors already returned to the caller.
    return out.view(-1)[: tokens * routes * hidden].view(tokens, routes, hidden).clone()


def run_expert(
    x, gate_up, down, *, threads: int = 8,
    workspace: ExpertWorkspace | None = None,
):
    """Convenience wrapper for one expert; returns ``[tokens, hidden]`` BF16."""
    return run_experts(x, [gate_up], [down], threads=threads, workspace=workspace)[:, 0, :]
