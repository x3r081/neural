"""Thin ctypes wrapper for gptoss_cpu_multi.dll (multi-token MXFP4 CPU experts).

Prompt processing routes one expert several tokens at once. ``gptoss_experts_multi``
reads each expert's 13.2 MB of weights once for all of its tokens, and for every
(expert, token) pair returns a result bit-identical to the single-token kernel
``gptoss_experts(1, ...)`` of gptoss_cpu_cap2.dll.

    import cpu_prefill as cp
    g = cp.group_pairs(topi, cpu_mask=mask)           # topi [N, K] expert ids of ONE layer
    Xp = X[g["pair_token"]]                           # [P, 2880] fp32, grouped by expert
    Wp = topw[g["pair_k"][:, 0], g["pair_k"][:, 1]]   # [P] routing weight of each pair
    slots = [ROWP[L*128 + e] for e in g["experts"]]   # row pointers of the used experts
    Y = cp.experts_multi(slots, g["off"], Xp, Wp, bgu_ptrs, bdn_ptrs, scale_layout=0)   # [P, 2880]; 1 = packed store

``Y[p] = 0 + Wp[p] * expert(Xp[p])`` per pair (NOT summed). To sum a token's experts
exactly the way ``gptoss_experts(E, ...)`` does, run the kernel with ``W = 1`` and use
``combine(g["idx"], topw, Y)``. Summing weighted pairs in numpy is NOT bit-identical:
the compiled decode combine uses rounded products for E = 4 (blocks of 8/4) and fused
multiply-adds for the last E mod 4 terms (E <= 3: pure FMA chain).

The DLL also exports ``gptoss_experts`` / ``gptoss_experts_cap`` built from the unchanged
gptoss_cpu_cap2.c source (``CpuMultiExperts.lib``), so a process can use one libgomp pool.

SCALE LAYOUT: slots are read in the layout ``set_scale_layout`` selects, which must match the store the
row pointers come from: 0 = raw (one E8M0 byte per group, 13,219,200 B/slot; store repr "mxfp4_g32"), 1 = packed
(base byte + 4-bit deltas per 90-group row, 12,839,040 B/slot; "mxfp4_g32_ps4"). The mode lives inside each DLL
(process-global): set it on this DLL AND on the decode kernel DLL (``bind_scale_layout``). The module-level
``default()`` / ``experts_multi()`` conveniences REQUIRE ``scale_layout=`` so a packed store can never be read as
raw by omission. Nothing here touches the GPU. Build:
    gcc -O3 -march=native -fopenmp -shared -o gptoss_cpu_multi.dll kernels/gptoss_cpu_multi.c
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from neural.moe.layout import GPTOSS_LAYOUT   # torch-free; the raw slot size of kernels that predate the layout switch

H, GU = 2880, 5760
_INT_MAX = 2**31 - 1
MAX_PAIRS = (_INT_MAX - 16) // (H + GU)       # 248,551: largest P with P*(H+GU) + 16 <= INT_MAX
_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVKIT = os.environ.get("NEURAL_DEVKIT", r"F:\AI\Neural\third_party\tools\w64devkit\bin")   # optional
VP = ctypes.c_void_p


def _scale_api(dll) -> bool:
    """Declare the scale-layout exports of a kernel DLL (gptoss_cpu_cap2.c, also exported by gptoss_cpu_multi.c);
    False for a build that predates the switch (it reads raw slots only)."""
    if not hasattr(dll, "gptoss_set_scale_layout"):
        return False
    dll.gptoss_set_scale_layout.argtypes = [ctypes.c_int]
    dll.gptoss_set_scale_layout.restype = ctypes.c_int
    dll.gptoss_get_scale_layout.argtypes = []
    dll.gptoss_get_scale_layout.restype = ctypes.c_int
    dll.gptoss_slot_bytes.argtypes = [ctypes.c_int]
    dll.gptoss_slot_bytes.restype = ctypes.c_longlong
    return True


def bind_scale_layout(dll, mode: int, slot_bytes: int | None = None, name: str = "kernel DLL") -> bool:
    """Point a kernel DLL (a ctypes.CDLL of gptoss_cpu_cap2.c / gptoss_cpu_multi.c) at the scale layout of the
    expert store (0 raw, 1 packed; ExpertLayout.scale_layout_mode) and, when `slot_bytes` is given, check that the
    kernel's slot size in that mode equals the store's. Process-global inside the DLL: call it once before serving,
    on EVERY kernel DLL the process loads. Returns True when the mode was set, False when the DLL predates the
    switch and mode == 0 (it reads raw slots, which is what the store holds). Raises RuntimeError - never
    misreads - when a packed store meets a DLL without the switch, or the slot sizes disagree."""
    mode = int(mode)
    if not _scale_api(dll):
        if mode != 0:
            raise RuntimeError(f"{name} predates gptoss_set_scale_layout but the expert store holds packed scales "
                               f"(mode {mode}): rebuild the kernels with tools\\build_kernels.bat")
        if slot_bytes is not None and int(slot_bytes) != GPTOSS_LAYOUT.slot_bytes:
            raise RuntimeError(f"{name} reads {GPTOSS_LAYOUT.slot_bytes:,} B slots only, the store has {int(slot_bytes):,} B")
        return False
    if dll.gptoss_set_scale_layout(mode) != 0:
        raise RuntimeError(f"{name} does not support scale layout {mode}")
    if slot_bytes is not None and dll.gptoss_slot_bytes(mode) != int(slot_bytes):
        raise RuntimeError(f"{name}: kernel slot size {dll.gptoss_slot_bytes(mode):,} B != store slot size "
                           f"{int(slot_bytes):,} B (scale layout {mode}); rebuild the kernels with tools\\build_kernels.bat")
    return True


def _cint(v, name):
    """int(v), refusing values that a C int cannot hold (ctypes would wrap them silently)."""
    v = int(v)
    if not -_INT_MAX - 1 <= v <= _INT_MAX:
        raise ValueError(f"{name}={v} does not fit a C int")
    return v


def _ptr(a, dtype, name, min_rows=None, width=None, ndim=None):
    """Data pointer of a C-contiguous CPU numpy array or torch tensor of `dtype`.
    width: require shape [*, width]; ndim: require exactly that many dimensions."""
    if hasattr(a, "data_ptr"):                                   # torch.Tensor
        import torch
        tdt = {np.float32: torch.float32, np.int32: torch.int32}[dtype]
        if a.device.type != "cpu" or a.dtype != tdt or not a.is_contiguous():
            raise ValueError(f"{name}: need a contiguous CPU {tdt} tensor")
        shape = tuple(a.shape)
        p = a.data_ptr()
    else:
        if not isinstance(a, np.ndarray) or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError(f"{name}: need a C-contiguous {np.dtype(dtype).name} numpy array")
        shape = a.shape
        p = a.ctypes.data
    if width is not None and (len(shape) != 2 or shape[1] != width):
        raise ValueError(f"{name}: shape {shape}, want [*, {width}]")
    if ndim is not None and len(shape) != ndim:
        raise ValueError(f"{name}: shape {shape}, want {ndim}-D")
    if min_rows is not None and (len(shape) == 0 or shape[0] < min_rows):
        raise ValueError(f"{name}: {shape[0] if shape else 0} rows < {min_rows}")
    return p


class CpuMultiExperts:
    """Loads gptoss_cpu_multi.dll; keeps a reusable, growing scratch buffer.

    Not thread-safe: one instance must not be called from two Python threads at once
    (shared scratch). Create one instance per calling thread if needed."""

    def __init__(self, threads: int = 8, dll: str | None = None, scale_layout: int | None = None):
        if os.path.isdir(_DEVKIT):
            os.add_dll_directory(_DEVKIT)
        self.dll_path = os.path.abspath(dll or os.path.join(_DIR, "gptoss_cpu_multi.dll"))
        self.lib = ctypes.CDLL(self.dll_path)
        L = self.lib
        self.has_scale_layout = _scale_api(L)
        L.gptoss_experts_multi.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]
        L.gptoss_experts_multi.restype = None
        L.gptoss_multi_scratch_floats.argtypes = [ctypes.c_int, ctypes.c_int]
        L.gptoss_multi_scratch_floats.restype = ctypes.c_int
        L.gptoss_combine_pairs.argtypes = [ctypes.c_int, ctypes.c_int, VP, VP, VP, VP, ctypes.c_int]
        L.gptoss_combine_pairs.restype = None
        L.gptoss_multi_set_tiling.argtypes = [ctypes.c_int, ctypes.c_int]
        L.gptoss_multi_set_tiling.restype = None
        L.gptoss_multi_set_schedule.argtypes = [ctypes.c_int, ctypes.c_int]
        L.gptoss_multi_set_schedule.restype = None
        L.gptoss_multi_get_config.argtypes = [VP]
        L.gptoss_multi_get_config.restype = None
        L.gptoss_multi_activation.argtypes = [ctypes.c_int, VP, VP, ctypes.c_int]
        L.gptoss_multi_activation.restype = None
        L.gptoss_multi_check_exp.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_longlong)]
        L.gptoss_multi_check_exp.restype = ctypes.c_longlong
        # the unchanged single-token kernels (same source as gptoss_cpu_cap2.dll)
        L.gptoss_experts.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]
        L.gptoss_experts.restype = None
        L.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
        L.gptoss_experts_cap.restype = None
        self.threads = int(threads)
        self._scr = np.zeros(0, np.float32)
        if scale_layout is not None:
            self.set_scale_layout(scale_layout)

    # ---- scale layout of the slots this DLL reads (process-global inside the DLL)
    def set_scale_layout(self, mode: int) -> None:
        """0 = raw slots (default), 1 = packed 4-bit scale deltas. Set once, before any call. A DLL built
        before the layout switch supports only mode 0 (raises for 1: it would misread packed rows)."""
        mode = int(mode)
        if not self.has_scale_layout:
            if mode != 0:
                raise RuntimeError("this kernel DLL predates gptoss_set_scale_layout: it can only read raw slots; "
                                   "rebuild it with tools\\build_kernels.bat")
            return
        if self.lib.gptoss_set_scale_layout(mode) != 0:
            raise ValueError(f"scale layout {mode} is not supported by this kernel")

    def scale_layout(self) -> int:
        return int(self.lib.gptoss_get_scale_layout()) if self.has_scale_layout else 0

    def slot_bytes(self, mode: int | None = None) -> int:
        """Bytes per slot the kernel expects in `mode` (default: the current one)."""
        m = self.scale_layout() if mode is None else int(mode)
        return int(self.lib.gptoss_slot_bytes(m)) if self.has_scale_layout else GPTOSS_LAYOUT.slot_bytes

    def scratch_floats(self, E: int, P: int) -> int:
        n = int(self.lib.gptoss_multi_scratch_floats(int(E), int(P)))
        if n < 0:
            raise ValueError(f"P={P} pairs is too large for one call "
                             f"(need P*8640 + 16 <= 2^31-1, i.e. P <= {MAX_PAIRS:,})")
        return n

    # ---- tuning knobs (process-global inside the DLL; set between calls, not during one)
    # Row counts are clamped to [1, 5760] inside the DLL (a larger value is equivalent to 5760);
    # values <= 0 and tokens_per_block outside 1..8 are ignored. ints must fit a C int.
    def set_tiling(self, rows_per_tile: int = 16, tokens_per_block: int = 8) -> None:
        """Rows per L2 tile and tokens per register block (1..8). Results are bit-identical
        for every setting; only speed changes. Defaults 16 / 8."""
        self.lib.gptoss_multi_set_tiling(_cint(rows_per_tile, "rows_per_tile"),
                                         _cint(tokens_per_block, "tokens_per_block"))

    def set_schedule(self, dynamic: bool = True, rows_per_item: int = 64) -> None:
        """dynamic=True: OpenMP dynamic items of `rows_per_item` rows of one expert (default);
        False: cost-weighted static contiguous row split. Bit-identical either way.
        The dynamic default deliberately deviates from the spec's static schedule: it is faster
        when other threads share the cores (see the C header for the measured numbers)."""
        self.lib.gptoss_multi_set_schedule(1 if dynamic else 0, _cint(rows_per_item, "rows_per_item"))

    def get_config(self) -> dict:
        """Effective (clamped) knob values inside the DLL."""
        cfg = (ctypes.c_int * 4)()
        self.lib.gptoss_multi_get_config(cfg)
        return {"rows_per_tile": cfg[0], "tokens_per_block": cfg[1], "dynamic": bool(cfg[2]),
                "rows_per_item": cfg[3]}

    # ---- test hooks
    def activation(self, GUin, vec: bool = True):
        """Clamped SwiGLU of pre-activations GUin fp32 [P, 5760] ([gate|up]) -> [P, 2880]
        de-interleaved [h_even(1440) | h_odd(1440)]; vec=False runs cap2's scalar expression."""
        P = GUin.shape[0]
        out = np.empty((P, H), np.float32)
        self.lib.gptoss_multi_activation(P, _ptr(GUin, np.float32, "GUin", P, GU), out.ctypes.data,
                                         1 if vec else 0)
        return out

    def check_exp(self, begin: int = 0, end: int = 1 << 32, threads: int | None = None):
        """Exhaustive check of the vector expf against the scalar expf over float bit patterns
        [begin, end). Returns (mismatches, first_bad_pattern, fallback_lanes)."""
        fb, nf = ctypes.c_uint(), ctypes.c_longlong()
        bad = self.lib.gptoss_multi_check_exp(int(begin), int(end), int(threads or self.threads),
                                              ctypes.byref(fb), ctypes.byref(nf))
        return int(bad), int(fb.value), int(nf.value)

    def _scratch(self, E: int, P: int):
        need = self.scratch_floats(E, P)
        if self._scr.size < need:
            self._scr = np.empty(max(need, int(self._scr.size * 1.25)), np.float32)
        return self._scr.ctypes.data

    def experts(self, slots, off, X, W, bgu, bdn, out=None, threads: int | None = None):
        """slots/bgu/bdn: E pointers (ints) to the expert's store row / fp32 gate_up bias
        [5760] / fp32 down bias [2880]. off: E+1 non-decreasing ints, pairs of expert e are
        [off[e], off[e+1]) (off[0] normally 0). X: fp32 [>=P, 2880] packed pair inputs,
        W: fp32 1-D [>=P] per-PAIR routing weights (not the [N, K] topw), P = off[E].
        Returns Y fp32 [P, 2880], row p = pair p. With `out` given, rows outside
        [off[0], P) of `out` are left untouched; with out=None, rows [0, off[0]) of the new
        array are +0.0 (never uninitialized memory)."""
        E = len(slots)
        if len(bgu) != E or len(bdn) != E:
            raise ValueError("slots/bgu/bdn length mismatch")
        off64 = np.asarray(off)
        if off64.dtype.kind not in "iu":
            raise ValueError(f"off must be integer, got {off64.dtype}")
        off64 = off64.astype(np.int64)
        if off64.shape != (E + 1,):
            raise ValueError(f"off must have E+1={E + 1} entries, got {off64.shape}")
        if off64[0] < 0 or np.any(np.diff(off64) < 0):
            raise ValueError("off must be >= 0 and non-decreasing")
        if off64[-1] > MAX_PAIRS:
            raise ValueError(f"P={int(off64[-1])} pairs is too large for one call (P <= {MAX_PAIRS:,})")
        offa = np.ascontiguousarray(off64.astype(np.int32))
        P, p_lo = int(offa[-1]), int(offa[0])
        if out is None:
            out = np.empty((P, H), np.float32)
            out[:p_lo] = 0.0                                # the kernel writes only [off[0], P)
        xp = _ptr(X, np.float32, "X", P, H)
        wp = _ptr(W, np.float32, "W", P, ndim=1)
        yp = _ptr(out, np.float32, "out", P, H)
        if E == 0 or P == 0:
            return out[:P]
        sl = (VP * E)(*[int(s) for s in slots])
        bg = (VP * E)(*[int(b) for b in bgu])
        bd = (VP * E)(*[int(b) for b in bdn])
        self.lib.gptoss_experts_multi(E, sl, offa.ctypes.data, xp, bg, bd, wp, yp,
                                      self._scratch(E, P), _cint(threads or self.threads, "threads"))
        return out[:P]

    def combine(self, idx, w, Yr, out=None, threads: int | None = None):
        """out[t] = 0 + sum_k w[t,k] * Yr[idx[t,k]] over idx >= 0 in k order, with exactly the
        rounding of gptoss_experts' compiled combine for E_t = #valid entries (see the C
        source). Yr = experts(..., W=ones). idx int32 [N, K] (-1 = skip), w fp32 [N, K]."""
        idx = np.ascontiguousarray(np.asarray(idx, dtype=np.int32))
        w = np.ascontiguousarray(np.asarray(w, dtype=np.float32))
        if idx.ndim != 2 or w.shape != idx.shape:
            raise ValueError("idx / w must both be [N, K]")
        N, K = idx.shape
        if N and idx.max(initial=-1) >= (Yr.shape[0] if hasattr(Yr, "shape") else 0):
            raise ValueError("idx out of range of Yr")
        if out is None:
            out = np.empty((N, H), np.float32)
        yp = _ptr(Yr, np.float32, "Yr", None, H)
        op = _ptr(out, np.float32, "out", N, H)
        if N:
            self.lib.gptoss_combine_pairs(N, K, idx.ctypes.data, w.ctypes.data, yp, op,
                                          int(threads or self.threads))
        return out[:N]


def group_pairs(topi, cpu_mask=None):
    """Group one layer's routing into expert-major pairs.

    topi: int [N, K] expert ids per token; cpu_mask: optional bool [N, K], True = this
    (token, k) is computed by the CPU kernel. Returns dict with
      experts    int64 [E]   distinct expert ids (ascending) -> slots/bgu/bdn order
      off        int32 [E+1] pair ranges per expert
      pair_token int64 [P]   token of each pair (tokens ascending within an expert)
      pair_k     int64 [P,2] (token, k) of each pair, e.g. W = topw[pk[:,0], pk[:,1]]
      idx        int32 [N,K] pair index of (token, k) or -1 -> combine()"""
    topi = np.asarray(topi)
    N, K = topi.shape
    m = np.ones((N, K), bool) if cpu_mask is None else np.asarray(cpu_mask, bool)
    tok, kk = np.nonzero(m)                                   # row-major: token-ascending
    ex = topi[tok, kk].astype(np.int64)
    order = np.argsort(ex, kind="stable")
    tok, kk, ex = tok[order], kk[order], ex[order]
    experts, counts = np.unique(ex, return_counts=True)
    off = np.zeros(len(experts) + 1, np.int32)
    np.cumsum(counts, out=off[1:])
    idx = np.full((N, K), -1, np.int32)
    idx[tok, kk] = np.arange(len(tok), dtype=np.int32)
    return {"experts": experts, "off": off, "pair_token": tok.astype(np.int64),
            "pair_k": np.stack([tok, kk], 1).astype(np.int64), "idx": idx}


_DEFAULT: CpuMultiExperts | None = None


def default(threads: int = 8, *, dll: str | None = None, scale_layout: int) -> CpuMultiExperts:
    """The process-wide CpuMultiExperts (created on first use; a different `dll` replaces it). `scale_layout` is
    REQUIRED and re-applied on every call: 0 = raw slots, 1 = packed slots (ExpertLayout.scale_layout_mode of the
    store the row pointers come from). There is deliberately no default, so a packed store cannot be read as raw
    by omission."""
    global _DEFAULT
    if _DEFAULT is None or (dll is not None and _DEFAULT.dll_path != os.path.abspath(dll)):
        _DEFAULT = CpuMultiExperts(threads, dll=dll, scale_layout=scale_layout)
    else:
        _DEFAULT.set_scale_layout(scale_layout)
    return _DEFAULT


def experts_multi(slots, off, X, W, bgu, bdn, out=None, threads: int = 8, *, dll: str | None = None,
                  scale_layout: int):
    """Module-level convenience over the process-wide CpuMultiExperts; ``scale_layout`` (0 raw / 1 packed) and
    optionally ``dll`` are required / accepted here, see default()."""
    return default(threads, dll=dll, scale_layout=scale_layout).experts(slots, off, X, W, bgu, bdn, out=out,
                                                                       threads=threads)
