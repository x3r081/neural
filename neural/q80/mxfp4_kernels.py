"""CAPACITY-4 — generic mxfp4_g32 GEMM kernel family (Triton).

Consumes the SOURCE representation of MXFP4 checkpoints directly from the
slot pool: packed fp4-e2m1 codes (2/byte, low nibble = even position) with one
E8M0 scale byte per 32-value group along K. The implicit dequantization is
EXACT (fp4 values and 2^(scale-127) are exactly representable; bf16 products
are exact in fp32), accumulation is fp32 via tl.dot, output bf16.

Layout consumed (u8 views over slot-pool rows, slot-strided):
  blocks [cap, N, K/32, 16]   scales [cap, N, K/32]         (raw: one E8M0 byte per group)
                              scales [cap, N, K/64 + 1]     (PACKED, weight_repr "mxfp4_g32_ps4")
A packed scale row is 1 + GK/2 bytes: byte 0 = base (the row's minimum scale), byte 1 + k = the 4-bit deltas
of groups 2k (low nibble) and 2k + 1 (high nibble). The kernels rebuild scale = base + delta as an exact
integer, i.e. precisely the byte the raw layout stores, so both layouts produce bit-identical outputs
(tools/scale_pack_gpu_test.py). The launchers pick the layout from the scales view's last dimension
(GK raw / GK//2 + 1 packed), or from the explicit ``packed=`` argument (checked against the view).

One kernel serves both GEMMs and both phases:
  decode  gate_up: slots=[k selected], x [1, K]      -> out [k, 1, N]
  decode  down   : slots=[one slot],   h [1, K2]     -> out [1, 1, N2]
  prefill (either): slots=[one slot],  x [M, K]      -> out [1, M, N]

Deterministic: fixed grid/order, no atomics, no autotune (CUDA-graph safe).
"""

from __future__ import annotations

import os
import operator

import torch
import triton
import triton.language as tl

# Host-side switches, read once at import (both are also plain module attributes, readable and settable at runtime).
#   NEURAL_FAST_LAUNCH=0  -> every launch goes through Triton's stock JIT dispatch (A/B against the fast path below;
#                            the server's _expert_gemms also reads it for its bit-exact torch-op reductions).
#   NEURAL_PREFILL_BM     -> BLOCK_M of the prefill GEMM as launched by the server's _expert_gemms (default 64; 16 =
#                            mxfp4_gemm's default). mxfp4_gemv (decode) has no BLOCK_M and is not affected.
FAST_LAUNCH = os.environ.get("NEURAL_FAST_LAUNCH", "1") != "0"
PREFILL_BLOCK_M = int(os.environ.get("NEURAL_PREFILL_BM", "64"))
if PREFILL_BLOCK_M < 16 or PREFILL_BLOCK_M & (PREFILL_BLOCK_M - 1):
    raise ValueError(f"NEURAL_PREFILL_BM={PREFILL_BLOCK_M}: BLOCK_M must be a power of two >= 16 (tl.dot tile)")
# Grouped epilogue of the server's layer-order prefill (server.py moe_prefill_layer / _pf_run_chunk): the GEMMs stay one
# launch per (expert, block) pair, the pointwise bias / SwiGLU / weight-scale ops run once per group of pairs.
#   NEURAL_PREFILL_GROUP=0          -> the per-pair _expert_gemms path (A/B; also taken when FAST_LAUNCH is off)
#   NEURAL_PREFILL_GROUP_ROWS       -> row cap of one group (default 16384; transient VRAM ~ rows * 28.8 KB for the
#                                      2880 / 5760-wide gpt-oss experts: gate_up + its bias gather + the activation)
#   NEURAL_PREFILL_GROUP_EXPERTS    -> staged experts per group (default 0 = one per scratch slot; the server clips it
#                                      to the slot count; e.g. 3 of 6 slots = double buffering between groups)
PREFILL_GROUP = os.environ.get("NEURAL_PREFILL_GROUP", "0") != "0"   # measured neutral (the copy pipeline becomes the wall); opt-in
# Distinct from PREFILL_GROUP: issue each group's ragged GEMMs in ONE launch.
# Opt-in until compiled-kernel identity and in-server timings have been measured.
PREFILL_GROUPED_GEMM = os.environ.get("NEURAL_PREFILL_GROUPED_GEMM", "0") != "0"
PREFILL_GROUP_ROWS = int(os.environ.get("NEURAL_PREFILL_GROUP_ROWS", "16384"))
PREFILL_GROUP_EXPERTS = int(os.environ.get("NEURAL_PREFILL_GROUP_EXPERTS", "0"))
if PREFILL_GROUP_ROWS < 1 or PREFILL_GROUP_EXPERTS < 0:
    raise ValueError(f"NEURAL_PREFILL_GROUP_ROWS={PREFILL_GROUP_ROWS} (>= 1), NEURAL_PREFILL_GROUP_EXPERTS="
                     f"{PREFILL_GROUP_EXPERTS} (>= 0)")


@triton.jit
def _e2m1(n):
    """fp4 e2m1 nibble -> value. bits: s|e e|m."""
    s = tl.where((n & 8) != 0, -1.0, 1.0)
    e = ((n >> 1) & 3).to(tl.float32)
    m = (n & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * m, tl.exp2(e - 1.0) * (1.0 + 0.5 * m))
    return s * mag


def _scale_layout(scales: torch.Tensor, GK: int, packed) -> bool:
    """PACKED constexpr of a launch: True when `scales` (…, N, w) holds packed rows (w == GK//2 + 1), False
    when raw (w == GK). `packed` (None = infer) is checked against the view, never trusted over it."""
    w = scales.shape[-1]
    inferred = (w == GK // 2 + 1)
    if not (inferred or w == GK):
        raise ValueError(f"scales view has row width {w}: want {GK} (raw E8M0) or {GK // 2 + 1} (packed 4-bit)")
    if packed is not None and bool(packed) != inferred:
        raise ValueError(f"packed={packed} but the scales view row width {w} says "
                         f"{'packed' if inferred else 'raw'} (GK={GK})")
    if scales.stride(-1) != 1 or scales.stride(-2) != w:
        raise ValueError(f"scales rows must be contiguous (strides {tuple(scales.stride())}, row width {w})")
    return inferred


@triton.jit
def _mxfp4_gemm(x_ptr, b_ptr, s_ptr, slots_ptr, out_ptr,
                M, sb_slot, ss_slot,
                N: tl.constexpr, GK: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                PACKED: tl.constexpr = False):
    pe = tl.program_id(0)
    pn = tl.program_id(1)
    pm = tl.program_id(2)
    slot = tl.load(slots_ptr + pe).to(tl.int64)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    nmask = rn < N
    mmask = rm < M
    bb = b_ptr + slot * sb_slot
    bs = s_ptr + slot * ss_slot
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if PACKED:
        # row pitch GK//2 + 1: byte 0 = base scale, byte 1 + g//2 = the nibble pair of groups (g & ~1, g | 1)
        base = tl.load(bs + rn * (GK // 2 + 1), mask=nmask, other=127).to(tl.int32)
    for g in range(GK):
        byts = tl.load(bb + (rn[:, None] * GK + g) * 16
                       + tl.arange(0, 16)[None, :],
                       mask=nmask[:, None], other=0)
        if PACKED:
            pb = tl.load(bs + rn * (GK // 2 + 1) + 1 + g // 2, mask=nmask, other=0).to(tl.int32)
            sc = (base + ((pb >> ((g & 1) * 4)) & 15)).to(tl.float32)     # == the raw E8M0 byte, exactly
        else:
            sc = tl.load(bs + rn * GK + g, mask=nmask, other=127).to(tl.float32)
        vlo = _e2m1((byts & 0x0F).to(tl.int32))
        vhi = _e2m1((byts >> 4).to(tl.int32))
        w = tl.interleave(vlo, vhi) * tl.exp2(sc - 127.0)[:, None]  # [BN, 32]
        xk = tl.load(x_ptr + rm[:, None] * (GK * 32) + g * 32
                     + tl.arange(0, 32)[None, :],
                     mask=mmask[:, None], other=0.0)                # [BM, 32]
        acc += tl.dot(xk.to(tl.bfloat16), w.to(tl.bfloat16).trans(),
                      out_dtype=tl.float32)
    out = acc.to(tl.bfloat16)
    optr = out_ptr + (pe * M + rm[:, None]) * N + rn[None, :]
    tl.store(optr, out, mask=mmask[:, None] & nmask[None, :])


# ---------------------------------------------------------------------------
# Launch fast path (FAST_LAUNCH, default on). Triton's JITFunction.run redoes on EVERY launch what only changes
# when the kernel specialization changes: it binds and specializes all 13 arguments, builds the cache key (including
# str(options)), looks the CompiledKernel up, builds launch metadata and goes through CudaLauncher.__call__. That is
# host work per launch, paid twice per expert in prompt processing. Here the FIRST launch of every distinct
# specialization goes through Triton unchanged (it compiles or loads the binary and returns its CompiledKernel);
# later launches with the same key call that kernel's launcher directly (triton/runtime/jit.py run(), the
# kernel.run(...) call). Same compiled binary, same grid, same arguments, same stream => bit-identical results.
# The key holds everything Triton specializes this kernel on: the constexprs, the dtype and 16-byte alignment of
# every pointer (a length-1 view of an int64 slot table alternates alignment with slot parity: two binaries), M
# (Triton's OWN specialize function classifies it: whatever it distinguishes, e.g. 1 and divisibility by 16, gets
# its own kernel entry) and the two strides (exact values). If the Triton API is not what is used here, the
# import-time probe fails and every launch takes the stock path.
# ---------------------------------------------------------------------------
try:
    import triton.knobs as _tknobs
    from triton.runtime.jit import native_specialize_impl as _tspec
    _EN, _EX = _tknobs.runtime.launch_enter_hook, _tknobs.runtime.launch_exit_hook
    _PRE = _mxfp4_gemm.pre_run_hooks
    _EN.calls, _EX.calls, _mxfp4_gemm.device_caches                 # probe: HookChain lists (profiler hooks), binder cache
    try:
        from torch._C import _cuda_getCurrentRawStream as _raw_stream       # what triton's driver uses
    except ImportError:
        def _raw_stream(i):
            return torch.cuda.current_stream(i).cuda_stream
    _FAST_API = True
except Exception:                                                   # other Triton: stock dispatch only
    _FAST_API = False

_KERNELS: dict = {}     # specialization key -> (kernel launcher, function handle, packed metadata)
_MSPEC: dict = {}       # (device, M) -> Triton's (type, key) for an int argument of that value
_cur_dev = torch.cuda.current_device


def _mspec(dev, M):
    """Triton's own classification of the int argument M, from the backend instance its binder for this kernel uses
    (JITFunction.device_caches[dev] = kernel cache, key cache, target, backend, binder). Any failure -> stock path."""
    global _FAST_API
    try:
        ms = _tspec(_mxfp4_gemm.device_caches[dev][3], M, False, True, True)
        hash(ms)
    except Exception:
        _FAST_API = False
        return None
    _MSPEC[(dev, M)] = ms
    return ms


def _launch_gemm(x, blocks, scales, slots, out, E, M, N, GK, block_n, block_m, pk):
    """Launch _mxfp4_gemm: grid (E, ceil(N / block_n), ceil(M / block_m)), num_warps 4. `out` holds E*M*N bf16."""
    sb_slot = blocks.stride(0)
    ss_slot = scales.stride(0)
    gx, gy, gz = E, -(-N // block_n), -(-M // block_m)              # triton.cdiv
    key = None
    if FAST_LAUNCH and _FAST_API:
        dev = _cur_dev()
        ms = _MSPEC.get((dev, M)) or _mspec(dev, M)
        if ms is not None:
            key = (dev, N, GK, block_n, block_m, pk, sb_slot, ss_slot, ms,
                   x.data_ptr() & 15, blocks.data_ptr() & 15, scales.data_ptr() & 15, slots.data_ptr() & 15,
                   out.data_ptr() & 15, x.dtype, blocks.dtype, scales.dtype, slots.dtype, out.dtype)
            ent = _KERNELS.get(key)
            if ent is not None and not (_EN.calls or _EX.calls or _PRE):     # hooks/pre-run hooks: stock path
                ent[0](gx, gy, gz, _raw_stream(dev), ent[1], ent[2], None, _EN, _EX,
                       x, blocks, scales, slots, out, M, sb_slot, ss_slot, N, GK, block_n, block_m, pk)
                return
    k = _mxfp4_gemm[(gx, gy, gz)](x, blocks, scales, slots, out, M, sb_slot, ss_slot,
                                   N=N, GK=GK, BLOCK_N=block_n, BLOCK_M=block_m,
                                   PACKED=pk, num_warps=4)
    if key is not None and hasattr(k, "packed_metadata") and hasattr(k, "function"):
        _KERNELS[key] = (k.run, k.function, k.packed_metadata)     # k.run: initialised by the launch above


def mxfp4_gemm(x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor,
               slots: torch.Tensor, *, block_n: int = 128,
               block_m: int = 16, packed: bool | None = None) -> torch.Tensor:
    """x [M, K] bf16; blocks/scales = slot-pool u8 views; slots int64 [E].
    Returns [E, M, N] bf16. E>1 requires M==1 (stacked decode gate_up).
    scales: [cap, N, GK] raw or [cap, N, GK//2 + 1] packed (``packed`` None = inferred from the view)."""
    E = slots.numel()
    M, K = x.shape
    cap, N, GK, _ = blocks.shape
    assert GK * 32 == K
    pk = _scale_layout(scales, GK, packed)
    out = torch.empty(E, M, N, dtype=torch.bfloat16, device=x.device)
    _launch_gemm(x, blocks, scales, slots, out, E, M, N, GK, block_n, block_m, pk)
    return out


def mxfp4_gemm_into(x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor,
                    slots: torch.Tensor, out: torch.Tensor, *, block_n: int = 128,
                    block_m: int = 16, packed: bool | None = None) -> torch.Tensor:
    """mxfp4_gemm writing into a caller-provided contiguous bf16 `out` with E*M*N elements (any shape; the kernel sees
    a pointer: [M, N] for one slot is the same bytes as [1, M, N]) instead of allocating. Same kernel, grid, block
    configuration and arithmetic; the caller owns the stream ordering of `out` (one stream: enqueue order)."""
    E = slots.numel()
    M, K = x.shape
    cap, N, GK, _ = blocks.shape
    assert GK * 32 == K
    pk = _scale_layout(scales, GK, packed)
    assert out.dtype is torch.bfloat16 and out.numel() == E * M * N and out.is_contiguous()
    _launch_gemm(x, blocks, scales, slots, out, E, M, N, GK, block_n, block_m, pk)
    return out


# ---------------------------------------------------------------------------
# Ragged multi-expert GEMM. A tile belongs to ONE original (expert, block) pair;
# rows from different pairs never share a tensor-core tile, even for one slot.
# The arithmetic below deliberately remains the stock _mxfp4_gemm arithmetic:
# same BM/BN, group-by-group dequantization, dot sequence and final bf16 cast.
# Only pointer selection and the launch grid change. GPU bit tests, rather than
# this structural argument alone, must establish identity for a given compiler.
# ---------------------------------------------------------------------------
@triton.jit
def _mxfp4_gemm_grouped(b_ptr, s_ptr, tiles_ptr, sb_slot, ss_slot,
                       N: tl.constexpr, GK: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                       PACKED: tl.constexpr = False,
                       ALIGNED: tl.constexpr = False):
    tile = tl.program_id(0)
    pn = tl.program_id(1)
    # descriptor = [input pointer, output pointer, physical slot, pair M, pair pm]
    desc = tiles_ptr + tile * 5
    x_address = tl.load(desc)
    out_address = tl.load(desc + 1)
    x_ptr = x_address.to(tl.pointer_type(tl.bfloat16))
    out_ptr = out_address.to(tl.pointer_type(tl.bfloat16))
    if ALIGNED:
        # Descriptor loads hide the original tensor argument alignment from
        # Triton. Host validation proves these byte addresses and row pitches
        # are multiples of 16. Hint pointers after the cast because Triton's
        # AxisInfo does not propagate integer hints through int_to_ptr.
        # Pointer divisibility is in bytes; offset views stay unhinted.
        x_ptr = tl.multiple_of(x_ptr, 16)
        out_ptr = tl.multiple_of(out_ptr, 16)
    slot = tl.load(desc + 2).to(tl.int64)
    M = tl.load(desc + 3).to(tl.int32)
    pm = tl.load(desc + 4).to(tl.int32)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    nmask = rn < N
    mmask = rm < M
    bb = b_ptr + slot * sb_slot
    bs = s_ptr + slot * ss_slot
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if PACKED:
        base = tl.load(bs + rn * (GK // 2 + 1), mask=nmask, other=127).to(tl.int32)
    for g in range(GK):
        byts = tl.load(bb + (rn[:, None] * GK + g) * 16
                       + tl.arange(0, 16)[None, :],
                       mask=nmask[:, None], other=0)
        if PACKED:
            pb = tl.load(bs + rn * (GK // 2 + 1) + 1 + g // 2, mask=nmask, other=0).to(tl.int32)
            sc = (base + ((pb >> ((g & 1) * 4)) & 15)).to(tl.float32)
        else:
            sc = tl.load(bs + rn * GK + g, mask=nmask, other=127).to(tl.float32)
        vlo = _e2m1((byts & 0x0F).to(tl.int32))
        vhi = _e2m1((byts >> 4).to(tl.int32))
        w = tl.interleave(vlo, vhi) * tl.exp2(sc - 127.0)[:, None]
        xk = tl.load(x_ptr + rm[:, None] * (GK * 32) + g * 32
                     + tl.arange(0, 32)[None, :],
                     mask=mmask[:, None], other=0.0)
        acc += tl.dot(xk.to(tl.bfloat16), w.to(tl.bfloat16).trans(),
                      out_dtype=tl.float32)
    out = acc.to(tl.bfloat16)
    optr = out_ptr + rm[:, None] * N + rn[None, :]
    tl.store(optr, out, mask=mmask[:, None] & nmask[None, :])


def _grouped_tile_rows(pair_descriptors, block_m):
    """Pure host plan builder: (x address, y address, slot, M) -> pair-local tiles.

    Empty pairs contribute no tile. Integer masks and pair-local origins remain
    those of ceil(M/BM) individual stock launches; no padding is concatenated.
    """
    block_m = operator.index(block_m)
    if block_m < 16 or block_m & (block_m - 1):
        raise ValueError("block_m must be a power of two >= 16")
    rows = []
    for desc in pair_descriptors:
        if len(desc) != 4:
            raise ValueError("pair descriptor must have four integers")
        xptr, yptr, slot, m = (operator.index(v) for v in desc)
        if xptr < 0 or yptr < 0 or slot < 0 or not 0 <= m <= 2**31 - 1:
            raise ValueError("invalid pointer, slot or pair row count")
        rows.extend((xptr, yptr, slot, m, pm) for pm in range(-(-m // block_m)))
    return rows


class GroupedGemmPlan:
    """Owned metadata for a ragged GEMM, reusable while pair tensors stay fixed.

    `pairs` is an iterable of (contiguous bf16 x[M,K], physical slot integer,
    contiguous bf16 out[M,N]). Inputs may repeat; output slices must be disjoint
    and may not overlap ANY input. GPU metadata is a small pinned asynchronous
    upload. This object retains the upload source and all pair tensors. Callers
    own stream ordering of inputs, pool slots and outputs, as with gemm_into.
    Constructing a CPU plan is allowed for validation; launching requires CUDA.
    """
    __slots__ = ("pairs", "metadata_host", "metadata", "block_m", "K", "N",
                 "device", "slots", "ntiles", "aligned")

    def __init__(self, pairs, *, block_m=64):
        self.pairs = tuple(pairs)
        self.block_m = operator.index(block_m)
        _grouped_tile_rows((), self.block_m)  # validate even an empty plan
        self.K = self.N = None
        self.device = None
        self.aligned = True
        descs, read_spans, write_spans, slots = [], [], [], []
        for x, slot, out in self.pairs:
            slot = operator.index(slot)
            if x.ndim != 2 or out.ndim != 2 or x.shape[0] != out.shape[0]:
                raise ValueError("each grouped pair needs x[M,K] and out[M,N]")
            if x.dtype is not torch.bfloat16 or out.dtype is not torch.bfloat16:
                raise ValueError("grouped GEMM inputs and outputs must be bf16")
            if not x.is_contiguous() or not out.is_contiguous():
                raise ValueError("grouped GEMM pair matrices must be contiguous")
            if x.device != out.device:
                raise ValueError("a grouped pair's input/output must share a device")
            if self.device is None:
                self.device, self.K, self.N = x.device, x.shape[1], out.shape[1]
            if (x.device != self.device or x.shape[1] != self.K or out.shape[1] != self.N
                    or self.K < 32 or self.K % 32 or self.N < 1):
                raise ValueError("grouped pairs need one device, common K (multiple of 32), and common N")
            m = x.shape[0]
            descs.append((x.data_ptr(), out.data_ptr(), slot, m))
            slots.append(slot)
            if m:
                self.aligned &= (x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0
                                 and x.shape[1] * x.element_size() % 16 == 0
                                 and out.shape[1] * out.element_size() % 16 == 0)
                read_spans.append((x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()))
                write_spans.append((out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()))
        ordered = sorted(write_spans)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("grouped output slices overlap")
        # Inputs may overlap each other, but no input may overlap a write. A
        # sorted sweep avoids quadratic Python work for many resident pairs.
        write_i = 0
        for read_lo, read_hi in sorted(read_spans):
            while write_i < len(ordered) and ordered[write_i][1] <= read_lo:
                write_i += 1
            if write_i < len(ordered) and ordered[write_i][0] < read_hi:
                raise ValueError("grouped outputs overlap an input slice")
        tiles = _grouped_tile_rows(descs, self.block_m)
        self.slots = tuple(slots)
        self.ntiles = len(tiles)
        self.metadata_host = torch.empty((self.ntiles, 5), dtype=torch.int64,
                                         pin_memory=self.device is not None and self.device.type == "cuda")
        if tiles:
            self.metadata_host.copy_(torch.tensor(tiles, dtype=torch.int64))
        self.metadata = (self.metadata_host.to(self.device, non_blocking=True)
                         if self.device is not None and self.device.type == "cuda" else self.metadata_host)


def mxfp4_gemm_grouped_into(pairs_or_plan, blocks: torch.Tensor, scales: torch.Tensor,
                            *, block_n=128, block_m=64, packed=None):
    """One ragged GEMM launch, writing directly to each pair's own output slice.

    Return the GroupedGemmPlan so a caller can reuse fixed metadata or retain its
    owners. `pairs_or_plan` may be an existing plan; its block_m must match. No
    arithmetic/tiling switch is applied to the stock per-pair or decode paths.
    """
    plan = (pairs_or_plan if isinstance(pairs_or_plan, GroupedGemmPlan)
            else GroupedGemmPlan(pairs_or_plan, block_m=block_m))
    if plan.block_m != block_m:
        raise ValueError("plan block_m differs from the requested launch")
    if not plan.ntiles:
        return plan
    if blocks.ndim != 4 or blocks.shape[-1] != 16:
        raise ValueError("grouped blocks must have shape [capacity,N,GK,16]")
    cap, n, gk, _ = blocks.shape
    if plan.K != gk * 32 or plan.N != n or max(plan.slots) >= cap:
        raise ValueError("grouped plan dimensions or slots do not match the pool")
    if (plan.device != blocks.device or scales.device != blocks.device
            or blocks.device.type != "cuda"):
        raise ValueError("grouped launch needs plan, blocks and scales on the same CUDA device")
    if blocks.dtype is not torch.uint8 or scales.dtype is not torch.uint8:
        raise ValueError("grouped blocks and scales must be uint8")
    if tuple(blocks.stride()[1:]) != (gk * 16, 16, 1) or scales.shape[:2] != (cap, n):
        raise ValueError("grouped pool matrix rows must be contiguous and scales must match")
    block_n = operator.index(block_n)
    if block_n < 16 or block_n & (block_n - 1):
        raise ValueError("block_n must be a power of two >= 16")
    pk = _scale_layout(scales, gk, packed)
    _mxfp4_gemm_grouped[(plan.ntiles, -(-n // block_n))](
        blocks, scales, plan.metadata, blocks.stride(0), scales.stride(0),
        N=n, GK=gk, BLOCK_N=block_n, BLOCK_M=block_m, PACKED=pk,
        ALIGNED=plan.aligned, num_warps=4)
    return plan


# ---------------------------------------------------------------------------
# OVERNIGHT-1: GEMV-specialized kernel (M == 1). Removes the 16x tensor-core
# padding waste of the GEMM kernel at decode, loads codes as vectorized u32,
# and supports per-expert inputs (x [E, K]) so the four routed down-GEMVs run
# as ONE launch. fp32 accumulation, fixed reduction order, no atomics.
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_gemv(x_ptr, b_ptr, s_ptr, slots_ptr, out_ptr,
                sb_slot, ss_slot,
                N: tl.constexpr, GK: tl.constexpr, X_PER_EXPERT: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr,
                PACKED: tl.constexpr = False):
    pe = tl.program_id(0)
    pn = tl.program_id(1)
    slot = tl.load(slots_ptr + pe).to(tl.int64)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    bb = b_ptr + slot * sb_slot
    bs = s_ptr + slot * ss_slot
    xb = x_ptr + (pe * GK * 32 if X_PER_EXPERT else 0)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    if PACKED:
        base = tl.load(bs + rn * (GK // 2 + 1), mask=nmask, other=127).to(tl.int32)   # [BLOCK_N]
    for g0 in range(0, GK, BLOCK_G):
        gs = g0 + tl.arange(0, BLOCK_G)
        gmask = gs < GK
        # codes: [BLOCK_N, BLOCK_G, 16] u8 loaded via row-contiguous offsets
        offs = (rn[:, None, None] * GK + gs[None, :, None]) * 16 \
            + tl.arange(0, 16)[None, None, :]
        byts = tl.load(bb + offs, mask=nmask[:, None, None] & gmask[None, :, None],
                       other=0)
        if PACKED:
            pb = tl.load(bs + rn[:, None] * (GK // 2 + 1) + 1 + (gs[None, :] >> 1),
                         mask=nmask[:, None] & gmask[None, :], other=0).to(tl.int32)
            sci = base[:, None] + ((pb >> ((gs[None, :] & 1) * 4)) & 15)
            # masked lanes read 127 exactly like the raw path's other=127 (code 0 there, so the value is 0 either way)
            sc = tl.where(nmask[:, None] & gmask[None, :], sci, 127).to(tl.float32)
        else:
            sc = tl.load(bs + rn[:, None] * GK + gs[None, :],
                         mask=nmask[:, None] & gmask[None, :], other=127
                         ).to(tl.float32)
        vlo = _e2m1((byts & 0x0F).to(tl.int32))
        vhi = _e2m1((byts >> 4).to(tl.int32))
        w = tl.interleave(vlo, vhi) * tl.exp2(sc - 127.0)[:, :, None]
        xk = tl.load(xb + gs[:, None] * 32 + tl.arange(0, 32)[None, :],
                     mask=gmask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(tl.sum(w * xk[None, :, :], axis=2), axis=1)
    tl.store(out_ptr + pe * N + rn, acc.to(tl.bfloat16), mask=nmask)


def mxfp4_gemv(x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor,
               slots: torch.Tensor, *, per_expert_x: bool = False,
               block_n: int = 64, block_g: int = 4,
               num_warps: int = 4, packed: bool | None = None) -> torch.Tensor:
    """Decode-path GEMV. x [1, K] shared, or [E, K] with per_expert_x=True.
    Returns [E, N] bf16. scales: raw [cap, N, GK] or packed [cap, N, GK//2 + 1] (``packed`` None = inferred)."""
    E = slots.numel()
    cap, N, GK, _ = blocks.shape
    pk = _scale_layout(scales, GK, packed)
    out = torch.empty(E, N, dtype=torch.bfloat16, device=x.device)
    grid = (E, triton.cdiv(N, block_n))
    _mxfp4_gemv[grid](x, blocks, scales, slots, out,
                      blocks.stride(0), scales.stride(0),
                      N=N, GK=GK, X_PER_EXPERT=per_expert_x,
                      BLOCK_N=block_n, BLOCK_G=block_g, PACKED=pk, num_warps=num_warps)
    return out


# Opt-in zero-contribution skip, distinct from moving CPU misses onto the GPU.
# The stock GEMV above is untouched. Hit rows use the same arithmetic; a zero
# active weight writes +0 without reading that expert's weights. Intermediate
# missed-row values and their signed zeros need not equal stock. A caller may
# use this only when the original zero-weight expert outputs are finite AND its
# final fixed-order weighted reduction has been proved bit-identical. In
# particular, NaN*0 is NaN in stock, so this is not a generic NaN-preserving API.
@triton.jit
def _mxfp4_gemv_masked(x_ptr, b_ptr, s_ptr, slots_ptr, active_ptr, out_ptr,
                       sb_slot, ss_slot,
                       N: tl.constexpr, GK: tl.constexpr, X_PER_EXPERT: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr,
                       PACKED: tl.constexpr = False):
    pe = tl.program_id(0)
    pn = tl.program_id(1)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    if tl.load(active_ptr + pe) != 0:
        slot = tl.load(slots_ptr + pe).to(tl.int64)
        bb = b_ptr + slot * sb_slot
        bs = s_ptr + slot * ss_slot
        xb = x_ptr + (pe * GK * 32 if X_PER_EXPERT else 0)
        acc = tl.zeros((BLOCK_N,), tl.float32)
        if PACKED:
            base = tl.load(bs + rn * (GK // 2 + 1), mask=nmask, other=127).to(tl.int32)
        for g0 in range(0, GK, BLOCK_G):
            gs = g0 + tl.arange(0, BLOCK_G)
            gmask = gs < GK
            offs = (rn[:, None, None] * GK + gs[None, :, None]) * 16 \
                + tl.arange(0, 16)[None, None, :]
            byts = tl.load(bb + offs, mask=nmask[:, None, None] & gmask[None, :, None], other=0)
            if PACKED:
                pb = tl.load(bs + rn[:, None] * (GK // 2 + 1) + 1 + (gs[None, :] >> 1),
                             mask=nmask[:, None] & gmask[None, :], other=0).to(tl.int32)
                sci = base[:, None] + ((pb >> ((gs[None, :] & 1) * 4)) & 15)
                sc = tl.where(nmask[:, None] & gmask[None, :], sci, 127).to(tl.float32)
            else:
                sc = tl.load(bs + rn[:, None] * GK + gs[None, :],
                             mask=nmask[:, None] & gmask[None, :], other=127).to(tl.float32)
            vlo = _e2m1((byts & 0x0F).to(tl.int32))
            vhi = _e2m1((byts >> 4).to(tl.int32))
            w = tl.interleave(vlo, vhi) * tl.exp2(sc - 127.0)[:, :, None]
            xk = tl.load(xb + gs[:, None] * 32 + tl.arange(0, 32)[None, :],
                         mask=gmask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(tl.sum(w * xk[None, :, :], axis=2), axis=1)
        tl.store(out_ptr + pe * N + rn, acc.to(tl.bfloat16), mask=nmask)
    else:
        tl.store(out_ptr + pe * N + rn, 0.0, mask=nmask)


def mxfp4_gemv_masked(x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor,
                      slots: torch.Tensor, active_weights: torch.Tensor, *, per_expert_x=False,
                      block_n=64, block_g=4, num_warps=4, packed=None):
    """GEMV with zero-weight experts skipped, retaining the original row ordering.

    Same return shape/dtype and per-hit arithmetic as mxfp4_gemv. active_weights
    is a contiguous one-dimensional bf16/fp32 vector, one scalar per slot. NaNs
    and nonzero weights take the stock arithmetic branch. Caller must establish
    the finite-result/final-reduction contract documented above; skipped rows
    themselves are intentionally not bit-identical to unused stock results.
    """
    E = slots.numel()
    cap, N, GK, _ = blocks.shape
    if (active_weights.ndim != 1 or active_weights.numel() != E or not active_weights.is_contiguous()
            or active_weights.device != x.device or active_weights.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("active_weights needs one contiguous bf16/fp32 scalar per slot on the input device")
    if (x.dtype is not torch.bfloat16 or not x.is_contiguous() or x.ndim != 2
            or x.shape != ((E if per_expert_x else 1), GK * 32)):
        raise ValueError("masked GEMV requires contiguous bf16 x[1,K] or per-expert x[E,K]")
    if slots.device != x.device or blocks.device != x.device or scales.device != x.device:
        raise ValueError("masked GEMV tensors must share a device")
    pk = _scale_layout(scales, GK, packed)
    out = torch.empty(E, N, dtype=torch.bfloat16, device=x.device)
    _mxfp4_gemv_masked[(E, triton.cdiv(N, block_n))](
        x, blocks, scales, slots, active_weights, out, blocks.stride(0), scales.stride(0),
        N=N, GK=GK, X_PER_EXPERT=per_expert_x, BLOCK_N=block_n, BLOCK_G=block_g,
        PACKED=pk, num_warps=num_warps)
    return out
