"""HYBRID-3 D: the gpt-oss per-layer decode core as 5 Triton kernels (was ~94
PyTorch kernels + ~15 DtoD copies, 311 us/layer measured, ~66 us bandwidth floor).

  K2 qkv    combine(mid, g_out, c_out) -> input RMSNorm -> fused Q|K|V GEMV + bias
            -> RoPE -> write K/V cache at pos, Q to a buffer
  K3 attn   grouped sink attention over the valid key range (1 program / kv head)
  K4 oproj  o_proj GEMV + bias + residual -> h_mid (written to `mid`)
  K5a rtr   post RMSNorm (+ writes h_norm / bx / host pack[:H]) -> router GEMV + bias
  K5b topk  top-4 -> softmax -> residency lookup -> B inputs + host pack

Numerics: every bf16 rounding point of the eager core (gptoss_core_graphs.
_LayerGraph._compute) is reproduced explicitly (round-to-bf16 after each bf16
op), and exp/cos/sin/rsqrt use libdevice (the same accurate functions PyTorch's
CUDA kernels call). What is NOT reproduced is the fp32 ACCUMULATION ORDER of the
GEMVs / norm sums / softmax sums / PV sums (cuBLAS picks its own), so results are
not bit-exact; they are validated directly against the eager core on real
decode states (router top-4 agreement, h_mid / h_norm / KV error) and end-to-end.

COMPONENT A additions (both OFF by default: FusedCore(ring=0, split_k=1) is the old path):

  ring=R>0   ROLLING K/V cache for the sliding-window layers (window W=128 <= R). Each
             sliding layer's cache becomes [1, NKV, R, HD]; absolute position j lives in
             slot j % R. k_qkv stores at pos % R, k_attn reads key j from slot j % R with the
             SAME iteration order and arithmetic as the linear layout, so decode outputs are
             bit-identical (torch.equal) to a linear cache holding the same values. Prompt
             processing uses prefill_attention(..., k_new=, v_new=, ring=R) (keys < p0 from
             the ring, keys >= p0 from the block's own K/V) followed by ring_write(). At
             SMAX=16384 this frees 18 layers x 2 x 8 x (16384-128) x 64 x 2 B = 0.56 GiB.
             !! With ring>0 the eager reference path (_LayerGraph._compute, which does
             index_copy_ at the absolute pos, and GptOssCoreGraphs.load_prefill_kv, which
             writes absolute positions) is NO LONGER VALID for sliding layers: do not run or
             capture them. FusedCore sets lg.ring on EVERY layer in every mode (R for sliding
             layers when ring=R>0, else 0; FusedCore.layer_ring(L) is equivalent) so a caller
             can check. Prefix reuse is only valid if the ring still holds positions
             c-W+1..c-1 of the reused prefix (see ring_reuse_ok()).
  split_k=NS SPLIT-K decode attention for the full-attention layers: grid (NKV, NS), the key
             range [0, pos] is cut into NS contiguous chunks of whole BN blocks computed from
             pos INSIDE the kernels (graph-capturable, empty chunks are no-ops). Four kernels
             (partial max -> global max + partial sums -> global sum + partial P@V ->
             ordered reduce) reproduce every per-element bf16 rounding of k_attn (global m
             and S are known before any p is formed); only the fp32 summation order of S
             and of P@V differs - across chunks AND inside a tile: tl.sum's reduction order
             follows the register layout of its input (k_attn sums the dot-output layout, the
             split path sums cached scores in a load layout), and each chunk's P@V is formed
             from zero before the ordered reduce. So outputs can differ from k_attn by rounding
             flips even when every chunk holds one block (measured on synthetic data: 0-5 of
             4096 outputs at pos 1000 / NS 16, depending on the data). The partial-max kernel
             also caches the rounded scores s (bf16, exact) so the later passes read 16 B/key
             instead of re-reading K (128 B/key); cache_s=False recomputes the scores (same
             values in every measured case) but sums them in the dot-output layout, so cache_s
             on/off can also differ by such flips (the partial maxes, hence m, were
             bit-identical in every measured case).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
try:
    from triton.language.extra.cuda import globaltimer      # %globaltimer (ns); only k_bell (--doorbell) uses it
except Exception:                                             # noqa: BLE001  (a Triton without it: k_bell then fails to compile)
    globaltimer = None

H, HP, NQ, NKV, HD, NE, K = 2880, 4096, 64, 8, 64, 128, 4
REP = NQ // NKV


@triton.jit
def _rb(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _combine(mid, gout, cout, offs, msk, FIRST: tl.constexpr):
    x = tl.load(mid + offs, mask=msk, other=0.).to(tl.float32)
    if not FIRST:
        g = tl.load(gout + offs, mask=msk, other=0.).to(tl.float32)
        c = _rb(tl.load(cout + offs, mask=msk, other=0.))
        x = _rb(x + _rb(g + c))
    return x


@triton.jit
def k_qkv(mid, gout, cout, xbuf, in_w, W, Bq, pos_ptr, inv_freq, kc, vc, qout, eps, att_scale,
          SMAX: tl.constexpr, FIRST: tl.constexpr, H: tl.constexpr, HP: tl.constexpr,
          BK: tl.constexpr, NQ: tl.constexpr, NKV: tl.constexpr, RING: tl.constexpr = 0):
    """RING=0: K/V cache [1, NKV, SMAX, 64], position pos stored at row pos.
    RING>0: rolling cache [1, NKV, RING, 64], position pos stored at slot pos % RING."""
    pid = tl.program_id(0)
    head = pid // 2
    half = pid % 2
    offs = tl.arange(0, HP)
    msk = offs < H
    x = _combine(mid, gout, cout, offs, msk, FIRST)
    if pid == 0:
        tl.store(xbuf + offs, x.to(tl.bfloat16), mask=msk)
    var = tl.sum(x * x, axis=0) * (1.0 / H)
    r = libdevice.rsqrt(var + eps)
    i16 = tl.arange(0, 16)
    ra = head * 64 + half * 16 + i16                 # rows of the first rope half
    rb = ra + 32                                     # their rope partners
    acc_a = tl.zeros([16, BK], tl.float32)
    acc_b = tl.zeros([16, BK], tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < H
        xc = _combine(mid, gout, cout, kk, km, FIRST)
        wn = tl.load(in_w + kk, mask=km, other=0.).to(tl.float32)
        hc = _rb(wn * (xc * r))
        wa = tl.load(W + ra[:, None].to(tl.int64) * H + kk[None, :], mask=km[None, :], other=0.).to(tl.float32)
        wb = tl.load(W + rb[:, None].to(tl.int64) * H + kk[None, :], mask=km[None, :], other=0.).to(tl.float32)
        acc_a += wa * hc[None, :]
        acc_b += wb * hc[None, :]
    ya = _rb(tl.sum(acc_a, axis=1) + tl.load(Bq + ra).to(tl.float32))
    yb = _rb(tl.sum(acc_b, axis=1) + tl.load(Bq + rb).to(tl.float32))
    pos = tl.load(pos_ptr)
    d = half * 16 + i16
    if head < NQ + NKV:                              # q and k heads: RoPE
        fr = tl.load(inv_freq + d) * pos.to(tl.float32)
        cs = _rb(libdevice.cos(fr) * att_scale)
        sn = _rb(libdevice.sin(fr) * att_scale)
        o1 = _rb(_rb(ya * cs) - _rb(yb * sn))
        o2 = _rb(_rb(yb * cs) + _rb(ya * sn))
    else:
        o1 = ya
        o2 = yb
    if RING > 0:
        slot = pos % RING                            # rolling cache: slot of absolute pos
        rows = RING
    else:
        slot = pos
        rows = SMAX
    if head < NQ:
        tl.store(qout + head * 64 + d, o1.to(tl.bfloat16))
        tl.store(qout + head * 64 + 32 + d, o2.to(tl.bfloat16))
    elif head < NQ + NKV:
        base = (head - NQ).to(tl.int64) * rows * 64 + slot * 64
        tl.store(kc + base + d, o1.to(tl.bfloat16))
        tl.store(kc + base + 32 + d, o2.to(tl.bfloat16))
    else:
        base = (head - NQ - NKV).to(tl.int64) * rows * 64 + slot * 64
        tl.store(vc + base + d, o1.to(tl.bfloat16))
        tl.store(vc + base + 32 + d, o2.to(tl.bfloat16))


@triton.jit
def k_attn(q, kc, vc, sinks, pos_ptr, out, scaling, SLIDING: tl.constexpr, SMAX: tl.constexpr,
           REP: tl.constexpr, BN: tl.constexpr, RING: tl.constexpr = 0):
    """RING>0 (sliding layers only, RING >= SLIDING): key/value of absolute position j is
    read from slot j % RING of a [1, NKV, RING, 64] cache. Iteration order and arithmetic
    are identical to RING=0, so outputs are bit-identical to a linear cache."""
    kvh = tl.program_id(0)
    pos = tl.load(pos_ptr)
    if SLIDING > 0:
        lo = tl.maximum(pos - SLIDING + 1, 0)
    else:
        lo = pos * 0
    lo_b = (lo // BN) * BN
    r = tl.arange(0, 16)
    rm = r < REP
    d = tl.arange(0, 64)
    qt = tl.load(q + (kvh * REP + r)[:, None] * 64 + d[None, :], mask=rm[:, None], other=0.)
    sink = tl.load(sinks + kvh * REP + r, mask=rm, other=0.).to(tl.float32)
    if RING > 0:
        kb = kc + kvh.to(tl.int64) * RING * 64
        vb = vc + kvh.to(tl.int64) * RING * 64
    else:
        kb = kc + kvh.to(tl.int64) * SMAX * 64
        vb = vc + kvh.to(tl.int64) * SMAX * 64
    NEG = float("-inf")
    m = sink
    for s0 in range(lo_b, pos + 1, BN):                        # pass 1: max
        j = s0 + tl.arange(0, BN)
        ok = (j >= lo) & (j <= pos)
        if RING > 0:
            jr = j % RING
        else:
            jr = j
        kt = tl.load(kb + jr[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.)
        s = _rb(_rb(tl.dot(qt, tl.trans(kt))) * scaling)
        s = tl.where(ok[None, :], s, NEG)
        m = tl.maximum(m, tl.max(s, axis=1))
    ssum = libdevice.exp(_rb(sink - m))                       # pass 2: denominator
    for s0 in range(lo_b, pos + 1, BN):
        j = s0 + tl.arange(0, BN)
        ok = (j >= lo) & (j <= pos)
        if RING > 0:
            jr = j % RING
        else:
            jr = j
        kt = tl.load(kb + jr[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.)
        s = _rb(_rb(tl.dot(qt, tl.trans(kt))) * scaling)
        e = tl.where(ok[None, :], libdevice.exp(_rb(s - m[:, None])), 0.)
        ssum += tl.sum(e, axis=1)
    acc = tl.zeros([16, 64], tl.float32)                       # pass 3: P @ V
    for s0 in range(lo_b, pos + 1, BN):
        j = s0 + tl.arange(0, BN)
        ok = (j >= lo) & (j <= pos)
        if RING > 0:
            jr = j % RING
        else:
            jr = j
        kt = tl.load(kb + jr[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.)
        s = _rb(_rb(tl.dot(qt, tl.trans(kt))) * scaling)
        e = tl.where(ok[None, :], libdevice.exp(_rb(s - m[:, None])), 0.)
        p = tl.div_rn(e, ssum[:, None]).to(tl.bfloat16)
        vt = tl.load(vb + jr[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.)
        acc += tl.dot(p, vt)
    tl.store(out + (kvh * REP + r)[:, None] * 64 + d[None, :], acc.to(tl.bfloat16), mask=rm[:, None])


# ------------------------------------------------------------------ split-K decode attention
# Full-attention layers only (SLIDING == 0, linear cache [1, NKV, SMAX, 64]). Grid (NKV, NS)
# for kernels 1-3, (NKV,) for the reduce. Chunking is computed from pos on the device:
#   nb = cdiv(pos+1, BN) key blocks, bps = cdiv(nb, NS) blocks per split,
#   split sp covers keys [sp*bps*BN, min((sp+1)*bps*BN, pos+1))   (empty when sp*bps >= nb)
# The blocks are the same absolute BN-aligned blocks k_attn visits (lo = 0), so every score s,
# the global max m and every p = bf16(exp(bf16(s - m)) / S) use k_attn's exact rounding points.
# Buffers (fp32 unless noted): pm [NKV, NS, 16], ps [NKV, NS, 16], pacc [NKV, NS, REP, 64],
# sbuf [NKV, REP, SMAX] bf16 (score cache, used when CACHE_S).

@triton.jit
def _split_range(pos, sp, NS: tl.constexpr, BN: tl.constexpr):
    n = pos + 1
    nb = (n + BN - 1) // BN
    bps = (nb + NS - 1) // NS
    c0 = sp * bps * BN
    c1 = tl.minimum(c0 + bps * BN, n)
    return c0, c1


@triton.jit
def _global_max(sinks, pm, kvh, r, rm, REP: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr):
    """m = max(sink, max over splits of the partial maxes) - max is exact, so this equals
    k_attn's sequential running max bit for bit."""
    sink = tl.load(sinks + kvh * REP + r, mask=rm, other=0.).to(tl.float32)
    i = tl.arange(0, NSP)
    t = tl.load(pm + (kvh * NS + i)[:, None] * 16 + r[None, :], mask=(i < NS)[:, None],
                other=float("-inf"))
    return sink, tl.maximum(sink, tl.max(t, axis=0))


@triton.jit
def k_attn_s1(q, kc, pos_ptr, pm, sbuf, scaling, SMAX: tl.constexpr, REP: tl.constexpr,
              BN: tl.constexpr, NS: tl.constexpr, CACHE_S: tl.constexpr):
    """Split pass 1: partial row max of s over the chunk (-inf for an empty chunk); with
    CACHE_S also stores s (already bf16-rounded, so the bf16 store is exact)."""
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    pos = tl.load(pos_ptr)
    c0, c1 = _split_range(pos, sp, NS, BN)
    r = tl.arange(0, 16)
    rm = r < REP
    d = tl.arange(0, 64)
    qt = tl.load(q + (kvh * REP + r)[:, None] * 64 + d[None, :], mask=rm[:, None], other=0.)
    kb = kc + kvh.to(tl.int64) * SMAX * 64
    sb = sbuf + kvh.to(tl.int64) * REP * SMAX
    m = tl.full([16], float("-inf"), tl.float32)
    for s0 in range(c0, c1, BN):
        j = s0 + tl.arange(0, BN)
        ok = j <= pos
        kt = tl.load(kb + j[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.)
        s = _rb(_rb(tl.dot(qt, tl.trans(kt))) * scaling)
        if CACHE_S:
            tl.store(sb + r[:, None] * SMAX + j[None, :], s.to(tl.bfloat16), mask=rm[:, None] & ok[None, :])
        s = tl.where(ok[None, :], s, float("-inf"))
        m = tl.maximum(m, tl.max(s, axis=1))
    tl.store(pm + (kvh * NS + sp) * 16 + r, m)


@triton.jit
def _scores(qt, kb, sb, j, ok, r, rm, d, scaling, SMAX: tl.constexpr, CACHE_S: tl.constexpr):
    if CACHE_S:
        s = tl.load(sb + r[:, None] * SMAX + j[None, :], mask=rm[:, None] & ok[None, :], other=0.).to(tl.float32)
    else:
        kt = tl.load(kb + j[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.)
        s = _rb(_rb(tl.dot(qt, tl.trans(kt))) * scaling)
    return s


@triton.jit
def k_attn_s2(q, kc, sinks, pos_ptr, pm, ps, sbuf, scaling, SMAX: tl.constexpr, REP: tl.constexpr,
              BN: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr, CACHE_S: tl.constexpr):
    """Split pass 2: global max m, then the chunk's partial sum of exp(bf16(s - m))."""
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    pos = tl.load(pos_ptr)
    c0, c1 = _split_range(pos, sp, NS, BN)
    r = tl.arange(0, 16)
    rm = r < REP
    d = tl.arange(0, 64)
    sink, m = _global_max(sinks, pm, kvh, r, rm, REP, NS, NSP)
    if CACHE_S:
        qt = tl.zeros([16, 64], tl.bfloat16)
    else:
        qt = tl.load(q + (kvh * REP + r)[:, None] * 64 + d[None, :], mask=rm[:, None], other=0.)
    kb = kc + kvh.to(tl.int64) * SMAX * 64
    sb = sbuf + kvh.to(tl.int64) * REP * SMAX
    ssum = tl.zeros([16], tl.float32)
    for s0 in range(c0, c1, BN):
        j = s0 + tl.arange(0, BN)
        ok = j <= pos
        s = _scores(qt, kb, sb, j, ok, r, rm, d, scaling, SMAX, CACHE_S)
        e = tl.where(ok[None, :], libdevice.exp(_rb(s - m[:, None])), 0.)
        ssum += tl.sum(e, axis=1)
    tl.store(ps + (kvh * NS + sp) * 16 + r, ssum)


@triton.jit
def k_attn_s3(q, kc, vc, sinks, pos_ptr, pm, ps, pacc, sbuf, scaling, SMAX: tl.constexpr,
              REP: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr,
              CACHE_S: tl.constexpr):
    """Split pass 3: global S = exp(bf16(sink - m)) + partial sums (split order), then the
    chunk's fp32 partial P @ V with p = bf16(exp(bf16(s - m)) / S) exactly as k_attn."""
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    pos = tl.load(pos_ptr)
    c0, c1 = _split_range(pos, sp, NS, BN)
    r = tl.arange(0, 16)
    rm = r < REP
    d = tl.arange(0, 64)
    sink, m = _global_max(sinks, pm, kvh, r, rm, REP, NS, NSP)
    ssum = libdevice.exp(_rb(sink - m))
    for i in tl.static_range(NS):
        ssum += tl.load(ps + (kvh * NS + i) * 16 + r)
    if CACHE_S:
        qt = tl.zeros([16, 64], tl.bfloat16)
    else:
        qt = tl.load(q + (kvh * REP + r)[:, None] * 64 + d[None, :], mask=rm[:, None], other=0.)
    kb = kc + kvh.to(tl.int64) * SMAX * 64
    vb = vc + kvh.to(tl.int64) * SMAX * 64
    sb = sbuf + kvh.to(tl.int64) * REP * SMAX
    acc = tl.zeros([16, 64], tl.float32)
    for s0 in range(c0, c1, BN):
        j = s0 + tl.arange(0, BN)
        ok = j <= pos
        s = _scores(qt, kb, sb, j, ok, r, rm, d, scaling, SMAX, CACHE_S)
        e = tl.where(ok[None, :], libdevice.exp(_rb(s - m[:, None])), 0.)
        p = tl.div_rn(e, ssum[:, None]).to(tl.bfloat16)
        vt = tl.load(vb + j[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.)
        acc += tl.dot(p, vt)
    tl.store(pacc + ((kvh * NS + sp) * REP + r)[:, None] * 64 + d[None, :], acc, mask=rm[:, None])


@triton.jit
def k_attn_s4(pacc, out, REP: tl.constexpr, NS: tl.constexpr):
    """Split pass 4: sum the partial accumulators in split order -> bf16 output."""
    kvh = tl.program_id(0)
    r = tl.arange(0, REP)
    d = tl.arange(0, 64)
    o = (r[:, None] * 64 + d[None, :])
    acc = tl.load(pacc + kvh * NS * REP * 64 + o)
    for i in tl.static_range(1, NS):
        acc += tl.load(pacc + (kvh * NS + i) * REP * 64 + o)
    tl.store(out + kvh * REP * 64 + o, acc.to(tl.bfloat16))


class SplitAttn:
    """Preallocated buffers + launcher for split-K decode attention (full layers). One
    instance can serve every full layer of a model (layers run sequentially on one stream).
    launch() allocates nothing, never syncs and never reads pos on the host, so it is CUDA
    graph capturable; the chunking follows the device-side pos at replay time."""

    def __init__(self, ns, smax, dev, bn=64, cache_s=True, num_warps=4):
        assert ns >= 1 and bn % 16 == 0
        self.ns, self.smax, self.bn, self.cache_s, self.num_warps = int(ns), int(smax), int(bn), bool(cache_s), num_warps
        self.nsp = triton.next_power_of_2(self.ns)
        self.pm = torch.empty(NKV * self.ns * 16, dtype=torch.float32, device=dev)
        self.ps = torch.empty(NKV * self.ns * 16, dtype=torch.float32, device=dev)
        self.pacc = torch.empty(NKV * self.ns * REP * HD, dtype=torch.float32, device=dev)
        self.sbuf = torch.empty(NKV * REP * (self.smax if self.cache_s else 1), dtype=torch.bfloat16, device=dev)

    def launch(self, q, kc, vc, sinks, pos_dev, out, scaling):
        assert kc.shape[-2] == self.smax, (kc.shape, self.smax)
        ns, bn, cs, w = self.ns, self.bn, self.cache_s, self.num_warps
        sc = float(scaling)
        k_attn_s1[(NKV, ns)](q, kc, pos_dev, self.pm, self.sbuf, sc, SMAX=self.smax, REP=REP, BN=bn,
                             NS=ns, CACHE_S=cs, num_warps=w)
        k_attn_s2[(NKV, ns)](q, kc, sinks, pos_dev, self.pm, self.ps, self.sbuf, sc, SMAX=self.smax,
                             REP=REP, BN=bn, NS=ns, NSP=self.nsp, CACHE_S=cs, num_warps=w)
        k_attn_s3[(NKV, ns)](q, kc, vc, sinks, pos_dev, self.pm, self.ps, self.pacc, self.sbuf, sc,
                             SMAX=self.smax, REP=REP, BN=bn, NS=ns, NSP=self.nsp, CACHE_S=cs, num_warps=w)
        k_attn_s4[(NKV,)](self.pacc, out, REP=REP, NS=ns, num_warps=4)


def decode_attention(q, kc, vc, sinks, pos_dev, out, scaling, sliding, smax, ring=0, bn=64, split=None):
    """Stand-alone decode attention (what FusedCore.run launches after k_qkv).
    q: [NQ*HD] bf16; kc/vc: [1, NKV, smax, HD] (or [1, NKV, ring, HD] with ring>0, sliding
    layers only); pos_dev: int64 [1] on the device; out: [NQ*HD] bf16. split: a SplitAttn
    (full layers only) -> split-K path; None -> single-program-per-kv-head k_attn."""
    if split is not None:
        assert not sliding and not ring, "split-K is for full-attention layers"
        split.launch(q, kc, vc, sinks, pos_dev, out, scaling)
        return out
    if ring:
        assert sliding and ring >= int(sliding) and kc.shape[-2] == ring, (sliding, ring, kc.shape)
    k_attn[(NKV,)](q, kc, vc, sinks, pos_dev, out, float(scaling), SLIDING=int(sliding or 0),
                   SMAX=(ring or smax), REP=REP, BN=bn, RING=int(ring), num_warps=4)
    return out


@triton.jit
def k_oproj(a, Wo, bo, xbuf, mid, H: tl.constexpr, KD: tl.constexpr, ROWS: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rm = rows < H
    acc = tl.zeros([ROWS, BK], tl.float32)
    for k0 in range(0, KD, BK):
        kk = k0 + tl.arange(0, BK)
        av = tl.load(a + kk).to(tl.float32)
        w = tl.load(Wo + rows[:, None].to(tl.int64) * KD + kk[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        acc += w * av[None, :]
    y = _rb(tl.sum(acc, axis=1) + tl.load(bo + rows, mask=rm, other=0.).to(tl.float32))
    x = tl.load(xbuf + rows, mask=rm, other=0.).to(tl.float32)
    tl.store(mid + rows, _rb(x + y).to(tl.bfloat16), mask=rm)


@triton.jit
def k_router(mid, post_w, eps, Wr, br, logits, hn_o, bx, pack, H: tl.constexpr, HP: tl.constexpr,
             ROWS: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, HP)
    msk = offs < H
    x = tl.load(mid + offs, mask=msk, other=0.).to(tl.float32)
    var = tl.sum(x * x, axis=0) * (1.0 / H)
    r = libdevice.rsqrt(var + eps)
    if pid == 0:
        hn = _rb(tl.load(post_w + offs, mask=msk, other=0.).to(tl.float32) * (x * r))
        tl.store(hn_o + offs, hn.to(tl.bfloat16), mask=msk)
        tl.store(bx + offs, hn.to(tl.bfloat16), mask=msk)
        tl.store(pack + offs, hn, mask=msk)
    rows = pid * ROWS + tl.arange(0, ROWS)
    acc = tl.zeros([ROWS, BK], tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < H
        xc = tl.load(mid + kk, mask=km, other=0.).to(tl.float32)
        hc = _rb(tl.load(post_w + kk, mask=km, other=0.).to(tl.float32) * (xc * r))
        w = tl.load(Wr + rows[:, None] * H + kk[None, :], mask=km[None, :], other=0.).to(tl.float32)
        acc += w * hc[None, :]
    lg = _rb(tl.sum(acc, axis=1) + tl.load(br + rows).to(tl.float32))
    tl.store(logits + rows, lg)


@triton.jit
def _emit(j: tl.constexpr, ij, ej, ssum, slot_tab, base, sc_o, idx_o, bslots, bgids, bw, pack, H: tl.constexpr):
    sc = _rb(tl.div_rn(ej, ssum))
    g = base + ij.to(tl.int64)
    sl = tl.load(slot_tab + g)
    hit = sl >= 0
    tl.store(sc_o + j, sc.to(tl.bfloat16))
    tl.store(idx_o + j, ij.to(tl.int64))
    tl.store(bslots + j, tl.where(hit, sl, 0))
    tl.store(bgids + j, g)
    tl.store(bw + j, tl.where(hit, sc, 0.).to(tl.bfloat16))
    tl.store(pack + H + j, sc)
    tl.store(pack + H + 4 + j, ij.to(tl.float32))
    tl.store(pack + H + 8 + j, tl.where(hit, 1.0, 0.0))


@triton.jit
def k_topk(logits, slot_tab, base, sc_o, idx_o, bslots, bgids, bw, pack, H: tl.constexpr, NE: tl.constexpr,
           NM: tl.constexpr = 0):
    """NM=1: also write the ids of ranks 5-8 to pack[H+12 .. H+16) (the pack must then hold H+16
    floats). NM=0 (default): exactly the previous kernel, pack[H+12:] untouched."""
    i = tl.arange(0, NE)
    v = tl.load(logits + i)
    v0 = tl.max(v, 0)
    i0 = tl.argmax(v, 0)
    v = tl.where(i == i0, float("-inf"), v)
    v1 = tl.max(v, 0)
    i1 = tl.argmax(v, 0)
    v = tl.where(i == i1, float("-inf"), v)
    v2 = tl.max(v, 0)
    i2 = tl.argmax(v, 0)
    v = tl.where(i == i2, float("-inf"), v)
    v3 = tl.max(v, 0)
    i3 = tl.argmax(v, 0)
    e0 = libdevice.exp(v0 - v0)
    e1 = libdevice.exp(v1 - v0)
    e2 = libdevice.exp(v2 - v0)
    e3 = libdevice.exp(v3 - v0)
    ssum = e0 + e1 + e2 + e3
    _emit(0, i0, e0, ssum, slot_tab, base, sc_o, idx_o, bslots, bgids, bw, pack, H)
    _emit(1, i1, e1, ssum, slot_tab, base, sc_o, idx_o, bslots, bgids, bw, pack, H)
    _emit(2, i2, e2, ssum, slot_tab, base, sc_o, idx_o, bslots, bgids, bw, pack, H)
    _emit(3, i3, e3, ssum, slot_tab, base, sc_o, idx_o, bslots, bgids, bw, pack, H)
    if NM:
        # NEAR MISSES: ranks 5-8 by logit, ids only, to pack[H+12 .. H+16). The top-4 selection, its
        # order, its softmax and every store above are unchanged; these four extra argmax rounds run
        # on the already-masked copy of the logits. Used for admission candidates / traces (exact).
        v = tl.where(i == i3, float("-inf"), v)
        i4 = tl.argmax(v, 0)
        v = tl.where(i == i4, float("-inf"), v)
        i5 = tl.argmax(v, 0)
        v = tl.where(i == i5, float("-inf"), v)
        i6 = tl.argmax(v, 0)
        v = tl.where(i == i6, float("-inf"), v)
        i7 = tl.argmax(v, 0)
        tl.store(pack + H + 12, i4.to(tl.float32))
        tl.store(pack + H + 13, i5.to(tl.float32))
        tl.store(pack + H + 14, i6.to(tl.float32))
        tl.store(pack + H + 15, i7.to(tl.float32))


class FusedCore:
    """Per-layer launcher. Concatenates each layer's q/k/v weights and biases into
    one buffer and re-points the Parameters at views of it (VRAM-neutral; the
    eager core keeps working on the views). Must run BEFORE any graph capture."""

    def __init__(self, cg, dev, qkv_bk=128, qkv_warps=4, o_rows=16, o_bk=256, o_warps=4,
                 r_rows=8, r_bk=256, attn_bn=64, ring=0, split_k=1, split_bn=64, split_cache_s=True,
                 near_miss=False):
        """ring=R>0: REPLACES every sliding layer's lg.k / lg.v with [1, NKV, R, HD] zeros
        (rolling cache, slot = pos % R; R >= the layer's window). lg.ring is set on every
        layer in every mode (R for sliding layers when ring=R>0, 0 otherwise, including all
        layers when ring=0). Construct BEFORE any CUDA-graph capture. After
        this the eager reference path (_LayerGraph._compute / GptOssCoreGraphs.
        load_prefill_kv) is NOT valid for sliding layers (absolute-position writes into a
        ring would index out of bounds / land in the wrong slot); prompt processing must use
        prefill_attention(..., k_new=, v_new=, ring=R) + ring_write().
        split_k=NS>1: full-attention layers use split-K decode attention (SplitAttn, grid
        (NKV, NS), buffers preallocated here and shared by all full layers).
        Defaults (ring=0, split_k=1) launch exactly the previous kernels."""
        self.cg, self.dev = cg, dev
        self.ring, self.split_k = int(ring), int(split_k)
        self.near_miss = bool(near_miss)     # k_topk also writes router ranks 5-8 to pack[H+12:H+16) (pack >= H+16)
        self.cfg = dict(qkv_bk=qkv_bk, qkv_warps=qkv_warps, o_rows=o_rows, o_bk=o_bk, o_warps=o_warps,
                        r_rows=r_rows, r_bk=r_bk, attn_bn=attn_bn, ring=self.ring, split_k=self.split_k,
                        split_bn=split_bn, split_cache_s=bool(split_cache_s), near_miss=self.near_miss)
        self.W, self.B = [], []
        for lg in cg.layers:
            a = lg.layer.self_attn
            with torch.no_grad():
                W = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).contiguous()
                B = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).contiguous()
                a.q_proj.weight.data = W[:NQ * HD]
                a.k_proj.weight.data = W[NQ * HD:(NQ + NKV) * HD]
                a.v_proj.weight.data = W[(NQ + NKV) * HD:]
                a.q_proj.bias.data = B[:NQ * HD]
                a.k_proj.bias.data = B[NQ * HD:(NQ + NKV) * HD]
                a.v_proj.bias.data = B[(NQ + NKV) * HD:]
            self.W.append(W)
            self.B.append(B)
        for lg in cg.layers:                     # lg.ring on EVERY layer, in every mode (0 = linear)
            if self.ring and lg.sliding:
                assert self.ring >= int(lg.sliding), (self.ring, lg.sliding)
                lg.k = torch.zeros(1, NKV, self.ring, HD, dtype=torch.bfloat16, device=dev)
                lg.v = torch.zeros_like(lg.k)
                lg.ring = self.ring
            else:
                lg.ring = 0
        torch.cuda.empty_cache()
        self.xbuf = torch.zeros(H, dtype=torch.bfloat16, device=dev)
        self.q = torch.zeros(NQ * HD, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(NQ * HD, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(NE, dtype=torch.float32, device=dev)
        self.split = None
        if self.split_k > 1:
            full = [lg.smax for lg in cg.layers if not lg.sliding]
            if full:
                assert len(set(full)) == 1, set(full)
                self.split = SplitAttn(self.split_k, full[0], dev, bn=split_bn, cache_s=split_cache_s)

    def layer_ring(self, L):
        """Ring size used for layer L (0 = linear cache)."""
        return self.ring if (self.ring and self.cg.layers[L].sliding) else 0

    def snapshot_rings(self):
        """Copy of every sliding layer's ring (18 x 2 x 8 x R x 64 x 2 B = 4.5 MiB at R=128).
        Take it where the next prompt will resume (e.g. right after a prefill, before decode)
        so a later prefix reuse from exactly that position is valid (see ring_reuse_ok)."""
        return [(lg.k.clone(), lg.v.clone()) if (self.ring and lg.sliding) else None for lg in self.cg.layers]

    def restore_rings(self, snap):
        """In-place restore (pointers unchanged, so captured graphs stay valid)."""
        for lg, kv in zip(self.cg.layers, snap):
            if kv is not None:
                lg.k.copy_(kv[0])
                lg.v.copy_(kv[1])

    def run(self, L, mid, g_out, c_out, slot_tab, bx, bslots, bgids, bw, pack, kc=None, vc=None,
            hn_o=None, sc_o=None, idx_o=None):
        """One layer. Reads mid (+ g_out + c_out for L>0); writes K/V at pos, mid <- h_mid,
        h_norm (hn_o, bx, pack[:H]), router outputs (sc_o, idx_o, pack, B inputs).
        kc/vc overrides must use the layer's layout: contiguous bf16 [1, NKV, ring, HD] for a
        sliding layer when ring>0, else [1, NKV, lg.smax, HD]; anything else raises
        AssertionError (host-side shape check). Nothing here depends on pos on the host (graph-capturable)."""
        cg, c = self.cg, self.cfg
        lg = cg.layers[L]
        ly = lg.layer
        a = ly.self_attn
        kc = lg.k if kc is None else kc
        vc = lg.v if vc is None else vc
        hn_o = lg.h_norm if hn_o is None else hn_o
        sc_o = lg.r_scores if sc_o is None else sc_o
        idx_o = lg.r_idx if idx_o is None else idx_o
        ring = self.layer_ring(L)
        rows = ring or lg.smax                       # cache rows per kv head
        # the kernels address kc/vc as contiguous [1, NKV, rows, HD] bf16 (slot pos % ring or row
        # pos): refuse anything else (e.g. a linear [1, NKV, smax, HD] override on a ring layer,
        # which would otherwise be written/read as a ring silently). Host-side metadata only: no
        # sync, no pos dependence, safe during graph capture.
        for t, nm in ((kc, "kc"), (vc, "vc")):
            assert (tuple(t.shape) == (1, NKV, rows, HD) and t.dtype == torch.bfloat16 and t.is_contiguous()), (
                f"layer {L}: {nm} must be a contiguous bf16 [1, {NKV}, {rows}, {HD}] "
                f"({'ring' if ring else 'linear'} layout), got {tuple(t.shape)} {t.dtype} stride {t.stride()}")
        k_qkv[(2 * (NQ + 2 * NKV),)](mid, g_out, c_out, self.xbuf, ly.input_layernorm.weight, self.W[L], self.B[L],
                                     cg.pos, cg.inv_freq, kc, vc, self.q,
                                     float(ly.input_layernorm.variance_epsilon), float(cg.att_scale),
                                     SMAX=rows, FIRST=(L == 0), H=H, HP=HP, BK=c["qkv_bk"], NQ=NQ, NKV=NKV,
                                     RING=ring, num_warps=c["qkv_warps"])
        if self.split is not None and not lg.sliding:
            self.split.launch(self.q, kc, vc, a.sinks, cg.pos, self.att, lg.scaling)
        else:
            k_attn[(NKV,)](self.q, kc, vc, a.sinks, cg.pos, self.att, float(lg.scaling),
                           SLIDING=int(lg.sliding or 0), SMAX=rows, REP=REP, BN=c["attn_bn"], RING=ring,
                           num_warps=4)
        k_oproj[(triton.cdiv(H, c["o_rows"]),)](self.att, a.o_proj.weight, a.o_proj.bias, self.xbuf, mid,
                                                H=H, KD=NQ * HD, ROWS=c["o_rows"], BK=c["o_bk"],
                                                num_warps=c["o_warps"])
        r = ly.mlp.router
        k_router[(NE // c["r_rows"],)](mid, ly.post_attention_layernorm.weight,
                                       float(ly.post_attention_layernorm.variance_epsilon),
                                       r.weight, r.bias, self.logits, hn_o, bx, pack,
                                       H=H, HP=HP, ROWS=c["r_rows"], BK=c["r_bk"], num_warps=4)
        k_topk[(1,)](self.logits, slot_tab, L * NE, sc_o, idx_o, bslots, bgids, bw, pack,
                     H=H, NE=NE, NM=int(self.near_miss), num_warps=4)


@triton.jit
def k_fetch(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    """Zero-copy fetch: SMs read a pinned (UVA-mapped) host buffer into device memory,
    so the per-layer c_out transfer never queues behind admission copies on the copy
    engine."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(dst + o, tl.load(src + o, mask=m), mask=m)


def fetch(src_pinned, dst):
    k_fetch[(triton.cdiv(dst.numel(), 1024),)](src_pinned, dst, N=dst.numel(), BLOCK=1024, num_warps=4)


@triton.jit
def k_bell(bell, POLL_CAP: tl.constexpr):
    """DOORBELL spin node (server.py --doorbell 1): the FIRST node of every graph that consumes the CPU experts'
    output, ahead of k_fetch. Spins on a word in pinned host memory until the host rings it (see doorbell.py for
    the word layout and the protocol), so the graph can be launched BEFORE the CPU kernel that feeds it.

      bell[0] FLAG (host writes)  bell[1] WANT (host writes before the launch)  bell[2] timeout, microseconds
      bell[3] FAILS (bumped when the spin gives up)  bell[4] polls of this spin  bell[5] spin time, microseconds

    Exit when (int32)(FLAG - WANT) >= 0 (wrapping compare: sequence numbers never need a reset), or after
    bell[2] microseconds of %globaltimer, or after POLL_CAP polls (a second bound in case the timer misbehaves):
    then FAILS is incremented and the graph runs on (on stale data: nothing in it can fault, it only computes floats
    and indexes with router ids < 128); the host sees FAILS change after the graph's router event, synchronizes,
    and re-runs the token in the classic order (server._bell_recover; doorbell.py). The flag is read with
    ld.volatile (relaxed, system scope): it is re-fetched from host memory every
    iteration. The data behind the flag (out_pin) is read by the NEXT node, k_fetch, in a separate kernel:
    the kernel boundary orders it after this spin, exactly as it used to be ordered after the host's launch.
    One warp, one program: the poll traffic is a single 4-byte PCIe read per iteration."""
    want = tl.load(bell + 1, volatile=True)
    limit = tl.load(bell + 2, volatile=True).to(tl.int64) * 1000
    t0 = globaltimer()
    flag = tl.load(bell, volatile=True)
    n = flag * 0
    while tl.condition((((flag - want) < 0) & ((globaltimer() - t0) < limit)) & (n < POLL_CAP), disable_licm=True):
        flag = tl.load(bell, volatile=True)
        n += 1
    tl.store(bell + 4, n)
    tl.store(bell + 5, ((globaltimer() - t0) // 1000).to(tl.int32))
    if (flag - want) < 0:
        fails = tl.load(bell + 3, volatile=True)
        tl.store(bell + 3, fails + 1)


BELL_POLL_CAP = 1 << 20      # ~1 s of polls at the MEASURED 1.1 polls/us (a second bound, below WDDM's ~2 s TDR; doorbell.MAX_POLL_CAP)


def bell_wait(bell_pin, poll_cap=BELL_POLL_CAP):
    k_bell[(1,)](bell_pin, POLL_CAP=poll_cap, num_warps=1)


@triton.jit
def k_prefill_attn(Q, Kc, Vc, Kn, Vn, sk_t, sk_h, sv_t, sv_h, sinks, O, p0, T, scaling, SLIDING: tl.constexpr,
                   SMAX: tl.constexpr, REP: tl.constexpr, NQ: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, RING: tl.constexpr = 0):
    """Prompt-processing attention (flash-style, online softmax) with gpt-oss sinks,
    causal + optional sliding window. Q/O: [T, NQ, 64] for absolute positions
    p0..p0+T-1. The sink joins the softmax denominator as an initial running max/sum
    (m = sink, l = 1).
      RING=0: K/V read from the static decode cache [1, NKV, SMAX, 64], which already
              holds positions < p0+T (Kn/Vn unused).
      RING>0: (sliding layers, RING >= SLIDING) keys j < p0 come from the rolling cache
              [1, NKV, RING, 64] (slot j % RING, holding positions p0-RING..p0-1), keys
              j >= p0 from the block's own K/V Kn/Vn [T, NKV, 64] (row/head strides
              sk_t/sk_h and sv_t/sv_h, last dim contiguous). Same blocks / order / arithmetic as RING=0,
              so the output is bit-identical to RING=0 on a linear cache with the same
              values. Keys the ring cannot hold (j <= p0-RING) are always outside every
              query's window, so whatever their slot contains is masked to -inf (finite
              contents -> p = 0 exactly); the ring must therefore never hold inf/NaN."""
    mb = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // REP
    offs = mb * BM + tl.arange(0, BM)
    mm = offs < T
    qpos = p0 + offs
    d = tl.arange(0, 64)
    q = tl.load(Q + (offs[:, None] * NQ + h) * 64 + d[None, :], mask=mm[:, None], other=0.)
    sink = tl.load(sinks + h).to(tl.float32)
    m_i = tl.zeros([BM], tl.float32) + sink
    l_i = tl.zeros([BM], tl.float32) + 1.0
    acc = tl.zeros([BM, 64], tl.float32)
    hi = tl.minimum(p0 + mb * BM + BM, p0 + T)
    if SLIDING > 0:
        lo = tl.maximum(p0 + mb * BM - SLIDING + 1, 0)
    else:
        lo = p0 * 0
    lo = (lo // BN) * BN
    if RING > 0:
        kb = Kc + kvh.to(tl.int64) * RING * 64
        vb = Vc + kvh.to(tl.int64) * RING * 64
        knb = Kn + kvh.to(tl.int64) * sk_h
        vnb = Vn + kvh.to(tl.int64) * sv_h
    else:
        kb = Kc + kvh.to(tl.int64) * SMAX * 64
        vb = Vc + kvh.to(tl.int64) * SMAX * 64
    for s0 in range(lo, hi, BN):
        j = s0 + tl.arange(0, BN)
        jm = j < hi
        if RING > 0:
            # ONE masked load per tile from a per-row selected pointer (ring slot for j < p0,
            # block row j - p0 otherwise). Two masked loads + tl.where would double the
            # pipelined shared-memory buffers (OutOfResources at BM/BN 64/128, 128/128, 256/64,
            # tilings the linear kernel handles) and cost ~1.8x time at 64/64. The unselected
            # pointer (e.g. a negative block row for j < p0) is never dereferenced.
            old = (j < p0)[:, None]
            jr = (j % RING)[:, None] * 64 + d[None, :]
            jn = (j - p0)[:, None].to(tl.int64)
            k = tl.load(tl.where(old, kb + jr, knb + jn * sk_t + d[None, :]), mask=jm[:, None], other=0.)
        else:
            k = tl.load(kb + j[:, None] * 64 + d[None, :], mask=jm[:, None], other=0.)
        s = tl.dot(q, tl.trans(k)) * scaling
        ok = (j[None, :] <= qpos[:, None]) & jm[None, :]
        if SLIDING > 0:
            ok = ok & (j[None, :] > qpos[:, None] - SLIDING)
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if RING > 0:
            v = tl.load(tl.where(old, vb + jr, vnb + jn * sv_t + d[None, :]), mask=jm[:, None], other=0.)
        else:
            v = tl.load(vb + j[:, None] * 64 + d[None, :], mask=jm[:, None], other=0.)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + (offs[:, None] * NQ + h) * 64 + d[None, :], o.to(tl.bfloat16), mask=mm[:, None])


def prefill_attention(q, kc, vc, sinks, p0, scaling, sliding, smax, BM=64, BN=64, k_new=None, v_new=None,
                      ring=0):
    """q: [T, NQ, 64] bf16 (contiguous) for positions p0..p0+T-1 -> o [T, NQ, 64].
    ring=0: kc/vc are the linear decode cache [1, NKV, smax, 64] ALREADY holding the block's
            K/V at p0..p0+T-1 (k_new/v_new ignored).
    ring=R: sliding layers only; kc/vc are the rolling cache [1, NKV, R, 64] holding
            positions p0-R..p0-1 (those that exist) and k_new/v_new [T, NKV, 64] (last dim
            contiguous) are the block's own K/V. T may exceed R. The ring is NOT updated
            here: call ring_write(kc, vc, k_new, v_new, p0, R) after attention."""
    T = q.shape[0]
    o = torch.empty_like(q)
    if ring:
        assert sliding and int(ring) >= int(sliding), (sliding, ring)
        assert kc.shape[-2] == ring and vc.shape[-2] == ring, (kc.shape, ring)
        assert k_new is not None and v_new is not None and k_new.shape == v_new.shape == (T, NKV, HD)
        assert k_new.stride(-1) == 1 and v_new.stride(-1) == 1, (k_new.stride(), v_new.stride())
        kn, vn, st = k_new, v_new, (k_new.stride(0), k_new.stride(1), v_new.stride(0), v_new.stride(1))
    else:
        kn, vn, st = kc, vc, (0, 0, 0, 0)
    k_prefill_attn[(triton.cdiv(T, BM), NQ)](q, kc, vc, kn, vn, *st, sinks, o, p0, T, float(scaling),
                                            SLIDING=int(sliding or 0), SMAX=(ring or smax), REP=REP, NQ=NQ,
                                            BM=BM, BN=BN, RING=int(ring), num_warps=4)
    return o


def ring_write(kc, vc, k_new, v_new, p0, ring):
    """Store the last min(T, ring) positions of a block (positions p0..p0+T-1, K/V given as
    [T, NKV, 64]) into their ring slots (slot = position % ring). Call AFTER the block's
    attention. At most two contiguous slice copies; no allocation, no host sync."""
    T = k_new.shape[0]
    n = min(T, ring)
    a = T - n                                  # block-relative index of the first kept position
    s0 = (p0 + a) % ring
    first = min(n, ring - s0)
    kc[0, :, s0:s0 + first] = k_new[a:a + first].transpose(0, 1)
    vc[0, :, s0:s0 + first] = v_new[a:a + first].transpose(0, 1)
    if first < n:
        kc[0, :, :n - first] = k_new[a + first:].transpose(0, 1)
        vc[0, :, :n - first] = v_new[a + first:].transpose(0, 1)


def ring_reuse_ok(n_written, c, ring, window=128):
    """Prefix reuse check for a rolling cache. The ring was last written with positions
    0..n_written-1, so it holds max(0, n_written-ring)..n_written-1. A block starting at
    p0=c (prefix reuse) needs the window before it, c-window+1..c-1. Valid iff c <= n_written
    and (n_written <= ring or n_written - c <= ring - window + 1). With ring == window == 128
    that is c >= n_written - 1: after a turn decoded past c the sliding layers can NOT reuse
    the prefix (their K/V for c-127..c-1 were overwritten), and because those K/V depend on
    ~18 x 127 earlier positions through the stacked sliding layers they cannot be patched by
    re-prefilling a short tail either. Remedies (caller side): a larger ring (slack R - W
    tokens of rewind for 2*NKV*HD*2 B per extra slot per sliding layer, e.g. R=1024 -> 36 MiB
    for all 18 layers), a snapshot of the 18 rings (4.5 MiB at R=128) taken at a position the
    next prompt will share (e.g. the end of each prefill; then reuse from exactly that
    position), or a full re-prefill."""
    if c > n_written:
        return False
    if n_written <= ring:
        return True
    return n_written - c <= ring - window + 1
