"""COMPONENT A tests: rolling (ring) K/V cache for sliding-window layers + split-K decode
attention for full-attention layers (fused_core.py). Synthetic data only - the model is
NOT loaded. q/k/v ~ N(0,1) bf16, sinks ~ N(0,1) bf16, scaling 1/8 (head_dim 64).

  (a) decode, sliding layer: ring (R=128) vs linear cache, incremental pos 0..999 (7+ wraps)
      and random pos up to 16383 -> torch.equal required
  (b) decode, full layer: split-K (NS in 4/8/16/32) vs single k_attn at pos in
      {0,1,63,64,65,1000,4095,8000,16383} -> max abs diff, fraction of differing bf16
      outputs, max ulp distance, and both vs an fp64 reference; score-cache on/off -> equal
      (strict check, holds for the default seed only - see NOTE in (b)) + the seed-independent
      invariant: partial maxes bit-identical, partial sums within the fp32 reordering bound,
      outputs = rounding flips only
  (c) prefill, sliding layer: ring mode (keys < p0 from the ring, >= p0 from k_new/v_new)
      vs linear kernel on a linear cache holding the same values -> torch.equal; ring_write
      contents; a chained multi-block prompt + decode continuation -> torch.equal
  (c2) prefill ring mode at EVERY BM/BN tiling the linear kernel accepts (incl. 64/128, 128/128,
      256/64, which used to hit OutOfResources in ring mode): compiles, torch.equal to linear on
      all (c) cases, compiled shared memory <= the linear kernel's
  (d) k_qkv: ring store (slot pos % 128) vs linear store with random weights -> the written
      K/V (and q, xbuf) equal, no other ring slot touched
  (e) CUDA graphs: split path captured once, replayed with pos changed between replays ==
      eager; FusedCore(ring=128, split_k=16) captured per layer on a synthetic 2-layer
      model (sliding + full) vs FusedCore() default eager: sliding layer bit-identical at
      every step, full layer within split-K tolerance (top-4 set at every step: strict check,
      default seed only - see NOTE in (e2); every disagreement must be a near-tie)
  (h) FusedCore API guards: lg.ring present on every layer in every mode (== layer_ring(L));
      wrong-layout kc/vc overrides (linear cache on a ring layer, ring-sized on a full layer,
      non-contiguous, fp32) raise AssertionError before any launch and write nothing;
      correct-layout overrides still work and equal running on the layer's own cache
  (f) timing (CUDA events, INDICATIVE): attention per layer single vs split at pos
      1k/4k/8k/16k (graph-replayed), whole FusedCore layer default vs ring+split, and
      sliding-layer prompt attention (T=2048 block at p0=4096, eager) linear vs ring mode

Usage:  F:\\AI\\Neural\\.venv\\Scripts\\python.exe dev\\test_attention.py [--no-timing] [--json out.json] [--seed N]
(prepend F:\\AI\\Neural\\third_party\\tools\\w64devkit\\bin to PATH first). Exit code 1 on any
exactness failure.
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import torch

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S))
import fused_core as fc                                                      # noqa: E402
from fused_core import H, HD, NE, NKV, NQ, REP                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-timing", action="store_true")
ap.add_argument("--json", default=None)
ap.add_argument("--timing-reps", type=int, default=30)
ap.add_argument("--seed", type=int, default=1234, help="synthetic-data generator seed")
A = ap.parse_args()

dev = torch.device("cuda")
bf = torch.bfloat16
SC = 0.125
W = 128                       # sliding window
RING = 128
SMAX = 16384
RES = {}
FAILS = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def rnd(*shape, g=None, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=g) * scale).to(bf)


def bf16_ulp_dist(a, b):
    """max distance in bf16 ulps (monotone int mapping of the bit patterns)."""
    def key(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return int((key(a) - key(b)).abs().max())


def ref64(q, kc, vc, sinks, pos, lo=0):
    """exact-math fp64 sink attention for one decode query over keys lo..pos."""
    qd = q.view(NKV, REP, HD).double()
    k = kc[0, :, lo:pos + 1].double()                                # [8, n, 64]
    v = vc[0, :, lo:pos + 1].double()
    s = torch.einsum("hrd,hnd->hrn", qd, k) * SC
    sk = sinks.view(NKV, REP, 1).double()
    m = torch.maximum(s.max(-1, keepdim=True).values, sk)
    e = torch.exp(s - m)
    den = e.sum(-1, keepdim=True) + torch.exp(sk - m)
    return torch.einsum("hrn,hnd->hrd", e / den, v).reshape(-1)


torch.manual_seed(0)
G = torch.Generator(device=dev)
G.manual_seed(A.seed)
t_start = time.perf_counter()

# ------------------------------------------------------------------ shared 16k linear cache
Kf = rnd(1, NKV, SMAX, HD, g=G)
Vf = rnd(1, NKV, SMAX, HD, g=G)
sinks = rnd(NQ, g=G)
pos_dev = torch.zeros(1, dtype=torch.int64, device=dev)

# ================================================================== (a) decode ring vs linear
print("(a) decode sliding window: ring vs linear", flush=True)
with torch.inference_mode():
    Kr = rnd(1, NKV, RING, HD, g=G, scale=3.0)          # garbage in never-written slots
    Vr = rnd(1, NKV, RING, HD, g=G, scale=3.0)
    ol = torch.empty(NQ * HD, dtype=bf, device=dev)
    orr = torch.empty_like(ol)
    bad, n = [], 0
    for pos in range(0, 1000):                           # incremental decode: write pos, attend
        Kr[0, :, pos % RING] = Kf[0, :, pos]
        Vr[0, :, pos % RING] = Vf[0, :, pos]
        q = rnd(NQ * HD, g=G)
        pos_dev.fill_(pos)
        fc.decode_attention(q, Kf, Vf, sinks, pos_dev, ol, SC, W, SMAX)
        fc.decode_attention(q, Kr, Vr, sinks, pos_dev, orr, SC, W, SMAX, ring=RING)
        n += 1
        if not torch.equal(ol, orr):
            bad.append(pos)
    # random far positions: fill the ring with the window of pos, other slots stay stale
    far = [4095, 4096, 8000, 12345, 16255, 16256, 16382, 16383]
    for pos in far:
        js = torch.arange(pos - RING + 1, pos + 1, device=dev)
        Kr[0, :, js % RING] = Kf[0, :, js]
        Vr[0, :, js % RING] = Vf[0, :, js]
        q = rnd(NQ * HD, g=G)
        pos_dev.fill_(pos)
        fc.decode_attention(q, Kf, Vf, sinks, pos_dev, ol, SC, W, SMAX)
        fc.decode_attention(q, Kr, Vr, sinks, pos_dev, orr, SC, W, SMAX, ring=RING)
        n += 1
        if not torch.equal(ol, orr):
            bad.append(pos)
    # sanity: the sliding kernel output is the real attention (fp64 reference, window only)
    err64 = (ol.double() - ref64(q, Kf, Vf, sinks, 16383, lo=16383 - W + 1)).abs().max().item()
    RES["a"] = {"cases": n, "not_equal": bad, "pos16383_maxabs_vs_fp64": err64}
    check("a: ring decode == linear decode (torch.equal)", not bad, f"{n} positions, mismatches={bad[:10]}")
    check("a: sliding output matches fp64 reference", err64 < 2e-2, f"max|out-ref64|={err64:.3g}")
    del Kr, Vr

# ================================================================== (b) split-K vs single
print("(b) decode full attention: split-K vs single program per kv head", flush=True)
POSB = [0, 1, 63, 64, 65, 1000, 4095, 8000, 16383]
NSS = [4, 8, 16, 32]
SPL = {ns: fc.SplitAttn(ns, SMAX, dev) for ns in NSS}
SPL_NC = {ns: fc.SplitAttn(ns, SMAX, dev, cache_s=False) for ns in NSS}
RES["b"] = []
with torch.inference_mode():
    o1 = torch.empty(NQ * HD, dtype=bf, device=dev)
    o2 = torch.empty_like(o1)
    o3 = torch.empty_like(o1)
    cache_equal = True
    worst_frac, worst_abs = 0.0, 0.0
    for pos in POSB:
        q = rnd(NQ * HD, g=G)
        pos_dev.fill_(pos)
        fc.decode_attention(q, Kf, Vf, sinks, pos_dev, o1, SC, 0, SMAX)
        r64 = ref64(q, Kf, Vf, sinks, pos)
        e_single = (o1.double() - r64).abs().max().item()
        me_single = (o1.double() - r64).abs().mean().item()
        oscale_ulp = 2.0 ** (math.floor(math.log2(o1.float().abs().max().item())) - 7)   # bf16 ulp at output scale
        for ns in NSS:
            fc.decode_attention(q, Kf, Vf, sinks, pos_dev, o2, SC, 0, SMAX, split=SPL[ns])
            fc.decode_attention(q, Kf, Vf, sinks, pos_dev, o3, SC, 0, SMAX, split=SPL_NC[ns])
            ce = torch.equal(o2, o3)
            cache_equal &= ce
            # what cache_s on/off MUST share (seed-independent): the pass-1 partial maxes (max is exact,
            # scores are the same bf16 values) -> identical global m; the partial sums may differ only by
            # fp32 summation order (tl.sum's reduction order follows the register layout of its input:
            # loaded scores vs the dot-output layout). Rigorous bound for two orderings of the same n
            # positive fp32 terms: |a - b| <= 2 (n - 1) 2^-24 * sum, n = keys per split
            pm_eq = torch.equal(SPL[ns].pm, SPL_NC[ns].pm)
            psa = SPL[ns].ps.view(NKV, ns, 16)[:, :, :REP].double()
            psb = SPL_NC[ns].ps.view(NKV, ns, 16)[:, :, :REP].double()
            ps_rel = ((psa - psb).abs() / psa.abs().clamp_min(1e-30)).max().item()
            dcn = (o2.float() - o3.float()).abs()
            dif = (o1.float() - o2.float()).abs()
            frac = (o1 != o2).float().mean().item()
            row = {"pos": pos, "ns": ns, "max_abs_diff": dif.max().item(), "frac_diff": frac,
                   "n_diff": int((o1 != o2).sum()), "max_ulp": bf16_ulp_dist(o1, o2),
                   "err_vs_fp64_single": e_single, "err_vs_fp64_split": (o2.double() - r64).abs().max().item(),
                   "mean_err_vs_fp64_single": me_single, "mean_err_vs_fp64_split": (o2.double() - r64).abs().mean().item(),
                   "max_abs_diff_in_output_scale_ulps": dif.max().item() / oscale_ulp,
                   "score_cache_equal": ce, "blocks": (pos + 64) // 64,
                   "one_block_per_split": (pos + 64) // 64 <= ns,
                   "cache_nocache_pm_equal": pm_eq, "cache_nocache_ps_max_rel": ps_rel,
                   "cache_nocache_ps_bound": 2 * (-(-((pos + 64) // 64) // ns) * 64 - 1) * 2.0 ** -24,
                   "cache_nocache_n_diff": int((o2 != o3).sum()),
                   "cache_nocache_output_scale_ulps": dcn.max().item() / oscale_ulp}
            RES["b"].append(row)
            worst_frac, worst_abs = max(worst_frac, frac), max(worst_abs, row["max_abs_diff"])
            print(f"    pos={pos:5d} NS={ns:2d} blocks={row['blocks']:3d}  diff: max_abs={row['max_abs_diff']:.3g} "
                  f"n={row['n_diff']:4d}/4096 ({frac:.4%}) max_ulp={row['max_ulp']} "
                  f"(={row['max_abs_diff_in_output_scale_ulps']:.2g} ulp at output scale)  |  vs fp64 max/mean: "
                  f"single {e_single:.3g}/{me_single:.3g} split {row['err_vs_fp64_split']:.3g}/"
                  f"{row['mean_err_vs_fp64_split']:.3g}", flush=True)
    # NOTE (found by running other --seed values): this bit-equality is NOT an invariant of the code -
    # it holds for the default seed, but seeds 1..5 each give 1-7 (pos, NS) cases with a few differing
    # outputs, because only the fp32 summation order of the partial sums differs (see the next check).
    # Kept unchanged (not loosened); the seed-independent invariant is checked right below.
    check("b: split score-cache on == off (torch.equal, all cases)", cache_equal)
    cn_pm = all(r["cache_nocache_pm_equal"] for r in RES["b"])
    cn_ps = max(r["cache_nocache_ps_max_rel"] for r in RES["b"])
    cn_ps_ratio = max(r["cache_nocache_ps_max_rel"] / r["cache_nocache_ps_bound"] for r in RES["b"])
    cn_ulp = max(r["cache_nocache_output_scale_ulps"] for r in RES["b"])
    cn_frac = max(r["cache_nocache_n_diff"] for r in RES["b"]) / (NQ * HD)
    RES["b_cache_nocache"] = {"pm_equal_all": cn_pm, "ps_max_rel": cn_ps, "ps_worst_frac_of_bound": cn_ps_ratio,
                              "worst_output_scale_ulps": cn_ulp,
                              "worst_frac": cn_frac,
                              "n_cases_bitexact": sum(r["cache_nocache_n_diff"] == 0 for r in RES["b"])}
    check("b: score-cache on vs off: partial maxes bit-identical, partial sums within the fp32 reordering bound, "
          "outputs = rounding flips only (<=2 ulp at output scale, <=5% of outputs)",
          cn_pm and cn_ps_ratio <= 1.0 and cn_ulp <= 2.0 and cn_frac <= 0.05,
          f"pm equal={cn_pm}, max rel ps diff={cn_ps:.3g} (= {cn_ps_ratio:.3g} x bound), worst {cn_ulp:.2f} "
          f"output-scale ulps, worst frac "
          f"{cn_frac:.4%}, bit-identical in {RES['b_cache_nocache']['n_cases_bitexact']}/{len(RES['b'])} cases")
    # criterion: differences are fp32-reassociation rounding flips only: every differing output
    # within 2 bf16 ulps AT THE OUTPUT'S SCALE (per-element ulp distance is meaningless for
    # outputs near 0: a 1e-7 absolute change is many ulps of a 1e-6 value), <= 5% of outputs
    # differ, and the split result is no less accurate than single vs an fp64 reference.
    w_sulp = max(r["max_abs_diff_in_output_scale_ulps"] for r in RES["b"])
    acc_ok = all(r["mean_err_vs_fp64_split"] <= 1.02 * r["mean_err_vs_fp64_single"] + 1e-9 for r in RES["b"])
    check("b: split vs single = rounding flips only (<=2 ulp at output scale, <=5% of outputs, accuracy vs "
          "fp64 not worse)", worst_frac <= 0.05 and w_sulp <= 2.0 and acc_ok,
          f"worst frac={worst_frac:.4%} worst abs={worst_abs:.3g} worst={w_sulp:.2f} output-scale ulps, "
          f"acc_ok={acc_ok}")
    eq_small = all(r["n_diff"] == 0 for r in RES["b"] if r["pos"] < 64)
    check("b: pos < 64 (single block) split == single bit-exact", eq_small)
    se = [r for r in RES["b"] if r["one_block_per_split"]]
    RES["b_summary"] = {"worst_frac": worst_frac, "worst_abs": worst_abs, "worst_output_scale_ulps": w_sulp,
                        "max_ulp": max(r["max_ulp"] for r in RES["b"]),
                        "n_cases_bitexact": sum(r["n_diff"] == 0 for r in RES["b"]), "n_cases": len(RES["b"]),
                        "one_block_per_split_cases_bitexact": f"{sum(r['n_diff'] == 0 for r in se)}/{len(se)}"}

# ================================================================== (c) prefill ring vs linear
print("(c) prefill sliding window: ring mode vs linear kernel", flush=True)
CASES = [(0, 1), (0, 7), (0, 128), (0, 700), (50, 1), (50, 300), (300, 64), (1000, 129),
         (127, 1), (128, 2), (129, 255), (4000, 1100)]
RES["c"] = []
with torch.inference_mode():
    for bm, bn in ((64, 64), (32, 32), (128, 64)):
        allok = True
        for p0, T in CASES:
            q = rnd(T, NQ, HD, g=G)
            # linear cache holding positions 0..p0+T-1 (+ stale values beyond)
            kl, vl = Kf, Vf
            # block's own K/V: strided like the server (K contiguous, V a view of a wider row)
            kn = kl[0, :, p0:p0 + T].transpose(0, 1).contiguous()                    # [T, 8, 64]
            vwide = torch.zeros(T, NKV * HD + 512, dtype=bf, device=dev)
            vwide[:, 256:256 + NKV * HD] = vl[0, :, p0:p0 + T].transpose(0, 1).reshape(T, -1)
            vn = vwide[:, 256:256 + NKV * HD].view(T, NKV, HD)                       # strides (T*.., 64, 1)
            kr = rnd(1, NKV, RING, HD, g=G, scale=5.0)                               # stale garbage
            vr = rnd(1, NKV, RING, HD, g=G, scale=5.0)
            js = torch.arange(max(0, p0 - RING), p0, device=dev)
            if js.numel():
                kr[0, :, js % RING] = kl[0, :, js]
                vr[0, :, js % RING] = vl[0, :, js]
            ol_ = fc.prefill_attention(q, kl, vl, sinks, p0, SC, W, SMAX, BM=bm, BN=bn)
            or_ = fc.prefill_attention(q, kr, vr, sinks, p0, SC, W, SMAX, BM=bm, BN=bn,
                                       k_new=kn, v_new=vn, ring=RING)
            eq = torch.equal(ol_, or_)
            fc.ring_write(kr, vr, kn, vn, p0, RING)
            jw = torch.arange(max(0, p0 + T - RING), p0 + T, device=dev)
            rw = torch.equal(kr[0, :, jw % RING], kl[0, :, jw]) and torch.equal(vr[0, :, jw % RING], vl[0, :, jw])
            allok &= eq and rw
            RES["c"].append({"bm": bm, "bn": bn, "p0": p0, "T": T, "equal": eq, "ring_write_ok": rw,
                             "max_abs_diff": (ol_.float() - or_.float()).abs().max().item()})
            if (bm, bn) == (64, 64):
                print(f"    p0={p0:5d} T={T:5d}: equal={eq} ring_write={rw}", flush=True)
        check(f"c: prefill ring == linear (torch.equal) + ring_write, BM={bm} BN={bn}", allok,
              f"{len(CASES)} cases")
    # chained: prompt in blocks, then decode continuation, ring only ever updated by ring_write /
    # slot writes; compared with the linear path at every step
    blocks = [(0, 300), (300, 64), (364, 1), (365, 129), (494, 700), (1194, 3)]
    kr = torch.zeros(1, NKV, RING, HD, dtype=bf, device=dev)
    vr = torch.zeros_like(kr)
    chain_ok = True
    for p0, T in blocks:
        q = rnd(T, NQ, HD, g=G)
        kn = Kf[0, :, p0:p0 + T].transpose(0, 1).contiguous()
        vn = Vf[0, :, p0:p0 + T].transpose(0, 1).contiguous()
        a_ = fc.prefill_attention(q, Kf, Vf, sinks, p0, SC, W, SMAX)
        b_ = fc.prefill_attention(q, kr, vr, sinks, p0, SC, W, SMAX, k_new=kn, v_new=vn, ring=RING)
        fc.ring_write(kr, vr, kn, vn, p0, RING)
        chain_ok &= torch.equal(a_, b_)
    o_l = torch.empty(NQ * HD, dtype=bf, device=dev)
    o_r = torch.empty_like(o_l)
    for pos in range(1197, 1197 + 300):
        kr[0, :, pos % RING] = Kf[0, :, pos]
        vr[0, :, pos % RING] = Vf[0, :, pos]
        q = rnd(NQ * HD, g=G)
        pos_dev.fill_(pos)
        fc.decode_attention(q, Kf, Vf, sinks, pos_dev, o_l, SC, W, SMAX)
        fc.decode_attention(q, kr, vr, sinks, pos_dev, o_r, SC, W, SMAX, ring=RING)
        chain_ok &= torch.equal(o_l, o_r)
    RES["c_chain"] = chain_ok
    check("c: chained prompt blocks + 300 decode steps, ring == linear", chain_ok)
    # reuse rule
    rr = [fc.ring_reuse_ok(1000, 999, 128), fc.ring_reuse_ok(1000, 998, 128), fc.ring_reuse_ok(100, 7, 128),
          fc.ring_reuse_ok(1000, 1001, 128), fc.ring_reuse_ok(2000, 1200, 1024)]
    check("c: ring_reuse_ok rule", rr == [True, False, True, False, True], str(rr))

# ================================================================== (c2) prefill ring: every tiling
# Review finding: ring mode used two masked loads + tl.where per K/V tile, which doubled the pipelined
# shared-memory buffers -> OutOfResources at BM/BN 64/128, 128/128, 256/64 where the linear kernel
# works. Required now: at EVERY tiling the linear kernel accepts, ring mode compiles, is torch.equal
# to linear on all CASES, and its compiled shared-memory footprint is <= the linear kernel's.
print("(c2) prefill ring mode at every BM/BN tiling the linear kernel accepts", flush=True)
TILINGS = [(64, 64), (32, 32), (128, 32), (128, 64), (64, 128), (128, 128), (256, 64), (16, 16), (32, 64),
           (64, 32), (256, 32), (32, 128)]


def _smem(ring, bm, bn):
    """compiled shared memory (bytes) of k_prefill_attn for this tiling / mode (tiny T=1 launch)."""
    q1 = torch.zeros(1, NQ, HD, dtype=bf, device=dev)
    o1_ = torch.empty_like(q1)
    s1 = torch.zeros(NQ, dtype=bf, device=dev)
    if ring:
        c1 = torch.zeros(1, NKV, RING, HD, dtype=bf, device=dev)
        n1 = torch.zeros(1, NKV, HD, dtype=bf, device=dev)
        ck = fc.k_prefill_attn[(1, NQ)](q1, c1, c1, n1, n1, NKV * HD, HD, NKV * HD, HD, s1, o1_, 0, 1, SC,
                                        SLIDING=W, SMAX=RING, REP=REP, NQ=NQ, BM=bm, BN=bn, RING=RING, num_warps=4)
    else:
        c1 = torch.zeros(1, NKV, 256, HD, dtype=bf, device=dev)
        ck = fc.k_prefill_attn[(1, NQ)](q1, c1, c1, c1, c1, 0, 0, 0, 0, s1, o1_, 0, 1, SC,
                                        SLIDING=W, SMAX=256, REP=REP, NQ=NQ, BM=bm, BN=bn, num_warps=4)
    torch.cuda.synchronize()
    return int(ck.metadata.shared)


RES["c2"] = []
with torch.inference_mode():
    c2_ok, n_lin_ok = True, 0
    for bm, bn in TILINGS:
        row = {"bm": bm, "bn": bn}
        try:
            row["smem_linear"] = _smem(0, bm, bn)
        except Exception as e:                                       # linear itself unsupported: nothing required
            row["linear"] = f"unsupported ({type(e).__name__})"
            RES["c2"].append(row)
            print(f"    BM={bm:3d} BN={bn:3d}: linear unsupported ({type(e).__name__}) - no requirement", flush=True)
            continue
        n_lin_ok += 1
        try:
            row["smem_ring"] = _smem(RING, bm, bn)
            neq = 0
            for p0, T in CASES:
                q = rnd(T, NQ, HD, g=G)
                kn = Kf[0, :, p0:p0 + T].transpose(0, 1).contiguous()
                vwide = torch.zeros(T, NKV * HD + 512, dtype=bf, device=dev)
                vwide[:, 256:256 + NKV * HD] = Vf[0, :, p0:p0 + T].transpose(0, 1).reshape(T, -1)
                vn = vwide[:, 256:256 + NKV * HD].view(T, NKV, HD)
                kr = rnd(1, NKV, RING, HD, g=G, scale=5.0)
                vr = rnd(1, NKV, RING, HD, g=G, scale=5.0)
                js = torch.arange(max(0, p0 - RING), p0, device=dev)
                if js.numel():
                    kr[0, :, js % RING] = Kf[0, :, js]
                    vr[0, :, js % RING] = Vf[0, :, js]
                a_ = fc.prefill_attention(q, Kf, Vf, sinks, p0, SC, W, SMAX, BM=bm, BN=bn)
                b_ = fc.prefill_attention(q, kr, vr, sinks, p0, SC, W, SMAX, BM=bm, BN=bn, k_new=kn, v_new=vn,
                                          ring=RING)
                neq += int(not torch.equal(a_, b_))
            row["cases"], row["not_equal"] = len(CASES), neq
            ok = neq == 0 and row["smem_ring"] <= row["smem_linear"]
        except Exception as e:
            row["ring"] = f"FAILS ({type(e).__name__}: {str(e)[:80]})"
            ok = False
        c2_ok &= ok
        RES["c2"].append(row)
        print(f"    BM={bm:3d} BN={bn:3d}: " + (f"ring == linear on {row['cases'] - row['not_equal']}/{row['cases']} "
              f"cases, shared mem ring {row['smem_ring']} B vs linear {row['smem_linear']} B"
              if "ring" not in row else row["ring"]), flush=True)
    check("c2: ring prefill compiles + torch.equal to linear + shared mem <= linear at every tiling linear accepts",
          c2_ok and n_lin_ok >= 7, f"{n_lin_ok} tilings supported by linear, {len(CASES)} cases each")
    for _n in ("q", "kn", "vn", "vwide", "kr", "vr", "a_", "b_"):     # keep the later sections' footprint small
        globals().pop(_n, None)
    torch.cuda.empty_cache()

# ================================================================== (d) k_qkv ring store
print("(d) k_qkv: ring store vs linear store", flush=True)
with torch.inference_mode():
    Wq = rnd((NQ + 2 * NKV) * HD, H, g=G, scale=0.02)
    Bq = rnd((NQ + 2 * NKV) * HD, g=G, scale=0.02)
    in_w = (1 + 0.1 * torch.randn(H, device=dev, generator=G)).to(bf)
    inv_freq = 1.0 / (150000 ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
    att_scale = 0.1 * math.log(32.0) + 1.0
    kl = torch.zeros(1, NKV, SMAX, HD, dtype=bf, device=dev)
    vl = torch.zeros_like(kl)
    d_ok = True
    for first in (True, False):
        for pos in (0, 1, 127, 128, 129, 300, 1000, 16383):
            mid = rnd(H, g=G)
            gout = rnd(H, g=G, scale=0.3)
            cout = torch.randn(H, device=dev, generator=G) * 0.3
            kr = rnd(1, NKV, RING, HD, g=G)
            vr = rnd(1, NKV, RING, HD, g=G)
            kr0, vr0 = kr.clone(), vr.clone()
            xa, xb = torch.empty(H, dtype=bf, device=dev), torch.empty(H, dtype=bf, device=dev)
            qa, qb = torch.empty(NQ * HD, dtype=bf, device=dev), torch.empty(NQ * HD, dtype=bf, device=dev)
            pos_dev.fill_(pos)
            for kc_, vc_, xb_, qo_, ring, rows in ((kl, vl, xa, qa, 0, SMAX), (kr, vr, xb, qb, RING, RING)):
                fc.k_qkv[(2 * (NQ + 2 * NKV),)](mid, gout, cout, xb_, in_w, Wq, Bq, pos_dev, inv_freq, kc_, vc_, qo_,
                                                1e-5, att_scale, SMAX=rows, FIRST=first, H=H, HP=fc.HP, BK=128,
                                                NQ=NQ, NKV=NKV, RING=ring, num_warps=4)
            sl = pos % RING
            ok = (torch.equal(kl[0, :, pos], kr[0, :, sl]) and torch.equal(vl[0, :, pos], vr[0, :, sl])
                  and torch.equal(qa, qb) and torch.equal(xa, xb))
            other = torch.ones(RING, dtype=torch.bool, device=dev)
            other[sl] = False
            ok &= torch.equal(kr[0, :, other], kr0[0, :, other]) and torch.equal(vr[0, :, other], vr0[0, :, other])
            ok &= bool(kl[0, :, pos].abs().sum() > 0)
            d_ok &= ok
    RES["d"] = d_ok
    check("d: k_qkv ring slot pos%128 == linear row pos (K, V, q, xbuf), other slots untouched", d_ok,
          "16 cases (FIRST x 8 pos)")
    del kl, vl, Wq

# ================================================================== (e) CUDA graphs
print("(e) CUDA graph capture + replay with changing pos", flush=True)
with torch.inference_mode():
    # e1: kernel level
    qg = rnd(NQ * HD, g=G)
    og = torch.empty(NQ * HD, dtype=bf, device=dev)
    oe = torch.empty_like(og)
    spg = fc.SplitAttn(16, SMAX, dev)
    pos_dev.fill_(5)
    spg.launch(qg, Kf, Vf, sinks, pos_dev, og, SC)                     # compile outside capture
    torch.cuda.synchronize()
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        spg.launch(qg, Kf, Vf, sinks, pos_dev, og, SC)
    e1 = True
    for pos in [16383, 0, 1, 63, 64, 65, 1000, 4095, 8000, 77, 16383, 12000]:
        pos_dev.fill_(pos)
        g1.replay()
        spg2 = SPL[16]
        spg2.launch(qg, Kf, Vf, sinks, pos_dev, oe, SC)
        e1 &= torch.equal(og, oe)
    check("e1: split-K graph replay (pos changed between replays) == eager", e1, "12 replays")
    del g1, spg, SPL_NC
    del Kf, Vf                  # keep the footprint small; (f) times on the synthetic model's full-layer cache
    torch.cuda.empty_cache()

    # e2: FusedCore on a synthetic 2-layer model (layer 0 sliding, layer 1 full)
    def lin(o, i):
        return SimpleNamespace(weight=rnd(o, i, g=G, scale=0.02), bias=rnd(o, g=G, scale=0.02))

    def norm():
        return SimpleNamespace(weight=(1 + 0.1 * torch.randn(H, device=dev, generator=G)).to(bf), variance_epsilon=1e-5)

    def layer():
        at = SimpleNamespace(q_proj=lin(NQ * HD, H), k_proj=lin(NKV * HD, H), v_proj=lin(NKV * HD, H),
                             o_proj=lin(H, NQ * HD), sinks=rnd(NQ, g=G))
        return SimpleNamespace(self_attn=at, input_layernorm=norm(), post_attention_layernorm=norm(),
                               mlp=SimpleNamespace(router=lin(NE, H)))

    class LG:
        def __init__(self, ly, pos, smax, sliding):
            self.layer, self.pos, self.smax, self.sliding, self.scaling = ly, pos, smax, sliding, SC
            self.k = torch.zeros(1, NKV, smax, HD, dtype=bf, device=dev)
            self.v = torch.zeros_like(self.k)
            self.h_norm = torch.zeros(1, H, dtype=bf, device=dev)
            self.r_scores = torch.zeros(1, 4, dtype=bf, device=dev)
            self.r_idx = torch.zeros(1, 4, dtype=torch.long, device=dev)

    class CG:
        def __init__(self, lys):
            self.pos = torch.zeros(1, dtype=torch.int64, device=dev)
            self.inv_freq = inv_freq
            self.att_scale = att_scale
            self.layers = [LG(lys[0], self.pos, SMAX, W), LG(lys[1], self.pos, SMAX, None)]

    LYS = [layer(), layer()]
    cgA, cgB = CG(LYS), CG(LYS)
    old_ptr = cgB.layers[0].k.data_ptr()
    FA = fc.FusedCore(cgA, dev)                                         # default: linear + single
    FB = fc.FusedCore(cgB, dev, ring=RING, split_k=16)                  # ring + split-K
    check("e2: FusedCore(ring) replaced sliding caches (full layer untouched)",
          tuple(cgB.layers[0].k.shape) == (1, NKV, RING, HD) and tuple(cgB.layers[0].v.shape) == (1, NKV, RING, HD)
          and cgB.layers[0].k.data_ptr() != old_ptr and tuple(cgB.layers[1].k.shape) == (1, NKV, SMAX, HD)
          and cgB.layers[0].ring == RING and cgB.layers[1].ring == 0 and FA.split is None and FB.split is not None)

    def bufs():
        return SimpleNamespace(mid=torch.zeros(1, H, dtype=bf, device=dev), g=torch.zeros(1, H, dtype=bf, device=dev),
                               c=torch.zeros(1, H, dtype=torch.float32, device=dev),
                               bx=torch.zeros(1, H, dtype=bf, device=dev), bs=torch.zeros(4, dtype=torch.long, device=dev),
                               bg=torch.zeros(4, dtype=torch.long, device=dev), bw=torch.zeros(4, dtype=bf, device=dev),
                               pack=torch.zeros(H + 12, dtype=torch.float32, device=dev))
    slot_tab = torch.full((36 * NE,), -1, dtype=torch.long, device=dev)
    slot_tab[::3] = torch.arange(0, 36 * NE, 3, device=dev) // 3
    BA, BB = bufs(), bufs()

    def run(F, b, L):
        F.run(L, b.mid, b.g, b.c, slot_tab, b.bx, b.bs, b.bg, b.bw, b.pack)

    for F, b in ((FA, BA), (FB, BB)):                                   # compile, then clear caches
        run(F, b, 0)
        run(F, b, 1)
    for cg in (cgA, cgB):
        for lg in cg.layers:
            lg.k.zero_()
            lg.v.zero_()
    torch.cuda.synchronize()
    GB = []
    for L in (0, 1):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(FB, BB, L)
        GB.append(g)
    GA = []                                                            # for timing only
    for L in (0, 1):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(FA, BA, L)
        GA.append(g)

    def outs(cg, b, L):
        lg = cg.layers[L]
        return [b.mid, b.bx, b.bs, b.bg, b.bw, b.pack, lg.h_norm, lg.r_scores, lg.r_idx]

    e2 = {"steps": 0, "sliding_bitexact": 0, "full_kv_equal": 0, "full_mid_equal": 0, "full_mid_maxabs": 0.0,
          "full_top4_set_agree": 0, "full_top4_order_agree": 0}
    phases = [(0, 300), (8000, 24), (SMAX - 40, 40)]
    for start, nsteps in phases:
        if start > 0:                                                  # jump: identical random history
            for L in (0, 1):
                kA, vA = cgA.layers[L].k, cgA.layers[L].v
                kA[0, :, :start] = rnd(NKV, start, HD, g=G)
                vA[0, :, :start] = rnd(NKV, start, HD, g=G)
                if L == 0:
                    js = torch.arange(start - RING, start, device=dev)
                    cgB.layers[0].k[0, :, js % RING] = kA[0, :, js]
                    cgB.layers[0].v[0, :, js % RING] = vA[0, :, js]
                else:
                    cgB.layers[1].k.copy_(kA)
                    cgB.layers[1].v.copy_(vA)
        for pos in range(start, start + nsteps):
            cgA.pos.fill_(pos)
            cgB.pos.fill_(pos)
            x0 = rnd(1, H, g=G)
            BA.mid.copy_(x0)
            BB.mid.copy_(x0)
            run(FA, BA, 0)                                             # reference: eager default path
            GB[0].replay()                                             # ring path: graph replay
            ok0 = all(torch.equal(a, b) for a, b in zip(outs(cgA, BA, 0), outs(cgB, BB, 0)))
            ok0 &= torch.equal(cgA.layers[0].k[0, :, pos], cgB.layers[0].k[0, :, pos % RING])
            ok0 &= torch.equal(cgA.layers[0].v[0, :, pos], cgB.layers[0].v[0, :, pos % RING])
            e2["sliding_bitexact"] += int(ok0)
            gg, cc = rnd(1, H, g=G, scale=0.3), torch.randn(1, H, device=dev, generator=G) * 0.3
            for b in (BA, BB):
                b.g.copy_(gg)
                b.c.copy_(cc)
            BB.mid.copy_(BA.mid)                                       # identical layer-1 input
            run(FA, BA, 1)
            GB[1].replay()
            e2["full_kv_equal"] += int(torch.equal(cgA.layers[1].k[0, :, pos], cgB.layers[1].k[0, :, pos])
                                       and torch.equal(cgA.layers[1].v[0, :, pos], cgB.layers[1].v[0, :, pos]))
            e2["full_mid_equal"] += int(torch.equal(BA.mid, BB.mid))
            e2["full_mid_maxabs"] = max(e2["full_mid_maxabs"], (BA.mid.float() - BB.mid.float()).abs().max().item())
            ia, ib = cgA.layers[1].r_idx[0].tolist(), cgB.layers[1].r_idx[0].tolist()
            e2["full_max_abs_dlogit"] = max(e2.get("full_max_abs_dlogit", 0.0),
                                            (FA.logits - FB.logits).abs().max().item())
            if sorted(ia) != sorted(ib):                               # diagnostic: is it a near-tie?
                la_ = FA.logits.double()
                srt = la_.sort(descending=True).values
                e2.setdefault("top4_set_disagreements", []).append(
                    {"pos": pos, "ref_margin_4th_5th": (srt[3] - srt[4]).item(),
                     "max_abs_dlogit": (la_ - FB.logits.double()).abs().max().item(),
                     "max_abs_dh_mid": (BA.mid.float() - BB.mid.float()).abs().max().item()})
            e2["full_top4_set_agree"] += int(sorted(ia) == sorted(ib))
            e2["full_top4_order_agree"] += int(ia == ib)
            e2["steps"] += 1
    RES["e2"] = e2
    snap = FB.snapshot_rings()                                         # snapshot / restore helpers
    k_ptr = cgB.layers[0].k.data_ptr()
    cgB.layers[0].k.normal_()
    cgB.layers[0].v.normal_()
    FB.restore_rings(snap)
    snap_ok = (snap[1] is None and cgB.layers[0].k.data_ptr() == k_ptr and torch.equal(cgB.layers[0].k, snap[0][0])
               and torch.equal(cgB.layers[0].v, snap[0][1]))
    check("e2: snapshot_rings / restore_rings (in place, pointer kept)", snap_ok)
    del snap
    print(f"    {e2}", flush=True)
    check("e2: sliding layer (ring, graph) bit-identical to default eager path at every step",
          e2["sliding_bitexact"] == e2["steps"], f"{e2['sliding_bitexact']}/{e2['steps']}")
    check("e2: full layer K/V write identical at every step", e2["full_kv_equal"] == e2["steps"])
    # NOTE (found by running other --seed values): "every step" is NOT an invariant - split-K's legitimate
    # rounding differences (<= 1 bf16 ulp of h_mid) flip a near-tied 4th/5th router choice on ~0.5% of
    # steps for some data (seed 1: 362/364, both flips with a reference 4th-5th margin of 1/64 = the
    # logit perturbation). Passes for the default seed; kept unchanged (not loosened). The
    # seed-independent property is the near-tie check right below.
    check("e2: full layer (split-K, graph) router top-4 set agrees at every step",
          e2["full_top4_set_agree"] == e2["steps"], f"{e2['full_top4_set_agree']}/{e2['steps']}; "
          f"h_mid bit-equal {e2['full_mid_equal']}/{e2['steps']}, max|dh_mid|={e2['full_mid_maxabs']:.3g}")
    dis = e2.get("top4_set_disagreements", [])
    check("e2: every full-layer top-4 set disagreement is a near-tie the logit perturbation can flip "
          "(reference 4th-5th margin <= 2 * max|dlogit| at that step)",
          all(d["ref_margin_4th_5th"] <= 2 * d["max_abs_dlogit"] for d in dis),
          f"{len(dis)} disagreements in {e2['steps']} steps, max|dlogit| over all steps "
          f"{e2['full_max_abs_dlogit']:.3g}, {dis[:3]}")

# ================================================================== (h) FusedCore API guards
# Review findings: (1) lg.ring existed only when ring>0; (2) run() accepted kc/vc overrides of the
# wrong layout (a linear [1,8,SMAX,64] cache on a ring layer was silently written/read as a ring).
print("(h) FusedCore API: lg.ring in every mode, kc/vc override layout checks", flush=True)
with torch.inference_mode():
    h_attr = ([getattr(lg, "ring", "MISSING") for lg in cgA.layers], [getattr(lg, "ring", "MISSING") for lg in cgB.layers])
    h_lr = ([FA.layer_ring(L) for L in (0, 1)], [FB.layer_ring(L) for L in (0, 1)])
    RES["h_lg_ring"] = {"ring0": h_attr[0], "ring128": h_attr[1], "layer_ring": h_lr}
    check("h: lg.ring set on every layer in every mode and == layer_ring(L)",
          h_attr == ([0, 0], [RING, 0]) and h_lr == ([0, 0], [RING, 0]), str(RES["h_lg_ring"]))

    def raises(fn):
        """'AssertionError' = refused by the host-side check (required); 'no error' = silently accepted;
        anything else (e.g. a Triton CompilationError from an fp32 cache) = not a clean refusal."""
        try:
            fn()
            torch.cuda.synchronize()
            return "no error"
        except AssertionError:
            return "AssertionError"
        except Exception as e:                                         # noqa: BLE001
            return type(e).__name__

    lin_big = torch.zeros(1, NKV, SMAX, HD, dtype=bf, device=dev)
    lin_big2 = torch.zeros_like(lin_big)
    ring_k0, ring_v0 = cgB.layers[0].k.clone(), cgB.layers[0].v.clone()
    mid0 = BB.mid.clone()
    cgB.pos.fill_(1000)
    bad_cases = {
        # linear cache as override for a ring (sliding) layer: the reviewer's R9 case
        "sliding_ring_layer_linear_override": lambda: FB.run(0, BB.mid, BB.g, BB.c, slot_tab, BB.bx, BB.bs, BB.bg, BB.bw,
                                                             BB.pack, kc=lin_big, vc=lin_big2),
        # only vc wrong
        "sliding_ring_layer_vc_only_wrong": lambda: FB.run(0, BB.mid, BB.g, BB.c, slot_tab, BB.bx, BB.bs, BB.bg, BB.bw,
                                                           BB.pack, kc=ring_k0.clone(), vc=lin_big2),
        # ring-sized cache as override for a full (linear) layer
        "full_layer_ring_override": lambda: FB.run(1, BB.mid, BB.g, BB.c, slot_tab, BB.bx, BB.bs, BB.bg, BB.bw,
                                                   BB.pack, kc=ring_k0.clone(), vc=ring_v0.clone()),
        # default FusedCore: sliding layer given a ring-sized cache
        "default_core_ring_sized_override": lambda: FA.run(0, BA.mid, BA.g, BA.c, slot_tab, BA.bx, BA.bs, BA.bg, BA.bw,
                                                           BA.pack, kc=ring_k0.clone(), vc=ring_v0.clone()),
        # right shape, wrong memory layout (non-contiguous view) / wrong dtype
        "non_contiguous_override": lambda: FB.run(0, BB.mid, BB.g, BB.c, slot_tab, BB.bx, BB.bs, BB.bg, BB.bw, BB.pack,
                                                  kc=torch.zeros(1, RING, NKV, HD, dtype=bf, device=dev).transpose(1, 2),
                                                  vc=ring_v0.clone()),
        "fp32_override": lambda: FB.run(0, BB.mid, BB.g, BB.c, slot_tab, BB.bx, BB.bs, BB.bg, BB.bw, BB.pack,
                                        kc=ring_k0.float(), vc=ring_v0.float()),
    }
    h_bad = {k: raises(f) for k, f in bad_cases.items()}
    untouched = (torch.equal(lin_big, torch.zeros_like(lin_big)) and torch.equal(lin_big2, torch.zeros_like(lin_big2))
                 and torch.equal(cgB.layers[0].k, ring_k0) and torch.equal(cgB.layers[0].v, ring_v0)
                 and torch.equal(BB.mid, mid0))
    RES["h_bad_override_refused"] = h_bad
    check("h: wrong-layout kc/vc overrides refused (AssertionError) before any launch; nothing written",
          all(v == "AssertionError" for v in h_bad.values()) and untouched, f"{h_bad}, untouched={untouched}")
    del lin_big, lin_big2, bad_cases
    # correct-layout overrides still accepted and equivalent to the layer's own cache (both modes; the
    # default-core linear clone is exactly what dev/fused_validate.py / dev/hybrid_quality.py pass)
    ok_good = True
    for F, b, cg, L in ((FB, BB, cgB, 0), (FB, BB, cgB, 1), (FA, BA, cgA, 0), (FA, BA, cgA, 1)):
        x0 = rnd(1, H, g=G)
        kov, vov = cg.layers[L].k.clone(), cg.layers[L].v.clone()
        # the kernels write exactly one row per kv head (slot pos % ring, or row pos): snapshot that
        # row of the layer's OWN cache (a full clone would cost 2 x 16 MiB more at SMAX=16384)
        slot = int(cg.pos.item()) % (F.layer_ring(L) or SMAX)
        k_own0, v_own0 = cg.layers[L].k[0, :, slot].clone(), cg.layers[L].v[0, :, slot].clone()
        b.mid.copy_(x0)
        F.run(L, b.mid, b.g, b.c, slot_tab, b.bx, b.bs, b.bg, b.bw, b.pack, kc=kov, vc=vov)
        r_ov = [t.clone() for t in outs(cg, b, L)]
        own_untouched = (torch.equal(cg.layers[L].k[0, :, slot], k_own0) and torch.equal(cg.layers[L].v[0, :, slot], v_own0)
                         and not torch.equal(kov[0, :, slot], k_own0))      # the override row WAS written
        b.mid.copy_(x0)
        F.run(L, b.mid, b.g, b.c, slot_tab, b.bx, b.bs, b.bg, b.bw, b.pack)
        ok_good &= (own_untouched and all(torch.equal(x, y) for x, y in zip(r_ov, outs(cg, b, L)))
                    and torch.equal(kov, cg.layers[L].k) and torch.equal(vov, cg.layers[L].v))
        del kov, vov, k_own0, v_own0
    check("h: correct-layout overrides accepted, == running on the layer's own cache (ring+split and default)",
          ok_good)

# ================================================================== (f) timing
if not A.no_timing:
    print("(f) timing (CUDA events, graph replay; INDICATIVE - GPU is shared)", flush=True)
    RES["f"] = {"attn_us": [], "layer_us": []}
    R_ = A.timing_reps
    INNER = 10

    def gtime(fn, reps=R_):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(INNER):
                fn()
        g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            e0, e1_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1_.record()
            e1_.synchronize()
            ts.append(e0.elapsed_time(e1_) * 1000 / INNER)
        ts.sort()
        return ts[len(ts) // 2]

    with torch.inference_mode():
        Kf = cgA.layers[1].k                           # 16k full-layer cache, random through pos 16383
        Vf = cgA.layers[1].v
        Kf[0, :, :SMAX - 40] = rnd(NKV, SMAX - 40, HD, g=G)
        Vf[0, :, :SMAX - 40] = rnd(NKV, SMAX - 40, HD, g=G)
        SPL_NC = {16: fc.SplitAttn(16, SMAX, dev, cache_s=False)}
        qt = rnd(NQ * HD, g=G)
        ot = torch.empty(NQ * HD, dtype=bf, device=dev)
        Krt = rnd(1, NKV, RING, HD, g=G)
        Vrt = rnd(1, NKV, RING, HD, g=G)
        for pos in (1024, 4096, 8192, 16383):
            pos_dev.fill_(pos)
            row = {"pos": pos,
                   "single": gtime(lambda: fc.decode_attention(qt, Kf, Vf, sinks, pos_dev, ot, SC, 0, SMAX))}
            for ns in NSS:
                row[f"split{ns}"] = gtime(lambda: fc.decode_attention(qt, Kf, Vf, sinks, pos_dev, ot, SC, 0, SMAX,
                                                                      split=SPL[ns]))
            row["split16_nocache"] = gtime(lambda: fc.decode_attention(qt, Kf, Vf, sinks, pos_dev, ot, SC, 0, SMAX,
                                                                       split=SPL_NC[16]))
            row["sliding_linear"] = gtime(lambda: fc.decode_attention(qt, Kf, Vf, sinks, pos_dev, ot, SC, W, SMAX))
            row["sliding_ring"] = gtime(lambda: fc.decode_attention(qt, Krt, Vrt, sinks, pos_dev, ot, SC, W, SMAX,
                                                                    ring=RING))
            RES["f"]["attn_us"].append(row)
            print("    attn us/layer " + " ".join(f"{k}={v:.1f}" if k != "pos" else f"pos={v}" for k, v in row.items()),
                  flush=True)
        for pos in (1024, 4096, 8192, 16383):
            cgA.pos.fill_(pos)
            cgB.pos.fill_(pos)
            row = {"pos": pos}
            for name, g in (("full_default", GA[1]), ("full_split16", GB[1]), ("sliding_default", GA[0]),
                            ("sliding_ring", GB[0])):
                g.replay()
                torch.cuda.synchronize()
                ts = []
                for _ in range(R_):
                    e0, e1_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1_.record()
                    e1_.synchronize()
                    ts.append(e0.elapsed_time(e1_) * 1000)
                ts.sort()
                row[name] = ts[len(ts) // 2]
            RES["f"]["layer_us"].append(row)
            print("    whole layer us " + " ".join(f"{k}={v:.1f}" if k != "pos" else f"pos={v}" for k, v in row.items()),
                  flush=True)
        # sliding-layer prompt attention for one 2048-token block at p0=4096: linear vs ring mode.
        # Timed eagerly (one ~100+ us launch per call, launch overhead negligible) so no graph pool
        # holds per-call outputs (keeps the footprint small).
        def etime(fn, reps=R_):
            fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                e0, e1_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1_.record()
                e1_.synchronize()
                ts.append(e0.elapsed_time(e1_) * 1000)
            ts.sort()
            return ts[len(ts) // 2]

        RES["f"]["prefill_us"] = []
        p0, T = 4096, 2048
        qp = rnd(T, NQ, HD, g=G)
        knp = Kf[0, :, p0:p0 + T].transpose(0, 1).contiguous()
        vnp = Vf[0, :, p0:p0 + T].transpose(0, 1).contiguous()
        js = torch.arange(p0 - RING, p0, device=dev)
        Krt[0, :, js % RING] = Kf[0, :, js]
        Vrt[0, :, js % RING] = Vf[0, :, js]
        for bm, bn in ((64, 64), (64, 32), (64, 128)):
            row = {"bm": bm, "bn": bn,
                   "linear": etime(lambda: fc.prefill_attention(qp, Kf, Vf, sinks, p0, SC, W, SMAX, BM=bm, BN=bn)),
                   "ring": etime(lambda: fc.prefill_attention(qp, Krt, Vrt, sinks, p0, SC, W, SMAX, BM=bm, BN=bn,
                                                              k_new=knp, v_new=vnp, ring=RING))}
            RES["f"]["prefill_us"].append(row)
            print(f"    prefill sliding T={T} p0={p0} BM={bm} BN={bn}: linear {row['linear']:.1f} us  ring {row['ring']:.1f} us",
                  flush=True)
        del qp, knp, vnp

RES["gpu_max_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
RES["gpu_max_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
RES["fails"] = FAILS
RES["wall_s"] = time.perf_counter() - t_start
print(f"GPU max allocated {RES['gpu_max_allocated_MiB']:.0f} MiB, reserved {RES['gpu_max_reserved_MiB']:.0f} MiB; "
      f"wall {RES['wall_s']:.1f} s", flush=True)
print("RESULT:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}", flush=True)
if A.json:
    with open(A.json, "w") as f:
        json.dump(RES, f, indent=1)
sys.exit(1 if FAILS else 0)
