"""Component B test: multi-token MXFP4 CPU expert kernel (gptoss_cpu_multi.dll).

(a) bit-identity: gptoss_experts_multi (via cpu_prefill.py) vs per-pair single-token calls
    of the EXISTING gptoss_cpu_cap2.dll, for c in {1,2,3,5,8,13,40}, E in {1,3,6}, mixed
    counts incl. an empty expert, several seeds; plus thread-count invariance, clamp-range
    inputs, the exact combine vs gptoss_experts(E=K) per token, and the re-exported
    single-token kernels of the multi DLL vs cap2.dll.
(b) timing: ms per expert for c in {1,2,4,8,16,32,64}, multi kernel vs c single-token calls.

CPU only (no CUDA). Reads ~12 expert rows (~160 MB) of the read-only store; writes nothing.
    set PATH=F:\\AI\\Neural\\third_party\\tools\\w64devkit\\bin;%PATH%
    F:\\AI\\Neural\\.venv\\Scripts\\python.exe dev\\test_cpu_multi.py [--no-timing] [--seeds 3]

The slot size and scale layout follow the store (layout_for_store): a raw store (weight_repr mxfp4_g32, the default
NEURAL_STORE_DIR) or a packed-scale store (mxfp4_g32_ps4, tools/pack_store.py): `test_cpu_multi.py --store <dir>`.
Both run against the same gptoss_cpu_cap2.dll / gptoss_cpu_multi.dll (cpu_prefill.bind_scale_layout sets the mode).
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.path.dirname(HERE)
sys.path.insert(0, SRV)

import torch  # noqa: E402

from neural.moe.gptoss_adapter import GPTOSS_MODEL_ID, layout_for_store  # noqa: E402
from neural.q80.prepacked_store import Q80PrepackedStore  # noqa: E402

import cpu_prefill as cp  # noqa: E402

H, GU, NE = 2880, 5760, 128
import paths as NP  # noqa: E402
VP = ctypes.c_void_p

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--no-timing", action="store_true")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--no-exhaustive", action="store_true", help="skip the 2^32 expf check (~20 s)")
ap.add_argument("--store", default=NP.STORE_DIR, help="prepacked store (raw or packed scales; default NEURAL_STORE_DIR)")
A = ap.parse_args()
STORE = A.store
LAYOUT = layout_for_store(STORE)             # GPTOSS_LAYOUT (raw scales) or GPTOSS_LAYOUT_PS4 (packed), from metadata.json
ROWB = LAYOUT.slot_bytes                     # 13,219,200 raw / 12,839,040 packed
SCALE_MODE = LAYOUT.scale_layout_mode

# ------------------------------------------------------------------ setup
ps = Q80PrepackedStore(STORE, layout=LAYOUT)
st = ps.open(expect_model_id=GPTOSS_MODEL_ID)
assert st["status"] == "ok", st
# fixed small pool: 8 experts of layer 5 + 4 of layer 20 (only these rows are touched)
POOL = [(5, e) for e in (0, 9, 17, 33, 64, 77, 101, 126)] + [(20, e) for e in (3, 40, 88, 127)]
GIDS = [l * NE + e for l, e in POOL]
ROWS = [ps.row(l, e) for l, e in POOL]
for r in ROWS:
    assert r.dtype == torch.uint8 and r.numel() == ROWB and r.is_contiguous()
RP = [r.data_ptr() for r in ROWS]
b = torch.load(os.path.join(STORE, "expert_biases.pt"), weights_only=True)
assert tuple(b["bias_gu"].shape) == (4608, GU) and tuple(b["bias_dn"].shape) == (4608, H)
BGU = np.ascontiguousarray(b["bias_gu"][GIDS].float().numpy())
BDN = np.ascontiguousarray(b["bias_dn"][GIDS].float().numpy())
del b
BGP = [BGU.ctypes.data + i * GU * 4 for i in range(len(POOL))]
BDP = [BDN.ctypes.data + i * H * 4 for i in range(len(POOL))]

NP.add_devkit_dll_dir()
cap2 = ctypes.CDLL(os.path.join(SRV, "gptoss_cpu_cap2.dll"))
cap2.gptoss_experts.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]
cap2.gptoss_experts.restype = None
cp.bind_scale_layout(cap2, SCALE_MODE, ROWB, "gptoss_cpu_cap2.dll")        # raises for a build without the switch + packed store
M = cp.CpuMultiExperts(threads=A.threads, scale_layout=SCALE_MODE)
assert M.slot_bytes() == ROWB, (M.slot_bytes(), ROWB)
CFG0 = M.get_config()                                         # DLL defaults, before any setter
SCR1 = np.zeros(12 * (GU + 3 * H) + H + 64, np.float32)      # cap2 scratch (E <= 12)


def single(lib, pool_ids, x, w, out, threads=None):
    """lib.gptoss_experts(E, ...) on pool experts `pool_ids` for one token x."""
    E = len(pool_ids)
    sl = (VP * E)(*[RP[i] for i in pool_ids])
    bg = (VP * E)(*[BGP[i] for i in pool_ids])
    bd = (VP * E)(*[BDP[i] for i in pool_ids])
    wv = np.ascontiguousarray(np.asarray(w, np.float32).reshape(E))
    lib.gptoss_experts(E, sl, x.ctypes.data, bg, bd, wv.ctypes.data, out.ctypes.data,
                       SCR1.ctypes.data, threads or A.threads)


def reference_pairs(lib, pool_ids, off, X, W):
    R = np.empty_like(X)
    for e, i in enumerate(pool_ids):
        for p in range(off[e], off[e + 1]):
            single(lib, [i], X[p], W[p:p + 1], R[p])
    return R


def multi(pool_ids, off, X, W, threads=None, out=None):
    return M.experts([RP[i] for i in pool_ids], off, X, W,
                     [BGP[i] for i in pool_ids], [BDP[i] for i in pool_ids],
                     out=out, threads=threads)


def bits_equal(a, b):
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def softmax_w(rng, n):
    lg = rng.normal(0.0, 1.5, (n, 4)).astype(np.float32)
    ex = np.exp(lg - lg.max(1, keepdims=True))
    sm = ex / ex.sum(1, keepdims=True)
    return np.ascontiguousarray(sm[np.arange(n), rng.integers(0, 4, n)].astype(np.float32))


def counts_for(rng, E, c):
    if E == 1:
        return [c]
    extra = [0, max(1, c // 2), 1, 2 * c + 3, 3, c + 1][: E - 1]   # always contains an empty expert
    cnt = [c] + extra
    return [int(v) for v in rng.permutation(cnt)]


print(f"store ok; pool={POOL}; threads={A.threads}", flush=True)
t0 = time.perf_counter()
warm_x = np.zeros(H, np.float32)
warm_o = np.zeros(H, np.float32)
for i in range(len(POOL)):                                   # page every pool row into RAM
    single(cap2, [i], warm_x, [0.0], warm_o)
print(f"pool rows paged in: {time.perf_counter() - t0:.2f} s", flush=True)

FAIL = []
results = {"identity": [], "threads": [], "clamp": [], "combine": [], "reexport": []}

# ------------------------------------------------------------------ (a) bit identity
print("\n(a) bit-identity vs gptoss_cpu_cap2.dll single-token calls")
print(f"{'seed':>4} {'E':>2} {'c':>3} {'counts':<28} {'P':>4} {'bit-equal':>9} {'max|d|':>8}")
for seed in range(A.seeds):
    for E in (1, 3, 6):
        for c in (1, 2, 3, 5, 8, 13, 40):
            rng = np.random.default_rng(1000 * seed + 10 * E + c)
            ids = [int(v) for v in rng.choice(len(POOL), E, replace=False)]
            cnt = counts_for(rng, E, c)
            off = np.zeros(E + 1, np.int32)
            off[1:] = np.cumsum(cnt)
            P = int(off[-1])
            X = (rng.standard_normal((P, H)) * 0.8).astype(np.float32)
            W = softmax_w(rng, P)
            Y = multi(ids, off, X, W)
            R = reference_pairs(cap2, ids, off, X, W)
            ok = bits_equal(Y, R) and bool(np.array_equal(Y, R))
            md = float(np.max(np.abs(Y - R))) if P else 0.0
            results["identity"].append((seed, E, c, cnt, P, ok, md))
            print(f"{seed:>4} {E:>2} {c:>3} {str(cnt):<28} {P:>4} {str(ok):>9} {md:>8.2e}", flush=True)
            if not ok:
                FAIL.append(("identity", seed, E, c))

# thread-count / tiling / schedule invariance (+ reuse of `out`, rows past P untouched)
rng = np.random.default_rng(77)
ids = [int(v) for v in rng.choice(len(POOL), 6, replace=False)]
cnt = [13, 0, 40, 1, 7, 29]
off = np.zeros(7, np.int32)
off[1:] = np.cumsum(cnt)
P = int(off[-1])
X = (rng.standard_normal((P, H)) * 0.8).astype(np.float32)
W = softmax_w(rng, P)
R = reference_pairs(cap2, ids, off, X, W)
OUT = np.full((P + 5, H), np.nan, np.float32)
print("\nthread-count / tiling / schedule invariance (E=6, counts", cnt, ")")
CONFIGS = [(th, 16, 8, True, 64) for th in (1, 2, 3, 5, 8, 16)] + [
    (8, 16, 8, False, 64), (7, 1, 8, False, 64), (8, 32, 4, True, 32), (8, 7, 3, True, 17),
    (3, 5, 1, False, 64), (8, 1000, 6, True, 1000), (16, 3, 5, False, 64)]
for th, rpt, tpb, dyn, irows in CONFIGS:
    M.set_tiling(rpt, tpb)
    M.set_schedule(dyn, irows)
    OUT[:] = np.nan
    Y = multi(ids, off, X, W, threads=th, out=OUT)
    ok = bits_equal(Y, R) and bool(np.all(np.isnan(OUT[P:])))
    results["threads"].append((th, rpt, tpb, dyn, irows, ok))
    print(f"  threads={th:>2} rows/tile={rpt:>4} tok/block={tpb} {'dynamic' if dyn else 'static '} "
          f"rows/item={irows:>4}: bit-equal={ok}", flush=True)
    if not ok:
        FAIL.append(("invariance", th, rpt, tpb, dyn, irows))
M.set_tiling(16, 8)
M.set_schedule(True, 64)

# clamp range: larger inputs so gate > 7 and |up| > 7 occur (exercises the clamped SwiGLU)
print("\nclamp-range inputs (x ~ N(0,1)*s)")
for s in (3.0, 8.0):
    rng = np.random.default_rng(int(s * 10))
    ids = [int(v) for v in rng.choice(len(POOL), 3, replace=False)]
    cnt = [5, 11, 2]
    off = np.zeros(4, np.int32)
    off[1:] = np.cumsum(cnt)
    P = int(off[-1])
    X = (rng.standard_normal((P, H)) * s).astype(np.float32)
    W = softmax_w(rng, P)
    Y = multi(ids, off, X, W)
    R = reference_pairs(cap2, ids, off, X, W)
    ok = bits_equal(Y, R)
    # how often the clamps engage: read the gate_up pre-activation left in the multi scratch
    # (documented layout: 64-B aligned base, XH [P][H] then GUS [P][GU])
    base = M._scr.ctypes.data
    a0 = (((base + 63) & ~63) - base) // 4
    gus = M._scr[a0 + P * H: a0 + P * H + P * GU].reshape(P, GU)
    fg = float(np.mean(gus[:, :H] > 7.0))
    fu = float(np.mean(np.abs(gus[:, H:]) > 7.0))
    results["clamp"].append((s, ok, fg, fu))
    print(f"  scale={s}: bit-equal={ok}  frac(gate>7)={fg:.4f}  frac(|up|>7)={fu:.4f}  "
          f"finite={bool(np.isfinite(Y).all())}", flush=True)
    if not ok:
        FAIL.append(("clamp", s))

# activation unit test: vector path vs cap2's scalar expression on special + random values
print("\nactivation (vector) vs cap2 scalar expression, special values + random")
f32 = np.float32
fmax = np.finfo(f32).max
specials = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 7.0, -7.0,
                     np.nextafter(f32(7), f32(8)), np.nextafter(f32(7), f32(0)),
                     np.nextafter(f32(-7), f32(-8)), np.nextafter(f32(-7), f32(0)),
                     1e-45, -1e-45, 1e-38, -1e-38, fmax, -fmax, 1.0, -1.0,
                     -86.0 / 1.702, -88.0 / 1.702, -51.7, -52.0, -60.0, -1e4, 1e4, 3.0e38],
                    dtype=f32)
rng = np.random.default_rng(4242)
rowsA = []
for sc in (0.5, 3.0, 10.0, 60.0):
    rowsA.append((rng.standard_normal(GU) * sc).astype(f32))
for _ in range(4):
    r_ = (rng.standard_normal(GU) * 5).astype(f32)
    pos = rng.choice(GU, 400, replace=False)
    r_[pos] = specials[rng.integers(0, len(specials), 400)]
    rowsA.append(r_)
r_ = np.resize(specials, GU).astype(f32)                  # every special in gate AND up halves
rowsA.append(r_)
rowsA.append(np.roll(r_, 7))
r_ = (rng.standard_normal(GU) * 5).astype(f32)             # every (gate, up) special combination
ns = len(specials)
r_[:ns * ns] = np.repeat(specials, ns)
r_[H:H + ns * ns] = np.tile(specials, ns)
rowsA.append(r_)
GA = np.ascontiguousarray(np.stack(rowsA))
hv, hr = M.activation(GA, vec=True), M.activation(GA, vec=False)
ok = bits_equal(hv, hr)
results["activation"] = (GA.shape[0], ok)
print(f"  {GA.shape[0]} rows x 2880: bit-equal={ok} (NaN outputs: {int(np.isnan(hr).sum())})", flush=True)
if not ok:
    FAIL.append(("activation",))

if not A.no_exhaustive:
    t0 = time.perf_counter()
    bad, first, nfb = M.check_exp(0, 1 << 32, A.threads)
    results["exp_exhaustive"] = (bad, first, nfb)
    print(f"  vector expf vs scalar expf, ALL 2^32 float inputs: mismatches={bad} "
          f"fallback lanes={nfb} ({time.perf_counter() - t0:.1f} s)", flush=True)
    bad2, _, nfb2 = M.check_exp(0x3E000000, 0x42B00000, A.threads)      # a in [0.125, 88): in-domain
    print(f"  in-domain sample [0.125, 88): mismatches={bad2} fallback rate={nfb2 / (0x42B00000 - 0x3E000000):.2e}",
          flush=True)
    if bad or bad2:
        FAIL.append(("exp_exhaustive", bad, hex(first)))

# exact combine: multi(W=1) + combine(idx, w) vs gptoss_experts(E_t, ...) per token (decode kernel).
# K=4 with random CPU masks gives E_t in 0..4 (the decode case); K=9 exercises the 8/4/FMA-tail
# structure of cap2's compiled combine for E_t up to 9.
print("\nexact combine vs cap2 gptoss_experts(E=E_t) per token (random CPU masks)")
for K, pm, npool in ((4, 0.7, 8), (9, 0.8, 12)):
    for seed in range(A.seeds):
        rng = np.random.default_rng(500 + 17 * K + seed)
        N = 37
        topi = np.stack([rng.choice(npool, K, replace=False) for _ in range(N)])
        lg = rng.normal(0, 1.5, (N, K)).astype(np.float32)
        topw = np.exp(lg - lg.max(1, keepdims=True))
        topw = (topw / topw.sum(1, keepdims=True)).astype(np.float32)
        mask = rng.random((N, K)) < pm
        mask[0] = False                                                    # a token with no CPU expert
        mask[1] = True                                                     # a token with all K
        X = (rng.standard_normal((N, H)) * 0.8).astype(np.float32)
        g = cp.group_pairs(topi, cpu_mask=mask)
        Xp = np.ascontiguousarray(X[g["pair_token"]])
        ones = np.ones(len(Xp), np.float32)
        Yr = multi([int(e) for e in g["experts"]], g["off"], Xp, ones)
        out = M.combine(g["idx"], topw, Yr)
        ref = np.zeros((N, H), np.float32)
        for t in range(N):
            ks = [k for k in range(K) if mask[t, k]]
            if ks:
                single(cap2, [int(topi[t, k]) for k in ks], X[t], topw[t, ks], ref[t])
        # weighted pairs summed in numpy: NOT expected to be bit-identical (reported only)
        Yw = multi([int(e) for e in g["experts"]], g["off"], Xp,
                   np.ascontiguousarray(topw[g["pair_k"][:, 0], g["pair_k"][:, 1]]))
        naive = np.zeros((N, H), np.float32)
        for p, (t, k) in enumerate(g["pair_k"]):
            naive[t] += Yw[p]
        ok = bits_equal(out, ref)
        nz = int(np.count_nonzero(naive.view(np.uint32) != ref.view(np.uint32)))
        et = np.bincount(mask.sum(1), minlength=K + 1)
        results["combine"].append((K, seed, len(Xp), ok, nz))
        print(f"  K={K} seed={seed}: pairs={len(Xp)} E_t histogram={et.tolist()} combine bit-equal={ok}; "
              f"numpy-sum of weighted pairs differs in {nz}/{N * H} values", flush=True)
        if not ok:
            bad = np.nonzero((out.view(np.uint32) != ref.view(np.uint32)).any(1))[0]
            print(f"    mismatching tokens: {bad.tolist()[:20]} with E_t={mask.sum(1)[bad].tolist()[:20]}")
            FAIL.append(("combine", K, seed))

# re-exported single-token kernel of the multi DLL == cap2.dll (decode-style E=4 calls)
print("\nre-exported gptoss_experts (multi DLL) vs cap2.dll, E=4")
rng = np.random.default_rng(9)
okall = True
for trial in range(8):
    ids = [int(v) for v in rng.choice(len(POOL), 4, replace=False)]
    x = (rng.standard_normal(H) * 0.8).astype(np.float32)
    w = softmax_w(rng, 4)
    o1, o2 = np.empty(H, np.float32), np.empty(H, np.float32)
    single(cap2, ids, x, w, o1)
    single(M.lib, ids, x, w, o2)
    okall &= bits_equal(o1, o2)
results["reexport"].append(okall)
print(f"  8 trials: bit-equal={okall}", flush=True)
if not okall:
    FAIL.append(("reexport",))

# re-exported gptoss_experts_cap (the call the server's decode path uses), with 2 of 4 capture
# buffers set: outputs bit-equal to cap2.dll and every captured row byte-equal to the store row
cap2.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
cap2.gptoss_experts_cap.restype = None
okc = True
for trial in range(3):
    ids = [int(v) for v in rng.choice(len(POOL), 4, replace=False)]
    x = (rng.standard_normal(H) * 0.8).astype(np.float32)
    w = softmax_w(rng, 4)
    outs = []
    for lib in (cap2, M.lib):
        bufs = [np.zeros(ROWB, np.uint8), None, np.zeros(ROWB, np.uint8), None]
        cp_ = (VP * 4)(*[None if bb is None else bb.ctypes.data for bb in bufs])
        o = np.empty(H, np.float32)
        lib.gptoss_experts_cap(4, (VP * 4)(*[RP[i] for i in ids]), x.ctypes.data,
                               (VP * 4)(*[BGP[i] for i in ids]), (VP * 4)(*[BDP[i] for i in ids]),
                               w.ctypes.data, o.ctypes.data, SCR1.ctypes.data, A.threads, cp_)
        rows_ok = all(np.array_equal(bufs[j], ROWS[ids[j]].numpy()) for j in (0, 2))
        outs.append((o, rows_ok))
    okc &= bits_equal(outs[0][0], outs[1][0]) and outs[0][1] and outs[1][1]
results["reexport_cap"] = okc
print(f"  gptoss_experts_cap, 3 trials (2 capture buffers each): bit-equal + captured rows exact={okc}",
      flush=True)
if not okc:
    FAIL.append(("reexport_cap",))

# degenerate inputs
Yz = multi([0, 1], np.array([0, 0, 0], np.int32), np.zeros((0, H), np.float32), np.zeros(0, np.float32))
assert Yz.shape == (0, H)
print("degenerate (all experts empty): ok")

# ------------------------------------------------------------------ review-fix regressions
print("\nreview-fix regressions")
INT_MAX = 2**31 - 1


def check(name, ok, detail=""):
    results.setdefault("review", []).append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(("review", name))


# defaults (the dynamic schedule is a documented, deliberate deviation from the spec's static one)
check("DLL default config", CFG0 == {"rows_per_tile": 16, "tokens_per_block": 8, "dynamic": True,
                                     "rows_per_item": 64}, str(CFG0))

# R1: knob values near INT_MAX are clamped to GU (5760) and stay bit-identical (they used to
# overflow: gate_up/down skipped, or an access violation in tile_rows)
rng = np.random.default_rng(31)
ids = [int(v) for v in rng.choice(len(POOL), 4, replace=False)]
cnt = [3, 0, 11, 5]
off = np.zeros(5, np.int32)
off[1:] = np.cumsum(cnt)
P = int(off[-1])
X = (rng.standard_normal((P, H)) * 0.8).astype(np.float32)
W = softmax_w(rng, P)
R = reference_pairs(cap2, ids, off, X, W)
KNOBS = [  # (rows_per_tile, tokens_per_block, dynamic, rows_per_item)
    (16, 8, True, INT_MAX), (16, 8, True, 2147480000), (16, 8, True, INT_MAX - GU + 1), (16, 8, True, 5761),
    (INT_MAX, 8, True, 64), (2147480000, 8, True, 64), (INT_MAX - 1, 3, False, 64),
    (INT_MAX, 8, True, INT_MAX), (INT_MAX, 1, False, INT_MAX), (5760, 8, True, 5760), (1 << 30, 8, True, 1 << 30)]
for rpt, tpb, dyn, irows in KNOBS:
    M.set_tiling(rpt, tpb)
    M.set_schedule(dyn, irows)
    cfg = M.get_config()
    want = {"rows_per_tile": min(rpt, GU), "tokens_per_block": tpb, "dynamic": dyn, "rows_per_item": min(irows, GU)}
    Y = multi(ids, off, X, W, out=np.full((P, H), np.nan, np.float32))
    check(f"knobs tile={rpt} tpb={tpb} {'dyn' if dyn else 'static'} item={irows}",
          cfg == want and bits_equal(Y, R), f"effective {cfg}")
M.set_tiling(16, 8)
M.set_schedule(True, 64)
M.set_tiling(0, 0); M.set_tiling(-7, 9); M.set_schedule(True, 0); M.set_schedule(True, -1 - INT_MAX)
check("invalid knob values ignored", M.get_config() == CFG0, str(M.get_config()))
try:
    M.set_tiling(INT_MAX + 1, 8)
    check("wrapper rejects knob > INT_MAX (ctypes would wrap it)", False)
except ValueError:
    check("wrapper rejects knob > INT_MAX (ctypes would wrap it)", M.get_config() == CFG0)

# R2: W must be the 1-D per-pair vector; a 2-D W ([P, K] topw-like, [P, 1]) or a 0-d W is rejected
slots_, bg_, bd_ = [RP[i] for i in ids], [BGP[i] for i in ids], [BDP[i] for i in ids]
for nm, Wbad in (("[P,4]", np.ascontiguousarray(np.repeat(W[:, None], 4, 1))), ("[P,1]", W[:, None].copy()),
                 ("0-d", np.array(0.5, dtype=np.float32)),
                 ("torch [P,4]", torch.from_numpy(np.ascontiguousarray(np.repeat(W[:, None], 4, 1))))):
    try:
        M.experts(slots_, off, X, Wbad, bg_, bd_)
        check(f"wrapper rejects W {nm}", False, "accepted")
    except ValueError as ex:
        check(f"wrapper rejects W {nm}", True, f"({ex})")
Wlong = np.concatenate([W, np.full(7, np.nan, np.float32)])        # 1-D, longer than P: fine
check("wrapper accepts 1-D W longer than P", bits_equal(M.experts(slots_, off, X, Wlong, bg_, bd_), R))

# R3: off[0] > 0. out=None -> rows [0, off[0]) are +0.0 (not np.empty garbage); out given ->
# rows outside [off[0], P) untouched
offs = off + 5
Xs = np.concatenate([np.full((5, H), np.nan, np.float32), X])
Ws = np.concatenate([np.full(5, np.nan, np.float32), W])
Rs = np.concatenate([np.zeros((5, H), np.float32), R])
for trial in range(3):                                             # fresh allocations each time
    junk = [np.full((P + 5, H), -3.0, np.float32) for _ in range(2)]   # dirty the allocator
    del junk
    Ys = M.experts(slots_, offs, Xs, Ws, bg_, bd_)
    check(f"out=None, off[0]=5 (trial {trial}): shape (P,H), rows [0,5) == +0.0, pairs bit-equal",
          Ys.shape == (P + 5, H) and bool(np.all(Ys[:5].view(np.uint32) == 0)) and bits_equal(Ys, Rs))
Ob = np.full((P + 9, H), 42.5, np.float32)
Yb = M.experts(slots_, offs, Xs, Ws, bg_, bd_, out=Ob)
check("out given, off[0]=5: rows [0,5) and [P,P+9) untouched, pairs bit-equal",
      Yb.shape == (P + 5, H) and bits_equal(Yb[5:], R) and bool(np.all(Ob[:5] == 42.5))
      and bool(np.all(Ob[P + 5:] == 42.5)))

# R4: the pair limit is P <= 248,551 (= floor((2^31-1-16)/8640)), not 248,547
LIM = (INT_MAX - 16) // (H + GU)
check("pair limit constant", cp.MAX_PAIRS == LIM == 248_551, f"MAX_PAIRS={cp.MAX_PAIRS}")
check("scratch_floats at the limit", M.lib.gptoss_multi_scratch_floats(1, LIM) == LIM * (H + GU) + 16
      and M.lib.gptoss_multi_scratch_floats(1, LIM + 1) == -1
      and M.lib.gptoss_multi_scratch_floats(1, 248_548) == 248_548 * (H + GU) + 16)
try:
    M.scratch_floats(1, LIM + 1)
    check("wrapper error message names the true limit", False, "no error")
except ValueError as ex:
    check("wrapper error message names the true limit", "248,551" in str(ex), f"({ex})")
for nm, offbad in (("P = limit+1", np.array([0, LIM + 1], np.int64)),
                   ("int64 off wrapping to a small int32", np.array([0, 2**32 + 3], np.int64)),
                   ("float off", np.array([0.0, 3.0]))):
    try:
        M.experts(slots_[:1], offbad, X, W, bg_[:1], bd_[:1])
        check(f"wrapper rejects {nm}", False, "accepted")
    except ValueError as ex:
        check(f"wrapper rejects {nm}", True, f"({ex})")

n_id = len(results["identity"])
n_ok = sum(1 for r in results["identity"] if r[5])
n_rv = len(results.get("review", []))
print(f"\nIDENTITY: {n_ok}/{n_id} configs bit-equal "
      f"({sum(r[4] for r in results['identity'])} pairs); review-fix checks "
      f"{sum(1 for r in results.get('review', []) if r[1])}/{n_rv}; failures: {FAIL or 'none'}", flush=True)

# ------------------------------------------------------------------ (b) timing
if not A.no_timing:
    print(f"\n(b) timing, threads={A.threads}, 8 distinct warm experts per call (rows in RAM, "
          f"8 x 13.2 MB > L3), median of reps; ms PER EXPERT")
    ET = 8
    ids = list(range(ET))
    hdr = (f"{'c':>3} {'multi':>9} {'multi-st':>9} {'c x cap2':>9} {'c x copy':>9} {'speedup':>8} "
           f"{'multi us/tok':>12} {'single us/tok':>13}")
    print(hdr)
    rows_out = []
    for c in (1, 2, 4, 8, 16, 32, 64):
        rng = np.random.default_rng(c)
        off = np.arange(ET + 1, dtype=np.int32) * c
        P = ET * c
        X = (rng.standard_normal((P, H)) * 0.8).astype(np.float32)
        W = softmax_w(rng, P)
        OUTB = np.empty((P, H), np.float32)
        reps = A.reps if c <= 16 else max(3, A.reps // 2)

        def run_multi():
            multi(ids, off, X, W, out=OUTB)

        def run_single(lib):
            def f():
                for e in range(ET):
                    for p in range(off[e], off[e + 1]):
                        single(lib, [ids[e]], X[p], W[p:p + 1], OUTB[p])
            return f

        def med(fn):
            fn()                                              # warm (pages, pool threads)
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t)
            return statistics.median(ts) * 1e3 / ET

        tm = med(run_multi)
        M.set_schedule(False, 64)
        tst = med(run_multi)                                  # static weighted split (spec variant)
        M.set_schedule(True, 64)
        time.sleep(0.05)                                      # let the other libgomp pool go idle
        ts_cap2 = med(run_single(cap2))
        time.sleep(0.05)
        ts_copy = med(run_single(M.lib))                      # same code, same pool as multi
        time.sleep(0.05)
        rows_out.append((c, tm, tst, ts_cap2, ts_copy))
        print(f"{c:>3} {tm:>9.3f} {tst:>9.3f} {ts_cap2:>9.3f} {ts_copy:>9.3f} {ts_cap2 / tm:>7.2f}x "
              f"{tm * 1e3 / c:>12.1f} {ts_cap2 * 1e3 / c:>13.1f}", flush=True)

sys.exit(1 if FAIL else 0)
