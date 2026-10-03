"""Bit-exactness (+ optional speed) A/B for the PACKED scale layout of the PREFILL (multi-token) CPU kernel
(nothing here touches a store or the server). Modelled on tools/scale_pack_ab.py.

gptoss_experts_multi (kernels/gptoss_cpu_multi.c) reads slots in one of two scale layouts, chosen with
gptoss_set_scale_layout (a process-global of the DLL, read once per call):
    0 = raw     : 13,219,200 B/slot, 90 scale bytes per row
    1 = packed  : 12,839,040 B/slot, 46 B per row = base byte + 45 bytes of 4-bit deltas (tools/scale_pack.py)
Both pools are built from the same logical data (SYNTHETIC experts with span<=15 scale rows that exercise every
nibble value; optionally REAL store slots via --real), so the outputs must be bit-identical.

PART 1 (default) bit-identity, no timing. For E in {1,3,4} experts with mixed token counts (incl. an empty expert
and c > 8, so several register blocks and both the dynamic and the static schedule run) x tiling/schedule settings:
    multi(packed) == multi(raw)                       (uint32 bit patterns: -0.0 != +0.0)
    multi(raw)    == the previous build (--ref), when given (raw-path regression check)
    multi(raw/packed)[pair p] == the single-token kernel gptoss_experts(1, ...) of the SAME DLL in the same
    layout (the multi kernel's documented contract, first setting only),
    and the decode kernel's captured bytes in packed mode == the packed slot.
PART 2 (--time) speed: multi kernel raw vs packed (+ raw again as a NULL control) INTERLEAVED per iteration, paired
    per-iteration ratio packed/raw with its IQR, for the requested token counts. The bandwidth-bound expectation for
    a memory-bound call is 0.9712; a compute-bound call (large c) should show ~1.0 (no penalty).

    identity only : python tools\\scale_pack_multi_ab.py --lib .\\gptoss_cpu_multi.dll --threads 1
    + old build   : ... --ref <a build from before the layout switch>
    negative ctrl : ... --mutate          (must report FAILURES: one packed delta nibble is flipped)
    timing        : ... --time --threads 8 --experts 32 --iters 100

Labels: bit-identity is a MEASURED fact of the build; ms and ratios are MEASURED on the machine they run on.
"""
import argparse, ctypes, os, statistics, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import scale_pack as sp                                   # noqa: E402
import cpu_prefill as cp                                  # noqa: E402
import paths as NP                                        # noqa: E402  (NEURAL_STORE_DIR, DEVKIT)

H, GU = sp.H, sp.GU
SLOT_RAW, SLOT_PK = sp.SLOT_RAW, sp.SLOT_PACKED
VP = ctypes.c_void_p

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lib", default=os.path.join(ROOT, "gptoss_cpu_multi.dll"), help="multi kernel with gptoss_set_scale_layout")
ap.add_argument("--ref", default=None, help="previous multi build (raw layout only), for a raw-path regression check")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--experts", type=int, default=6, help="synthetic experts in each pool (>= 16 keeps the L3 cold for --time)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--single-check", type=int, default=1, help="check each pair against the single-token kernel (1st setting); 0 = skip")
ap.add_argument("--time", action="store_true", help="also run the timing A/B (NOT run by default)")
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--tokens", default="1,4,8,16", help="tokens per expert for the timing part")
ap.add_argument("--real", type=int, default=0, help="replace this many pool experts by REAL slots (read-only memmap)")
ap.add_argument("--real-store", default=NP.STORE_DIR, help="a RAW store (default NEURAL_STORE_DIR)")
ap.add_argument("--real-layer", type=int, default=0)
ap.add_argument("--mutate", action="store_true", help="NEGATIVE CONTROL: flip one scale nibble in the packed pool; the run must report FAILURES")
A = ap.parse_args()

M = cp.CpuMultiExperts(threads=A.threads, dll=os.path.abspath(A.lib))
if not M.has_scale_layout:
    print("the --lib kernel has no gptoss_set_scale_layout (rebuild it with tools\\build_kernels.bat)", file=sys.stderr)
    sys.exit(2)
R = cp.CpuMultiExperts(threads=A.threads, dll=os.path.abspath(A.ref)) if A.ref else None
if R is not None and R.has_scale_layout:
    R.set_scale_layout(0)

# ------------------------------------------------------------------ pools (raw + packed, same logical data)
rng = np.random.default_rng(A.seed)
NEXP = A.experts
t0 = time.perf_counter()
raw = np.empty((NEXP, SLOT_RAW), np.uint8)
for i in range(NEXP):
    raw[i] = sp.synthetic_slot(rng)
real_note = ""
if A.real:
    mm = np.memmap(os.path.join(A.real_store, f"layer_{A.real_layer}.slots"), dtype=np.uint8, mode="r").reshape(-1, SLOT_RAW)
    for j, e in enumerate(np.linspace(0, mm.shape[0] - 1, A.real).astype(int)):
        raw[j] = mm[int(e)]
    real_note = f" (first {A.real} are REAL slots of layer {A.real_layer})"
pk = np.empty((NEXP, SLOT_PK), np.uint8)
for i in range(NEXP):
    pk[i] = sp.pack_slot(raw[i])                        # raises SpanError if a row cannot be stored in 4 bits
for i in (0, NEXP - 1):
    assert np.array_equal(sp.unpack_slot(pk[i]), raw[i])
if A.mutate:
    pk[1, sp.OFF_GS + 46 * 5000 + 20] ^= 0x04             # one delta of gate row 5000 in expert 1
POOL = {0: raw, 1: pk}
BGU = (rng.standard_normal((NEXP, GU)) * 0.05).astype(np.float32)
BDN = (rng.standard_normal((NEXP, H)) * 0.05).astype(np.float32)
PMAX = 64
X = (rng.standard_normal((PMAX, H)) * 0.8).astype(np.float32)
Wp = rng.uniform(0.05, 0.9, PMAX).astype(np.float32)
print(f"pools: {NEXP} synthetic experts{real_note}; raw {SLOT_RAW:,} B/slot, packed {SLOT_PK:,} B/slot; "
      f"threads {A.threads}; built in {time.perf_counter() - t0:.1f} s", flush=True)


def multi(lib, layout, sel, cnt, threads=None):
    """gptoss_experts_multi over experts `sel` with `cnt` tokens each, slots in `layout` (None = previous build)."""
    if layout is not None:
        lib.set_scale_layout(layout)
    pool = POOL[0 if layout is None else layout]
    off = np.zeros(len(sel) + 1, np.int32)
    off[1:] = np.cumsum(cnt)
    P = int(off[-1])
    Y = lib.experts([pool[i].ctypes.data for i in sel], off, X[:P], Wp[:P],
                    [BGU[i].ctypes.data for i in sel], [BDN[i].ctypes.data for i in sel], threads=threads)
    return Y.copy()


SCR1 = np.zeros(4 * (GU + 3 * H) + H + 64, np.float32)
M.lib.gptoss_experts.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]


def single(layout, i, x, w):
    """The same DLL's single-token kernel (decode path) on expert i, slot in `layout`."""
    M.set_scale_layout(layout)
    out = np.empty(H, np.float32)
    wv = np.array([w], np.float32)
    M.lib.gptoss_experts(1, (VP * 1)(POOL[layout][i].ctypes.data), x.ctypes.data, (VP * 1)(BGU[i].ctypes.data),
                         (VP * 1)(BDN[i].ctypes.data), wv.ctypes.data, out.ctypes.data, SCR1.ctypes.data, A.threads)
    return out


def same(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


bad = 0
checks = 0


def fail(msg):
    global bad
    bad += 1
    print("  FAIL:", msg)


# ------------------------------------------------------------------ 1. bit-identity
print(f"kernel reports slot bytes raw {M.slot_bytes(0):,} / packed {M.slot_bytes(1):,}", flush=True)
if M.slot_bytes(0) != SLOT_RAW or M.slot_bytes(1) != SLOT_PK:
    fail(f"kernel slot bytes {M.slot_bytes(0)}, {M.slot_bytes(1)}")
try:
    M.set_scale_layout(2)
    fail("an unsupported scale layout was accepted")
except ValueError:
    pass
M.set_scale_layout(0)

SETTINGS = [("dyn64 tile16/8", 16, 8, True, 64), ("static tile7/3", 7, 3, False, 64), ("dyn17 tile1000/6", 1000, 6, True, 17)]
CASES = [([0], [1]), ([1], [5]), ([2], [8]), ([3], [13]), ([0, 1, 2], [3, 0, 9]), ([4, 2, 5], [8, 1, 5]),
         ([1, 3, 0, 5], [2, 13, 0, 8]), ([5, 4, 3, 2], [17, 1, 1, 6])]
CASES = [(sel, cnt) for sel, cnt in CASES if max(sel) < NEXP]
t0 = time.perf_counter()
for si, (sname, rpt, tpb, dyn, irows) in enumerate(SETTINGS):
    for lib in (M, R):
        if lib is not None:
            lib.set_tiling(rpt, tpb)
            lib.set_schedule(dyn, irows)
    for sel, cnt in CASES:
        y_raw = multi(M, 0, sel, cnt)
        y_pk = multi(M, 1, sel, cnt)
        checks += 2
        if not np.isfinite(y_raw).all():
            fail(f"non-finite output {sname} sel={sel} cnt={cnt}")
        if not same(y_pk, y_raw):
            fail(f"PACKED != RAW ({sname}) sel={sel} cnt={cnt} max|d|={np.abs(y_pk - y_raw).max():.3e}")
        if not same(multi(M, 0, sel, cnt), y_raw):                      # the layout switch toggles back cleanly
            fail(f"raw result changed after a packed call ({sname}) sel={sel} cnt={cnt}")
        if R is not None:
            checks += 1
            if not same(multi(R, None, sel, cnt), y_raw):
                fail(f"raw path differs from --ref build ({sname}) sel={sel} cnt={cnt}")
        if si == 0 and A.single_check:
            off = np.concatenate([[0], np.cumsum(cnt)])
            for e, i in enumerate(sel):
                for p in range(off[e], off[e + 1]):
                    for lay, y in ((0, y_raw), (1, y_pk)):
                        checks += 1
                        if not same(single(lay, i, X[p], Wp[p]), y[p]):
                            fail(f"multi != single-token kernel: layout {lay}, expert {i}, pair {p} ({sname})")
    print(f"  setting {sname}: done ({time.perf_counter() - t0:.1f} s elapsed)", flush=True)
for lib in (M, R):
    if lib is not None:
        lib.set_tiling(16, 8)
        lib.set_schedule(True, 64)
M.set_scale_layout(0)

# the decode kernel re-exported by this DLL, packed mode, with capture (same code as gptoss_cpu_cap2.c)
M.lib.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
sel = [i for i in (0, 1, 2) if i < NEXP]
E = len(sel)
w = np.full(E, 1.0 / E, np.float32)
o = {}
caps = {}
for lay in (0, 1):
    M.set_scale_layout(lay)
    cb = [np.zeros(POOL[lay].shape[1] + 64, np.uint8) for _ in range(E)]
    base = [(c.ctypes.data + 63) // 64 * 64 for c in cb]
    out = np.zeros(H, np.float32)
    M.lib.gptoss_experts_cap(E, (VP * E)(*[POOL[lay][i].ctypes.data for i in sel]), X[0].ctypes.data,
                             (VP * E)(*[BGU[i].ctypes.data for i in sel]), (VP * E)(*[BDN[i].ctypes.data for i in sel]),
                             w.ctypes.data, out.ctypes.data, SCR1.ctypes.data, A.threads, (VP * E)(*base))
    o[lay] = out
    for j, i in enumerate(sel):
        off0 = base[j] - cb[j].ctypes.data
        checks += 1
        if not np.array_equal(cb[j][off0:off0 + POOL[lay].shape[1]], POOL[lay][i]):
            fail(f"decode-kernel capture != slot in layout {lay}, expert {i}")
checks += 1
if not same(o[0], o[1]):
    fail("re-exported gptoss_experts_cap: packed output != raw output")
M.set_scale_layout(0)
print(f"bit-identity ({checks} checks; multi packed vs raw as uint32 bit patterns, multi vs the single-token kernel, "
      f"decode capture bytes" + (", raw path vs --ref build" if R is not None else "") + f"): "
      f"{'OK' if bad == 0 else f'{bad} FAILURES'}  [{time.perf_counter() - t0:.1f} s]", flush=True)

# ------------------------------------------------------------------ 2. speed (interleaved, paired ratios)
if A.time:
    toks = [int(v) for v in A.tokens.split(",")]
    variants = [("raw", M, 0), ("packed", M, 1), ("raw#2", M, 0)] + ([("ref-raw", R, None)] if R is not None else [])
    print(f"\nspeed: multi kernel, {A.threads} threads, E=4 experts x c tokens per call; order reversed every iteration; "
          f"ratio = per-iteration paired time vs raw [p25, p75]; byte ratio packed/raw = {SLOT_PK / SLOT_RAW:.4f}")
    print(f"{'c':>3} {'variant':>8} | {'ms/call':>8} {'GB/s':>6} | {'vs raw':>7} {'[p25':>7} {'p75]':>7}")
    k = 0
    for c in toks:
        acc = {v[0]: [] for v in variants}
        for it in range(A.iters):
            order = variants if it % 2 == 0 else variants[::-1]
            for name, lib, lay in order:
                sel = [(k + j) % NEXP for j in range(4)]
                k += 4
                cnt = [c] * 4
                if sum(cnt) > PMAX:
                    raise SystemExit("--tokens too large for the fixed 64-pair input")
                if lay is not None:
                    lib.set_scale_layout(lay)
                offs = np.zeros(5, np.int32); offs[1:] = np.cumsum(cnt)
                P = int(offs[-1])
                ptrs = [POOL[lay or 0][i].ctypes.data for i in sel]
                bg = [BGU[i].ctypes.data for i in sel]; bd = [BDN[i].ctypes.data for i in sel]
                t = time.perf_counter()
                lib.experts(ptrs, offs, X[:P], Wp[:P], bg, bd, threads=A.threads)
                dt = time.perf_counter() - t
                if it >= 4:
                    acc[name].append(dt)
        base_t = np.array(acc["raw"])
        for name, lib, lay in variants:
            t = np.array(acc[name])
            m = statistics.median(t)
            nbytes = 4 * (SLOT_PK if lay else SLOT_RAW)
            if name == "raw":
                rat = ""
            else:
                q = np.percentile(t / base_t, [25, 50, 75])
                rat = f"{q[1]:7.4f} {q[0]:7.4f} {q[2]:7.4f}"
            print(f"{c:3d} {name:>8s} | {m * 1e3:8.3f} {nbytes / m / 1e9:6.1f} | {rat}")
    M.set_scale_layout(0)
sys.exit(1 if bad else 0)
