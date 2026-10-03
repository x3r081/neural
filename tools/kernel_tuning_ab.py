"""Bit-exactness + speed A/B for the decode CPU kernel's memory-level-parallelism knobs.

Runs gptoss_experts_cap on SYNTHETIC experts (random MXFP4 codes, E8M0 scales in a sane
range, random biases / inputs; no model or store needed) under every combination of
    prefetch_bytes in {0, 1024, 2048}  x  pair_rows in {0, 1}  x  fuse in {0, 1}
and checks that every output is bit-identical to the untuned kernel (prefetch 0, pair 0, fuse 0),
optionally also against a REFERENCE build of the previous kernel (--ref old.dll / old.so).
Then it times each setting (median of interleaved runs) and prints GB/s, for E = 4 experts per
call (what the earlier A/B timed) and for E = 2 (closer to the server's ~2.4 misses per layer,
where per-call fixed costs matter).

Thread pinning cannot be undone inside a process (libgomp keeps its pool threads), so the
pinned rows are measured in a CHILD process (--affinity S spawns it); do not compare pinned and
unpinned rows from one process.

    Windows:  python tools\kernel_tuning_ab.py --lib .\gptoss_cpu_cap2.dll [--ref .\old_cap2.dll] --affinity 2
    Linux  :  python tools/kernel_tuning_ab.py --lib ./cap2.so --ref ./cap2_orig.so

Labels: bit-identity is a MEASURED fact of the build; GB/s is MEASURED on the machine it runs
on (the sandbox that validated this script is not the target host).
"""
import argparse, ctypes, os, statistics, subprocess, sys, time
import numpy as np

H, GU, GK, RB = 2880, 5760, 90, 1440
OFF_DC, OFF_GS, OFF_DS = GU * RB, GU * RB + H * RB, GU * RB + H * RB + GU * GK
SLOT = OFF_DS + H * GK
assert SLOT == 13_219_200      # the RAW slot layout (mode 0, the kernel default); the packed layout is tools/scale_pack_ab.py's
VP = ctypes.c_void_p

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True, help="new kernel (gptoss_cpu_cap2.dll or a Linux .so of the same source)")
ap.add_argument("--ref", default=None, help="previous kernel build to compare against (optional)")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--experts", type=int, default=32, help="synthetic experts in the pool (>= 16 keeps the L3 cold)")
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--affinity", type=int, default=0, help="also time every setting PINNED with this stride, in a child process")
ap.add_argument("--_pinned", type=int, default=0, help=argparse.SUPPRESS)   # child mode: pin first, time only
A = ap.parse_args()
CHILD = A._pinned > 0


def load(path):
    L = ctypes.CDLL(os.path.abspath(path))          # Windows does not search the cwd for a bare DLL name
    L.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
    if hasattr(L, "gptoss_set_tuning"):
        L.gptoss_set_tuning.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.gptoss_get_tuning.argtypes = [VP]
    if hasattr(L, "gptoss_set_fuse"):
        L.gptoss_set_fuse.argtypes = [ctypes.c_int]
    return L


def tune(L, pf, pr, fuse, aff=0):
    L.gptoss_set_tuning(pf, pr, aff)
    if hasattr(L, "gptoss_set_fuse"):
        L.gptoss_set_fuse(fuse)


lib = load(A.lib)
ref = load(A.ref) if (A.ref and not CHILD) else None
if not hasattr(lib, "gptoss_set_tuning"):
    print("the --lib kernel has no gptoss_set_tuning: it is the previous build; nothing to A/B", file=sys.stderr)
    sys.exit(2)
HAS_FUSE = hasattr(lib, "gptoss_set_fuse")
rng = np.random.default_rng(0)
NEXP = A.experts
pool = rng.integers(0, 256, size=(NEXP, SLOT), dtype=np.uint8)
pool[:, OFF_GS:] = rng.integers(112, 136, size=(NEXP, SLOT - OFF_GS), dtype=np.uint8)   # 2^(s-127) in [2^-15, 2^8]
BGU = (rng.standard_normal((NEXP, GU)) * 0.05).astype(np.float32)
BDN = (rng.standard_normal((NEXP, H)) * 0.05).astype(np.float32)
X = (rng.standard_normal((8, H)) * 0.8).astype(np.float32)
scr = np.zeros(4 * (GU + 3 * H) + H + 64, np.float32)
out = np.zeros(H, np.float32)
caps = [np.zeros(SLOT + 64, np.uint8) for _ in range(4)]


def run(L, sel, w, x, capture=False):
    E = len(sel)
    P = (VP * E)(*[pool[i].ctypes.data for i in sel])
    BG = (VP * E)(*[BGU[i].ctypes.data for i in sel])
    BD = (VP * E)(*[BDN[i].ctypes.data for i in sel])
    CP = (VP * E)(*([(caps[j].ctypes.data + 63) // 64 * 64 for j in range(E)] if capture else [None] * E))
    L.gptoss_experts_cap(E, P, x.ctypes.data, BG, BD, w.ctypes.data, out.ctypes.data, scr.ctypes.data, A.threads, CP)
    return out.copy()


def cap_bytes(j):
    off = (caps[j].ctypes.data + 63) // 64 * 64 - caps[j].ctypes.data
    return caps[j][off:off + SLOT]


settings = [(pf, pr, fu) for fu in ((0, 1) if HAS_FUSE else (0,)) for pr in (0, 1) for pf in (0, 1024, 2048)
            if not (fu and pr)]                     # fuse ignores pair_rows

# ------------------------------------------------------------------ 1. bit-identity (parent only)
bad = 0
if not CHILD:
    print(f"pool: {NEXP} synthetic experts x {SLOT/1e6:.1f} MB; threads {A.threads}; fuse available: {HAS_FUSE}")
    cases = []
    for E in (1, 2, 3, 4):
        for trial in range(4):
            sel = [int(i) for i in rng.choice(NEXP, E, replace=False)]
            w = rng.dirichlet(np.ones(E)).astype(np.float32)
            cases.append((sel, w, X[trial % len(X)]))
    for sel, w, x in cases:
        tune(lib, 0, 0, 0)
        base = run(lib, sel, w, x, capture=True)
        for j, i in enumerate(sel):
            if not np.array_equal(cap_bytes(j), pool[i]):
                bad += 1; print(f"  CAPTURE MISMATCH (untuned) expert {i}")
        if ref is not None:
            r = run(ref, sel, w, x, capture=True)
            if not np.array_equal(r, base):
                bad += 1; print(f"  MISMATCH vs --ref: E={len(sel)} sel={sel}")
            for j, i in enumerate(sel):
                if not np.array_equal(cap_bytes(j), pool[i]):
                    bad += 1; print(f"  REF CAPTURE MISMATCH expert {i}")
        for pf, pr, fu in settings[1:]:
            tune(lib, pf, pr, fu)
            o = run(lib, sel, w, x, capture=True)
            if not np.array_equal(o, base):
                bad += 1; print(f"  MISMATCH prefetch={pf} pair={pr} fuse={fu}: E={len(sel)} sel={sel} max|d|={np.abs(o-base).max():.3e}")
            for j, i in enumerate(sel):
                if not np.array_equal(cap_bytes(j), pool[i]):
                    bad += 1; print(f"  CAPTURE MISMATCH prefetch={pf} pair={pr} fuse={fu} expert {i}")
    print(f"bit-identity over {len(cases)} cases x {len(settings)} settings"
          + (" (+ reference build)" if ref is not None else "") + f": {'OK' if bad == 0 else f'{bad} MISMATCHES'}", flush=True)

# ------------------------------------------------------------------ 2. speed (interleaved, median)
aff = A._pinned if CHILD else 0
tag = f"PINNED stride {aff}" if CHILD else "unpinned"
print(f"\nspeed ({tag}), median of interleaved runs; GB/s = expert bytes / call time")
print(f"{'E':>2s} {'prefetch':>8s} {'pair':>4s} {'fuse':>4s} {'aff':>3s} | {'ms/call':>8s} {'GB/s':>6s}")
for E in (4, 2):
    acc = {t: [] for t in settings}
    k = 0
    wE = np.full(E, 1.0 / E, np.float32)
    for it in range(A.iters):
        order = settings if it % 2 == 0 else settings[::-1]
        for t in order:
            sel = [(k + j) % NEXP for j in range(E)]; k += E
            tune(lib, t[0], t[1], t[2], aff)
            t0 = time.perf_counter(); run(lib, sel, wE, X[it % len(X)]); dt = time.perf_counter() - t0
            if it >= 6:
                acc[t].append(dt)
    for t in settings:
        m = statistics.median(acc[t])
        print(f"{E:2d} {t[0]:8d} {t[1]:4d} {t[2]:4d} {aff:3d} | {m*1e3:8.3f} {E*SLOT/m/1e9:6.1f}")
    if ref is not None:
        tr = []
        for it in range(A.iters):
            sel = [(k + j) % NEXP for j in range(E)]; k += E
            t0 = time.perf_counter(); run(ref, sel, wE, X[it % len(X)]); dt = time.perf_counter() - t0
            if it >= 6:
                tr.append(dt)
        m = statistics.median(tr)
        print(f"{E:2d} {'ref':>8s} {'-':>4s} {'-':>4s} {'-':>3s} | {m*1e3:8.3f} {E*SLOT/m/1e9:6.1f}")
tune(lib, 0, 0, 0)

if A.affinity and not CHILD:
    print(f"\n--- child process: every setting with threads pinned (stride {A.affinity}) ---", flush=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--lib", A.lib, "--threads", str(A.threads),
           "--experts", str(A.experts), "--iters", str(A.iters), "--_pinned", str(A.affinity)]
    sys.exit(subprocess.call(cmd) or (1 if bad else 0))
sys.exit(1 if bad else 0)
