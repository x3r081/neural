"""Bit-exactness + speed A/B for the PACKED scale layout of the decode CPU kernel (kernels/gptoss_cpu_cap2.c;
nothing here touches a store or the server). Modelled on tools/kernel_tuning_ab.py.

gptoss_experts_cap reads slots in one of two scale layouts (gptoss_set_scale_layout):
    0 = raw     : 13,219,200 B/slot, 90 scale bytes per row
    1 = packed  : 12,839,040 B/slot, 46 B per row = base byte + 45 bytes of 4-bit deltas (tools/scale_pack.py)
Both are built from the same logical data (SYNTHETIC experts: random MXFP4 codes, scale rows with span <= 15 that
exercise every nibble value; optionally real store slots via --real), so the two outputs must be bit-identical.

PART 1 (default) bit-identity, no timing. For E = 1..4 x 4 trials x every prefetch/pair/fuse setting it checks
    packed output == raw output == the untuned raw output (compared as uint32 bit patterns, so -0.0 != +0.0),
    captured bytes == the slot in its own layout (raw capture == raw slot; packed capture == packed slot),
    and, with --ref old.dll, that the refactored raw path still equals the previous kernel build.
PART 2 (--time) speed: raw vs packed (and a raw-again NULL control, and --ref) INTERLEAVED, median of per-call
    times, GB/s over the bytes each layout actually holds, and the PAIRED per-iteration ratio packed/raw with its
    inter-quartile range. E = 4 and E = 2, fuse 0 and 1. The bandwidth-bound expectation is packed/raw = 0.9712.

    bit-identity only : python tools\\scale_pack_ab.py --lib .\\gptoss_cpu_cap2.dll --threads 1 --experts 6 --iters 2
    + old build       : ... --ref <a build from before the layout switch>      (raw-path regression check)
    + real slots      : ... --real 3        (reads a RAW store, read-only; --real-store, default NEURAL_STORE_DIR)
    timing            : python tools\\scale_pack_ab.py --lib .\\gptoss_cpu_cap2.dll --ref <old build> --time --threads 8 --experts 32 --iters 200

Labels: bit-identity is a MEASURED fact of the build; GB/s and ratios are MEASURED on the machine they run on.
"""
import argparse, ctypes, os, statistics, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scale_pack as sp
import paths as NP                                            # noqa: E402  (NEURAL_STORE_DIR)

H, GU = sp.H, sp.GU
SLOT_RAW, SLOT_PK = sp.SLOT_RAW, sp.SLOT_PACKED
BYTE_RATIO = SLOT_PK / SLOT_RAW
VP = ctypes.c_void_p
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lib", default=os.path.join(ROOT, "gptoss_cpu_cap2.dll"), help="kernel with gptoss_set_scale_layout")
ap.add_argument("--ref", default=None, help="previous kernel build (raw layout only), for bit-identity and a raw-path regression check")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--experts", type=int, default=32, help="synthetic experts in each pool (>= 16 keeps the L3 cold)")
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--time", action="store_true", help="also run the timing A/B (NOT run by default)")
ap.add_argument("--pf", type=int, default=0, help="prefetch bytes for the timing part (server default 0)")
ap.add_argument("--pair", type=int, default=0, help="pair_rows for the timing part (server default 0)")
ap.add_argument("--aff", type=int, default=0, help="affinity stride for the timing part (server default 0)")
ap.add_argument("--fuse-list", default="0,1", help="fuse settings to time")
ap.add_argument("--real", type=int, default=0, help="replace this many pool experts by REAL slots (read-only memmap)")
ap.add_argument("--real-store", default=NP.STORE_DIR, help="a RAW store (default NEURAL_STORE_DIR)")
ap.add_argument("--real-layer", type=int, default=0)
ap.add_argument("--mutate", action="store_true", help="NEGATIVE CONTROL: flip one scale nibble in the packed pool; the run must then report FAILURES")
A = ap.parse_args()


def load(path):
    L = ctypes.CDLL(os.path.abspath(path))          # Windows does not search the cwd for a bare DLL name
    L.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
    L.gptoss_set_tuning.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.gptoss_set_fuse.argtypes = [ctypes.c_int]
    if hasattr(L, "gptoss_set_scale_layout"):
        L.gptoss_set_scale_layout.argtypes = [ctypes.c_int]; L.gptoss_set_scale_layout.restype = ctypes.c_int
        L.gptoss_get_scale_layout.restype = ctypes.c_int
        L.gptoss_slot_bytes.argtypes = [ctypes.c_int]; L.gptoss_slot_bytes.restype = ctypes.c_longlong
    return L


def tune(L, pf, pr, fuse, aff=0):
    L.gptoss_set_tuning(pf, pr, aff)
    L.gptoss_set_fuse(fuse)


lib = load(A.lib)
if not hasattr(lib, "gptoss_set_scale_layout"):
    print("the --lib kernel has no gptoss_set_scale_layout (rebuild it with tools\\build_kernels.bat)", file=sys.stderr)
    sys.exit(2)
ref = load(A.ref) if A.ref else None

# ------------------------------------------------------------------ pools (raw + packed, same logical data)
rng = np.random.default_rng(0)
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
    pk[i] = sp.pack_slot(raw[i])                     # raises SpanError if a row cannot be stored in 4 bits
for i in (0, NEXP - 1):
    assert np.array_equal(sp.unpack_slot(pk[i]), raw[i])
if A.mutate:
    pk[1, sp.OFF_GS + 46 * 5000 + 20] ^= 0x04          # one delta of gate row 5000 in expert 1
BGU = (rng.standard_normal((NEXP, GU)) * 0.05).astype(np.float32)
BDN = (rng.standard_normal((NEXP, H)) * 0.05).astype(np.float32)
X = (rng.standard_normal((8, H)) * 0.8).astype(np.float32)
scr = np.zeros(4 * (GU + 3 * H) + H + 64, np.float32)
out = np.zeros(H, np.float32)
caps_raw = [np.zeros(SLOT_RAW + 64, np.uint8) for _ in range(4)]
caps_pk = [np.zeros(SLOT_PK + 64, np.uint8) for _ in range(4)]
POOL = {0: raw, 1: pk}
CAPS = {0: caps_raw, 1: caps_pk}
SLOTB = {0: SLOT_RAW, 1: SLOT_PK}
print(f"pools: {NEXP} synthetic experts{real_note}; raw {SLOT_RAW:,} B/slot, packed {SLOT_PK:,} B/slot "
      f"({100 * (1 - BYTE_RATIO):.2f}% fewer bytes); threads {A.threads}; built in {time.perf_counter() - t0:.1f} s", flush=True)


def al(a):
    return (a.ctypes.data + 63) // 64 * 64


def run(L, layout, sel, w, x, capture=False):
    """layout 0/1 = set that scale layout on the new kernel; None = the previous (raw-only) kernel."""
    lay = 0 if layout is None else layout
    if layout is not None:
        L.gptoss_set_scale_layout(layout)
    E = len(sel)
    P = (VP * E)(*[POOL[lay][i].ctypes.data for i in sel])
    BG = (VP * E)(*[BGU[i].ctypes.data for i in sel])
    BD = (VP * E)(*[BDN[i].ctypes.data for i in sel])
    CP = (VP * E)(*([al(CAPS[lay][j]) for j in range(E)] if capture else [None] * E))
    L.gptoss_experts_cap(E, P, x.ctypes.data, BG, BD, w.ctypes.data, out.ctypes.data, scr.ctypes.data, A.threads, CP)
    return out.copy()


def cap_bytes(layout, j):
    c = CAPS[layout][j]
    off = al(c) - c.ctypes.data
    return c[off:off + SLOTB[layout]]


def same(a, b):
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


# ------------------------------------------------------------------ 1. bit-identity
settings = [(pf, pr, fu) for fu in (0, 1) for pr in (0, 1) for pf in (0, 1024, 2048) if not (fu and pr)]   # fuse ignores pair
bad = 0
checks = 0


def fail(msg):
    global bad
    bad += 1
    print("  FAIL:", msg)


# API sanity
if lib.gptoss_slot_bytes(0) != SLOT_RAW or lib.gptoss_slot_bytes(1) != SLOT_PK or lib.gptoss_slot_bytes(2) != -1:
    fail(f"gptoss_slot_bytes = {lib.gptoss_slot_bytes(0)}, {lib.gptoss_slot_bytes(1)}")
lib.gptoss_set_scale_layout(1)
if lib.gptoss_set_scale_layout(2) != -1 or lib.gptoss_get_scale_layout() != 1:
    fail("an unsupported scale layout was accepted")
lib.gptoss_set_scale_layout(0)
print(f"kernel reports slot bytes raw {lib.gptoss_slot_bytes(0):,} / packed {lib.gptoss_slot_bytes(1):,}; mode 2 rejected", flush=True)

cases = []
for E in (1, 2, 3, 4):
    for trial in range(4):
        sel = [int(i) for i in rng.choice(NEXP, E, replace=False)]
        w = rng.dirichlet(np.ones(E)).astype(np.float32)
        cases.append((sel, w, X[trial % len(X)]))
for sel, w, x in cases:
    tune(lib, 0, 0, 0)
    base = run(lib, 0, sel, w, x, capture=True)
    if not np.isfinite(base).all():
        fail(f"non-finite output E={len(sel)} sel={sel}")
    for j, i in enumerate(sel):
        checks += 1
        if not np.array_equal(cap_bytes(0, j), raw[i]):
            fail(f"raw CAPTURE mismatch (untuned) expert {i}")
    if ref is not None:
        tune(ref, 0, 0, 0)
        r = run(ref, None, sel, w, x, capture=True)
        checks += 1
        if not same(r, base):
            fail(f"raw path differs from --ref: E={len(sel)} sel={sel}")
        for j, i in enumerate(sel):
            checks += 1
            if not np.array_equal(cap_bytes(0, j), raw[i]):
                fail(f"--ref capture mismatch expert {i}")
    for pf, pr, fu in settings:
        tune(lib, pf, pr, fu)
        o_raw = run(lib, 0, sel, w, x, capture=False)
        o_pk = run(lib, 1, sel, w, x, capture=True)
        checks += 2
        if not same(o_raw, base):
            fail(f"RAW mode differs from untuned raw: prefetch={pf} pair={pr} fuse={fu} E={len(sel)} sel={sel}")
        if not same(o_pk, base):
            fail(f"PACKED != RAW: prefetch={pf} pair={pr} fuse={fu} E={len(sel)} sel={sel} max|d|={np.abs(o_pk - base).max():.3e}")
        for j, i in enumerate(sel):
            checks += 1
            if not np.array_equal(cap_bytes(1, j), pk[i]):
                fail(f"packed CAPTURE mismatch prefetch={pf} pair={pr} fuse={fu} expert {i}")
        # also: raw capture under this setting (fuse/pair paths copy rows differently)
        run(lib, 0, sel, w, x, capture=True)
        for j, i in enumerate(sel):
            checks += 1
            if not np.array_equal(cap_bytes(0, j), raw[i]):
                fail(f"raw CAPTURE mismatch prefetch={pf} pair={pr} fuse={fu} expert {i}")
tune(lib, 0, 0, 0)
lib.gptoss_set_scale_layout(0)
print(f"bit-identity over {len(cases)} cases x {len(settings)} settings ({checks} checks; packed vs raw output as uint32 bit patterns, "
      f"captured bytes vs the slot in its own layout" + (", raw path vs --ref build" if ref is not None else "") + f"): "
      f"{'OK' if bad == 0 else f'{bad} FAILURES'}", flush=True)

# ------------------------------------------------------------------ 2. speed (interleaved, paired ratios)
if A.time:
    fuses = [int(v) for v in A.fuse_list.split(",")]
    variants = [("raw", lib, 0), ("packed", lib, 1), ("raw#2", lib, 0)] + ([("ref-raw", ref, None)] if ref is not None else [])
    combos = [(f, v[0]) for f in fuses for v in variants]
    vmap = {v[0]: v for v in variants}
    WARM = 6
    print(f"\nspeed: prefetch {A.pf} B, pair {A.pair}, affinity {A.aff}; median of interleaved runs (order reversed every iteration); "
          f"GB/s = bytes that layout holds / call time; ratio = per-iteration paired time vs raw, [p25, p75]; "
          f"byte ratio packed/raw = {BYTE_RATIO:.4f}")
    print(f"{'E':>2s} {'fuse':>4s} {'variant':>8s} | {'ms/call':>8s} {'GB/s':>6s} | {'vs raw':>7s} {'[p25':>7s} {'p75]':>7s}  {'faster in':>9s}")
    summary = {}
    k = 0
    for E in (4, 2):
        wE = np.full(E, 1.0 / E, np.float32)
        acc = {c: [] for c in combos}
        for it in range(A.iters):
            order = combos if it % 2 == 0 else combos[::-1]
            for c in order:
                f, name = c
                _, L, lay = vmap[name]
                sel = [(k + j) % NEXP for j in range(E)]; k += E
                tune(L, A.pf, A.pair, f, A.aff)
                t0 = time.perf_counter(); run(L, lay, sel, wE, X[it % len(X)]); dt = time.perf_counter() - t0
                if it >= WARM:
                    acc[c].append(dt)
        for f in fuses:
            base_t = np.array(acc[(f, "raw")])
            for v in variants:
                t = np.array(acc[(f, v[0])])
                m = statistics.median(t)
                nbytes = SLOTB[1] if v[0] == "packed" else SLOT_RAW
                if v[0] == "raw":
                    rat = ""
                else:
                    r = t / base_t
                    q = np.percentile(r, [25, 50, 75])
                    rat = f"{q[1]:7.4f} {q[0]:7.4f} {q[2]:7.4f}  {100 * float((r < 1).mean()):8.0f}%"
                    summary[(E, f, v[0])] = float(q[1])
                print(f"{E:2d} {f:4d} {v[0]:>8s} | {m * 1e3:8.3f} {E * nbytes / m / 1e9:6.1f} | {rat}")
    print("\nverdict inputs (median paired ratio packed/raw; bandwidth-bound ideal %.4f; raw#2/raw is the noise floor):" % BYTE_RATIO)
    for E in (4, 2):
        for f in fuses:
            p, n = summary[(E, f, "packed")], summary[(E, f, "raw#2")]
            extra = f", ref-raw/raw {summary[(E, f, 'ref-raw')]:.4f}" if ref is not None else ""
            print(f"  E={E} fuse={f}: packed/raw {p:.4f} ({100 * (1 - p):+.2f}% time) vs noise raw#2/raw {n:.4f}{extra}")
    tune(lib, 0, 0, 0)
    lib.gptoss_set_scale_layout(0)
sys.exit(1 if bad else 0)
