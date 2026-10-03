"""A/B for the DECODE CPU path: gptoss_cpu_cap2.gptoss_experts (current) vs
gptoss_cpu_multi (one token per expert) + gptoss_combine_pairs, for decode shapes
(E experts, 1 token). Checks bit-identity of the combined output, then speed."""
import ctypes, os, statistics, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as NP
from neural.moe.gptoss_adapter import GPTOSS_LAYOUT, GPTOSS_MODEL_ID
from neural.q80.prepacked_store import Q80PrepackedStore
import cpu_prefill as CP
NP.add_devkit_dll_dir()
VP = ctypes.c_void_p
cap2 = ctypes.CDLL(os.path.join(NP.REPO, "gptoss_cpu_cap2.dll"))
cap2.gptoss_experts.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]
M = CP.CpuMultiExperts(threads=8)
H, SLOT = 2880, GPTOSS_LAYOUT.slot_bytes           # raw store + the shipped (raw-only) DLLs
ps = Q80PrepackedStore(NP.STORE_DIR, layout=GPTOSS_LAYOUT)
assert ps.open(expect_model_id=GPTOSS_MODEL_ID)["status"] == "ok"
rows = [ps.row(L, e) for L in (10, 11, 12, 13) for e in range(0, 128, 4)]
for r in rows:
    r.sum(dtype=torch.int64)
b = torch.load(os.path.join(NP.STORE_DIR, "expert_biases.pt"), weights_only=True)
BGU = np.ascontiguousarray(b["bias_gu"][:128].float().numpy()); BDN = np.ascontiguousarray(b["bias_dn"][:128].float().numpy())
rng = np.random.default_rng(0)
x = (rng.standard_normal(H) * 0.8).astype(np.float32)
scr = np.zeros(4 * (5760 + 3 * H) + H + 64, np.float32)
out_a = np.zeros(H, np.float32)
X4 = np.ascontiguousarray(np.tile(x, (4, 1)))
ONES = np.ones(4, np.float32)
Y = np.zeros((4, H), np.float32)


def run_cap2(sel, w):
    E = len(sel)
    P, BG, BD = (VP * E)(*[rows[i].data_ptr() for i in sel]), (VP * E)(*[BGU[i % 128].ctypes.data for i in sel]), (VP * E)(*[BDN[i % 128].ctypes.data for i in sel])
    cap2.gptoss_experts(E, P, x.ctypes.data, BG, BD, w.ctypes.data, out_a.ctypes.data, scr.ctypes.data, 8)
    return out_a


def run_multi(sel, w):
    E = len(sel)
    off = np.arange(E + 1, dtype=np.int32)
    Yr = M.experts([rows[i].data_ptr() for i in sel], off, X4, ONES, [BGU[i % 128].ctypes.data for i in sel],
                   [BDN[i % 128].ctypes.data for i in sel], out=Y)
    idx = np.arange(E, dtype=np.int32)[None, :]
    return M.combine(idx, w[None, :E], Yr)[0]


same = True
for E in (1, 2, 3, 4):
    for trial in range(5):
        sel = list(rng.choice(len(rows), E, replace=False))
        w = rng.dirichlet(np.ones(E)).astype(np.float32)
        a = run_cap2(sel, w).copy()
        m = run_multi(sel, w).copy()
        same &= bool(np.array_equal(a, m))
print("bit-identical combined outputs:", same, flush=True)
for E in (1, 2, 3, 4):
    ta, tm, k = [], [], 0
    for it in range(160):
        sel = [(k + j) % len(rows) for j in range(E)]; k += E
        w = np.full(E, 1.0 / E, np.float32)
        for name, fn, acc in (("a", run_cap2, ta), ("m", run_multi, tm)) if it % 2 == 0 else (("m", run_multi, tm), ("a", run_cap2, ta)):
            t0 = time.perf_counter(); fn(sel, w); dt = time.perf_counter() - t0
            if it >= 20:
                acc.append(dt)
    a_, m_ = statistics.median(ta), statistics.median(tm)
    print(f"E={E}: cap2 {a_*1e3:.3f} ms ({E*SLOT/a_/1e9:.1f} GB/s) | multi+combine {m_*1e3:.3f} ms ({E*SLOT/m_/1e9:.1f} GB/s) -> {a_/m_:.2f}x", flush=True)
