"""CPU-ONLY check of the packed-scale Triton kernels through Triton's INTERPRETER (TRITON_INTERPRET=1): no GPU, no JIT
compilation, no CUDA. It runs the very kernel source of neural/q80/mxfp4_kernels.py (PACKED = False / True) on small CPU
tensors and asserts raw == packed bit for bit. The interpreter emulates tl.* in numpy, so this validates the KERNEL LOGIC of
the packed path (scale = base + nibble, masked-lane values, group/row addressing, slot stride) and catches Python/typing
mistakes in the kernel bodies; it does NOT prove the compiled GPU code (that is tools/scale_pack_gpu_test.py, run on the GPU).

    python tools\\scale_pack_triton_interp.py                 # single thread; --n 40 --parts gemv for a quick run

Labels: bit-identity here is MEASURED on the interpreter (numpy float32), not on the GPU.
"""
import argparse, os, sys, time

os.environ["TRITON_INTERPRET"] = "1"                 # must precede the triton import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import scale_pack as sp                                                    # noqa: E402
from neural.q80.mxfp4_kernels import mxfp4_gemm, mxfp4_gemv, _scale_layout   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="48,40", help="rows per matrix to test (comma list); 40 leaves a masked row tail for BLOCK_N = 32")
ap.add_argument("--parts", default="gemv,gemm", help="which kernels to run")
A = ap.parse_args()
GK, H = sp.GK, sp.H
rng = np.random.default_rng(3)
CAP = 3
bad = checks = 0


def fail(m):
    global bad
    bad += 1
    print("  FAIL:", m, flush=True)


def make(N):
    """CAP slots of N rows: random codes, scale rows with span <= 15 (all nibble values), raw + packed scale tensors laid out
    as a slot-strided pool (row pitch 90 / 46) with a padding gap after each slot's scales, like the real pool."""
    blocks = rng.integers(0, 256, size=(CAP, N, GK, 16), dtype=np.uint8)
    sc = np.empty((CAP, N, GK), np.uint8)
    for c in range(CAP):
        sc[c] = sp.synthetic_slot(rng)[sp.OFF_GS:sp.OFF_GS + N * GK].reshape(N, GK)
    # synthetic_slot rows are span<=15 already; force a few edge rows
    sc[0, 0, :] = 105
    sc[0, 1, :] = 105; sc[0, 1, 5] = 120
    sc[1, 2, :] = 130; sc[1, 2, ::2] = 115
    pk = np.stack([sp.pack_scale_rows(sc[c]) for c in range(CAP)])
    pad = 64
    raw_pool = np.zeros((CAP, N * GK + pad), np.uint8)
    pk_pool = np.zeros((CAP, N * (GK // 2 + 1) + pad), np.uint8)
    raw_pool[:, :N * GK] = sc.reshape(CAP, -1)
    pk_pool[:, :N * (GK // 2 + 1)] = pk.reshape(CAP, -1)
    t = torch.from_numpy
    return (t(blocks), t(raw_pool)[:, :N * GK].view(CAP, N, GK), t(pk_pool)[:, :N * (GK // 2 + 1)].view(CAP, N, GK // 2 + 1))


def eq(name, a, b):
    global checks
    checks += 1
    if not torch.equal(a, b):
        fail(f"{name}: raw != packed ({int((a != b).sum())} of {a.numel()} elements differ)")
    elif not torch.isfinite(a.float()).all():
        fail(f"{name}: non-finite")


t0 = time.perf_counter()
g = torch.Generator().manual_seed(0)
for N in [int(v) for v in A.n.split(",")]:
    blocks, s_raw, s_pk = make(N)
    slots = torch.tensor([2, 0, 1], dtype=torch.long)
    x1 = (torch.randn(1, H, generator=g) * 0.8).to(torch.bfloat16)
    x3 = (torch.randn(3, H, generator=g) * 0.8).to(torch.bfloat16)
    assert _scale_layout(s_raw, GK, None) is False and _scale_layout(s_pk, GK, None) is True
    for cfg in ((dict(block_n=32, block_g=4, num_warps=8), dict(block_n=16, block_g=8, num_warps=4)) if "gemv" in A.parts else ()):
        # 90 % 4 and 90 % 8 leave masked group tails
        eq(f"gemv shared-x N={N} {cfg}", mxfp4_gemv(x1, blocks, s_raw, slots, **cfg), mxfp4_gemv(x1, blocks, s_pk, slots, **cfg))
        eq(f"gemv per-expert-x N={N} {cfg}", mxfp4_gemv(x3, blocks, s_raw, slots, per_expert_x=True, **cfg),
           mxfp4_gemv(x3, blocks, s_pk, slots, per_expert_x=True, **cfg))
    for M in ((1, 5, 17) if "gemm" in A.parts else ()):
        xm = (torch.randn(M, H, generator=g) * 0.8).to(torch.bfloat16)
        one = torch.tensor([1], dtype=torch.long)
        eq(f"gemm M={M} N={N}", mxfp4_gemm(xm, blocks, s_raw, one, block_n=32, block_m=16),
           mxfp4_gemm(xm, blocks, s_pk, one, block_n=32, block_m=16))
    # the comparison can fail: flip one delta nibble
    s_bad = s_pk.clone()
    s_bad[1, 3, 20] ^= 0x04                                                                        # slot 1, row 3, one nibble byte
    a = mxfp4_gemv(x1, blocks, s_raw, torch.tensor([1]), block_n=32, block_g=4, num_warps=8)
    b = mxfp4_gemv(x1, blocks, s_bad, torch.tensor([1]), block_n=32, block_g=4, num_warps=8)
    checks += 1
    if torch.equal(a, b):
        fail("negative control not detected (vacuous test)")
    print(f"N={N}: done ({time.perf_counter() - t0:.1f} s elapsed)", flush=True)

print(f"RESULT: {'ALL BIT-IDENTICAL' if bad == 0 else f'{bad} FAILURES'} ({checks} checks, Triton interpreter, {time.perf_counter() - t0:.1f} s)")
sys.exit(1 if bad else 0)
