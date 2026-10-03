"""GPU bit-identity test of the PACKED scale layout for the Triton mxfp4 kernels. RUN ON THE GPU (it uses CUDA + Triton JIT);
first run on the dev PC: 134 checks, all bit-identical (MEASURED).

The runtime's slot pool holds either raw slots (weight_repr "mxfp4_g32", 13,219,200 B, scales [N, 90]) or packed slots
("mxfp4_g32_ps4", 12,839,040 B, scales [N, 46] = base byte + 45 nibble bytes; tools/scale_pack.py). neural.q80.mxfp4_kernels
rebuilds scale = base + delta as the exact integer the raw layout stores, so both layouts must give BIT-IDENTICAL outputs.
This test loads real raw slots (and, with --packed-store, the same slots from a converted store, cross-checked against
pack_slot) plus synthetic slots that exercise every nibble value, builds two small device pools with the runtime's OWN view
builder (neural.moe.layout.mxfp4_pool_views), and runs both kernel variants on random inputs:

  GEMV (decode)   gate_up  x [1, 2880] shared,   E = 4 slots (mixed indices), N = 5760      server config and defaults
                  down     x [4, 2880] per expert (per_expert_x=True),        N = 2880      + other BLOCK_N / BLOCK_G tilings
  GEMM (prefill)  gate_up and down, one slot, M in {1, 2, 15, 16, 17, 64, 100, 257}; and the decode-style E = 4, M = 1 stack

and asserts torch.equal(raw output, packed output) for every one. It also checks that the views address the same logical
scales (unpack(packed view) == raw view), that each kernel is deterministic (raw run twice), prints the max relative error of
one raw GEMV against an fp32 dequantized reference (so "identical" is not "identically wrong"), and finishes with a built-in
NEGATIVE CONTROL (one flipped nibble in a copy of the packed pool must make the comparison FAIL, otherwise the test is vacuous).

    python tools\\scale_pack_gpu_test.py                                   # real slots from NEURAL_STORE_DIR + synthetic
    python tools\\scale_pack_gpu_test.py --packed-store <dir>              # also read the converter's output (tools\\pack_store.py)
    python tools\\scale_pack_gpu_test.py --time --iters 200                # + interleaved GEMV/GEMM speed, raw vs packed

Labels: bit-identity is a MEASURED fact of the run; ms and ratios are MEASURED on the GPU they run on.
"""
import argparse, os, statistics, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import scale_pack as sp                                                           # noqa: E402
import paths as NP                                                                # noqa: E402
from neural.moe.layout import GPTOSS_LAYOUT, GPTOSS_LAYOUT_PS4, mxfp4_pool_views  # noqa: E402
from neural.q80.mxfp4_kernels import mxfp4_gemm, mxfp4_gemv                       # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--store", default=NP.STORE_DIR, help="raw store the real slots come from (read-only memmap)")
ap.add_argument("--packed-store", default=None, help="converted store: read the packed slots from it (and cross-check pack_slot)")
ap.add_argument("--layer", type=int, default=17)
ap.add_argument("--experts", default="0,63,127", help="real slots of --layer to use ('' = none)")
ap.add_argument("--synthetic", type=int, default=3, help="synthetic slots (every nibble value, all span classes)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--time", action="store_true", help="also time raw vs packed (interleaved CUDA-event timings)")
ap.add_argument("--iters", type=int, default=100)
A = ap.parse_args()

assert torch.cuda.is_available(), "this test needs the GPU"
dev = torch.device("cuda")
L_RAW, L_PK = GPTOSS_LAYOUT, GPTOSS_LAYOUT_PS4
H, GU, GK = sp.H, sp.GU, sp.GK
rng = np.random.default_rng(A.seed)

# ------------------------------------------------------------------ slots
raw_slots, pk_slots, names = [], [], []
real = [int(v) for v in A.experts.split(",") if v.strip() != ""]
if real:
    mm = np.memmap(os.path.join(A.store, f"layer_{A.layer}.slots"), dtype=np.uint8, mode="r").reshape(-1, sp.SLOT_RAW)
    pmm = None
    if A.packed_store:
        pmm = np.memmap(os.path.join(A.packed_store, f"layer_{A.layer}.slots"), dtype=np.uint8, mode="r").reshape(-1, sp.SLOT_PACKED)
    for e in real:
        r = np.array(mm[e])
        p = sp.pack_slot(r)
        if pmm is not None:
            assert np.array_equal(np.array(pmm[e]), p), f"converted store slot L{A.layer}E{e} != pack_slot(raw slot)"
        raw_slots.append(r)
        pk_slots.append(p)
        names.append(f"real L{A.layer}E{e}")
for i in range(A.synthetic):
    r = sp.synthetic_slot(rng)
    raw_slots.append(r)
    pk_slots.append(sp.pack_slot(r))
    names.append(f"synthetic {i}")
CAP = len(raw_slots)
assert CAP >= 4, "need at least 4 slots (E = 4 stacks)"
raw_pool = torch.from_numpy(np.stack(raw_slots)).to(dev)
pk_pool = torch.from_numpy(np.stack(pk_slots)).to(dev)
print(f"pools: {CAP} slots ({', '.join(names)}); raw {sp.SLOT_RAW:,} B/slot, packed {sp.SLOT_PACKED:,} B/slot"
      + (f"; packed slots read from {A.packed_store} (== pack_slot: OK)" if A.packed_store else ""), flush=True)

gb_r, db_r, gs_r, ds_r = mxfp4_pool_views(raw_pool, L_RAW)
gb_p, db_p, gs_p, ds_p = mxfp4_pool_views(pk_pool, L_PK)
bad = 0
checks = 0


def fail(msg):
    global bad
    bad += 1
    print("  FAIL:", msg, flush=True)


def eq(name, a, b):
    global checks
    checks += 1
    if a.shape != b.shape or not torch.equal(a, b):
        d = (a.float() - b.float()).abs()
        fail(f"{name}: raw != packed (max|d| = {d.max().item():.3e}, {int((a != b).sum().item())} of {a.numel()} elements differ)")
    elif not torch.isfinite(a.float()).all():
        fail(f"{name}: non-finite output")


# views: same blocks, and the packed scale view unpacks to the raw scale view
assert gs_r.shape == (CAP, L_RAW.gate_up_n, GK) and gs_p.shape == (CAP, L_PK.gate_up_n, GK // 2 + 1), (gs_r.shape, gs_p.shape)
assert ds_r.shape == (CAP, L_RAW.down_n, GK) and ds_p.shape == (CAP, L_PK.down_n, GK // 2 + 1)
assert gb_r.stride(0) == sp.SLOT_RAW and gb_p.stride(0) == sp.SLOT_PACKED and gs_p.stride(0) == sp.SLOT_PACKED
for i in range(CAP):
    checks += 4
    if not torch.equal(gb_r[i], gb_p[i]) or not torch.equal(db_r[i], db_p[i]):
        fail(f"slot {i}: code views differ")
    for nm, sr_, spk in (("gate", gs_r, gs_p), ("down", ds_r, ds_p)):
        un = sp.unpack_scale_rows(spk[i].reshape(-1, GK // 2 + 1).cpu().numpy())
        if not np.array_equal(un, sr_[i].reshape(-1, GK).cpu().numpy()):
            fail(f"slot {i}: unpack({nm} packed scale view) != raw {nm} scale view")
print("views: packed scale views [cap, N, 46] unpack to the raw views [cap, N, 90] for every slot", flush=True)

g = torch.Generator(device="cpu").manual_seed(A.seed + 1)


def rx(*shape, s=0.8):
    return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16).to(dev)


# ------------------------------------------------------------------ GEMV (decode)
slots4 = torch.tensor([CAP - 1, 1, CAP // 2, 0], dtype=torch.long, device=dev)      # 4 distinct, out of order
x1 = rx(1, H)
x4 = rx(4, H)
GEMV_CFGS = [dict(block_n=32, block_g=4, num_warps=8),      # what the server runs
             dict(),                                        # kernel defaults (64 / 4 / 4)
             dict(block_n=16, block_g=8, num_warps=4),      # 90 % 8 = 2: masked group tail
             dict(block_n=128, block_g=2, num_warps=8),
             dict(block_n=64, block_g=16, num_warps=4)]
for cfg in GEMV_CFGS:
    a = mxfp4_gemv(x1, gb_r, gs_r, slots4, **cfg)
    a2 = mxfp4_gemv(x1, gb_r, gs_r, slots4, **cfg)
    checks += 1
    if not torch.equal(a, a2):
        fail(f"raw gemv nondeterministic: {cfg}")
    eq(f"gemv gate_up {cfg}", a, mxfp4_gemv(x1, gb_p, gs_p, slots4, **cfg))
    eq(f"gemv gate_up {cfg} (explicit packed=True)", a, mxfp4_gemv(x1, gb_p, gs_p, slots4, packed=True, **cfg))
    eq(f"gemv down {cfg}", mxfp4_gemv(x4, db_r, ds_r, slots4, per_expert_x=True, **cfg),
       mxfp4_gemv(x4, db_p, ds_p, slots4, per_expert_x=True, **cfg))
    eq(f"gemv down shared-x {cfg}", mxfp4_gemv(x1, db_r, ds_r, slots4, **cfg), mxfp4_gemv(x1, db_p, ds_p, slots4, **cfg))
    for s in range(CAP):                                          # every single slot too (slot * ss_slot addressing)
        one = torch.tensor([s], dtype=torch.long, device=dev)
        eq(f"gemv gate_up slot {s} {cfg}", mxfp4_gemv(x1, gb_r, gs_r, one, **cfg), mxfp4_gemv(x1, gb_p, gs_p, one, **cfg))
print(f"GEMV: {len(GEMV_CFGS)} tilings x (gate_up, down, shared-x down, {CAP} single slots): "
      f"{'all bit-identical' if bad == 0 else f'{bad} FAILURES so far'}", flush=True)

# the wrong layout flag must be refused, never silently misread
for nm, args in (("packed=True on raw views", (gb_r, gs_r, True)), ("packed=False on packed views", (gb_p, gs_p, False))):
    checks += 1
    try:
        mxfp4_gemv(x1, args[0], args[1], slots4, packed=args[2])
        fail(f"{nm}: accepted")
    except ValueError:
        pass

# ------------------------------------------------------------------ GEMM (prefill and decode-style stack)
n0 = bad
for M in (1, 2, 15, 16, 17, 64, 100, 257):
    xm = rx(M, H)
    for s in sorted({0, CAP - 1, 2 % CAP}):
        one = torch.tensor([s], dtype=torch.long, device=dev)
        eq(f"gemm gate_up M={M} slot {s}", mxfp4_gemm(xm, gb_r, gs_r, one), mxfp4_gemm(xm, gb_p, gs_p, one))
        eq(f"gemm down M={M} slot {s}", mxfp4_gemm(xm, db_r, ds_r, one), mxfp4_gemm(xm, db_p, ds_p, one))
eq("gemm stacked E=4 M=1 gate_up (runtime eager decode)", mxfp4_gemm(x1, gb_r, gs_r, slots4), mxfp4_gemm(x1, gb_p, gs_p, slots4))
for bn, bm in ((64, 16), (32, 32), (128, 64)):
    xm = rx(50, H)
    one = torch.tensor([1], dtype=torch.long, device=dev)
    eq(f"gemm gate_up block_n={bn} block_m={bm}", mxfp4_gemm(xm, gb_r, gs_r, one, block_n=bn, block_m=bm),
       mxfp4_gemm(xm, gb_p, gs_p, one, block_n=bn, block_m=bm))
print(f"GEMM: 8 batch sizes x gate_up/down x 3 slots + stacked decode + 3 tilings: "
      f"{'all bit-identical' if bad == n0 else f'{bad - n0} FAILURES'}", flush=True)

# ------------------------------------------------------------------ not identically wrong: raw GEMV vs an fp32 dequantized reference
from neural.moe.gptoss_adapter import mxfp4_dequant_nk                             # noqa: E402
s0 = 0
w = mxfp4_dequant_nk(gb_r[s0].contiguous(), gs_r[s0].contiguous())                 # [5760, 2880] fp32
ref = (x1.float() @ w.t())[0]
out = mxfp4_gemv(x1, gb_p, gs_p, torch.tensor([s0], dtype=torch.long, device=dev), block_n=32, block_g=4, num_warps=8)[0].float()
rel = ((out - ref).abs().max() / ref.abs().max()).item()
print(f"sanity: packed GEMV vs fp32 dequantized reference (slot {s0}, {names[0]}): max|d|/max|ref| = {rel:.2e} (bf16 output rounding is ~4e-3)", flush=True)
checks += 1
if not rel < 2e-2:
    fail(f"packed GEMV disagrees with the dequantized reference (rel {rel:.2e})")

# ------------------------------------------------------------------ negative control: the comparison must be able to fail
mut = pk_pool.clone()
gb_m, db_m, gs_m, ds_m = mxfp4_pool_views(mut, L_PK)
row = 4321
mut[0, sp.OFF_GS + 46 * row + 17] ^= 0x04                                          # one delta nibble of gate row 4321, slot 0
one = torch.tensor([0], dtype=torch.long, device=dev)
a = mxfp4_gemv(x1, gb_r, gs_r, one, block_n=32, block_g=4, num_warps=8)
b = mxfp4_gemv(x1, gb_m, gs_m, one, block_n=32, block_g=4, num_warps=8)
detected = not torch.equal(a, b)
print(f"negative control: one flipped scale nibble in a copy of the packed pool -> outputs differ: {detected} "
      f"({int((a != b).sum().item())} elements)", flush=True)
if not detected:
    fail("NEGATIVE CONTROL NOT DETECTED - the comparison is vacuous")

# ------------------------------------------------------------------ optional speed
if A.time:
    def cuda_ms(fn, iters):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / iters

    big = torch.empty(256 << 20, dtype=torch.uint8, device=dev)                    # L2 flush
    print(f"\nspeed (E=4 decode stack, x{A.iters} per sample, L2 flushed between samples; raw / packed / raw#2 interleaved):")
    for name, fn_r, fn_p in (
            ("gemv gate_up 32/4/8", lambda: mxfp4_gemv(x1, gb_r, gs_r, slots4, block_n=32, block_g=4, num_warps=8),
             lambda: mxfp4_gemv(x1, gb_p, gs_p, slots4, block_n=32, block_g=4, num_warps=8)),
            ("gemv down 32/4/8", lambda: mxfp4_gemv(x4, db_r, ds_r, slots4, per_expert_x=True, block_n=32, block_g=4, num_warps=8),
             lambda: mxfp4_gemv(x4, db_p, ds_p, slots4, per_expert_x=True, block_n=32, block_g=4, num_warps=8)),
            ("gemm gate_up M=64", (lambda xm=rx(64, H), one=torch.tensor([1], device=dev): mxfp4_gemm(xm, gb_r, gs_r, one)),
             (lambda xm=rx(64, H), one=torch.tensor([1], device=dev): mxfp4_gemm(xm, gb_p, gs_p, one)))):
        fn_r(); fn_p(); torch.cuda.synchronize()                                   # compile + warm
        t = {"raw": [], "packed": [], "raw#2": []}
        for it in range(20):
            order = [("raw", fn_r), ("packed", fn_p), ("raw#2", fn_r)]
            if it % 2:
                order = order[::-1]
            for k, fn in order:
                big.zero_()
                t[k].append(cuda_ms(fn, max(1, A.iters // 20)))
        m = {k: statistics.median(v) for k, v in t.items()}
        print(f"  {name:22s} raw {m['raw']:.4f} ms | packed {m['packed']:.4f} ms ({m['packed'] / m['raw']:.3f}x) | raw#2 {m['raw#2']:.4f} ms "
              f"({m['raw#2'] / m['raw']:.3f}x = noise floor)")

print(f"\nRESULT: {'ALL BIT-IDENTICAL' if bad == 0 else f'{bad} FAILURES'} ({checks} checks; packed vs raw outputs compared with torch.equal)")
sys.exit(1 if bad else 0)
