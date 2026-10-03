"""Offline compile check of the server's Triton kernels (no GPU needed).

Compiles the pool GEMV `_mxfp4_gemv` (used for GPU-computed misses) and `fused_core.k_topk`
(near-miss ranks, with and without the switch) to PTX for a chosen SM with Triton's AOT compiler. Catches type errors, unsupported casts
(int64 -> pointer) and shape/mask mistakes before the kernels ever run on the Windows host.
torch is stubbed if missing: only the kernels' source is needed.

    python tools/compile_check_kernels.py [--sm 86]
"""
import argparse
import os
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--sm", type=int, default=86, help="RTX 3080 Ti = 86")
A = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import torch  # noqa: F401
except ImportError:                                   # annotations / dtype constants only at import time
    stub = types.ModuleType("torch")
    for name in ("Tensor", "dtype", "device"):
        setattr(stub, name, object)
    for name in ("bfloat16", "float16", "float32", "int64", "uint8", "int32", "long"):
        setattr(stub, name, name)
    stub.cuda = types.SimpleNamespace(Stream=object, Event=object, CUDAGraph=object)
    stub.nn = types.SimpleNamespace(functional=types.SimpleNamespace())
    sys.modules["torch"] = stub
    sys.modules["torch.nn"] = stub.nn
    sys.modules["torch.nn.functional"] = stub.nn.functional

import triton                                          # noqa: E402
from triton.backends.compiler import GPUTarget         # noqa: E402
from triton.compiler import ASTSource                  # noqa: E402

target = GPUTarget("cuda", A.sm, 32)


def compile_kernel(fn, sig, cx, num_warps):
    """Handle the two ASTSource conventions (constexprs= in 3.x; constants= before)."""
    full_sig = dict(sig)
    for k in cx:
        full_sig[k] = "constexpr"
    try:
        src = ASTSource(fn=fn, signature=full_sig, constexprs=cx)
    except TypeError:
        src = ASTSource(fn=fn, signature=sig, constants=cx)
    return triton.compile(src, target=target, options={"num_warps": num_warps})


def report(name, k):
    ptx = k.asm.get("ptx", "")
    n_reg = getattr(k, "n_regs", None)
    spills = getattr(k, "n_spills", None)
    print(f"  {name}: OK  ptx {len(ptx)//1024} KiB  regs {n_reg}  spills {spills}", flush=True)


ok = True
print(f"target sm_{A.sm}, triton {triton.__version__}")
try:
    from neural.q80.mxfp4_kernels import _mxfp4_gemv
    sig_pool = {"x_ptr": "*bf16", "b_ptr": "*u8", "s_ptr": "*u8", "slots_ptr": "*i64", "out_ptr": "*bf16",
                "sb_slot": "i64", "ss_slot": "i64"}
    for packed in (False, True):
        tag = "packed scales" if packed else "raw scales"
        k = compile_kernel(_mxfp4_gemv, sig_pool, {"N": 5760, "GK": 90, "X_PER_EXPERT": False, "BLOCK_N": 32, "BLOCK_G": 4,
                                                   "PACKED": packed}, 8)
        report(f"_mxfp4_gemv (pool GEMV, gate_up, {tag})", k)
        k = compile_kernel(_mxfp4_gemv, sig_pool, {"N": 2880, "GK": 90, "X_PER_EXPERT": True, "BLOCK_N": 32, "BLOCK_G": 4,
                                                   "PACKED": packed}, 8)
        report(f"_mxfp4_gemv (pool GEMV, down, {tag})", k)
    from neural.q80.mxfp4_kernels import _mxfp4_gemm
    sig_gemm = {"x_ptr": "*bf16", "b_ptr": "*u8", "s_ptr": "*u8", "slots_ptr": "*i64", "out_ptr": "*bf16",
                "M": "i32", "sb_slot": "i64", "ss_slot": "i64"}
    for packed in (False, True):
        tag = "packed scales" if packed else "raw scales"
        k = compile_kernel(_mxfp4_gemm, sig_gemm, {"N": 5760, "GK": 90, "BLOCK_N": 128, "BLOCK_M": 16, "PACKED": packed}, 4)
        report(f"_mxfp4_gemm (prefill GEMM, gate_up, {tag})", k)
except Exception as e:                                  # noqa: BLE001
    ok = False
    print(f"  mxfp4_kernels FAILED: {type(e).__name__}: {str(e)[:800]}", flush=True)
try:
    from fused_core import k_topk
    sig_tk = {"logits": "*fp32", "slot_tab": "*i64", "base": "i32", "sc_o": "*bf16", "idx_o": "*i64",
              "bslots": "*i64", "bgids": "*i64", "bw": "*bf16", "pack": "*fp32"}
    for nm in (0, 1):
        k = compile_kernel(k_topk, sig_tk, {"H": 2880, "NE": 128, "NM": nm}, 4)
        report(f"k_topk (NM={nm})", k)
except Exception as e:                                  # noqa: BLE001
    ok = False
    print(f"  k_topk FAILED: {type(e).__name__}: {str(e)[:800]}", flush=True)
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
