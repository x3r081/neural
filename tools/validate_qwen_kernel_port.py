"""Tiny exact-output A/B check for the Qwen BF16 scheduling/fusion port.

This loads two native DLLs directly and never loads a model. By default it
checks the preserved baseline DLL hash before comparing small seeded BF16
expert tensors against the candidate DLL.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".tools/qwen36-baseline-7a46a86/qwen_bf16_experts.dll"
CANDIDATE = ROOT / ".tools/qwen36-port-optimizations/qwen_bf16_experts.dll"
BASELINE_SHA256 = "11076a97409428a22765425e7bdeda05091114dec1256665a9d4b4b4e897b3bf"


def _load(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    function = library.qwen_bf16_experts
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    return library, function


def _run(function, x, gate_up, down, threads):
    tokens, hidden = x.shape
    routes, two_i, gu_hidden = gate_up.shape
    if gu_hidden != hidden or two_i % 2:
        raise ValueError("invalid test gate_up shape")
    intermediate = two_i // 2
    if tuple(down.shape) != (routes, hidden, intermediate):
        raise ValueError("invalid test down shape")
    y = torch.empty((tokens, routes, hidden), dtype=torch.bfloat16)
    gu_tmp = torch.empty((tokens, routes, 2 * intermediate), dtype=torch.bfloat16)
    act_tmp = torch.empty((tokens, routes, intermediate), dtype=torch.bfloat16)
    gu_ptrs = (ctypes.c_void_p * routes)(*(gate_up[r].data_ptr() for r in range(routes)))
    dn_ptrs = (ctypes.c_void_p * routes)(*(down[r].data_ptr() for r in range(routes)))
    status = function(
        ctypes.c_void_p(x.data_ptr()), gu_ptrs, dn_ptrs,
        ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(gu_tmp.data_ptr()),
        ctypes.c_void_p(act_tmp.data_ptr()), tokens, routes, hidden,
        intermediate, threads,
    )
    if status:
        raise RuntimeError(f"kernel returned status {status}")
    return y


def _random_case(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden, intermediate, routes, tokens = 2048, 512, 8, 3
    x = (torch.randn((tokens, hidden), generator=generator) * 0.5).to(torch.bfloat16).contiguous()
    gate_up = (torch.randn((routes, 2 * intermediate, hidden), generator=generator) * 0.02)
    down = (torch.randn((routes, hidden, intermediate), generator=generator) * 0.02)
    return x, gate_up.to(torch.bfloat16).contiguous(), down.to(torch.bfloat16).contiguous()


def _edge_case():
    hidden, intermediate, routes, tokens = 2048, 512, 8, 3
    x = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
    x[:, 0] = 1
    x[0, 1] = 1.0 / 256.0  # halfway between BF16 1.0 and its next value
    x[1, 0] = -1
    x[1, 1] = -1.0 / 256.0
    x[2, 0] = 0.5
    x[2, 2] = -0.25
    gate_up = torch.zeros((routes, 2 * intermediate, hidden), dtype=torch.bfloat16)
    down = torch.zeros((routes, hidden, intermediate), dtype=torch.bfloat16)
    gate_values = (None, -1.0, 12.0, -12.0, 0.5, 0.0, 4.0, -4.0)
    for route, value in enumerate(gate_values):
        if route == 0:
            gate_up[route, 0, 0] = 1
            gate_up[route, 0, 1] = 1
        else:
            gate_up[route, 0, 0] = value
        gate_up[route, intermediate, 0] = -2.0 if route & 1 else 2.0
        gate_up[route, intermediate + 1, 0] = 1.0 / 128.0
        down[route, 0, 0] = 1
        down[route, 0, 1] = -1
    return x.contiguous(), gate_up.contiguous(), down.contiguous()


def _exact_case(name, reference_fn, candidate_fn, tensors, threads):
    ref = _run(reference_fn, *tensors, threads)
    got = _run(candidate_fn, *tensors, threads)
    mismatches = int(torch.count_nonzero(ref != got))
    result = {
        "case": name,
        "threads": threads,
        "shape": list(got.shape),
        "mismatch_count": mismatches,
        "max_abs_error": float((ref.float() - got.float()).abs().max()),
        "nonzero_outputs": int(torch.count_nonzero(got)),
        "finite_outputs": bool(torch.isfinite(got).all()),
        "passed": mismatches == 0 and bool(torch.isfinite(got).all()),
    }
    if not result["passed"]:
        raise AssertionError(f"{name}: candidate differs from preserved DLL: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=BASELINE)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-reference-hash", action="store_true")
    args = parser.parse_args()

    ref_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    if not args.skip_reference_hash and ref_hash != BASELINE_SHA256:
        raise SystemExit(f"reference DLL hash mismatch: {ref_hash}")
    reference_library, reference_fn = _load(args.reference)
    candidate_library, candidate_fn = _load(args.candidate)
    report = {
        "reference": str(args.reference.resolve()),
        "reference_sha256": ref_hash,
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "results": [],
    }

    random = _random_case(20261002)
    report["results"].append(_exact_case("model_shape_random_t3_k8", reference_fn, candidate_fn, random, 8))
    report["results"].append(_exact_case("model_shape_random_single_thread", reference_fn, candidate_fn, random, 1))
    report["results"].append(_exact_case("model_shape_sign_tie_extremes", reference_fn, candidate_fn, _edge_case(), 8))

    # Exercise simultaneous independent OpenMP calls with small matrices so
    # per-invocation barrier state cannot leak across teams.
    small = _random_case(51)
    small_x, small_gu, small_dn = small
    small_x, small_gu, small_dn = small_x[:2, :64].contiguous(), small_gu[:, :32, :64].contiguous(), small_dn[:, :64, :16].contiguous()
    def concurrent_candidate(_):
        return _run(candidate_fn, small_x, small_gu, small_dn, 2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(concurrent_candidate, range(2)))
    concurrent_ref = _run(reference_fn, small_x, small_gu, small_dn, 2)
    concurrent_equal = all(torch.equal(result, concurrent_ref) for result in concurrent)
    report["results"].append({"case": "concurrent_reentrant_small", "threads_per_call": 2,
                              "calls": len(concurrent), "passed": concurrent_equal})
    if not concurrent_equal:
        raise AssertionError("concurrent calls were not bit-identical to baseline")

    report["status"] = "passed"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    # Keep references alive until all native calls are finished.
    del reference_library, candidate_library
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
