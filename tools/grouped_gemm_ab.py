"""Small-footprint GPU identity/microbenchmark for ragged grouped MXFP4 GEMMs.

No model/store is opened. Synthetic raw and byte-exact packed scales exercise
repeated slots, ragged M around tile boundaries, partial N, offset pointers and
gapped output slices. CUDA graph replay is checked as well. Identity is against
the STOCK per-pair kernel with the SAME BM/BN, not a float reference.

  project-python tools/grouped_gemm_ab.py --output <json>
  project-python tools/grouped_gemm_ab.py --real-shape --time --output <json>

Default correctness dimensions are small. --real-shape adds gpt-oss gate/down
dimensions (K=2880, N=5760/2880) with three synthetic slots, <100 MiB per case.
Timing reports both prebuilt-metadata launches and end-to-end metadata creation;
the latter is the relevant host overhead for changing prefill routes. Repeats
alternate AB/BA and report GPU event time separately from host submission time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import neural.q80.mxfp4_kernels as mk


def make_pool(k, n, device, seed):
    """Three slot-strided matrices; pad both regions to exercise slot stride."""
    rng = np.random.default_rng(seed)
    gk = k // 32
    cap = 3
    codes = rng.integers(0, 256, (cap, n, gk, 16), dtype=np.uint8)
    scales = rng.integers(0, 16, (cap, n, gk), dtype=np.uint8) + np.uint8(116)
    scales[:, 0] = 120
    if n > 1:
        scales[:, 1, ::2] = 116
        scales[:, 1, 1::2] = 131
    # scale_pack.pack_scale_rows is deliberately fixed to the real GK=90;
    # this probe also uses tiny GK=4 matrices. Apply the same exact format for
    # every even GK: row minimum followed by low/high 4-bit delta pairs.
    base = scales.min(axis=-1, keepdims=True)
    delta = scales - base
    assert gk % 2 == 0 and int(delta.max()) <= 15
    packed = np.empty((cap, n, gk // 2 + 1), dtype=np.uint8)
    packed[..., 0] = base[..., 0]
    packed[..., 1:] = delta[..., 0::2] | (delta[..., 1::2] << 4)
    codes_host = np.full((cap, n * gk * 16 + 64), 199, np.uint8)
    codes_host[:, :n * gk * 16] = codes.reshape(cap, -1)
    raw_host = np.full((cap, n * gk + 64), 177, np.uint8)
    raw_host[:, :n * gk] = scales.reshape(cap, -1)
    packed_host = np.full((cap, n * (gk // 2 + 1) + 64), 173, np.uint8)
    packed_host[:, :n * (gk // 2 + 1)] = packed.reshape(cap, -1)
    cb = torch.from_numpy(codes_host).to(device)[:, :n * gk * 16].view(cap, n, gk, 16)
    sr = torch.from_numpy(raw_host).to(device)[:, :n * gk].view(cap, n, gk)
    ps = torch.from_numpy(packed_host).to(device)[:, :n * (gk // 2 + 1)].view(cap, n, gk // 2 + 1)
    return cb, sr, ps


def make_pairs(ms, k, n, device, seed, offsets=True):
    gen = torch.Generator().manual_seed(seed)
    pairs, refs, owners = [], [], []
    used = sum(ms) * n
    # Shared gapped output storage catches unintended writes outside a pair.
    gap = 23 if offsets else 24
    actual = torch.full((used + gap * len(ms) + 31,), -777, dtype=torch.bfloat16, device=device)
    reference = actual.clone()
    cursor = 7 if offsets else 8
    write_mask = torch.zeros(actual.numel(), dtype=torch.bool)
    for j, m in enumerate(ms):
        off = (1, 7, 8, 16)[j % 4] if offsets else 16
        host_x = (torch.randn(m * k + off + 17, generator=gen) * 0.8).to(torch.bfloat16)
        xp = host_x.to(device)
        x = xp[off:off + m * k].view(m, k)
        y = actual[cursor:cursor + m * n].view(m, n)
        ref = reference[cursor:cursor + m * n].view(m, n)
        slot = (2, 0, 2, 1)[j % 4]
        pairs.append((x, slot, y))
        refs.append((x, slot, ref))
        owners.append(xp)
        write_mask[cursor:cursor + m * n] = True
        cursor += m * n + gap
    return pairs, refs, actual, reference, write_mask.to(device), owners


def stock(pairs, blocks, scales, bm, bn):
    for x, slot, y in pairs:
        if x.shape[0]:
            # Views of one common slot table reproduce the server's slot table.
            sl = SLOT_TABLE[slot:slot + 1]
            mk.mxfp4_gemm_into(x, blocks, scales, sl, y, block_m=bm, block_n=bn)


def bits(a, b):
    # Compare storage bits (including signed zero), not torch.equal's float rule.
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def timed(fn, iters):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_s = time.perf_counter() - t0
    b.record()
    b.synchronize()
    return {"gpu_ms": a.elapsed_time(b) / iters, "host_ms": host_s * 1000 / iters}


def bench(pairs, refs, blocks, scales, plan, bm, bn, iters, repeats):
    fns = {"stock": lambda: stock(refs, blocks, scales, bm, bn),
           "grouped_reuse": lambda: mk.mxfp4_gemm_grouped_into(plan, blocks, scales, block_m=bm, block_n=bn),
           "grouped_build": lambda: mk.mxfp4_gemm_grouped_into(pairs, blocks, scales, block_m=bm, block_n=bn)}
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    legs = []
    for r in range(repeats):
        names = list(fns) if r % 2 == 0 else list(reversed(fns))
        for name in names:
            legs.append({"round": r, "kind": name, **timed(fns[name], iters)})
    med = {name: {metric: statistics.median(row[metric] for row in legs if row["kind"] == name)
                  for metric in ("gpu_ms", "host_ms")} for name in fns}
    return {"label": "MEASURED", "iterations": iters, "legs": legs, "medians": med,
            "gpu_ratio_stock_over_grouped_build": med["stock"]["gpu_ms"] / med["grouped_build"]["gpu_ms"]}


def main():
    global SLOT_TABLE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.iters < 1 or args.repeats < 2:
        ap.error("iters must be positive and repeats must be >= 2")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this tool does not fall back to CPU")
    torch.set_num_threads(1)
    dev = torch.device("cuda")
    SLOT_TABLE = torch.arange(3, dtype=torch.long, device=dev)
    shapes = [(128, 160), (128, 128)]
    if args.real_shape:
        shapes += [(2880, 5760), (2880, 2880)]
    out = {"label": "MEASURED", "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "triton": mk.triton.__version__, "pid": os.getpid(), "cases": [], "checks": 0,
           "negative_control": False, "all_equal": True}
    try:
        for k, n in shapes:
            blocks, raw, packed = make_pool(k, n, dev, 41)
            for bm, ms, offsets in ((16, (1, 15, 16, 17, 31), True),
                                    (64, (0, 1, 63, 64, 65, 129), True),
                                    (64, (0, 1, 63, 64, 65, 129), False)):
                pairs, refs, actual, reference, writes, owners = make_pairs(ms, k, n, dev, 7, offsets=offsets)
                plan = mk.GroupedGemmPlan(pairs, block_m=bm)
                expected_aligned = not offsets
                if plan.aligned != expected_aligned:
                    raise AssertionError(f"bad plan alignment classification: {plan.aligned} vs {expected_aligned}")
                for name, scales in (("raw", raw), ("packed", packed)):
                    stock(refs, blocks, scales, bm, 128)
                    mk.mxfp4_gemm_grouped_into(plan, blocks, scales, block_m=bm)
                    torch.cuda.synchronize()
                    equal = bits(actual, reference)
                    finite = bool(torch.isfinite(actual).all())
                    guard = bool((actual[~writes] == -777).all())
                    first = actual.clone()
                    mk.mxfp4_gemm_grouped_into(plan, blocks, scales, block_m=bm)
                    torch.cuda.synchronize()
                    deterministic = bits(first, actual)
                    # Capture replays use uploaded descriptors with stable owners.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        mk.mxfp4_gemm_grouped_into(plan, blocks, scales, block_m=bm)
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_equal = bits(actual, reference)
                    row = {"K": k, "N": n, "BM": bm, "BN": 128, "M": list(ms), "layout": name,
                           "aligned": plan.aligned, "tiles": plan.ntiles, "equal": equal,
                           "finite": finite, "guard_intact": guard,
                           "deterministic": deterministic, "graph_equal": graph_equal}
                    out["checks"] += 5
                    out["all_equal"] &= equal and finite and guard and deterministic and graph_equal
                    if args.time and bm == 64:
                        row["timing"] = bench(pairs, refs, blocks, scales, plan, bm, 128, args.iters, args.repeats)
                    out["cases"].append(row)
                    print(json.dumps(row), flush=True)
                    del graph, first
                # Corrupt a descriptor output cell: equality MUST detect it.
                first_real = next(y for x, slot, y in pairs if x.shape[0])
                first_real.view(torch.int16).reshape(-1)[0].bitwise_xor_(1)
                out["negative_control"] = not bits(actual, reference)
                if not out["negative_control"]:
                    raise AssertionError("negative control did not detect a modified output")
                del plan, pairs, refs, actual, reference, owners, writes
            del blocks, raw, packed
        out["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        if not out["all_equal"]:
            raise AssertionError("a grouped GEMM identity check failed")
    except BaseException as exc:
        out["error"] = repr(exc)
        out["all_equal"] = False
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {out['checks']} checks; negative control detected; peak {out['peak_allocated_mib']:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
