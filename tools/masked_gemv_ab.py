"""Opt-in zero-weight GEMV skip: complete production b_body equivalence gate.

Uses synthetic native MXFP4 matrices, never a substitute model or checkpoint.
Compare stock and masked GEMVs with the production bf16 bias adds, clamped
SwiGLU closure, routing multiply and original FOUR-row torch.sum. Test all 16
hit masks on signed/zero inputs and mixed/zero biases, raw and packed scales,
then replay each candidate CUDA graph and compare FINAL OUTPUT BITS and hit
rows. Missing slots are mapped to slot 0, exactly as the production router.

  project-python tools/masked_gemv_ab.py --output <json>
  project-python tools/masked_gemv_ab.py --real-shape --time --output <json>

Default hidden width is 128; --real-shape additionally tests H=2880, gate=5760,
down=2880 with three synthetic slots (<100 MiB). Optional graph microtimings
are MEASURED kernel/body costs, not an inference speedup. A forced-NaN negative
control demonstrates why the skip is restricted to finite expert results:
stock NaN*0 stays NaN, whereas a skipped row can become finite. Signed zero in
unused intermediates is allowed to differ; the final four-row sum must match
bitwise on every accepted case. Failure is a hard nonzero exit with saved JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import neural.q80.mxfp4_kernels as mk
from neural.q80.expert_runtime import _make_clamped_swiglu
from grouped_gemm_ab import make_pool

ACT = _make_clamped_swiglu(1.702, 7.0)


def bits(a, b):
    return bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))


def body(x, gb, gs, db, ds, slots, gids, bg, bd, weights, *, masked):
    """Op-for-op server.b_body, exposing unused intermediates for the finite gate."""
    hwidth = x.shape[1]
    if masked:
        gu_raw = mk.mxfp4_gemv_masked(x, gb, gs, slots, weights, block_n=32, block_g=4, num_warps=8)
    else:
        gu_raw = mk.mxfp4_gemv(x, gb, gs, slots, block_n=32, block_g=4, num_warps=8)
    gu = gu_raw.view(4, 2 * hwidth) + bg.index_select(0, gids)
    h = ACT(gu)
    if masked:
        y_raw = mk.mxfp4_gemv_masked(h, db, ds, slots, weights, per_expert_x=True,
                                    block_n=32, block_g=4, num_warps=8)
    else:
        y_raw = mk.mxfp4_gemv(h, db, ds, slots, per_expert_x=True,
                             block_n=32, block_g=4, num_warps=8)
    y = y_raw + bd.index_select(0, gids)
    weighted = y * weights.view(4, 1)
    gout = weighted.sum(0, keepdim=True)
    return {"gu_raw": gu_raw, "gu": gu, "h": h, "y_raw": y_raw,
            "y": y, "weighted": weighted, "g_out": gout}


def graph_timing(fn, iters):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_ms = (time.perf_counter() - t0) * 1000 / iters
    b.record()
    b.synchronize()
    return {"gpu_event_ms_per_body": a.elapsed_time(b) / iters, "host_submit_ms_per_body": host_ms}


def benchmark(args, stock_fn, masked_fn):
    graphs, results = {}, {}
    for name, fn in (("stock", stock_fn), ("masked", masked_fn)):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            results[name] = fn()
        graphs[name] = g
    legs = []
    for r in range(args.repeats):
        for name in (("stock", "masked") if r % 2 == 0 else ("masked", "stock")):
            legs.append({"round": r, "kind": name, **graph_timing(graphs[name].replay, args.iters)})
    med = {name: {metric: statistics.median(v[metric] for v in legs if v["kind"] == name)
                  for metric in ("gpu_event_ms_per_body", "host_submit_ms_per_body")}
           for name in graphs}
    return {"label": "MEASURED", "scope": "synthetic full GPU b_body graph; not token/inference throughput",
            "iterations": args.iters, "legs": legs, "medians": med,
            "stock_over_masked_gpu_time": med["stock"]["gpu_event_ms_per_body"] / med["masked"]["gpu_event_ms_per_body"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.iters < 1 or args.repeats < 2:
        ap.error("iters must be positive, repeats >= 2")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; there is no CPU inference fallback")
    torch.set_num_threads(1)
    dev = torch.device("cuda")
    out = {"label": "MEASURED", "scope": "synthetic complete GPU b_body; no inference speed claimed",
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": mk.triton.__version__,
           "finite_contract": "Original gu/h/y and final weighted results must be finite. Zero-weight intermediate zero signs may differ; original four-row sum and hit rows must match bitwise.",
           "cases": [], "all_finite_cases_bit_equal": False, "nan_scope_negative_control": [],
           "output_bitflip_negative_control": False}
    gen = torch.Generator().manual_seed(20261001)
    dims = [128, 2880] if args.real_shape else [128]
    try:
        for hwidth in dims:
            gb, gs_raw, gs_pk = make_pool(hwidth, 2 * hwidth, dev, 7)
            db, ds_raw, ds_pk = make_pool(hwidth, hwidth, dev, 11)
            gids = torch.tensor([7, 1, 3, 0], device=dev)
            nominal_slots = torch.tensor([2, 0, 2, 1], device=dev)
            for layout, gs, ds in (("raw", gs_raw, ds_raw), ("packed", gs_pk, ds_pk)):
                for bias_kind in ("mixed", "zero"):
                    bg = (torch.randn(8, 2 * hwidth, generator=gen) * 0.2).to(torch.bfloat16).to(dev)
                    bd = (torch.randn(8, hwidth, generator=gen) * 0.05).to(torch.bfloat16).to(dev)
                    if bias_kind == "zero":
                        bg.zero_(); bd.zero_()
                        bg[:, 1::2] = -0.0; bd[:, 1::2] = -0.0
                    base = torch.randn(1, hwidth, generator=gen).to(torch.bfloat16).to(dev)
                    for input_kind in ("mixed", "positive", "negative", "zero"):
                        x = base.clone()
                        if input_kind == "positive":
                            x.abs_()
                        elif input_kind == "negative":
                            x.abs_().neg_()
                        elif input_kind == "zero":
                            x.zero_(); x[:, 1::2] = -0.0
                        for mask in range(16):
                            hit_host = torch.tensor([bool(mask & (1 << j)) for j in range(4)])
                            hit = hit_host.to(dev)
                            weights = torch.tensor([0.125, 0.25, 0.375, 0.5], dtype=torch.bfloat16, device=dev)
                            weights[~hit] = 0.0
                            # Exercise both signs of exactly zero inactive weights.
                            if mask % 2:
                                weights[~hit] = -0.0
                            slots = torch.where(hit, nominal_slots, torch.zeros_like(nominal_slots))
                            params = (x, gb, gs, db, ds, slots, gids, bg, bd, weights)
                            stock_fn = lambda params=params: body(*params, masked=False)
                            masked_fn = lambda params=params: body(*params, masked=True)
                            ref, alt = stock_fn(), masked_fn()
                            torch.cuda.synchronize()
                            finite = all(bool(torch.isfinite(v).all()) for v in ref.values())
                            hit_equal = all(bits(ref[key][hit], alt[key][hit])
                                            for key in ("gu_raw", "gu", "h", "y_raw", "y", "weighted"))
                            final_equal = bits(ref["g_out"], alt["g_out"])
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                captured = masked_fn()
                            g.replay()
                            torch.cuda.synchronize()
                            graph_equal = bits(ref["g_out"], captured["g_out"])
                            row = {"H": hwidth, "layout": layout, "bias": bias_kind, "input": input_kind,
                                   "hit_mask": mask, "finite": finite, "hit_rows_bit_equal": hit_equal,
                                   "g_out_bit_equal": final_equal, "graph_g_out_bit_equal": graph_equal}
                            if args.time and input_kind == "mixed" and bias_kind == "mixed" and mask in (0, 1, 3, 7, 15):
                                row["timing"] = benchmark(args, stock_fn, masked_fn)
                            out["cases"].append(row)
                            if not (finite and hit_equal and final_equal and graph_equal):
                                raise AssertionError(f"masked GEMV finite/identity gate failed: {row}")
                            # Equality test must detect a one-bit output corruption.
                            if not out["output_bitflip_negative_control"]:
                                bad = alt["g_out"].clone()
                                bad.view(torch.int16).reshape(-1)[0].bitwise_xor_(1)
                                out["output_bitflip_negative_control"] = not bits(ref["g_out"], bad)
                                if not out["output_bitflip_negative_control"]:
                                    raise AssertionError("output bit-flip was not detected")
                            del ref, alt, captured, g
                        print(f"PASS H={hwidth} {layout} {bias_kind} {input_kind}: all16 masks", flush=True)
                    # Scope control: all-miss NaN input is deliberately outside
                    # the finite contract and MUST differ, preventing a false
                    # claim that skipping0*NaN is universally IEEE-equivalent.
                    nan_x = torch.full((1, hwidth), float("nan"), dtype=torch.bfloat16, device=dev)
                    zeros = torch.zeros(4, dtype=torch.bfloat16, device=dev)
                    zero_slots = torch.zeros(4, dtype=torch.long, device=dev)
                    params = (nan_x, gb, gs, db, ds, zero_slots, gids, bg, bd, zeros)
                    ref, alt = body(*params, masked=False), body(*params, masked=True)
                    torch.cuda.synchronize()
                    noticed = bool(torch.isnan(ref["g_out"]).any()) and bool(torch.isfinite(alt["g_out"]).all())
                    out["nan_scope_negative_control"].append({"H": hwidth, "layout": layout,
                                                               "bias": bias_kind, "expected_mismatch_detected": noticed})
                    if not noticed:
                        raise AssertionError("NaN scope negative control did not demonstrate finite restriction")
                    del ref, alt, nan_x, bg, bd, base
            del gb, gs_raw, gs_pk, db, ds_raw, ds_pk
        out["all_finite_cases_bit_equal"] = True
        out["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    except BaseException as exc:
        out["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {len(out['cases'])} finite cases, graph replay and negative controls", flush=True)


if __name__ == "__main__":
    main()
