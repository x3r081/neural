"""Full-model exactness gate for the review's grouped GEMMs. Invoked by server.py.

Hashes every final hidden-state element, every prompt position's logits and all K/V
buffers, including unused ring slots. No new checkpoint, routing, precision or context
limit is substituted. Hashing and the logits audit are outside the prefill timing.
"""
import hashlib
import json
import os
from pathlib import Path
import time


def _hash_tensor(tensor):
    import torch
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def run(s):
    import torch
    args, mx = s["A"], s["_MXK"]
    if not hasattr(mx, "PREFILL_GROUPED_GEMM"):
        raise RuntimeError("grouped GEMM implementation is not installed")
    if not args.fast_prefill or not mx.FAST_LAUNCH:
        raise ValueError("grouped dispatch proof requires --fast-prefill 1 and NEURAL_FAST_LAUNCH=1")
    root = Path(s["S"])
    control = root.parent / "neuralserver-base"
    parts = [(control / p).read_text(encoding="utf-8") for p in
             ("fused_core.py", "harmony_render.py", "cpu_prefill.py", "server.py")]
    source = "\n\n".join(parts)
    ids = s["tok"](source, add_special_tokens=False).input_ids
    lengths = [int(n) for n in args.review_prefill_lengths.split(",")]
    group_sizes = [int(n) for n in args.review_prefill_groups.split(",")]
    if not group_sizes or min(group_sizes) < 1:
        raise ValueError("expert group sizes must be positive")
    if not lengths or min(lengths) < 1 or max(lengths) >= args.smax - 16:
        raise ValueError("review lengths must be positive and fit the configured context")
    if max(lengths) > len(ids):
        raise ValueError("source does not contain the requested number of prompt tokens")
    s["setup_levers"]()
    if args.prewarm:
        s["prewarm"]()
    if args.ws_min:
        s["hold_working_set"]()
    before_slots = dict(s["rt"].slot_of)
    original = (mx.PREFILL_GROUP, mx.PREFILL_GROUPED_GEMM, mx.PREFILL_GROUP_EXPERTS)
    out = {"evidence": "MEASURED", "model_path": args.model_dir,
           "store": args.store_dir, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "chunk_size": args.prefill_chunk, "block_m": mx.PREFILL_BLOCK_M,
           "group_row_cap": mx.PREFILL_GROUP_ROWS,
           "group_max_tokens": args.grouped_prefill_max_tokens,
           "group_min_tokens": args.grouped_prefill_min_tokens,
           "lengths": lengths, "group_sizes": group_sizes, "runs": [], "all_equal": False}
    dest = Path(args.review_prefill_check)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        os.replace(tmp, dest)

    def one(n, group, grouped, label):
        mx.PREFILL_GROUP, mx.PREFILL_GROUPED_GEMM = group, grouped
        for layer in s["LAY"]:
            layer.k.zero_()
            layer.v.zero_()
        s["PSTAT"].update(res=0, cpu=0, stg=0, stg_pairs=0)
        hidden = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        last = s["prefill"](ids[:n], 0, order="layer", hidden_out=hidden)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        hh, lh = [], []
        for block in hidden:
            hh.append(_hash_tensor(block))
            for j in range(0, block.shape[0], 128):
                lh.append(_hash_tensor(s["_logits"](block[j:j + 128])))
        kv = [(_hash_tensor(layer.k), _hash_tensor(layer.v)) for layer in s["LAY"]]
        row = {"label": label, "tokens": n, "prefill_s": elapsed,
               "hidden_sha256": hh, "all_position_logits_sha256": lh,
               "last_logits_sha256": _hash_tensor(last), "kv_sha256": kv,
               "routes": dict(s["PSTAT"]), "residency_unchanged": dict(s["rt"].slot_of) == before_slots,
               "epilogue_groups": s["PTIME"]["n_epi_groups"]}
        eligible = (args.grouped_prefill_max_tokens == 0
                    or args.grouped_prefill_min_tokens <= n <= min(args.prefill_chunk, args.grouped_prefill_max_tokens))
        expected_grouped = grouped and eligible and (row["routes"]["res"] + row["routes"]["stg_pairs"] > 0)
        row["grouped_path_expected"] = expected_grouped
        row["grouped_dispatch_correct"] = (row["epilogue_groups"] > 0) == expected_grouped
        del hidden, last, block
        torch.cuda.empty_cache()
        out["runs"].append(row)
        save()
        print("REVIEW_PREFILL", label, n, round(elapsed, 4), flush=True)
        if not row["grouped_dispatch_correct"]:
            raise AssertionError(f"unexpected grouped dispatch for {label} at {n} tokens")
        return row

    try:
        with torch.inference_mode():
            # Each production candidate is compared directly to the current ungrouped production path.
            for n in lengths:
                ref = one(n, False, False, "production")
                for size in group_sizes:
                    mx.PREFILL_GROUP_EXPERTS = size
                    alt = one(n, True, True, f"grouped_gemm_g{size}")
                    fields = ("hidden_sha256", "all_position_logits_sha256", "last_logits_sha256", "kv_sha256")
                    equal = all(ref[k] == alt[k] for k in fields)
                    routes = all(ref["routes"][k] == alt["routes"][k] for k in ("res", "cpu", "stg_pairs"))
                    alt["bit_equal_to_production"] = equal
                    alt["same_route_counts"] = routes
                    save()
                    if not (equal and routes and alt["residency_unchanged"]):
                        raise AssertionError(f"full-model grouped prefill mismatch at {n} tokens, group {size}")
            out["all_equal"] = True
            save()
    except BaseException as exc:
        out["error"] = repr(exc)
        save()
        raise
    finally:
        mx.PREFILL_GROUP, mx.PREFILL_GROUPED_GEMM, mx.PREFILL_GROUP_EXPERTS = original
    print("REVIEW_PREFILL_ALL_BIT_EQUAL", flush=True)
