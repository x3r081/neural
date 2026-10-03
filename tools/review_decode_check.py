"""Full-model frozen-residency exactness gate for masked decode GEMV.

Invoked by server.py after model, pool and production graphs are initialized. It
replays the same fixed prompt/decode inputs through the already-captured default
graphs and graphs recaptured with --masked-gemv 1. Hashing is deliberately part
of this check, not a performance measurement.
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


def _hash_array(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _clear_kv(s):
    for layer in s["LAY"]:
        layer.k.zero_()
        layer.v.zero_()
    s["CACHE"].clear()
    s["GEN"][:] = 0
    s["PREF"][:] = 0
    if "COUNTS" in s:
        s["COUNTS"][:] = 0
    s["CAND"].clear()
    s["PEND"].clear()
    s["PENDG"].clear()
    s["RC"].update(captures=0, cand_marked=0)
    s["reset"]()


def _fingerprint(s):
    import numpy as np
    rt = s["rt"]
    return {"slot_of": dict(rt.slot_of), "TAB": np.asarray(s["TAB"]).copy(),
            "OWNER": np.asarray(s["OWNER"]).copy()}


def _json_slot_map(mapping):
    rows = []
    for key, value in mapping.items():
        key = key if isinstance(key, tuple) else (key,)
        rows.append([*(int(part) for part in key), int(value)])
    return sorted(rows)


def _same_fingerprint(before, after):
    import numpy as np
    return (before["slot_of"] == after["slot_of"] and
            np.array_equal(before["TAB"], after["TAB"]) and
            np.array_equal(before["OWNER"], after["OWNER"]))


def _finite(tensor, label):
    import torch
    if not bool(torch.isfinite(tensor).all().item()):
        raise AssertionError(f"non-finite values in {label}")


def _run_one(s, prompt_ids, n_decode, label, frozen, decode_setup=None):
    import torch
    args = s["A"]
    _clear_kv(s)
    torch.cuda.synchronize()
    decode_flag = args.masked_gemv
    args.masked_gemv = 0       # hold prompt processing to the same implementation in both legs
    start = time.perf_counter()
    logits = s["prefill"](prompt_ids, 0)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - start
    _finite(logits, f"{label} prompt logits")
    prompt_token = int(logits.argmax().item())
    prompt_logits_sha = _hash_tensor(logits)
    del logits
    prompt_routes = s["PREF"].copy()
    prompt_routes_sha = _hash_array(prompt_routes)
    s["fold_prefill_counts"](len(prompt_ids))
    args.masked_gemv = decode_flag
    if decode_setup is not None:
        decode_setup()  # CPU-DLL comparisons keep the reference path for both prefills.

    # The seeded first output comes from prompt logits; subsequent outputs come
    # from exactly n_decode calls to the production decode_token implementation.
    current = prompt_token
    tokens, hidden_hashes, logits_hashes = [], [], []
    actual_ids_hashes = []
    token_t0 = time.perf_counter()
    for i in range(n_decode):
        actual = int(s["decode_token"](current, len(prompt_ids) + i))
        final_hidden = s["mid"] + (s["g_out"] + s["c_out"].to(s["bf"]))
        full_logits = s["_logits"](final_hidden)
        torch.cuda.synchronize()
        _finite(final_hidden, f"{label} hidden state at decode {i}")
        _finite(full_logits, f"{label} logits at decode {i}")
        pred = int(full_logits.argmax().item())
        if pred != actual:
            raise AssertionError(f"{label} decode {i}: full-logits argmax {pred} != actual id {actual}")
        tokens.append(actual)
        actual_ids_hashes.append(hashlib.sha256(int(actual).to_bytes(8, "little", signed=False)).hexdigest())
        hidden_hashes.append(_hash_tensor(final_hidden))
        logits_hashes.append(_hash_tensor(full_logits))
        current = actual
        del final_hidden, full_logits
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - token_t0
    kv_hashes = [{"layer": i, "k_sha256": _hash_tensor(layer.k), "v_sha256": _hash_tensor(layer.v)}
                 for i, layer in enumerate(s["LAY"])]
    finite_kv = [torch.isfinite(t).all() for layer in s["LAY"] for t in (layer.k, layer.v)]
    if not bool(torch.stack(finite_kv).all().item()):
        raise AssertionError(f"{label}: non-finite K/V cache values")
    after = _fingerprint(s)
    unchanged = _same_fingerprint(frozen, after)
    if s["CAND"] or s["PEND"] or s["PENDG"]:
        raise AssertionError(f"{label}: admission state changed; candidate/pending slots are nonempty")
    if s["RC"].get("captures", 0) or s["RC"].get("cand_marked", 0) or \
       s["C"].get("gpuadm", 0) or s["C"].get("gpumiss", 0):
        raise AssertionError(f"{label}: admission or GPU-miss work occurred during the isolated check")
    if not unchanged:
        raise AssertionError(f"{label}: slot mapping/residency changed during frozen-residency check")
    gen = s["GEN"].copy()
    counts = s["COUNTS"].copy() if "COUNTS" in s else None
    if counts is not None:
        routing_sha = _hash_array(counts)
    else:
        routing_sha = None
    row = {"label": label, "prompt_tokens": len(prompt_ids), "decode_calls": n_decode,
           "prefill_s_MEASURED_not_performance_claim": round(prefill_s, 4),
           "decode_s_MEASURED_not_performance_claim": round(decode_s, 4),
           "prompt_next_token": prompt_token, "prompt_logits_sha256": prompt_logits_sha,
           "generated_ids": tokens, "generated_ids_sha256": _hash_array(
               __import__("numpy").asarray(tokens, dtype="<u4")),
           "actual_id_step_sha256": actual_ids_hashes,
           "final_hidden_per_step_sha256": hidden_hashes,
           "full_logits_per_step_sha256": logits_hashes,
           "kv_per_layer_sha256": kv_hashes,
           "prompt_routing_counts": prompt_routes.tolist(),
           "prompt_routing_counts_sha256": prompt_routes_sha,
           "GEN_counts": gen.tolist(), "GEN_counts_sha256": _hash_array(gen),
           "routing_COUNTS": counts.tolist() if counts is not None else None,
           "routing_COUNTS_sha256": routing_sha,
           "counters": dict(s["C"]),
           "captures": s["RC"].get("captures"), "PEND_empty": not bool(s["PEND"]),
           "CAND_empty": not bool(s["CAND"]), "residency_unchanged": unchanged}
    return row


def run(s):
    import torch
    args = s["A"]
    if args.no_fold:
        raise RuntimeError("--review-decode-check requires folded production graphs (no --no-fold)")
    if args.doorbell or args.admit_gpu or args.zc_misses or args.near_miss:
        raise RuntimeError("--review-decode-check isolates masked GEMV; doorbell, GPU admission/misses, and near-miss must be off")
    if getattr(args, "masked_gemv", 0) != 0:
        raise RuntimeError("start the review check with --masked-gemv 0 so installed production graphs are the reference")
    if args.refresh_every:
        raise RuntimeError("--review-decode-check requires --refresh-every 0 (no residency refresh during the check)")
    if s["GB"] is not None or s["GT"] is None or not s["GA"]:
        raise RuntimeError("--review-decode-check requires the folded GA/GT graph path")
    if s["CAND"] or s["PEND"] or s["PENDG"]:
        raise RuntimeError("startup left candidates/admissions pending; exactness check requires empty admission state")

    if s["BELL"] is not None or s["GPU_MISS"] or s["ADMIT_GPU"] or s["ZC_MISSES"] or s["NEAR_MISS"]:
        raise RuntimeError("runtime setup enabled doorbell/admission/GPU-miss/near-miss; disable those features")

    lengths = [int(v.strip()) for v in getattr(args, "review_decode_lengths", "128,3000,13000").split(",")
               if v.strip()]
    if lengths != sorted(set(lengths)) or not lengths or min(lengths) < 1 or max(lengths) >= args.smax - 16:
        raise ValueError("review decode lengths must be unique, ascending, positive and fit --smax")
    n_decode = int(args.review_decode_tokens)
    if n_decode < 1 or max(lengths) + n_decode >= args.smax - 16:
        raise ValueError("review decode token count must be positive and fit after every prompt")

    root = Path(s["S"])
    control = root.parent / "neuralserver-base"
    files = ("fused_core.py", "harmony_render.py", "cpu_prefill.py", "server.py")
    source_parts = [(control / p).read_text(encoding="utf-8") for p in files]
    source = "\n\n".join(source_parts)
    ids = s["tok"](source, add_special_tokens=False).input_ids
    if max(lengths) > len(ids):
        raise ValueError("baseline source does not contain requested prompt lengths")

    # Set up any optional host state before taking the frozen-slot snapshot.
    s["setup_levers"]()
    if s["GPU_MISS"] or s["ADMIT_GPU"] or s["ZC_MISSES"] or s["NEAR_MISS"]:
        raise RuntimeError("runtime setup enabled admission/GPU-miss/near-miss; disable those features")
    if args.prewarm:
        s["prewarm"]()
    if args.ws_min:
        s["hold_working_set"]()
    torch.cuda.synchronize()

    baseline_graphs = s["GA"]
    original_flag = args.masked_gemv
    frozen = _fingerprint(s)
    original_counting = s["MODE"].get("counting", False)
    original_gt = s.get("GT")
    out = {"evidence": "MEASURED", "purpose": "exactness_only_no_performance_claim",
           "model_path": args.model_dir, "store": args.store_dir,
           "baseline_source_files": [{"path": str(control / p),
                                      "sha256": hashlib.sha256((control / p).read_bytes()).hexdigest()}
                                     for p in files],
           "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "masked_gemv_reference": 0, "masked_gemv_candidate": 1,
           "lengths": lengths, "decode_calls_per_length": n_decode,
           "slot_mapping_before": {"slot_of": _json_slot_map(frozen["slot_of"]),
                                   "TAB_sha256": _hash_array(frozen["TAB"]),
                                   "OWNER_sha256": _hash_array(frozen["OWNER"])},
           "runs": [], "all_equal": False}
    dest = Path(args.review_decode_check)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)

    def one(length, label):
        row = _run_one(s, ids[:length], n_decode, label, frozen)
        row["prompt_token_ids_sha256"] = hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little") for token in ids[:length])).hexdigest()
        out["runs"].append(row)
        save()
        print("REVIEW_DECODE", label, length, row["generated_ids_sha256"], flush=True)
        return row

    try:
        s["GEN"][:] = 0
        s["PREF"][:] = 0
        if "COUNTS" in s:
            s["COUNTS"][:] = 0
        s["MODE"]["counting"] = True
        s["GT"] = s["GT_GREEDY"]
        args.masked_gemv = 1
        with torch.inference_mode():
            candidate_graphs = [s["capture"](lambda L=L: s["a_full"](L))
                                for L in range(len(s["LAY"]))]
            torch.cuda.synchronize()
            # Existing captures were installed with masked GEMV disabled.
            s["GA"] = baseline_graphs
            args.masked_gemv = 0
            for length in lengths:
                ref = one(length, "masked_gemv_0")
                args.masked_gemv = 1
                s["GA"] = candidate_graphs
                alt = one(length, "masked_gemv_1")
                fields = ("prompt_next_token", "prompt_logits_sha256", "generated_ids",
                          "prompt_token_ids_sha256",
                          "final_hidden_per_step_sha256", "full_logits_per_step_sha256",
                          "kv_per_layer_sha256", "GEN_counts_sha256", "prompt_routing_counts_sha256",
                          "routing_COUNTS_sha256")
                equal = all(ref[k] == alt[k] for k in fields)
                alt["bit_equal_to_masked_gemv_0"] = equal
                alt["same_routing_counts"] = ref["GEN_counts"] == alt["GEN_counts"] and \
                    ref["routing_COUNTS_sha256"] == alt["routing_COUNTS_sha256"]
                alt["same_residency_mapping"] = ref["residency_unchanged"] and alt["residency_unchanged"]
                save()
                if not (equal and alt["same_routing_counts"] and alt["same_residency_mapping"]):
                    raise AssertionError(f"masked GEMV full-model mismatch at prompt length {length}")
                # Next length starts again from the reference captures, with empty KV and counters.
                args.masked_gemv = 0
                s["GA"] = baseline_graphs
            out["all_equal"] = True
            save()
    except BaseException as exc:
        out["error"] = repr(exc)
        save()
        raise
    finally:
        args.masked_gemv = original_flag
        s["GA"] = baseline_graphs
        s["MODE"]["counting"] = original_counting
        if original_gt is not None:
            s["GT"] = original_gt
    print("REVIEW_DECODE_ALL_BIT_EQUAL", flush=True)
