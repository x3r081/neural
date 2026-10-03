"""Frozen-residency, full-model CPU-DLL comparison; never an inference speed test."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

from tools.review_decode_check import _fingerprint, _hash_array, _json_slot_map, _run_one


def run(s):
    import torch

    args = s["A"]
    if args.kernel_persistent or args.masked_gemv or args.refresh_every:
        raise RuntimeError("CPU check starts with kernel-persistent=0, masked-gemv=0, refresh-every=0")
    if not args.kernel_fuse or args.no_fold or args.doorbell or args.admit_gpu or args.zc_misses or args.near_miss:
        raise RuntimeError("CPU check requires fused CPU/folded GPU paths and no alternate admission/doorbell")
    if s["GB"] is not None or not s["GA"] or s["GT"] is None:
        raise RuntimeError("CPU check requires production folded graphs")
    lengths = [int(v) for v in args.review_decode_lengths.split(",")]
    n_decode = args.review_decode_tokens
    if not lengths or lengths != sorted(set(lengths)) or min(lengths) < 1 or n_decode < 1 or max(lengths) + n_decode >= args.smax - 16:
        raise ValueError("positive unique ascending prompt lengths and decode span must fit context")

    root = Path(s["S"])
    reference_path = (root / args.kdll).resolve()
    candidate_path = (root / args.review_cpu_dll).resolve()
    if candidate_path == reference_path:
        raise ValueError("reference and candidate DLL paths must differ")
    ref = s["lib"]
    candidate = ctypes.CDLL(str(candidate_path))
    vp = ctypes.c_void_p
    candidate.gptoss_experts.argtypes = [ctypes.c_int, vp, vp, vp, vp, vp, vp, vp, ctypes.c_int]
    candidate.gptoss_experts_cap.argtypes = [ctypes.c_int, vp, vp, vp, vp, vp, vp, vp, ctypes.c_int, vp]
    s["_CP"].bind_scale_layout(candidate, s["SCALE_MODE"], s["SLOTB"], str(candidate_path))
    candidate.gptoss_set_tuning.argtypes = [ctypes.c_int] * 3
    candidate.gptoss_set_tuning(args.kernel_prefetch, args.kernel_pair, args.kernel_affinity)
    candidate.gptoss_set_fuse.argtypes = [ctypes.c_int]
    candidate.gptoss_set_fuse(args.kernel_fuse)
    if hasattr(candidate, "gptoss_set_cold_prefetch"):
        candidate.gptoss_set_cold_prefetch.argtypes = [ctypes.c_int, ctypes.c_int]
        candidate.gptoss_set_cold_prefetch(args.kernel_cold_prefetch, 0)
    elif args.kernel_cold_prefetch:
        raise RuntimeError("candidate cannot reproduce reference cold-prefetch setting")
    candidate.gptoss_set_persistent.argtypes = [ctypes.c_int]
    candidate.gptoss_set_persistent.restype = ctypes.c_int
    candidate.gptoss_persistent_shutdown.argtypes = []
    candidate.gptoss_persistent_shutdown.restype = ctypes.c_int
    candidate.gptoss_persistent_status.argtypes = [vp]
    candidate.gptoss_persistent_status.restype = None
    if candidate.gptoss_set_persistent(1):
        raise RuntimeError("candidate persistent mode could not be enabled")

    dest = Path(args.review_cpu_decode_check)
    if dest.exists():
        candidate.gptoss_persistent_shutdown()
        raise FileExistsError(f"use a fresh result path: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    source_files = ("fused_core.py", "harmony_render.py", "cpu_prefill.py", "server.py")
    control = root.parent / "neuralserver-base"
    source = "\n\n".join((control / name).read_text(encoding="utf-8") for name in source_files)
    ids = s["tok"](source, add_special_tokens=False).input_ids
    if len(ids) < max(lengths):
        raise ValueError("control source too short")
    out = {"evidence": "MEASURED", "purpose": "exactness_only_no_performance_claim",
           "model_path": args.model_dir, "store": args.store_dir,
           "reference_dll": str(reference_path), "candidate_dll": str(candidate_path),
           "reference_dll_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
           "candidate_dll_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
           "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "lengths": lengths, "decode_calls_per_length": n_decode,
           "runs": [], "all_equal": False}

    def save():
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)

    def select(lib):
        s["lib"] = lib

    def status():
        values = (ctypes.c_uint32 * 8)()
        candidate.gptoss_persistent_status(values)
        return {"enabled": int(values[0]), "team_threads": int(values[1]),
                "worker_handles": int(values[2]), "wait_on_address": int(values[3]),
                "math_jobs": int(values[4]) | (int(values[5]) << 32),
                "fallback_calls": int(values[6]) | (int(values[7]) << 32)}

    old_counting = s["MODE"]["counting"]
    old_gt = s["GT"]
    try:
        s["setup_levers"]()
        if s["BELL"] is not None or s["GPU_MISS"] or s["ADMIT_GPU"] or s["ZC_MISSES"] or s["NEAR_MISS"]:
            raise RuntimeError("alternate execution or placement path enabled")
        if s["CAND"] or s["PEND"] or s["PENDG"]:
            raise RuntimeError("admissions already pending")
        if args.prewarm:
            s["prewarm"]()
        if args.ws_min:
            s["hold_working_set"]()
        frozen = _fingerprint(s)
        out["slot_mapping_before"] = {"slot_of": _json_slot_map(frozen["slot_of"]),
                                       "TAB_sha256": _hash_array(frozen["TAB"]),
                                       "OWNER_sha256": _hash_array(frozen["OWNER"])}
        s["MODE"]["counting"] = True
        s["GT"] = s["GT_GREEDY"]
        with torch.inference_mode():
            for length in lengths:
                pair = []
                for label, lib in (("stock_cpu", ref), ("persistent_cpu", candidate)):
                    select(ref)  # Both prefills always use the original DLL.
                    before = status()
                    row = _run_one(s, ids[:length], n_decode, label, frozen,
                                   decode_setup=lambda lib=lib: select(lib))
                    after = status()
                    row["persistent_status_before"] = before
                    row["persistent_status_after"] = after
                    if label == "persistent_cpu":
                        if after["math_jobs"] <= before["math_jobs"] or after["fallback_calls"] != before["fallback_calls"]:
                            raise AssertionError("candidate must execute persistent jobs without fallback")
                        if after["team_threads"] != args.threads or not after["enabled"]:
                            raise AssertionError("candidate team configuration does not match requested threads")
                    out["runs"].append(row)
                    pair.append(row)
                    save()
                    print("REVIEW_CPU_DECODE", label, length, row["generated_ids_sha256"], flush=True)
                fields = ("prompt_next_token", "prompt_logits_sha256", "generated_ids",
                          "final_hidden_per_step_sha256", "full_logits_per_step_sha256",
                          "kv_per_layer_sha256", "GEN_counts_sha256", "prompt_routing_counts_sha256",
                          "routing_COUNTS_sha256", "residency_unchanged")
                pair[1]["bit_equal_to_stock_cpu"] = all(pair[0][k] == pair[1][k] for k in fields)
                save()
                if not pair[1]["bit_equal_to_stock_cpu"]:
                    raise AssertionError(f"CPU DLL full-model mismatch at prompt length {length}")
            out["all_equal"] = True
            save()
    except BaseException as exc:
        out["error"] = repr(exc)
        save()
        raise
    finally:
        select(ref)
        s["MODE"]["counting"] = old_counting
        s["GT"] = old_gt
        candidate.gptoss_persistent_shutdown()
    print("REVIEW_CPU_DECODE_ALL_BIT_EQUAL", flush=True)
