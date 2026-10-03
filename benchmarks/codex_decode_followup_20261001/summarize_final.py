"""Read-only, reproducible summary of the final matched decode trials.

This script intentionally does not start a model or modify benchmark artifacts.
It exits with an error if any replay archive is missing or incomplete.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import review_bench  # noqa: E402

HERE = Path(__file__).resolve().parent
FROZEN_PATH = HERE / "persistent_cpu_frozen.json"
LLAMA_PATH = HERE / "persistent_vs_llama.json"
MIRRORED_PATH = HERE / "persistent_cpu_mirrored.json"


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def med(values):
    vals = [float(x) for x in values if numeric(x)]
    return round(statistics.median(vals), 6) if vals else None


def turn_hash(turn, key):
    value = turn.get(key) or (turn.get("neural") or {}).get(key)
    return value if isinstance(value, str) and value else None


def turn_value(turn, key, *aliases):
    neural = turn.get("neural") or {}
    for name in (key, *aliases):
        value = turn.get(name)
        if value is None:
            value = neural.get(name)
        if value is not None:
            return value
    return None


def load_bundle(path, expected_sessions):
    if not path.is_file():
        raise ValueError(f"missing result file: {path}")
    result = review_bench.read(str(path))
    archive = review_bench.archive_for(str(path), result)
    if archive is None:
        raise ValueError(f"missing replay archive: {path}.replay.json")
    summary, integrity = review_bench.aggregate(result, archive)
    archive_sessions = archive.get("sessions", [])
    requests = [r for session in archive_sessions for r in session.get("requests", [])]
    capture = integrity.get("archive_capture_state") or {}
    expected_requests = expected_sessions * 4
    if (len(archive_sessions) != expected_sessions or len(requests) != expected_requests or
            capture.get("complete") != expected_requests or capture.get("failed") != 0 or
            capture.get("pending") != 0 or capture.get("legacy_unknown") != 0):
        raise ValueError(f"incomplete replay archive: {path} (sessions={len(archive_sessions)}, "
                         f"requests={len(requests)}, capture={capture})")
    if not all(integrity.get(k) is True for k in (
            "request_archive_integrity", "response_archive_integrity", "result_archive_link_integrity")):
        raise ValueError(f"archive integrity failed: {path}: {integrity}")
    if any(not isinstance(r.get("request_sha256"), str) or not r.get("request_sha256")
           for r in requests):
        raise ValueError(f"request hash missing: {path}")
    if any(r.get("status", "complete") != "complete" for r in requests):
        raise ValueError(f"non-complete archived request: {path}")
    return {"path": path, "result": result, "archive": archive,
            "summary": summary, "integrity": integrity}


def weighted(summary, kind, key="all_turns_weighted_decode"):
    row = (summary.get(kind) or {}).get(key) or {}
    return {"tokens": row.get("tokens", 0), "seconds": row.get("seconds"),
            "tok_s": row.get("tok_s")}


def weighted_ratio(new, base):
    nt, ns = new.get("tokens"), new.get("seconds")
    bt, bs = base.get("tokens"), base.get("seconds")
    if not all(numeric(x) for x in (nt, ns, bt, bs)) or ns <= 0 or bs <= 0 or bt <= 0:
        return {"ratio": None, "percent_change": None}
    value = (nt * bs) / (ns * bt)
    return {"ratio": round(value, 8), "percent_change": round((value - 1) * 100, 6)}


def ids_by_turn(sessions, session_filter=None):
    chosen = [s for s in sessions if session_filter is None or session_filter(s)]
    count = max((len(s.get("turns", [])) for s in chosen), default=0)
    report = []
    for i in range(count):
        pids = [turn_hash(s["turns"][i], "prompt_ids_sha256")
                for s in chosen if len(s.get("turns", [])) > i]
        gids = [turn_hash(s["turns"][i], "generated_ids_sha256")
                for s in chosen if len(s.get("turns", [])) > i]
        pknown, gknown = [x for x in pids if x], [x for x in gids if x]
        report.append({"turn": i, "samples": len(pids), "prompt_hashes_present": len(pknown),
                       "generated_hashes_present": len(gknown),
                       "prompt_distinct_hashes": len(set(pknown)),
                       "generated_distinct_hashes": len(set(gknown)),
                       "prompt_identical": (len(pids) == len(chosen) and len(pknown) == len(chosen) and
                                            len(set(pknown)) == 1),
                       "generated_identical": (len(gids) == len(chosen) and len(gknown) == len(chosen) and
                                               len(set(gknown)) == 1)})
    return report


def request_hashes(archive):
    return [[r.get("request_sha256") for r in s.get("requests", [])]
            for s in archive.get("sessions", [])]


def context_identity(bundle):
    context = ((bundle["result"].get("provenance") or {}).get("context") or {})
    return {"ctx_token_ids_sha256": context.get("ctx_token_ids_sha256"),
            "decoded_context_sha256": context.get("decoded_context_sha256"),
            "ctx_token_count": context.get("ctx_token_count"),
            "files": [{"path": f.get("path"), "sha256": f.get("sha256")}
                      for f in context.get("files", [])]}


def selected_dll(bundle, kind):
    roots = (bundle["result"].get("provenance") or {}).get("neural_server_roots") or {}
    return ((roots.get(kind) or {}).get("artifacts") or {}).get("selected_decode_dll") or {}


def equal_dll_hashes(records):
    hashes = [r.get("sha256") for r in records]
    return bool(hashes) and all(isinstance(h, str) and len(h) == 64 and
        all(c in "0123456789abcdef" for c in h) for h in hashes) and len(set(hashes)) == 1


def metrics_by_kind(bundle):
    output = {}
    for kind, row in bundle["summary"].items():
        sessions = [s for s in bundle["result"].get("sessions", []) if s.get("kind") == kind]
        turns = [t for s in sessions for t in s.get("turns", [])]
        first_turns = [s["turns"][0] for s in sessions if s.get("turns")]
        followup_turns = [t for s in sessions for t in s.get("turns", [])[1:]]
        stages = {
            "client_wall_s": med([turn_value(t, "wall_s") for t in turns]),
            "generation_wall_s": med([turn_value(t, "generation_wall_s") for t in turns]),
            "prefill_prompt_s": med([turn_value(t, "prompt_s") for t in turns]),
            "prompt_tokens": med([turn_value(t, "prompt_tokens") for t in turns]),
            "vram_reserved_gib": med([turn_value(t, "vram_reserved_gib") for t in turns]),
            "cpu_ms_per_tok": med([turn_value(t, "cpu_ms_per_tok") for t in turns]),
            "replay_ms_per_tok": med([turn_value(t, "replay_ms_per_tok") for t in turns]),
            "gpu_wait_ms_per_tok": med([turn_value(t, "gpu_wait_ms_per_tok") for t in turns]),
            "admission_ms_per_tok": med([turn_value(t, "admission_ms_per_tok") for t in turns]),
            "other_ms_per_tok": med([turn_value(t, "other_ms_per_tok") for t in turns]),
        }
        output[kind] = {"sessions": len(sessions), "turns": len(turns),
                        "all_turns_weighted_decode": weighted(bundle["summary"], kind),
                        "first_weighted_decode": weighted(bundle["summary"], kind, "first_weighted_decode"),
                        "followup_weighted_decode": weighted(bundle["summary"], kind,
                                                              "followup_weighted_decode"),
                        "first_decode_tok_s_median": row.get("first_decode_tok_s_median"),
                        "followup_decode_tok_s_median": row.get("followup_decode_tok_s_median"),
                        "first_turn_medians": {
                            "client_wall_s": med([turn_value(t, "wall_s") for t in first_turns]),
                            "prefill_prompt_s": med([turn_value(t, "prompt_s") for t in first_turns]),
                        },
                        "followup_turn_medians": {
                            "client_wall_s": med([turn_value(t, "wall_s") for t in followup_turns]),
                            "prefill_prompt_s": med([turn_value(t, "prompt_s") for t in followup_turns]),
                        },
                        "stage_medians": stages}
    return output


def request_match_across(bundles):
    reports = []
    for turn in range(4):
        hashes = []
        counts = {}
        for name, bundle in bundles.items():
            rows = request_hashes(bundle["archive"])
            vals = [row[turn] for row in rows if len(row) > turn]
            counts[name] = len(vals)
            hashes.extend(vals)
        known = [h for h in hashes if h]
        reports.append({"turn": turn, "request_count_by_file": counts,
                        "hashes_present": len(known), "hashes_expected": len(hashes),
                        "distinct_request_hashes": len(set(known)),
                        "matched_across_files_and_sessions": bool(hashes) and
                            len(known) == len(hashes) and len(set(known)) == 1})
    return reports


def summarize():
    frozen = load_bundle(FROZEN_PATH, 4)
    llama = load_bundle(LLAMA_PATH, 4)
    mirrored = load_bundle(MIRRORED_PATH, 6)
    frozen_sessions = frozen["result"].get("sessions", [])
    llama_sessions = llama["result"].get("sessions", [])
    mirrored_sessions = mirrored["result"].get("sessions", [])
    frozen_ids = ids_by_turn(frozen_sessions)
    llama_opt_ids = ids_by_turn(llama_sessions, lambda s: s.get("kind") == "opt")
    llama_prompt_ids = ids_by_turn(llama_sessions)
    contexts = {"frozen": context_identity(frozen), "mirrored": context_identity(mirrored),
                "llama_comparison": context_identity(llama)}
    context_ids = [v.get("ctx_token_ids_sha256") for v in contexts.values()]
    decoded_context_ids = [v.get("decoded_context_sha256") for v in contexts.values()]
    req_match = request_match_across({"frozen": frozen, "mirrored": mirrored, "llama": llama})
    candidate_dlls = {"frozen_cpu": selected_dll(frozen, "cpu"),
                      "mirrored_cpu": selected_dll(mirrored, "cpu"),
                      "mirrored_opt": selected_dll(mirrored, "opt"),
                      "fresh_opt": selected_dll(llama, "opt")}
    control_dlls = {"frozen_off": selected_dll(frozen, "off"),
                    "mirrored_off": selected_dll(mirrored, "off")}

    frozen_metrics = metrics_by_kind(frozen)
    llama_metrics = metrics_by_kind(llama)
    comparisons = {
        "cpu_vs_off": weighted_ratio(frozen_metrics.get("cpu", {}).get("all_turns_weighted_decode", {}),
                                      frozen_metrics.get("off", {}).get("all_turns_weighted_decode", {})),
        "opt_vs_llama": weighted_ratio(llama_metrics.get("opt", {}).get("all_turns_weighted_decode", {}),
                                        llama_metrics.get("llama", {}).get("all_turns_weighted_decode", {})),
    }

    def persistent_checks(sessions, kind):
        checks = []
        for session in sessions:
            if session.get("kind") != kind:
                continue
            for i, turn in enumerate(session.get("turns", [])):
                neural = turn.get("neural") or {}
                jobs = neural.get("persistent_cpu_jobs", turn.get("persistent_cpu_jobs"))
                fallback = neural.get("persistent_cpu_fallback_calls", turn.get("persistent_cpu_fallback_calls"))
                threads = neural.get("persistent_cpu_team_threads", turn.get("persistent_cpu_team_threads"))
                checks.append({"tag": session.get("tag"), "turn": i, "jobs": jobs,
                               "fallback_calls": fallback, "team_threads": threads,
                               "pass": numeric(jobs) and jobs > 0 and numeric(fallback) and fallback == 0 and
                                      numeric(threads) and threads == 8})
        return checks

    cpu_checks = persistent_checks(frozen_sessions, "cpu")
    opt_checks = persistent_checks(llama_sessions, "opt")
    kinds_frozen = {s.get("kind") for s in frozen_sessions}
    kinds_llama = {s.get("kind") for s in llama_sessions}
    complete_gates = {
        "candidate_dll_hashes_present_and_identical_across_trials": equal_dll_hashes(list(candidate_dlls.values())),
        "control_dll_hashes_present_and_identical_across_trials": equal_dll_hashes(list(control_dlls.values())),
        "frozen_four_sessions_four_turns": len(frozen_sessions) == 4 and
            all(len(s.get("turns", [])) == 4 and s.get("status") == "complete" for s in frozen_sessions),
        "llama_trial_four_sessions_four_turns": len(llama_sessions) == 4 and
            all(len(s.get("turns", [])) == 4 and s.get("status") == "complete" for s in llama_sessions),
        "expected_frozen_kinds": kinds_frozen == {"off", "cpu"} and
            all(sum(s.get("kind") == k for s in frozen_sessions) == 2 for k in ("off", "cpu")),
        "expected_llama_kinds": kinds_llama == {"opt", "llama"} and
            all(sum(s.get("kind") == k for s in llama_sessions) == 2 for k in ("opt", "llama")),
        "cpu_persistent_jobs_fallback_and_threads_pass_each_turn": len(cpu_checks) == 8 and
            all(x["pass"] for x in cpu_checks),
        "opt_persistent_jobs_fallback_and_threads_pass_each_turn": len(opt_checks) == 8 and
            all(x["pass"] for x in opt_checks),
        "same_source_context_across_all_trials": bool(context_ids[0]) and len(set(context_ids)) == 1 and
            bool(decoded_context_ids[0]) and len(set(decoded_context_ids)) == 1,
        "same_nonempty_llama_executable_hash_in_mirrored_and_llama_trials": bool(
            ((llama["result"].get("provenance") or {}).get("llama_server") or {}).get("sha256")) and
            ((llama["result"].get("provenance") or {}).get("llama_server") or {}).get("sha256") ==
            (((mirrored["result"].get("provenance") or {}).get("llama_server") or {}).get("sha256")),
        "same_exact_api_requests_across_frozen_mirrored_llama": len(req_match) == 4 and
            all(x["matched_across_files_and_sessions"] for x in req_match),
    }
    identity_gates = {
        "frozen_prompt_ids_identical_all_four_sessions": len(frozen_ids) == 4 and
            all(x["samples"] == 4 and x["prompt_hashes_present"] == 4 and x["prompt_identical"]
                for x in frozen_ids),
        "frozen_generated_ids_identical_all_four_sessions": len(frozen_ids) == 4 and
            all(x["samples"] == 4 and x["generated_hashes_present"] == 4 and x["generated_identical"]
                for x in frozen_ids),
        "opt_prompt_ids_repeat_between_two_opt_sessions": len(llama_opt_ids) == 4 and
            all(x["samples"] == 2 and x["prompt_hashes_present"] == 2 and x["prompt_identical"]
                for x in llama_opt_ids),
        "opt_generated_ids_repeat_between_two_opt_sessions": len(llama_opt_ids) == 4 and
            all(x["samples"] == 2 and x["generated_hashes_present"] == 2 and x["generated_identical"]
                for x in llama_opt_ids),
    }

    prov = llama["result"].get("provenance") or {}
    llama_exe = prov.get("llama_server") or {}
    mirrored_exe = ((mirrored["result"].get("provenance") or {}).get("llama_server") or {})
    cfg = llama["result"].get("config") or {}
    dll = ((prov.get("neural_server_roots") or {}).get("opt") or {}).get("artifacts", {}).get(
        "selected_decode_dll") or {}
    return {
        "success": all(complete_gates.values()) and all(identity_gates.values()),
        "completeness_and_matched_inputs": complete_gates,
        "quality_identity": identity_gates,
        "frozen_identity_by_turn": frozen_ids,
        "opt_identity_by_turn": llama_opt_ids,
        "llama_internal_prompt_token_ids": {
            "status": "unknown_not_reported_by_backend",
            "per_turn_observations": llama_prompt_ids,
            "api_request_hash_equality_does_not_establish_internal_prompt_token_identity": True},
        "cpu_persistent_turn_checks": cpu_checks,
        "opt_persistent_turn_checks": opt_checks,
        "weighted_decode_and_stage_medians": {"frozen": frozen_metrics, "opt_vs_llama": llama_metrics},
        "weighted_comparisons": comparisons,
        "exact_api_request_hashes_by_turn": req_match,
        "source_context_identity": contexts,
        "effective_opt_decode_dll": dll,
        "candidate_decode_dlls": candidate_dlls,
        "control_decode_dlls": control_dlls,
        "llama_requested_args": cfg.get("llama_args"),
        "configured_llama_directory_label": cfg.get("llama_dir"),
        "llama_server_executable": {"path": llama_exe.get("path"), "size": llama_exe.get("size"),
                                     "sha256": llama_exe.get("sha256"),
                                     "sha256_in_mirrored_trial": mirrored_exe.get("sha256"),
                                     "same_nonempty_sha256_as_mirrored_trial": (
                                         bool(llama_exe.get("sha256")) and
                                         llama_exe.get("sha256") == mirrored_exe.get("sha256"))},
        "llama_version_claim": "none; the directory label is not a version claim, and the hash is only an executable identity",
        "quality_caveat": "Exact request-body hashes establish matched API inputs. Backend-internal prompt token IDs are not reported for llama; generated-ID equality is checked only between the two opt sessions. Cross-backend quality equality is not established.",
        "review_bench_integrity": {"frozen": frozen["integrity"], "mirrored": mirrored["integrity"],
                                   "llama": llama["integrity"]},
    }


def main():
    try:
        report = summarize()
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
