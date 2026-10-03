"""Summarize/audit the matched persistent-CPU decode trial (read-only, no inference).

Uses the existing ``review_bench`` helpers for archive loading, integrity, and
weighted decode aggregation. Prints one JSON document to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_bench  # noqa: E402

DEFAULT_RESULT = (HERE.parent / "benchmarks" / "codex_decode_followup_20261001" /
                  "persistent_cpu_mirrored.json")
EXPECTED_KINDS = ("off", "cpu", "opt")


def numeric(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))


def turn_value(turn, key):
    value = turn.get(key)
    if value is None:
        value = (turn.get("neural") or {}).get(key)
    return value


def turn_hash(turn, key):
    value = turn_value(turn, key)
    return value if isinstance(value, str) and value else None


def mean_median(values):
    vals = [float(x) for x in values if numeric(x)]
    return {"count": len(vals), "mean": round(statistics.fmean(vals), 6) if vals else None,
            "median": round(statistics.median(vals), 6) if vals else None}


def args_for_kind(config, session):
    args = [str(config.get("neural_args", ""))]
    for item in config.get("kinds", []):
        name, sep, extra = str(item).partition("=")
        if name == session.get("kind") and sep:
            args.append(extra)
    args.append(str(session.get("args", "")))
    return " ".join(args).split()


def last_option(tokens, flag, default=None):
    value = default
    for i, token in enumerate(tokens):
        if token.startswith(flag + "="):
            value = token.split("=", 1)[1]
        elif token == flag and i + 1 < len(tokens):
            value = tokens[i + 1]
    return value


def selected_dll(provenance, kind):
    root = (provenance.get("neural_server_roots") or {}).get(kind) or {}
    artifacts = root.get("artifacts") or {}
    return artifacts.get("selected_decode_dll")


def stage_stats(turns):
    fields = {"decode_tok_s": "decode_tok_s", "cpu_ms_per_tok": "cpu_ms_per_tok",
              "replay_ms_per_tok": "replay_ms_per_tok",
              "gpu_wait_ms_per_tok": "gpu_wait_ms_per_tok",
              "admission_ms_per_tok": "admission_ms_per_tok",
              "other_ms_per_tok": "other_ms_per_tok"}
    return {out: mean_median([turn_value(t, src) for t in turns]) for out, src in fields.items()}


def weighted_for(summary, key):
    return summary.get(key) or {"tokens": 0, "seconds": None, "tok_s": None}


def weighted_ratio(new, base):
    """Compare weighted token/time totals without using aggregate's rounded tok/s."""
    nt, ns = new.get("tokens"), new.get("seconds")
    bt, bs = base.get("tokens"), base.get("seconds")
    if not all(numeric(x) for x in (nt, ns, bt, bs)) or ns <= 0 or bs <= 0 or bt <= 0:
        return {"ratio": None, "percent_change": None}
    value = (nt * bs) / (ns * bt)
    return {"ratio": round(value, 8),
            "percent_change": round((value - 1) * 100, 6)}


def weighted_metric(aggregate_summary, kind, key="all_turns_weighted_decode"):
    return weighted_for(aggregate_summary.get(kind) or {}, key)


def matched_output_subset(sessions, identity):
    """Aggregate only turns whose prompt and generated IDs match in all sessions."""
    identity_maps = {field: {row["turn"]: row for row in rows}
                     for field, rows in identity.items()}
    shared = sorted(set(identity_maps.get("prompt_ids_sha256", {})) &
                    set(identity_maps.get("generated_ids_sha256", {})))
    matched = [i for i in shared if all(
        identity_maps[field][i].get("identical_across_all_sessions") is True and
        identity_maps[field][i].get("samples") == len(sessions) and
        identity_maps[field][i].get("hashes_present") == len(sessions)
        for field in ("prompt_ids_sha256", "generated_ids_sha256"))]

    subset_sessions = []
    observations = []
    for sess in sessions:
        chosen = []
        for index in matched:
            if len(sess.get("turns", [])) <= index:
                continue
            turn = sess["turns"][index]
            chosen.append(turn)
            neural = turn.get("neural") or {}
            observations.append({
                "kind": sess.get("kind"), "tag": sess.get("tag"), "session_status": sess.get("status"),
                "turn": index, "prompt_ids_sha256": turn_hash(turn, "prompt_ids_sha256"),
                "generated_ids_sha256": turn_hash(turn, "generated_ids_sha256"),
                "decode_tokens_timed": turn.get("decode_tokens_timed"),
                "completion_tokens": turn.get("completion_tokens"),
                "decode_s": turn_value(turn, "decode_s"),
                "decode_tok_s": turn_value(turn, "decode_tok_s"),
                "cpu_ms_per_tok": turn_value(turn, "cpu_ms_per_tok"),
                "replay_ms_per_tok": turn_value(turn, "replay_ms_per_tok"),
                "gpu_wait_ms_per_tok": turn_value(turn, "gpu_wait_ms_per_tok"),
                "admission_ms_per_tok": turn_value(turn, "admission_ms_per_tok"),
                "persistent_cpu_jobs": neural.get("persistent_cpu_jobs", turn.get("persistent_cpu_jobs")),
                "persistent_cpu_fallback_calls": neural.get(
                    "persistent_cpu_fallback_calls", turn.get("persistent_cpu_fallback_calls")),
                "persistent_cpu_team_threads": neural.get(
                    "persistent_cpu_team_threads", turn.get("persistent_cpu_team_threads")),
            })
        subset_sessions.append({"kind": sess.get("kind"), "status": sess.get("status"),
                                "turns": chosen})

    subset_summary, _ = review_bench.aggregate({"sessions": subset_sessions}, None)
    per_turn = {}
    for index in matched:
        one_turn = [{"kind": sess.get("kind"), "status": sess.get("status"),
                     "turns": [sess["turns"][index]] if len(sess.get("turns", [])) > index else []}
                    for sess in sessions]
        one_summary, _ = review_bench.aggregate({"sessions": one_turn}, None)
        per_turn[str(index)] = {kind: weighted_metric(one_summary, kind)
                                for kind in EXPECTED_KINDS}

    comparisons = {}
    for challenger in ("cpu", "opt"):
        comparisons[f"{challenger}_vs_off"] = weighted_ratio(
            weighted_metric(subset_summary, challenger),
            weighted_metric(subset_summary, "off"))
    comparisons["opt_vs_cpu"] = weighted_ratio(
        weighted_metric(subset_summary, "opt"),
        weighted_metric(subset_summary, "cpu"))
    per_turn_comparisons = {}
    for index, values in per_turn.items():
        per_turn_comparisons[index] = {
            "cpu_vs_off": weighted_ratio(values["cpu"], values["off"]),
            "opt_vs_off": weighted_ratio(values["opt"], values["off"]),
            "opt_vs_cpu": weighted_ratio(values["opt"], values["cpu"]),
        }
    return {"turn_indices": matched,
            "criterion": ("posthoc subset: prompt and generated token ID hashes are present and identical "
                          "across every session for each included turn"),
            "sessions_included": len(sessions),
            "summary_by_kind": {kind: subset_summary.get(kind, {}) for kind in EXPECTED_KINDS},
            "comparisons": comparisons, "per_turn_weighted_decode": per_turn,
            "per_turn_comparisons": per_turn_comparisons,
            "raw_observations": observations}


def analyze(result_path: str):
    result = review_bench.read(result_path)
    archive = review_bench.archive_for(result_path, result)
    summary, integrity = review_bench.aggregate(result, archive)
    sessions = result.get("sessions", [])
    by_kind = {kind: [s for s in sessions if s.get("kind") == kind] for kind in EXPECTED_KINDS}
    flat_turns = {kind: [t for s in rows for t in s.get("turns", [])]
                  for kind, rows in by_kind.items()}

    # Exact token identity across all scheduled sessions, grouped by turn index.
    identity = {}
    for field in ("prompt_ids_sha256", "generated_ids_sha256"):
        turn_rows = []
        for turn_index in range(max((len(s.get("turns", [])) for s in sessions), default=0)):
            hashes = [turn_hash(s["turns"][turn_index], field)
                      for s in sessions if len(s.get("turns", [])) > turn_index]
            known = [h for h in hashes if h]
            turn_rows.append({"turn": turn_index, "samples": len(hashes), "hashes_present": len(known),
                              "missing_hash_count": len(hashes) - len(known),
                              "distinct_hashes": len(set(known)),
                              "identical_across_all_sessions": (
                                  len(hashes) == 6 and len(known) == 6 and len(set(known)) == 1)})
        identity[field] = turn_rows
    matched_subset = matched_output_subset(sessions, identity)

    provenance = result.get("provenance") or {}
    archive_requests = [req for sess in (archive or {}).get("sessions", [])
                        for req in sess.get("requests", [])]
    messages_hashes_valid = bool(archive) and all(
        isinstance(req.get("messages_sha256"), str) and req.get("messages_sha256") and
        review_bench.sha256_json((req.get("request") or {}).get("messages", [])) == req.get("messages_sha256")
        for req in archive_requests)
    capture_state = integrity.get("archive_capture_state") or {}
    archive_capture_complete = (len(archive_requests) == 24 and capture_state.get("complete") == 24 and
                                capture_state.get("failed") == 0 and capture_state.get("pending") == 0 and
                                capture_state.get("legacy_unknown") == 0)
    effective_dlls = {}
    persistence = {}
    per_kind = {}
    for kind in EXPECTED_KINDS:
        rows = by_kind[kind]
        aggregate = summary.get(kind) or {}
        kind_tokens = args_for_kind(result.get("config") or {}, rows[0] if rows else {"kind": kind})
        dll = selected_dll(provenance, kind) or {}
        dll = {**dll, "requested_filename": last_option(kind_tokens, "--kdll", "gptoss_cpu_cap2.dll"),
               "kernel_persistent_enabled": last_option(kind_tokens, "--kernel-persistent", "0") == "1"}
        effective_dlls[kind] = dll
        kind_report = {"sessions": len(rows), "session_statuses": [s.get("status") for s in rows],
                       "all_four_turns": all(len(s.get("turns", [])) == 4 for s in rows),
                       "all_turns_weighted_decode": weighted_for(aggregate, "all_turns_weighted_decode"),
                       "first_weighted_decode": weighted_for(aggregate, "first_weighted_decode"),
                       "followup_weighted_decode": weighted_for(aggregate, "followup_weighted_decode"),
                       "per_turn": [], "effective_decode_dll": dll}
        for turn_index in range(4):
            turn_rows = [s["turns"][turn_index] for s in rows if len(s.get("turns", [])) > turn_index]
            turn_aggregate, _ = review_bench.aggregate(
                {"sessions": [{"kind": kind, "turns": turn_rows}]}, None)
            kind_report["per_turn"].append({"turn": turn_index,
                                             "phase": "first" if turn_index == 0 else "followup",
                                             **stage_stats(turn_rows),
                                             "weighted_decode": weighted_metric(turn_aggregate, kind)})
        per_kind[kind] = kind_report

        enabled = last_option(kind_tokens, "--kernel-persistent", "0") == "1"
        threads_text = last_option(kind_tokens, "--threads", "8")
        try:
            effective_threads = int(threads_text)
        except (TypeError, ValueError):
            effective_threads = None
        turn_values = []
        for sess in rows:
            per_turn = []
            for turn_index, turn in enumerate(sess.get("turns", [])):
                neural = turn.get("neural") or {}
                jobs = neural.get("persistent_cpu_jobs", turn.get("persistent_cpu_jobs"))
                fallback = neural.get("persistent_cpu_fallback_calls",
                                      turn.get("persistent_cpu_fallback_calls"))
                team_threads = neural.get("persistent_cpu_team_threads",
                                          turn.get("persistent_cpu_team_threads"))
                per_turn.append({"turn": turn_index, "jobs": jobs,
                                 "fallback_calls": fallback, "team_threads": team_threads,
                                 "jobs_positive": numeric(jobs) and jobs > 0,
                                 "fallback_zero": numeric(fallback) and fallback == 0,
                                 "team_threads_match_effective_args": (
                                     effective_threads is not None and numeric(team_threads) and
                                     team_threads == effective_threads)})
            job_vals = [x["jobs"] for x in per_turn if numeric(x["jobs"])]
            fallback_vals = [x["fallback_calls"] for x in per_turn if numeric(x["fallback_calls"])]
            thread_vals = [x["team_threads"] for x in per_turn if numeric(x["team_threads"])]
            turn_values.append({"tag": sess.get("tag"), "status": sess.get("status"),
                                "persistent_enabled": enabled, "turns": per_turn,
                                "persistent_jobs_total": sum(job_vals) if job_vals else None,
                                "fallback_calls_total": sum(fallback_vals) if fallback_vals else None,
                                "team_threads_observed": sorted(set(thread_vals)),
                                "metrics_present_each_turn": len(per_turn) == 4 and
                                    all(numeric(x["jobs"]) and numeric(x["fallback_calls"]) and
                                        numeric(x["team_threads"]) for x in per_turn),
                                "effective_threads_from_args": effective_threads,
                                "enabled_turn_checks_pass": (len(per_turn) == 4 and all(
                                    x["jobs_positive"] and x["fallback_zero"] and
                                    x["team_threads_match_effective_args"] for x in per_turn)
                                    if enabled else None)})
        active = [x for x in turn_values if x["persistent_enabled"]]
        persistence_ok = all(x["metrics_present_each_turn"] and
                             x["enabled_turn_checks_pass"] is True for x in active)
        disabled_ok = all((x["persistent_jobs_total"] in (None, 0) and
                           x["fallback_calls_total"] in (None, 0)) for x in turn_values if not x["persistent_enabled"])
        persistence[kind] = {"enabled_from_effective_args": enabled,
                             "effective_threads_from_args": effective_threads,
                             "sessions": turn_values,
                             "persistent_metrics_pass": (persistence_ok if enabled else disabled_ok),
                             "fallback_calls_zero": (all(x["fallback_calls_total"] == 0 for x in active)
                                                     if enabled else disabled_ok)}

    comparisons = {}
    for challenger in ("cpu", "opt"):
        comparisons[f"{challenger}_vs_off"] = {}
        for phase, summary_key in (("all", "all_turns_weighted_decode"),
                                   ("first", "first_weighted_decode"),
                                   ("followup", "followup_weighted_decode")):
            a = weighted_for(summary.get(challenger, {}), summary_key)
            b = weighted_for(summary.get("off", {}), summary_key)
            comparisons[f"{challenger}_vs_off"][phase] = weighted_ratio(a, b)
        comparisons[f"{challenger}_vs_off"]["per_turn"] = [
            {"turn": i, "phase": "first" if i == 0 else "followup",
             **weighted_ratio(per_kind[challenger]["per_turn"][i]["weighted_decode"],
                              per_kind["off"]["per_turn"][i]["weighted_decode"])}
            for i in range(4)]
    comparisons["opt_vs_cpu"] = {}
    for phase, summary_key in (("all", "all_turns_weighted_decode"),
                               ("first", "first_weighted_decode"),
                               ("followup", "followup_weighted_decode")):
        comparisons["opt_vs_cpu"][phase] = weighted_ratio(
            weighted_for(summary.get("opt", {}), summary_key),
            weighted_for(summary.get("cpu", {}), summary_key))
    comparisons["opt_vs_cpu"]["per_turn"] = [
        {"turn": i, "phase": "first" if i == 0 else "followup",
         **weighted_ratio(per_kind["opt"]["per_turn"][i]["weighted_decode"],
                          per_kind["cpu"]["per_turn"][i]["weighted_decode"])}
        for i in range(4)]

    gates = {
        "six_sessions": len(sessions) == 6,
        "two_sessions_per_kind": all(len(by_kind[k]) == 2 for k in EXPECTED_KINDS),
        "expected_kinds": set(s.get("kind") for s in sessions) == set(EXPECTED_KINDS),
        "all_sessions_complete": len(sessions) == 6 and all(s.get("status") == "complete" for s in sessions),
        "four_turns_each": len(sessions) == 6 and all(len(s.get("turns", [])) == 4 for s in sessions),
        "no_missing_prompt_or_generated_hashes": len(sessions) == 6 and all(
            all(row["samples"] == 6 and row["hashes_present"] == 6 and row["missing_hash_count"] == 0
                for row in identity[field]) and len(identity[field]) == 4
            for field in identity),
        "prompt_ids_identical_across_all_sessions": len(identity["prompt_ids_sha256"]) == 4 and all(
            row["identical_across_all_sessions"] for row in identity["prompt_ids_sha256"]),
        "generated_ids_identical_across_all_sessions": len(identity["generated_ids_sha256"]) == 4 and all(
            row["identical_across_all_sessions"] for row in identity["generated_ids_sha256"]),
        "request_response_archive_and_result_links_valid": all(
            integrity.get(key) is True for key in ("request_archive_integrity", "response_archive_integrity",
                                                  "result_archive_link_integrity")),
        "all_24_archived_requests_complete": archive_capture_complete,
        "request_message_hashes_present_and_valid": messages_hashes_valid,
        "requests_matched_by_turn": len(integrity.get("request_match_across_sessions_by_turn", [])) == 4 and all(
            row.get("matched") is True for row in integrity.get("request_match_across_sessions_by_turn", [])),
        "effective_dll_hashes_present": all(isinstance(effective_dlls[k], dict) and
                                             effective_dlls[k].get("path") and
                                             effective_dlls[k].get("size") is not None and
                                             effective_dlls[k].get("sha256") for k in EXPECTED_KINDS),
        "persistent_runtime_checks_pass": all(persistence[k]["persistent_metrics_pass"]
                                               for k in EXPECTED_KINDS),
    }
    gates["success"] = all(gates.values())
    return {"result_path": str(Path(result_path).resolve()),
            "archive_path": str(Path(result_path + ".replay.json").resolve()) if archive else None,
            "success": gates["success"], "gates": gates,
            "summary": per_kind, "comparisons": comparisons,
            "identity_by_turn": identity, "persistent_runtime": persistence,
            "matched_output_subset": matched_subset,
            "effective_decode_dll_provenance": effective_dlls,
            "archive_message_hashes_valid": messages_hashes_valid,
            "archive_capture_state": capture_state,
            "review_bench_integrity": integrity}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", nargs="?", default=str(DEFAULT_RESULT),
                    help="result JSON; defaults to the persistent mirrored trial")
    args = ap.parse_args()
    try:
        report = analyze(args.result)
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
