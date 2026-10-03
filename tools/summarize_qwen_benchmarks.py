"""Summarize frozen Qwen HTTP benchmark records without scoring answer quality.

The common decode metric counts generated tokens after the first sample:
llama.cpp defines this as predicted_n - 1, while GenerationEngine records the
same count directly as decode_calls. The first token comes from prompt logits.
Client throughput instead uses all output tokens divided by client_wall_s.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _named_paths(items: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option} expects LABEL=PATH, got {item!r}")
        label, value = item.split("=", 1)
        if not label or not value or label in result:
            raise ValueError(f"invalid or duplicate {option} label: {label!r}")
        result[label] = Path(value)
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _int_arg(command: list[str], names: tuple[str, ...]) -> int | None:
    for index, value in enumerate(command):
        if value in names and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                return None
        for name in names:
            if value.startswith(name + "="):
                try:
                    return int(value.split("=", 1)[1])
                except ValueError:
                    return None
    return None


def _execution_settings(meta: dict[str, Any]) -> dict[str, Any]:
    command = meta.get("command") or []
    if not isinstance(command, list):
        command = []
    context = _int_arg(command, ("-c", "--ctx-size", "--context"))
    if context is None:
        context = meta.get("context")
    weight_precision = meta.get("weight_precision", meta.get("model_precision"))
    cache_precision = meta.get("kv_cache_precision", meta.get("cache_dtype"))
    # llama.cpp spells out both cache dtypes on its command line. Do not infer
    # the neural runtime's cache dtype from its backend name or placement.
    if cache_precision is None:
        k = _arg_value(command, ("-ctk", "--cache-type-k"))
        v = _arg_value(command, ("-ctv", "--cache-type-v"))
        if k and v and k.lower() == v.lower():
            cache_precision = k.lower()
        elif k or v:
            cache_precision = {"k": k, "v": v}
    if weight_precision is None:
        weight_precision = _arg_value(command, ("--weight-precision", "--dtype"))
    return {
        "source_revision": meta.get("model_revision", meta.get("source_revision")),
        "context": context,
        "weight_precision": weight_precision.lower() if isinstance(weight_precision, str) else weight_precision,
        "kv_cache_precision": cache_precision.lower() if isinstance(cache_precision, str) else cache_precision,
    }


def _arg_value(command: list[str], names: tuple[str, ...]) -> str | None:
    for index, value in enumerate(command):
        if value in names and index + 1 < len(command):
            return str(command[index + 1])
        for name in names:
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
    return None


def _finite_number(value: Any, field: str, run_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{run_id}: {field} must be a finite number, got {value!r}")
    return float(value)


def _summarize_engine(
    label: str,
    result_path: Path,
    meta_path: Path,
    expected_hash: str,
    expected_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _read_json(result_path)
    meta = _read_json(meta_path)
    if result.get("status") != "completed":
        raise ValueError(f"{label}: benchmark status is {result.get('status')!r}, expected 'completed'")
    if result.get("requests_file_sha256") != expected_hash:
        raise ValueError(f"{label}: results use a different requests-file SHA-256")
    if "cold conversation state per request" not in str(result.get("scope", "")):
        raise ValueError(f"{label}: run scope does not confirm independent conversation state per request")
    runs = result.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{label}: no run records")

    expected_by_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in expected_requests:
        expected_by_id[str(entry["id"])].append(entry)
    seen_by_id: collections.Counter[str] = collections.Counter()
    counts: collections.Counter[str] = collections.Counter()
    stop_counts: collections.Counter[str] = collections.Counter()
    rows = []
    sum_tokens = sum_steps = 0
    sum_decode_ms = sum_wall_s = sum_prompt_n = sum_prompt_ms = 0.0
    sum_prompt_ms_count = 0
    max_tokens = 0

    for run in runs:
        run_id = str(run.get("id"))
        occurrences = expected_by_id.get(run_id, [])
        occurrence = seen_by_id[run_id]
        if occurrence >= len(occurrences):
            raise ValueError(f"{label}/{run_id}: unexpected duplicate or unknown request ID")
        expected = occurrences[occurrence]
        seen_by_id[run_id] += 1

        request = run.get("request") or {}
        response = run.get("response") or {}
        timings = response.get("timings") or response.get("neural") or {}
        if run.get("prompt_token_ids_match") is not True:
            raise ValueError(f"{label}/{run_id}: prompt token ID check did not pass")
        if request.get("prompt") != expected.get("prompt"):
            raise ValueError(f"{label}/{run_id}: raw prompt differs from frozen request")
        if request.get("cache_prompt") is not False:
            raise ValueError(f"{label}/{run_id}: cache_prompt must be false")
        cap = expected.get("max_tokens")
        if request.get("n_predict", request.get("max_tokens")) != cap:
            raise ValueError(f"{label}/{run_id}: token cap differs from frozen request")
        for key in ("temperature", "top_k", "top_p", "seed"):
            if request.get(key) != {"temperature": 0, "top_k": 20, "top_p": 0.95, "seed": 0}[key]:
                raise ValueError(f"{label}/{run_id}: request generation setting {key} is unexpected")

        prompt_n = int(timings.get("prompt_n", -1))
        if prompt_n != len(expected.get("prompt_ids", [])):
            raise ValueError(f"{label}/{run_id}: prompt_n does not match the frozen prompt token IDs")
        cache_n = timings.get("cache_n")
        if cache_n is not None and int(cache_n) != 0:
            raise ValueError(f"{label}/{run_id}: prompt cache was reused ({cache_n} tokens)")
        if cache_n is None:
            counts["cache_count_not_reported"] += 1
        else:
            counts["cache_zero"] += 1

        predicted_n = int(timings.get("predicted_n", response.get("tokens_predicted", -1)))
        if predicted_n < 0:
            raise ValueError(f"{label}/{run_id}: no generated-token count")
        reported_n = response.get("tokens_predicted")
        if reported_n is None:
            reported_n = (response.get("usage") or {}).get("completion_tokens")
        if reported_n is not None and int(reported_n) != predicted_n:
            raise ValueError(f"{label}/{run_id}: response token count disagrees with timings.predicted_n")
        token_ids = response.get("tokens")
        if isinstance(token_ids, list) and token_ids and len(token_ids) != predicted_n:
            raise ValueError(f"{label}/{run_id}: returned token-ID count disagrees with predicted_n")
        decode_steps = int(timings.get("decode_calls", max(0, predicted_n - 1)))
        expected_steps = max(0, predicted_n - 1)
        if decode_steps != expected_steps:
            raise ValueError(f"{label}/{run_id}: decode_calls={decode_steps} differs from predicted_n-1={expected_steps}")
        decode_ms = _finite_number(timings.get("predicted_ms"), "predicted_ms", run_id)
        wall_s = _finite_number(run.get("client_wall_s"), "client_wall_s", run_id)
        if decode_ms <= 0 or wall_s <= 0:
            raise ValueError(f"{label}/{run_id}: decode and client elapsed times must be positive")
        stop = str(response.get("stop_type", response.get("finish_reason", "unknown")))
        stop_counts[stop] += 1
        at_cap = predicted_n == cap
        counts["at_cap"] += int(at_cap)
        counts["below_cap"] += int(predicted_n < cap)
        counts["truncated"] += int(bool(response.get("truncated", False)))
        counts["prompt_ids_validated"] += 1
        sum_tokens += predicted_n
        sum_steps += decode_steps
        sum_decode_ms += decode_ms
        sum_wall_s += wall_s
        sum_prompt_n += prompt_n
        if timings.get("prompt_ms") is not None:
            prompt_ms = _finite_number(timings["prompt_ms"], "prompt_ms", run_id)
            sum_prompt_ms += prompt_ms
            sum_prompt_ms_count += 1
        max_tokens = max(max_tokens, predicted_n)
        rows.append({
            "id": run_id,
            "prompt_tokens": prompt_n,
            "cache_tokens": cache_n,
            "output_tokens": predicted_n,
            "decode_steps": decode_steps,
            "decode_elapsed_ms": decode_ms,
            "decode_steps_per_second": decode_steps * 1000.0 / decode_ms,
            "client_wall_s": wall_s,
            "client_output_tokens_per_second": predicted_n / wall_s,
            "output_cap": cap,
            "at_cap": at_cap,
            "stop": stop,
            "truncated": bool(response.get("truncated", False)),
        })

    if seen_by_id != collections.Counter({key: len(value) for key, value in expected_by_id.items()}):
        raise ValueError(f"{label}: run IDs/counts do not cover the frozen request cohort exactly once")
    if max_tokens == 0:
        raise ValueError(f"{label}: no output tokens were generated")

    summary = {
        "status": result["status"],
        "result_path": str(result_path.resolve()),
        "run_metadata_path": str(meta_path.resolve()),
        "execution_settings": _execution_settings(meta),
        "run_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "stop_counts": dict(sorted(stop_counts.items())),
        "prompt_tokens_total": int(sum_prompt_n),
        "prompt_prefill_ms_total": sum_prompt_ms if sum_prompt_ms_count == len(rows) else None,
        "prompt_prefill_tokens_per_second_weighted": (
            sum_prompt_n * 1000.0 / sum_prompt_ms
            if sum_prompt_ms_count == len(rows) and sum_prompt_ms > 0 else None
        ),
        "output_tokens_total": sum_tokens,
        "decode_steps_total": sum_steps,
        "decode_elapsed_ms_total": sum_decode_ms,
        "decode_steps_per_second_weighted": sum_steps * 1000.0 / sum_decode_ms,
        "client_wall_s_total": sum_wall_s,
        "client_output_tokens_per_second_weighted": sum_tokens / sum_wall_s,
        "per_request": rows,
    }
    return summary


def _comparison_gate(summaries: dict[str, dict[str, Any]], cohort_hash: str) -> dict[str, Any]:
    labels = list(summaries)
    if len(labels) < 2:
        return {"eligible": False, "reasons": ["at least two result sets are required"]}
    baseline = summaries[labels[0]]["execution_settings"]
    reasons = []
    for label in labels[1:]:
        current = summaries[label]["execution_settings"]
        for key in ("source_revision", "context", "weight_precision", "kv_cache_precision"):
            left, right = baseline.get(key), current.get(key)
            if left is None or right is None:
                reasons.append(f"{key} is missing from run metadata for {labels[0]} or {label}")
            elif left != right:
                reasons.append(f"{key} differs between {labels[0]} and {label}: {left!r} vs {right!r}")
    if not cohort_hash:
        reasons.append("requests cohort SHA-256 is unavailable")
    return {"eligible": not reasons, "cohort_sha256": cohort_hash, "placement_may_differ": True,
            "reasons": reasons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path, help="frozen requests_v*.json")
    parser.add_argument("--run", required=True, action="append", metavar="LABEL=RESULTS_JSON")
    parser.add_argument("--meta", required=True, action="append", metavar="LABEL=RUN_JSON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_paths = _named_paths(args.run, "--run")
    meta_paths = _named_paths(args.meta, "--meta")
    if list(run_paths) != list(meta_paths):
        parser.error("--run and --meta must provide the same labels in the same order")
    request_bytes = args.requests.read_bytes()
    request_hash = hashlib.sha256(request_bytes).hexdigest()
    request_doc = json.loads(request_bytes)
    expected_requests = request_doc.get("requests")
    if not isinstance(expected_requests, list) or not expected_requests:
        parser.error("requests file must contain a non-empty requests list")

    summaries = {
        label: _summarize_engine(label, run_paths[label], meta_paths[label],
                                 request_hash, expected_requests)
        for label in run_paths
    }
    gate = _comparison_gate(summaries, request_hash)
    relative_rates = None
    if gate["eligible"]:
        reference_label = next(iter(summaries))
        reference = summaries[reference_label]
        relative_rates = {
            label: {
                "reference": reference_label,
                "decode_steps_per_second_ratio": (
                    summary["decode_steps_per_second_weighted"] /
                    reference["decode_steps_per_second_weighted"]
                ),
                "client_output_tokens_per_second_ratio": (
                    summary["client_output_tokens_per_second_weighted"] /
                    reference["client_output_tokens_per_second_weighted"]
                ),
            }
            for label, summary in summaries.items() if label != reference_label
        }
    report = {
        "status": "SUMMARIZED_NO_QUALITY_SCORING",
        "requests_file": str(args.requests.resolve()),
        "requests_file_sha256": request_hash,
        "requested_cohort_count": len(expected_requests),
        "comparison_gate": gate,
        "relative_rates": relative_rates,
        "engines": summaries,
        "interpretation": {
            "decode_rate": "weighted sum of output_tokens-1 decode calls divided by summed predicted_ms. The numerators match; internal timer boundaries do not: llama.cpp starts after sampling output token 1, while GenerationEngine's predicted_ms starts before sampling it and includes incremental tokenizer.decode work.",
            "client_rate": "weighted sum of all generated output tokens divided by summed HTTP client_wall_s; includes prompt, generation, serialization and transport",
            "prompt_rate": "weighted from server-reported prompt_n/prompt_ms; llama.cpp's prompt timer includes sampling output token 1, GenerationEngine's prompt_ms stops immediately before that sample",
            "quality": "not scored; generated text/token identity is not assumed across engines",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "comparison_eligible": report["comparison_gate"]["eligible"],
                      "reason_count": len(report["comparison_gate"]["reasons"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
