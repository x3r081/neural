"""Recompute warm-prefill medians and verify the archived evidence, without inference.

Run with the project Python. Prints JSON; redirect to a NEW file when repeating.
"""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def audit(name, expected_sessions):
    path = ROOT / (name + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    sessions = data["sessions"]
    assert not data["errors"], data["errors"]
    assert len(sessions) == expected_sessions
    expected_prompts = [r["prompt"] for r in sessions[0]["requests"]]
    assert expected_prompts
    for s in sessions:
        assert s.get("free_ram_gib_after_stop") is not None, "unfinished session"
        assert [r["prompt"] for r in s["requests"]] == expected_prompts
        for r in s["requests"]:
            assert digest(r["request"]) == r["request_sha256"]
            assert digest(r["response"]) == r["response_sha256"]
            assert r["stats"]["cached_tokens"] == 0
    rows = []
    for prompt in expected_prompts:
        entries = [(s["kind"], next(r for r in s["requests"] if r["prompt"] == prompt))
                   for s in sessions]
        for key in ("request_sha256", "prompt_ids_sha256", "generated_ids_sha256"):
            values = [r[key] for _, r in entries]
            assert all(values) and len(set(values)) == 1, (name, prompt, key)
        variants = {}
        for kind in dict.fromkeys(s["kind"] for s in sessions):
            sample = [r for k, r in entries if k == kind]
            times = [r["stats"]["prompt_s"] for r in sample]
            wall = [r["stats"]["wall_s"] for r in sample]
            variants[kind] = {
                "samples_s": times, "median_s": statistics.median(times),
                "wall_samples_s": wall, "median_wall_s": statistics.median(wall),
                "group_counts": [r["stats"]["neural"]["prefill_epi_groups"] for r in sample],
                "reserved_gib": [r["stats"]["neural"]["vram_reserved_gib"] for r in sample],
            }
        row = {"prompt": prompt, "measured": entries[0][1]["measured"],
               "rendered_tokens": entries[0][1]["stats"]["prompt_tokens"], "variants": variants}
        row["group3_latency_reduction_pct"] = 100 * (1 - variants["g3"]["median_s"] / variants["off"]["median_s"])
        rows.append(row)
    return {"source": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sessions": len(sessions), "all_request_response_hashes_verified": True,
            "all_corresponding_request_prompt_output_identities_match": True,
            "all_cached_tokens_zero": True, "config": data["config"], "rows": rows}


if __name__ == "__main__":
    result = {"evidence": "MEASURED", "studies": {
        name: audit(name, count) for name, count in (
            ("warm_prefill_bounded_abba", 6),
            ("warm_prefill_short_abba", 4),
            ("warm_profile_abba", 4),
        )}}
    print(json.dumps(result, indent=2))
