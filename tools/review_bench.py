"""Audit and aggregate bench_vs_llama.py result/replay artifacts (no model loading)."""
import argparse
import json
import os
import statistics
import sys
import hashlib


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def median(values):
    values = [x for x in values if isinstance(x, (int, float))]
    return round(statistics.median(values), 4) if values else None


def sha256_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def archive_for(path, result):
    rp = path + ".replay.json"
    if not os.path.isfile(rp):
        return None
    ar = read(rp)
    if ar.get("schema") != "neuralserver-bench-replay-v1":
        raise ValueError(f"unsupported replay archive schema: {rp}")
    return ar


def aggregate(result, archive):
    by_kind = {}
    for sidx, sess in enumerate(result.get("sessions", [])):
        kind = sess.get("kind", "unknown")
        bucket = by_kind.setdefault(kind, {"sessions": [], "turns": []})
        bucket["sessions"].append(sess)
        for turn in sess.get("turns", []):
            bucket["turns"].append(turn)
    replay_by_kind = {}
    if archive:
        for sess in archive.get("sessions", []):
            replay_by_kind.setdefault(sess.get("kind", "unknown"), []).append(sess)
    report = {}
    for kind, bucket in by_kind.items():
        sessions, turns = bucket["sessions"], bucket["turns"]
        first = [s["turns"][0] for s in sessions if s.get("turns")]
        follow = [t for s in sessions for t in s.get("turns", [])[1:]]
        def timing(t):
            nested = t.get("neural") or {}
            llama = t.get("timings") or {}
            is_neural = bool(nested) or "decode_tokens_timed" in t
            n = t.get("decode_tokens_timed")
            if n is None:
                n = max(0, (t.get("completion_tokens") or 0) - 1) if is_neural else (
                    llama.get("predicted_n") or t.get("completion_tokens"))
            dt = t.get("decode_s") or nested.get("decode_s")
            if dt is None and llama.get("predicted_ms") is not None:
                dt = llama["predicted_ms"] / 1000
            speed = t.get("decode_tok_s")
            if dt is None and speed and n:
                dt = n / speed
            return n, dt

        def weighted(rows):
            total_n, total_s = 0, 0.0
            for turn in rows:
                n, dt = timing(turn)
                if isinstance(n, (int, float)) and isinstance(dt, (int, float)) and dt > 0:
                    total_n += n
                    total_s += dt
            return {"tokens": total_n, "seconds": round(total_s, 4) if total_s else None,
                    "tok_s": round(total_n / total_s, 4) if total_s else None}

        all_weighted = weighted(turns)
        first_weighted = weighted(first)
        follow_weighted = weighted(follow)
        replay = replay_by_kind.get(kind, [])
        turn_identities = []
        for i in range(max((len(s.get("turns", [])) for s in sessions), default=0)):
            hashes = []
            for s in sessions:
                if len(s.get("turns", [])) <= i:
                    continue
                turn = s["turns"][i]
                nested = turn.get("neural") or {}
                h = (turn.get("generated_ids_sha256") or nested.get("generated_ids_sha256") or
                     turn.get("content_sha256") or turn.get("content_sha1"))
                if h:
                    hashes.append(h)
            n_expected = sum(len(s.get("turns", [])) > i for s in sessions)
            complete = len(hashes) == n_expected
            verified = n_expected >= 2 and complete
            turn_identities.append({"turn": i, "samples": n_expected, "hashes_present": len(hashes),
                                    "missing_identity_count": n_expected - len(hashes),
                                    "distinct_output_hashes": len(set(hashes)),
                                    "identical": (len(set(hashes)) == 1 if verified else None),
                                    "verification": ("verified" if verified else
                                                     ("not_verified_single_session" if n_expected < 2 else
                                                      "unknown_missing_hash"))})
        req_hashes_by_turn = []
        for i in range(max((len(s.get("requests", [])) for s in replay), default=0)):
            hs = {s["requests"][i].get("request_sha256") for s in replay if len(s.get("requests", [])) > i}
            req_hashes_by_turn.append({"turn": i, "distinct_hashes": len(hs),
                                       "matched": len(hs) == 1 and None not in hs})
        report[kind] = {
            "sessions": len(sessions), "turns": len(turns),
            "session_status_counts": {status: sum(1 for s in sessions if s.get("status", "legacy_unknown") == status)
                                       for status in ("complete", "failed", "running", "legacy_unknown")},
            "all_sessions_complete": (all(s.get("status") == "complete" for s in sessions)
                                      if all("status" in s for s in sessions) else None),
            "first_prompt_s_median": median([t.get("prompt_s") for t in first]),
            "first_decode_tok_s_median": median([t.get("decode_tok_s") for t in first]),
            "followup_decode_tok_s_median": median([t.get("decode_tok_s") for t in follow]),
            "all_turns_weighted_decode_tok_s": all_weighted["tok_s"],
            "all_turns_weighted_decode": all_weighted,
            "first_weighted_decode": first_weighted, "followup_weighted_decode": follow_weighted,
            "finish_reasons": sorted({(t.get("neural") or {}).get("finish") or
                                       ((t.get("response") or {}).get("choices") or [{}])[0].get("finish_reason")
                                       for t in turns if (t.get("neural") or {}).get("finish") or
                                       ((t.get("response") or {}).get("choices") or [{}])[0].get("finish_reason")}),
            "output_identity_by_turn": turn_identities,
            "request_hashes_by_turn": req_hashes_by_turn,
        }
    checks = {"request_archive_integrity": True, "response_archive_integrity": True,
              "result_archive_link_integrity": True,
              "archive_capture_state": {"complete": 0, "failed": 0, "pending": 0, "legacy_unknown": 0},
              "request_match_across_sessions_by_turn": [], "prompt_token_ids_match_by_turn": []}
    if archive:
        archive_sessions = archive.get("sessions", [])
        for sess in archive_sessions:
            for req in sess.get("requests", []):
                status = req.get("status", "complete" if "response" in req and "response_sha256" in req
                                 else "legacy_unknown")
                key = status if status in checks["archive_capture_state"] else "legacy_unknown"
                checks["archive_capture_state"][key] += 1
                if sha256_json(req.get("request")) != req.get("request_sha256"):
                    checks["request_archive_integrity"] = False
                if status == "complete":
                    if "response" not in req or sha256_json(req.get("response")) != req.get("response_sha256"):
                        checks["response_archive_integrity"] = False
                elif status in ("pending", "failed", "legacy_unknown"):
                    if checks["response_archive_integrity"] is not False:
                        checks["response_archive_integrity"] = None
        result_sessions = result.get("sessions", [])
        if len(result_sessions) != len(archive_sessions):
            checks["result_archive_link_integrity"] = False
        for rsess, asess in zip(result_sessions, archive_sessions):
            if rsess.get("kind") != asess.get("kind"):
                checks["result_archive_link_integrity"] = False
                continue
            complete_requests = [req for req in asess.get("requests", []) if req.get("status", "complete") == "complete"]
            if len(rsess.get("turns", [])) != len(complete_requests):
                checks["result_archive_link_integrity"] = False
            for turn, req in zip(rsess.get("turns", []), complete_requests):
                if turn.get("request_sha256") != req.get("request_sha256") or \
                   turn.get("response_sha256") != req.get("response_sha256"):
                    checks["result_archive_link_integrity"] = False
        max_turns = max((len(s.get("requests", [])) for s in archive_sessions), default=0)
        for i in range(max_turns):
            reqs, pids = set(), set()
            for sess in archive_sessions:
                if len(sess.get("requests", [])) > i and sess["requests"][i].get("status", "complete") == "complete":
                    reqs.add(sess["requests"][i].get("request_sha256"))
            expected_ids = 0
            known_ids = 0
            for sess in result.get("sessions", []):
                if len(sess.get("turns", [])) > i:
                    expected_ids += 1
                    turn = sess["turns"][i]
                    h = turn.get("prompt_ids_sha256") or (turn.get("neural") or {}).get("prompt_ids_sha256")
                    if h:
                        known_ids += 1
                        pids.add(h)
            expected_reqs = sum(len(s.get("requests", [])) > i and
                                s["requests"][i].get("status", "complete") == "complete"
                                for s in archive_sessions)
            checks["request_match_across_sessions_by_turn"].append({"turn": i, "distinct_hashes": len(reqs),
                "matched": (len(reqs) == 1 and None not in reqs if expected_reqs else None)})
            checks["prompt_token_ids_match_by_turn"].append({"turn": i, "samples": expected_ids,
                "hashes_present": known_ids, "missing_hash_count": expected_ids - known_ids,
                "distinct_hashes": len(pids),
                "matched": (len(pids) == 1 if expected_ids >= 2 and known_ids == expected_ids else None),
                "matched_among_known": (len(pids) == 1 if known_ids >= 2 else None)})
    else:
        checks["request_archive_integrity"] = None
        checks["response_archive_integrity"] = None
    return report, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="benchmark JSON result files")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    docs = []
    errors = []
    for path in args.results:
        try:
            result = read(path)
            archive = archive_for(path, result)
            summary, checks = aggregate(result, archive)
            docs.append({"path": os.path.abspath(path), "result": result, "archive": archive,
                         "summary": summary, "checks": checks})
        except Exception as e:
            errors.append(f"{path}: {type(e).__name__}: {e}")
    if errors:
        for error in errors:
            print("ERROR", error, file=sys.stderr)
        return 2
    compatibility = []
    for label, pick, known in (
        ("model_artifact", lambda d: (d["result"].get("provenance") or {}).get("gguf"),
         lambda v: isinstance(v, dict) and all(k in v for k in ("path", "size", "mtime_ns"))),
        ("replay_prompts", lambda d: [{"turn": i, "hashes": sorted({s["requests"][i].get("request_sha256")
            for s in (d["archive"] or {}).get("sessions", []) if len(s.get("requests", [])) > i},
            key=lambda x: "" if x is None else x)}
            for i in range(max((len(s.get("requests", [])) for s in (d["archive"] or {}).get("sessions", [])), default=0))],
         lambda v: bool(v) and all(h.get("hashes") for h in v)),
        ("context_source", lambda d: (d["result"].get("provenance") or {}).get("context"),
         lambda v: isinstance(v, dict) and bool(v.get("files"))),
        ("internal_prompt_token_ids", lambda d: [
            {"turn": i, "hashes": sorted({h for s in d["result"].get("sessions", [])
                if len(s.get("turns", [])) > i
                for h in [(s["turns"][i].get("prompt_ids_sha256") or
                           (s["turns"][i].get("neural") or {}).get("prompt_ids_sha256"))] if h})}
            for i in range(max((len(s.get("turns", [])) for s in d["result"].get("sessions", [])), default=0))],
         lambda v: bool(v) and all(r["hashes"] for r in v))):
        values = [pick(d) for d in docs]
        if len(values) > 1:
            is_known = all(known(v) for v in values)
            same = all(v == values[0] for v in values) if is_known else None
            compatibility.append({"check": label, "known_for_all_inputs": is_known,
                                  "matches_across_inputs": same,
                                  "evidence": values if same is not True else values[0]})
    output = {"inputs": [{"path": d["path"], "config": d["result"].get("config"),
                          "provenance": d["result"].get("provenance"),
                          "has_replay_archive": bool(d["archive"]), "checks": d["checks"],
                          "aggregate": d["summary"]}
                         for d in docs], "cross_input_checks": compatibility}
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        for d in docs:
            print(f"\n{d['path']}")
            cfg = d["result"].get("config", {})
            prov = d["result"].get("provenance") or {}
            print(f"  archive={'present' if d['archive'] else 'MISSING'}  "
                  f"model={prov.get('gguf')}  ctx_root={cfg.get('ctx_root')}")
            print(f"  request/response/link hashes valid={d['checks']['request_archive_integrity']}/"
                  f"{d['checks']['response_archive_integrity']}/{d['checks']['result_archive_link_integrity']}; request matches by turn="
                  f"{d['checks']['request_match_across_sessions_by_turn']}; internal token prompts by turn="
                  f"{d['checks']['prompt_token_ids_match_by_turn']}")
            for kind, row in d["summary"].items():
                w = row["all_turns_weighted_decode"]
                print(f"  {kind}: n={row['sessions']} sessions/{row['turns']} turns; "
                      f"first={row['first_decode_tok_s_median']} tok/s; "
                      f"follow-up median={row['followup_decode_tok_s_median']} tok/s; "
                      f"weighted={row['all_turns_weighted_decode_tok_s']} tok/s "
                      f"(n={w['tokens']} / {w['seconds']} s); "
                      f"output identities={row['output_identity_by_turn']}")
        for check in compatibility:
            state = "MATCH" if check["matches_across_inputs"] is True else (
                "DIFFER" if check["matches_across_inputs"] is False else "UNKNOWN")
            print(f"\n{check['check']}: {state}; {check['evidence']}")
        if any(not d["archive"] for d in docs):
            print("\nCAVEAT: historical results lack exact request/response replay archives; "
                  "their saved hashes and summaries cannot prove prompt identity or full output quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
