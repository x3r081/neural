import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


TOOL = Path(__file__).parents[1] / "tools" / "review_bench.py"
spec = importlib.util.spec_from_file_location("review_bench", TOOL)
review_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_bench)
bench_path = TOOL.with_name("bench_vs_llama.py")
bench_spec = importlib.util.spec_from_file_location("bench_vs_llama", bench_path)
bench_vs_llama = importlib.util.module_from_spec(bench_spec)
old_argv = sys.argv
try:
    sys.argv = [str(bench_path), "unused.json"]
    bench_spec.loader.exec_module(bench_vs_llama)
finally:
    sys.argv = old_argv


def test_canonical_json_hash_ignores_object_key_order():
    assert review_bench.sha256_json({"a": 1, "b": 2}) == review_bench.sha256_json({"b": 2, "a": 1})


def test_provenance_fingerprints_effective_dll_and_store_override(tmp_path):
    import hashlib
    (tmp_path / "candidate.dll").write_bytes(b"separate candidate build")
    (tmp_path / "old_store").mkdir()
    store = tmp_path / "new_store"
    store.mkdir()
    (store / "meta.json").write_text('{"revision": 2}')
    result = bench_vs_llama.root_artifacts(str(tmp_path),
        "--kdll old.dll --store-dir old_store --kdll=candidate.dll --store-dir=new_store",
        str(tmp_path / "tokenizer"))
    assert result["selected_decode_dll"]["sha256"] == hashlib.sha256(b"separate candidate build").hexdigest()
    assert result["store_dir"]["path"] == str(store.resolve())
    assert "store/meta.json" in result


def test_aggregate_uses_neural_steady_decode_tokens_and_checks_hashes():
    request = {"model": "x", "messages": [{"role": "user", "content": "fixed"}]}
    response = {"choices": [{"message": {"content": "answer"}}]}
    req_hash = review_bench.sha256_json(request)
    response_hash = review_bench.sha256_json(response)
    result = {"sessions": [{"kind": "neural", "turns": [{
        "completion_tokens": 512, "decode_tokens_timed": 511, "decode_s": 25.0,
        "decode_tok_s": 20.44, "prompt_ids_sha256": "prompt", "generated_ids_sha256": "generated"
    }]}]}
    archive = {"sessions": [{"kind": "neural", "requests": [{
        "request": request, "request_sha256": req_hash, "response": response,
        "response_sha256": response_hash
    }]}]}
    summary, checks = review_bench.aggregate(result, archive)
    assert summary["neural"]["all_turns_weighted_decode"]["tokens"] == 511
    assert summary["neural"]["all_turns_weighted_decode_tok_s"] == 20.44
    assert checks["request_archive_integrity"] is True
    assert checks["response_archive_integrity"] is True
    assert checks["request_match_across_sessions_by_turn"] == [{"turn": 0, "distinct_hashes": 1, "matched": True}]


def test_aggregate_flags_prompt_date_or_template_mismatch():
    result = {"sessions": [
        {"kind": "neural", "turns": [{"prompt_ids_sha256": "day-a"}]},
        {"kind": "llama", "turns": [{"prompt_ids_sha256": "day-b"}]},
    ]}
    archive = {"sessions": [
        {"kind": "neural", "requests": [{"request": {}, "request_sha256": review_bench.sha256_json({}),
                                             "response": {}, "response_sha256": review_bench.sha256_json({})}]},
        {"kind": "llama", "requests": [{"request": {}, "request_sha256": review_bench.sha256_json({}),
                                            "response": {}, "response_sha256": review_bench.sha256_json({})}]},
    ]}
    _, checks = review_bench.aggregate(result, archive)
    assert checks["prompt_token_ids_match_by_turn"] == [{"turn": 0, "samples": 2, "hashes_present": 2,
        "missing_hash_count": 0, "distinct_hashes": 2, "matched": False, "matched_among_known": False}]


def test_request_snapshot_is_immutable_after_messages_mutate():
    body = {"messages": [{"role": "user", "content": "first"}]}
    saved, request_hash, messages_hash = bench_vs_llama.snapshot_request(body)
    body["messages"].append({"role": "assistant", "content": "later"})
    assert saved["messages"] == [{"role": "user", "content": "first"}]
    assert request_hash == review_bench.sha256_json(saved)
    assert messages_hash == review_bench.sha256_json(saved["messages"])


def test_first_session_archive_replay_returns_verified_copy_and_source_metadata():
    request = {"model": "x", "messages": [{"role": "user", "content": "seed"}]}
    body, request_hash, messages_hash = bench_vs_llama.snapshot_request(request)
    archive = {"sessions": [{"kind": "off", "tag": "s0", "requests": [{
        "request": body, "request_sha256": request_hash, "messages_sha256": messages_hash
    }]}]}
    requests, source = bench_vs_llama.replay_requests_from_archive(archive)
    archive["sessions"][0]["requests"][0]["request"]["messages"][0]["content"] = "mutated"
    assert requests[0]["messages"][0]["content"] == "seed"
    assert source["kind"] == "off" and source["tag"] == "s0"
    assert source["request_sha256_by_turn"] == [request_hash]


def test_old_nested_and_new_flat_stats_are_aggregated():
    old = {"sessions": [{"kind": "neural", "turns": [{
        "completion_tokens": 11, "decode_tok_s": 2.0,
        "neural": {"decode_s": 5.0, "prompt_ids_sha256": "p0", "generated_ids_sha256": "g0"}
    }]}]}
    new = {"sessions": [{"kind": "neural", "turns": [{
        "completion_tokens": 11, "decode_tokens_timed": 10, "decode_s": 5.0,
        "decode_tok_s": 2.0, "prompt_ids_sha256": "p0", "generated_ids_sha256": "g0"
    }]}]}
    old_summary, _ = review_bench.aggregate(old, None)
    new_summary, _ = review_bench.aggregate(new, None)
    assert old_summary["neural"]["all_turns_weighted_decode"] == new_summary["neural"]["all_turns_weighted_decode"]
    assert old_summary["neural"]["all_turns_weighted_decode"]["tokens"] == 10


def test_missing_hashes_are_unknown_and_tampering_is_detected():
    req, response = {"messages": []}, {"choices": []}
    req_hash = review_bench.sha256_json(req)
    res_hash = review_bench.sha256_json(response)
    result = {"sessions": [
        {"kind": "neural", "turns": [{"completion_tokens": 1}]},
        {"kind": "neural", "turns": [{"completion_tokens": 1}]},
    ]}
    archive = {"sessions": [
        {"kind": "neural", "requests": [{"request": req, "request_sha256": req_hash,
                                             "response": response, "response_sha256": res_hash}]},
        {"kind": "neural", "requests": [{"request": req, "request_sha256": req_hash,
                                             "response": {"tampered": True}, "response_sha256": res_hash}]},
    ]}
    summary, checks = review_bench.aggregate(result, archive)
    assert summary["neural"]["output_identity_by_turn"][0]["identical"] is None
    assert checks["prompt_token_ids_match_by_turn"][0]["matched"] is None
    assert checks["request_archive_integrity"] is True
    assert checks["response_archive_integrity"] is False


def test_one_session_output_hash_is_not_claimed_as_verified_identity():
    result = {"sessions": [{"kind": "neural", "turns": [{"content_sha1": "stable"}]}]}
    summary, _ = review_bench.aggregate(result, None)
    identity = summary["neural"]["output_identity_by_turn"][0]
    assert identity["identical"] is None
    assert identity["verification"] == "not_verified_single_session"
    assert identity["missing_identity_count"] == 0


def test_current_neural_launcher_defaults_are_used_without_overriding_store_env():
    args = bench_vs_llama.A.neural_args.split()
    assert "--pool" in args and args[args.index("--pool") + 1] == "6.05"
    assert "--kv-ring" in args and args[args.index("--kv-ring") + 1] == "256"
    assert "--scratch" in args and args[args.index("--scratch") + 1] == "6"
    assert "--kernel-fuse" in args and args[args.index("--kernel-fuse") + 1] == "1"
    assert "--prefill-order" in args and args[args.index("--prefill-order") + 1] == "layer"
    assert "--refresh-m" in args and args[args.index("--refresh-m") + 1] == "16"
    assert "--store-dir" not in args


def test_failed_post_is_archived_as_failed_and_reviewed_as_partial():
    replay = {"kind": "neural", "tag": "failed", "requests": []}
    result = {"kind": "neural", "turns": []}
    saves = []
    with patch.object(bench_vs_llama, "free_ram_gib", return_value=5.0), \
         patch.object(bench_vs_llama, "start", return_value=(object(), "http://fake", "ready")), \
         patch.object(bench_vs_llama, "wait_ready", return_value=0.1), \
         patch.object(bench_vs_llama, "post", side_effect=OSError("mock transport failure")), \
         patch.object(bench_vs_llama, "stop"):
        try:
            bench_vs_llama.session("neural", "failed", "context", replay_record=replay,
                                   result_record=result, on_progress=lambda: saves.append(True))
        except OSError:
            pass
        else:
            raise AssertionError("mock request should fail")
    assert result["status"] == "failed"
    assert replay["requests"][0]["status"] == "failed"
    assert replay["requests"][0]["request"]
    assert "mock transport failure" in replay["requests"][0]["error"]
    assert len(saves) >= 2  # pending request saved before the POST, failure saved after it
    summary, checks = review_bench.aggregate({"sessions": [result]}, {"sessions": [replay]})
    assert summary["neural"]["all_sessions_complete"] is False
    assert summary["neural"]["session_status_counts"]["failed"] == 1
    assert checks["archive_capture_state"]["failed"] == 1
    assert checks["response_archive_integrity"] is None


def _prompt_id_check(hashes):
    result_sessions, archive_sessions = [], []
    req = {"request": {}, "request_sha256": review_bench.sha256_json({}),
           "response": {}, "response_sha256": review_bench.sha256_json({}), "status": "complete"}
    for i, prompt_hash in enumerate(hashes):
        turn = {} if prompt_hash is None else {"prompt_ids_sha256": prompt_hash}
        result_sessions.append({"kind": "neural", "status": "complete", "turns": [turn]})
        archive_sessions.append({"kind": "neural", "requests": [{**req, "turn": 0}]})
    _, checks = review_bench.aggregate({"sessions": result_sessions}, {"sessions": archive_sessions})
    return checks["prompt_token_ids_match_by_turn"][0]


def test_prompt_id_match_uses_observation_count_not_distinct_set_size():
    matched = _prompt_id_check(["same", "same"])
    assert matched["samples"] == 2 and matched["hashes_present"] == 2
    assert matched["matched"] is True and matched["matched_among_known"] is True


def test_prompt_id_match_reports_known_subset_when_some_engines_lack_hashes():
    partial = _prompt_id_check(["same", None])
    assert partial["missing_hash_count"] == 1
    assert partial["matched"] is None and partial["matched_among_known"] is None
    subset = _prompt_id_check(["same", "same", None])
    assert subset["matched"] is None and subset["matched_among_known"] is True


def test_prompt_id_mismatch_is_reported_with_two_known_hashes():
    mismatch = _prompt_id_check(["left", "right"])
    assert mismatch["matched"] is False and mismatch["matched_among_known"] is False


def test_response_integrity_failure_is_not_downgraded_by_later_failed_request():
    good = {"request": {}, "request_sha256": review_bench.sha256_json({}),
            "response": {"ok": True}, "response_sha256": "tampered", "status": "complete"}
    failed = {"request": {}, "request_sha256": review_bench.sha256_json({}),
              "status": "failed", "error": "transport"}
    archive = {"sessions": [{"kind": "neural", "requests": [good, failed]}]}
    result = {"sessions": [{"kind": "neural", "status": "failed", "turns": []}]}
    _, checks = review_bench.aggregate(result, archive)
    assert checks["response_archive_integrity"] is False
