"""Fresh-server, no-prefix-reuse prefill A/B for Neural's grouped GEMM switch.

Each scheduled session starts a fresh server, sends a discarded 13k warm-up,
then measured 13k code, 3k file, and a second unique 13k source ordering. Every
request is archived with its full response. Nothing is benchmarked at import.

Example (from experiments/neuralserver-opt):
  python tools/review_warm_prefill.py benchmarks/warm_prefill.json \
      --schedule off,g1,g3,g3,g1,off --quick --ctx-root F:\\AI\\NeuralServer
  python tools/review_warm_prefill.py benchmarks/warm_prefill_llama.json \
      --schedule llama --ctx-root F:\\AI\\NeuralServer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ("fused_core.py", "harmony_render.py", "cpu_prefill.py", "server.py")
BASE_NEURAL_ARGS = (
    "--pool 6.05 --kv-ring 256 --scratch 6 --kernel-fuse 1 "
    "--prefill-order layer --smax 16384 --threads 8 --hotset hotset_code.json "
    "--refresh-every 8 --refresh-m 16 --capbufs 16 --prefill-m 64 "
    "--early-every 4 --early-tokens 32 --static 250 --skip-quality "
    r"--store-dir G:\NeuralStores\gptoss120b_ps4"
)
PROMPT_SPECS = (
    ("warmup_13k", ("fused_core.py", "harmony_render.py", "cpu_prefill.py", "server.py"), 13000,
     "Warm-up. Summarize this Neural source snapshot in one sentence."),
    ("code_13k", ("server.py", "cpu_prefill.py", "harmony_render.py", "fused_core.py"), 13000,
     "Measured code prompt. Summarize this Neural source snapshot in one sentence."),
    ("file_3k", ("cpu_prefill.py", "harmony_render.py"), 3000,
     "Measured short file prompt. Summarize this Neural source snapshot in one sentence."),
    ("unique_13k", ("cpu_prefill.py", "fused_core.py", "server.py", "harmony_render.py"), 13000,
     "Measured second long prompt. Summarize this differently ordered Neural source snapshot in one sentence."),
)
SHORT_SWEEP_SPECS = (
    ("short_128", ("cpu_prefill.py",), 128,
     "Measured 128-token file prompt. Summarize this source excerpt in one sentence."),
    ("short_512", ("harmony_render.py",), 512,
     "Measured 512-token file prompt. Summarize this source excerpt in one sentence."),
    ("short_1024", ("fused_core.py",), 1024,
     "Measured 1024-token file prompt. Summarize this source excerpt in one sentence."),
    ("short_3000", ("server.py", "cpu_prefill.py", "harmony_render.py"), 3000,
     "Measured 3000-token file prompt. Summarize this source excerpt in one sentence."),
)
SHORT_SWEEP_WARMUP_SPEC = (
    "warmup_short_3k", ("fused_core.py", "server.py", "cpu_prefill.py", "harmony_render.py"), 3000,
    "Discarded short-sweep warm-up. Summarize this differently ordered Neural source snapshot in one sentence.")
MAX_CACHED_PREFIX = 64


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return sha_bytes(raw)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_bench_helpers(args):
    """Import the shared safe process/HTTP/provenance helpers without running its CLI."""
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    neural_args = (BASE_NEURAL_ARGS + f" --grouped-prefill-min-tokens {args.group_min_tokens}"
                   + f" --grouped-prefill-max-tokens {args.group_max_tokens}")
    old_argv = sys.argv
    sys.argv = [str(ROOT / "tools" / "bench_vs_llama.py"), "_helper_unused.json",
                "--ctx-root", str(args.ctx_root), "--neural-args", neural_args,
                "--python", args.python, "--log-dir", str(args.log_dir),
                "--llama-dir", args.llama_dir, "--gguf", args.gguf]
    try:
        import bench_vs_llama as bench  # noqa: PLC0415
    finally:
        sys.argv = old_argv
    # The helper stores CLI settings in a module global; point it at this runner's
    # fixed snapshot and server checkout while retaining its start/stop/post code.
    bench.A.ctx_root = str(args.ctx_root)
    bench.A.neural_args = neural_args
    bench.A.python = args.python
    bench.A.log_dir = str(args.log_dir)
    bench.A.llama_dir = args.llama_dir
    bench.A.gguf = args.gguf
    bench.A.llama_args = args.llama_args
    bench.A.server_roots = [f"{kind}={args.server_root}" for kind in ("off", "group", "g1", "g3", "g6")]
    return bench


def build_prompts(bench, ctx_root: Path, short_sweep: bool = False):
    """Read source files once, then freeze all prompt text and provenance."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(bench.NP.MODEL_DIR, local_files_only=True)
    files = {}
    file_meta = {}
    for name in SOURCE_NAMES:
        path = ctx_root / name
        raw = path.read_bytes()
        files[name] = raw.decode("utf-8")
        file_meta[name] = {"path": str(path.resolve()), "size": len(raw), "sha256": sha_bytes(raw)}

    specs = (PROMPT_SPECS[0], SHORT_SWEEP_WARMUP_SPEC, *SHORT_SWEEP_SPECS) if short_sweep else PROMPT_SPECS
    prompts = []
    for name, order, count, instruction in specs:
        source = "\n\n".join(f"# ===== {f} =====\n{files[f]}" for f in order)
        token_ids = tok(source, add_special_tokens=False).input_ids[:count]
        context = tok.decode(token_ids)
        # Distinct marker at the very beginning prevents a large shared prefix
        # even if the same source files occur in another order.
        marker = (f"Neural prefill short-sweep case {name} ({count} context tokens; "
                  f"source={','.join(order)})." if short_sweep and name != "warmup_13k" else
                  f"Neural prefill review request {name}.")
        user = (f"{marker}\n{instruction}\n\n"
                "```python\n" + context + "\n```")
        body = {"model": "x", "messages": [{"role": "user", "content": user}],
                "max_tokens": 1, "temperature": 0, "reasoning_effort": "low",
                "chat_template_kwargs": {"reasoning_effort": "low"}}
        is_warmup = name.startswith("warmup_")
        prompts.append({"name": name, "measured": not is_warmup, "body": body,
                        "prompt_marker": marker, "short_sweep": short_sweep,
                        "no_prefix_reuse_expected": short_sweep and not is_warmup,
                        "source_order": list(order), "context_tokens": len(token_ids),
                        "context_token_ids_sha256": sha_bytes(
                            b"".join(int(i).to_bytes(4, "little") for i in token_ids)),
                        "context_sha256": sha_bytes(context.encode("utf-8")),
                        "request_sha256": sha_json(body)})
    if short_sweep:
        markers = [p["prompt_marker"] for p in prompts if p["measured"]]
        if len(markers) != len(set(markers)):
            raise ValueError("short-sweep prompt markers must be unique to prevent long-prefix reuse")
    tok_meta = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "tokenizer.json", "config.json"):
        p = Path(bench.NP.MODEL_DIR) / name
        if p.is_file():
            data = p.read_bytes()
            tok_meta[name] = {"path": str(p.resolve()), "size": len(data), "sha256": sha_bytes(data)}
    return prompts, {"files_read_once": file_meta, "tokenizer_dir": str(Path(bench.NP.MODEL_DIR).resolve()),
                    "tokenizer_artifacts": tok_meta}


def process_memory(proc=None):
    if proc is None:
        return None
    try:
        import psutil
    except ImportError:
        return {"unavailable": "psutil is not installed"}
    try:
        root = psutil.Process(proc.pid)
        procs = [root] + root.children(recursive=True)
        rows = []
        for p in procs:
            try:
                info = p.memory_info()
                rows.append({"pid": p.pid, "rss_gib": round(info.rss / 2**30, 3),
                             "vms_gib": round(info.vms / 2**30, 3)})
            except psutil.Error:
                continue
        return {"scope": "server process and recursive children", "processes": rows,
                "rss_total_gib": round(sum(x["rss_gib"] for x in rows), 3)}
    except psutil.Error as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def summarize(sessions):
    rows = {}
    for kind in ("off", "group", "g1", "g3", "g6", "llama"):
        eligible = [s for s in sessions if s.get("kind") == kind]
        if not eligible:
            continue
        prompts = {}
        prompt_names = list(dict.fromkeys(
            r.get("prompt") for s in eligible for r in s.get("requests", [])
            if r.get("measured") and r.get("prompt")))
        for p_name in prompt_names:
            values = [r["stats"].get("prompt_s") for s in eligible for r in s["requests"]
                      if r["prompt"] == p_name and r.get("stats", {}).get("prompt_s") is not None]
            prompts[p_name] = {"count": len(values), "prompt_s": values,
                               "median_prompt_s": statistics.median(values) if values else None}
        rows[kind] = {"session_count": len(eligible), "measured_prompts": prompts}
    return rows


def run_session(bench, kind: str, tag: str, group_size: int, group_rows: int | None, prompts, result,
                save_progress, source_meta, server_root: Path, group_trim_cache: int = 0):
    is_neural = kind != "llama"
    grouped = kind != "off"
    effective_group_size = int(kind[1:]) if kind in ("g1", "g3", "g6") else group_size
    extra = "--grouped-prefill-gemm 1" if grouped else "--grouped-prefill-gemm 0"
    if is_neural and grouped:
        extra += f" --trim-prefill-cache {group_trim_cache}"
    start_kind = f"{kind}={extra}" if is_neural else "llama"
    env_keys = ("NEURAL_PREFILL_GROUP_EXPERTS", "NEURAL_PREFILL_GROUP_ROWS")
    old_env = {key: os.environ.get(key) for key in env_keys}
    old_exists = {key: key in os.environ for key in env_keys}
    effective_group_rows = (str(group_rows) if group_rows is not None else
                            (old_env["NEURAL_PREFILL_GROUP_ROWS"] if old_exists["NEURAL_PREFILL_GROUP_ROWS"]
                             else "default(16384)"))
    proc = None
    session = {"kind": kind, "tag": tag,
              "server_args": bench.A.neural_args + (" " + extra if is_neural else ""),
              "llama_args": None if is_neural else bench.A.llama_args,
              "group_experts_env": effective_group_size if is_neural else None,
              "group_rows_env_override": group_rows if is_neural else None,
              "group_rows_env_effective": effective_group_rows if is_neural else None,
              "trim_prefill_cache": group_trim_cache if is_neural and grouped else (0 if is_neural else None),
              "requests": [], "process_memory": {},
              "free_ram_gib_at_start": bench.free_ram_gib()}
    result["sessions"].append(session)
    save_progress()
    try:
        if is_neural:
            os.environ["NEURAL_PREFILL_GROUP_EXPERTS"] = str(effective_group_size)
            if group_rows is not None:
                os.environ["NEURAL_PREFILL_GROUP_ROWS"] = str(group_rows)
        try:
            proc, base, ready_url = bench.start(start_kind, tag)
        finally:
            # The child inherits the setting at process creation; restore the
            # parent environment immediately, before readiness waits or HTTP work.
            for key in env_keys:
                if old_exists[key]:
                    os.environ[key] = old_env[key]
                else:
                    os.environ.pop(key, None)
        session["server_log"] = str(Path(bench.A.log_dir) / f"bench_{tag}_{kind}.log")
        session["startup_s"] = round(bench.wait_ready(ready_url, proc), 2)
        session["free_ram_gib_when_ready"] = bench.free_ram_gib()
        session["process_memory"]["when_ready"] = process_memory(proc)
        save_progress()
        for index, prompt in enumerate(prompts):
            saved_body, request_hash, messages_hash = bench.snapshot_request(prompt["body"])
            response, wall = bench.post(base + "/v1/chat/completions", prompt["body"], timeout=3600)
            stats = bench.turn_stats(kind, response, wall)
            message = (response.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            entry = {"index": index, "prompt": prompt["name"], "measured": prompt["measured"],
                     "source_order": prompt["source_order"], "context_tokens": prompt["context_tokens"],
                     "context_token_ids_sha256": prompt["context_token_ids_sha256"],
                     "context_sha256": prompt["context_sha256"],
                     "prompt_marker": prompt["prompt_marker"],
                     "short_sweep": prompt["short_sweep"],
                     "no_prefix_reuse_expected": prompt["no_prefix_reuse_expected"],
                     "request": saved_body, "request_sha256": request_hash,
                     "messages_sha256": messages_hash, "response": response,
                     "response_sha256": sha_json(response), "prompt_ids_sha256": stats.get("prompt_ids_sha256"),
                     "generated_ids_sha256": stats.get("generated_ids_sha256"),
                     "cached_prefix_tokens": stats.get("cached_tokens"), "stats": stats,
                     "content_sha256": sha_bytes(content.encode("utf-8")),
                     "reasoning_sha256": sha_bytes(reasoning.encode("utf-8")),
                     "request_source_metadata": source_meta,
                     "free_ram_gib_after": bench.free_ram_gib(),
                     "process_memory_after": process_memory(proc)}
            session["requests"].append(entry)
            save_progress()
            print(f"[{tag} {kind}] {prompt['name']}: {json.dumps(stats, ensure_ascii=False)}", flush=True)
            if stats.get("cached_tokens") is not None and stats["cached_tokens"] > MAX_CACHED_PREFIX:
                raise RuntimeError(f"rejected long-KV reuse for {prompt['name']}: "
                                   f"cached_prefix_tokens={stats['cached_tokens']} > {MAX_CACHED_PREFIX}")
        session["process_memory"]["before_stop"] = process_memory(proc)
    except BaseException:
        session["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if proc is not None:
                bench.stop(proc)
        finally:
            for key in env_keys:
                if old_exists[key]:
                    os.environ[key] = old_env[key]
                else:
                    os.environ.pop(key, None)
            session["process_memory"]["after_stop"] = process_memory(proc)
            session["free_ram_gib_after_stop"] = bench.free_ram_gib()
            save_progress()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="incremental atomic JSON result path")
    ap.add_argument("--schedule", default="off,g1,g3,g3,g1,off",
                    help="comma-separated fresh-server order: off, group, g1, g3, g6, or llama")
    ap.add_argument("--group-size", type=int, default=0,
                    help="fallback NEURAL_PREFILL_GROUP_EXPERTS for the 'group' kind")
    ap.add_argument("--group-rows", type=int, default=None,
                    help="optional positive NEURAL_PREFILL_GROUP_ROWS cap for grouped-prefill sessions")
    ap.add_argument("--group-min-tokens", type=int, default=0,
                    help="explicit server guard minimum; default 0 tests tiny prompts too")
    ap.add_argument("--group-max-tokens", type=int, default=4096,
                    help="explicit server guard maximum; 0 tests unbounded grouping")
    ap.add_argument("--group-trim-cache", type=int, choices=(0, 1), default=0,
                    help="enable CUDA cache release for grouped variants only; the off control keeps cache release off")
    ap.add_argument("--short-sweep", action="store_true",
                    help="replace the default measured prompts with unique-source 128/512/1024/3000-token file prompts")
    ap.add_argument("--quick", action="store_true",
                    help="run only a 13k discarded warm-up and the measured 3k file prompt per server")
    ap.add_argument("--ctx-root", default=str(ROOT), help="fixed checkout used for the prompt sources")
    ap.add_argument("--server-root", default=str(ROOT), help="checkout used by fresh Neural servers")
    ap.add_argument("--python", default=r"F:\AI\Neural\.venv\Scripts\python.exe")
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--llama-dir", default=r"G:\Tools\llama-b10361")
    ap.add_argument("--gguf", default=r"G:\Models\gpt-oss-120b-gguf\gpt-oss-120b-MXFP4.gguf")
    ap.add_argument("--llama-args", default="-ngl 99 -ncmoe 31 -t 8 -c 16384 -fa on --jinja")
    args = ap.parse_args()
    schedule = [x.strip().lower() for x in args.schedule.split(",") if x.strip()]
    if not schedule or any(x not in ("off", "group", "g1", "g3", "g6", "llama") for x in schedule):
        ap.error("--schedule entries must be off, group, g1, g3, g6, or llama")
    if args.group_size < 0:
        ap.error("--group-size must be >= 0")
    if args.group_rows is not None and args.group_rows <= 0:
        ap.error("--group-rows must be a positive integer")
    if (args.group_min_tokens < 0 or args.group_max_tokens < 0 or
            (args.group_max_tokens and args.group_min_tokens > args.group_max_tokens)):
        ap.error("group token bounds must be nonnegative, with minimum <= a positive maximum")
    ctx_root, server_root = Path(args.ctx_root).resolve(), Path(args.server_root).resolve()
    out_path = Path(args.out).resolve()
    error_path = Path(str(out_path) + ".error.json")
    bench = load_bench_helpers(args)
    prompts, source_meta = build_prompts(bench, ctx_root, short_sweep=args.short_sweep)
    if args.quick:
        measured_quick_name = "short_3000" if args.short_sweep else "file_3k"
        warmup_names = {"warmup_13k", "warmup_short_3k"} if args.short_sweep else {"warmup_13k"}
        prompts = [p for p in prompts if p["name"] in (warmup_names | {measured_quick_name})]
    neural_prov = bench.repo_provenance(str(server_root)) if hasattr(bench, "repo_provenance") else None
    artifacts = bench.root_artifacts(str(server_root), bench.A.neural_args, bench.NP.MODEL_DIR)
    result = {"schema": "neural-warm-prefill-review-v1",
              "config": {"schedule": schedule, "group_size_env_fallback": args.group_size,
                         "group_size_by_kind": {"g1": 1, "g3": 3, "g6": 6, "group": args.group_size},
                         "group_rows_env_override": args.group_rows,
                         "group_rows_default_if_unset": 16384,
                         "group_min_tokens": args.group_min_tokens,
                         "group_max_tokens": args.group_max_tokens,
                         "group_trim_cache": args.group_trim_cache,
                         "short_sweep": args.short_sweep,
                         "short_sweep_context_tokens": [128, 512, 1024, 3000] if args.short_sweep else None,
                         "warmup_prompts": [p["name"] for p in prompts if not p["measured"]],
                         "quick": args.quick,
                         "neural_args": bench.A.neural_args, "llama_args": args.llama_args,
                         "ctx_root": str(ctx_root), "server_root": str(server_root),
                         "max_tokens": 1, "temperature": 0, "effort": "low",
                         "max_cached_prefix_tokens": MAX_CACHED_PREFIX,
                         "measured_prompts": [p["name"] for p in prompts if p["measured"]]},
              "provenance": {"source": source_meta, "neural_repo": neural_prov,
                             "source_and_dll_store_hashes": artifacts,
                             "runner": str(Path(__file__).resolve()),
                             "runner_sha256": sha_bytes(Path(__file__).read_bytes()),
                             "ctx_root_git": bench.repo_provenance(str(ctx_root)),
                             "server_root_git": bench.repo_provenance(str(server_root))},
              "sessions": [], "summary": {}, "errors": []}
    save = lambda: atomic_json(out_path, result)
    save()
    try:
        for index, kind in enumerate(schedule):
            try:
                run_session(bench, kind, f"s{index}", args.group_size, args.group_rows, prompts, result, save,
                            source_meta, server_root, group_trim_cache=args.group_trim_cache)
            except BaseException as exc:
                result["errors"].append({"session": index, "kind": kind,
                                         "error": f"{type(exc).__name__}: {exc}"})
                result["summary"] = summarize(result["sessions"])
                save()
                atomic_json(error_path, {"schema": result["schema"], "out": str(out_path),
                                         "errors": result["errors"], "traceback": traceback.format_exc()})
                raise
            result["summary"] = summarize(result["sessions"])
            save()
    except BaseException:
        return 1
    if error_path.exists():
        error_path.unlink()
    print("SUMMARY", json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
