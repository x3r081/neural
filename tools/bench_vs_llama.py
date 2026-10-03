"""Interleaved A/B of the Neural server against llama-server on one long coding conversation.

Each session starts one server, sends a ~13k-token code context with a question, then
follow-up questions in the same conversation (prefix cached), and stops the server. Sessions
alternate Neural / llama.cpp for --rounds rounds, so both see the same machine state (the
other server's weights in the page cache at start). Decode speed comes from each server's
own timings (Neural: `neural.decode_tok_s`; llama-server: `timings.predicted_per_second`).

    python tools/bench_vs_llama.py benchmarks/vs_llama.json [--rounds 2] [--kinds neural llama]
    # Neural variants: --kinds "neural" "old=--prefill-weight 1073741824" llama  (NAME=extra server args)

Nothing else should run on the machine meanwhile (RAM bandwidth is the measured resource).
"""
import argparse, copy, ctypes, hashlib, json, os, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import paths as NP                                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--schedule", default=None,
                help="explicit comma-separated session order by kind name (overrides --rounds); the page "
                     "cache left by the previous session matters, so give each variant the same predecessors")
ap.add_argument("--kinds", nargs="+", default=["neural", "llama"],
                help="sessions per round: llama, neural, or NAME=extra Neural args (a Neural variant)")
ap.add_argument("--workload", choices=["conversation", "rewrite"], default="conversation",
                help="conversation: 13k-token code context + 4 questions; rewrite: read one file, rewrite it whole "
                     "(the answer continues the prompt, like an editor apply step)")
ap.add_argument("--ctx-tokens", type=int, default=None, help="default 13000 (conversation) / 3000 (rewrite)")
ap.add_argument("--max-tokens", type=int, default=None, help="default 512 (conversation) / 1536 (rewrite)")
ap.add_argument("--effort", default="low")
ap.add_argument("--llama-dir", default=r"G:\Tools\llama-b10361")
ap.add_argument("--gguf", default=r"G:\Models\gpt-oss-120b-gguf\gpt-oss-120b-MXFP4.gguf")
ap.add_argument("--llama-args", default="-ngl 99 -ncmoe 31 -t 8 -c 16384 -fa on --jinja")
ap.add_argument("--neural-args", default="--pool 6.05 --smax 16384 --threads 8 --hotset hotset_code.json "
                "--refresh-every 8 --refresh-m 16 --capbufs 16 --prefill-m 64 --early-every 4 "
                "--early-tokens 32 --static 250 --skip-quality --kv-ring 256 --scratch 6 "
                "--kernel-fuse 1 --prefill-order layer")
ap.add_argument("--python", default=None, help="python for server.py (default: repo .venv, else F:\\AI\\Neural\\.venv)")
ap.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
ap.add_argument("--server-roots", nargs="*", default=[],
                help="NAME=PATH: start server.py for session kind NAME from another checkout (A/B across trees, "
                     "e.g. main=F:\\AI\\NeuralServer_main); other kinds run from this checkout")
ap.add_argument("--ctx-root", default=ROOT,
                help="checkout whose source files form the prompt, so every session sees the same tokens")
replay_group = ap.add_mutually_exclusive_group()
replay_group.add_argument("--replay-prompts", default=None,
                help="replay exact request bodies from a prior <out>.replay.json archive for matched comparisons")
replay_group.add_argument("--replay-first-session", action="store_true",
                help="run the first scheduled session normally, then replay its exact request sequence in every later session")
ap.add_argument("--replay-session", type=int, default=0,
                help="session index in --replay-prompts whose request sequence is replayed (default 0)")
A = ap.parse_args()
if A.ctx_tokens is None:
    A.ctx_tokens = 13000 if A.workload == "conversation" else 3000
if A.max_tokens is None:
    A.max_tokens = 512 if A.workload == "conversation" else 1536

QUESTIONS = [
    "Explain in detail how the split-K decode attention in this code works, step by step, "
    "then write a small standalone test for it.",
    "Write a Python function that checks the prompt attention against a straightforward reference "
    "implementation, with detailed comments.",
    "Refactor the CPU multi-token expert wrapper into a small class with type hints. Show the full code.",
    "List five potential bugs or edge cases in this code and show a fix for each.",
]


def sha256_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def snapshot_request(body):
    saved = json.loads(json.dumps(body, ensure_ascii=False))
    return saved, sha256_json(saved), sha256_json(saved.get("messages", []))


def replay_requests_from_archive(archive, session_index=0):
    sessions = archive.get("sessions", [])
    if not 0 <= session_index < len(sessions):
        raise ValueError(f"replay session {session_index} outside archive session count {len(sessions)}")
    source = sessions[session_index]
    entries = source.get("requests", [])
    if not entries:
        raise ValueError("selected replay session has no saved requests")
    requests = []
    for i, entry in enumerate(entries):
        saved, request_hash, messages_hash = snapshot_request(entry["request"])
        if request_hash != entry.get("request_sha256"):
            raise ValueError(f"replay archive request hash validation failed at turn {i}")
        if messages_hash != entry.get("messages_sha256"):
            raise ValueError(f"replay archive message hash validation failed at turn {i}")
        requests.append(saved)
    metadata = {"session_index": session_index, "kind": source.get("kind"), "tag": source.get("tag"),
                "request_sha256_by_turn": [entry.get("request_sha256") for entry in entries],
                "messages_sha256_by_turn": [entry.get("messages_sha256") for entry in entries]}
    return requests, metadata


def file_provenance(path):
    path = os.path.abspath(path)
    try:
        st = os.stat(path)
        return {"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        return {"path": path, "missing": True}


def repo_provenance(root):
    def git(*args):
        try:
            return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except Exception:
            return None
    status = git("status", "--porcelain")
    diff = git("diff", "--binary", "HEAD")
    source_files = []
    try:
        listed = subprocess.run(["git", "-C", root, "ls-files", "-co", "--exclude-standard"],
                                capture_output=True, text=True, timeout=20, check=True).stdout.splitlines()
        for rel in listed:
            norm = rel.replace("\\", "/")
            if not norm.lower().endswith((".py", ".c", ".h", ".cpp", ".bat")):
                continue
            if any(part.lower() in {"benchmarks", "benchmark", "__pycache__", ".venv", "venv", ".git"}
                   for part in norm.split("/")):
                continue
            path = os.path.join(root, rel)
            try:
                st = os.stat(path)
                if st.st_size > 64 * 1024 * 1024:
                    source_files.append({"path": norm, "size": st.st_size, "sha256": None,
                                         "hash_status": "too_large"})
                else:
                    source_files.append({"path": norm, **small_file_provenance(path)})
            except OSError:
                source_files.append({"path": norm, "missing": True})
    except Exception:
        source_files = None
    return {"root": os.path.abspath(root), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(status) if status is not None else None,
            "working_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff is not None else None,
            "source_manifest": source_files}


def context_text(with_metadata=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NP.MODEL_DIR, local_files_only=True)
    src = []
    files = ("fused_core.py", "harmony_render.py", "cpu_prefill.py", "server.py") if A.workload == "conversation" \
        else ("harmony_render.py",)
    for f in files:
        src.append(f"# ===== {f} =====\n" + open(os.path.join(A.ctx_root, f), encoding="utf-8").read())
    ids = tok("\n\n".join(src), add_special_tokens=False).input_ids[:A.ctx_tokens]
    ctx = tok.decode(ids)
    meta = {"files": [{"path": os.path.abspath(os.path.join(A.ctx_root, f)),
                             "sha256": hashlib.sha256(open(os.path.join(A.ctx_root, f), "rb").read()).hexdigest()}
                            for f in files],
                 "tokenizer_dir": os.path.abspath(NP.MODEL_DIR), "ctx_tokens_requested": A.ctx_tokens,
                 "ctx_token_count": len(ids), "ctx_token_ids_sha256": hashlib.sha256(
                     b"".join(int(i).to_bytes(4, "little") for i in ids)).hexdigest(),
                 "decoded_context_sha256": hashlib.sha256(ctx.encode("utf-8")).hexdigest()}
    return (ctx, meta) if with_metadata else ctx


def small_file_provenance(path, include_hash=True, max_hash_bytes=64 * 1024 * 1024):
    info = file_provenance(path)
    if "size" in info and info["size"] <= max_hash_bytes and include_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def root_artifacts(root, neural_args, tokenizer_dir):
    files = {"server": os.path.join(root, "server.py")}
    for nm in ("gptoss_cpu_cap2.dll", "gptoss_cpu_multi.dll"):
        p = os.path.join(root, nm)
        if os.path.isfile(p):
            files[nm] = p
    for nm in ("tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
               "tokenizer.json", "config.json"):
        p = os.path.join(tokenizer_dir, nm)
        if os.path.isfile(p):
            files["tokenizer/" + nm] = p
    for nm in ("tools/bench_vs_llama.py", "tools/review_bench.py"):
        p = os.path.join(root, nm.replace("/", os.sep))
        if os.path.isfile(p):
            files[nm] = p
    args = neural_args.split()
    # Match argparse's last-option-wins behavior, including per-variant overrides.
    def last_option(flag, default=None):
        value = default
        for i, arg in enumerate(args):
            if arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            elif arg == flag and i + 1 < len(args):
                value = args[i + 1]
        return value

    kdll = last_option("--kdll", "gptoss_cpu_cap2.dll")
    files["selected_decode_dll"] = os.path.abspath(os.path.join(root, kdll))
    store = last_option("--store-dir")
    if store:
        store = os.path.abspath(store if os.path.isabs(store) else os.path.join(root, store))
        files["store_dir"] = store
        try:
            for p in __import__("glob").glob(os.path.join(store, "*.json")):
                if os.path.getsize(p) <= 1024 * 1024:
                    files["store/" + os.path.basename(p)] = p
        except OSError:
            pass
    return {name: (file_provenance(path) if os.path.isdir(path) else small_file_provenance(path))
            for name, path in files.items()}


def free_ram_gib():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong),
                    ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullAvailPhys / 2**30, 2)


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, time.perf_counter() - t0


def wait_ready(url, proc, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("server did not become ready")


def stop(proc):
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    try:
        proc.wait(30)
    except Exception:
        pass
    time.sleep(5)


def python_exe():
    for p in (A.python, os.path.join(ROOT, ".venv", "Scripts", "python.exe"), r"F:\AI\Neural\.venv\Scripts\python.exe"):
        if p and os.path.exists(p):
            return p
    return sys.executable


def port_free(port):
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start(kind, tag):
    os.makedirs(A.log_dir, exist_ok=True)
    port = 8080 if kind.partition("=")[0] == "llama" else 8000
    assert port_free(port), f"port {port} is in use: stop the running server first (it would be benchmarked instead)"
    name, _, extra = kind.partition("=")
    log = open(os.path.join(A.log_dir, f"bench_{tag}_{name}.log"), "w", encoding="utf-8")
    if name != "llama":
        cmd = [python_exe(), "server.py"] + A.neural_args.split() + extra.split()
        roots = dict(kv.split("=", 1) for kv in A.server_roots)
        proc = subprocess.Popen(cmd, cwd=roots.get(name, ROOT), stdout=log, stderr=subprocess.STDOUT)
        return proc, "http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/models"
    cmd = [os.path.join(A.llama_dir, "llama-server.exe"), "-m", A.gguf, "--port", "8080"] + A.llama_args.split()
    proc = subprocess.Popen(cmd, cwd=A.llama_dir, stdout=log, stderr=subprocess.STDOUT)
    return proc, "http://127.0.0.1:8080", "http://127.0.0.1:8080/health"


def turn_stats(kind, r, wall):
    u = r.get("usage") or {}
    if "neural" in r:
        n = r.get("neural") or {}
        return {"prompt_tokens": u.get("prompt_tokens"),
                "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "prompt_s": n.get("prefill_s"), "decode_s": n.get("decode_s"),
                "decode_tokens_timed": max(0, (u.get("completion_tokens") or 0) - 1),
                "completion_tokens": u.get("completion_tokens"),
                "decode_tok_s": n.get("decode_tok_s"), "vram_hit": n.get("vram_hit"),
                "cpu_ms_per_tok": n.get("cpu_ms_per_tok"), "gpu_wait_ms_per_tok": n.get("gpu_wait_ms_per_tok"),
                "cpu_GBps": n.get("cpu_GBps"), "prefill_weight": n.get("prefill_weight"),
                "prefill_fold_w": n.get("prefill_fold_w"), "cand_marked_start": n.get("cand_marked_start"),
                "captures": n.get("captures"), "prompt_ids_sha256": n.get("prompt_ids_sha256"),
                "generated_ids_sha256": n.get("generated_ids_sha256"), "wall_s": round(wall, 2),
                "neural": {k: v for k, v in n.items() if not k.startswith("splice")}}
    t = r.get("timings") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "cached_tokens": t.get("cache_n"),
            "prompt_s": (t["prompt_ms"] / 1e3) if t.get("prompt_ms") is not None else None,
            "decode_s": (t["predicted_ms"] / 1e3) if t.get("predicted_ms") is not None else None,
            "decode_tokens_timed": t.get("predicted_n") or u.get("completion_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "decode_tok_s": t.get("predicted_per_second"), "prompt_ids_sha256": None,
            "generated_ids_sha256": None, "wall_s": round(wall, 2), "timings": t}


def session(kind, tag, ctx, replay_requests=None, replay_record=None, result_record=None, on_progress=None):
    res = result_record if result_record is not None else {}
    res.update({"kind": kind.partition("=")[0], "args": kind.partition("=")[2], "tag": tag,
                "free_ram_gib_at_start": None, "turns": res.get("turns", []), "status": "running"})
    if replay_record is None:
        replay_record = {"kind": res["kind"], "tag": tag, "requests": []}
    proc = None
    try:
        res["free_ram_gib_at_start"] = free_ram_gib()
        proc, base, ready = start(kind, tag)
        res["startup_s"] = round(wait_ready(ready, proc), 1)
        res["free_ram_gib_ready"] = free_ram_gib()
        if A.workload == "rewrite":
            qs = ["Rewrite the whole file with type hints on every function. Output the complete file and nothing else."]
            msgs = [{"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Here is harmony_render.py:\n\n```python\n" + ctx + "\n```\n\n" + qs[0]}]
        else:
            qs = QUESTIONS
            msgs = [{"role": "user", "content": "Here is part of a codebase:\n\n```python\n" + ctx + "\n```\n\n" + qs[0]}]
        if replay_requests is not None:
            if len(replay_requests) != len(qs):
                raise ValueError(f"replay request count {len(replay_requests)} != workload turn count {len(qs)}")
        for i in range(len(qs)):
            if i:
                msgs.append({"role": "user", "content": qs[i]})
            body = {"model": "x", "messages": msgs, "max_tokens": A.max_tokens, "temperature": 0,
                    "reasoning_effort": A.effort, "chat_template_kwargs": {"reasoning_effort": A.effort}}
            if replay_requests is not None:
                body = replay_requests[i]
            saved_body, request_hash, messages_hash = snapshot_request(body)
            request_entry = {"turn": i, "request": saved_body, "request_sha256": request_hash,
                             "messages_sha256": messages_hash, "status": "pending"}
            replay_record["requests"].append(request_entry)
            if on_progress:
                on_progress()
            try:
                r, wall = post(base + "/v1/chat/completions", body)
            except BaseException as exc:
                request_entry.update({"status": "failed", "error_type": type(exc).__name__,
                                      "error": repr(exc)})
                raise
            st = turn_stats(kind, r, wall)
            st["turn"] = i
            message = (r.get("choices") or [{}])[0].get("message") or {}
            _c = message.get("content") or ""
            st["content_sha1"] = hashlib.sha1(_c.encode("utf-8")).hexdigest()   # compatibility with historical reports
            st["content_sha256"] = hashlib.sha256(_c.encode("utf-8")).hexdigest()
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            st["reasoning_sha256"] = hashlib.sha256(reasoning.encode("utf-8")).hexdigest()
            st["tool_calls_sha256"] = sha256_json(message.get("tool_calls") or [])
            semantic = {"role": message.get("role"), "content": message.get("content"),
                        "reasoning": reasoning,
                        "tool_calls": [{"type": tc.get("type"),
                                        "function": tc.get("function") or {}}
                                       for tc in (message.get("tool_calls") or [])]}
            st["semantic_message_sha256"] = sha256_json(semantic)
            st["response_sha256"] = sha256_json(r)
            st["request_sha256"] = request_entry["request_sha256"]
            st["messages_sha256"] = request_entry["messages_sha256"]
            st["free_ram_gib"] = free_ram_gib()
            res["turns"].append(st)
            request_entry["response"] = r
            request_entry["response_sha256"] = st["response_sha256"]
            request_entry["status"] = "complete"
            print(f"[{tag} {res['kind']}] turn {i}: {json.dumps(st)}", flush=True)
            if on_progress:
                on_progress()
            msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""})
        res["status"] = "complete"
    except BaseException as exc:
        res["status"] = "failed"
        res["error_type"] = type(exc).__name__
        res["error"] = repr(exc)
        if replay_record.get("requests") and replay_record["requests"][-1].get("status") == "pending":
            replay_record["requests"][-1].update({"status": "failed", "error_type": type(exc).__name__,
                                                  "error": repr(exc)})
        if on_progress:
            on_progress()
        raise
    finally:
        if proc is not None:
            stop(proc)
    return res


def atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ctx, ctx_meta = context_text(with_metadata=True)
    roots = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in A.server_roots if "=" in kv}
    provenance = {"neural_repo": repo_provenance(ROOT), "context_repo": repo_provenance(A.ctx_root),
                  "context": ctx_meta,
                  "gguf": {**file_provenance(A.gguf), "identity_strength": "metadata_only_no_large_file_hash"},
                  "llama_server": small_file_provenance(os.path.join(A.llama_dir, "llama-server.exe")),
                  "neural_server_roots": {},
                  "runner_artifacts": root_artifacts(ROOT, A.neural_args, NP.MODEL_DIR)}
    for kind in A.kinds:
        name, _, extra = kind.partition("=")
        root = roots.get(name, ROOT)
        provenance["neural_server_roots"][name] = {**repo_provenance(root),
            "artifacts": root_artifacts(root, A.neural_args + " " + extra, NP.MODEL_DIR)}
    out = {"config": vars(A), "provenance": provenance, "sessions": []}
    replay_source = None
    replay_source_meta = None
    external_replay_requests = None
    if A.replay_prompts:
        with open(A.replay_prompts, encoding="utf-8") as f:
            replay_source = json.load(f)
        external_replay_requests, replay_source_meta = replay_requests_from_archive(replay_source, A.replay_session)
    replay_path = os.path.abspath(A.out + ".replay.json")
    replay = {"schema": "neuralserver-bench-replay-v1", "source_result": os.path.abspath(A.out),
              "source_replay": os.path.abspath(A.replay_prompts) if A.replay_prompts else None,
              "source_session": A.replay_session if A.replay_prompts else None,
              "replay_mode": "external" if A.replay_prompts else
                             ("first-session" if A.replay_first_session else "none"),
              "first_session_source": replay_source_meta,
              "provenance": provenance, "sessions": []}
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)

    def save_progress():
        atomic_json(A.out, out)
        atomic_json(replay_path, replay)

    def first_session_replay():
        requests, metadata = replay_requests_from_archive(replay, 0)
        turns = out.get("sessions", [{}])[0].get("turns", [])
        metadata["response_sha256_by_turn"] = [r.get("response_sha256")
                                                for r in replay["sessions"][0].get("requests", [])]
        metadata["prompt_ids_sha256_by_turn"] = [t.get("prompt_ids_sha256") or
                                                   (t.get("neural") or {}).get("prompt_ids_sha256")
                                                   for t in turns]
        metadata["generated_ids_sha256_by_turn"] = [t.get("generated_ids_sha256") or
                                                     (t.get("neural") or {}).get("generated_ids_sha256")
                                                     for t in turns]
        return requests, metadata
    kinds = list(A.kinds)
    names = [k.partition("=")[0] for k in kinds]
    assert len(set(names)) == len(names), f"--kinds names must be unique: {names}"
    if A.schedule:
        bad = [n for n in A.schedule.split(",") if n.strip() not in names]
        assert not bad, f"--schedule has unknown names {bad}; known: {names}"
    if A.schedule:                     # explicit order, e.g. "llama,neural,old,llama,old,neural"
        spec = {k.partition("=")[0]: k for k in kinds}
        for i, name in enumerate(A.schedule.split(",")):
            kind = spec[name.strip()]
            rec = {"kind": name.strip(), "tag": f"s{i}", "requests": []}
            res = {"kind": name.strip(), "args": kind.partition("=")[2], "tag": f"s{i}", "turns": []}
            out["sessions"].append(res)
            replay["sessions"].append(rec)
            save_progress()
            use_requests = external_replay_requests
            if A.replay_first_session and i > 0:
                use_requests, replay_source_meta = first_session_replay()
                replay["first_session_source"] = replay_source_meta
                out["replay_source"] = replay_source_meta
                rec["replay_source"] = replay_source_meta
            session(kind, f"s{i}", ctx, use_requests, rec, res, save_progress)
            if A.replay_first_session and i == 0:
                _, replay_source_meta = first_session_replay()
                replay["first_session_source"] = replay_source_meta
                out["replay_source"] = replay_source_meta
                save_progress()
    else:
        session_index = 0
        for rd in range(A.rounds):
            for k in (kinds if rd % 2 == 0 else kinds[::-1]):       # ABBA order
                rec = {"kind": k.partition("=")[0], "tag": f"r{rd}", "requests": []}
                res = {"kind": k.partition("=")[0], "args": k.partition("=")[2], "tag": f"r{rd}", "turns": []}
                out["sessions"].append(res)
                replay["sessions"].append(rec)
                save_progress()
                use_requests = external_replay_requests
                if A.replay_first_session and session_index > 0:
                    use_requests, replay_source_meta = first_session_replay()
                    replay["first_session_source"] = replay_source_meta
                    out["replay_source"] = replay_source_meta
                    rec["replay_source"] = replay_source_meta
                session(k, f"r{rd}", ctx, use_requests, rec, res, save_progress)
                if A.replay_first_session and session_index == 0:
                    _, replay_source_meta = first_session_replay()
                    replay["first_session_source"] = replay_source_meta
                    out["replay_source"] = replay_source_meta
                    save_progress()
                session_index += 1
    summ = {}
    for k in [x.partition("=")[0] for x in kinds]:
        ss = [s for s in out["sessions"] if s["kind"] == k]
        first = [s["turns"][0]["decode_tok_s"] for s in ss if s["turns"]]
        fol = [t["decode_tok_s"] for s in ss for t in s["turns"][1:]]
        pr = [s["turns"][0]["prompt_s"] for s in ss if s["turns"]]
        def med(v):
            v = [x for x in v if x is not None]
            return statistics.median(v) if v else None
        summ[k] = {"prompt_s_median": med(pr),
                   "first_decode_median": med(first), "first_decode_all": first,
                   "followup_decode_median": med(fol), "followup_decode_all": fol,
                   "first_vram_hit_median": med([s["turns"][0].get("vram_hit") for s in ss if s["turns"]]),
                   "followup_vram_hit_median": med([t.get("vram_hit") for s in ss for t in s["turns"][1:]]),
                   "first_cpu_ms_median": med([s["turns"][0].get("cpu_ms_per_tok") for s in ss if s["turns"]]),
                   "followup_prompt_s_median": med([t["prompt_s"] for s in ss for t in s["turns"][1:]])}
    out["summary"] = summ
    save_progress()
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
