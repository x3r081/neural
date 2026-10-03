"""Sequential GPT-OSS launcher parity/performance audit using the same checkpoint.

No llama or Qwen jobs. Owns and stops only the server processes it creates.
All request bodies, token hashes, timings and launch arguments are archived.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_runtime.__main__ import parser, model_paths, native_plan, native_arguments
from neural_runtime.model_spec import inspect_model


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stop_owned(proc):
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for item in children[::-1]:
            try: item.terminate()
            except psutil.NoSuchProcess: pass
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=20)
        for item in alive:
            item.kill()
        psutil.wait_procs(alive, timeout=10)
    except psutil.NoSuchProcess:
        pass
    proc.wait(timeout=30)


def request(base, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
          headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("NEURAL_API_KEY", "local")})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return {"request_sha256": digest(body), "wall_s": time.perf_counter()-start,
            "usage": result["usage"], "neural": result.get("neural"), "choices": result["choices"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmarks/gptoss_agnostic_20261003/native_abba.json")
    ap.add_argument("--schedule", default="direct,launcher,launcher,direct")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--native-manifest")
    ap.add_argument("--candidate-warm-method", choices=("kernel", "pages", "pages-parallel"))
    ap.add_argument("--replay", help="prior audit file whose exact request bodies should be replayed")
    a = ap.parse_args()
    out = (ROOT / a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = ["serve", "--port", str(a.port), "--pool-gib", "6.05"]
    if a.native_manifest:
        flags += ["--native-manifest", a.native_manifest]
    # Frozen placement: arithmetic is sensitive to CPU/GPU reduction order.
    flags += ["--", "--refresh-every", "0", "--prefill-m", "0"]
    args = parser().parse_args(flags)
    model, store = model_paths(args)
    spec = inspect_model(model, store).to_dict()
    hw, plan = native_plan(args, spec)
    if not plan["supported"]:
        raise RuntimeError(plan)
    code = (ROOT / "cpu_prefill.py").read_text(encoding="utf-8")[:9000]
    bodies = [{"model": "gpt-oss-120b-neural", "messages": [
        {"role": "system", "content": "You are a careful coding assistant. Explain your reasoning and give concrete code."},
        {"role": "user", "content": "Review this CPU expert wrapper and propose a safe boundary-condition test:\n" + code}],
        "temperature": 0, "max_tokens": a.max_tokens, "stream": False}]
    # Repeating exact prompt exercises the production KV/prompt rollback path.
    bodies.append(json.loads(json.dumps(bodies[0])))
    bodies.append({"model": "gpt-oss-120b-neural", "messages": [
        {"role": "user", "content": "Write a Python function to merge overlapping half-open integer intervals. Explain edge cases and give three tests."}],
        "temperature": 0, "max_tokens": a.max_tokens, "stream": False})
    if a.replay:
        bodies = json.loads(Path(a.replay).read_text(encoding="utf-8"))["requests"]
    report = {"evidence": "MEASURED", "status": "running", "scope": "frozen GPT-OSS production-path launcher parity",
              "requests": bodies, "model": spec, "hardware": hw, "plan": plan, "sessions": [],
              "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ["server.py", "fused_core.py", "cpu_prefill.py", "neural_runtime/__main__.py"]}}
    def save(): out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    env = os.environ.copy()
    env.update(NEURAL_PREFILL_GROUP="0", NEURAL_PREFILL_GROUP_EXPERTS="3", NEURAL_PREFILL_GROUP_ROWS="1024",
               PYTHONUNBUFFERED="1")
    base = f"http://127.0.0.1:{a.port}"
    try:
        for i, kind in enumerate(a.schedule.split(",")):
            if kind not in {"direct", "launcher", "portable"}:
                raise ValueError("schedule must contain direct/launcher/portable only")
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", a.port)) == 0:
                    raise RuntimeError("benchmark port already in use")
            selected_flags = list(flags)
            if kind == "portable":
                selected_flags[1:1] = ["--native-manifest", str(ROOT / "artifacts/portable_cpu_build.json")]
            if a.candidate_warm_method and kind != "direct":
                selected_flags += ["--warm-method", a.candidate_warm_method]
            cmd = ([sys.executable] + native_arguments(args, spec, plan) + ["--warm-method", "kernel"] if kind == "direct" else
                   [sys.executable, "-m", "neural_runtime", *selected_flags])
            logpath = out.with_name(out.stem + f"_{i}_{kind}.log")
            rec = {"kind": kind, "command": cmd, "log": str(logpath), "status": "starting", "turns": [],
                   "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                      for name in report["source_hashes"]}}
            report["sessions"].append(rec); save()
            print(f"Starting session {i+1}: {kind}", flush=True)
            with logpath.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                rec["pid"] = proc.pid
                try:
                    start = time.monotonic()
                    while time.monotonic()-start < 900:
                        if proc.poll() is not None:
                            raise RuntimeError(f"server exited {proc.returncode}; see {logpath}")
                        try:
                            with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
                                if response.status == 200: break
                        except OSError: pass
                        time.sleep(2)
                    else: raise TimeoutError("server readiness timeout")
                    rec["startup_s"] = time.monotonic()-start
                    rec["status"] = "ready"; save()
                    for j, body in enumerate(bodies):
                        result = request(base, body)
                        rec["turns"].append(result); save()
                        if i:
                            baseline = report["sessions"][0]["turns"][j]
                            for field in ("prompt_ids_sha256", "generated_ids_sha256"):
                                if (not result["neural"].get(field) or
                                        result["neural"][field] != baseline["neural"].get(field)):
                                    raise AssertionError(f"session {i+1} turn {j+1}: {field} differs; stopping audit")
                        print(f"Session {i+1} turn {j+1}: {result['neural']['decode_tok_s']:.3f} tok/s, "
                              f"{result['usage']['completion_tokens']} tokens", flush=True)
                    rec["status"] = "complete"
                finally:
                    stop_owned(proc)
                    rec["server_stopped"] = True
                    save()
            time.sleep(3)
        hashes = []
        for j in range(len(bodies)):
            turns = [s["turns"][j] for s in report["sessions"]]
            hashes.append({field: len({t["neural"].get(field) for t in turns}) == 1
                           and all(t["neural"].get(field) for t in turns)
                           for field in ("prompt_ids_sha256", "generated_ids_sha256")})
        report["token_identity"] = hashes
        report["status"] = "PASS" if all(all(x.values()) for x in hashes) else "FAIL"
        for kind in {s["kind"] for s in report["sessions"]}:
            turns = [t for s in report["sessions"] if s["kind"] == kind for t in s["turns"]]
            seconds = sum(t["neural"]["decode_s"] for t in turns)
            tokens = sum(max(0,t["usage"]["completion_tokens"]-1) for t in turns)
            report.setdefault("aggregate", {})[kind] = {"decode_tokens": tokens, "decode_s": seconds,
                                                      "decode_tok_s": tokens/seconds}
        save()
        print(json.dumps({"status": report["status"], "aggregate": report["aggregate"]}), flush=True)
        return 0 if report["status"] == "PASS" else 1
    except BaseException as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}"); save()
        raise

if __name__ == "__main__":
    raise SystemExit(main())
