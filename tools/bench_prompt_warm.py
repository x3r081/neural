"""Warm prompt-processing speed: Neural vs llama-server, both with their weights already in RAM.

A server's FIRST prompt after start is not comparable: llama-server (mmap) loads its weights
from disk inside that prompt, while Neural warms RAM at startup, outside prompt timing. So each
session sends a discarded warm-up prompt first, then prompts with no shared prefix:
13,000 tokens of code (files in a different order) and 3,000 tokens (one file).

    python tools/bench_prompt_warm.py benchmarks/prompt_warm.json [--kinds llama neural] [--rounds 1]
"""
import argparse, json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.argv, _argv = [sys.argv[0], "unused.json"], sys.argv            # reuse bench_vs_llama's helpers/args
import bench_vs_llama as B                                           # noqa: E402
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--kinds", nargs="+", default=["llama", "neural"])
ap.add_argument("--rounds", type=int, default=1)
A = ap.parse_args()


def text(files, n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B.NP.MODEL_DIR, local_files_only=True)
    src = "\n\n".join(f"# ===== {f} =====\n" + open(os.path.join(ROOT, f), encoding="utf-8").read() for f in files)
    return tok.decode(tok(src, add_special_tokens=False).input_ids[:n])


PROMPTS = [
    ("warmup_13k", ("fused_core.py", "harmony_render.py", "cpu_prefill.py", "server.py"), 13000),
    ("code_13k", ("server.py", "cpu_prefill.py", "harmony_render.py", "fused_core.py"), 13000),
    ("file_3k", ("cpu_prefill.py", "harmony_render.py"), 3000),
]


def main():
    ctx = {name: text(files, n) for name, files, n in PROMPTS}
    out = {"sessions": []}
    for rd in range(A.rounds):
        for kind in (A.kinds if rd % 2 == 0 else A.kinds[::-1]):
            proc, base, ready = B.start(kind, f"pw{rd}")
            res = {"kind": kind, "round": rd, "prompts": []}
            try:
                res["startup_s"] = round(B.wait_ready(ready, proc), 1)
                for name, _, _ in PROMPTS:
                    body = {"model": "x", "max_tokens": 1, "temperature": 0, "reasoning_effort": "low",
                            "messages": [{"role": "user", "content": "Summarize this code:\n\n```python\n" + ctx[name] + "\n```"}]}
                    r, wall = B.post(base + "/v1/chat/completions", body)
                    u = r.get("usage") or {}
                    split = {}
                    if "neural" in r:
                        ps, cached = r["neural"]["prefill_s"], (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                        # host-time split of prompt processing (server.py PTIME): prefill_memcpy_s, prefill_ring_wait_s, ...
                        split = {k: v for k, v in r["neural"].items() if k.startswith("prefill_") and k != "prefill_s"}
                    else:
                        t = r.get("timings") or {}
                        ps, cached = (t["prompt_ms"] / 1e3) if t.get("prompt_ms") is not None else None, t.get("cache_n")
                    row = {"prompt": name, "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": cached,
                           "prompt_s": ps, "wall_s": round(wall, 2), **split}
                    res["prompts"].append(row)
                    print(f"[{kind} r{rd}] {json.dumps(row)}", flush=True)
            finally:
                B.stop(proc)
            out["sessions"].append(res)
            json.dump(out, open(A.out, "w"), indent=1)


main()
