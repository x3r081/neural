"""Calibrate the VRAM hot set on coding-agent traffic.

Requires the server started with --count-routing. Sends coding-agent requests (Continue-style
system prompt + tools; tasks deliberately DISJOINT from dev/bench_agent.py's snake/LRU/summary
tasks), then reads the routing counters and writes hotset_code.json in the same format as
hotset_freq.json: {"counts": [36][128]}.

counts = normalized(code decode counts) + 0.25 * normalized(general calibration counts)
(decode routing is what the VRAM set serves; the small general term keeps prose/reasoning
experts that coding sessions also use).

Usage: python calibrate_hotset.py [max_tokens]
"""
import json, os, sys, time
import httpx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = "http://127.0.0.1:8000"
MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 700
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c = httpx.Client(timeout=3600)

# reuse the Continue-style tools/system prompt definitions without running the benchmark
src = open(os.path.join(os.path.dirname(__file__), "bench_agent.py"), encoding="utf-8").read()
ns = {}
exec(src[src.index("def fn(name"):src.index("FS = {}")], ns)
TOOLS, SYSTEM = ns["TOOLS"], ns["SYSTEM"]

TASKS = [
    "Create a Python CLI tool rename_by_date.py that renames photos in a folder by EXIF date taken, with --dry-run.",
    "Write a React + TypeScript component DataTable.tsx: sortable, filterable table with pagination.",
    "Build a FastAPI todo service in app.py with SQLite persistence, plus tests in test_app.py.",
    "Make a Tetris game in a single file tetris.html using canvas and arrow keys.",
    "Write backup.sh: tar.gz a directory into a timestamped archive and prune archives older than 14 days.",
    "My Express server returns 404 for files in ./public. Create server.js that serves static files correctly and logs requests.",
    "Create schema.sql for a library (books, members, loans) with indexes, and queries.sql with 5 useful reports.",
    "Write csvparse.rs: a Rust function that parses a CSV line with quoted fields and escaped quotes, with unit tests.",
    "Create index.html and styles.css for a responsive SaaS landing page with hero, features and pricing cards.",
    "Write crawler.go: a concurrent web crawler with a worker pool, depth limit and same-host filter.",
    "Implement Dijkstra in dijkstra.cpp with a priority queue and a small main() demo graph.",
    "Add validation and error handling to a signup form: create form.js that validates email, password strength and shows inline errors.",
    "Write test_intervals.py: pytest tests for merge_intervals(intervals) including edge cases, and intervals.py implementing it.",
    "Create a Dockerfile and docker-compose.yml for a Flask app with Redis caching, plus the minimal app.py.",
    "Write latency_report.py: read a JSON-lines access log and print per-endpoint p50/p95/p99 latency tables.",
    "Create a Kotlin data class and repository for a notes app with an in-memory store and unit tests.",
]

t0 = time.perf_counter()
for i, task in enumerate(TASKS):
    body = {"model": "gpt-oss-120b-neural", "temperature": 0.7, "max_tokens": MAXTOK, "tools": TOOLS,
            "reasoning_effort": "low" if i % 2 else "medium",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]}
    r = c.post(f"{BASE}/v1/chat/completions", json=body, headers={"Authorization": "Bearer local"})
    j = r.json()
    n = j.get("neural", {})
    tc = (j["choices"][0]["message"].get("tool_calls") or [{}])[0].get("function", {}).get("name")
    print(f"[{i + 1}/{len(TASKS)}] {j['usage']['completion_tokens']} tok  {n.get('decode_tok_s', 0):.1f} tok/s  "
          f"hit {n.get('vram_hit', 0):.2f}  tool={tc}  ({time.perf_counter() - t0:.0f}s)", flush=True)

rt = c.get(f"{BASE}/neural/routing").json()
dec = np.array(rt["decode_counts"], dtype=np.float64)
pre = np.array(rt["prefill_counts"], dtype=np.float64)
gen = np.array(json.load(open(os.path.join(HERE, "hotset_freq.json")))["counts"], dtype=np.float64)
counts = dec / max(dec.sum(), 1) + 0.25 * gen / max(gen.sum(), 1)
out = {"counts": (counts * 1e9).round().astype(np.int64).tolist(), "source": "coding-agent calibration",
       "decode_tokens": rt["decode_tokens"], "tasks": len(TASKS), "max_tokens": MAXTOK,
       "mix": "code_decode_norm + 0.25 * general_norm", "raw_decode_counts": rt["decode_counts"],
       "raw_prefill_counts": rt["prefill_counts"]}
json.dump(out, open(os.path.join(HERE, "hotset_code.json"), "w"))
# how different is the code set from the general one?
k = 414
top_code = set(np.argsort(-counts.ravel())[:k].tolist())
top_gen = set(np.argsort(-gen.ravel())[:k].tolist())
share_dec_code = float(np.sort(dec.ravel())[::-1][:k].sum() / max(dec.sum(), 1))
share_dec_gen = float(dec.ravel()[list(top_gen)].sum() / max(dec.sum(), 1))
print(f"decode tokens counted: {rt['decode_tokens']}; top-{k} overlap code vs general: {len(top_code & top_gen)}/{k}")
print(f"share of coding decode routing covered by top-{k}: code set {share_dec_code:.3f} vs general set {share_dec_gen:.3f}")
print("wrote hotset_code.json")
