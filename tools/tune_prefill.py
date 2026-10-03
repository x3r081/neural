import json, sys, time, httpx
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = "http://127.0.0.1:8000"
c = httpx.Client(timeout=3600)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths as NP  # noqa: E402
CODE = open(NP.SAMPLE_CODE, encoding="utf-8").read()
res = []
for i, cpm in enumerate([16, 32, 64, 128, 16, 32, 64, 128]):
    c.post(f"{B}/neural/config", json={"cpu_prefill_max": cpm, "idle_canon": 0})
    msgs = [{"role": "system", "content": f"[tuning run {i}] You are a coding agent."},
            {"role": "user", "content": "File:\n```python\n" + CODE[:34000] + "\n```\nOne-line summary."}]
    row = {"cpm": cpm}
    for label, extra in (("cold", None), ("+300", CODE[60000:61100]), ("+1500", CODE[62000:67400])):
        if extra:
            msgs = msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "Tool output:\n" + extra + "\nAck."}]
        j = c.post(f"{B}/v1/chat/completions", json={"model": "x", "messages": msgs, "max_tokens": 4, "temperature": 0,
                                                     "reasoning_effort": "low"}).json()
        n = j["neural"]
        row[label] = (n["prompt_tokens"] - n["cached_tokens"], round(n["prefill_s"], 2))
    res.append(row)
    print(json.dumps(row), flush=True)
json.dump(res, open(_os.path.join(NP.REPO, "benchmarks", "tune_prefill_latest.json"), "w"), indent=1)
