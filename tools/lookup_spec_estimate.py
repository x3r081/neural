"""Offline estimate of prompt-lookup speculative decoding for this server (result: no gain).

Generates two tasks on a RUNNING server (a full-file rewrite like Continue apply requests, and new
code), then replays n-gram prompt-lookup drafting against the real output tokens. Verify cost is
the MEASURED cost of processing n new tokens through the prompt path on the dev machine.
    python tools/lookup_spec_estimate.py
"""
import json, os, urllib.request, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as NP
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(NP.MODEL_DIR)
URL = "http://127.0.0.1:8000/v1/chat/completions"
src = open(os.path.join(NP.REPO, "chat_client.py"), encoding="utf-8").read()
TASKS = {
 "apply_rewrite": ("Here is a Python file. Add type hints to the function `stream_chat` and change the default "
                   "--max-tokens to 8192. Output the complete updated file in one code block, nothing else.\n\n```python\n" + src + "\n```"),
 "new_code": ("Write a Python module implementing an LRU cache class with get/put, a TTL option and thread safety, "
              "plus pytest tests. Output only code."),
}
# MEASURED verify cost (ms) for n new tokens through the prompt path (this server, real code tokens)
def verify_ms(n):
    pts = [(5, 236), (7, 279), (8, 323), (12, 387), (15, 472), (26, 625)]
    if n <= pts[0][0]: return pts[0][1]
    for (a, ca), (b, cb) in zip(pts, pts[1:]):
        if n <= b: return ca + (cb - ca) * (n - a) / (b - a)
    return pts[-1][1] + 12 * (n - pts[-1][0])
res = {}
for name, prompt in TASKS.items():
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000,
                       "temperature": 0, "reasoning_effort": "low"}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}), timeout=3600))
    msg = r["choices"][0]["message"]; out = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    dec_ms = 1000 / r["neural"]["decode_tok_s"]
    P = tok.encode(prompt); O = tok.encode(out)
    print(f"\n== {name}: prompt {len(P)} tok, output {len(O)} tok, decode {r['neural']['decode_tok_s']:.1f} tok/s ({dec_ms:.0f} ms/tok)")
    for N in (3, 2):
        for D in (4, 8, 16):
            i = 0; t_ms = 0.0; steps = drafted = accepted = 0
            ctx = P[:]
            while i < len(O):
                cand = []
                hist = ctx + O[:i]
                key = hist[-N:]
                if len(key) == N:
                    for j in range(len(hist) - N - 1, -1, -1):            # most recent earlier occurrence
                        if hist[j:j + N] == key:
                            cand = hist[j + N:j + N + D]; break
                if cand:
                    k = 0
                    while k < len(cand) and i + k < len(O) and cand[k] == O[i + k]: k += 1
                    t_ms += verify_ms(len(cand) + 1); steps += 1; drafted += len(cand); accepted += k
                    i += k + 1                                             # accepted drafts + 1 model token
                else:
                    t_ms += dec_ms; i += 1
            base = len(O) * dec_ms
            print(f"  ngram {N} draft<={D:2d}: verify steps {steps:4d}, accept {accepted}/{drafted} "
                  f"({(accepted / max(drafted, 1)):.2f}), est. speedup {base / t_ms:.2f}x")
