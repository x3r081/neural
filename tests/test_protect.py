"""Functional checks on a running server:
 1. side-request protection: conversation -> unrelated short request -> conversation continues;
    the continuation must reuse (almost) the whole conversation prefix.
 2. aborted stream: disconnect mid-generation -> the server must not run the idle job for it,
    and the next continuation must still reuse the cache.
 3. error path: an invalid request type is answered with JSON, not a dropped connection.
"""
import json, sys, time
import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = "http://127.0.0.1:8000/v1"
c = httpx.Client(timeout=900)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths as NP  # noqa: E402
CODE = open(NP.SAMPLE_CODE, encoding="utf-8").read()


def chat(msgs, **kw):
    j = c.post(f"{B}/chat/completions", json=dict({"model": "x", "messages": msgs, "temperature": 0,
                                                   "reasoning_effort": "low", "max_tokens": 40}, **kw)).json()
    return j, j.get("usage", {}), j.get("neural", {})


ok = True
conv = [{"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "File:\n```python\n" + CODE[:9000] + "\n```\nName one class in it."}]
j, u, n = chat(conv)
conv.append({"role": "assistant", "content": j["choices"][0]["message"]["content"] or "ok"})
time.sleep(8)                                        # let the idle job run (natural final answer)
print("1a conversation:", u["prompt_tokens"], "prompt tokens")
# unrelated side request (like an 'apply' or title request)
j2, u2, n2 = chat([{"role": "system", "content": "Generate a short title."}, {"role": "user", "content": "Title for: snake game"}])
print("1b side request: protected =", n2.get("side_protected"), "| cached", u2["prompt_tokens_details"]["cached_tokens"])
conv.append({"role": "user", "content": "And one function name?"})
j3, u3, n3 = chat(conv)
reuse = u3["prompt_tokens_details"]["cached_tokens"] / u3["prompt_tokens"]
print(f"1c continuation: cached {u3['prompt_tokens_details']['cached_tokens']}/{u3['prompt_tokens']} ({reuse:.0%}), "
      f"prefill {n3.get('prefill_s', 0):.2f}s")
ok &= bool(n2.get("side_protected")) and reuse > 0.9
conv.append({"role": "assistant", "content": j3["choices"][0]["message"]["content"] or "ok"})
time.sleep(8)

# 2. aborted stream
conv2 = conv + [{"role": "user", "content": "Write a long essay about this code."}]
with c.stream("POST", f"{B}/chat/completions", json={"model": "x", "messages": conv2, "stream": True,
                                                     "temperature": 0, "reasoning_effort": "low", "max_tokens": 600}) as r:
    got = 0
    for line in r.iter_lines():
        if line.startswith("data: "):
            got += 1
            if got > 25:
                break                                  # hang up mid-generation
time.sleep(10)
conv3 = conv + [{"role": "user", "content": "Is it thread-safe? One word."}]
j4, u4, n4 = chat(conv3)
reuse4 = u4["prompt_tokens_details"]["cached_tokens"] / u4["prompt_tokens"]
print(f"2  after an aborted stream: cached {u4['prompt_tokens_details']['cached_tokens']}/{u4['prompt_tokens']} ({reuse4:.0%})")
ok &= reuse4 > 0.9

# 3. error path
r5 = c.post(f"{B}/chat/completions", json={"model": "x", "messages": "not a list"})
print("3  bad request ->", r5.status_code, r5.text[:120])
ok &= r5.status_code in (400, 500) and r5.headers.get("content-type", "").startswith("application/json")
print("RESULT:", "PASS" if ok else "FAIL")

# 4. new chat taking over: first message looks like a side request (protected), the second
#    message of that new chat must still reuse its first turn (swapped back in from host RAM)
new = [{"role": "system", "content": "You are a different assistant for a new chat."},
       {"role": "user", "content": "Here is some context:\n" + CODE[20000:27000] + "\nSummarize in one line."}]
j6, u6, n6 = chat(new)
new.append({"role": "assistant", "content": j6["choices"][0]["message"]["content"] or "ok"})
new.append({"role": "user", "content": "Now name one function in it."})
j7, u7, n7 = chat(new)
reuse7 = u7["prompt_tokens_details"]["cached_tokens"] / u7["prompt_tokens"]
print(f"4  new chat: first protected={n6.get('side_protected')}, second message cached "
      f"{u7['prompt_tokens_details']['cached_tokens']}/{u7['prompt_tokens']} ({reuse7:.0%}), prefill {n7.get('prefill_s', 0):.2f}s")
print("RESULT (with 4):", "PASS" if ok and reuse7 > 0.8 else "FAIL")
