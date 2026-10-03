"""Terminal chat against the running Neural server (standard library only).

    python chat_client.py [--url http://127.0.0.1:8000/v1] [--effort low|medium|high]

Commands: /reset  /effort low|medium|high  /quit      Ctrl+C stops an answer.
"""
import argparse, json, os, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
ap.add_argument("--max-tokens", type=int, default=4096)
A = ap.parse_args()
os.system("")                                           # ANSI colours on Windows consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIM, BOLD, CYAN, RST = "\x1b[2m", "\x1b[1m", "\x1b[36m", "\x1b[0m"
KEY = os.environ.get("NEURAL_API_KEY", "local")


def stream_chat(messages, effort):
    body = json.dumps({"model": "gpt-oss-120b-neural", "messages": messages, "stream": True,
                       "stream_options": {"include_usage": True}, "reasoning_effort": effort,
                       "max_tokens": A.max_tokens}).encode()
    req = urllib.request.Request(A.url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    answer, mode, stats, t0 = [], None, {}, time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])
            if ch.get("error"):
                print(f"\n[server error: {ch['error'].get('message')}]")
            if ch.get("usage"):
                stats = dict(ch["usage"], **(ch.get("neural") or {}))
            for c in ch.get("choices", []):
                d = c.get("delta", {})
                if d.get("reasoning_content"):
                    if mode != "r":
                        sys.stdout.write(DIM + "[thinking] ")
                        mode = "r"
                    sys.stdout.write(d["reasoning_content"])
                if d.get("content"):
                    if mode != "a":
                        sys.stdout.write(RST + ("\n" if mode else "") + BOLD + "answer: " + RST)
                        mode = "a"
                    sys.stdout.write(d["content"])
                    answer.append(d["content"])
                sys.stdout.flush()
    print(RST)
    if stats:
        print(DIM + f"[{stats.get('completion_tokens')} tokens | {stats.get('decode_tok_s', 0):.1f} tok/s | "
              f"prompt {stats.get('prompt_tokens')} ({stats.get('prompt_tokens_details', {}).get('cached_tokens', 0)} cached) "
              f"processed in {stats.get('prefill_s', 0):.1f} s | {time.perf_counter() - t0:.1f} s total]" + RST)
    return "".join(answer)


def main():
    effort, messages = A.effort, []
    print(BOLD + "Neural chat" + RST + f" -> {A.url}  (reasoning effort {effort}; /reset /effort /quit)")
    while True:
        try:
            q = input("\n" + CYAN + BOLD + "you> " + RST).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in ("/quit", "/exit"):
            return
        if q == "/reset":
            messages = []
            print(DIM + "(conversation cleared)" + RST)
            continue
        if q.startswith("/effort"):
            v = q.split()[-1]
            effort = v if v in ("low", "medium", "high") else effort
            print(DIM + f"(reasoning effort: {effort})" + RST)
            continue
        messages.append({"role": "user", "content": q})
        try:
            messages.append({"role": "assistant", "content": stream_chat(messages, effort) or "(no answer)"})
        except KeyboardInterrupt:
            print(RST + DIM + "\n(stopped)" + RST)
            messages.pop()
        except OSError as e:
            print(f"cannot reach the server at {A.url} ({e}). Start it with start_neural_server.bat")
            messages.pop()


main()
