"""Agent-workload benchmark for the Neural server (OpenAI chat-completions over HTTP).

  1. decode   : fixed-length generation (ignore_eos) at ~0.3k / ~4k / ~10k tokens of context
  2. prefill  : cold prompt, then agent-step increments (+50 / +300 / +1500 new tokens) on a cached prefix
  3. scenario : Continue-style agent session (Continue's 13 tool names, a coding-agent system prompt),
                "build a snake game" -> tool calls against a virtual workspace -> follow-up edit request.
                Temperature 0. Wall time per request and total.

Usage: python bench_agent.py <out.json> [base_url]
"""
import json, re, sys, time
import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = sys.argv[1] if len(sys.argv) > 1 else "bench.json"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/v1"
c = httpx.Client(timeout=3600)
R = {"decode": [], "prefill": [], "scenario": []}
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths as NP  # noqa: E402
CODE = open(NP.SAMPLE_CODE, encoding="utf-8").read()


def chat(messages, **kw):
    body = {"model": "gpt-oss-120b-neural", "messages": messages, "temperature": 0}
    body.update(kw)
    t0 = time.perf_counter()
    r = c.post(f"{BASE}/chat/completions", json=body, headers={"Authorization": "Bearer local"})
    r.raise_for_status()
    j = r.json()
    j["_wall"] = time.perf_counter() - t0
    return j


def stats(j):
    n = j.get("neural", {})
    return {"wall_s": round(j["_wall"], 2), "prompt": j["usage"]["prompt_tokens"],
            "cached": j["usage"]["prompt_tokens_details"]["cached_tokens"], "completion": j["usage"]["completion_tokens"],
            "prefill_s": n.get("prefill_s"), "decode_tok_s": n.get("decode_tok_s"), "vram_hit": n.get("vram_hit"),
            "cpu_GBps": n.get("cpu_GBps"), "cpu_ms_per_tok": n.get("cpu_ms_per_tok"),
            "gpu_wait_ms_per_tok": n.get("gpu_wait_ms_per_tok"), "finish": j["choices"][0]["finish_reason"]}


# ------------------------------------------------------------------ 1. decode at context sizes
TASK = "Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL expiry, plus unit tests."
for label, nchars in (("ctx~0.3k", 0), ("ctx~4k", 14000), ("ctx~10k", 36000)):
    ctx = ("Reference code from the repository:\n```python\n" + CODE[:nchars] + "\n```\n\n") if nchars else ""
    j = chat([{"role": "system", "content": f"You are a senior engineer. [{label}]"},
              {"role": "user", "content": ctx + TASK}],
             max_tokens=384, ignore_eos=True, reasoning_effort="low")
    s = stats(j)
    s["label"] = label
    R["decode"].append(s)
    print("decode", json.dumps(s), flush=True)

# ------------------------------------------------------------------ 2. prefill: cold + agent-step increments
base_msgs = [{"role": "system", "content": "You are a coding agent. [prefill probe]"},
             {"role": "user", "content": "Here is a file:\n```python\n" + CODE[40000:54000] + "\n```\nSummarize it in one line."}]
j = chat(base_msgs, max_tokens=24, reasoning_effort="low")
s = stats(j); s["label"] = "cold ~4k"; R["prefill"].append(s); print("prefill", json.dumps(s), flush=True)
msgs = base_msgs + [{"role": "assistant", "content": j["choices"][0]["message"]["content"] or "ok"}]
for label, nchars in (("+~50 tok", 180), ("+~300 tok", 1100), ("+~1500 tok", 5400)):
    msgs = msgs + [{"role": "user", "content": "Tool output:\n" + CODE[60000:60000 + nchars] + "\nAcknowledge in one word."}]
    j = chat(msgs, max_tokens=16, reasoning_effort="low")
    s = stats(j); s["label"] = label; R["prefill"].append(s); print("prefill", json.dumps(s), flush=True)
    msgs = msgs + [{"role": "assistant", "content": j["choices"][0]["message"]["content"] or "ok"}]

# ------------------------------------------------------------------ 3. Continue-style agent scenario
def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


S_ = {"type": "string"}
TOOLS = [fn("read_file", "Use this tool if you need to view the contents of an existing file.", {"filepath": dict(S_, description="The path of the file to read, relative to the root of the workspace")}, ["filepath"]),
         fn("create_new_file", "Create a new file. Only use this when a file doesn't exist and should be created", {"filepath": dict(S_, description="The path where the new file should be created, relative to the root of the workspace"), "contents": dict(S_, description="The contents to write to the new file")}, ["filepath", "contents"]),
         fn("run_terminal_command", "Run a terminal command in the current directory. The shell is not stateful.", {"command": dict(S_, description="The command to run"), "waitForCompletion": {"type": "boolean", "description": "Whether to wait for the command to complete"}}, ["command"]),
         fn("file_glob_search", "Search for files recursively in the project using glob patterns.", {"pattern": dict(S_, description="Glob pattern for file path matching")}, ["pattern"]),
         fn("view_diff", "View the current diff of working changes", {}, []),
         fn("read_currently_open_file", "Read the currently open file in the IDE.", {}, []),
         fn("ls", "List files and folders in a given directory", {"dirPath": dict(S_, description="The directory path relative to the root"), "recursive": {"type": "boolean"}}, []),
         fn("fetch_url_content", "Can be used to view the contents of a website using a URL.", {"url": dict(S_, description="The URL to read")}, ["url"]),
         fn("read_skill", "Read a skill's instructions.", {"skillName": S_}, ["skillName"]),
         fn("search_web", "Performs a web search, returning top results.", {"query": dict(S_, description="The natural language search query")}, ["query"]),
         fn("edit_existing_file", "Use this tool to edit an existing file. If you don't know the contents of the file, read it first. When addressing code modification requests, present a concise code snippet that emphasizes only the necessary changes and uses abbreviated placeholders for unmodified sections.", {"filepath": dict(S_, description="The path of the file to edit"), "changes": dict(S_, description="Any modifications to the file, showing only needed changes.")}, ["filepath", "changes"]),
         fn("single_find_and_replace", "Performs exact string replacement in a file.", {"filepath": S_, "old_string": S_, "new_string": S_, "replace_all": {"type": "boolean"}}, ["filepath", "old_string", "new_string"]),
         fn("grep_search", "Perform a search over the repository using ripgrep.", {"query": dict(S_, description="The search query to use.")}, ["query"])]
SYSTEM = ("<important_rules>\n  You are in agent mode.\n\n  If you need to use multiple tools, you can call multiple read only tools simultaneously.\n\n"
          "  Always include the language and file name in the info string when you write code blocks.\n"
          "  If you are editing \"src/main.py\" for example, your code block should start with '```python src/main.py'\n\n"
          "  For larger codeblocks (>20 lines), use brief language-appropriate placeholders for unmodified sections.\n"
          "  Use tools to read, create and edit files in the workspace. Do not just describe code - write it to files.\n"
          "</important_rules>")
FS = {}


def run_tool(name, args):
    p = args.get("filepath") or args.get("dirPath") or ""
    if name == "create_new_file":
        FS[p] = args.get("contents", "")
        return f"File {p} created successfully"
    if name == "read_file" or name == "read_currently_open_file":
        return FS.get(p or (list(FS)[-1] if FS else ""), f"File {p} does not exist")
    if name == "edit_existing_file":
        FS[p] = FS.get(p, "") + "\n" + args.get("changes", "")
        return f"Edits applied to {p}"
    if name == "single_find_and_replace":
        FS[p] = FS.get(p, "").replace(args.get("old_string", ""), args.get("new_string", ""))
        return f"Replacement applied to {p}"
    if name in ("ls", "file_glob_search"):
        return "\n".join(FS) or "(empty)"
    if name == "run_terminal_command":
        return "Command completed with exit code 0 (no output)"
    return "(no results)"


def agent_turn(msgs, max_steps=6):
    steps = []
    for _ in range(max_steps):
        j = chat(msgs, tools=TOOLS, max_tokens=6000)
        s = stats(j)
        m = j["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        s["tool"] = tcs[0]["function"]["name"] if tcs else None
        steps.append(s)
        print("scenario", json.dumps(s), flush=True)
        msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": tcs} if tcs else {"role": "assistant", "content": m.get("content") or ""})
        if not tcs:
            break
        try:
            args = json.loads(tcs[0]["function"]["arguments"] or "{}")
        except ValueError:
            args = {}
        msgs.append({"role": "tool", "tool_call_id": tcs[0]["id"], "content": run_tool(tcs[0]["function"]["name"], args)})
    return steps


t_scn = time.perf_counter()
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Build me a snake game in index.html that I can run through WASD and play to test."}]
R["scenario"].append({"turn": "snake", "steps": agent_turn(msgs)})
msgs.append({"role": "user", "content": "Add a score display and make the snake speed up as the score increases."})
R["scenario"].append({"turn": "score+speed", "steps": agent_turn(msgs)})
R["scenario_total_wall_s"] = round(time.perf_counter() - t_scn, 1)
R["scenario_generated_tokens"] = sum(s["completion"] for t in R["scenario"] for s in t["steps"])
R["files"] = {k: len(v) for k, v in FS.items()}
print("SCENARIO total wall", R["scenario_total_wall_s"], "s | generated", R["scenario_generated_tokens"], "tokens | files", R["files"], flush=True)
json.dump(R, open(OUT, "w"), indent=1)
