"""FROZEN copy of server.py's prompt rendering as of 2026-09-23 (before exact
token splicing), used by dev/test_splice.py as the regression oracle.

The three functions below (_text, _defang, render) are copied VERBATIM from
server.py lines 993-1072 (extracted with sed, not retyped). Only the globals they
read are provided here: tok, SPECIAL_TEXT (built exactly as server.py lines
720-721 build it) and A.identity. Call setup(tok, identity) first.
"""
import json


class _Args:
    identity = None


A = _Args()
tok = None
SPECIAL_TEXT = []


def setup(tokenizer, identity=None):
    global tok, SPECIAL_TEXT
    tok = tokenizer
    A.identity = identity
    SPECIAL_TEXT = sorted({t for t in tok.all_special_tokens} | {t for t in tok.get_added_vocab()}, key=len, reverse=True)
    SPECIAL_TEXT = [t for t in SPECIAL_TEXT if t.startswith("<|") and t.endswith("|>")]


# ---------------------------------------------------------------- verbatim from server.py
def _text(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return str(c)


def _defang(s):
    """Stop text that spells a control token (e.g. a file containing '<|end|>') from
    being tokenized as that control token."""
    for sp in SPECIAL_TEXT:
        if sp in s:
            s = s.replace(sp, "<\u200b" + sp[1:])
    return s


def render(messages, tools, effort):
    sys_parts, conv = [], []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            sys_parts.append(_defang(_text(m.get("content"))))
        elif role == "user":
            conv.append({"role": "user", "content": _defang(_text(m.get("content")))})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            think = _defang(_text(m.get("reasoning_content") or m.get("reasoning") or m.get("thinking")))
            content = _defang(_text(m.get("content")))
            if tcs:
                for i, tc in enumerate(tcs):              # the template renders ONE call per message
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except ValueError:
                        args = {"_raw": args}
                    msg = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": fn.get("name"), "arguments": args}}]}
                    th = (think or content) if i == 0 else ""
                    if th:
                        msg["thinking"] = th
                    conv.append(("assistant_call", msg, tc.get("id")))
            else:
                conv.append({"role": "assistant", "content": content})
        elif role == "tool":
            conv.append(("tool", _defang(_text(m.get("content"))), m.get("tool_call_id")))
    # re-order so every tool result directly follows the call it answers
    out, pending_results = [], {}
    for item in conv:
        if isinstance(item, tuple) and item[0] == "tool":
            pending_results.setdefault(item[2], []).append(item[1])
    used = set()
    for item in conv:
        if isinstance(item, tuple) and item[0] == "assistant_call":
            out.append(item[1])
            res = pending_results.get(item[2])
            if res:
                out.extend({"role": "tool", "content": r} for r in res)
                used.add(item[2])
        elif isinstance(item, tuple) and item[0] == "tool":
            if item[2] not in used and item[2] not in {c[2] for c in conv if isinstance(c, tuple) and c[0] == "assistant_call"}:
                out.append({"role": "tool", "content": item[1]})
        else:
            out.append(item)
    raws = {}
    for i, m in enumerate(out):
        if m["role"] == "tool":
            key = f"NEURALTOOLRESULT{i}X"
            raws[key] = m["content"]
            m["content"] = key
    msgs = ([{"role": "developer", "content": "\n\n".join(p for p in sys_parts if p)}] if any(sys_parts) else []) + out
    kw = {"reasoning_effort": effort}
    if A.identity:
        kw["model_identity"] = A.identity
    text = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False, **kw)
    for key, raw in raws.items():                       # tool output raw (harmony), not JSON-quoted
        text = text.replace(json.dumps(key), raw, 1)
    return tok(text, add_special_tokens=False).input_ids
