"""Harmony prompt rendering for the gpt-oss OpenAI-compatible server (pure: no model,
no torch; every function takes the tokenizer as a parameter).

render() turns OpenAI chat messages into prompt token ids exactly as server.py always
has (HF chat template, _defang on every text field, tool results rendered raw, one
tool call per assistant message) with one addition: EXACT TOKEN SPLICING of the
model's own tool-call turns.

Why. When the model calls a tool it generates

    <|channel|>analysis<|message|>REASONING<|end|><|start|>assistant<|channel|>commentary
    to=functions.NAME <|constrain|>json<|message|>ARGS<|call|>

but the harness echoes back only the call (no reasoning), and the template re-renders
it as "<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>
ARGS_TOJSON<|call|>". The model loses its reasoning across consecutive tool calls
(harmony keeps the analysis of tool-call turns until the next final answer; so does
the HF template when a 'thinking' field is given) and the prefix KV cache stops
matching at that turn, so every agent step re-processes the previous assistant turn.

Splicing. The server remembers, per returned tool_call id, the exact generated token
ids of that response (first token after the prompt through <|call|>), in a bounded
LRU (SpliceMemory: entry count AND total-token budget, 4 bytes per token). An
assistant message with tool_calls is SPLICEABLE when
  * its FIRST tool_call id is remembered,
  * the remembered call has the same function name and the same arguments: string-
    equal, or TYPE-STRICT equal after JSON parsing (key order / whitespace / escapes
    may differ; true != 1, 3 != 3.0, duplicate keys never match) - i.e. the client
    did not edit the call, and
  * policy after_final="drop" (default, harmony-canonical): no assistant message
    WITHOUT tool_calls follows it anywhere later in the conversation (the template's
    future_final_message rule: once a final answer exists, harmony drops the analysis
    of earlier tool-call turns). after_final="keep" (opt-in) splices those too.
A spliceable message renders as tokens("<|start|>assistant") + remembered tokens in
place of the template's rendering of its first call (analysis + call block). With
several tool_calls in one message (the server itself only ever returns one) the
remaining calls render exactly as today, each followed by its results. Everything
else, and every message when nothing is remembered (e.g. after a restart, unknown
ids), renders byte-identically to the pre-splice server.

COST OF THE CANONICAL "drop" POLICY (SIMULATED: real tokenizer + a replay of the
server's prefix-cache bookkeeping, dev/test_splice.py::test_followup_turn_policy_ab;
the model was not run). Within an agent loop splicing keeps the prefix cache
matching. But the first request of the NEXT user turn (after the loop's final
answer) renders every tool-call turn of that loop in template form again, so it
stops matching the cache at the loop's FIRST tool call and re-prefills all of the
loop's tool results once more. The pre-splice server lost only the final answer
there (unless the model spoke a commentary preamble, which already cost it the same).
Total prompt tokens prefilled over a 2-user-turn session, old / drop / keep:
  read-heavy (three ~2.5k-token results)       9,518 / 18,581 / 9,432
  read-heavy, commentary preambles            18,699 / 18,581 / 9,432
  write-heavy (file contents in call args)     7,559 /  6,434 / 3,285
  write-heavy, commentary preambles           10,740 /  6,434 / 3,285
Follow-up request alone, read-heavy: 100 / 9,249 / 100 tokens. The server's own
log shows ~150-200 prefill tok/s at 4-5k context (INDICATIVE), i.e. ~45-60 s
re-processed per follow-up turn of read-heavy work under "drop". "keep" avoids it
(the cache then diverges at the final answer, like the old server) but is NOT the
harmony rendering the model was trained on (the template drops that analysis):
earlier loops' reasoning stays in context. Its quality effect is unmeasured, and it
uses more context (read-heavy follow-up prompt 9,645 vs 9,471 tokens).

LENGTH FALLBACK. A spliced prompt is longer (it carries the reasoning). With
max_tokens (the server's prompt bound, smax-16) render() only returns a spliced
prompt that leaves at least `reserve` tokens below it for generation; otherwise it
falls back to the canonical plan ("keep" only) and then to the unspliced render
(== the pre-splice server's prompt, byte for byte). Splicing can therefore never
turn a request the old server served into a 400. The fallback is all-or-nothing on
purpose: it re-prefills the loop ONCE, then the prompt is the old one and grows like
it. Partial fallbacks move the cache divergence point every step. The price, in the
review's overflow scenario (--smax 16384, 20-step read loop, ~260-word reasoning,
~0.6k-token results; SIMULATED), as total prefill over the 19 requests every policy
serves: old server 15,866; all-or-nothing 27,523 (the switch at request 15
re-prefills 12,850; 26,715 with the default 1024-token reserve, which switches one
request earlier); drop-oldest-splices-first 55,881; drop-newest-first 46,337. Note
that before the switch a read loop saves only ~19 tokens of prefill per step (the
re-rendered call block): for read-heavy work, splicing buys reasoning retention, not
prefill. Extra cost of a fallback: one more render, ~25 ms at ~20k tokens.

Method. The template is run as today, except that each spliceable call carries a
unique sentinel as its arguments and no thinking. Its rendered block is then cut out
of the text, the text segments are tokenized separately, and the remembered token
lists are spliced in between. Every cut sits between two special tokens (the text
before a cut ends with <|end|>/<|call|>/<|return|>, the text after starts with
<|start|>); the HF tokenizer extracts added special tokens before normalization /
pre-tokenization, so separate tokenization of the segments equals whole-string
tokenization (dev/test_splice.py checks this at every such boundary).

Sentinels (tool-result placeholders included) carry a random per-render nonce, so
no client text can collide with them. The old fixed keys "NEURALTOOLRESULT{i}X"
could be spoofed by content containing that literal quoted string; that corner case
now renders the literal text instead of misplacing tool output. It is the only
intended output difference from the pre-splice renderer.
"""
import json
import re
import secrets
import threading
from array import array
from collections import OrderedDict

START_ASSISTANT = "<|start|>assistant"
AFTER_FINAL_POLICIES = ("drop", "keep")
DEFAULT_TOKEN_BUDGET = 1 << 22                             # 4,194,304 tokens = 16 MiB at 4 B/token
_SEGMENT_END = ("<|end|>", "<|call|>", "<|return|>")      # what the text before a cut must end with
_SEGMENT_START = "<|start|>"                               # what the text after a cut must start with


# ---------------------------------------------------------------- text helpers (verbatim server.py logic)
def special_texts(tok):
    """Control-token strings to defang, longest first (== server.py SPECIAL_TEXT)."""
    s = sorted({t for t in tok.all_special_tokens} | {t for t in tok.get_added_vocab()}, key=len, reverse=True)
    return [t for t in s if t.startswith("<|") and t.endswith("|>")]


def text_of(c):
    """OpenAI content (str / list of parts / None) -> plain text (== server.py _text)."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return str(c)


def defang(s, special_text):
    """Stop text that spells a control token (e.g. a file containing '<|end|>') from
    being tokenized as that control token (== server.py _defang)."""
    for sp in special_text:
        if sp in s:
            s = s.replace(sp, "<​" + sp[1:])
    return s


# ---------------------------------------------------------------- splice memory
class SpliceEntry:
    __slots__ = ("tokens", "name", "arguments")

    def __init__(self, tokens, name, arguments):
        self.tokens, self.name, self.arguments = tokens, name, arguments


class SpliceMemory:
    """Bounded LRU: tool_call id -> exact generated token ids of the response that
    produced that call (first token after the prompt through <|call|> inclusive),
    plus the call's function name and arguments text as returned to the client.
    Bounded by entry count (capacity) AND by the total number of stored tokens
    (max_tokens; None = no token bound). Tokens are kept as array('I'): 4 bytes each."""

    def __init__(self, capacity=512, call_token=None, max_tokens=DEFAULT_TOKEN_BUDGET):
        self.capacity = int(capacity)
        self.call_token = call_token
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self._d = OrderedDict()
        self._lock = threading.Lock()
        self._total = 0
        self.stats = {"stored": 0, "rejected": 0, "evicted": 0, "lookups": 0, "hits": 0}

    def remember(self, call_id, tokens, name=None, arguments=None):
        """Store; returns False (and stores nothing) for an empty id / token list, a
        token list longer than the whole token budget or not representable as uint32,
        or, when call_token is set, one that does not end with exactly one <|call|>
        (e.g. a tool message the model closed with <|end|>)."""
        try:
            toks = array("I", (int(t) for t in (tokens or ())))
        except (OverflowError, TypeError, ValueError):
            toks = array("I")
        ok = bool(call_id) and bool(toks) and self.capacity > 0
        if ok and self.max_tokens is not None:
            ok = len(toks) <= self.max_tokens
        if ok and self.call_token is not None:
            ok = toks[-1] == self.call_token and toks.count(self.call_token) == 1
        with self._lock:
            if not ok:
                self.stats["rejected"] += 1
                return False
            prev = self._d.pop(call_id, None)
            if prev is not None:
                self._total -= len(prev.tokens)
            self._d[call_id] = SpliceEntry(toks, name, arguments)
            self._total += len(toks)
            self.stats["stored"] += 1
            while len(self._d) > self.capacity or (self.max_tokens is not None and self._total > self.max_tokens):
                _, old = self._d.popitem(last=False)
                self._total -= len(old.tokens)
                self.stats["evicted"] += 1
        return True

    @property
    def total_tokens(self):
        return self._total

    def get(self, call_id, touch=True):
        with self._lock:
            self.stats["lookups"] += 1
            e = self._d.get(call_id) if call_id is not None else None
            if e is not None:
                self.stats["hits"] += 1
                if touch:
                    self._d.move_to_end(call_id)
            return e

    def touch(self, call_id):
        with self._lock:
            if call_id in self._d:
                self._d.move_to_end(call_id)

    def forget(self, call_id):
        with self._lock:
            e = self._d.pop(call_id, None)
            if e is not None:
                self._total -= len(e.tokens)
            return e is not None

    def clear(self):
        with self._lock:
            self._d.clear()
            self._total = 0

    def keys(self):
        with self._lock:
            return list(self._d.keys())

    def __len__(self):
        return len(self._d)

    def __contains__(self, call_id):
        return call_id in self._d


def _no_duplicate_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r}")
        d[k] = v
    return d


def _canonical_json(value):
    """Type-strict canonical form of a parsed JSON value: json.dumps tells true from 1
    and 3 from 3.0 at every depth (Python == does not), and sorting keys makes object
    key order irrelevant."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=True)


def call_matches(entry, name, raw_arguments):
    """True when the echoed call is the one that was remembered: same function name,
    and arguments either string-equal, or equal as JSON VALUES with types preserved
    (clients may re-serialize: key order, whitespace and string escapes may differ;
    true vs 1, 3 vs 3.0, 0 vs false are edits; a duplicate key on either side never
    matches). Anything else is treated as a client edit -> not spliced."""
    if entry.name is not None and name != entry.name:
        return False
    if entry.arguments is None:
        return True
    if isinstance(raw_arguments, str) and raw_arguments == entry.arguments:
        return True
    try:
        stored = json.loads(entry.arguments, object_pairs_hook=_no_duplicate_keys)
        echo = (json.loads(raw_arguments, object_pairs_hook=_no_duplicate_keys)
                if isinstance(raw_arguments, str) else raw_arguments)
        return _canonical_json(echo) == _canonical_json(stored)
    except (TypeError, ValueError, RecursionError):
        return False


# ---------------------------------------------------------------- messages -> template messages
def convert(messages, special_text):
    """server.py's message conversion (same logic, same order). Returns
    (sys_parts, out, calls): `out` is the template's loop_messages (tool results
    re-ordered to follow the call they answer), `calls` maps an index in `out` of an
    assistant tool-call entry to (call_index_in_its_message, call_id, name, raw_arguments)."""
    sys_parts, conv = [], []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            sys_parts.append(defang(text_of(m.get("content")), special_text))
        elif role == "user":
            conv.append({"role": "user", "content": defang(text_of(m.get("content")), special_text)})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            think = defang(text_of(m.get("reasoning_content") or m.get("reasoning") or m.get("thinking")), special_text)
            content = defang(text_of(m.get("content")), special_text)
            if tcs:
                for i, tc in enumerate(tcs):              # the template renders ONE call per message
                    fn = tc.get("function", tc)
                    raw_args = args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except ValueError:
                        args = {"_raw": args}
                    msg = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": fn.get("name"), "arguments": args}}]}
                    th = (think or content) if i == 0 else ""
                    if th:
                        msg["thinking"] = th
                    conv.append(("assistant_call", msg, tc.get("id"), (i, fn.get("name"), raw_args)))
            else:
                conv.append({"role": "assistant", "content": content})
        elif role == "tool":
            conv.append(("tool", defang(text_of(m.get("content")), special_text), m.get("tool_call_id")))
    # re-order so every tool result directly follows the call it answers
    out, pending_results, calls = [], {}, {}
    for item in conv:
        if isinstance(item, tuple) and item[0] == "tool":
            pending_results.setdefault(item[2], []).append(item[1])
    call_ids = {c[2] for c in conv if isinstance(c, tuple) and c[0] == "assistant_call"}
    used = set()
    for item in conv:
        if isinstance(item, tuple) and item[0] == "assistant_call":
            calls[len(out)] = (item[3][0], item[2], item[3][1], item[3][2])
            out.append(item[1])
            res = pending_results.get(item[2])
            if res:
                out.extend({"role": "tool", "content": r} for r in res)
                used.add(item[2])
        elif isinstance(item, tuple) and item[0] == "tool":
            if item[2] not in used and item[2] not in call_ids:
                out.append({"role": "tool", "content": item[1]})
        else:
            out.append(item)
    return sys_parts, out, calls


def plan_splices(out, calls, memory, after_final="drop"):
    """Which `out` entries to splice. Returns (plan, before_final, stats): plan =
    {index: SpliceEntry}; before_final = the plan indices that a final answer follows
    (non-empty only with after_final="keep"; they are what the canonical plan drops)."""
    stats = {"candidates": 0, "skipped_final": 0, "skipped_mismatch": 0, "kept_after_final": 0}
    if memory is None or not len(memory):
        return {}, set(), stats
    last_final = -1                                    # template future_final_message rule
    for k, m in enumerate(out):
        if m.get("role") == "assistant" and "tool_calls" not in m:
            last_final = k
    plan, before_final = {}, set()
    for j, (ci, cid, name, raw_args) in calls.items():
        if ci != 0 or cid is None or not isinstance(name, str):
            continue
        e = memory.get(cid, touch=False)
        if e is None:
            continue
        stats["candidates"] += 1
        if j < last_final and after_final != "keep":
            stats["skipped_final"] += 1
        elif not call_matches(e, name, raw_args):
            stats["skipped_mismatch"] += 1
        else:
            plan[j] = e
            if j < last_final:
                before_final.add(j)
                stats["kept_after_final"] += 1
    return plan, before_final, stats


# ---------------------------------------------------------------- server glue helpers (pure)
def splice_reserve(max_new, default, limit=None):
    """Generation room a spliced prompt must leave below the prompt bound: `default`
    (--splice-reserve), or the request's max_tokens when that is smaller, and never
    more than a quarter of the prompt bound `limit` (so a small --smax does not switch
    splicing off). Invalid or negative values never make it negative."""
    try:
        default = max(0, int(default))
    except (TypeError, ValueError):
        default = 0
    if limit is not None:
        default = min(default, max(0, int(limit)) // 4)
    if not max_new or isinstance(max_new, bool):
        return default
    try:
        return max(0, min(int(max_new), default))
    except (TypeError, ValueError, OverflowError):
        return default


def report_fields(info):
    """render() info -> flat keys for the response's `neural` object, logs/requests.jsonl
    and the console, so a disabled or degraded splice path is visible."""
    return {"spliced_calls": info.get("spliced", 0), "spliced_tokens": info.get("spliced_tokens", 0),
            "splice_policy": info.get("policy"), "splice_candidates": info.get("candidates", 0),
            "splice_skipped_final": info.get("skipped_final", 0),
            "splice_kept_after_final": info.get("kept_after_final", 0),
            "splice_skipped_mismatch": info.get("skipped_mismatch", 0),
            "splice_fallback": info.get("fallback"), "splice_length_fallback": info.get("length_fallback"),
            "splice_rejected_prompt_tokens": info.get("spliced_prompt_tokens")}


def report_note(fields):
    """Console suffix for the per-request line ('' when there is nothing to say)."""
    s = ""
    if fields.get("spliced_calls"):
        s += f" | {fields['spliced_calls']} tool turn(s) spliced"
    if fields.get("splice_length_fallback"):
        s += (f" | splice length fallback -> {fields['splice_length_fallback']} (spliced prompt "
              f"{fields.get('splice_rejected_prompt_tokens')} tok did not fit)")
    if fields.get("splice_skipped_mismatch"):
        s += f" | {fields['splice_skipped_mismatch']} edited tool call(s) not spliced"
    if fields.get("splice_fallback"):
        s += f" | SPLICE DISABLED FOR THIS REQUEST: {fields['splice_fallback']}"
    return s


# ---------------------------------------------------------------- render
class SpliceError(Exception):
    """The template output did not have the expected shape around a splice point."""


class _Collision(Exception):
    """A sentinel was not found exactly once (only possible if input text contains the nonce)."""


def _render_once(tok, sys_parts, out, plan, tools, effort, identity, nonce):
    msgs_out, raws, skeys = [], {}, {}
    for i, m in enumerate(out):
        if i in plan:
            key = f"NEURALSPLICE{i}X{nonce}"
            skeys[i] = key
            name = m["tool_calls"][0]["function"]["name"]
            m = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": key}}]}
        elif m["role"] == "tool":
            key = f"NEURALTOOLRESULT{i}X{nonce}"
            raws[json.dumps(key)] = m["content"]
            m = {"role": "tool", "content": key}
        msgs_out.append(m)
    msgs = ([{"role": "developer", "content": "\n\n".join(p for p in sys_parts if p)}] if any(sys_parts) else []) + msgs_out
    kw = {"reasoning_effort": effort}
    if identity:
        kw["model_identity"] = identity
    text = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False, **kw)
    if raws:                                           # tool output raw (harmony), not JSON-quoted; one pass
        pat = re.compile('"NEURALTOOLRESULT[0-9]+X' + nonce + '"')
        found = pat.findall(text)
        if sorted(found) != sorted(raws):
            raise _Collision("tool-result sentinel count")
        text = pat.sub(lambda mo: raws[mo.group(0)], text)
    if not plan:
        return tok(text, add_special_tokens=False).input_ids, text
    cuts = []
    for i, key in skeys.items():
        name = out[i]["tool_calls"][0]["function"]["name"]
        block = f"{START_ASSISTANT} to=functions.{name}<|channel|>commentary json<|message|>{json.dumps(key)}<|call|>"
        if text.count(json.dumps(key)) != 1:
            raise _Collision("splice sentinel count")
        p = text.find(block)
        if p < 0:
            raise SpliceError(f"template did not render the expected call block for message {i}")
        cuts.append((p, p + len(block), plan[i].tokens))
    cuts.sort()
    sa = tok(START_ASSISTANT, add_special_tokens=False).input_ids
    ids, pos = [], 0
    for a, b, toks in cuts:
        before, after = text[pos:a], text[b:]
        if (before and not before.endswith(_SEGMENT_END)) or not after.startswith(_SEGMENT_START):
            raise SpliceError("splice point is not between two special tokens")
        if before:
            ids += tok(before, add_special_tokens=False).input_ids
        ids += sa
        ids += toks
        pos = b
    ids += tok(text[pos:], add_special_tokens=False).input_ids
    return ids, text


def render(tok, messages, tools, effort, *, identity=None, special_text=None, memory=None, info=None,
           after_final="drop", max_tokens=None, reserve=0):
    """OpenAI messages -> harmony prompt token ids ending with "<|start|>assistant".

    memory: SpliceMemory or None (None / empty -> exactly the pre-splice rendering).
    after_final: "drop" (harmony-canonical: tool-call turns followed by a final answer
        render as the template renders them) or "keep" (splice them too; keeps the
        prefix cache across user turns, but keeps earlier loops' analysis in context).
    max_tokens: the caller's prompt bound (the server rejects len(ids) >= smax-16, so it
        passes smax-16). None = no length check. A spliced render is returned only if
        len(ids) + reserve < max_tokens; otherwise the canonical plan ("keep" only) and
        then the unspliced render are tried. The unspliced render is returned whatever
        its length (the caller applies its own bound, exactly as before splicing).
    reserve: generation room (tokens) a SPLICED prompt must leave below max_tokens.
    info: optional dict, filled with candidates / skipped_final / skipped_mismatch /
        kept_after_final / spliced / spliced_tokens / fallback (template-shape
        problem, str or None) / length_fallback (None, "canonical" or "unspliced") /
        spliced_prompt_tokens (length of the first spliced render that did not fit,
        or None) / policy."""
    if after_final not in AFTER_FINAL_POLICIES:
        raise ValueError(f"after_final must be one of {AFTER_FINAL_POLICIES}, not {after_final!r}")
    if special_text is None:
        special_text = special_texts(tok)
    sys_parts, out, calls = convert(messages, special_text)
    plan, before_final, stats = plan_splices(out, calls, memory, after_final)

    def attempt(pl):
        for _ in range(8):
            try:
                return _render_once(tok, sys_parts, out, pl, tools, effort, identity, secrets.token_hex(8))[0]
            except _Collision:
                continue
        raise RuntimeError("could not place prompt sentinels (input contains the random nonce?)")

    levels = [(None, plan)]                            # (length_fallback label, plan), longest first
    canonical = {j: e for j, e in plan.items() if j not in before_final}
    if before_final and canonical:
        levels.append(("canonical", canonical))
    if plan:
        levels.append(("unspliced", {}))
    reserve = max(0, int(reserve or 0))
    fallback = length_fallback = over = None
    for label, pl in levels:
        try:
            ids = attempt(pl)
        except SpliceError as e:                       # never fail a request over splicing
            fallback, pl = str(e), {}
            ids = attempt(pl)
            length_fallback = "unspliced" if over is not None else None
            break
        if not pl or max_tokens is None or len(ids) + reserve < max_tokens:
            length_fallback = label
            break
        if over is None:
            over = len(ids)
    for j in pl:                                       # LRU: touch only what was actually spliced
        memory.touch(calls[j][1])
    if info is not None:
        info.update(stats)
        info.update(policy=after_final, spliced=len(pl), spliced_tokens=sum(len(e.tokens) for e in pl.values()),
                    fallback=fallback, length_fallback=length_fallback, spliced_prompt_tokens=over)
    return ids
