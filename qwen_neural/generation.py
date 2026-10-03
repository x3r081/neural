"""Text generation and Qwen's chat/tool protocol, separate from the GPT-OSS server."""
from __future__ import annotations

import copy
import json
import math
import re
import time
import uuid

import torch


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    result = copy.deepcopy(messages)
    for message in result:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(item, dict) or item.get("type") != "text"
                   or not isinstance(item.get("text"), str) for item in content):
                raise ValueError("This runtime currently supports text inputs only")
        elif content is not None and not isinstance(content, str):
            raise ValueError("message content must be text")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("each tool call must be an object")
            fn = call.get("function", call)
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ValueError("tool function must have a name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("tool call arguments must be a JSON object")
            fn["arguments"] = args
    return result


def render_prompt(tokenizer, request):
    options = request.get("chat_template_kwargs") or {}
    if not isinstance(options, dict):
        raise ValueError("chat_template_kwargs must be an object")
    for key in ("enable_thinking", "preserve_thinking"):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f"{key} must be a boolean")
    thinking = options.get("enable_thinking", True)
    tools = request.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("only function tools are supported")
            fn = tool.get("function")
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ValueError("each tool function must have a name")
            parameters = fn.get("parameters", {})
            if not isinstance(parameters, dict) or not isinstance(parameters.get("properties", {}), dict):
                raise ValueError("tool parameters and properties must be objects")
            if any(not isinstance(p, dict) for p in parameters.get("properties", {}).values()):
                raise ValueError("tool parameter schemas must be objects")
    prompt = tokenizer.apply_chat_template(
        normalize_messages(request["messages"]), tools=request.get("tools"),
        tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking, preserve_thinking=options.get("preserve_thinking", True),
    )
    return prompt, thinking


def split_response(raw, thinking, tools=None, *, final=False):
    """Return reasoning, visible answer and complete structured tool calls.

    While a tool call is incomplete its markup is withheld from visible content.
    No tool is executed by this parser.
    """
    reasoning = ""
    content = raw
    if thinking:
        if "</think>" not in raw:
            return raw.removeprefix("<think>\n"), "", []
        reasoning, content = raw.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>\n").strip()
        content = content.lstrip("\n")
    # Plain chat may legitimately discuss XML/tool-call markup as code. Only
    # interpret it as a protocol when the caller supplied function tools.
    if not tools:
        return reasoning, content, []
    schemas = {t["function"]["name"]: t["function"].get("parameters", {})
               for t in tools or [] if t.get("type") == "function"}
    calls = []
    for block in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.S):
        match = re.fullmatch(r"\s*<function=([^>]+)>(.*?)</function>\s*", block.group(1), re.S)
        if match is None:
            raise ValueError("Model generated malformed tool-call markup")
        name, body = match.groups()
        name = name.strip()
        if name not in schemas:
            raise ValueError(f"Model requested an unavailable tool: {name}")
        args = {}
        properties = schemas[name].get("properties", {})
        for key, value in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S):
            # Remove the template's single framing newline on each side,
            # preserving newlines that belong to string/file contents.
            key, value = key.strip(), value.removeprefix("\n").removesuffix("\n")
            if properties.get(key, {}).get("type") != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            args[key] = value
        calls.append({"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S)
    if "<tool_call>" in content:
        if final:
            raise ValueError("Model output ended with an incomplete tool call")
        content = content.split("<tool_call>", 1)[0].rstrip()
    # Avoid leaking a partial protocol marker across streaming boundaries.
    for n in range(1, len("<tool_call>")):
        if content.endswith("<tool_call>"[:n]):
            content = content[:-n]
            break
    return reasoning, content, calls


def sample(logits, temperature=0.0, top_k=20, top_p=0.95, generator=None):
    scores = logits.float()
    if temperature <= 0:
        return torch.argmax(scores, dim=-1, keepdim=True)
    scores = scores / temperature
    if top_k > 0:
        threshold = torch.topk(scores, min(top_k, scores.shape[-1])).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if 0 < top_p < 1:
        ordered, indices = torch.sort(scores, descending=True)
        excluded = torch.softmax(ordered, -1).cumsum(-1) > top_p
        excluded[..., 1:] = excluded[..., :-1].clone()
        excluded[..., 0] = False
        ordered.masked_fill_(excluded, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, indices, ordered)
    return torch.multinomial(torch.softmax(scores, -1), 1, generator=generator)


class GenerationEngine:
    def __init__(self, backend, context=16384, prefill_chunk=64):
        self.backend, self.context, self.prefill_chunk = backend, context, prefill_chunk
        # The tokenizer is immutable for this engine. Some tokenizer backends
        # rebuild vocabulary maps while calculating their full length.
        self.vocab_size = len(backend.tokenizer)

    @torch.inference_mode()
    def generate(self, prompt, *, max_tokens=256, temperature=0.0, top_k=20,
                 top_p=0.95, seed=0, ignore_eos=False, on_text=None):
        backend = self.backend
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("invalid temperature, top_p or top_k")
        ids = (prompt if isinstance(prompt, list) else
               backend.tokenizer.encode(prompt, add_special_tokens=False))
        if not ids or any(not isinstance(i, int) or i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("invalid or empty prompt token IDs")
        if max_tokens < 1 or len(ids) + max_tokens > self.context:
            raise ValueError(f"prompt plus output must fit context={self.context}")
        device = next(p.device for p in backend.model.parameters() if not p.is_meta)
        generator = torch.Generator(device=device).manual_seed(seed)
        # Requests get independent recurrent and attention state. The immutable
        # expert cache may stay warm; it does not contain conversation state.
        past = None
        generated = []
        eos = backend.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        eos.update([248044, 248046])  # official pinned checkpoint stop IDs
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for start in range(0, len(ids), self.prefill_chunk):
            token_batch = torch.tensor([ids[start:start+self.prefill_chunk]], device=device)
            out = backend.forward(token_batch, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
        torch.cuda.synchronize(device)
        prefill_s = time.perf_counter() - started
        finish_reason = "length"
        decode_calls = 0
        first_token_s = None
        raw = ""
        for step in range(max_tokens):
            token = sample(out.logits[:, -1, :], temperature, top_k, top_p, generator)
            token_id = int(token.item())
            generated.append(token_id)
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
            if token_id in eos and not ignore_eos:
                finish_reason = "stop"
                break
            raw = backend.tokenizer.decode(generated, skip_special_tokens=False).rstrip("\ufffd")
            if on_text:
                on_text(raw)
            if step + 1 < max_tokens:
                out = backend.forward(token, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = out.past_key_values
                decode_calls += 1
        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - started
        decode_s = wall_s - prefill_s
        result = {"content": raw, "tokens": generated, "prompt_tokens": ids,
                  "finish_reason": finish_reason, "timings": {
                      "prompt_n": len(ids), "prompt_ms": prefill_s*1000,
                      "prompt_per_second": len(ids)/prefill_s,
                      "predicted_n": len(generated), "decode_calls": decode_calls,
                      "predicted_ms": decode_s*1000,
                      "predicted_per_second": decode_calls/decode_s if decode_s else 0,
                      "first_token_s": first_token_s, "wall_s": wall_s,
                      "generation_tokens_per_wall_s": len(generated)/wall_s,
                      "clock": "request prefill through generation, including sampling, detokenization and streaming callback; excludes startup"}}
        del past, out
        return result
