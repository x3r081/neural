"""Model-aware chat formatting and tool-call protocol helpers.

The serving layer treats generated protocol text as data only. It never invokes
tools or rewrites the target model's routing or generation behavior.
"""
from __future__ import annotations

import copy
import json
import re
import uuid


class ProtocolError(ValueError):
    """A malformed or unsupported chat protocol request/response."""


_QWEN_XML_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}
_QWEN_JSON_TYPES = {"qwen3_moe"}


def _model_type(backend) -> str:
    config = getattr(backend, "config", None)
    model_type = getattr(config, "model_type", None)
    if not model_type:
        model_type = getattr(getattr(config, "text_config", None), "model_type", "")
    return str(model_type or "").lower()


def capabilities(backend) -> dict[str, bool | str]:
    """Return conservative, overridable capabilities for the configured model."""
    explicit = getattr(backend, "protocol_capabilities", None)
    if callable(explicit):
        explicit = explicit()
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise TypeError("backend.protocol_capabilities must be a mapping")
        protocol = explicit.get("tool_protocol", "none")
        if protocol not in {"none", "qwen_xml", "qwen_json"}:
            raise ValueError("unsupported backend tool_protocol")
        return {"tool_protocol": protocol,
                "thinking": bool(explicit.get("thinking", False)),
                "template_kwargs": bool(explicit.get("template_kwargs", False))}
    model_type = _model_type(backend)
    if model_type in _QWEN_XML_TYPES:
        return {"tool_protocol": "qwen_xml", "thinking": True, "template_kwargs": True}
    if model_type in _QWEN_JSON_TYPES:
        return {"tool_protocol": "qwen_json", "thinking": True, "template_kwargs": True}
    # No tool or reasoning convention is inferred for architectures without an
    # explicit protocol implementation (including Mixtral).
    return {"tool_protocol": "none", "thinking": False, "template_kwargs": False}


def _validate_schema(schema, name):
    if not isinstance(schema, dict):
        raise ProtocolError(f"tool {name!r} parameters must be an object schema")
    if schema.get("type", "object") != "object":
        raise ProtocolError(f"tool {name!r} parameters must have type object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not all(isinstance(k, str) and isinstance(v, dict)
                                                   for k, v in properties.items()):
        raise ProtocolError(f"tool {name!r} properties must map names to schemas")
    if not isinstance(required, list) or not all(isinstance(k, str) for k in required):
        raise ProtocolError(f"tool {name!r} required must be a list of names")
    unknown_required = set(required) - set(properties)
    if unknown_required:
        raise ProtocolError(f"tool {name!r} required names must exist in properties")
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise ProtocolError(f"tool {name!r} additionalProperties must be a boolean")
    for key, prop in properties.items():
        typ = prop.get("type")
        if typ not in {None, "string", "integer", "number", "boolean", "object", "array", "null"}:
            raise ProtocolError(f"tool {name!r} property {key!r} has unsupported type")
    return properties, set(required), additional


def validate_tools(tools):
    if tools is None:
        return {}
    if not isinstance(tools, list):
        raise ProtocolError("tools must be a list")
    result = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ProtocolError("only function tools are supported")
        fn = tool.get("function")
        if not isinstance(fn, dict):
            raise ProtocolError("each function tool must include a function object")
        name = fn.get("name")
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ProtocolError("function tool name must be a nonempty string of at most 128 characters")
        if name in result:
            raise ProtocolError(f"duplicate function tool name: {name}")
        params = fn.get("parameters", {"type": "object", "properties": {}})
        result[name] = _validate_schema(params, name)
    return result


def _normalize_messages(messages, *, tool_schemas):
    if not isinstance(messages, list) or not messages:
        raise ProtocolError("messages must be a nonempty list")
    result = copy.deepcopy(messages)
    for message in result:
        if not isinstance(message, dict):
            raise ProtocolError("each message must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ProtocolError("unsupported message role")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(item, dict) or item.get("type") != "text"
                   or not isinstance(item.get("text"), str) for item in content):
                raise ProtocolError("this runtime supports text-only message content")
        elif content is not None and not isinstance(content, str):
            raise ProtocolError("message content must be text or null")
        calls = message.get("tool_calls", [])
        if calls is None:
            calls = []
            message["tool_calls"] = calls
        if not isinstance(calls, list):
            raise ProtocolError("assistant tool_calls must be a list")
        if calls and role != "assistant":
            raise ProtocolError("tool_calls are only valid on assistant messages")
        for call in calls:
            if not isinstance(call, dict):
                raise ProtocolError("each assistant tool call must be an object")
            fn = call.get("function", call)
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ProtocolError("tool function must have a name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as exc:
                    raise ProtocolError("assistant tool-call arguments must contain valid JSON") from exc
            if not isinstance(args, dict):
                raise ProtocolError("assistant tool-call arguments must be a JSON object")
            fn["arguments"] = args
    return result


def render_chat_prompt(backend, request):
    """Render via the loaded tokenizer's template and return (prompt, thinking, tools)."""
    caps = capabilities(backend)
    declared_schemas = validate_tools(request.get("tools"))
    tool_choice = request.get("tool_choice", "auto")
    if tool_choice not in {"auto", "none"}:
        raise ProtocolError("tool_choice must be 'auto' or 'none'")
    tools = request.get("tools") if tool_choice != "none" else None
    schemas = declared_schemas if tools is not None else {}
    if schemas and caps["tool_protocol"] == "none":
        raise ProtocolError("function tools are not supported by this model protocol")
    kwargs = request.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ProtocolError("chat_template_kwargs must be an object")
    kwargs = copy.deepcopy(kwargs)
    if kwargs and not caps["template_kwargs"]:
        raise ProtocolError("chat_template_kwargs are unsupported by this model protocol")
    if "enable_thinking" in kwargs and not isinstance(kwargs["enable_thinking"], bool):
        raise ProtocolError("enable_thinking must be a boolean")
    if "preserve_thinking" in kwargs and not isinstance(kwargs["preserve_thinking"], bool):
        raise ProtocolError("preserve_thinking must be a boolean")
    messages = _normalize_messages(request.get("messages"), tool_schemas=schemas)
    tokenizer = backend.tokenizer
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception as exc:
        raise ProtocolError(f"could not render model chat template: {exc}") from exc
    if not isinstance(prompt, str):
        raise ProtocolError("tokenizer chat template must return text when tokenize=False")
    # Never infer hidden reasoning for a model whose protocol does not declare it.
    thinking = bool(caps["thinking"] and (
        kwargs.get("enable_thinking", False) or re.search(r"<think>\s*$", prompt) is not None))
    return prompt, thinking, schemas, caps["tool_protocol"]


def _validate_arguments(name, args, tool_schemas):
    if not isinstance(args, dict):
        raise ProtocolError("tool arguments must be a JSON object")
    if name not in tool_schemas:
        raise ProtocolError(f"model requested unavailable tool {name!r}")
    properties, required, additional = tool_schemas[name]
    missing = required - set(args)
    if missing:
        raise ProtocolError(f"tool {name!r} omitted required arguments: {sorted(missing)}")
    if additional is False and set(args) - set(properties):
        raise ProtocolError(f"tool {name!r} returned undeclared arguments")
    for key, value in args.items():
        typ = properties.get(key, {}).get("type")
        valid = {
            None: lambda v: True,
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "null": lambda v: v is None,
        }[typ]
        if not valid(value):
            raise ProtocolError(f"tool {name!r} argument {key!r} has the wrong type")
    return args


def _make_call(name, args, schemas):
    args = _validate_arguments(name, args, schemas)
    return {"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def _parse_qwen_xml(raw, schemas, *, final):
    calls = []
    for block in re.finditer(r"<tool_call>(.*?)</tool_call>", raw, re.S):
        match = re.fullmatch(r"\s*<function=([^>]+)>(.*?)</function>\s*", block.group(1), re.S)
        if match is None:
            raise ProtocolError("model generated malformed Qwen XML tool-call markup")
        name, body = match.groups()
        args = {}
        for key, value in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S):
            key = key.strip()
            value = value.removeprefix("\n").removesuffix("\n")
            prop_type = schemas.get(name.strip(), ({}, set(), True))[0].get(key, {}).get("type")
            if prop_type != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            args[key] = value
        calls.append(_make_call(name.strip(), args, schemas))
    visible = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.S)
    if "<tool_call>" in visible:
        if final:
            raise ProtocolError("model output ended with an incomplete tool call")
        visible = visible.split("<tool_call>", 1)[0].rstrip()
    for n in range(1, len("<tool_call>")):
        if visible.endswith("<tool_call>"[:n]):
            visible = visible[:-n]
            break
    if not final:
        visible = visible.rstrip()
    return visible, calls


def _parse_qwen_json(raw, schemas, *, final):
    calls = []
    for block in re.finditer(r"<tool_call>(.*?)</tool_call>", raw, re.S):
        try:
            payload = json.loads(block.group(1).strip())
        except json.JSONDecodeError as exc:
            raise ProtocolError("model generated malformed Qwen JSON tool-call markup") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise ProtocolError("Qwen JSON tool call must contain a function name")
        args = payload.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise ProtocolError("Qwen JSON tool arguments are invalid") from exc
        calls.append(_make_call(payload["name"], args, schemas))
    visible = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.S)
    if "<tool_call>" in visible:
        if final:
            raise ProtocolError("model output ended with an incomplete tool call")
        visible = visible.split("<tool_call>", 1)[0].rstrip()
    for n in range(1, len("<tool_call>")):
        if visible.endswith("<tool_call>"[:n]):
            visible = visible[:-n]
            break
    if not final:
        visible = visible.rstrip()
    return visible, calls


def split_chat_output(raw, *, thinking, tools, tool_protocol, final=False):
    """Parse only declared model protocols; tool calls are returned, never run."""
    reasoning, content = "", raw
    if thinking and tool_protocol in {"qwen_xml", "qwen_json"}:
        if "</think>" not in content:
            body = content.removeprefix("<think>").lstrip("\n")
            if not final:
                close = "</think>"
                partial = max((n for n in range(1, len(close)) if body.endswith(close[:n])), default=0)
                body = body[:-partial] if partial else body
                # Keep unstable terminal whitespace until we know whether it
                # is internal reasoning text or stripped by the close marker.
                body = body.rstrip()
            return body, "", []
        reasoning, content = content.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
        content = content.lstrip("\n")
    if not tools:
        return reasoning, content, []
    schemas = validate_tools(tools)
    if tool_protocol == "qwen_xml":
        content, calls = _parse_qwen_xml(content, schemas, final=final)
    elif tool_protocol == "qwen_json":
        content, calls = _parse_qwen_json(content, schemas, final=final)
    else:
        raise ProtocolError("function tools are not supported by this model protocol")
    return reasoning, content, calls
