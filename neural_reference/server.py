"""Small OpenAI-compatible HTTP server over the model-neutral engine contract."""
from __future__ import annotations

import hmac
import json
import math
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .protocol import ProtocolError, render_chat_prompt, split_chat_output


MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _generation_config(backend):
    config = getattr(backend, "generation_config", None)
    if config is None:
        config = getattr(getattr(backend, "model", None), "generation_config", None)
    return config


def _sampling_defaults(backend, *, chat):
    config = _generation_config(backend)
    do_sample = getattr(config, "do_sample", None) if config is not None else None
    temperature = getattr(config, "temperature", None) if config is not None else None
    if do_sample is False:
        temperature = 0.0
    elif temperature is None or not _finite_number(temperature) or temperature < 0:
        temperature = 1.0 if chat else 0.0
    top_k = getattr(config, "top_k", None) if config is not None else None
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        top_k = 20
    top_p = getattr(config, "top_p", None) if config is not None else None
    if not _finite_number(top_p) or not 0 < top_p <= 1:
        top_p = 0.95
    return float(temperature), int(top_k), float(top_p)


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _strict_int(obj, field, default, *, minimum=None):
    value = obj.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{field} must be >= {minimum}")
    return value


def _strict_bool(obj, field, default=False):
    value = obj.get(field, default)
    if not isinstance(value, bool):
        raise ProtocolError(f"{field} must be a boolean")
    return value


def _strict_float(obj, field, default, *, minimum=None, maximum=None, exclusive_min=False):
    value = obj.get(field, default)
    if not _finite_number(value):
        raise ProtocolError(f"{field} must be a finite number")
    value = float(value)
    if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
        op = ">" if exclusive_min else ">="
        raise ProtocolError(f"{field} must be {op} {minimum}")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{field} must be <= {maximum}")
    return value


def _read_json(handler):
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise ProtocolError("Content-Length is required")
    try:
        size = int(raw_length)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Content-Length must be an integer") from exc
    if size < 1 or size > MAX_REQUEST_BYTES:
        raise ProtocolError(f"request body must be between 1 byte and {MAX_REQUEST_BYTES} bytes")
    raw = handler.rfile.read(size)
    if len(raw) != size:
        raise ProtocolError("request body ended before Content-Length bytes were read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("request body must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("request body must be a JSON object")
    return value


def _error_body(message, error_type="invalid_request_error"):
    return {"error": {"message": str(message), "type": error_type}}


def _extract_result(result):
    if not isinstance(result, dict):
        raise RuntimeError("engine.generate must return an object")
    tokens = result.get("tokens", [])
    prompt_tokens = result.get("prompt_tokens", [])
    content = result.get("content", "")
    finish = result.get("finish_reason", result.get("finish", "stop"))
    timings = result.get("timings", {})
    prefix_cache = result.get("prefix_cache")
    if not isinstance(tokens, list) or not isinstance(prompt_tokens, list):
        raise RuntimeError("engine result tokens and prompt_tokens must be lists")
    if not isinstance(content, str) or not isinstance(finish, str) or not isinstance(timings, dict):
        raise RuntimeError("engine result content/finish/timings have invalid types")
    if prefix_cache is not None and not isinstance(prefix_cache, dict):
        raise RuntimeError("engine result prefix_cache must be an object when present")
    return tokens, prompt_tokens, content, finish, timings, prefix_cache


def _format_runtime_info(backend, engine, runtime_info):
    info = runtime_info() if callable(runtime_info) else runtime_info
    if info is None:
        info = {}
    if not isinstance(info, dict):
        raise TypeError("runtime_info must be a dict or zero-argument callable returning a dict")
    report = getattr(backend, "memory_report", None)
    if callable(report):
        info = {**info, "execution": report()}
    config = getattr(backend, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type:
        info.setdefault("model_type", model_type)
    for key, source in (("context", getattr(engine, "context", None)),
                        ("prefill_chunk", getattr(engine, "prefill_chunk", None))):
        if source is not None:
            info.setdefault(key, source)
    return info


def make_server(backend, engine, *, host="127.0.0.1", port=8001,
                model_id, runtime_info=None) -> ThreadingHTTPServer:
    """Create (but do not start) a serialized-inference HTTP server.

    backend provides tokenizer, config/model metadata and optionally
    memory_report(). engine.generate(prompt, ...) returns tokens, prompt_tokens,
    content, finish_reason/finish, and timings. Streaming callbacks receive the
    engine's cumulative decoded text. Tools are parsed as data and never run.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    inference_lock = __import__("threading").Lock()
    api_key = os.environ.get("NEURAL_API_KEY") or None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "NeuralRuntime/1"

        def log_message(self, fmt, *args):
            return

        def _send_json(self, status, value):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self):
            if api_key is None:
                return True
            supplied = self.headers.get("Authorization", "")
            expected = "Bearer " + api_key
            if hmac.compare_digest(supplied, expected):
                return True
            self._send_json(401, _error_body("Unauthorized", "authentication_error"))
            return False

        def _handle_error(self, exc, *, streaming=False):
            if streaming:
                try:
                    item = _error_body(str(exc), "invalid_request_error" if isinstance(exc, (ValueError, KeyError)) else "server_error")
                    self.wfile.write(("data: " + json.dumps(item, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    pass
                return
            status = 400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 500
            try:
                self._send_json(status, _error_body(str(exc), "invalid_request_error" if status == 400 else "server_error"))
            except OSError:
                pass

        def do_GET(self):
            if not self._authorized():
                return
            path = urlsplit(self.path).path.rstrip("/")
            if path in {"", "/health", "/v1/health"}:
                info = _format_runtime_info(backend, engine, runtime_info)
                return self._send_json(200, {**info, "status": "ok", "model": model_id})
            if path in {"/v1/models", "/models"}:
                return self._send_json(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "owned_by": "neural-local"}]})
            return self._send_json(404, _error_body("Not found", "not_found_error"))

        def do_POST(self):
            if not self._authorized():
                return
            path = urlsplit(self.path).path.rstrip("/")
            if path not in {"/tokenize", "/completion", "/v1/chat/completions", "/chat/completions"}:
                return self._send_json(404, _error_body("Not found", "not_found_error"))
            locked = False
            streaming = False
            try:
                req = _read_json(self)
                if path == "/tokenize":
                    content = req.get("content")
                    if not isinstance(content, str):
                        raise ProtocolError("content must be a string")
                    add_special = _strict_bool(req, "add_special", False)
                    tokens = backend.tokenizer.encode(content, add_special_tokens=add_special)
                    return self._send_json(200, {"tokens": list(tokens)})

                chat = path in {"/v1/chat/completions", "/chat/completions"}
                if "model" in req and req["model"] != model_id:
                    raise ProtocolError(f"model must be {model_id}")
                if not chat:
                    if _strict_bool(req, "stream", False):
                        raise ProtocolError("streaming is supported on /v1/chat/completions, not /completion")
                    if any(field in req for field in ("messages", "tools", "tool_choice", "chat_template_kwargs")):
                        raise ProtocolError("chat protocol fields require /v1/chat/completions")
                if _strict_int(req, "n", 1, minimum=1) != 1:
                    raise ProtocolError("only n=1 is supported")
                unsupported = {"logit_bias", "presence_penalty", "frequency_penalty", "best_of",
                               "guided_json", "guided_regex", "response_format"}
                used_unsupported = [field for field in unsupported if field in req and req[field] not in (None, {}, False)]
                if used_unsupported:
                    raise ProtocolError("unsupported request fields: " + ", ".join(sorted(used_unsupported)))
                stop = req.get("stop")
                if stop not in (None, [], ""):
                    raise ProtocolError("stop sequences are not supported by this engine")
                if "reasoning_effort" in req:
                    raise ProtocolError("reasoning_effort is not supported; configure the model chat template explicitly")
                stream = _strict_bool(req, "stream", False) if chat else False
                ignore_eos = _strict_bool(req, "ignore_eos", False)
                cache_prompt = _strict_bool(req, "cache_prompt", True)
                limit_fields = [field for field in ("max_completion_tokens", "max_tokens", "n_predict") if field in req]
                limits = [_strict_int(req, field, None, minimum=1) for field in limit_fields]
                if len(set(limits)) > 1:
                    raise ProtocolError("max_completion_tokens, max_tokens and n_predict must agree when combined")
                max_tokens = limits[0] if limits else 256
                temperature_default, top_k_default, top_p_default = _sampling_defaults(backend, chat=chat)
                temperature = _strict_float(req, "temperature", temperature_default, minimum=0)
                top_k = _strict_int(req, "top_k", top_k_default, minimum=0)
                top_p = _strict_float(req, "top_p", top_p_default, minimum=0, maximum=1, exclusive_min=True)
                seed = _strict_int(req, "seed", 0)

                if chat:
                    prompt, thinking, schemas, tool_protocol = render_chat_prompt(backend, req)
                    active_tools = req.get("tools") if req.get("tool_choice", "auto") != "none" else None
                    if active_tools == []:
                        active_tools = None
                else:
                    prompt = req.get("prompt", req.get("prompt_tokens"))
                    if isinstance(prompt, str):
                        pass
                    elif isinstance(prompt, list):
                        if not all(isinstance(token, int) and not isinstance(token, bool) for token in prompt):
                            raise ProtocolError("prompt token IDs must be integers")
                    else:
                        raise ProtocolError("prompt must be text or a list of token IDs")
                    thinking, schemas, tool_protocol, active_tools = False, {}, "none", None

                if not inference_lock.acquire(blocking=False):
                    return self._send_json(409, _error_body("Server is processing another inference request", "server_busy"))
                locked = True
                request_id = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                last_sent = {"reasoning_content": "", "content": ""}

                def send_chunk(delta=None, finish=None, usage=None, neural=None):
                    payload = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                               "model": model_id, "choices": [{"index": 0, "delta": delta or {},
                                                                  "finish_reason": finish}]}
                    if usage is not None:
                        payload["usage"] = usage
                    if neural is not None:
                        payload["neural"] = neural
                    self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()

                def on_text(raw):
                    if not stream:
                        return
                    reason, visible, _ = split_chat_output(raw, thinking=thinking, tools=active_tools,
                                                           tool_protocol=tool_protocol, final=False)
                    delta = {}
                    for key, text in (("reasoning_content", reason), ("content", visible)):
                        old = last_sent[key]
                        if text.startswith(old):
                            suffix = text[len(old):]
                            if suffix:
                                delta[key] = suffix
                                last_sent[key] = text
                        elif text:
                            delta[key] = text
                            last_sent[key] = text
                    if delta:
                        send_chunk(delta)

                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    streaming = True
                    send_chunk({"role": "assistant"})

                result = engine.generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                         top_k=top_k, top_p=top_p, seed=seed,
                                         ignore_eos=ignore_eos, on_text=on_text if stream else None,
                                         cache_prompt=cache_prompt)
                tokens, prompt_tokens, raw, finish_reason, timings, prefix_cache = _extract_result(result)
                if not chat:
                    return self._send_json(200, {"content": raw, "tokens": tokens,
                                                 "prompt_tokens": prompt_tokens,
                                                 "finish_reason": finish_reason, "timings": timings,
                                                 "prefix_cache": prefix_cache})
                reason, content, calls = split_chat_output(raw, thinking=thinking, tools=active_tools,
                                                           tool_protocol=tool_protocol, final=True)
                finish = "tool_calls" if calls else finish_reason
                usage = {"prompt_tokens": len(prompt_tokens), "completion_tokens": len(tokens),
                         "total_tokens": len(prompt_tokens) + len(tokens)}
                if stream:
                    for key, text in (("reasoning_content", reason), ("content", content)):
                        old = last_sent[key]
                        if text.startswith(old) and len(text) > len(old):
                            send_chunk({key: text[len(old):]})
                            last_sent[key] = text
                    # Final parse may reveal whole structured calls; their payload
                    # is sent as inert protocol data and is never executed here.
                    if calls:
                        send_chunk({"tool_calls": [dict(call, index=i) for i, call in enumerate(calls)]})
                    metrics = dict(timings)
                    if prefix_cache is not None:
                        metrics["prefix_cache"] = prefix_cache
                    send_chunk(finish=finish, usage=usage, neural=metrics)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                message = {"role": "assistant", "content": content}
                if reason:
                    message["reasoning_content"] = reason
                if calls:
                    message["tool_calls"] = calls
                metrics = dict(timings)
                if prefix_cache is not None:
                    metrics["prefix_cache"] = prefix_cache
                return self._send_json(200, {"id": request_id, "object": "chat.completion",
                    "created": created, "model": model_id,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage, "neural": metrics})
            except (BrokenPipeError, ConnectionResetError):
                # GenerationEngine invalidates its checkpoint on callback failure.
                return
            except Exception as exc:
                self._handle_error(exc, streaming=streaming)
            finally:
                if locked:
                    inference_lock.release()

    return ThreadingHTTPServer((host, port), Handler)
