import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from neural_reference.server import make_server


class FakeTokenizer:
    def __init__(self, suffix=""):
        self.suffix = suffix

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) % 97 for ch in text]

    def apply_chat_template(self, messages, *, tools=None, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(str(m.get("content", "")) for m in messages) + self.suffix


class FakeBackend:
    def __init__(self, model_type="qwen3_5_moe", suffix="<think>"):
        self.tokenizer = FakeTokenizer(suffix)
        self.config = SimpleNamespace(model_type=model_type, max_position_embeddings=4096)
        self.model = SimpleNamespace(generation_config=SimpleNamespace(
            do_sample=True, temperature=0.8, top_k=13, top_p=0.9))
        self.device = "cpu"

    def memory_report(self):
        return {"device": "cpu", "memory_bytes": 123}


class FakeEngine:
    context = 4096
    prefill_chunk = 64

    def __init__(self, backend, content="answer", *, gate=None, fail_once=False):
        self.backend = backend
        self.content = content
        self.gate = gate
        self.fail_once = fail_once
        self.calls = []
        self.stream_values = None
        self.prefix_cache = {"hit": True, "cached_tokens": 64, "new_prefill_tokens": 20}

    def generate(self, prompt, *, max_tokens, temperature, top_k, top_p, seed,
                 ignore_eos, on_text=None, cache_prompt=True):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                           "temperature": temperature, "top_k": top_k,
                           "top_p": top_p, "seed": seed, "ignore_eos": ignore_eos})
        if self.gate is not None:
            self.gate.set()
            time.sleep(0.25)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("injected engine failure")
        if on_text is not None:
            for value in (self.stream_values or [self.content]):
                on_text(value)
        prompt_tokens = (prompt if isinstance(prompt, list)
                         else self.backend.tokenizer.encode(prompt))
        return {"content": self.content, "tokens": [1, 2],
                "prompt_tokens": list(prompt_tokens), "finish_reason": "stop",
                "timings": {"prompt_ms": 2.5, "predicted_per_second": 3.0,
                            "computed_prompt_n": 20, "cached_prompt_n": 64,
                            "effective_prompt_per_second": 33.6},
                "prefix_cache": dict(self.prefix_cache)}


@pytest.fixture
def running_server(monkeypatch):
    monkeypatch.delenv("NEURAL_API_KEY", raising=False)
    servers = []

    def start(backend=None, engine=None, **kwargs):
        backend = backend or FakeBackend()
        engine = engine or FakeEngine(backend)
        srv = make_server(backend, engine, host="127.0.0.1", port=0,
                          model_id=kwargs.pop("model_id", "fake-dynamic-model"), **kwargs)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        servers.append((srv, thread, backend, engine))
        return f"http://127.0.0.1:{srv.server_address[1]}", backend, engine

    yield start
    for srv, thread, _, _ in servers:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)


def request(base, path, body=None, *, method=None, headers=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, response.read().decode("utf-8"), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8"), error.headers


def test_health_models_tokenize_and_raw_token_or_text_completion(running_server):
    base, backend, engine = running_server(model_id="qwen-custom-id",
        runtime_info={"precision": "BF16", "backend": "fixture"})
    status, raw, _ = request(base, "/health")
    health = json.loads(raw)
    assert status == 200 and health["model"] == "qwen-custom-id"
    assert health["precision"] == "BF16" and health["execution"] == {"device": "cpu", "memory_bytes": 123}
    status, raw, _ = request(base, "/v1/models", method="GET")
    assert json.loads(raw)["data"][0]["id"] == "qwen-custom-id"
    status, raw, _ = request(base, "/tokenize", {"content": "abc", "add_special": False})
    assert status == 200 and json.loads(raw)["tokens"] == backend.tokenizer.encode("abc")
    status, raw, _ = request(base, "/completion", {"prompt": [1, 2, 3], "n_predict": 4})
    result = json.loads(raw)
    assert status == 200 and result["prompt_tokens"] == [1, 2, 3]
    assert engine.calls[-1]["max_tokens"] == 4 and engine.calls[-1]["temperature"] == 0.8
    assert result["prefix_cache"]["cached_tokens"] == 64
    status, raw, _ = request(base, "/completion", {"prompt": "raw text"})
    assert status == 200 and engine.calls[-1]["prompt"] == "raw text"


def test_chat_json_and_sse_emit_reasoning_and_dynamic_model_id(running_server):
    content = "<think>reasoning</think>\n\nfinal answer"
    base, _, engine = running_server(engine=None, model_id="runtime-v2")
    engine.content = content
    body = {"model": "runtime-v2", "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 9, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}
    status, raw, _ = request(base, "/v1/chat/completions", body)
    result = json.loads(raw)
    message = result["choices"][0]["message"]
    assert status == 200 and result["model"] == "runtime-v2"
    assert message["reasoning_content"] == "reasoning" and message["content"] == "final answer"
    assert engine.calls[-1]["max_tokens"] == 9 and engine.calls[-1]["temperature"] == 0

    body["stream"] = True
    status, raw, headers = request(base, "/v1/chat/completions", body)
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    assert raw.rstrip().endswith("data: [DONE]")
    deltas = [choice["delta"] for item in chunks for choice in item.get("choices", [])]
    assert any(delta.get("reasoning_content") == "reasoning" for delta in deltas)
    assert any(delta.get("content") == "final answer" for delta in deltas)
    assert chunks[-1]["model"] == "runtime-v2"
    assert chunks[-1]["neural"]["prefix_cache"]["cached_tokens"] == 64


@pytest.mark.parametrize("split_at", range(1, len("</think>")))
def test_streaming_thinking_delimiter_is_buffered_and_matches_json(split_at, running_server):
    raw_text = "<think>reasoning  </think>\n\nanswer"
    close_at = raw_text.index("</think>")
    backend = FakeBackend(suffix="<think>")
    engine = FakeEngine(backend, raw_text)
    engine.stream_values = [raw_text[:close_at + split_at], raw_text]
    base, _, _ = running_server(backend, engine)
    body = {"messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": True}}
    _, raw_json, _ = request(base, "/v1/chat/completions", body)
    expected = json.loads(raw_json)["choices"][0]["message"]
    _, raw_sse, _ = request(base, "/v1/chat/completions", body | {"stream": True})
    chunks = [json.loads(line[6:]) for line in raw_sse.splitlines() if line.startswith("data: {")]
    deltas = [choice["delta"] for item in chunks for choice in item.get("choices", [])]
    streamed_reason = "".join(delta.get("reasoning_content", "") for delta in deltas)
    streamed_content = "".join(delta.get("content", "") for delta in deltas)
    assert streamed_reason == expected["reasoning_content"] == "reasoning"
    assert streamed_content == expected["content"] == "answer"


def test_qwen_tool_calls_are_returned_as_data_and_plain_markup_is_preserved(running_server):
    backend = FakeBackend(model_type="qwen3_5_moe", suffix="")
    engine = FakeEngine(backend, content=("<tool_call><function=weather>"
        "<parameter=city>\nOslo\n</parameter></function></tool_call>"))
    base, _, _ = running_server(backend, engine)
    history = [{"role": "assistant", "content": None, "tool_calls": [{"id": "old",
        "type": "function", "function": {"name": "weather", "arguments": '{"city":"Oslo"}'}}]}]
    original = json.loads(json.dumps(history))
    body = {"messages": history + [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"]}}}]}
    status, raw, _ = request(base, "/v1/chat/completions", body)
    result = json.loads(raw)
    assert status == 200 and result["choices"][0]["finish_reason"] == "tool_calls"
    call = result["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Oslo"}
    assert history == original

    full_tool_text = "Preparing.  <tool_call><function=weather><parameter=city>\nOslo\n</parameter></function></tool_call>"
    engine.content = full_tool_text
    marker = full_tool_text.index("<tool_call>")
    engine.stream_values = [full_tool_text[:marker + i] for i in range(1, len("<tool_call>"))]
    engine.stream_values.append(full_tool_text)
    status, raw, _ = request(base, "/v1/chat/completions", body | {"stream": True})
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    deltas = [choice["delta"] for item in chunks for choice in item.get("choices", [])]
    assert status == 200
    assert "".join(delta.get("content", "") for delta in deltas) == "Preparing.  "
    streamed_calls = [delta["tool_calls"][0] for delta in deltas if delta.get("tool_calls")]
    assert json.loads(streamed_calls[0]["function"]["arguments"]) == {"city": "Oslo"}

    literal = "Please explain <tool_call> as text"
    engine.content = literal
    engine.stream_values = None
    status, raw, _ = request(base, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "literal"}]})
    assert json.loads(raw)["choices"][0]["message"]["content"] == literal


def test_unsupported_model_tools_and_bad_numeric_fields_are_client_errors(running_server):
    base, _, _ = running_server(FakeBackend(model_type="mixtral", suffix=""))
    status, raw, _ = request(base, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hello"}], "tools": [
            {"type": "function", "function": {"name": "unsafe", "parameters": {}}}]})
    assert status == 400 and "not supported" in json.loads(raw)["error"]["message"]
    for body in ({"prompt": "x", "temperature": True},
                 {"prompt": "x", "top_p": float("nan")},
                 {"prompt": "x", "max_tokens": True}):
        status, raw, _ = request(base, "/completion", body)
        assert status == 400, raw
    status, raw, _ = request(base, "/completion", {"prompt": "x", "stream": True})
    assert status == 400 and "streaming" in json.loads(raw)["error"]["message"]


def test_api_key_required_when_configured(running_server, monkeypatch):
    monkeypatch.setenv("NEURAL_API_KEY", "secret")
    base, _, _ = running_server()
    status, _, _ = request(base, "/v1/models", method="GET")
    assert status == 401
    status, raw, _ = request(base, "/v1/models", method="GET",
                             headers={"Authorization": "Bearer secret"})
    assert status == 200 and json.loads(raw)["data"][0]["id"] == "fake-dynamic-model"


def test_inference_busy_returns_409_and_engine_exception_releases_lock(running_server):
    entered = threading.Event()
    backend = FakeBackend(suffix="")
    engine = FakeEngine(backend, gate=entered, fail_once=True)
    base, _, _ = running_server(backend, engine)
    body = {"prompt": "first"}
    first_result = []

    def first_request():
        first_result.append(request(base, "/completion", body)[0])

    worker = threading.Thread(target=first_request)
    worker.start()
    assert entered.wait(1)
    status, raw, _ = request(base, "/completion", {"prompt": "second"})
    assert status == 409 and "processing another" in json.loads(raw)["error"]["message"]
    worker.join(timeout=2)
    assert first_result == [500]
    engine.gate = None
    status, raw, _ = request(base, "/completion", {"prompt": "third"})
    assert status == 200, raw


def test_model_mismatch_and_missing_prompt_are_rejected(running_server):
    base, _, _ = running_server(model_id="expected")
    status, raw, _ = request(base, "/v1/chat/completions", {
        "model": "other", "messages": [{"role": "user", "content": "x"}]})
    assert status == 400 and "model must be expected" in json.loads(raw)["error"]["message"]
    status, raw, _ = request(base, "/completion", {"max_tokens": 3})
    assert status == 400 and "prompt" in json.loads(raw)["error"]["message"]
