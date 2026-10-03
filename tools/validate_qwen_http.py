"""Exercise the local Qwen HTTP/chat/tool protocol and save an evidence report.

This client never starts a server and never executes generated tools. The
get_weather round trip uses a fixed local tool-result string supplied as a
normal tool message.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "qwen3.6-35b-a3b-neural"
DEFAULT_URL = "http://localhost:8001"
DEFAULT_OUTPUT = "reports/qwen_http_validation.json"
TIMEOUT = 7200
WEATHER_FIXTURE = "18 C and sunny"


def _url_root(url: str) -> str:
    root = url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "neural-qwen-http-validator/1"}
    key = os.environ.get("NEURAL_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    return headers


def _request(method: str, url: str, body=None, *, stream: bool = False) -> dict:
    headers = _headers()
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    record = {
        "method": method,
        "url": url,
        "request_body": body,
        "timeout_seconds": TIMEOUT,
        "streaming": stream,
    }
    try:
        try:
            response_context = urlopen(request, timeout=TIMEOUT)
        except HTTPError as exc:
            response_context = exc
        with response_context as response:
            record["status"] = response.status
            record["response_headers"] = dict(response.headers.items())
            if stream:
                raw_lines = []
                events = []
                errors = []
                answer_parts = []
                done = False
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    raw_lines.append(line)
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        done = True
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        errors.append(f"invalid SSE JSON: {exc}: {payload}")
                        continue
                    events.append(event)
                    if event.get("error"):
                        errors.append(str(event["error"]))
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if isinstance(text, str):
                            answer_parts.append(text)
                record["response_raw"] = "\n".join(raw_lines)
                record["response_events"] = events
                record["response_text"] = "".join(answer_parts)
                record["saw_done"] = done
                record["stream_errors"] = errors
            else:
                response_bytes = response.read()
                raw_body = response_bytes.decode("utf-8", errors="replace")
                record["response_raw"] = raw_body
                try:
                    record["response_json"] = json.loads(raw_body)
                except json.JSONDecodeError:
                    record["response_json"] = None
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        record["transport_error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.perf_counter() - start
    return record


def _body(record: dict) -> dict:
    value = record.get("response_json")
    return value if isinstance(value, dict) else {}


def _has_answer(text: str, expected: str) -> bool:
    if expected == "42":
        return re.search(r"(?<!\d)42(?!\d)", text) is not None
    if expected == "18 C":
        return re.search(r"(?<!\d)18\s*(?:°\s*)?(?:C\b|degrees\s+Celsius\b)", text, re.I) is not None
    return expected.casefold() in text.casefold()


def _save_report(path: Path, report: dict) -> None:
    report["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_check(report: dict, path: Path, name: str, request_record: dict,
               passed: bool, detail: str) -> bool:
    entry = {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "timing_seconds": request_record.get("elapsed_seconds"),
        "request": request_record,
    }
    report["checks"].append(entry)
    _save_report(path, report)
    print(f"{'PASS' if passed else 'FAIL'} {name}: {detail} "
          f"({request_record.get('elapsed_seconds', 0):.3f}s)", flush=True)
    return bool(passed)


def _chat_body(messages, *, max_tokens: int, stream: bool = False, tools=None,
               tool_choice=None) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="server root or /v1 URL (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="JSON report path (default: %(default)s)")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = repo / output
    root = _url_root(args.url)
    api = root + "/v1"
    report = {
        "validator": "tools/validate_qwen_http.py",
        "server_url": root,
        "model": MODEL,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timeout_seconds": TIMEOUT,
        "functional_request_settings": {
            "temperature": 0,
            "enable_thinking": False,
            "note": "These short functional checks intentionally do not use the benchmark default thinking mode.",
        },
        "checks": [],
        "scope_note": "Local HTTP protocol smoke checks only; no external tools are run and this does not establish broad agent quality.",
    }
    _save_report(output, report)
    all_passed = True
    server_reachable = True

    health = _request("GET", root + "/health")
    body = _body(health)
    passed = health.get("status") == 200 and body.get("status") == "ok"
    all_passed &= _run_check(report, output, "health", health, passed,
                             "GET /health returns status=ok" if passed else "health endpoint did not return status=ok")
    server_reachable &= "transport_error" not in health

    models = _request("GET", api + "/models")
    models_body = _body(models)
    model_ids = [entry.get("id") for entry in models_body.get("data", []) if isinstance(entry, dict)]
    passed = models.get("status") == 200 and MODEL in model_ids
    all_passed &= _run_check(report, output, "models", models, passed,
                             "model list includes the configured Qwen ID" if passed else "model list lacks the configured Qwen ID")
    server_reachable &= "transport_error" not in models

    if not server_reachable:
        report["final_status"] = "failed_unreachable"
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _save_report(output, report)
        return 1

    invalid_requests = [
        ("reject_top_level_array", []),
        ("reject_null_message", {"model": MODEL, "messages": [None], "max_tokens": 1}),
        ("reject_string_false_stream", _chat_body(
            [{"role": "user", "content": "validation only"}], max_tokens=1, stream="false"
        )),
    ]
    for name, request_body in invalid_requests:
        result = _request("POST", api + "/chat/completions", request_body)
        passed = result.get("status") == 400
        all_passed &= _run_check(report, output, name, result, passed,
                                 "malformed request rejected with HTTP 400" if passed else
                                 f"expected HTTP 400, got {result.get('status')}")

    prompt = "What is 6 times 7? Reply with only the integer."
    plain_body = _chat_body([{"role": "user", "content": prompt}], max_tokens=16)
    plain = _request("POST", api + "/chat/completions", plain_body)
    plain_json = _body(plain)
    choices = plain_json.get("choices") or []
    plain_message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    plain_answer = plain_message.get("content") if isinstance(plain_message, dict) else ""
    plain_answer = plain_answer if isinstance(plain_answer, str) else ""
    passed = plain.get("status") == 200 and not plain_json.get("error") and _has_answer(plain_answer, "42")
    all_passed &= _run_check(report, output, "plain_greedy_math", plain, passed,
                             f"answer contains 42: {plain_answer!r}" if passed else
                             f"expected answer containing 42; got {plain_answer!r}")

    stream_body = _chat_body([{"role": "user", "content": prompt}], max_tokens=16, stream=True)
    streamed = _request("POST", api + "/chat/completions", stream_body, stream=True)
    streamed_answer = streamed.get("response_text", "")
    stream_errors = streamed.get("stream_errors", [streamed.get("transport_error", "unknown stream error")])
    passed = (streamed.get("status") == 200 and streamed.get("saw_done") is True
              and not stream_errors and _has_answer(streamed_answer, "42"))
    all_passed &= _run_check(report, output, "streamed_greedy_math", streamed, passed,
                             f"SSE ended with [DONE], had no errors, and answer contains 42: {streamed_answer!r}"
                             if passed else f"SSE failure/errors={stream_errors!r}, answer={streamed_answer!r}")

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for the requested location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "enum": ["Stockholm"]}},
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }]
    tool_body = _chat_body(
        [{"role": "user", "content": "Use get_weather for Stockholm. Do not guess the weather."}],
        max_tokens=128, tools=tools, stream=True,
    )
    tool_result = _request("POST", api + "/chat/completions", tool_body, stream=True)
    assembled_calls = {}
    streamed_finish = None
    for event in tool_result.get('response_events', []):
        for choice in event.get('choices', []):
            if choice.get('finish_reason'):
                streamed_finish = choice['finish_reason']
            for delta_call in choice.get('delta', {}).get('tool_calls', []):
                slot = assembled_calls.setdefault(delta_call.get('index', 0), {
                    'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if delta_call.get('id'):
                    slot['id'] = delta_call['id']
                for key in ('name', 'arguments'):
                    slot['function'][key] += delta_call.get('function', {}).get(key, '')
    tool_json = {'choices': [{'finish_reason': streamed_finish, 'message': {
        'role': 'assistant', 'content': tool_result.get('response_text', ''),
        'tool_calls': [assembled_calls[i] for i in sorted(assembled_calls)]}}]}
    tool_choices = tool_json.get("choices") or []
    tool_choice = tool_choices[0] if tool_choices and isinstance(tool_choices[0], dict) else {}
    tool_message = tool_choice.get("message", {})
    returned_calls = tool_message.get("tool_calls", []) if isinstance(tool_message, dict) else []
    parsed_call = None
    validation_error = None
    if tool_result.get("status") == 200 and isinstance(returned_calls, list):
        matching = [
            call for call in returned_calls
            if isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == "get_weather"
        ]
        if len(matching) == 1:
            try:
                parsed_call = matching[0]
                args_obj = json.loads(parsed_call["function"].get("arguments", ""))
                if (not isinstance(args_obj, dict)
                        or args_obj.get("location") != "Stockholm"
                        or not parsed_call.get("id")):
                    validation_error = "function arguments/id did not satisfy the declared schema"
                else:
                    validation_error = None
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                validation_error = f"invalid function-call arguments: {exc}"
        else:
            validation_error = f"expected exactly one get_weather call, got {len(matching)}"
    else:
        validation_error = f"HTTP status {tool_result.get('status')} or malformed tool-call response"
    function_ok = (validation_error is None and tool_choice.get("finish_reason") == "tool_calls"
                   and tool_result.get('saw_done') and not tool_result.get('stream_errors'))
    if validation_error is None and not function_ok:
        validation_error = f"expected finish_reason=tool_calls; got {tool_choice.get('finish_reason')!r}"
    all_passed &= _run_check(report, output, "weather_function_call", tool_result, function_ok,
                             "one valid get_weather(location=Stockholm) call returned" if function_ok
                             else validation_error or "function call failed")

    if function_ok:
        call_id = parsed_call["id"]
        history = [
            {"role": "user", "content": "Use get_weather for Stockholm. Do not guess the weather."},
            tool_message,
            {"role": "tool", "tool_call_id": call_id, "content": WEATHER_FIXTURE},
            {"role": "user", "content": "Summarize the weather from the tool result."},
        ]
        follow_body = _chat_body(history, max_tokens=128, tools=tools, tool_choice="none")
        follow = _request("POST", api + "/chat/completions", follow_body)
        follow_json = _body(follow)
        follow_choices = follow_json.get("choices") or []
        follow_choice = follow_choices[0] if follow_choices and isinstance(follow_choices[0], dict) else {}
        follow_message = follow_choice.get("message", {})
        follow_content = follow_message.get("content", "") if isinstance(follow_message, dict) else ""
        follow_content = follow_content if isinstance(follow_content, str) else ""
        has_weather = _has_answer(follow_content, "18 C") and _has_answer(follow_content, "sunny")
        follow_ok = (follow.get("status") == 200 and not follow_json.get("error")
                     and has_weather and isinstance(follow_message, dict)
                     and not follow_message.get("tool_calls"))
        all_passed &= _run_check(report, output, "tool_result_consumed", follow, follow_ok,
                                 f"assistant used the local fixture without another tool call: {follow_content!r}"
                                 if follow_ok else f"tool-result follow-up failed: {follow_content!r}")
    else:
        skipped = {
            "method": "POST", "url": api + "/chat/completions", "request_body": None,
            "timeout_seconds": TIMEOUT, "elapsed_seconds": 0.0,
            "skipped": "function call did not validate; the follow-up is intentionally not attempted",
        }
        _run_check(report, output, "tool_result_consumed", skipped, False,
                   "skipped after invalid/missing function call")

    if all_passed:
        coding_report = output.with_name(output.stem + '_coding.json')
        before = time.perf_counter()
        child = subprocess.run([sys.executable, str(Path(__file__).with_name('validate_qwen_coding.py')),
            '--url', root, '--output', str(coding_report)], check=False)
        coding_ok = child.returncode == 0
        all_passed &= _run_check(report, output, 'generated_python_function',
            {'report_path': str(coding_report), 'elapsed_seconds': time.perf_counter()-before,
             'validator_returncode': child.returncode}, coding_ok,
            'generated function passed five behavior/input-preservation cases' if coding_ok
            else 'coding validator failed; see linked report')

    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["final_status"] = "passed" if all_passed else "failed"
    report["checks_passed"] = sum(1 for entry in report["checks"] if entry["passed"])
    report["checks_total"] = len(report["checks"])
    report["output_path"] = str(output)
    _save_report(output, report)
    print(f"Report: {output}", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
