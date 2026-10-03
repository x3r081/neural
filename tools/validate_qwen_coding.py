"""Bounded code-generation check against a running Neural Qwen HTTP server.

Does not start a server. Generated code is AST-restricted and is executed only
in a short-lived subprocess with a small builtin allowlist against fixed local
test data. The full response is saved whether validation passes or fails.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "qwen3.6-35b-a3b-neural"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/qwen36_20261002/coding_validation01.json"
PROMPT = """Write only one Python code block containing a function named `merge_intervals(intervals)`.
It must return a new list of sorted [start, end] pairs, merging overlapping and touching intervals.
Handle an empty list and nested intervals. Preserve the input list and its elements unchanged.
Output pairs must be new lists, with no aliases into the input.
Use no imports, no other top-level statements, and only ordinary Python list/loop/condition code."""

TEST_CASES = [
    {"input": [], "expected": []},
    {"input": [[5, 7], [1, 2], [9, 10]], "expected": [[1, 2], [5, 7], [9, 10]]},
    {"input": [[1, 3], [2, 6], [8, 10]], "expected": [[1, 6], [8, 10]]},
    {"input": [[1, 2], [2, 4], [5, 6]], "expected": [[1, 4], [5, 6]]},
    {"input": [[1, 10], [2, 3], [4, 8]], "expected": [[1, 10]]},
]

ALLOWED_BUILTINS = {"sorted", "list", "len", "range", "enumerate", "min", "max"}
ALLOWED_ANNOTATION_NAMES = {"int", "float", "str", "bool", "list", "tuple"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="Base URL of the already-running Neural API.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP request timeout in seconds.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def _post_chat(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("NEURAL_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    before = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        body = error.read()
        headers_out = dict(error.headers.items()) if error.headers else {}
        return {
            "status_code": int(error.code),
            "headers": headers_out,
            "body_bytes": body,
            "client_wall_s": time.perf_counter() - before,
        }
    with response:
        body = response.read()
        return {
            "status_code": int(response.status),
            "headers": dict(response.headers.items()),
            "body_bytes": body,
            "client_wall_s": time.perf_counter() - before,
        }


def _raw_response_record(response: dict[str, Any]) -> dict[str, Any]:
    body = response["body_bytes"]
    decoded = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        parsed = None
    return {
        "status_code": response["status_code"],
        "headers": response["headers"],
        "body_utf8": decoded,
        "body_base64": base64.b64encode(body).decode("ascii"),
        "json": parsed,
        "client_wall_s": response["client_wall_s"],
    }


def _response_content(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if response.get("status_code") != 200:
        raise ValueError(f"HTTP status {response.get('status_code')} is not 200")
    parsed = response.get("json")
    if not isinstance(parsed, dict) or parsed.get("object") != "chat.completion":
        raise ValueError("Response is not a chat.completion JSON object")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Response choices must contain exactly one object")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Response choice has no string message.content")
    if parsed.get("model") != MODEL_NAME:
        raise ValueError(f"Unexpected response model: {parsed.get('model')!r}")
    return message["content"], parsed


def _extract_code(content: str) -> str:
    fences = list(re.finditer(r"```([^\r\n`]*)\r?\n(.*?)```", content, flags=re.S))
    any_fence = "```" in content
    python_fences = [
        match for match in fences
        if match.group(1).strip().lower() in {"python", "py"}
    ]
    if any_fence:
        if len(fences) != 1 or len(python_fences) != 1:
            raise ValueError("Expected exactly one Python code fence")
        code = python_fences[0].group(2)
    else:
        code = content
    code = code.strip()
    if not code:
        raise ValueError("Emitted code is empty")
    return code + "\n"


class _RestrictedAst(ast.NodeVisitor):
    _simple_nodes = {
        ast.Module, ast.FunctionDef, ast.Lambda, ast.arguments, ast.arg, ast.Return,
        ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.If, ast.For,
        ast.While, ast.Break, ast.Continue, ast.Pass, ast.Name, ast.Load,
        ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Subscript,
        ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.IfExp, ast.ListComp, ast.comprehension, ast.keyword,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In,
        ast.NotIn,
    }

    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.function_depth = 0
        self.local_names = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.local_names.update(
            node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
        )
        self.annotation = False

    def generic_visit(self, node):
        if type(node) not in self._simple_nodes:
            raise ValueError(f"Disallowed Python syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.function_depth:
            raise ValueError("Nested function definitions are not allowed")
        if node.name != "merge_intervals":
            raise ValueError("Only the function named merge_intervals is allowed")
        if node.decorator_list or node.type_comment or node.args.defaults or node.args.kw_defaults:
            raise ValueError("Decorators, type comments, and default arguments are not allowed")
        if node.args.vararg or node.args.kwarg or node.args.kwonlyargs or node.args.posonlyargs:
            raise ValueError("Only a normal intervals parameter is allowed")
        if len(node.args.args) != 1 or node.args.args[0].arg != "intervals":
            raise ValueError("Function signature must be merge_intervals(intervals)")
        self.function_depth += 1
        try:
            self.annotation = True
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)
            self.annotation = False
            for statement in node.body:
                self.visit(statement)
        finally:
            self.annotation = False
            self.function_depth -= 1

    def visit_arg(self, node: ast.arg):
        if "__" in node.arg:
            raise ValueError("Dunder identifiers are disallowed")
        if node.annotation is not None:
            self.visit(node.annotation)

    def visit_Name(self, node: ast.Name):
        if "__" in node.id:
            raise ValueError("Dunder identifiers are disallowed")
        if node.id in self.local_names or node.id in ALLOWED_BUILTINS:
            return
        if node.id in ALLOWED_ANNOTATION_NAMES:
            return
        raise ValueError(f"Name is outside the local/builtin allowlist: {node.id}")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_BUILTINS:
                raise ValueError(f"Call is not allowed: {node.func.id}")
            self.visit(node.func)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in {"append", "extend"} or "__" in node.func.attr:
                raise ValueError("Only list append/extend method calls are allowed")
            if not isinstance(node.func.value, ast.Name) or node.func.value.id not in self.local_names:
                raise ValueError("append/extend may only be called on a local list variable")
            self.visit(node.func.value)
        else:
            raise ValueError("Only allowlisted builtin or list append/extend calls are allowed")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None or "__" in keyword.arg:
                raise ValueError("Keyword unpacking and dunder keywords are disallowed")
            self.visit(keyword.value)


def _validate_code(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Generated code does not parse: {exc}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Module must contain exactly one top-level function definition")
    validator = _RestrictedAst(tree)
    validator.visit(tree)


_SUBPROCESS_RUNNER = r'''
import json, sys
payload = json.load(sys.stdin)
cases = payload["cases"]
safe_builtins = {
    "sorted": sorted, "list": list, "len": len, "range": range,
    "enumerate": enumerate, "min": min, "max": max,
    "int": int, "float": float, "str": str, "bool": bool, "tuple": tuple,
}
scope = {"__builtins__": safe_builtins}
exec(compile(payload["code"], "<validated-generated-code>", "exec"), scope, scope)
function = scope.get("merge_intervals")
if not callable(function):
    raise AssertionError("merge_intervals definition missing")
observed = []
for case in cases:
    value = [item[:] for item in case["input"]]
    original = [item[:] for item in value]
    actual = function(value)
    if value != original:
        raise AssertionError("input was mutated")
    if actual is value or any(out is pair for out in actual for pair in value):
        raise AssertionError("result must contain new list objects, not aliases into the input")
    if actual != case["expected"]:
        raise AssertionError("wrong output: expected %r, got %r" % (case["expected"], actual))
    observed.append({"input": original, "actual": actual, "expected": case["expected"], "input_unchanged": True, "passed": True})
print(json.dumps({"status": "passed", "tests": observed}, ensure_ascii=False))
'''


def _execute_sandboxed(code: str, timeout: float = 2.0) -> dict[str, Any]:
    request = {"code": code, "cases": TEST_CASES}
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _SUBPROCESS_RUNNER],
        input=json.dumps(request, ensure_ascii=False),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        shell=False,
        cwd=str(REPO_ROOT),
    )
    result: dict[str, Any] = {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timeout_s": timeout,
    }
    if completed.returncode != 0:
        result["status"] = "failed"
        return result
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        result.update(status="failed", parse_error=str(exc))
        return result
    result["status"] = payload.get("status", "failed")
    result["tests"] = payload.get("tests", [])
    return result


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = _args()
    payload = {
        "model": MODEL_NAME,
        "endpoint": args.url.rstrip("/") + "/v1/chat/completions",
        "request": {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        },
    }
    report: dict[str, Any] = {
        "status": "running",
        "scope": "One bounded code-generation request; response is validated before isolated execution; this is not a model-quality suite.",
        "prompt": PROMPT,
        "request": payload,
        "test_cases": TEST_CASES,
    }
    try:
        response = _post_chat(args.url, payload['request'], args.timeout)
        report["raw_http_response"] = _raw_response_record(response)
        report["timing"] = {"client_wall_s": response["client_wall_s"]}
        try:
            content, parsed = _response_content(report["raw_http_response"])
            report["timing"]["server_neural"] = parsed.get("neural")
            report["usage"] = parsed.get("usage")
            report["finish_reason"] = parsed["choices"][0].get("finish_reason")
            report["emitted_content"] = content
            code = _extract_code(content)
            report["emitted_code"] = code
            _validate_code(code)
            report["ast_validation"] = {"status": "passed", "policy": "single merge_intervals function; no imports/dunders; allowlisted syntax and calls only"}
            report["sandbox_execution"] = _execute_sandboxed(code, timeout=2.0)
            report["status"] = "passed" if report["sandbox_execution"].get("status") == "passed" else "failed"
        except Exception as exc:
            report["status"] = "failed"
            report["validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        report["status"] = "failed"
        report["request_error"] = {"type": type(exc).__name__, "message": str(exc)}
    _write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
