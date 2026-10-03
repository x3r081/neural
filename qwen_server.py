"""Local Qwen3.6 text server. GPT-OSS's server.py remains an inherited reference."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from qwen_neural.backend import QwenBackend
from qwen_neural.generation import GenerationEngine, render_prompt, split_response

MODEL_NAME = "qwen3.6-35b-a3b-neural"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"F:\Models\Qwen3.6-35B-A3B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--expert-backend", choices=["staged", "native"], default="native")
    ap.add_argument("--gdn-backend", choices=["torch", "fla"], default="torch")
    ap.add_argument("--native-gpu-layers", type=int, default=2)
    ap.add_argument("--gpu-expert-budget-gib", type=float, default=3.0)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--prefill-chunk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--native-workspace", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--recurrent-graph", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    backend = QwenBackend(args.model_dir, expert_backend=args.expert_backend,
                          gpu_expert_budget_gib=args.gpu_expert_budget_gib,
                          gdn_backend=args.gdn_backend,
                          native_gpu_layers=args.native_gpu_layers,
                          native_workspace=args.native_workspace,
                          recurrent_graph=args.recurrent_graph,
                          native_threads=args.threads)
    engine = GenerationEngine(backend, args.context, args.prefill_chunk)
    inference_lock = threading.Lock()
    key = os.environ.get("NEURAL_API_KEY")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            if key and self.headers.get("Authorization") != "Bearer " + key:
                self.respond({"error": {"message": "Unauthorized"}}, 401)
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path.rstrip("/") in {"", "/health", "/v1/health"}:
                return self.respond({"status": "ok", "model": MODEL_NAME,
                                     "precision": "original BF16", "context": args.context,
                                     "expert_backend": args.expert_backend,
                                     "gdn_backend": args.gdn_backend,
                                     "native_workspace": args.native_workspace,
                                     "recurrent_graph": args.recurrent_graph,
                                     "execution": backend.memory_report()})
            if self.path.rstrip("/") == "/v1/models":
                return self.respond({"object": "list", "data": [
                    {"id": MODEL_NAME, "object": "model", "owned_by": "local"}]})
            self.respond({"error": {"message": "Not found"}}, 404)

        def do_POST(self):
            if not self.authorized():
                return
            streaming = False
            locked = False
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 8*1024*1024:
                    raise ValueError("request body must be between 1 byte and 8 MiB")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("request body must be an object")
                for field in ("stream", "ignore_eos", "add_special"):
                    if field in request and not isinstance(request[field], bool):
                        raise ValueError(f"{field} must be a boolean")
                for field in ("max_tokens", "n_predict", "top_k", "seed", "n"):
                    if field in request and (isinstance(request[field], bool) or not isinstance(request[field], int)):
                        raise ValueError(f"{field} must be an integer")
                for field in ("temperature", "top_p"):
                    if field in request and (isinstance(request[field], bool) or not isinstance(request[field], (int, float))):
                        raise ValueError(f"{field} must be a number")
                path = self.path.rstrip("/")
                if path == "/tokenize":
                    return self.respond({"tokens": backend.tokenizer.encode(
                        request["content"], add_special_tokens=request.get("add_special", False))})
                if path not in {"/completion", "/v1/chat/completions"}:
                    return self.respond({"error": {"message": "Not found"}}, 404)
                chat = path == "/v1/chat/completions"
                if chat and request.get("model", MODEL_NAME) != MODEL_NAME:
                    raise ValueError(f"model must be {MODEL_NAME}")
                if request.get("n", 1) != 1:
                    raise ValueError("only n=1 is supported")
                if request.get("stop") or request.get("logit_bias") or request.get("response_format"):
                    raise ValueError("stop, logit_bias and response_format are not implemented")
                if request.get("tool_choice", "auto") not in ("auto", "none"):
                    raise ValueError("only automatic or disabled tool selection is supported")
                if request.get("tool_choice") == "none":
                    request.pop("tools", None)
                prompt, thinking = render_prompt(backend.tokenizer, request) if chat else (request["prompt"], False)
                locked = inference_lock.acquire(blocking=False)
                if not locked:
                    return self.respond({"error": {"message": "Server is processing another request"}}, 409)
                request_id = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                already = {"reasoning_content": "", "content": ""}

                def emit(delta=None, finish=None, usage=None, metrics=None):
                    packet = {"id": request_id, "object": "chat.completion.chunk",
                              "created": created, "model": MODEL_NAME,
                              "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
                    if usage is not None:
                        packet["usage"] = usage
                    if metrics is not None:
                        packet["neural"] = metrics
                    self.wfile.write(("data: " + json.dumps(packet, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()

                def on_text(raw):
                    reason, content, _ = split_response(raw, thinking, request.get("tools"))
                    delta = {}
                    for field, text in (("reasoning_content", reason), ("content", content)):
                        if text.startswith(already[field]) and len(text) > len(already[field]):
                            delta[field] = text[len(already[field]):]
                            already[field] = text
                    if delta:
                        emit(delta)

                streaming = chat and bool(request.get("stream"))
                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    emit({"role": "assistant"})
                result = engine.generate(
                    prompt, max_tokens=int(request.get("max_tokens", request.get("n_predict", 256))),
                    temperature=float(request.get("temperature", 1.0 if chat else 0.0)),
                    top_k=int(request.get("top_k", 20)), top_p=float(request.get("top_p", 0.95)),
                    seed=int(request.get("seed", 0)), ignore_eos=bool(request.get("ignore_eos", False)),
                    on_text=on_text if streaming else None)
                if not chat:
                    return self.respond(result)
                reason, content, calls = split_response(result["content"], thinking, request.get("tools"), final=True)
                finish = "tool_calls" if calls else result["finish_reason"]
                usage = {"prompt_tokens": len(result["prompt_tokens"]),
                         "completion_tokens": len(result["tokens"]),
                         "total_tokens": len(result["prompt_tokens"])+len(result["tokens"])}
                metrics = dict(result["timings"], decode_tok_s=result["timings"]["predicted_per_second"],
                               prefill_s=result["timings"]["prompt_ms"]/1000)
                if streaming:
                    if calls:
                        emit({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                    emit(finish=finish, usage=usage, metrics=metrics)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    message = {"role": "assistant", "content": content, "reasoning_content": reason}
                    if calls:
                        message["tool_calls"] = calls
                    self.respond({"id": request_id, "object": "chat.completion", "created": created,
                                  "model": MODEL_NAME, "choices": [{"index": 0, "message": message,
                                  "finish_reason": finish}], "usage": usage, "neural": metrics})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                traceback.print_exc()
                error = {"error": {"message": str(exc), "type": type(exc).__name__}}
                if streaming:
                    try:
                        self.wfile.write(("data: " + json.dumps(error) + "\n\ndata: [DONE]\n\n").encode())
                        self.wfile.flush()
                    except OSError:
                        pass
                else:
                    self.respond(error, 400 if isinstance(exc, (ValueError, KeyError)) else 500)
            finally:
                if locked:
                    inference_lock.release()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"READY: {MODEL_NAME} at http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
