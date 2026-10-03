"""Instrument a short teacher-forced Qwen decode for wall-time attribution.

This deliberately synchronizes CUDA before and after every measured module
call. The synchronization and Python wrappers perturb execution time; use the
report to locate expensive categories, not as an uninstrumented throughput
measurement or a quality evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _ModuleProfiler:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.enabled = False
        self.stats: dict[str, dict] = {}
        self._wrapped = []

    def sync(self):
        self.torch.cuda.synchronize(self.device)

    def instrument(self, owner, attribute: str, category: str, layer: int):
        original = getattr(owner, attribute)
        class_name = type(owner).__name__
        self._wrapped.append((owner, attribute, original))

        def timed_forward(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            self.sync()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.sync()
                elapsed = time.perf_counter() - start
                bucket = self.stats.setdefault(category, {"calls": 0, "seconds": 0.0, "layers": {}})
                bucket["calls"] += 1
                bucket["seconds"] += elapsed
                layer_entry = bucket["layers"].setdefault(str(layer), {
                    "calls": 0,
                    "seconds": 0.0,
                    "module_attribute": attribute,
                    "module_class": class_name,
                })
                layer_entry["calls"] += 1
                layer_entry["seconds"] += elapsed

        setattr(owner, attribute, timed_forward)

    def instrument_model(self, model, gpu_expert_layer_start: int):
        layers = model.model.layers
        for layer_index, layer in enumerate(layers):
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            if experts is not None and hasattr(experts, "forward"):
                category = "gpu_experts" if layer_index >= gpu_expert_layer_start else "cpu_experts"
                self.instrument(experts, "forward", category, layer_index)
            # Qwen3.5/3.6 layers expose exactly the active attention family as
            # self_attn (full attention) or linear_attn (Gated DeltaNet). Probe
            # the actual instantiated attributes/classes rather than infer from
            # a periodic layer pattern.
            for attribute, category in (("self_attn", "attention"), ("linear_attn", "deltanet")):
                module = getattr(layer, attribute, None)
                if module is not None and hasattr(module, "forward"):
                    self.instrument(module, "forward", category, layer_index)

    def restore(self):
        for owner, attribute, original in reversed(self._wrapped):
            setattr(owner, attribute, original)
        self._wrapped.clear()


def _write_report(path: Path, report: dict):
    report["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _forward(backend, torch, input_ids, past, *, timing: bool, profiler=None):
    if timing:
        profiler.sync()
        started = time.perf_counter()
    output = backend.forward(
        input_ids,
        past_key_values=past,
        use_cache=True,
        logits_to_keep=1,
    )
    if timing:
        profiler.sync()
        return output, time.perf_counter() - started
    return output, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=r"F:\Models\Qwen3.6-35B-A3B")
    parser.add_argument("--output", default="reports/qwen_decode_profile.json")
    parser.add_argument("--gdn-backend", choices=("torch", "fla"), default="torch")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.warmup_tokens < 0 or args.decode_tokens < 1 or args.threads < 1:
        parser.error("warmup-tokens must be >=0; decode-tokens and threads must be >=1")

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    report = {
        "profile": "Qwen3.6 short teacher-forced decode wall-time attribution",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "configuration": {
            "expert_backend": "native",
            "native_dispatch": "grouped",
            "native_gpu_layers": 2,
            "gpu_expert_cache_gib": 3.0,
            "gdn_backend": args.gdn_backend,
            "native_threads": args.threads,
            "warmup_decode_tokens": args.warmup_tokens,
            "measured_decode_tokens": args.decode_tokens,
        },
        "prompt": "What is 6 times 7? Reply with one number.",
        "thinking_enabled": False,
        "teacher_forcing": True,
        "profiler_perturbs_time": True,
        "profiler_caveat": (
            "CUDA is synchronized before and after every measured module and forward call. "
            "These wrappers serialize work and add host overhead; timings attribute costs and "
            "are not an uninstrumented throughput claim."
        ),
        "category_definition": {
            "cpu_experts": "whole native expert-module call for layers before the final two; includes input/device transfers, store reads, CPU compute, aggregation, and output transfer",
            "gpu_experts": "whole expert-module call for the final two CUDA-resident expert layers",
            "attention": "self_attn module calls",
            "deltanet": "linear_attn module calls",
            "remainder": "measured full-forward wall time minus the non-overlapping expert, attention, and DeltaNet module wall times",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "warmup": {"tokens": 0, "seconds": None},
        "measured": {"tokens": 0, "forward_seconds": [], "categories": {}},
        "final_status": "initializing",
    }
    _write_report(output_path, report)

    backend = None
    profiler = None
    try:
        import torch
        from qwen_neural.backend import QwenBackend
        from qwen_neural.generation import render_prompt

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen decode profiler requires CUDA")
        backend = QwenBackend(
            args.model_dir,
            expert_backend="native",
            gdn_backend=args.gdn_backend,
            gpu_expert_budget_gib=3.0,
            native_threads=args.threads,
            native_dispatch="grouped",
            native_gpu_layers=2,
        )
        device = backend.device
        properties = torch.cuda.get_device_properties(device)
        report["environment"].update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "model_layers": len(backend.model.model.layers),
            "cpu_expert_layer_range": [0, backend.native_gpu_layer_start],
            "gpu_expert_layer_range": [backend.native_gpu_layer_start, len(backend.model.model.layers)],
        })
        profiler = _ModuleProfiler(torch, device)
        profiler.instrument_model(backend.model, backend.native_gpu_layer_start)

        request = {
            "messages": [{"role": "user", "content": report["prompt"]}],
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        }
        prompt, thinking = render_prompt(backend.tokenizer, request)
        if thinking:
            raise RuntimeError("fixed profiler prompt unexpectedly enabled thinking")
        prompt_ids = backend.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise RuntimeError("tokenizer returned an empty fixed profiler prompt")
        report["prompt_tokens"] = len(prompt_ids)
        continuation = backend.tokenizer.encode(" 42", add_special_tokens=False)
        if not continuation:
            continuation = prompt_ids[-1:]
        decode_ids = [continuation[i % len(continuation)]
                      for i in range(args.warmup_tokens + args.decode_tokens)]

        # Prefill the short fixed prompt. This is timed separately and is not
        # included in decode-category totals.
        prefill_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        profiler.sync()
        prefill_start = time.perf_counter()
        out = backend.forward(prefill_input, past_key_values=None, use_cache=True, logits_to_keep=1)
        profiler.sync()
        report["prefill_seconds"] = time.perf_counter() - prefill_start
        past = out.past_key_values
        del prefill_input, out

        # Warm page cache, CPU code paths, CUDA kernels and the two-GPU-layer
        # LRU without collecting warmup measurements.
        profiler.enabled = False
        warm_start = time.perf_counter()
        for token_id in decode_ids[:args.warmup_tokens]:
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            out = backend.forward(token, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
        profiler.sync()
        report["warmup"] = {
            "tokens": args.warmup_tokens,
            "seconds": time.perf_counter() - warm_start,
            "teacher_forced_token_ids": decode_ids[:args.warmup_tokens],
        }
        if args.warmup_tokens:
            del out

        profiler.sync()
        profiler.enabled = True
        for token_id in decode_ids[args.warmup_tokens:]:
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            out, elapsed = _forward(backend, torch, token, past, timing=True, profiler=profiler)
            past = out.past_key_values
            report["measured"]["forward_seconds"].append(elapsed)
        profiler.sync()
        profiler.enabled = False
        report["measured"]["tokens"] = args.decode_tokens
        report["measured"]["categories"] = profiler.stats

        overall = sum(report["measured"]["forward_seconds"])
        expert_s = sum(profiler.stats.get(key, {}).get("seconds", 0.0)
                       for key in ("cpu_experts", "gpu_experts"))
        attention_s = profiler.stats.get("attention", {}).get("seconds", 0.0)
        deltanet_s = profiler.stats.get("deltanet", {}).get("seconds", 0.0)
        report["measured"].update({
            "overall_forward_seconds": overall,
            "overall_forward_ms_per_token": overall * 1000 / args.decode_tokens,
            "cpu_expert_seconds": profiler.stats.get("cpu_experts", {}).get("seconds", 0.0),
            "gpu_expert_seconds": profiler.stats.get("gpu_experts", {}).get("seconds", 0.0),
            "attention_seconds": attention_s,
            "deltanet_seconds": deltanet_s,
            "remainder_seconds": overall - expert_s - attention_s - deltanet_s,
            "remainder_definition": "overall minus disjoint timed module buckets; includes router/norm/MLP non-expert work and dispatch/runtime overhead",
        })
        report["final_status"] = "complete"
    except Exception as exc:
        report["final_status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if profiler is not None:
            profiler.enabled = False
            profiler.restore()
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(f"{type(exc).__name__}: {exc}")
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["output_path"] = str(output_path)
        _write_report(output_path, report)

    print(f"Profile status: {report['final_status']}; report: {output_path}", flush=True)
    return 0 if report["final_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
