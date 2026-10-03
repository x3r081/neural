"""Teacher-forced numerical comparison of Qwen expert/GDN execution paths.

This is an execution tool, not an automatic pass/fail benchmark. Run it only
after checkpoint integrity and payload checks have completed and GPU use is
authorized. It never writes to model files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path(r"F:\Models\Qwen3.6-35B-A3B")
DEFAULT_VERIFICATION = REPO_ROOT / "benchmarks/qwen36_20261002/checkpoint_verification.json"
DEFAULT_MODES = (
    "staged-torch",
    "native-legacy-torch",
    "native-grouped-torch",
    "native-grouped-fla",
)

PROMPTS = {
    "coding": (
        "Write a Python function `merge_intervals` that accepts a list of "
        "[start, end] pairs, sorts them by start, and merges overlapping "
        "intervals. Return the result and briefly explain the time complexity."
    ),
    "reasoning": (
        "A shop discounts an item by 20%, then applies a 10% tax to the "
        "discounted price. The final price is 88 dollars. What was the original "
        "price? Show the arithmetic and explain why the percentages are applied "
        "in that order."
    ),
}

CONTINUATIONS = {
    "coding": (
        "\n\ndef merge_intervals(intervals):\n"
        "    intervals = sorted(intervals)\n"
        "    merged = []\n"
        "    for start, end in intervals:\n"
        "        if not merged or start > merged[-1][1]:\n"
        "            merged.append([start, end])\n"
    ),
    "reasoning": (
        "Let the original price be x. After a 20% discount, the price is 0.8x. "
        "After 10% tax, the final price is 0.88x. Therefore x equals 100 dollars."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES,
        help="Execution paths to compare. staged-torch is required as reference.",
    )
    parser.add_argument("--tokens", type=int, default=16, help="Frozen continuation length (default: 16).")
    parser.add_argument("--gpu-expert-budget-gib", type=float, default=0.0)
    parser.add_argument("--gpu-headroom-gib", type=float, default=2.0)
    parser.add_argument(
        "--native-gpu-layers", type=int, default=0,
        help="Keep the final N MoE layers on CUDA in native modes; budget must fit every expert in those layers.",
    )
    parser.add_argument(
        "--checkpoint-verification-json", type=Path, default=DEFAULT_VERIFICATION,
        help="Prior verifier report whose per-shard hashes are recorded separately from arithmetic results.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional JSON report path; without this, print JSON to stdout.",
    )
    args = parser.parse_args()
    if args.tokens < 1 or args.tokens > 256:
        parser.error("--tokens must be between 1 and 256")
    if "staged-torch" not in args.modes:
        parser.error("--modes must include staged-torch as the reference path")
    args.modes = ["staged-torch"] + [mode for mode in args.modes if mode != "staged-torch"]
    return args


def _checkpoint_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_supplied_or_missing", "report_path": str(path)}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        shards = [f for f in report.get("files", []) if f.get("name", "").endswith(".safetensors")]
        hashes_match = bool(shards) and all(
            item.get("sha256")
            and item.get("sha256") == item.get("actual_sha256")
            and item.get("lfs_digest_verified") is True
            for item in shards
        )
        expected_identity = (
            report.get("repo_id") == "Qwen/Qwen3.6-35B-A3B"
            and report.get("revision") == "995ad96eacd98c81ed38be0c5b274b04031597b0"
        )
        return {
            "status": report.get("status", "unknown"),
            "report_path": str(path),
            "repo_id": report.get("repo_id"),
            "revision": report.get("revision"),
            "verified_identity": expected_identity,
            "safetensors_shard_count": len(shards),
            "all_safetensors_hashes_match": hashes_match,
            "interpretation": "Prior checkpoint-file integrity evidence only; separate from floating-point inference comparisons.",
        }
    except Exception as exc:
        return {"status": "unreadable", "report_path": str(path), "error": repr(exc)}


def _render_inputs(tokenizer, token_count: int) -> dict[str, dict[str, Any]]:
    rendered: dict[str, dict[str, Any]] = {}
    for name, prompt in PROMPTS.items():
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ids.numel() < 30 or ids.numel() > 80:
            raise ValueError(
                f"Rendered {name} prompt has {ids.numel()} tokens; expected approximately 30-80"
            )
        candidate = tokenizer(CONTINUATIONS[name], add_special_tokens=False)["input_ids"]
        if len(candidate) < token_count:
            raise ValueError(
                f"Frozen {name} continuation has only {len(candidate)} tokens; "
                f"requested {token_count}"
            )
        continuation = [int(v) for v in candidate[:token_count]]
        rendered[name] = {
            "prompt": prompt,
            "rendered_prompt": chat_text,
            "prompt_token_ids": [int(v) for v in ids.tolist()],
            "prompt_token_count": int(ids.numel()),
            "continuation_text_source": CONTINUATIONS[name],
            "continuation_token_ids": continuation,
            "continuation_decoded": tokenizer.decode(continuation, skip_special_tokens=False),
        }
    return rendered


def _cache_dtypes(cache: Any) -> list[dict[str, str]]:
    """Read dtype metadata for small cache attributes without copying tensors."""
    import torch

    found: list[dict[str, str]] = []

    def add(name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            found.append({"name": name, "dtype": str(value.dtype), "device": str(value.device)})
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, torch.Tensor):
                    found.append({"name": f"{name}.{key}", "dtype": str(item.dtype), "device": str(item.device)})
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    found.append({"name": f"{name}[{index}]", "dtype": str(item.dtype), "device": str(item.device)})

    if cache is None:
        return found
    for name, value in getattr(cache, "__dict__", {}).items():
        add(name, value)
    for index, layer in enumerate(getattr(cache, "layers", ())):
        for name, value in getattr(layer, "__dict__", {}).items():
            add(f"layers[{index}].{name}", value)
    # Keep only unique dtype slots; some caches expose the same list at both levels.
    unique = { (item["name"], item["dtype"], item["device"]): item for item in found }
    return list(unique.values())


def _run_teacher_forced(backend, prompt_ids: list[int], continuation: list[int]) -> tuple[list[Any], list[Any]]:
    """Capture prefill and per-token decode logits on CPU FP32 only."""
    import torch

    device = backend.device
    cache = None
    snapshots: list[Any] = []
    cache_dtype_slots: dict[str, Any] = {"after_prefill": [], "after_decode": []}
    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = backend.forward(
            input_ids,
            past_key_values=None,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = output.logits[:, -1, :].detach().to(device="cpu", dtype=torch.float32).contiguous()[0]
        cache = output.past_key_values
        snapshots.append(logits)
        cache_dtype_slots["after_prefill"] = _cache_dtypes(cache)
        del output, logits, input_ids

        for token_id in continuation:
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
            output = backend.forward(
                input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = output.logits[:, -1, :].detach().to(device="cpu", dtype=torch.float32).contiguous()[0]
            cache = output.past_key_values
            snapshots.append(logits)
            del output, logits, input_ids
        cache_dtype_slots["after_decode"] = _cache_dtypes(cache)
    del cache
    return snapshots, cache_dtype_slots


def _select_path(backend, mode: str) -> None:
    from qwen_neural.fast_delta import bind_fla_gdn, bind_torch_gdn
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    backend.expert_backend = "staged" if mode == "staged-torch" else "native"
    backend.native_dispatch = "legacy" if mode == "native-legacy-torch" else "grouped"
    if backend._gdn_binding is not None:
        backend._gdn_binding.close()
    gdn_mode = "fla" if mode.endswith("-fla") else "torch"
    backend._gdn_binding = (
        bind_fla_gdn(backend.model)
        if gdn_mode == "fla"
        else bind_torch_gdn(backend.model, modeling_module=modeling_qwen3_5_moe)
    )


def _compare(reference: list[Any], candidate: list[Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if len(reference) != len(candidate):
        raise ValueError("Path logits have different step counts")
    ref = torch.stack(reference, dim=0)
    test = torch.stack(candidate, dim=0)
    delta = test - ref
    abs_delta = delta.abs()
    ref_norm = torch.linalg.vector_norm(ref)
    test_norm = torch.linalg.vector_norm(test)
    ref_logp = F.log_softmax(ref, dim=-1)
    test_logp = F.log_softmax(test, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - test_logp)).sum(dim=-1)
    ref_ids = ref.argmax(dim=-1)
    test_ids = test.argmax(dim=-1)
    ref_top = ref.topk(2, dim=-1).values
    test_top = test.topk(2, dim=-1).values
    ref_gaps = ref_top[:, 0] - ref_top[:, 1]
    test_gaps = test_top[:, 0] - test_top[:, 1]
    q10 = _finite_float(torch.quantile(ref_gaps, 0.10).item())
    divergences = []
    per_step = []
    for index in range(ref.shape[0]):
        disagree = bool(ref_ids[index] != test_ids[index])
        low_margin = q10 is not None and math.isfinite(float(ref_gaps[index])) and bool(ref_gaps[index] <= q10)
        if disagree:
            divergences.append({
                "logit_step": index,
                "ref_top1_id": int(ref_ids[index]),
                "candidate_top1_id": int(test_ids[index]),
                "ref_top1_gap": _finite_float(ref_gaps[index]),
                "candidate_top1_gap": _finite_float(test_gaps[index]),
                "ref_gap_at_or_below_q10": low_margin,
            })
        per_step.append({
            "logit_step": index,
            "max_abs": _finite_float(abs_delta[index].max()),
            "rms": _finite_float(torch.sqrt(torch.mean(delta[index] * delta[index]))),
            "relative_l2": _finite_float(torch.linalg.vector_norm(delta[index]) / ref[index].norm().clamp_min(1e-30)),
            "kl_ref_to_candidate": _finite_float(kl[index]),
            "ref_top1_id": int(ref_ids[index]),
            "candidate_top1_id": int(test_ids[index]),
            "top1_agreement": not disagree,
            "ref_top1_gap": _finite_float(ref_gaps[index]),
            "candidate_top1_gap": _finite_float(test_gaps[index]),
            "ref_gap_at_or_below_q10": low_margin,
        })
    return {
        "finite": bool(torch.isfinite(ref).all() and torch.isfinite(test).all()),
        "max_abs": _finite_float(abs_delta.max()),
        "rms": _finite_float(torch.sqrt(torch.mean(delta * delta))),
        "relative_l2": _finite_float(torch.linalg.vector_norm(delta) / ref_norm.clamp_min(1e-30)),
        "reference_l2": _finite_float(ref_norm),
        "candidate_l2": _finite_float(test_norm),
        "mean_kl_ref_to_candidate": _finite_float(kl.mean()),
        "max_kl_ref_to_candidate": _finite_float(kl.max()),
        "top1_agreement_fraction": float((ref_ids == test_ids).float().mean()),
        "top1_disagreement_count": len(divergences),
        "reference_top1_gap_q10_descriptive_only": q10,
        "divergences": divergences,
        "per_step": per_step,
        "automatic_tolerance_or_quality_pass": None,
    }


def _finite_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    backend = None
    results: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "modes": list(args.modes),
        "tokens": args.tokens,
        "native_gpu_layers": args.native_gpu_layers,
        "checkpoint_integrity_evidence": _checkpoint_evidence(
            args.checkpoint_verification_json
            if args.checkpoint_verification_json.is_absolute()
            else REPO_ROOT / args.checkpoint_verification_json
        ),
        "arithmetic_note": (
            "Checkpoint-file integrity evidence is reported separately above. "
            "This tool does not modify checkpoint files; numerical differences "
            "below describe execution arithmetic only."
        ),
        "prompts": {},
        "interpretation": "Measurements only; no numeric tolerance or quality pass is assigned automatically.",
    }
    failure = None
    try:
        if not args.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")

        # Imports remain inside main so --help works without loading Torch/Transformers.
        import torch
        torch.set_num_threads(8)
        from qwen_neural.backend import QwenBackend

        print("[numerics] loading staged/Torch backend", file=sys.stderr, flush=True)
        backend = QwenBackend(
            args.model_dir,
            expert_backend="staged",
            gdn_backend="torch",
            native_dispatch="grouped",
            native_threads=8,
            native_gpu_layers=args.native_gpu_layers,
            gpu_expert_budget_gib=args.gpu_expert_budget_gib,
            gpu_headroom_gib=args.gpu_headroom_gib,
        )
        inputs = _render_inputs(backend.tokenizer, args.tokens)
        results["prompt_metadata"] = {
            name: item
            for name, item in inputs.items()
        }
        reference_mode = "staged-torch"
        for prompt_name, prompt in inputs.items():
            prompt_result: dict[str, Any] = {"comparisons_vs": reference_mode, "paths": {}}
            results["prompts"][prompt_name] = prompt_result
            reference_logits = None
            reference_cache_dtypes = None
            for mode in args.modes:
                print(f"[numerics] prompt={prompt_name} mode={mode} start", file=sys.stderr, flush=True)
                _select_path(backend, mode)
                backend.reset(clear_gpu_experts=True)
                logits, cache_dtypes = _run_teacher_forced(
                    backend,
                    prompt["prompt_token_ids"],
                    prompt["continuation_token_ids"],
                )
                memory = backend.memory_report()
                path_entry = {
                    "expert_backend": backend.expert_backend,
                    "native_dispatch": backend.native_dispatch if backend.expert_backend == "native" else None,
                    "native_gpu_layers": backend.native_gpu_layers,
                    "native_gpu_layer_start": backend.native_gpu_layer_start,
                    "native_gpu_required_bytes": backend.native_gpu_required_bytes,
                    "gdn_backend": "fla" if mode.endswith("-fla") else "torch",
                    "cache_tensor_dtypes": cache_dtypes,
                    "captured_logit_steps": len(logits),
                    "all_logits_finite": all(bool(torch.isfinite(row).all()) for row in logits),
                    "backend_counters": {
                        "forward_calls": memory["forward_calls"],
                        "input_tokens": memory["input_tokens"],
                        "gpu_expert_cache_hits": memory["gpu_expert_cache_hits"],
                        "gpu_expert_cache_misses": memory["gpu_expert_cache_misses"],
                        "cpu_expert_fetches": memory["store"].get("expert_reads"),
                    },
                }
                if mode == reference_mode:
                    reference_logits = logits
                    reference_cache_dtypes = cache_dtypes
                    path_entry["logit_capture"] = "reference"
                else:
                    if reference_logits is None:
                        raise RuntimeError("staged-torch must run before candidate modes")
                    path_entry["comparison"] = _compare(reference_logits, logits)
                    del logits
                prompt_result["paths"][mode] = path_entry
                print(f"[numerics] prompt={prompt_name} mode={mode} complete", file=sys.stderr, flush=True)
            prompt_result["reference_cache_tensor_dtypes"] = reference_cache_dtypes
            del reference_logits
    except Exception as exc:
        failure = exc
        results["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"[numerics] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                if failure is None:
                    failure = exc
                    results["error"] = {"type": type(exc).__name__, "message": str(exc), "during": "backend.close"}
                    print(f"[numerics] close FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    results["elapsed_s"] = time.perf_counter() - started
    results["status"] = "failed" if failure is not None else "complete"
    encoded = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Wrote numerical comparison report: {args.output}", file=sys.stderr, flush=True)
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
