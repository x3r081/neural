"""Capability detection and conservative, model-derived CUDA memory planning."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[1]
NATIVE_FLAGS = ("AVX2", "FMA3", "AVX512F", "AVX512BW", "AVX512VL", "AVX512DQ")


def cpu_features():
    try:
        from numpy._core._multiarray_umath import __cpu_features__
        return {name: bool(__cpu_features__.get(name, False)) for name in NATIVE_FLAGS}
    except (ImportError, AttributeError):
        return {name: False for name in NATIVE_FLAGS}


def native_kernel_status(features=None):
    """Only approve an explicitly built, hashed library for a compatible ISA."""
    features = cpu_features() if features is None else features
    manifest_path = ROOT / "artifacts" / "native_runtime_build.json"
    result = {"available": False, "reason": "no verified native build", "library": None}
    if not manifest_path.is_file():
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = manifest["cpu_features"]
        if required != list(NATIVE_FLAGS):
            raise ValueError("unrecognized native build ISA contract")
        library = ROOT / manifest["library"]
        if library.parent.resolve() != ROOT.resolve():
            raise ValueError("native library must be in the project root")
        if not all(features.get(flag, False) for flag in required):
            return dict(result, reason="CPU/OS lacks the native kernel instruction set")
        if not library.is_file():
            return dict(result, reason="native library is missing")
        if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("native library hash differs from its build manifest")
        # Hash and ISA checks do not prove that this machine can load the DLL:
        # dependent runtimes (for example OpenMP) may be absent, or the expected
        # entry point may not be exported. Reuse the runtime loader's cached
        # binding check so auto-selection falls back cleanly during preflight.
        from qwen_neural.cpu_experts import load_library
        load_library(library)
        return {"available": True, "reason": "verified ISA and library hash", "library": str(library)}
    except (OSError, AttributeError, ValueError, KeyError, TypeError) as exc:
        return dict(result, reason=str(exc))


def probe_hardware(device="cuda:0"):
    import psutil
    import torch

    flags = cpu_features()
    report = {
        "platform": platform.system(), "machine": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        "logical_cores": os.cpu_count() or 1,
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "cpu_features": flags, "native_kernel": native_kernel_status(flags),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": device,
    }
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("select one CUDA device as cuda:0, cuda:1, etc.")
    if report["cuda_available"]:
        index = int(device[5:])
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        report.update(gpu_name=props.name, gpu_free_bytes=free, gpu_total_bytes=total,
                      compute_capability=[props.major, props.minor],
                      bf16_supported=props.major >= 8)
    return report


def estimate_state_bytes(profile, context):
    """CALCULATED upper bound for supported attention and recurrent cache layouts."""
    cfg = profile["text_config"]
    layers = int(cfg["num_hidden_layers"])
    types = cfg.get("layer_types") or ["full_attention"] * layers
    kv_heads = int(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
    head_dim = int(cfg.get("head_dim", int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])))
    # The backend requires source expert and activation dtypes to agree. Use
    # source headers, not a potentially stale dtype hint in config.json.
    source_dtypes = profile.get("expert_dtypes") or profile.get("dtypes")
    if not source_dtypes:
        source_dtypes = [cfg.get("dtype", cfg.get("torch_dtype", profile.get("dtype", "bfloat16")))]
    itemsize = 4 if {str(x).upper() for x in source_dtypes} & {"FLOAT32", "TORCH.FLOAT32", "F32"} else 2
    total = 0
    for kind in types:
        if kind == "linear_attention":
            nk = int(cfg["linear_num_key_heads"])
            nv = int(cfg["linear_num_value_heads"])
            kd = int(cfg["linear_key_head_dim"])
            vd = int(cfg["linear_value_head_dim"])
            total += nv * kd * vd * 4  # FP32 DeltaNet recurrent state
            total += (2 * nk * kd + nv * vd) * int(cfg["linear_conv_kernel_dim"]) * itemsize
        else:
            total += 2 * context * kv_heads * head_dim * itemsize
    return total


def plan_runtime(profile, hardware, *, context=16384, prefix_cache=True,
                 expert_backend="auto", threads=None, gpu_budget_gib=None,
                 gpu_layers=None, headroom_gib=1.5, host_cache_gib=None):
    """Plan without allocating model tensors or changing their representation."""
    if isinstance(context, bool) or not isinstance(context, int) or context < 128:
        raise ValueError("context must be an integer >= 128")
    maximum = int(profile["text_config"].get("max_position_embeddings", context))
    if context > maximum:
        raise ValueError(f"context {context} exceeds the checkpoint's configured limit {maximum}")
    if not hardware.get("cuda_available"):
        raise RuntimeError("Neural serving requires a CUDA-enabled PyTorch installation and NVIDIA GPU")
    dtypes = {str(x).upper() for x in profile.get("dtypes", [profile.get("dtype", "BF16")])}
    if dtypes & {"BF16", "BFLOAT16", "TORCH.BFLOAT16"} and not hardware.get("bf16_supported"):
        raise RuntimeError("this BF16 checkpoint requires a BF16-capable CUDA GPU; weights will not be converted")
    if not math.isfinite(headroom_gib) or headroom_gib < 0.5:
        raise ValueError("GPU headroom must be finite and at least 0.5 GiB")
    native = hardware.get("native_kernel", {}).get("available", False)
    expert_dtypes = {str(x).upper() for x in profile.get("expert_dtypes", profile.get("dtypes", []))}
    native_model = bool(expert_dtypes) and expert_dtypes <= {"BF16", "BFLOAT16", "TORCH.BFLOAT16"} and profile["text_config"].get("hidden_act", "silu") == "silu"
    if expert_backend == "auto":
        expert_backend = "native" if native and native_model else "torch-cpu"
    if expert_backend not in {"native", "torch-cpu", "staged"}:
        raise ValueError("expert backend must be auto, native, torch-cpu, or staged")
    if expert_backend == "native" and not (native and native_model):
        raise RuntimeError("native BF16/SiLU execution is unavailable; use torch-cpu or build a compatible kernel")
    threads = int(hardware["physical_cores"]) if threads is None else threads
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    state_bytes = estimate_state_bytes(profile, context)
    core_bytes = int(profile["core_bytes"])
    # Two states during snapshot/working-copy overlap; transient expert transfers
    # occur before insertion into the cache and need their own explicit reserve.
    state_reserve = state_bytes * (2 if prefix_cache else 1)
    largest = int(profile.get("largest_expert_bytes", 0))
    layer_bytes = {int(k): int(v) for k, v in profile["layer_expert_bytes"].items()}
    if not largest and layer_bytes:
        largest = max(layer_bytes.values()) // int(profile["expert_count"])
    reserve = math.ceil(headroom_gib * GIB) + state_reserve + 2 * largest
    available = int(hardware["gpu_free_bytes"]) - core_bytes - reserve
    if available < max(largest, 0):
        raise MemoryError(
            f"model core/context do not fit currently free VRAM: core={core_bytes/GIB:.2f}, "
            f"state+workspace={reserve/GIB:.2f}, free={hardware['gpu_free_bytes']/GIB:.2f} GiB")
    if gpu_budget_gib is not None:
        if not math.isfinite(gpu_budget_gib) or gpu_budget_gib < 0:
            raise ValueError("GPU cache budget must be finite and nonnegative")
        requested = int(gpu_budget_gib * GIB)
        if requested > available:
            raise MemoryError("requested GPU expert budget exceeds the memory plan")
        available = requested
    ordered = sorted(layer_bytes, reverse=True)
    selected, used = [], 0
    for layer in ordered:
        if used + layer_bytes[layer] > available:
            break
        selected.append(layer)
        used += layer_bytes[layer]
    if gpu_layers is not None:
        if isinstance(gpu_layers, bool) or not isinstance(gpu_layers, int) or not 0 <= gpu_layers <= len(ordered):
            raise ValueError("GPU expert-layer count is outside the sparse-layer range")
        selected = ordered[:gpu_layers]
        used = sum(layer_bytes[layer] for layer in selected)
        if used > available:
            raise MemoryError("requested resident expert layers exceed the GPU budget")
    if host_cache_gib is None:
        host_cache_gib = min(1.0, hardware["ram_available_bytes"] / GIB * 0.05)
    if not math.isfinite(host_cache_gib) or host_cache_gib < 0:
        raise ValueError("host cache budget must be finite and nonnegative")
    return {
        "evidence_class": "CALCULATED_MEMORY_PLAN", "device": hardware["device"],
        "context": context, "threads": threads, "expert_backend": expert_backend,
        "core_bytes": core_bytes, "cache_state_bytes": state_bytes,
        "state_reserve_bytes": state_reserve, "workspace_headroom_bytes": reserve - state_reserve,
        "gpu_expert_budget_bytes": available, "gpu_expert_budget_gib": available / GIB,
        "native_gpu_layers": len(selected), "resident_sparse_layers": sorted(selected),
        "resident_expert_bytes": used, "host_expert_cache_gib": host_cache_gib,
        "prefix_cache": bool(prefix_cache), "prefix_cache_max_bytes": state_bytes,
        "native_library": hardware.get("native_kernel", {}).get("library") if expert_backend == "native" else None,
        "note": "capacity estimate, not a speed prediction; OS mmap residency is separate from the host tensor cache",
    }
