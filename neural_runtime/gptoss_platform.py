"""GPT-OSS platform preflight and metadata-derived memory planning.

This describes the Windows/CUDA fused engine with either the inherited AVX-512
CPU kernel or a separately validated portable CPU fallback. CPU-only and
non-Windows GPT-OSS serving are not implemented here.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform

GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GPTOSS_ISA = ("AVX2", "FMA3", "AVX512F", "AVX512BW", "AVX512VL", "AVX512DQ")
REQUIRED_NATIVE_SYMBOLS = {
    "gptoss_experts", "gptoss_experts_cap", "gptoss_set_scale_layout",
    "gptoss_get_scale_layout", "gptoss_slot_bytes",
}
REQUIRED_PERSISTENT_SYMBOLS = REQUIRED_NATIVE_SYMBOLS | {
    "gptoss_set_persistent", "gptoss_persistent_shutdown", "gptoss_persistent_status",
    "gptoss_set_fuse",
    "gptoss_team_probe_batch",
}
REQUIRED_PREFILL_SYMBOLS = {
    "gptoss_experts", "gptoss_experts_cap", "gptoss_set_scale_layout",
    "gptoss_get_scale_layout", "gptoss_slot_bytes", "gptoss_experts_multi",
    "gptoss_multi_scratch_floats", "gptoss_combine_pairs",
    "gptoss_multi_set_tiling", "gptoss_multi_set_schedule", "gptoss_multi_get_config",
    "gptoss_multi_activation", "gptoss_multi_check_exp",
}
DEFAULT_NATIVE_MANIFEST = ROOT / "artifacts" / "gptoss_native_build.json"


def _cpu_features() -> dict[str, bool]:
    """Read OS-enabled SIMD features without executing any native kernel."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except (ImportError, AttributeError):
        return {name: False for name in REQUIRED_GPTOSS_ISA}
    return {name: bool(__cpu_features__.get(name, False)) for name in REQUIRED_GPTOSS_ISA}


def _cpu_signature() -> str:
    return "|".join((platform.system(), platform.machine(), platform.processor().strip()))


def _project_file(name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("manifest file path is empty")
    path = (ROOT / name).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("manifest file must remain inside the project directory") from exc
    return path


def _verified_project_library(name: str, expected_hash: str) -> tuple[Path, str]:
    path = _project_file(name)
    if not path.is_file():
        raise FileNotFoundError(f"native DLL is missing: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash.lower() != str(expected_hash).lower():
        raise ValueError(f"native DLL SHA-256 differs from its build manifest: {path.name}")
    return path, actual_hash


def _native_library_status(os_name: str, features: dict[str, bool], manifest_path: Path,
                           cpu_signature: str):
    result = {"available": False, "reason": "no verified GPT-OSS native build manifest", "library": None,
              "sha256": None}
    if os_name != "Windows":
        return dict(result, reason="the current GPT-OSS native DLL is Windows-only")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kernel_kind = manifest.get("kernel_kind", "avx512-persistent")
        if kernel_kind not in {"avx512-persistent", "portable-avx2-fma", "portable-scalar"}:
            raise ValueError(f"unsupported GPT-OSS kernel kind {kernel_kind!r}")
        required = manifest["cpu_features"]
        if not isinstance(required, list) or any(name not in REQUIRED_GPTOSS_ISA for name in required):
            raise ValueError("native manifest cpu_features must be an explicit known ISA feature list")
        if kernel_kind == "avx512-persistent" and not set(REQUIRED_GPTOSS_ISA).issubset(required):
            raise ValueError("AVX-512 persistent manifest omits required AVX2/FMA3/AVX-512 features")
        if kernel_kind == "portable-avx2-fma" and not {"AVX2", "FMA3"}.issubset(required):
            raise ValueError("AVX2 portable manifest must require AVX2 and FMA3")
        if kernel_kind == "portable-scalar" and required:
            raise ValueError("scalar portable manifest must not require SIMD features")
        if kernel_kind == "avx512-persistent" and manifest.get("cpu_signature") != cpu_signature:
            raise ValueError("native DLL is approved for a different CPU signature")
        if kernel_kind == "avx512-persistent" and not manifest.get("model_id"):
            raise ValueError("AVX-512 native manifest must restrict the DLL to a model identity")
        missing = [flag for flag in required if not features.get(flag, False)]
        if missing:
            # Do not load a DLL until all required CPU/OS instruction bits pass;
            # CRT initialization may execute compiled instructions.
            return dict(result, reason="CPU/OS lacks required native ISA: " + ", ".join(missing))
        if kernel_kind.startswith("portable-"):
            exact = manifest.get("exact_validation")
            if not isinstance(exact, dict) or exact.get("status") != "PASS":
                raise ValueError("portable native manifest requires passing exact-validation evidence")
            validation_path = _project_file(exact.get("report"))
            validation_hash = exact.get("report_sha256")
            if not validation_hash:
                raise ValueError("portable native manifest requires a hashed parity-validation report")
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            actual_validation_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
            if actual_validation_hash.lower() != str(validation_hash).lower():
                raise ValueError("portable parity-validation report SHA-256 mismatch")
            report_library_hash = validation.get("portable_dll_sha256")
            if not report_library_hash or str(report_library_hash).lower() != str(manifest["sha256"]).lower():
                raise ValueError("portable parity report does not identify the candidate DLL SHA-256")
            if (validation.get("result") != "PASS" or validation.get("mode") != "full"
                    or validation.get("full_validation") is not True
                    or validation.get("full_validation_passed") is not True):
                raise ValueError("portable parity-validation report must be a passing full validation")
            thread_counts = validation.get("thread_counts_tested", [])
            parallel_validated = (
                validation.get("openmp_transient_regions") is True
                and validation.get("persistent_worker_pool") is False
                and isinstance(thread_counts, list)
                and {1, 8}.issubset({n for n in thread_counts if isinstance(n, int) and not isinstance(n, bool)})
            )
        else:
            parallel_validated = True

        library, actual_hash = _verified_project_library(manifest["library"], manifest["sha256"])
        prefill_path, prefill_hash = _verified_project_library(
            manifest.get("prefill_library", manifest["library"]),
            manifest.get("prefill_sha256", manifest["sha256"]))
        if kernel_kind.startswith("portable-") and (
                library.resolve() != prefill_path.resolve() or actual_hash != prefill_hash):
            raise ValueError("portable validation covers one joint decode/prefill DLL; paths and hashes must match")
        # Safe only after the hard ISA gate above. Loading checks dependent DLLs
        # and exported ABI symbols; no expert function is called here.
        dll = ctypes.WinDLL(str(library))
        extra_symbols = manifest.get("symbols", [])
        if not isinstance(extra_symbols, list) or not all(isinstance(x, str) for x in extra_symbols):
            raise ValueError("native manifest symbols must be a list of symbol names")
        required_symbols = sorted((REQUIRED_PERSISTENT_SYMBOLS if kernel_kind == "avx512-persistent"
                                   else REQUIRED_NATIVE_SYMBOLS) | set(extra_symbols))
        missing_symbols = [name for name in required_symbols if not hasattr(dll, name)]
        if missing_symbols:
            raise AttributeError("native DLL lacks required symbols: " + ", ".join(missing_symbols))
        prefill_dll = ctypes.WinDLL(str(prefill_path))
        extra_prefill_symbols = manifest.get("prefill_symbols", [])
        if not isinstance(extra_prefill_symbols, list) or not all(isinstance(x, str) for x in extra_prefill_symbols):
            raise ValueError("native manifest prefill_symbols must be a list of symbol names")
        prefill_symbols = sorted(REQUIRED_PREFILL_SYMBOLS | set(extra_prefill_symbols))
        missing_symbols = [name for name in prefill_symbols if not hasattr(prefill_dll, name)]
        if missing_symbols:
            raise AttributeError("prefill DLL lacks required symbols: " + ", ".join(missing_symbols))
        del dll, prefill_dll
        persistent = kernel_kind == "avx512-persistent"
        return {"available": True, "reason": "verified ISA, CPU/model scope, SHA-256, loadability and symbols",
                "kernel_kind": kernel_kind, "library": str(library), "sha256": actual_hash,
                "prefill_library": str(prefill_path), "prefill_sha256": prefill_hash,
                "cpu_signature": cpu_signature, "cpu_features": list(required),
                "model_id": manifest.get("model_id"),
                "persistent_supported": persistent,
                "fused_cpu_kernel_supported": persistent,
                "parallel_supported": parallel_validated}
    except (OSError, AttributeError, ValueError, KeyError, TypeError) as exc:
        return dict(result, reason=str(exc))


def probe_hardware(device: str = "cuda:0", *, native_manifest: str | os.PathLike | None = None) -> dict:
    """Return a capability snapshot; never executes the GPT-OSS expert DLL."""
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("device must be a CUDA device name such as cuda:0")
    os_name = platform.system()
    features = _cpu_features()
    report = {
        "platform": os_name,
        "machine": platform.machine(),
        "cpu_signature": _cpu_signature(),
        "cpu_features": features,
        "physical_cores": os.cpu_count() or 1,
        "logical_cores": os.cpu_count() or 1,
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "cuda_available": False,
        "device": device,
        "gpu_name": None,
        "gpu_total_bytes": None,
        "gpu_free_bytes": None,
        "gpu_memory_clock_rate_khz": None,
        "gpu_memory_bus_width_bits": None,
        "gpu_is_integrated": None,
        "compute_capability": None,
        "bf16_supported": False,
    }
    try:
        import psutil
        report["physical_cores"] = psutil.cpu_count(logical=False) or report["logical_cores"]
        mem = psutil.virtual_memory()
        report["ram_total_bytes"] = int(mem.total)
        report["ram_available_bytes"] = int(mem.available)
    except ImportError:
        pass
    try:
        import torch
        report["torch_version"] = str(torch.__version__)
        report["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            index = int(device[5:])
            if index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {index} is not available")
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            report.update(cuda_available=True, gpu_name=props.name,
                          gpu_free_bytes=int(free), gpu_total_bytes=int(total),
                          compute_capability=[int(props.major), int(props.minor)],
                          bf16_supported=bool(props.major >= 8))
            for attr, key in (("memory_clock_rate", "gpu_memory_clock_rate_khz"),
                              ("memory_bus_width", "gpu_memory_bus_width_bits"),
                              ("is_integrated", "gpu_is_integrated")):
                value = getattr(props, attr, None)
                if value is not None:
                    report[key] = int(value) if key != "gpu_is_integrated" else bool(value)
    except ImportError as exc:
        report["torch_error"] = str(exc)
    manifest_path = Path(native_manifest) if native_manifest else DEFAULT_NATIVE_MANIFEST
    report["native_kernel"] = _native_library_status(os_name, features, manifest_path,
                                                     report["cpu_signature"])
    report["gpu_graphs_supported"] = bool(
        os_name == "Windows" and report["cuda_available"] and report["bf16_supported"]
    )
    # Preserve the older field as an alias for the GPU graph capability. CPU
    # kernel capability is reported separately and may be portable/nonpersistent.
    report["gptoss_fused_supported"] = report["gpu_graphs_supported"]
    native = report["native_kernel"]
    report["cpu_acceleration"] = {
        "available": bool(native.get("available")),
        "kernel_kind": native.get("kernel_kind"),
        "persistent_team_supported": bool(native.get("persistent_supported", False)),
        "fused_cpu_kernel_supported": bool(native.get("fused_cpu_kernel_supported", False)),
        "parallel_supported": bool(native.get("parallel_supported", False)),
    }
    report["gptoss_full_fastpath_supported"] = bool(
        report["gpu_graphs_supported"] and report["cpu_acceleration"]["available"]
    )
    if report["gptoss_full_fastpath_supported"]:
        report["fallback_requirement"] = None
    else:
        report["fallback_requirement"] = (
            "A separately validated GPT-OSS CPU kernel and/or Windows CUDA/BF16 graph capability is required; "
            "no unvalidated fallback is selected. See cpu_acceleration and gpu_graphs_supported for the "
            "capabilities that are missing."
        )
    return report


def _spec_dict(spec) -> dict:
    if hasattr(spec, "to_dict") and callable(spec.to_dict):
        spec = spec.to_dict()
    if not isinstance(spec, dict):
        raise TypeError("spec must be a metadata dictionary or expose to_dict()")
    return spec


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"spec {name} must be a positive integer")
    return value


def _core_bytes(spec: dict) -> int:
    memory = spec.get("memory", {})
    # The model spec reports source_core_bytes including CPU-resident token
    # embeddings and gpu_core_bytes for the current placement. Budget only the
    # latter; count CPU embeddings against host RAM, never against VRAM.
    value = memory.get("gpu_core_bytes") if isinstance(memory, dict) else None
    if value is None:
        value = spec.get("core_bytes")
    return _positive_int(value, "memory.core_bytes")


def _unsupported(reason: str) -> dict:
    return {"supported": False, "status": "unsupported", "reason": reason,
            "fallback_requirement": "An explicit, validated GPT-OSS fallback is required; no unvalidated fallback is auto-enabled.",
            "evidence_class": "CAPABILITY_PREFLIGHT"}


def plan_gptoss(spec, hardware: dict, context: int = 16384, *, pool_gib: float | None = None,
                kv_ring: int | None = None, threads: int | None = None,
                static_slots: int | None = None, capbufs: int | None = None,
                scratch_slots: int = 6, headroom_gib: float = 1.5,
                target_pool_gib: float = 6.05) -> dict:
    """Plan the validated fused GPT-OSS route from checkpoint/store metadata.

    User-specified capacities are strict requests and fail if they do not fit.
    The default pool is the proven profile ceiling clamped to actual free VRAM.
    """
    s = _spec_dict(spec)
    if not isinstance(hardware, dict):
        raise TypeError("hardware must be the result of probe_hardware or a compatible snapshot")
    geometry = s.get("geometry")
    store = s.get("store")
    if not isinstance(geometry, dict) or not isinstance(store, dict):
        raise ValueError("spec must include geometry and store metadata")
    if s.get("model_type") != "gpt_oss":
        return _unsupported("the GPT-OSS planner only accepts model_type=gpt_oss")
    layers = _positive_int(geometry.get("layers"), "geometry.layers")
    experts = _positive_int(geometry.get("experts"), "geometry.experts")
    top_k = _positive_int(geometry.get("top_k"), "geometry.top_k")
    hidden = _positive_int(geometry.get("hidden"), "geometry.hidden")
    kv_heads = _positive_int(geometry.get("num_kv_heads"), "geometry.num_kv_heads")
    head_dim = _positive_int(geometry.get("head_dim"), "geometry.head_dim")
    layer_types = geometry.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise ValueError("geometry.layer_types must describe every layer")
    slot_bytes = _positive_int(store.get("slot_bytes"), "store.slot_bytes")
    store_format = str(store.get("format", "")).lower()
    source_format = s.get("source_format")
    if not isinstance(source_format, dict):
        raise ValueError("source_format must be the verified metadata object")
    if ("mxfp4" not in store_format
            or str(source_format.get("quant_method", "")).lower() != "mxfp4"
            or source_format.get("weight_preserving") is not True):
        return _unsupported("GPT-OSS source/store format is not verified MXFP4; refusing conversion or guessing")
    layout = (store.get("metadata") or {}).get("layout", {}) if isinstance(store.get("metadata"), dict) else {}
    intermediate = _positive_int(geometry.get("intermediate"), "geometry.intermediate")
    for field, expected in (("n_layers", layers), ("n_experts", experts), ("top_k", top_k),
                            ("hidden", hidden), ("down_k", intermediate),
                            ("gate_up_n", 2 * intermediate), ("down_n", hidden),
                            ("slot_bytes", slot_bytes)):
        actual = layout.get(field)
        if actual is not None and actual != expected:
            raise ValueError(f"store metadata {field}={actual} disagrees with checkpoint geometry {expected}")
    declared_repr = layout.get("weight_repr")
    if declared_repr is not None and declared_repr != store.get("format"):
        raise ValueError("store metadata weight_repr disagrees with store format")
    if top_k > experts:
        raise ValueError("top_k cannot exceed expert count")
    maximum = (s.get("context_limits") or {}).get("max_position_embeddings")
    if isinstance(context, bool) or not isinstance(context, int) or context < 128:
        raise ValueError("context must be an integer >= 128")
    if maximum is not None and context > int(maximum):
        raise ValueError(f"context {context} exceeds checkpoint limit {maximum}")
    if (isinstance(headroom_gib, bool) or not isinstance(headroom_gib, (int, float))
            or not math.isfinite(headroom_gib) or headroom_gib < 0.5):
        raise ValueError("headroom_gib must be finite and at least 0.5")
    if (isinstance(target_pool_gib, bool) or not isinstance(target_pool_gib, (int, float))
            or not math.isfinite(target_pool_gib) or target_pool_gib <= 0):
        raise ValueError("target_pool_gib must be a finite positive number")
    if hardware.get("platform") != "Windows":
        return _unsupported("the current fused GPT-OSS server uses Windows APIs and Windows-built DLLs")
    kernel = hardware.get("native_kernel", {})
    kernel_kind = kernel.get("kernel_kind")
    supported_kernels = {"avx512-persistent", "portable-avx2-fma", "portable-scalar"}
    if kernel_kind not in supported_kernels:
        return _unsupported("native manifest does not identify a supported verified kernel kind")
    required_isa = kernel.get("cpu_features", [])
    if kernel_kind == "avx512-persistent":
        required_isa = REQUIRED_GPTOSS_ISA
    missing_isa = [name for name in required_isa if not hardware.get("cpu_features", {}).get(name, False)]
    if missing_isa:
        return _unsupported("CPU/OS lacks the verified kernel ISA: " + ", ".join(missing_isa))
    if not hardware.get("cuda_available") or not hardware.get("bf16_supported"):
        return _unsupported("a CUDA device with BF16 support is required by the fused GPT-OSS runtime")
    if not hardware.get("native_kernel", {}).get("available"):
        return _unsupported("the matching GPT-OSS native DLL failed hash/loadability/symbol validation")
    native_model_id = hardware["native_kernel"].get("model_id")
    if native_model_id is not None and native_model_id != s.get("model_id"):
        return _unsupported("the verified native DLL manifest restricts it to a different model identity")
    core_bytes = _core_bytes(s)
    gpu_free = _positive_int(hardware.get("gpu_free_bytes"), "hardware.gpu_free_bytes")
    ram_free = _positive_int(hardware.get("ram_available_bytes"), "hardware.ram_available_bytes")
    if isinstance(scratch_slots, bool) or not isinstance(scratch_slots, int) or scratch_slots < 0:
        raise ValueError("scratch_slots must be a nonnegative integer")
    sliding = {"sliding_attention", "sliding"}
    if kv_ring is None:
        kv_ring = min(context, max(_positive_int(geometry.get("window"), "geometry.window"), 256))
    if isinstance(kv_ring, bool) or not isinstance(kv_ring, int) or kv_ring < 1:
        raise ValueError("kv_ring must be a positive integer")
    if kv_ring > context:
        raise ValueError("kv_ring cannot exceed context")
    window = geometry.get("window")
    if window is not None and kv_ring < int(window):
        raise ValueError("kv_ring cannot be smaller than the checkpoint's sliding window")
    memory = s.get("memory") or {}
    if not isinstance(memory, dict):
        raise ValueError("spec memory must be an object")
    source_dtype = str(source_format.get("source_dtype", "bfloat16")).lower()
    if source_dtype in {"bfloat16", "bf16", "float16", "fp16"}:
        inferred_kv_bytes = 2
    elif source_dtype in {"float32", "fp32"}:
        inferred_kv_bytes = 4
    else:
        raise ValueError(f"unsupported GPT-OSS source dtype for KV planning: {source_dtype!r}")
    kv_bytes_per_element = int(memory.get("kv_bytes_per_element", inferred_kv_bytes))
    if kv_bytes_per_element not in (2, 4):
        raise ValueError("memory.kv_bytes_per_element must be 2 or 4")
    kv_tokens = sum(kv_ring if str(kind).lower() in sliding else context for kind in layer_types)
    kv_bytes = kv_tokens * 2 * kv_heads * head_dim * kv_bytes_per_element
    headroom = math.ceil(float(headroom_gib) * GIB)
    max_pool_bytes = gpu_free - core_bytes - kv_bytes - headroom
    if max_pool_bytes < slot_bytes * (scratch_slots + 1):
        return _unsupported("current free VRAM cannot fit model core, requested context/KV, headroom and a minimal slot pool")
    max_slots = max_pool_bytes // slot_bytes
    target_bytes = math.floor(float(target_pool_gib) * GIB)
    if pool_gib is not None:
        if isinstance(pool_gib, bool) or not isinstance(pool_gib, (int, float)) or not math.isfinite(pool_gib) or pool_gib <= 0:
            raise ValueError("pool_gib must be a finite positive number")
        requested_bytes = math.floor(float(pool_gib) * GIB)
        if requested_bytes > max_pool_bytes:
            raise MemoryError("requested pool exceeds free VRAM after core, KV and headroom reserves")
    else:
        requested_bytes = min(target_bytes, max_pool_bytes)
    pool_slots = requested_bytes // slot_bytes
    if pool_slots <= scratch_slots:
        return _unsupported("pool budget leaves no decode slot after scratch reservation")
    actual_pool_bytes = pool_slots * slot_bytes
    usable_slots = pool_slots - scratch_slots
    if static_slots is None:
        static_slots = min(250, usable_slots)
    if isinstance(static_slots, bool) or not isinstance(static_slots, int) or not 0 <= static_slots <= usable_slots:
        raise ValueError("static_slots must fit within pool slots after scratch reservation")
    if static_slots > layers * experts:
        raise ValueError("static_slots cannot exceed the checkpoint's total expert count")
    if threads is None:
        parallel = bool(kernel.get("parallel_supported", kernel_kind == "avx512-persistent"))
        threads = (min(8, _positive_int(hardware.get("physical_cores"), "hardware.physical_cores"))
                   if parallel else 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if threads > _positive_int(hardware.get("logical_cores"), "hardware.logical_cores"):
        raise ValueError("threads cannot exceed logical CPU count")
    max_capture_buffers = max(0, (ram_free - math.ceil(1.5 * GIB)) // slot_bytes)
    if capbufs is None:
        capbufs = min(16, int(max_capture_buffers))
    if isinstance(capbufs, bool) or not isinstance(capbufs, int) or capbufs < 2:
        return _unsupported("insufficient RAM for the minimum two capture buffers after the warm-page margin")
    if capbufs > max_capture_buffers:
        raise MemoryError("capture buffers exceed available RAM after the warm-page margin")
    return {
        "supported": True, "status": "ready", "evidence_class": "CALCULATED_CAPACITY_PLAN",
        "model_id": s.get("model_id"), "model_type": s.get("model_type"),
        "source_format": s["source_format"], "store_format": store["format"],
        "store_slot_bytes": slot_bytes, "layers": layers, "experts_per_layer": experts,
        "top_k": top_k, "context": context, "kv_ring": kv_ring, "kv_bytes": kv_bytes,
        "core_bytes": core_bytes, "headroom_bytes": headroom,
        "gpu_free_bytes": gpu_free, "pool_bytes": actual_pool_bytes,
        "pool_gib": actual_pool_bytes / GIB, "pool_slots": pool_slots,
        "pool_capacity_slots_at_current_free_vram": int(max_slots),
        "scratch_slots": scratch_slots, "usable_slots": usable_slots,
        "static_slots": static_slots, "adaptive_slots": usable_slots - static_slots,
        "threads": threads, "capbufs": capbufs,
        "capbuf_bytes": capbufs * slot_bytes,
        "ram_available_bytes": ram_free,
        "warmable_expert_store_bytes": min(layers * experts * slot_bytes, max(0, ram_free - 2 * GIB)),
        "native_kernel": hardware["native_kernel"],
        "kernel_kind": kernel_kind,
        "cpu_library": kernel.get("library"),
        "prefill_library": kernel.get("prefill_library"),
        "kernel_persistent": kernel_kind == "avx512-persistent",
        "fallback_requirement": None,
        "note": "MXFP4 source codes and verified store representation are preserved; capacities are derived from metadata and current free memory.",
    }
