"""Hardware inputs for the local model advisor.

Bandwidth values here are either user-supplied assumptions, a measured CPU
copy proxy, or a GPU memory-interface peak. None is an inference throughput estimate.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time
from typing import Any

from . import gptoss_platform

GIB = 1024 ** 3
_COPY_TOTAL_BYTES = 256 * 1024 ** 2
_CPU_CACHE_TTL_SECONDS = 60.0
_MIN_OVERRIDE_GBPS = 0.1
_MAX_OVERRIDE_GBPS = 100_000.0

# This lookup is intentionally exact after whitespace/case normalization.
# The published board specification gives 19 Gbps over a 384-bit interface:
# 19 * 384 / 8 = 912 GB/s (decimal). Other GPUs remain unknown until a primary
# source is added for their exact marketing name.
_GPU_PUBLISHED_PEAKS = {
    "nvidia geforce rtx 3080 ti": {
        "gbps": 912.0,
        "source": "https://www.gigabyte.com/us/Graphics-Card/GV-N308TVISION-OC-12GD/sp",
        "detail": "Published 19 Gbps memory rate and 384-bit interface; 19×384/8 = 912 GB/s.",
    },
    "geforce rtx 3080 ti": {
        "gbps": 912.0,
        "source": "https://www.gigabyte.com/us/Graphics-Card/GV-N308TVISION-OC-12GD/sp",
        "detail": "Published 19 Gbps memory rate and 384-bit interface; 19×384/8 = 912 GB/s.",
    },
    "rtx 3080 ti": {
        "gbps": 912.0,
        "source": "https://www.gigabyte.com/us/Graphics-Card/GV-N308TVISION-OC-12GD/sp",
        "detail": "Published 19 Gbps memory rate and 384-bit interface; 19×384/8 = 912 GB/s.",
    },
}
_NVIDIA_MEMORY_BANDWIDTH_SOURCES = [
    "https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/",
    "https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/",
]

_cpu_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_cpu_cache_lock = threading.Lock()


def _unknown(note: str) -> dict[str, Any]:
    return {"gbps_low": None, "gbps_high": None, "evidence": "UNKNOWN",
            "note": note, "sources": []}


def _override(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}_bandwidth_gbps must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or not _MIN_OVERRIDE_GBPS <= result <= _MAX_OVERRIDE_GBPS:
        raise ValueError(
            f"{label}_bandwidth_gbps must be between {_MIN_OVERRIDE_GBPS} and {_MAX_OVERRIDE_GBPS} GB/s"
        )
    return result


def _copy_cache_key(hw: dict[str, Any]) -> tuple[Any, ...]:
    features = hw.get("cpu_features")
    feature_key = tuple(sorted((str(k), bool(v)) for k, v in features.items())) if isinstance(features, dict) else ()
    return (str(hw.get("cpu_signature") or ""), int(hw.get("physical_cores") or 0),
            int(hw.get("logical_cores") or 0), int(hw.get("ram_total_bytes") or 0), feature_key)


def _measure_cpu_copy_proxy(hw: dict[str, Any]) -> dict[str, Any]:
    """Measure a bounded host-memory copy proxy, never model inference."""
    available = hw.get("ram_available_bytes")
    if not isinstance(available, int) or available < GIB:
        return _unknown("CPU copy proxy skipped: available RAM is unknown or below the 1 GiB safety threshold.")

    workers = min(8, int(hw.get("physical_cores") or 1))
    if workers < 1:
        workers = 1
    try:
        import numpy as np
    except ImportError:
        return _unknown("CPU copy proxy unavailable because NumPy is not installed.")

    try:
        one_buffer_bytes = _COPY_TOTAL_BYTES // 2
        source = np.empty(one_buffer_bytes, dtype=np.uint8)
        destination = np.empty(one_buffer_bytes, dtype=np.uint8)
        source.fill(0x5A)
        destination.fill(0)
        chunk = (one_buffer_bytes + workers - 1) // workers

        def copy_chunk(index: int) -> None:
            start = index * chunk
            end = min(one_buffer_bytes, start + chunk)
            if start < end:
                np.copyto(destination[start:end], source[start:end])

        sample_gbps: list[float] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Touch every page and start worker threads before the timed samples.
            list(pool.map(copy_chunk, range(workers)))
            for _ in range(3):
                started = time.perf_counter()
                list(pool.map(copy_chunk, range(workers)))
                elapsed = time.perf_counter() - started
                if elapsed > 0:
                    sample_gbps.append((2.0 * one_buffer_bytes) / elapsed / 1_000_000_000)
        if len(sample_gbps) != 3 or any(not math.isfinite(v) or v <= 0 for v in sample_gbps):
            return _unknown("CPU copy proxy did not produce three valid timing samples.")
        return {
            "gbps_low": min(sample_gbps),
            "gbps_high": max(sample_gbps),
            "evidence": "MEASURED_COPY_PROXY",
            "note": ("Three host-memory copy samples using private 256 MiB source/destination buffers and up to "
                     "eight physical-core workers. This is not an inference calibration; expert reads and model "
                     "compute behave differently."),
            "sources": [{"kind": "numpy_copyto", "sample_gbps": sample_gbps,
                         "samples": len(sample_gbps), "buffer_bytes_total": _COPY_TOTAL_BYTES,
                         "workers": workers}],
        }
    except (MemoryError, OSError, RuntimeError, ValueError) as exc:
        return _unknown(f"CPU copy proxy unavailable: {type(exc).__name__}: {exc}")


def _cached_cpu_copy_proxy(hw: dict[str, Any]) -> dict[str, Any]:
    key = _copy_cache_key(hw)
    now = time.monotonic()
    with _cpu_cache_lock:
        cached = _cpu_cache.get(key)
        if cached is not None and now - cached[0] < _CPU_CACHE_TTL_SECONDS:
            return dict(cached[1])
        result = _measure_cpu_copy_proxy(hw)
        _cpu_cache[key] = (now, result)
        return dict(result)


def _normalized_gpu_name(name: Any) -> str:
    return " ".join(str(name or "").casefold().split())


def bandwidth_profile(hw: dict[str, Any], cpu_bandwidth_gbps: float | None = None,
                      gpu_bandwidth_gbps: float | None = None) -> dict[str, dict[str, Any]]:
    """Return transparent bandwidth scenarios, not predicted model tok/s.

    Optional positive overrides are caller assumptions. CPU automatic values
    come from a short copy proxy; GPU automatic values use queried CUDA device
    properties where available or a published peak for exact known names.
    """
    if not isinstance(hw, dict):
        raise TypeError("hw must be a hardware report dictionary")
    cpu_override = _override(cpu_bandwidth_gbps, "cpu")
    gpu_override = _override(gpu_bandwidth_gbps, "gpu")

    if cpu_override is not None:
        cpu = {"gbps_low": cpu_override, "gbps_high": cpu_override,
               "evidence": "ASSUMED_USER_OVERRIDE",
               "note": "User-supplied CPU bandwidth scenario; not a measurement or inference rate.",
               "sources": []}
    else:
        cpu = _cached_cpu_copy_proxy(hw)

    if gpu_override is not None:
        gpu = {"gbps_low": gpu_override, "gbps_high": gpu_override,
               "evidence": "ASSUMED_USER_OVERRIDE",
               "note": "User-supplied GPU bandwidth scenario; not a measurement or inference rate.",
               "sources": []}
    else:
        memory_clock_khz = hw.get("gpu_memory_clock_rate_khz")
        memory_bus_width_bits = hw.get("gpu_memory_bus_width_bits")
        device_rate = None
        if (isinstance(memory_clock_khz, (int, float)) and not isinstance(memory_clock_khz, bool)
                and math.isfinite(memory_clock_khz) and memory_clock_khz > 0
                and isinstance(memory_bus_width_bits, (int, float))
                and not isinstance(memory_bus_width_bits, bool)
                and math.isfinite(memory_bus_width_bits) and memory_bus_width_bits > 0):
            # NVIDIA reports memory_clock_rate in kHz. GDDR transfers twice per
            # clock; divide interface bits by 8 to obtain bytes per transfer.
            device_rate = (2 * float(memory_clock_khz) * 1000
                           * (float(memory_bus_width_bits) / 8) / 1_000_000_000)
        if device_rate is not None:
            gpu = {"gbps_low": device_rate, "gbps_high": device_rate,
                   "evidence": "CALCULATED_DEVICE_PEAK",
                   "note": ("Calculated theoretical memory-interface peak from CUDA device properties: "
                            "2 × memory_clock_rate_kHz × 1000 × (memory_bus_width_bits / 8) / 1e9. "
                            "This physical ceiling is not measured application bandwidth or inference tok/s."),
                   "sources": list(_NVIDIA_MEMORY_BANDWIDTH_SOURCES)}
        else:
            name = _normalized_gpu_name(hw.get("gpu_name"))
            published = _GPU_PUBLISHED_PEAKS.get(name)
            if published is None:
                gpu = _unknown(f"No device memory clock/bus width or exact-name published bandwidth entry for GPU {hw.get('gpu_name')!r}.")
            else:
                peak = float(published["gbps"])
                gpu = {"gbps_low": peak, "gbps_high": peak, "evidence": "PUBLISHED_PEAK",
                       "note": (published["detail"] + " Peak bandwidth is a physical ceiling, not expected "
                                "application bandwidth or inference tok/s."),
                       "sources": [published["source"]]}
    return {"cpu": cpu, "gpu": gpu}


def probe_advisor_hardware(device: str = "cuda:0") -> dict[str, Any]:
    """Probe host capabilities using the known build, then validated portable fallback."""
    root = gptoss_platform.ROOT
    known = root / "artifacts" / "gptoss_known_build.json"
    portable = root / "artifacts" / "portable_cpu_build.json"
    report = gptoss_platform.probe_hardware(device, native_manifest=known)
    if not (report.get("native_kernel") or {}).get("available"):
        report = gptoss_platform.probe_hardware(device, native_manifest=portable)
    return report
