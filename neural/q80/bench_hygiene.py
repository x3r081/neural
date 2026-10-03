"""Benchmark hygiene: record whether the machine was IDLE or CONTENDED.

Q80-FLOOR-1 exists because an entire removal-oracle battery was run through
contention from this project's own measurement processes and produced
physically impossible results (removals that made the token SLOWER). The
contamination was invisible in the artifacts, so every performance artifact
now carries a hygiene stamp and callers can refuse to measure through load.

Nothing here kills or throttles anything: unrelated user processes are the
user's business. It observes, stamps, and optionally refuses.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

# Names that indicate a competing inference/benchmark workload. Matching is
# substring, case-insensitive, on the process image name.
COMPETITOR_HINTS = ("python", "llama", "ollama", "koboldcpp", "lmstudio",
                    "text-generation", "vllm", "tgi", "comfyui")

IDLE_GPU_UTIL_PCT = 25.0
IDLE_GPU_MEM_MIB = 2048.0
IDLE_CPU_PCT = 35.0


def _nvidia(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _cpu_pct(samples: int = 3, interval: float = 0.4) -> float:
    try:
        import psutil
    except Exception:
        return float("nan")
    vals = [psutil.cpu_percent(interval=interval) for _ in range(samples)]
    return round(sum(vals) / len(vals), 1)


def snapshot(*, own_pid: int | None = None,
             own_gpu_mib: float | None = None) -> dict[str, Any]:
    """Observe the machine. Cheap enough to call around every measurement.

    ``own_gpu_mib`` is subtracted from the observed GPU memory so a snapshot
    taken DURING our own run does not flag us as our own contention - the first
    version of this gate did exactly that and marked every arm inadmissible."""
    import os

    own = own_pid if own_pid is not None else os.getpid()
    gpu = _nvidia("gpu=utilization.gpu,memory.used,memory.total,clocks.sm,temperature.gpu")
    util = mem = total = clk = temp = None
    if gpu:
        parts = [p.strip() for p in gpu[0].split(",")]
        if len(parts) >= 5:
            util, mem, total, clk, temp = (float(parts[0]), float(parts[1]),
                                           float(parts[2]), float(parts[3]),
                                           float(parts[4]))
    others = []
    try:
        import psutil

        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            if p.info["pid"] == own:
                continue
            nm = (p.info["name"] or "").lower()
            if any(h in nm for h in COMPETITOR_HINTS):
                rss = getattr(p.info.get("memory_info"), "rss", 0) or 0
                others.append({"pid": p.info["pid"], "name": p.info["name"],
                               "rss_gib": round(rss / 2**30, 3)})
    except Exception:
        pass
    cpu = _cpu_pct()
    # A competitor that is merely resident is not contention; one holding real
    # memory is. Idle ollama sits at ~0.02 GiB and must not fail the gate.
    heavy = [o for o in others if o["rss_gib"] >= 0.5]
    reasons = []
    # GPU utilisation is only meaningful BEFORE our own work starts; during a
    # run it is our own kernels. Callers pass own_gpu_mib for the after-shot,
    # which also switches the utilisation check off.
    if own_gpu_mib is None and util is not None and util > IDLE_GPU_UTIL_PCT:
        reasons.append(f"gpu_util={util}% > {IDLE_GPU_UTIL_PCT}%")
    if own_gpu_mib is not None and cpu == cpu and cpu > 95.0:
        reasons.append(f"cpu saturated at {cpu}% during the run")
    other_mem = None if mem is None else max(0.0, mem - (own_gpu_mib or 0.0))
    if other_mem is not None and other_mem > IDLE_GPU_MEM_MIB:
        reasons.append(f"gpu_mem_used_by_others={other_mem:.0f} MiB "
                       f"> {IDLE_GPU_MEM_MIB} MiB")
    if own_gpu_mib is None and cpu == cpu and cpu > IDLE_CPU_PCT:
        reasons.append(f"cpu={cpu}% > {IDLE_CPU_PCT}%")
    if heavy:
        reasons.append(f"competing processes: {heavy}")
    return {
        "state": "IDLE" if not reasons else "CONTENDED",
        "reasons": reasons,
        "gpu_util_pct": util, "gpu_mem_used_mib": mem,
        "gpu_mem_used_by_others_mib": other_mem,
        "own_gpu_mib": own_gpu_mib, "gpu_mem_total_mib": total,
        "gpu_sm_clock_mhz": clk, "gpu_temp_c": temp,
        "cpu_pct": cpu,
        "competitor_processes": others,
        "t": round(time.time(), 1),
    }


def require_idle(*, allow_contended: bool = False,
                 own_pid: int | None = None) -> dict[str, Any]:
    """Snapshot and refuse to proceed unless the machine is idle."""
    s = snapshot(own_pid=own_pid)
    if s["state"] != "IDLE" and not allow_contended:
        raise RuntimeError(
            "REFUSING TO BENCHMARK THROUGH CONTENTION: " + "; ".join(s["reasons"])
            + " -- wait for the machine to clear, or pass allow_contended=True "
              "and the artifact will be stamped CONTENDED.")
    return s


def stamp(before: dict[str, Any], after: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine a before/after pair into one artifact field."""
    out = {"before": before, "after": after}
    states = {before["state"]} | ({after["state"]} if after else set())
    out["state"] = "IDLE" if states == {"IDLE"} else "CONTENDED"
    return out
