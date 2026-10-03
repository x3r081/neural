"""Offline model suggestions with explicit bandwidth scenarios, not speed promises.

No checkpoint is opened or downloaded. Model facts come from the bundled,
revision-pinned catalog. Only a matching host/kernel may show historical
GPT-OSS observations; those are never extrapolated to reference adapters.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def load_catalog(path=None):
    catalog = json.loads(Path(path or ROOT / "neural_runtime/model_catalog.json").read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("models"), list):
        raise ValueError("unrecognized model catalog")
    ids = [model["id"] for model in catalog["models"]]
    if len(set(ids)) != len(ids):
        raise ValueError("catalog contains duplicate model identifiers")
    return catalog


def _validate_inputs(context, workload, scenario):
    if isinstance(context, bool) or not isinstance(context, int) or not 128 <= context <= 1_048_576:
        raise ValueError("context must be an integer from 128 to 1048576 tokens")
    if workload not in {"coding", "general"}:
        raise ValueError("workload must be coding or general")
    if scenario not in {"dedicated", "current"}:
        raise ValueError("scenario must be dedicated or current")


def _state_bytes(model, context):
    state = model["inference_state"]
    # Full-attention and sliding-attention rates are separate. For GPT-OSS the
    # native runtime's ring capacity is 256, even though the window is 128.
    full = int(state.get("attention_kv_bytes_per_token", 0)) * context
    ring = min(context, max(256, int(state.get("sliding_window", 0))))
    sliding = int(state.get("sliding_kv_bytes_per_token", 0)) * ring
    recurrent = int(state.get("recurrent_state_bytes", 0))
    return full + sliding + recurrent


def _model_memory(model):
    memory = model["storage_bytes"]
    core = int(memory.get("gpu_core", memory["core_non_expert"]))
    cpu_core = int(memory.get("cpu_core", 0))
    experts = int(memory["expert_all"])
    active = int(memory["active_experts_per_token"])
    if model["native_profile"] == "gptoss-120b-mxfp4":
        # Existing lossless PS4 store, not a new quantization. The catalog's
        # original source bytes remain visible separately.
        experts = 36 * 128 * 12_839_040
        active = 36 * 4 * 12_839_040
    return core, cpu_core, experts, active


def _placement(model, hw, context, dedicated):
    core, cpu_core, experts, active = _model_memory(model)
    state = _state_bytes(model, context)
    native = model["support_tier"] == "optimized"
    gpu = int(hw.get("gpu_total_bytes" if dedicated else "gpu_free_bytes") or 0)
    ram = int(hw.get("ram_total_bytes" if dedicated else "ram_available_bytes") or 0)
    # Dedicated capacity reserves OS RAM; current free memory already excludes
    # current OS/process use. The runtime reserve is additional in both cases.
    ram_budget = max(0, ram - (4 * GIB if dedicated else 0) - 2 * GIB)
    workspace = GIB
    expert_size = experts // (model["geometry"]["expert_layers"] * model["geometry"]["total_experts"])
    # Reference caches need one snapshot plus writable working state and two
    # transient expert copies, matching the serving memory planner.
    state_reserved = state if native else 2 * state
    temporary = 6 * expert_size if native else 2 * expert_size
    expert_budget = max(0, gpu - workspace - core - state_reserved - temporary)
    if native:
        # Retain the proven default pool ceiling; do not assume all larger
        # cards already have a validated throughput-tuned placement profile.
        expert_budget = min(expert_budget, int(6.05 * GIB) - temporary)
    # The native pool holds whole expert slots. Reference auto/torch-cpu
    # execution places whole expert layers on CUDA, not fractional layers.
    granularity = expert_size if native else experts // model["geometry"]["expert_layers"]
    gpu_experts = min(experts, (expert_budget // granularity) * granularity)
    # Nonresident experts need RAM. GPU weight duplicates can be evicted from
    # the OS file cache, but CPU embeddings and reference tensor cache remain.
    host_cache = 0 if native else min(GIB, int(ram * 0.05))
    ram_needed = experts - gpu_experts + cpu_core + host_cache
    gpu_fits = gpu >= workspace + core + state_reserved + temporary + expert_size
    ram_fits = ram_needed <= ram_budget
    return {
        "fits": bool(gpu_fits and ram_fits), "gpu_fits": gpu_fits, "ram_fits": ram_fits,
        "model_gib": model["storage_bytes"]["total_model"] / GIB,
        "gpu_core_gib": core / GIB, "cpu_core_gib": cpu_core / GIB,
        "kv_gib": state / GIB, "state_reserved_gib": state_reserved / GIB,
        "gpu_expert_gib": gpu_experts / GIB, "ram_needed_gib": ram_needed / GIB,
        "ram_budget_gib": ram_budget / GIB, "gpu_budget_gib": gpu / GIB,
        "expert_resident_fraction": gpu_experts / experts,
        "active_expert_bytes": active, "core_bytes": core,
        "state_bytes": state, "expert_bytes": experts,
    }


def _matched_observations(hw, model, context):
    if model["native_profile"] != "gptoss-120b-mxfp4":
        return None
    kernel = hw.get("native_kernel", {})
    if not kernel.get("available"):
        return None
    portable = kernel.get("kernel_kind") == "portable-avx2-fma"
    name = "portable_avx2_abba.json" if portable else "native_abba.json"
    if kernel.get("kernel_kind") not in {"portable-avx2-fma", "avx512-persistent"}:
        return None
    path = ROOT / "benchmarks/gptoss_agnostic_20261003" / name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        old = data["hardware"]
        if (data["status"] != "PASS" or old["cpu_signature"] != hw.get("cpu_signature")
                or old["gpu_name"] != hw.get("gpu_name")
                or abs(old["gpu_total_bytes"] - int(hw.get("gpu_total_bytes") or 0)) > 64 * 1024**2
                or old["physical_cores"] != hw.get("physical_cores")):
            return None
        expected_manifest = ROOT / "artifacts" / ("portable_cpu_build.json" if portable else "gptoss_known_build.json")
        expected = json.loads(expected_manifest.read_text(encoding="utf-8"))
        if expected["sha256"] != kernel.get("sha256"):
            return None
        kinds = {"portable"} if portable else {"direct", "launcher"}
        values = [float(value["decode_tok_s"]) for kind, value in data["aggregate"].items() if kind in kinds]
        if not values:
            return None
        return {
            "low_tps": min(values), "high_tps": max(values), "evidence": "MEASURED_HISTORICAL",
            "label": "Same PC and kernel; frozen placement, 87–2597 prompt tokens, one sequence",
            "source": f"https://github.com/x3r081/neural/blob/main/benchmarks/gptoss_agnostic_20261003/{name}",
            "note": f"Historical observation, not a measurement at the selected {context}-token context or under current contention.",
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _speed(model, placement, hw, bandwidth, context):
    fraction = placement["expert_resident_fraction"]
    active = placement["active_expert_bytes"]
    cpu_bytes = active * (1 - fraction)
    # A decode pass attends to the selected context. State traffic is included
    # once; attention arithmetic and implementation overhead are not known.
    gpu_bytes = placement["core_bytes"] + active * fraction + placement["state_bytes"]
    measured = _matched_observations(hw, model, context)
    assumptions = [
        "One sequence; steady-state output generation at the selected context length; prompt processing excluded.",
        "Original checkpoint precision; GPT-OSS uses its existing lossless PS4 store.",
        "Expert selection is uniform: residency fraction is a capacity ratio, not a measured routing hit rate.",
        "No disk faults, competing inference or extra copies; compute, launch, dequantization and framework overhead are omitted.",
        "CPU/GPU activation transfers, synchronization and adaptive expert upload costs are omitted.",
        "These are bandwidth-only scenarios, not expected speed or guaranteed lower/upper bounds; actual throughput may fall below both values.",
    ]
    cpu = bandwidth.get("cpu", {})
    gpu = bandwidth.get("gpu", {})
    cpu_low, cpu_high = cpu.get("gbps_low"), cpu.get("gbps_high")
    gpu_low, gpu_high = gpu.get("gbps_low"), gpu.get("gbps_high")
    if (not gpu_low or not gpu_high or (cpu_bytes > 0 and (not cpu_low or not cpu_high))):
        return {"low_tps": None, "high_tps": None, "evidence": "INSUFFICIENT_BANDWIDTH_DATA",
                "label": "Theoretical speed unavailable; supply a bandwidth scenario", "confidence": "unknown",
                "assumptions": assumptions, "measured": measured}
    cpu_slow = cpu_bytes / (float(cpu_low) * 1e9) if cpu_bytes else 0
    cpu_fast = cpu_bytes / (float(cpu_high) * 1e9) if cpu_bytes else 0
    gpu_slow = gpu_bytes / (float(gpu_low) * 1e9)
    gpu_fast = gpu_bytes / (float(gpu_high) * 1e9)
    # Native has measured asynchronous CPU/GPU scheduling; reference expert
    # dispatch is sequential, so it must not receive an invented overlap gain.
    native = model["support_tier"] == "optimized"
    slow_time = cpu_slow + gpu_slow
    fast_time = max(cpu_fast, gpu_fast) if native else cpu_fast + gpu_fast
    assumptions.append("Native range spans serial to ideal CPU/GPU overlap." if native else
                       "Reference auto/torch-cpu placement is modeled sequentially with whole GPU-resident expert layers; staged PCIe weight streaming is not estimated.")
    return {"low_tps": 1 / slow_time, "high_tps": 1 / fast_time,
            "evidence": "CALCULATED_THEORETICAL", "label": "Theoretical bandwidth scenario (not a forecast)",
            "confidence": "low", "assumptions": assumptions, "measured": measured,
            "cpu_weight_bytes_per_token": cpu_bytes, "gpu_bytes_per_token": gpu_bytes,
            "formula": "1 / (CPU bytes / CPU GB/s + GPU bytes / GPU GB/s); native optimistic endpoint allows overlap"}


def suggest_models(catalog, hardware, bandwidth, *, context=16384, workload="coding", scenario="dedicated"):
    _validate_inputs(context, workload, scenario)
    recommendations, excluded = [], []
    for model in catalog["models"]:
        dedicated = _placement(model, hardware, context, True)
        current = _placement(model, hardware, context, False)
        placement = dedicated if scenario == "dedicated" else current
        native = model["support_tier"] == "optimized"
        reason = None
        if not hardware.get("cuda_available"):
            reason = "CUDA is unavailable; Neural cannot serve this model here."
        elif hardware.get("gpu_is_integrated"):
            reason = "Shared CPU/GPU memory capacity is not validated by this advisor's separate-pool model."
        elif not hardware.get("bf16_supported"):
            reason = "The model requires BF16-capable CUDA with its original precision."
        elif context > model["geometry"]["checkpoint_context_tokens"]:
            reason = "Selected context exceeds the checkpoint's advertised limit."
        elif native and hardware.get("platform") != "Windows":
            reason = "The current optimized GPT-OSS engine requires Windows."
        elif native and not hardware.get("native_kernel", {}).get("available"):
            reason = "No validated GPT-OSS CPU kernel is available for this host."
        elif not placement["gpu_fits"]:
            reason = "Core, context state and workspace do not fit the selected VRAM budget."
        elif not placement["ram_fits"]:
            reason = "The remaining expert weights exceed the RAM budget; disk-paging speed is not predicted."
        notes = []
        if scenario == "dedicated" and not current["fits"]:
            notes.append("Does not fit currently available memory. This suggestion assumes other model servers are stopped.")
        if not native:
            notes.append("Reference adapter only: architecture fixtures passed; this full checkpoint is not validated for Neural performance or quality.")
        if context > model["geometry"].get("runtime_context_tokens", 16384):
            notes.append("This context exceeds the runtime's validated workload; capacity alone does not establish correct long-context performance.")
        notes.extend(model.get("notes", []))
        fit = {key: value for key, value in placement.items() if key.endswith("_gib")}
        fit.update(fits_dedicated=dedicated["fits"], fits_now=current["fits"])
        result = {
            "id": model["id"], "name": model["display_name"], "rank": None,
            "support_tier": model["support_tier"], "validation": model["validation"],
            "status": "blocked" if reason else ("recommended" if native else "reference-experiment"),
            "reason": reason or ("Validated optimized path; strongest local evidence." if native else
                                 "Capacity-compatible architecture; experimental candidate, not a proven speed upgrade."),
            "memory": fit,
            "speed": _speed(model, placement, hardware, bandwidth, context) if not reason else {
                "low_tps": None, "high_tps": None, "evidence": "NOT_RUNNABLE_IN_SCENARIO",
                "label": "No runnable speed estimate", "confidence": "unknown", "measured": None, "assumptions": []},
            "notes": notes, "sources": model["sources"], "revision": model["revision"],
            "task_match": workload in model.get("task_tags", []),
        }
        (excluded if reason else recommendations).append(result)
    # Evidence is ranked ahead of theoretical speed. No invented intelligence
    # score makes an untested reference model outrank a validated deployment.
    recommendations.sort(key=lambda row: (row["support_tier"] != "optimized", not row["task_match"],
                                           -(row["speed"]["low_tps"] or 0), row["name"]))
    for rank, row in enumerate(recommendations, 1):
        row["rank"] = rank
    default_model = next((row for row in recommendations + excluded if row["id"] == "gpt-oss-120b"), None)
    return {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_verified_date": catalog["verified_date"], "context": context,
        "workload": workload, "scenario": scenario, "hardware": hardware, "bandwidth": bandwidth,
        "recommendations": recommendations, "excluded": excluded,
        "default_model": default_model,
        "notices": [
            "GPT-OSS 120B is the default. Other models are advanced experiments and are never selected automatically when GPT-OSS cannot run.",
            "No models were downloaded or loaded, and no model output was generated.",
            "The curated catalog is revision-pinned and offline; it is not an exhaustive or automatically updated leaderboard.",
            "Ranking prioritizes validated Neural support, then task tags and theoretical throughput; task tags are not quality benchmark scores.",
            "Dedicated capacity reserves 4 GiB RAM for the OS, 2 GiB for runtime and 1 GiB VRAM for workspace; free-memory fit is a separate snapshot.",
            "Historical measured tok/s and theoretical scenarios are different evidence; neither predicts successful coding tasks per hour.",
        ],
    }


def build_report(*, context=16384, workload="coding", scenario="dedicated",
                 cpu_bandwidth_gbps=None, gpu_bandwidth_gbps=None, device="cuda:0"):
    _validate_inputs(context, workload, scenario)
    from .advisor_hardware import probe_advisor_hardware, bandwidth_profile
    hardware = probe_advisor_hardware(device)
    bandwidth = bandwidth_profile(hardware, cpu_bandwidth_gbps=cpu_bandwidth_gbps,
                                  gpu_bandwidth_gbps=gpu_bandwidth_gbps)
    return suggest_models(load_catalog(), hardware, bandwidth, context=context, workload=workload, scenario=scenario)


def format_text(report):
    hw = report["hardware"]
    lines = ["Neural model suggestions", f"GPU: {hw.get('gpu_name') or 'CUDA unavailable'}",
             "Default model: GPT-OSS 120B. Install once with install_neural.bat; open chat with start_neural.bat.",
             f"Context: {report['context']:,} tokens | capacity scenario: {report['scenario']}",
             "Theoretical values are bandwidth scenarios, not measured or expected model speed.", ""]
    if (report.get("default_model") or {}).get("status") == "blocked":
        lines += ["Default model cannot run in this scenario: " + report["default_model"]["reason"],
                  "No alternative model will be selected automatically.", ""]
    for row in report["recommendations"]:
        speed = row["speed"]
        if speed["low_tps"] is None:
            rate = "not estimable (missing bandwidth data)"
        else:
            rate = f"{speed['low_tps']:.1f}–{speed['high_tps']:.1f} output tok/s [CALCULATED; low confidence]"
        lines += [f"{row['rank']}. {row['name']} — {row['status']}", f"   Theoretical: {rate}",
                  f"   {row['reason']}"]
        if speed.get("measured"):
            m = speed["measured"]
            lines.append(f"   Historical measurement: {m['low_tps']:.2f}–{m['high_tps']:.2f} tok/s; {m['label']}")
            lines.append(f"   {m['note']}")
        lines.append(f"   RAM needed: {row['memory']['ram_needed_gib']:.1f} GiB; fits currently: {row['memory']['fits_now']}")
        lines += ["   " + note for note in row["notes"]]
        lines.append("")
    if report["excluded"]:
        lines.append("Excluded from this capacity scenario:")
        lines += [f"- {row['name']}: {row['reason']}" for row in report["excluded"]]
    for side in ("cpu", "gpu"):
        profile = report["bandwidth"][side]
        lines += ["", f"{side.upper()} bandwidth evidence: {profile['evidence']} — {profile['note']}"]
    lines += ["", "Inspect assumptions and sources with --json, or open the browser advisor with --serve --open."]
    # Console code pages on Windows often cannot render the UI's typography.
    return "\n".join(lines).replace("–", "-").replace("—", "-").replace("×", "x")
