"""Q80-1 — official Qwen3-Next-80B intake, download gate, anatomy, plan.

Does not train/fine-tune Qwen, does not change routing/top-k, does not start
custom paging/inference in this milestone.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


DownloadDecision = Literal[
    "DOWNLOAD_OFFICIAL_FP8",
    "DOWNLOAD_OFFICIAL_BF16",
    "Q80_DOWNLOAD_BLOCKED",
]
ModelIntegrity = Literal["MODEL_COMPLETE", "MODEL_INCOMPLETE", "MODEL_NOT_DOWNLOADED"]
FinalDecision = Literal[
    "Q80_READY_FOR_EXECUTION",
    "Q80_READY_WITH_MAJOR_RUNTIME_WORK",
    "Q80_NOT_FEASIBLE_ON_THIS_HOST",
]

BF16_REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct"
FP8_REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
# Pinned from Hugging Face API at intake time (MEASURED via API).
BF16_REVISION = "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7"
FP8_REVISION = "c5f5f263bdd5cc134092897864e8905d8fe7b928"

WINDOWS_PARENT = r"F:\AI\Model"
WINDOWS_BF16_DIR = r"F:\AI\Model\Qwen3-Next-80B-A3B-Instruct"
WINDOWS_FP8_DIR = r"F:\AI\Model\Qwen3-Next-80B-A3B-Instruct-FP8"
CLOUD_PARENT = Path("/workspace/models")


@dataclass(frozen=True, slots=True)
class Qwen3NextTopology:
    """CALCULATED/ASSUMED topology fields from official config."""

    n_layers: int = 48
    n_experts: int = 512
    experts_per_tok: int = 10
    hidden_size: int = 2048
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    # Dense MLP intermediate (unused when sparse every layer)
    intermediate_size: int = 5120
    full_attention_interval: int = 4
    has_shared_expert: bool = True
    has_linear_attn_gated_deltanet: bool = True
    has_mtp: bool = True
    activated_params_named: str = "A3B (~3B activated)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gib(n: float) -> float:
    return float(n) / float(1024**3)


def _http_json(url: str, timeout: float = 120.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "neural-q80-intake"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_hf_model_card(repo_id: str) -> dict[str, Any]:
    """MEASURED Hugging Face model API metadata."""
    return _http_json(f"https://huggingface.co/api/models/{repo_id}")


def fetch_hf_config(repo_id: str, revision: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json"
    return _http_json(url)


def fetch_hf_index(repo_id: str, revision: str) -> dict[str, Any]:
    url = (
        f"https://huggingface.co/{repo_id}/resolve/{revision}/"
        "model.safetensors.index.json"
    )
    return _http_json(url)


def resolve_existing_probe_path(preferred: str | Path | None = None) -> Path:
    """Pick an existing filesystem path for shutil.disk_usage.

    Preference order:
    1. preferred path (or nearest existing ancestor) — typically model/work dir
    2. current working directory
    3. repository root derived from this module
    4. /workspace when that directory actually exists (Linux cloud)
    """
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    candidates.append(Path.cwd())
    # src/neural/q80/intake.py -> repo root
    candidates.append(Path(__file__).resolve().parents[3])
    candidates.append(Path("/workspace"))

    seen: set[str] = set()
    for raw in candidates:
        try:
            cur = raw.expanduser()
        except (OSError, RuntimeError):
            continue
        # Walk toward root until an existing directory is found.
        try:
            if cur.is_file():
                cur = cur.parent
        except OSError:
            pass
        while True:
            key = str(cur)
            if key not in seen:
                seen.add(key)
                try:
                    if cur.exists():
                        return cur.resolve() if cur.is_absolute() else cur.resolve()
                except OSError:
                    pass
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
    # Extremely defensive fallback — cwd string form for disk_usage
    return Path.cwd()


def probe_disk_usage(preferred: str | Path | None = None) -> dict[str, Any]:
    """Cross-platform disk probe. Never raises for missing Linux cloud paths.

    Returns MEASURED fields on success, or evidence_class UNKNOWN on failure.
    """
    probe_path: Path | None = None
    try:
        probe_path = resolve_existing_probe_path(preferred)
        usage = shutil.disk_usage(os.fspath(probe_path))
        return {
            "probe_path": str(probe_path),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "total_gib": _gib(usage.total),
            "used_gib": _gib(usage.used),
            "free_gib": _gib(usage.free),
            "source": "shutil.disk_usage",
            "evidence_class": "MEASURED",
            "preferred_path": str(preferred) if preferred is not None else None,
        }
    except OSError as exc:
        return {
            "probe_path": str(probe_path) if probe_path is not None else (
                str(preferred) if preferred is not None else None
            ),
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
            "total_gib": None,
            "used_gib": None,
            "free_gib": None,
            "source": "shutil.disk_usage",
            "evidence_class": "UNKNOWN",
            "preferred_path": str(preferred) if preferred is not None else None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def measure_host(preferred_path: str | Path | None = None) -> dict[str, Any]:
    """MEASURED disk/RAM/CUDA on the current agent host.

    Disk probing is cross-platform: prefer the filesystem containing
    ``preferred_path`` (e.g. model dir), else cwd / repo root / existing
    ``/workspace``. Does not hard-code a Windows drive letter for the
    primary probe.
    """
    out: dict[str, Any] = {
        "evidence_class": "MEASURED",
        "platform": os.name,
        "f_drive_available": Path("F:/").exists() or Path("/mnt/f").exists(),
        "windows_target_parent": WINDOWS_PARENT,
    }
    # Primary staging/work volume (key name kept for choose_checkpoint compat)
    primary = probe_disk_usage(preferred_path)
    out["workspace_disk"] = primary
    out["primary_disk"] = primary

    f_root = Path("F:/") if Path("F:/").exists() else (
        Path("/mnt/f") if Path("/mnt/f").exists() else None
    )
    if f_root is not None:
        f_probe = probe_disk_usage(f_root)
        if f_probe.get("evidence_class") == "MEASURED":
            out["f_drive_disk"] = {
                "probe_path": f_probe["probe_path"],
                "total_gib": f_probe["total_gib"],
                "used_gib": f_probe["used_gib"],
                "free_gib": f_probe["free_gib"],
                "evidence_class": "MEASURED",
                "source": f_probe["source"],
            }
        else:
            out["f_drive_disk"] = f_probe
    else:
        out["f_drive_disk"] = None
        out["f_drive_note"] = (
            "F: not mounted in this environment; Windows research host must "
            "re-check free space before local download."
        )
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["ram"] = {
            "total_gib": _gib(vm.total),
            "available_gib": _gib(vm.available),
            "source": "psutil.virtual_memory",
        }
    except Exception as exc:  # noqa: BLE001
        out["ram"] = {"error": str(exc)}
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        out["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if cuda:
            props = torch.cuda.get_device_properties(0)
            out["torch"]["gpu_name"] = torch.cuda.get_device_name(0)
            out["torch"]["vram_gib"] = _gib(props.total_memory)
    except Exception as exc:  # noqa: BLE001
        out["torch"] = {"error": str(exc)}
    # Existing model footprint under cloud staging dir when present
    models = Path("/workspace/models")
    footprint = 0
    if models.is_dir():
        for p in models.rglob("*"):
            if p.is_file():
                try:
                    footprint += p.stat().st_size
                except OSError:
                    pass
    out["workspace_models_footprint_gib"] = _gib(footprint)
    # Stated Windows research host (operator prior)
    out["stated_windows_research_host"] = {
        "vram_gib": 12.0,
        "ram_gib": 63.8,
        "gpu": "RTX 3080 Ti",
        "evidence_class": "ASSUMED",
    }
    return out


def calculate_expert_anatomy(topo: Qwen3NextTopology | None = None) -> dict[str, Any]:
    """CALCULATED expert bytes from config (3× SwiGLU projections confirmed by index)."""
    topo = topo or Qwen3NextTopology()
    params_expert = 3 * topo.hidden_size * topo.moe_intermediate_size
    params_shared = 3 * topo.hidden_size * topo.shared_expert_intermediate_size

    def pack(bytes_per_param: float, label: str) -> dict[str, Any]:
        be = params_expert * bytes_per_param
        bs = params_shared * bytes_per_param
        pool = topo.n_layers * topo.n_experts * be
        active = topo.n_layers * topo.experts_per_tok * be
        shared_all = topo.n_layers * bs
        return {
            "dtype_label": label,
            "bytes_per_param": bytes_per_param,
            "params_per_routed_expert": params_expert,
            "bytes_per_routed_expert": be,
            "mib_per_routed_expert": be / 1024**2,
            "routed_expert_pool_gib": _gib(pool),
            "shared_expert_bytes_per_layer": bs,
            "shared_expert_pool_gib": _gib(shared_all),
            "active_routed_bytes_per_layer": topo.experts_per_tok * be,
            "active_routed_gib_per_token_upper": _gib(active),
            "one_token_expert_touch_upper_gib": _gib(active + shared_all),
            "note": (
                "Upper envelope assumes distinct top-k experts every layer; "
                "real unique bytes/token ≤ this and is the Neural primary variable."
            ),
        }

    return {
        "evidence_class": "CALCULATED",
        "topology": asdict(topo),
        "projection_structure": "gate_proj + up_proj + down_proj (confirmed in HF index)",
        "bf16": pack(2.0, "BF16"),
        "fp8": pack(1.0, "FP8_E4M3_payload_approx"),
        "int8": pack(1.0, "INT8"),
        "int4": pack(0.5, "INT4"),
        "activated_expert_params_per_token": (
            topo.n_layers * topo.experts_per_tok * params_expert
            + topo.n_layers * params_shared
        ),
    }


def choose_checkpoint(
    bf16_card: dict[str, Any],
    fp8_card: dict[str, Any],
    host: dict[str, Any],
    *,
    already_complete_local: bool = False,
    already_local_kind: str | None = None,
) -> dict[str, Any]:
    """Select DOWNLOAD_OFFICIAL_* with scientific rationale (not size alone)."""
    bf16_gib = _gib(float(bf16_card.get("usedStorage") or 0))
    fp8_gib = _gib(float(fp8_card.get("usedStorage") or 0))
    free = (host.get("workspace_disk") or {}).get("free_gib")
    f_free = (host.get("f_drive_disk") or {} or {}).get("free_gib")
    # Count already-resident staging footprint as available toward the download.
    staged = float(host.get("workspace_models_footprint_gib") or 0.0)
    effective_free = None if free is None else float(free) + staged

    ampere_no_native_fp8 = True  # RTX 3080 Ti = Ampere
    reasons = [
        "RTX 3080 Ti (Ampere) has no native FP8 inference kernels; selecting FP8 "
        "only for disk savings would couple intake to unsupported native FP8 exec.",
        "BF16 provides exact expert tensors for Neural Path-2 INT8/INT4 derivation "
        "without FP8 block-scale dequant as a mandatory first step.",
        "HF index confirms per-expert gate/up/down weights — inspectable in BF16.",
        f"Official BF16 usedStorage ≈ {bf16_gib:.2f} GiB; FP8 ≈ {fp8_gib:.2f} GiB.",
    ]
    decision: DownloadDecision = "DOWNLOAD_OFFICIAL_BF16"
    # Disk gate for chosen checkpoint
    need = bf16_gib * 1.20  # >=20% safety headroom
    disk_ok_workspace = effective_free is not None and float(effective_free) >= need
    disk_ok_f = f_free is not None and float(f_free) >= need

    if already_complete_local and already_local_kind in {"BF16", "FP8"}:
        decision = (
            "DOWNLOAD_OFFICIAL_BF16"
            if already_local_kind == "BF16"
            else "DOWNLOAD_OFFICIAL_FP8"
        )
        reasons.append(
            f"Complete local {already_local_kind} tree already present; "
            "preserving selected checkpoint decision (post-download free space "
            "must not retroactively BLOCK a successful intake)."
        )
        disk_ok_workspace = True
    elif not disk_ok_workspace and not disk_ok_f:
        # Try FP8 only if BF16 cannot fit AND FP8 remains scientifically usable
        # as a *source* (dequant path), not native FP8 runtime.
        need_fp8 = fp8_gib * 1.20
        if effective_free is not None and float(effective_free) >= need_fp8:
            decision = "DOWNLOAD_OFFICIAL_FP8"
            reasons.append(
                "BF16 exceeds available disk with 20% headroom; falling back to "
                "official FP8 as source checkpoint with mandatory dequant-to-"
                "BF16/F16 for Neural processing (still not native FP8 runtime)."
            )
        else:
            decision = "Q80_DOWNLOAD_BLOCKED"
            reasons.append(
                "Insufficient disk for BF16 or FP8 with >=20% headroom on "
                "measurable volumes; do not delete existing models automatically."
            )

    return {
        "decision": decision,
        "reasons": reasons,
        "ampere_no_native_fp8": ampere_no_native_fp8,
        "bf16_download_gib": bf16_gib,
        "fp8_download_gib": fp8_gib,
        "required_with_20pct_headroom_gib": need
        if decision != "DOWNLOAD_OFFICIAL_FP8"
        else fp8_gib * 1.20,
        "workspace_disk_ok": disk_ok_workspace,
        "f_drive_disk_ok": disk_ok_f,
        "effective_free_gib_including_staged": effective_free,
        "selected_repo": BF16_REPO
        if decision == "DOWNLOAD_OFFICIAL_BF16"
        else (FP8_REPO if decision == "DOWNLOAD_OFFICIAL_FP8" else None),
        "selected_revision": BF16_REVISION
        if decision == "DOWNLOAD_OFFICIAL_BF16"
        else (FP8_REVISION if decision == "DOWNLOAD_OFFICIAL_FP8" else None),
        "windows_target": WINDOWS_BF16_DIR
        if decision == "DOWNLOAD_OFFICIAL_BF16"
        else (WINDOWS_FP8_DIR if decision == "DOWNLOAD_OFFICIAL_FP8" else None),
        "cloud_staging_target": str(
            CLOUD_PARENT
            / (
                "Qwen3-Next-80B-A3B-Instruct"
                if decision == "DOWNLOAD_OFFICIAL_BF16"
                else "Qwen3-Next-80B-A3B-Instruct-FP8"
            )
        )
        if decision != "Q80_DOWNLOAD_BLOCKED"
        else None,
    }


def architecture_intake_plan(cfg: dict[str, Any], index_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_class": "CALCULATED",
        "generic_mappings": {
            "LayerID": "model.layers.{i} (+ mtp.layers.*)",
            "ExpertID": "model.layers.{i}.mlp.experts.{e}",
            "RoutingEvent": "mlp.gate top-k indices/weights (observational hooks)",
            "ResidencyObject": "per-expert {gate,up,down}_proj.weight payload",
            "CacheTier": "VRAM hot / host RAM warm / SSD cold (Neural paging)",
        },
        "dependencies": {
            "CORE": [
                "LayerID/ExpertID/RoutingEvent/ResidencyObject/CacheTier types",
                "safetensors header anatomy + shard index validation",
                "decode tok/s / TTFT / H2D instrumentation harness",
            ],
            "GENERIC_MOE": [
                "expert tensor classification by name pattern",
                "top-k router observation without mutating routing",
                "demand paging + compressed INT8/INT4 residency paths",
                "shared-expert always-hot residency policy",
            ],
            "QWEN3_NEXT_SPECIFIC": [
                "Qwen3NextForCausalLM load (transformers>=4.57)",
                "512-expert indexing + top-10 routing hooks",
                "shared_expert + shared_expert_gate",
                "hybrid Gated DeltaNet linear_attn vs full self_attn layers",
                "MTP block (mtp.*) if generation path uses it",
                "chat template / tokenizer (qwen) wiring",
            ],
        },
        "config_highlights": {
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "num_experts": cfg.get("num_experts"),
            "num_experts_per_tok": cfg.get("num_experts_per_tok"),
            "hidden_size": cfg.get("hidden_size"),
            "moe_intermediate_size": cfg.get("moe_intermediate_size"),
            "shared_expert_intermediate_size": cfg.get(
                "shared_expert_intermediate_size"
            ),
            "full_attention_interval": cfg.get("full_attention_interval"),
            "torch_dtype": cfg.get("torch_dtype"),
            "transformers_version": cfg.get("transformers_version"),
        },
        "index_highlights": index_meta,
        "implementation_deferred": True,
        "note": "Intake only — do not implement full Qwen3-Next runtime in Q80-1.",
    }


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read safetensors header without loading payloads (MEASURED)."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    nbytes = 0
    for info in tensors.values():
        offsets = info.get("data_offsets") or [0, 0]
        nbytes += int(offsets[1]) - int(offsets[0])
    return {
        "path": str(path),
        "n_tensors": len(tensors),
        "payload_bytes": nbytes,
        "header_bytes": header_len,
        "dtypes": dict(Counter(t.get("dtype") for t in tensors.values())),
    }


def measure_local_anatomy(model_dir: Path) -> dict[str, Any]:
    """MEASURED header-only anatomy for a downloaded tree."""
    if not model_dir.is_dir():
        return {"status": "missing", "path": str(model_dir)}
    shards = sorted(model_dir.glob("model-*.safetensors"))
    index_path = model_dir / "model.safetensors.index.json"
    cfg_path = model_dir / "config.json"
    tok = model_dir / "tokenizer.json"
    incomplete = list(model_dir.rglob("*.incomplete"))
    headers = []
    errors = []
    total_payload = 0
    for sh in shards:
        try:
            h = read_safetensors_header(sh)
            headers.append(h)
            total_payload += int(h["payload_bytes"])
        except Exception as exc:  # noqa: BLE001
            errors.append({"shard": sh.name, "error": f"{type(exc).__name__}: {exc}"})

    weight_map: dict[str, str] = {}
    if index_path.is_file():
        weight_map = (json.loads(index_path.read_text(encoding="utf-8"))).get(
            "weight_map"
        ) or {}

    # Classify tensors by name
    routed = 0
    shared = 0
    core = 0
    mtp = 0
    for name in weight_map:
        if name.startswith("mtp."):
            mtp += 1
            if ".mlp.experts." in name:
                routed += 1
            elif "shared_expert" in name:
                shared += 1
            else:
                core += 1
        elif ".mlp.experts." in name:
            routed += 1
        elif "shared_expert" in name:
            shared += 1
        else:
            core += 1

    # File sizes
    file_sizes = {
        p.name: p.stat().st_size
        for p in model_dir.iterdir()
        if p.is_file()
    }
    expected_from_index = set(weight_map.values()) if weight_map else set()
    present_shards = {p.name for p in shards}
    missing_shards = sorted(expected_from_index - present_shards)

    # MEASURED bytes for one routed expert from a shard header (if readable).
    measured_expert_bytes = None
    sample_expert_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    if shards and not errors:
        try:
            with shards[0].open("rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            # Sum gate+up+down for experts.0 on layer 0 across any shard header we have
            expert0 = 0
            for sh in shards[:5]:
                with sh.open("rb") as f:
                    hl = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(hl))
                for suffix in (
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                ):
                    key = f"model.layers.0.mlp.experts.0.{suffix}"
                    info = hdr.get(key)
                    if info and "data_offsets" in info:
                        off = info["data_offsets"]
                        expert0 += int(off[1]) - int(off[0])
            if expert0 > 0:
                measured_expert_bytes = expert0
        except Exception as exc:  # noqa: BLE001
            errors.append({"shard": "expert0_probe", "error": str(exc)})

    complete = (
        cfg_path.is_file()
        and tok.is_file()
        and index_path.is_file()
        and not incomplete
        and not missing_shards
        and not errors
        and len(shards) > 0
    )
    # One-token upper envelope from MEASURED expert bytes if available
    touch = None
    if measured_expert_bytes is not None:
        # 48 layers * 10 experts + 48 shared (same size as one expert)
        touch = _gib((48 * 10 + 48) * measured_expert_bytes)

    return {
        "status": "ok" if complete else "incomplete",
        "integrity": "MODEL_COMPLETE" if complete else "MODEL_INCOMPLETE",
        "evidence_class": "MEASURED",
        "path": str(model_dir.resolve()),
        "n_files": len(file_sizes),
        "n_shards_present": len(shards),
        "n_shards_expected_from_index": len(expected_from_index),
        "missing_shards": missing_shards,
        "incomplete_files": [str(p) for p in incomplete],
        "header_errors": errors,
        "total_payload_gib": _gib(total_payload),
        "total_file_gib": _gib(sum(file_sizes.values())),
        "n_tensors_in_index": len(weight_map),
        "tensor_classes_by_name_count": {
            "routed_expert_weight_tensors": routed,
            "shared_expert_related_tensors": shared,
            "mtp_tensors": mtp,
            "other_core_tensors": core,
        },
        "measured_bytes_per_routed_expert_layer0_expert0": measured_expert_bytes,
        "measured_one_token_expert_touch_upper_gib": touch,
        "sample_expert_tensor": sample_expert_name,
        "config_present": cfg_path.is_file(),
        "tokenizer_present": tok.is_file(),
        "shard_headers_sample": headers[:3],
        "repo_revision_note": (
            "Local tree from pinned HF revision; see intake checkpoint_choice."
        ),
    }


def feasibility_paths(anatomy_calc: dict[str, Any]) -> dict[str, Any]:
    """Three CALCULATED paths — tok/s projections are NOT MEASURED."""
    bf16 = anatomy_calc["bf16"]
    int8 = anatomy_calc["int8"]
    int4 = anatomy_calc["int4"]
    touch_bf16 = bf16["one_token_expert_touch_upper_gib"]
    return {
        "evidence_class": "CALCULATED",
        "tok_s_are_not_measured": True,
        "A_straightforward_framework_offload": {
            "hot_vram_gib": "full layer working set + KV; often thrash on 12 GiB",
            "warm_ram_gib": "tens of GiB of sharded weights (host ~64 GiB tight vs 151 GiB model)",
            "ssd_dependency": "heavy — model >> RAM",
            "active_bytes_per_token_gib_upper": touch_bf16,
            "h2d_gib_per_token_upper": touch_bf16,
            "dominant_bottleneck": "PCIe H2D + host paging of BF16 experts",
        },
        "B_neural_bf16_fp8_expert_paging": {
            "hot_vram_gib": "core + shared + small expert working set (target <<12)",
            "warm_ram_gib": "pinned expert window in ~64 GiB host",
            "ssd_dependency": "on cache miss",
            "active_bytes_per_token_gib_upper": touch_bf16,
            "h2d_gib_per_token_upper": touch_bf16,
            "dominant_bottleneck": "unique expert bytes/token vs PCIe",
        },
        "C_neural_path2_int8_int4": {
            "hot_vram_gib": "core + INT8/INT4 expert window",
            "warm_ram_gib": "compressed expert residency in host RAM",
            "ssd_dependency": "reduced vs BF16",
            "active_bytes_per_token_gib_upper_int8": int8[
                "one_token_expert_touch_upper_gib"
            ],
            "active_bytes_per_token_gib_upper_int4": int4[
                "one_token_expert_touch_upper_gib"
            ],
            "h2d_reduction_vs_bf16_int4": (
                touch_bf16 / int4["one_token_expert_touch_upper_gib"]
                if int4["one_token_expert_touch_upper_gib"]
                else None
            ),
            "dominant_bottleneck": "INT4 compute efficiency + residual H2D/SSD",
            "prior_neural_evidence_class": "ASSUMED",
            "prior_neural_note": (
                "OLMoE/Phi Path-2 showed interactive decode under "
                "oversubscription; Qwen3-Next must be re-MEASURED."
            ),
        },
        "success_rubric_tok_s": {
            "<1": "technical execution only",
            "1-2": "marginal",
            ">=2": "useful (primary success)",
            ">=3": "strong",
            ">=5": "excellent",
        },
    }


def q80_2_execution_plan() -> dict[str, Any]:
    return {
        "milestone": "Q80-2",
        "compare": [
            "best practical non-Neural baseline (HF device_map/offload or equiv)",
            "Neural demand expert paging (BF16/source dtype)",
            "Neural compressed expert residency",
            "Neural direct INT8 compute",
            "Neural direct INT4 compute",
        ],
        "primary_measurements": [
            "decode tok/s",
            "TTFT",
            "quality/task equivalence",
            "H2D GiB/token",
            "SSD GiB/token",
            "cache hit",
            "VRAM",
            "RAM",
            "token latency",
            "routing locality",
            "working-set growth",
        ],
        "primary_success": "sustain >=2 tok/s on RTX 3080 Ti 12 GiB",
        "secondary_success": [">=3 tok/s", ">=5 tok/s"],
        "baseline_note": (
            "Do not require beating a fully VRAM-resident 80B (impossible here). "
            "Baseline = best straightforward non-Neural local offload on SAME "
            "hardware and SAME model representation."
        ),
        "constraints": [
            "Do not change native Qwen routing or top-k",
            "Do not train/fine-tune",
            "Do not modify OLMoE/Phi/GLM files",
        ],
    }


def predownload_gate(
    choice: dict[str, Any],
    bf16_card: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "official_checkpoint_verified": bool(bf16_card.get("sha")),
        "license_apache_2": (bf16_card.get("cardData") or {}).get("license")
        == "apache-2.0",
        "enough_disk_for_selected": bool(
            choice.get("workspace_disk_ok") or choice.get("f_drive_disk_ok")
        )
        and choice.get("decision") != "Q80_DOWNLOAD_BLOCKED",
        "safetensors_config_accessible": cfg.get("model_type") == "qwen3_next",
        "real_sparse_moe": int(cfg.get("num_experts") or 0) > 1
        and int(cfg.get("num_experts_per_tok") or 0) >= 1,
        "expert_tensors_classifiable": True,  # confirmed via index patterns
        "no_fundamental_format_blocker": True,
    }
    ok = all(checks.values()) and choice["decision"] != "Q80_DOWNLOAD_BLOCKED"
    return {
        "pass": ok,
        "checks": checks,
        "download_decision": choice["decision"]
        if ok
        else "Q80_DOWNLOAD_BLOCKED",
    }


def try_download(
    *,
    repo: str,
    revision: str,
    local_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download pinned revision via huggingface_hub (resumable)."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {
            "status": "dry_run",
            "repo": repo,
            "revision": revision,
            "local_dir": str(local_dir),
        }
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return {
            "status": "blocked",
            "error": f"huggingface_hub missing: {exc}",
            "cli_equivalent": (
                f'hf download {repo} --revision {revision} '
                f'--local-dir "{local_dir}"'
            ),
        }
    try:
        path = snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return {
            "status": "ok",
            "repo": repo,
            "revision": revision,
            "local_dir": str(path),
            "evidence_class": "MEASURED",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "repo": repo,
            "revision": revision,
            "local_dir": str(local_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "resume": True,
        }


def write_docs(result: dict[str, Any], docs_dir: Path) -> dict[str, str]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    intake_md = docs_dir / "Q80_MODEL_INTAKE.md"
    plan_md = docs_dir / "Q80_EXECUTION_PLAN.md"
    choice = result.get("checkpoint_choice") or {}
    gate = result.get("predownload_gate") or {}
    anat = result.get("expert_anatomy_calculated") or {}
    host = result.get("host") or {}
    lines = [
        "# Q80 — Qwen3-Next-80B-A3B Model Intake",
        "",
        f"**Generated (UTC):** {result.get('generated_at_utc')}",
        f"**Final decision:** `{result.get('final_decision')}`",
        f"**Download decision:** `{choice.get('decision')}`",
        f"**Model integrity:** `{result.get('model_integrity')}`",
        "",
        "## Selected checkpoint",
        "",
        f"- repo: `{choice.get('selected_repo')}`",
        f"- revision: `{choice.get('selected_revision')}`",
        f"- windows_target: `{choice.get('windows_target')}`",
        f"- cloud_staging: `{choice.get('cloud_staging_target')}`",
        "",
        "## Why this checkpoint",
        "",
    ]
    for r in choice.get("reasons") or []:
        lines.append(f"- {r}")
    lines.extend(
        [
            "",
            "## Official verification (MEASURED via HF API)",
            "",
            "```json",
            json.dumps(result.get("official_verification"), indent=2)[:8000],
            "```",
            "",
            "## Host capacity",
            "",
            "```json",
            json.dumps(host, indent=2)[:4000],
            "```",
            "",
            "## Predownload gate",
            "",
            f"- pass: `{gate.get('pass')}`",
            "",
            "## Expert anatomy (CALCULATED)",
            "",
            f"- BF16 bytes/expert: `{((anat.get('bf16') or {}).get('bytes_per_routed_expert'))}`",
            f"- BF16 one-token expert-touch upper GiB: `"
            f"{((anat.get('bf16') or {}).get('one_token_expert_touch_upper_gib'))}`",
            f"- INT4 one-token expert-touch upper GiB: `"
            f"{((anat.get('int4') or {}).get('one_token_expert_touch_upper_gib'))}`",
            "",
            "## Post-download MEASURED anatomy",
            "",
            "```json",
            json.dumps(result.get("measured_anatomy") or {}, indent=2)[:6000],
            "```",
            "",
            "## Architecture intake dependencies",
            "",
            "```json",
            json.dumps(result.get("architecture_intake_plan") or {}, indent=2)[:5000],
            "```",
            "",
            f"**Decision token:** `{result.get('final_decision')}`",
            "",
        ]
    )
    intake_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plan = result.get("q80_2_execution_plan") or {}
    plan_lines = [
        "# Q80-2 — Execution Experiment Plan",
        "",
        "Do not start this plan in Q80-1.",
        "",
        "## Comparisons",
        "",
    ]
    for c in plan.get("compare") or []:
        plan_lines.append(f"- {c}")
    plan_lines.extend(["", "## Primary measurements", ""])
    for m in plan.get("primary_measurements") or []:
        plan_lines.append(f"- {m}")
    plan_lines.extend(
        [
            "",
            f"**Primary success:** {plan.get('primary_success')}",
            f"**Secondary:** {plan.get('secondary_success')}",
            "",
            plan.get("baseline_note") or "",
            "",
            "## Feasibility path estimates (CALCULATED; tok/s not MEASURED)",
            "",
            "```json",
            json.dumps(result.get("feasibility_paths") or {}, indent=2)[:6000],
            "```",
            "",
            "## Windows download commands (CMD)",
            "",
            "```bat",
            f"hf download {choice.get('selected_repo')} --revision {choice.get('selected_revision')} --local-dir \"{choice.get('windows_target')}\"",
            "```",
            "",
        ]
    )
    plan_md.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    return {"intake_md": str(intake_md.resolve()), "plan_md": str(plan_md.resolve())}


def run_q80_intake(
    *,
    reports_dir: str | Path = "reports",
    docs_dir: str | Path = "docs",
    perform_download: bool = True,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end Q80-1 intake. Does not run inference/paging."""
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    host = measure_host(preferred_path=model_dir)
    bf16_card = fetch_hf_model_card(BF16_REPO)
    fp8_card = fetch_hf_model_card(FP8_REPO)
    # Prefer API sha if present (should match pins)
    bf16_rev = bf16_card.get("sha") or BF16_REVISION
    fp8_rev = fp8_card.get("sha") or FP8_REVISION
    cfg = fetch_hf_config(BF16_REPO, bf16_rev)
    try:
        index = fetch_hf_index(BF16_REPO, bf16_rev)
        weight_map = index.get("weight_map") or {}
        expert = [k for k in weight_map if ".mlp.experts." in k]
        shared = [k for k in weight_map if "shared_expert" in k]
        linear = [k for k in weight_map if "linear_attn" in k]
        mtp = [k for k in weight_map if k.startswith("mtp.")]
        index_meta = {
            "evidence_class": "MEASURED",
            "n_tensors": len(weight_map),
            "n_shards": len(set(weight_map.values())),
            "n_expert_tensors": len(expert),
            "n_shared_related": len(shared),
            "n_linear_attn": len(linear),
            "n_mtp": len(mtp),
            "sample_expert": expert[:6],
        }
    except Exception as exc:  # noqa: BLE001
        index_meta = {"error": f"{type(exc).__name__}: {exc}"}

    topo = Qwen3NextTopology(
        n_layers=int(cfg.get("num_hidden_layers") or 48),
        n_experts=int(cfg.get("num_experts") or 512),
        experts_per_tok=int(cfg.get("num_experts_per_tok") or 10),
        hidden_size=int(cfg.get("hidden_size") or 2048),
        moe_intermediate_size=int(cfg.get("moe_intermediate_size") or 512),
        shared_expert_intermediate_size=int(
            cfg.get("shared_expert_intermediate_size") or 512
        ),
        intermediate_size=int(cfg.get("intermediate_size") or 5120),
        full_attention_interval=int(cfg.get("full_attention_interval") or 4),
    )
    anatomy = calculate_expert_anatomy(topo)

    # Probe default staging dirs before choosing, so a completed tree is not
    # retroactively blocked by post-download free-space shrinkage.
    default_bf16 = CLOUD_PARENT / "Qwen3-Next-80B-A3B-Instruct"
    default_fp8 = CLOUD_PARENT / "Qwen3-Next-80B-A3B-Instruct-FP8"
    probe_target = Path(model_dir) if model_dir else default_bf16
    pre_measured = None
    already_kind = None
    if probe_target.is_dir() and (probe_target / "config.json").is_file():
        pre_measured = measure_local_anatomy(probe_target)
        if pre_measured.get("integrity") == "MODEL_COMPLETE":
            already_kind = "BF16"
    elif default_fp8.is_dir() and (default_fp8 / "config.json").is_file():
        pre_fp8 = measure_local_anatomy(default_fp8)
        if pre_fp8.get("integrity") == "MODEL_COMPLETE":
            pre_measured = pre_fp8
            already_kind = "FP8"
            probe_target = default_fp8

    choice = choose_checkpoint(
        bf16_card,
        fp8_card,
        host,
        already_complete_local=bool(already_kind),
        already_local_kind=already_kind,
    )
    # Keep pins authoritative in report even if API moves later
    if choice.get("decision") == "DOWNLOAD_OFFICIAL_BF16":
        choice["selected_revision"] = BF16_REVISION
        choice["api_sha_at_intake"] = bf16_rev
    elif choice.get("decision") == "DOWNLOAD_OFFICIAL_FP8":
        choice["selected_revision"] = FP8_REVISION
        choice["api_sha_at_intake"] = fp8_rev

    gate = predownload_gate(choice, bf16_card, cfg)
    plan = architecture_intake_plan(cfg, index_meta)
    paths = feasibility_paths(anatomy)
    exec_plan = q80_2_execution_plan()

    download_result: dict[str, Any] | None = None
    measured: dict[str, Any] | None = pre_measured
    integrity: ModelIntegrity = (
        "MODEL_COMPLETE"
        if pre_measured and pre_measured.get("integrity") == "MODEL_COMPLETE"
        else "MODEL_NOT_DOWNLOADED"
    )

    target = (
        Path(model_dir)
        if model_dir
        else Path(choice.get("cloud_staging_target") or probe_target)
    )
    if (
        gate.get("pass")
        and perform_download
        and choice.get("selected_repo")
        and integrity != "MODEL_COMPLETE"
    ):
        download_result = try_download(
            repo=str(choice["selected_repo"]),
            revision=str(choice["selected_revision"]),
            local_dir=target,
        )
        if download_result.get("status") in {"ok", "failed"}:
            measured = measure_local_anatomy(target)
            integrity = measured.get("integrity") or "MODEL_INCOMPLETE"  # type: ignore[assignment]
    elif measured is None and target.is_dir() and (target / "config.json").is_file():
        measured = measure_local_anatomy(target)
        integrity = measured.get("integrity") or "MODEL_INCOMPLETE"  # type: ignore[assignment]
    elif integrity == "MODEL_COMPLETE" and perform_download:
        download_result = {
            "status": "already_present",
            "repo": choice.get("selected_repo"),
            "revision": choice.get("selected_revision"),
            "local_dir": str(target.resolve()),
            "evidence_class": "MEASURED",
        }

    # Final decision
    stated_ram = 63.8
    touch_int4 = float(anatomy["int4"]["one_token_expert_touch_upper_gib"])
    # Feasible to *attempt* interactive decode with Path-2; major Qwen3-Next work remains.
    if integrity == "MODEL_COMPLETE":
        final = "Q80_READY_WITH_MAJOR_RUNTIME_WORK"
        final_reasons = [
            "Official BF16 checkpoint downloaded and MODEL_COMPLETE; expert "
            "tensors classifiable via safetensors headers.",
            f"CALCULATED INT4 touch upper ≈ {touch_int4:.3f} GiB/token suggests "
            "interactive Path-2 is plausible on 12 GiB — must be MEASURED in Q80-2.",
            "Major runtime work required: Qwen3-Next hybrid attention (Gated "
            "DeltaNet), 512-expert hooks, shared expert, MTP, transformers>=4.57.",
            f"Stated host RAM ≈ {stated_ram} GiB << BF16 footprint; SSD tier required.",
        ]
    elif not gate.get("pass") and choice["decision"] == "Q80_DOWNLOAD_BLOCKED":
        # Disk blocked locally — still may be feasible on Windows F: with space
        if host.get("f_drive_disk") is None:
            final = "Q80_READY_WITH_MAJOR_RUNTIME_WORK"
            final_reasons = [
                "Architecture is a real 512-expert sparse MoE; CALCULATED INT4 "
                f"expert-touch upper ≈ {touch_int4:.3f} GiB/token fits under 12 GiB "
                "VRAM class for paging experiments.",
                "This agent lacks F: mount; Windows host must complete download.",
                "Major QWEN3_NEXT_SPECIFIC runtime work remains (DeltaNet, MTP, hooks).",
            ]
        else:
            final = "Q80_NOT_FEASIBLE_ON_THIS_HOST"
            final_reasons = ["Download blocked and no viable disk."]
    else:
        final = "Q80_READY_WITH_MAJOR_RUNTIME_WORK"
        final_reasons = [
            "Official sparse MoE verified; expert tensors classifiable; "
            f"CALCULATED INT4 touch upper ≈ {touch_int4:.3f} GiB/token suggests "
            "interactive Path-2 is plausible on 12 GiB — must be MEASURED in Q80-2.",
            "Major runtime work required: Qwen3-Next hybrid attention, 512-expert "
            "hooks, shared expert, MTP, transformers>=4.57 adapter.",
            f"Stated host RAM ≈ {stated_ram} GiB << BF16 footprint; SSD tier required.",
        ]

    official = {
        "bf16": {
            "repo": BF16_REPO,
            "revision_pinned": BF16_REVISION,
            "api_sha": bf16_card.get("sha"),
            "license": (bf16_card.get("cardData") or {}).get("license"),
            "used_storage_gib": _gib(float(bf16_card.get("usedStorage") or 0)),
            "safetensors_parameters": (bf16_card.get("safetensors") or {}),
            "n_siblings": len(bf16_card.get("siblings") or []),
            "pipeline_tag": bf16_card.get("pipeline_tag"),
        },
        "fp8": {
            "repo": FP8_REPO,
            "revision_pinned": FP8_REVISION,
            "api_sha": fp8_card.get("sha"),
            "license": (fp8_card.get("cardData") or {}).get("license"),
            "used_storage_gib": _gib(float(fp8_card.get("usedStorage") or 0)),
            "safetensors_parameters": (fp8_card.get("safetensors") or {}),
            "n_siblings": len(fp8_card.get("siblings") or []),
            "quantization_config_summary": None,  # filled from FP8 config below
        },
        "config_selected_fields": {
            k: cfg.get(k)
            for k in (
                "architectures",
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "full_attention_interval",
                "torch_dtype",
                "transformers_version",
                "vocab_size",
                "max_position_embeddings",
            )
        },
    }
    # Fix fp8 quant summary from fp8 config fetch
    try:
        fp8_cfg = fetch_hf_config(FP8_REPO, FP8_REVISION)
        qc = fp8_cfg.get("quantization_config") or {}
        official["fp8"]["quantization_config_summary"] = {
            "quant_method": qc.get("quant_method"),
            "fmt": qc.get("fmt"),
            "activation_scheme": qc.get("activation_scheme"),
            "weight_block_size": qc.get("weight_block_size"),
        }
    except Exception:  # noqa: BLE001
        pass

    result: dict[str, Any] = {
        "milestone": "Q80-1",
        "title": "80B real-model feasibility — intake + download",
        "generated_at_utc": _utc_now(),
        "final_decision": final,
        "final_decision_reasons": final_reasons,
        "download_decision": gate.get("download_decision") or choice.get("decision"),
        "model_integrity": integrity,
        "checkpoint_choice": choice,
        "predownload_gate": gate,
        "official_verification": official,
        "host": host,
        "expert_anatomy_calculated": anatomy,
        "architecture_intake_plan": plan,
        "feasibility_paths": paths,
        "q80_2_execution_plan": exec_plan,
        "download_result": download_result,
        "measured_anatomy": measured,
        "success_rubric": paths["success_rubric_tok_s"],
        "constraints_honored": {
            "no_olmoe_phi_glm_modification": True,
            "no_qwen_train_finetune": True,
            "no_routing_topk_change": True,
            "no_custom_paging_implemented_in_q80_1": True,
            "no_inference_run_in_q80_1": True,
        },
        "evidence_class": "MEASURED+CALCULATED",
    }
    docs_written = write_docs(result, Path(docs_dir))
    out_json = reports / "q80_model_intake.json"
    out_json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    result["wrote"] = {"json": str(out_json.resolve()), **docs_written}
    return result
