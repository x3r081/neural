"""One-click setup for the pinned, original-MXFP4 GPT-OSS 120B runtime.

The setup downloads only the official checkpoint files at the pinned revision,
builds the repository's raw store, then uses the existing PS4 packer with full
verification. It never quantizes or deletes model/store data.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

from .defaults import DEFAULT_MODEL_REPO, DEFAULT_MODEL_REVISION, default_paths

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
RAW_SLOT_BYTES = 13_219_200
PS4_SLOT_BYTES = 12_839_040
N_LAYERS = 36
N_EXPERTS = 128
# pip's CUDA wheel set is the largest external dependency in this setup. This
# fixed reserve covers the pinned cu130 torch/triton wheels and their cache.
DEPENDENCY_RESERVE_BYTES = 12 * GIB
DEPENDENCY_VENV_RESERVE_BYTES = 4 * GIB
DEPENDENCY_CACHE_RESERVE_BYTES = DEPENDENCY_RESERVE_BYTES - DEPENDENCY_VENV_RESERVE_BYTES
STORE_BIAS_RESERVE_BYTES = 160 * 1024 ** 2
CONFIG_NAME = "neural.local.json"


class SetupError(RuntimeError):
    """Actionable setup failure."""


def _resolve(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_config_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def setup_paths(data_dir=None, model_dir=None, raw_store_dir=None, store_dir=None) -> dict[str, Path]:
    paths = default_paths(data_dir)
    result = {key: Path(value).resolve() for key, value in paths.items()}
    if model_dir is not None:
        result["model_dir"] = _resolve(model_dir)
    if raw_store_dir is not None:
        result["raw_store_dir"] = _resolve(raw_store_dir)
    if store_dir is not None:
        result["store_dir"] = _resolve(store_dir)
    return result


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"Cannot read {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SetupError(f"{config_path} must contain a JSON object")
    return data


def _checkpoint_model_type(model_dir: Path) -> str | None:
    config = model_dir / "config.json"
    if not config.is_file():
        return None
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"Cannot read model config {config}: {exc}") from exc
    return data.get("model_type") if isinstance(data, dict) else None


def _selected_files(info) -> list[Any]:
    """Select official top-level original safetensors and tokenizer/config assets."""
    included = []
    excluded_extensions = (".gguf", ".bin", ".pt", ".pth", ".onnx", ".tflite")
    for sibling in info.siblings:
        if not isinstance(sibling.rfilename, str):
            raise SetupError("Hugging Face metadata contains an invalid file name")
        name = sibling.rfilename.replace("\\", "/")
        parts = name.split("/")
        if (name in {"", ".", ".."} or ".." in parts
                or any(part.lower() in {"original", "metal"} for part in parts)):
            continue
        if len(parts) != 1 or name.lower().endswith(excluded_extensions):
            continue
        if (name.endswith(".safetensors") or name.endswith(".jinja") or name in {
            "config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "README.md", "LICENSE", "USAGE_POLICY",
        } or name.startswith("tokenizer.")):
            if sibling.size is None or int(sibling.size) < 0:
                raise SetupError(f"Hugging Face metadata did not provide a file size for {name}")
            included.append(sibling)
    names = {item.rfilename for item in included}
    if "config.json" not in names or "model.safetensors.index.json" not in names:
        raise SetupError("Pinned repository metadata lacks config.json or model.safetensors.index.json")
    if not any(name.endswith(".safetensors") for name in names):
        raise SetupError("Pinned repository metadata contains no original safetensors checkpoint shards")
    return included


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _same_volume_key(path: Path) -> str:
    # Existing parent matters when the destination directory is not created yet.
    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.path.normcase(str(parent.anchor or parent))


def disk_requirements(files, model_dir: Path, raw_dir: Path, packed_dir: Path,
                      *, dependency_dir: Path = ROOT, dependency_cache_dir: Path = ROOT,
                      include_dependencies: bool = True,
                      raw_complete: bool = False,
                      packed_complete: bool = False) -> dict[str, int]:
    """Return exact required free bytes per volume for known payloads and reserve."""
    required: dict[str, int] = {}
    groups: dict[str, list[Path]] = {}
    model_key, raw_key, packed_key = (_same_volume_key(p) for p in (model_dir, raw_dir, packed_dir))
    for sibling in files:
        expected = int(sibling.size)
        present = _file_bytes(model_dir / sibling.rfilename)
        remaining = max(0, expected - present) if present == expected else expected
        required[model_key] = required.get(model_key, 0) + remaining
    if not raw_complete:
        required[raw_key] = required.get(raw_key, 0) + N_LAYERS * N_EXPERTS * RAW_SLOT_BYTES + STORE_BIAS_RESERVE_BYTES // 2
    if not packed_complete:
        required[packed_key] = required.get(packed_key, 0) + N_LAYERS * N_EXPERTS * PS4_SLOT_BYTES + STORE_BIAS_RESERVE_BYTES // 2
    if include_dependencies:
        dep_key = _same_volume_key(dependency_dir)
        required[dep_key] = required.get(dep_key, 0) + DEPENDENCY_VENV_RESERVE_BYTES
        cache_key = _same_volume_key(dependency_cache_dir)
        required[cache_key] = required.get(cache_key, 0) + DEPENDENCY_CACHE_RESERVE_BYTES
    return required


def check_disk_space(required: dict[str, int], paths: dict[str, Path], dependency_dir: Path = ROOT,
                     dependency_cache_dir: Path = ROOT) -> None:
    grouped = {"model_dir": paths["model_dir"], "raw_store_dir": paths["raw_store_dir"],
               "store_dir": paths["store_dir"], "dependency environment": dependency_dir,
               "dependency cache": dependency_cache_dir}
    checked = set()
    for path in grouped.values():
        key = _same_volume_key(path)
        if key in checked:
            continue
        checked.add(key)
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        need = required.get(key, 0)
        if free < need:
            raise SetupError(f"Insufficient free space on {path.anchor}: {free / GIB:.1f} GiB free, "
                             f"{need / GIB:.1f} GiB required for remaining checkpoint files, stores, bias files, and configured reserves. "
                             "Choose paths on a volume with more free space.")


def _validated_existing_store(model_dir: Path, store_dir: Path) -> bool:
    from .model_spec import ModelSpecError, inspect_model
    try:
        spec = inspect_model(model_dir, store_dir, mode="fast")
    except (ModelSpecError, OSError, ValueError, KeyError, TypeError):
        return False
    return spec.model_type == "gpt_oss" and spec.source_format.get("store_format") == "mxfp4_g32_ps4"


def _check_hardware(spec) -> dict[str, Any]:
    from .advisor_hardware import probe_advisor_hardware
    from .gptoss_platform import plan_gptoss
    try:
        hardware = probe_advisor_hardware("cuda:0")
    except Exception as exc:
        raise SetupError(f"Could not validate CUDA and the GPT-OSS runtime on this host: {type(exc).__name__}: {exc}. "
                         "Check the pinned dependencies and NVIDIA driver.") from exc
    plan = plan_gptoss(spec.to_dict(), hardware, context=16384)
    if not plan.get("supported"):
        reason = plan.get("reason") or "hardware capacity check failed"
        native_reason = (hardware.get("native_kernel") or {}).get("reason")
        raise SetupError(f"GPT-OSS setup is not supported on this host: {reason}"
                         + (f"; {native_reason}" if native_reason else "")
                         + ". Install a supported CUDA 13.0 PyTorch runtime and use a host with a verified GPT-OSS kernel.")
    return {"hardware": hardware, "plan": plan}


def _preflight_host_capacity() -> dict[str, Any]:
    """Reject unsupported host/CUDA/kernel/capacity before the large download."""
    from .advisor import load_catalog, suggest_models
    from .advisor_hardware import probe_advisor_hardware
    try:
        hardware = probe_advisor_hardware("cuda:0")
    except Exception as exc:
        raise SetupError(f"Could not validate CUDA and the GPT-OSS runtime on this host: {type(exc).__name__}: {exc}. "
                         "Check the pinned dependencies and NVIDIA driver; no checkpoint download was started.") from exc
    if hardware.get("platform") != "Windows":
        raise SetupError("The GPT-OSS optimized runtime requires Windows.")
    if not hardware.get("cuda_available") or not hardware.get("bf16_supported"):
        raise SetupError("A CUDA device with BF16 support is required; no checkpoint download was started.")
    if not (hardware.get("native_kernel") or {}).get("available"):
        why = (hardware.get("native_kernel") or {}).get("reason", "no verified GPT-OSS kernel is available")
        raise SetupError(f"GPT-OSS native kernel preflight failed: {why}. No checkpoint download was started.")
    report = suggest_models(load_catalog(), hardware, {"cpu": {}, "gpu": {}}, context=16384,
                            workload="coding", scenario="dedicated")
    default = report.get("default_model") or {}
    if default.get("status") == "blocked":
        raise SetupError("GPT-OSS capacity preflight failed: " + str(default.get("reason"))
                         + ". Stop other GPU workloads or use a machine with more VRAM/RAM; no checkpoint download was started.")
    return {"hardware_preflight": {"gpu_name": hardware.get("gpu_name"),
                                   "cuda_available": hardware.get("cuda_available"),
                                   "bf16_supported": hardware.get("bf16_supported"),
                                   "native_kernel": hardware.get("native_kernel"),
                                   "default_model_status": default.get("status"),
                                   "memory": default.get("memory")}}


def _atomic_config(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_config(path)
    current.update(values)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(current, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _save_config_if_needed(args, path: Path, values: dict[str, str]) -> None:
    current = _read_config(path)
    if not path.exists() or "model_dir" not in current:
        _atomic_config(path, values)
        return
    if not (args.data_dir or args.model_dir or args.raw_store_dir or args.store_dir):
        return
    configured = _resolve_config_path(current["model_dir"])
    if _checkpoint_model_type(configured) == "gpt_oss":
        _atomic_config(path, values)


def _select_paths(args, config_path: Path) -> dict[str, Path]:
    config = _read_config(config_path)
    for key in ("model_dir", "store_dir", "raw_store_dir"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise SetupError(f"{config_path} field {key!r} must be a non-empty path string. "
                             "The configuration was preserved; repair it or pass --config with a separate file.")
    configured_model = config.get("model_dir")
    if config and ("model_dir" in config or "store_dir" in config) and not configured_model:
        raise SetupError(f"{config_path} has no usable model_dir. It was preserved. Repair it or pass --config with a separate file.")
    kind = _checkpoint_model_type(_resolve_config_path(configured_model)) if configured_model else None
    if config and configured_model and kind is None:
        raise SetupError(f"{config_path} points to a missing or unreadable model path. It was preserved. "
                         "Repair that path or pass --config with a separate configuration file.")
    if kind and kind != "gpt_oss":
        raise SetupError(f"{config_path} points to a {kind!r} model. It was preserved. To set up GPT-OSS, "
                         "pass --config with a separate file; the installer will not replace a custom model configuration.")
    paths = setup_paths(args.data_dir)
    if kind == "gpt_oss":
        if not args.model_dir:
            paths["model_dir"] = _resolve_config_path(configured_model)
        if not args.store_dir and config.get("store_dir"):
            paths["store_dir"] = _resolve_config_path(config["store_dir"])
        if not args.raw_store_dir and config.get("raw_store_dir"):
            paths["raw_store_dir"] = _resolve_config_path(config["raw_store_dir"])
    if args.model_dir is not None:
        paths["model_dir"] = _resolve(args.model_dir)
    if args.raw_store_dir is not None:
        paths["raw_store_dir"] = _resolve(args.raw_store_dir)
    if args.store_dir is not None:
        paths["store_dir"] = _resolve(args.store_dir)
    return paths


def _assert_python() -> None:
    if sys.version_info[:2] != (3, 12) or platform.architecture()[0] != "64bit":
        raise SetupError("Neural setup requires 64-bit Python 3.12. Install it from https://www.python.org/downloads/release/python-31210/ "
                         "and rerun install_neural.bat.")


def _dependency_versions_ok() -> tuple[bool, str]:
    expected = {"torch": "2.10.0+cu130", "triton-windows": "3.7.1.post27", "transformers": "5.14.1",
                "tokenizers": "0.22.2", "safetensors": "0.8.0", "numpy": "2.5.1",
                "psutil": "7.2.2", "huggingface-hub": "1.27.0", "accelerate": "1.14.0",
                "httpx": "0.28.1"}
    mismatches = []
    for package, wanted in expected.items():
        try:
            found = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found = "missing"
        if found != wanted:
            mismatches.append(f"{package}={found} (requires {wanted})")
    modules = {"torch": "torch", "triton-windows": "triton", "transformers": "transformers",
               "tokenizers": "tokenizers", "safetensors": "safetensors", "numpy": "numpy",
               "psutil": "psutil", "huggingface-hub": "huggingface_hub",
               "accelerate": "accelerate", "httpx": "httpx"}
    for package, module in modules.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            mismatches.append(f"{package} import failed ({type(exc).__name__}: {exc})")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "all pinned runtime dependencies are present"


def _validate_checkpoint_only(model_dir: Path) -> None:
    """Validate GPT-OSS config, shard headers and HF markers without tensor payload reads."""
    from .model_spec import (GPTOSS_EXPECTED_REVISION, ModelSpecError, _checkpoint_memory,
                             _geometry, _read_json, _verify_download_revision,
                             _verified_source_dtype)
    try:
        config, _ = _read_json(model_dir / "config.json", "GPT-OSS config")
        if config.get("model_type") != "gpt_oss":
            raise SetupError("checkpoint config model_type must be 'gpt_oss'")
        quant = config.get("quantization_config")
        if not isinstance(quant, dict) or str(quant.get("quant_method", "")).lower() != "mxfp4":
            raise SetupError("checkpoint must use the official original MXFP4 format")
        geometry = _geometry(config)
        expected = {"layers": 36, "experts": 128, "top_k": 4, "hidden": 2880,
                    "intermediate": 2880, "num_heads": 64, "num_kv_heads": 8, "head_dim": 64, "window": 128}
        if any(geometry.get(key) != value for key, value in expected.items()):
            raise SetupError("checkpoint GPT-OSS geometry does not match the supported 120B profile")
        memory, checkpoint_paths = _checkpoint_memory(model_dir)
        index_path = checkpoint_paths.get("safetensors_index")
        if not index_path:
            raise SetupError("checkpoint is missing model.safetensors.index.json")
        index, _ = _read_json(Path(index_path), "Safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SetupError("checkpoint Safetensors index has no weight_map")
        marker_info = _verify_download_revision(model_dir, weight_map)
        if any(revision != GPTOSS_EXPECTED_REVISION for revision in marker_info.values()):
            raise SetupError(f"checkpoint download markers do not match pinned revision {GPTOSS_EXPECTED_REVISION}")
        dtype = _verified_source_dtype(config, memory)
        if (dtype != "BF16" or memory.get("expert_weight_dtypes") != ["U8"]
                or memory.get("expert_bias_dtypes") != ["BF16"]):
            raise SetupError("checkpoint tensors are not the original BF16 core/bias and U8 MXFP4 expert bytes")
    except ModelSpecError as exc:
        raise SetupError(f"checkpoint metadata validation failed: {exc}") from exc


def check_existing(model_dir: Path, store_dir: Path, *, hardware: bool = True) -> dict[str, Any]:
    from .model_spec import inspect_model
    try:
        spec = inspect_model(model_dir, store_dir, mode="fast")
    except Exception as exc:
        raise SetupError(f"Existing model/store failed metadata validation: {exc}") from exc
    if spec.model_type != "gpt_oss" or spec.source_format.get("store_format") != "mxfp4_g32_ps4":
        raise SetupError("The configured model/store is not the supported original-MXFP4 PS4 GPT-OSS profile")
    result = {"model": spec.to_dict()}
    result["validation_scope"] = "metadata schema and file sizes; model/store payload hashes were not recomputed"
    if hardware:
        result.update(_check_hardware(spec))
    return result


def _download_checkpoint(model_dir: Path, files, *, api=None, downloader=None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = downloader or snapshot_download
    info = api.model_info(DEFAULT_MODEL_REPO, revision=DEFAULT_MODEL_REVISION, files_metadata=True)
    if info.sha != DEFAULT_MODEL_REVISION:
        raise SetupError(f"Hugging Face returned revision {info.sha}, expected pinned {DEFAULT_MODEL_REVISION}")
    current = _selected_files(info)
    chosen = {item.rfilename: int(item.size) for item in files}
    if {item.rfilename: int(item.size) for item in current} != chosen:
        raise SetupError("Pinned Hugging Face file metadata changed between preflight and download; rerun setup")
    print(f"Downloading pinned {DEFAULT_MODEL_REPO}@{DEFAULT_MODEL_REVISION} to {model_dir}", flush=True)
    downloader(DEFAULT_MODEL_REPO, revision=DEFAULT_MODEL_REVISION, local_dir=str(model_dir),
               allow_patterns=sorted(chosen), max_workers=2)
    missing = [name for name, size in chosen.items() if _file_bytes(model_dir / name) != size]
    if missing:
        raise SetupError("Checkpoint download is incomplete or has wrong file sizes: " + ", ".join(missing[:8]))


class HuggingFaceMetadataAPI:
    """Small stdlib-only reader for pinned public Hugging Face file metadata."""

    def model_info(self, repo_id: str, *, revision: str, files_metadata: bool = True):
        repo = quote(repo_id, safe="/")
        ref = quote(revision, safe="")
        request = Request(f"https://huggingface.co/api/models/{repo}/revision/{ref}?blobs=true",
                          headers={"User-Agent": "Neural-GPT-OSS-setup/1"})
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.load(response)
        except Exception as exc:
            raise SetupError(f"Could not read pinned Hugging Face model metadata: {exc}") from exc
        siblings = []
        for item in payload.get("siblings", []):
            if not isinstance(item, dict):
                continue
            size = item.get("size")
            if size is None and isinstance(item.get("lfs"), dict):
                size = item["lfs"].get("size")
            siblings.append(SimpleNamespace(rfilename=item.get("rfilename", ""), size=size))
        return SimpleNamespace(sha=payload.get("sha"), siblings=siblings)


def _store_state(raw_dir: Path, packed_dir: Path, model_dir: Path) -> tuple[bool, bool]:
    from .model_spec import ModelSpecError, inspect_model
    raw_complete = False
    packed_complete = False
    if raw_dir.exists() and any(raw_dir.iterdir()):
        try:
            raw = inspect_model(model_dir, raw_dir, mode="fast")
            raw_complete = raw.model_type == "gpt_oss" and raw.store["format"] == "mxfp4_g32"
        except (ModelSpecError, OSError, ValueError, KeyError, TypeError):
            if not (packed_dir.exists() and (packed_dir / "metadata.json").is_file()):
                raise SetupError(f"{raw_dir} contains an incomplete or unrecognized store. Setup will not overwrite it. "
                                 "Choose a new --raw-store-dir or inspect and move the partial folder yourself.")
    if packed_dir.exists() and any(packed_dir.iterdir()):
        if (packed_dir / "metadata.json").is_file():
            packed_complete = _validated_existing_store(model_dir, packed_dir)
            if not packed_complete:
                raise SetupError(f"{packed_dir} contains a store that fails validation; setup will not overwrite it")
        elif not (packed_dir / "pack_progress.json").is_file():
            raise SetupError(f"{packed_dir} contains partial files without pack_progress.json. Setup will not overwrite it; "
                             "choose a fresh --store-dir or inspect and move the partial folder yourself.")
    return raw_complete, packed_complete


def run_setup(args, *, api=None, downloader=None,
              run_command: Callable[..., Any] = subprocess.run,
              hardware_check: bool = True) -> dict[str, Any]:
    paths = _select_paths(args, _resolve_config_path(args.config))
    model_dir, raw_dir, packed_dir = paths["model_dir"], paths["raw_store_dir"], paths["store_dir"]
    dependency_dir = _resolve(args.dependency_dir) if args.dependency_dir else ROOT
    dependency_cache_dir = _resolve(args.dependency_cache_dir) if args.dependency_cache_dir else ROOT
    if _checkpoint_model_type(model_dir) not in (None, "gpt_oss"):
        raise SetupError(f"{model_dir} contains a non-GPT-OSS model. Choose a separate --model-dir; setup will not overwrite it.")
    if args.skip_dependencies and not args.dry_run:
        okay, detail = _dependency_versions_ok()
        if not okay:
            raise SetupError("Pinned runtime dependencies are unavailable in this environment: " + detail + ". "
                             "Create a fresh repository .venv and run install_neural.bat.")
    if args.check:
        report = check_existing(model_dir, packed_dir, hardware=hardware_check)
        report["status"] = "CHECK_OK"
        return report
    if _validated_existing_store(model_dir, packed_dir):
        report = check_existing(model_dir, packed_dir, hardware=hardware_check and not args.dry_run)
        if not args.dry_run:
            _save_config_if_needed(args, _resolve_config_path(args.config), {"model_dir": str(model_dir),
                                     "store_dir": str(packed_dir), "raw_store_dir": str(raw_dir)})
        report["status"] = "ALREADY_READY" if not args.dry_run else "DRY_RUN_READY"
        return report

    api_obj = api or HuggingFaceMetadataAPI()
    info = api_obj.model_info(DEFAULT_MODEL_REPO, revision=DEFAULT_MODEL_REVISION, files_metadata=True)
    if info.sha != DEFAULT_MODEL_REVISION:
        raise SetupError(f"Hugging Face returned revision {info.sha}, expected pinned {DEFAULT_MODEL_REVISION}")
    files = _selected_files(info)
    raw_complete, packed_complete = _store_state(raw_dir, packed_dir, model_dir)
    needed = disk_requirements(files, model_dir, raw_dir, packed_dir,
                               dependency_dir=dependency_dir, dependency_cache_dir=dependency_cache_dir,
                               include_dependencies=not args.skip_dependencies,
                               raw_complete=raw_complete, packed_complete=packed_complete)
    check_disk_space(needed, paths, dependency_dir, dependency_cache_dir)
    plan = {"repo_id": DEFAULT_MODEL_REPO, "revision": DEFAULT_MODEL_REVISION,
            "checkpoint_bytes": sum(int(file.size) for file in files),
            "raw_store_bytes": N_LAYERS * N_EXPERTS * RAW_SLOT_BYTES,
            "ps4_store_bytes": N_LAYERS * N_EXPERTS * PS4_SLOT_BYTES,
            "paths": {k: str(v) for k, v in paths.items()}, "raw_store_complete": raw_complete,
            "ps4_store_complete": packed_complete, "disk_required_bytes_by_volume": needed}
    if args.dry_run:
        plan["status"] = "DRY_RUN_OK"
        plan["hardware_preflight"] = "not run by storage-only dry-run; actual setup checks CUDA, kernel, and capacity before downloading"
        return plan

    print("Checking Windows, CUDA, verified kernel, and GPT-OSS memory capacity before checkpoint download.", flush=True)
    host_preflight = _preflight_host_capacity()
    _download_checkpoint(model_dir, files, api=api_obj, downloader=downloader)
    _validate_checkpoint_only(model_dir)
    if not raw_complete:
        if raw_dir.exists() and any(raw_dir.iterdir()):
            raise SetupError(f"{raw_dir} is nonempty but not a valid raw store; refusing to rebuild over partial data")
        print(f"Building raw original-MXFP4 store at {raw_dir}", flush=True)
        run_command([sys.executable, str(ROOT / "tools" / "build_store.py"),
                     "--model-dir", str(model_dir), "--store-dir", str(raw_dir)], check=True, cwd=ROOT)
    if not packed_complete:
        print(f"Packing and fully verifying PS4 store at {packed_dir}", flush=True)
        command = [sys.executable, str(ROOT / "tools" / "pack_store.py"), "--src", str(raw_dir),
                   "--dst", str(packed_dir), "--verify"]
        run_command(command, check=True, cwd=ROOT)
    report = check_existing(model_dir, packed_dir, hardware=hardware_check)
    config_path = _resolve_config_path(args.config)
    _save_config_if_needed(args, config_path, {"model_dir": str(model_dir), "store_dir": str(packed_dir),
                                               "raw_store_dir": str(raw_dir)})
    plan.update(report)
    plan.update(host_preflight)
    plan["status"] = "SETUP_COMPLETE"
    plan["payload_verification"] = "PS4 store fully checked by tools/pack_store.py --verify; checkpoint payload hashes were not recomputed"
    plan["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return plan


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--data-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--raw-store-dir")
    ap.add_argument("--store-dir")
    ap.add_argument("--config", default=str(ROOT / CONFIG_NAME))
    ap.add_argument("--dependency-dir", help=argparse.SUPPRESS)
    ap.add_argument("--dependency-cache-dir", help=argparse.SUPPRESS)
    ap.add_argument("--check", action="store_true", help="validate an existing GPT-OSS model/store and hardware without downloading")
    ap.add_argument("--dry-run", action="store_true", help="show pinned download and exact disk plan without writing")
    ap.add_argument("--skip-dependencies", action="store_true", help="verify installed dependency versions; never install packages")
    ap.add_argument("--json", action="store_true", help="print the full machine-readable diagnostic report")
    return ap


def _format_result(report: dict[str, Any]) -> str:
    status = report.get("status", "UNKNOWN")
    if status == "DRY_RUN_OK":
        lines = ["Storage preflight passed for the pinned original-MXFP4 GPT-OSS 120B checkpoint.",
                 f"Checkpoint: {report['checkpoint_bytes'] / GIB:.1f} GiB; raw store: {report['raw_store_bytes'] / GIB:.1f} GiB; "
                 f"PS4 store: {report['ps4_store_bytes'] / GIB:.1f} GiB.",
                 "Hardware will be checked after dependencies, before model download."]
        for name, path in report["paths"].items():
            lines.append(f"{name.replace('_', ' ')}: {path}")
        for volume, required in report["disk_required_bytes_by_volume"].items():
            lines.append(f"Minimum free space on {volume}: {required / GIB:.1f} GiB")
        return "\n".join(lines)
    if status == "CHECK_OK":
        model = report["model"]
        plan = report.get("plan", {})
        lines = ["GPT-OSS model and PS4 store metadata and file sizes are valid; stored payload hashes were not recomputed.",
                 f"Model: {model['paths']['model_dir']}", f"Store: {model['paths']['store_dir']}"]
        if plan:
            lines.append(f"Hardware capacity: {'ready' if plan.get('supported') else plan.get('reason', 'unsupported')}.")
        return "\n".join(lines)
    if status in {"SETUP_COMPLETE", "ALREADY_READY", "DRY_RUN_READY"}:
        paths = report.get("paths") or report.get("model", {}).get("paths", {})
        model_path = paths.get("model_dir", "")
        store_path = paths.get("store_dir", "")
        if status == "SETUP_COMPLETE":
            lines = ["GPT-OSS 120B setup completed. The pinned checkpoint, raw MXFP4 store, and verified PS4 store are ready.",
                     f"Model: {model_path}", f"Raw store: {paths.get('raw_store_dir', '')}", f"PS4 store: {store_path}",
                     "Start the local chat with start_neural.bat."]
            return "\n".join(lines)
        label = ("Existing GPT-OSS 120B model and PS4 store passed metadata and file-size checks; payload hashes were not recomputed."
                 if status == "ALREADY_READY" else
                 "Existing GPT-OSS model and PS4 store passed metadata and file-size checks; payload hashes were not recomputed.")
        return "\n".join((label, f"Model: {model_path}", f"PS4 store: {store_path}"))
    return f"Setup status: {status}"


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        _assert_python()
        report = run_setup(args)
        print(json.dumps(report, indent=2, default=str) if args.json else _format_result(report))
        return 0
    except (SetupError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Neural setup: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
