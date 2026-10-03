"""Verify an unquantized Qwen3.6 text GGUF against its source safetensors.

Metadata mode reads only SafeTensors headers. Full mode streams one source
tensor at a time, reuses the pinned llama.cpp Qwen conversion transforms, and
compares the exact BF16/F32 GGUF payload. It does not hash model weight files.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA_ROOT = REPO_ROOT / ".tools" / "llama.cpp"
EXPECTED_ARCH = "Qwen3_5MoeForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_5_moe"


def _imports():
    if not (LLAMA_ROOT / "conversion").is_dir():
        raise RuntimeError(f"Pinned llama.cpp source is missing: {LLAMA_ROOT}")
    sys.path.insert(0, str(LLAMA_ROOT))
    sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))
    import numpy as np
    import torch
    import gguf
    from conversion import get_model_class
    return np, torch, gguf, get_model_class


def _load_source(model_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(f"Expected model_type {EXPECTED_MODEL_TYPE}, got {config.get('model_type')!r}")
    if config.get("architectures") != [EXPECTED_ARCH]:
        raise ValueError(f"Unexpected model architectures: {config.get('architectures')!r}")
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("SafeTensors index has no non-empty weight_map")
    return config, {name: model_dir / shard for name, shard in weight_map.items()}


def _source_metadata(model_dir: Path, weight_map: dict[str, Path]) -> dict[str, Any]:
    from safetensors import safe_open

    dtype_counts: collections.Counter[str] = collections.Counter()
    name_counts: collections.Counter[str] = collections.Counter()
    actual_names: set[str] = set()
    for shard in sorted(set(weight_map.values())):
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                actual_names.add(name)
                dtype_counts[sf.get_slice(name).get_dtype()] += 1
                if "visual." in name or "vision." in name or "audio." in name:
                    name_counts["vision_or_audio"] += 1
                elif name.startswith(("mtp.", "model.mtp.")) or ".mtp." in name:
                    name_counts["mtp"] += 1
                else:
                    name_counts["other"] += 1
    if actual_names != set(weight_map):
        raise ValueError(
            f"Index/header tensor-name mismatch: index={len(weight_map)}, headers={len(actual_names)}"
        )
    return {
        "source_tensor_count": len(actual_names),
        "source_dtype_counts": dict(sorted(dtype_counts.items())),
        "source_categories": dict(sorted(name_counts.items())),
        "index_total_size_bytes": json.loads(
            (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        ).get("metadata", {}).get("total_size"),
        "weight_files": [
            {"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(set(weight_map.values()))
        ],
        "weights_hashed": False,
    }


def _new_converter_model(config: dict[str, Any], include_mtp: bool):
    np, torch, gguf, get_model_class = _imports()
    arch = config["architectures"][0]
    cls = get_model_class(arch)
    text_hparams = {**config, **config.get("text_config", {})}
    layer_count = int(text_hparams["num_hidden_layers"])

    # Build only the converter state used by filter_tensors()/modify_tensors().
    # This avoids indexing or materializing the 72 GB checkpoint in the verifier.
    model = cls.__new__(cls)
    model.hparams = text_hparams
    model.block_count = layer_count + (int(text_hparams.get("mtp_num_hidden_layers", 0)) if include_mtp else 0)
    model.tensor_map = gguf.get_tensor_name_map(cls.model_arch, model.block_count)
    model.no_mtp = not include_mtp
    model.mtp_only = False
    model._original_block_count = layer_count
    model.opt_num_mtp_layers = 0
    cls.no_mtp = not include_mtp
    cls.mtp_only = False
    cls._original_block_count = layer_count
    cls.opt_num_mtp_layers = 0
    model.fuse_gate_up_exps = False
    model.fuse_qkv = False
    model._gate_exp_buffer = {}
    model._up_exp_buffer = {}
    model._q_buffer = {}
    model._k_buffer = {}
    model._v_buffer = {}
    model._q_bias_buffer = {}
    model._k_bias_buffer = {}
    model._v_bias_buffer = {}
    return np, torch, gguf, model


def _filter_source_name(model, raw_name: str) -> str | None:
    selected = type(model).filter_tensors((raw_name, lambda: None))
    if selected is None:
        return None
    name, _ = selected
    if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
        return None
    return name


def _block_id(name: str) -> int | None:
    return next((int(part) for part in name.split(".") if part.isdecimal()), None)


def _plan_outputs(model, torch, name: str, shape: list[int]) -> list[tuple[str, Any]]:
    meta_tensor = torch.empty(tuple(shape), dtype=torch.float32, device="meta")
    plans = list(model.modify_tensors(meta_tensor, name, _block_id(name)))
    return [(output_name, tensor) for output_name, tensor in plans]


def _full_verify(model_dir: Path, gguf_path: Path, weight_map: dict[str, Path],
                 config: dict[str, Any], include_mtp: bool) -> dict[str, Any]:
    from safetensors import safe_open

    np, torch, gguf, model = _new_converter_model(config, include_mtp)
    reader = gguf.GGUFReader(str(gguf_path))
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensor_map) != len(reader.tensors):
        raise ValueError("GGUF contains duplicate tensor names")

    arch_field = reader.get_field("general.architecture")
    if arch_field is None or arch_field.contents() != "qwen35moe":
        raise ValueError("GGUF general.architecture is not qwen35moe")

    qtype_counts = collections.Counter(t.tensor_type.name for t in reader.tensors)
    unsupported_qtypes = set(qtype_counts) - {"BF16", "F32"}
    if unsupported_qtypes:
        raise ValueError(f"GGUF contains non-BF16/F32 tensors: {sorted(unsupported_qtypes)}")

    expected: dict[str, tuple[str, str, Any]] = {}
    skipped_vision = skipped_mtp = skipped_known = 0
    by_shard: dict[Path, list[str]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    # First pass uses SafeTensors slice metadata plus meta tensors only.
    for shard, names in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for raw_name in names:
                name = _filter_source_name(model, raw_name)
                if name is None:
                    if "visual." in raw_name or "vision." in raw_name or "audio." in raw_name:
                        skipped_vision += 1
                    elif raw_name.startswith(("mtp.", "model.mtp.")) or ".mtp." in raw_name:
                        skipped_mtp += 1
                    else:
                        skipped_known += 1
                    continue
                outputs = _plan_outputs(model, torch, name, sf.get_slice(raw_name).get_shape())
                if not outputs:
                    raise ValueError(f"Pinned converter buffered/omitted an unexpected text tensor: {raw_name}")
                for output_name, planned_tensor in outputs:
                    if output_name in expected:
                        raise ValueError(f"Multiple source tensors map to GGUF tensor {output_name}")
                    expected[output_name] = (raw_name, name, planned_tensor)

    expected_names, actual_names = set(expected), set(tensor_map)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            f"GGUF/source tensor-name mismatch: missing={len(missing)} {missing[:8]!r}; "
            f"unexpected={len(unexpected)} {unexpected[:8]!r}"
        )

    checked = 0
    checked_elements = 0
    mismatches: list[dict[str, Any]] = []
    for shard, names in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for raw_name in names:
                name = _filter_source_name(model, raw_name)
                if name is None:
                    continue
                # Qwen converter prepare_tensors widens BF16 to FP32 before its
                # architecture-specific modify_tensors() transforms.
                source = sf.get_tensor(raw_name).to(torch.float32)
                for output_name, expected_tensor in model.modify_tensors(source, name, _block_id(name)):
                    destination = tensor_map[output_name]
                    expected_np = np.ascontiguousarray(expected_tensor.detach().cpu().numpy(), dtype=np.float32)
                    if destination.tensor_type == gguf.GGMLQuantizationType.F32:
                        equal = np.array_equal(destination.data, expected_np, equal_nan=True)
                    elif destination.tensor_type == gguf.GGMLQuantizationType.BF16:
                        expected_bytes = gguf.quants.quantize(
                            expected_np, gguf.GGMLQuantizationType.BF16
                        )
                        equal = destination.data.shape == expected_bytes.shape and np.array_equal(
                            destination.data, expected_bytes
                        )
                    else:  # Guard in case the allowed qtype set is ever changed above.
                        equal = False
                    if not equal and len(mismatches) < 8:
                        mismatches.append({
                            "source": raw_name,
                            "gguf": output_name,
                            "source_shape": list(source.shape),
                            "gguf_shape": list(destination.data.shape),
                            "gguf_type": destination.tensor_type.name,
                        })
                    checked += 1
                    checked_elements += expected_np.size
                del source
    if checked != len(expected):
        raise ValueError(f"Internal verifier count mismatch: compared={checked}, expected={len(expected)}")
    if mismatches:
        raise ValueError(f"Numerical mismatches found: {mismatches!r}")

    mtp_field = reader.get_field("qwen35moe.nextn_predict_layers")
    if include_mtp and mtp_field is None:
        raise ValueError("MTP was requested but GGUF has no nextn_predict_layers metadata")
    if not include_mtp and mtp_field is not None:
        raise ValueError("MTP was excluded but GGUF contains nextn_predict_layers metadata")

    return {
        "gguf_path": str(gguf_path.resolve()),
        "gguf_bytes": gguf_path.stat().st_size,
        "gguf_architecture": arch_field.contents(),
        "gguf_tensor_count": len(tensor_map),
        "gguf_tensor_type_counts": dict(sorted(qtype_counts.items())),
        "source_tensor_outputs_compared": checked,
        "source_tensor_elements_compared": checked_elements,
        "skipped_source_vision_or_audio_tensors": skipped_vision,
        "skipped_source_mtp_tensors": skipped_mtp,
        "skipped_source_known_nonweights": skipped_known,
        "mtp_included": include_mtp,
        "exact_payload_comparison": True,
        "weight_files_hashed": False,
        "status": "PASS_EXACT_BF16_OR_F32_PAYLOADS",
    }


def _plan_only(model_dir: Path, weight_map: dict[str, Path],
               config: dict[str, Any], include_mtp: bool) -> dict[str, Any]:
    """Plan output names/shapes through pinned transforms using meta tensors only."""
    from safetensors import safe_open

    _, torch, _, model = _new_converter_model(config, include_mtp)
    by_shard: dict[Path, list[str]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    output_names: set[str] = set()
    examples: dict[str, dict[str, Any]] = {}
    skipped_vision = skipped_mtp = skipped_known = 0
    for shard, names in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for raw_name in names:
                name = _filter_source_name(model, raw_name)
                if name is None:
                    if "visual." in raw_name or "vision." in raw_name or "audio." in raw_name:
                        skipped_vision += 1
                    elif raw_name.startswith(("mtp.", "model.mtp.")) or ".mtp." in raw_name:
                        skipped_mtp += 1
                    else:
                        skipped_known += 1
                    continue
                outputs = _plan_outputs(model, torch, name, sf.get_slice(raw_name).get_shape())
                if not outputs:
                    raise ValueError(f"Pinned converter buffered/omitted an unexpected text tensor: {raw_name}")
                for output_name, planned_tensor in outputs:
                    if output_name in output_names:
                        raise ValueError(f"Multiple source tensors map to GGUF tensor {output_name}")
                    output_names.add(output_name)
                    if len(examples) < 12:
                        examples[output_name] = {
                            "source": raw_name,
                            "source_shape": list(sf.get_slice(raw_name).get_shape()),
                            "output_shape": list(planned_tensor.shape),
                        }
    return {
        "planned_gguf_tensor_count": len(output_names),
        "example_name_shape_mappings": examples,
        "skipped_source_vision_or_audio_tensors": skipped_vision,
        "skipped_source_mtp_tensors": skipped_mtp,
        "skipped_source_known_nonweights": skipped_known,
        "mtp_included": include_mtp,
        "payloads_read": False,
        "weights_hashed": False,
        "status": "CONVERTER_METADATA_PLAN_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(r"F:\Models\Qwen3.6-35B-A3B"))
    parser.add_argument("--gguf", type=Path, help="converted text GGUF; omit for header-only metadata mode")
    parser.add_argument("--include-mtp", action="store_true", help="expect default converter behavior (MTP included)")
    parser.add_argument("--plan-only", action="store_true", help="also run pinned converter name/shape transforms on meta tensors only")
    parser.add_argument("--report", type=Path, help="write the JSON report to this path")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    config, weight_map = _load_source(model_dir)
    report = {
        "source_repo": "Qwen/Qwen3.6-35B-A3B",
        "source_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "source_path": str(model_dir),
        "source_model_type": config["model_type"],
        "source_architecture": config["architectures"][0],
        "text_layers": config["text_config"]["num_hidden_layers"],
        "source_metadata": _source_metadata(model_dir, weight_map),
        "gguf_conversion_commit": "fb4b2737a808a3fb7c2117a498f43815dc9be53e",
    }
    if args.gguf and args.plan_only:
        parser.error("--gguf and --plan-only are separate modes")
    if args.gguf:
        report["conversion_verification"] = _full_verify(
            model_dir, args.gguf.resolve(), weight_map, config, args.include_mtp
        )
    elif args.plan_only:
        report["conversion_plan"] = _plan_only(
            model_dir, weight_map, config, args.include_mtp
        )
        report["status"] = "SOURCE_HEADER_AND_CONVERTER_PLAN_ONLY"
    else:
        report["status"] = "SOURCE_HEADER_METADATA_ONLY"

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
