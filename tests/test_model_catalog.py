"""Offline validation for the curated official MoE model catalog."""
from __future__ import annotations

import json
from pathlib import Path


CATALOG = Path(__file__).resolve().parents[1] / "neural_runtime" / "model_catalog.json"
EXPECTED_MODELS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ("qwen3_moe", "reference", "architecture-fixtures"),
    "Qwen/Qwen3.6-35B-A3B": ("qwen3_5_moe", "reference", "architecture-fixtures"),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ("mixtral", "reference", "architecture-fixtures"),
    "openai/gpt-oss-120b": ("gpt_oss", "optimized", "full-model"),
}


def _catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_catalog_has_only_curated_supported_official_checkpoints():
    catalog = _catalog()
    assert catalog["schema_version"] == 1
    assert catalog["verified_date"] == "2026-10-03"
    models = {model["hf_repo"]: model for model in catalog["models"]}
    assert set(models) == set(EXPECTED_MODELS)

    for repo, model in models.items():
        registry_key, tier, validation = EXPECTED_MODELS[repo]
        assert model["official"] is True
        assert model["original_format"] is True
        assert model["registry_key"] == registry_key
        assert model["support_tier"] == tier
        assert model["validation"] == validation
        assert model["revision"]
        assert model["native_profile"]
        assert model["task_tags"]
        assert model["quantization"] in {"none", "mxfp4"}
        assert model["storage_bytes"]["total_model"] > 0
        assert model["storage_bytes"]["core_non_expert"] > 0
        assert model["storage_bytes"]["expert_all"] > 0
        assert model["storage_bytes"]["active_experts_per_token"] > 0
        assert (
            model["storage_bytes"]["core_non_expert"]
            + model["storage_bytes"]["expert_all"]
            == model["storage_bytes"]["total_model"]
        )
        assert model["storage_bytes"]["active_experts_per_token"] <= model["storage_bytes"]["expert_all"]
        assert model["geometry"]["runtime_context_tokens"] == 16384
        assert model["geometry"]["checkpoint_context_tokens"] >= model["geometry"]["runtime_context_tokens"]
        assert model["geometry"]["expert_parameters_per_token"] > 0
        assert model["inference_state"]["attention_kv_bytes_per_token"] > 0
        assert model["inference_state"]["recurrent_state_bytes"] >= 0
        assert any(source["kind"] == "config" for source in model["sources"])
        assert any(source["kind"] == "safetensors-index" for source in model["sources"])
        assert all(source["revision"] == model["revision"] for source in model["sources"])
        assert all(source["verified_date"] == catalog["verified_date"] for source in model["sources"])
        assert all(source["url"].startswith("https://huggingface.co/") for source in model["sources"])
        assert all(item["label"] in {"CALCULATED", "MEASURED"} for item in model["byte_derivation"])


def test_gpt_oss_uses_native_mxfp4_and_pinned_memory_split():
    model = next(m for m in _catalog()["models"] if m["hf_repo"] == "openai/gpt-oss-120b")
    memory = model["storage_bytes"]
    state = model["inference_state"]
    assert model["revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    assert model["native_profile"] == "gptoss-120b-mxfp4"
    assert model["original_dtype"].startswith("BF16 core")
    assert model["quantization"] == "mxfp4"
    assert memory["gpu_core"] == 3_176_475_264
    assert memory["cpu_core"] == 1_158_266_880
    assert memory["gpu_core"] + memory["cpu_core"] == memory["core_non_expert"]
    assert state["sliding_layers"] == 18
    assert state["sliding_window"] == 128
    assert state["sliding_kv_bytes_per_token"] > 0
    assert any(source["kind"] == "original-format-index" for source in model["sources"])


def test_qwen36_tracks_nested_text_config_and_recurrent_state():
    model = next(m for m in _catalog()["models"] if m["hf_repo"] == "Qwen/Qwen3.6-35B-A3B")
    assert model["registry_key"] == "qwen3_5_moe"
    assert model["geometry"]["experts_per_token"] == 8
    assert model["geometry"]["shared_expert_intermediate_size"] == 512
    assert model["inference_state"]["recurrent_state_bytes"] > 0
    assert any("vision_config" in source["facts"] for source in model["sources"] if source["kind"] == "config")
    assert any("text_config.model_type" in source["facts"] for source in model["sources"] if source["kind"] == "config")
    assert any("vision encoder" in item["notes"] for item in model["byte_derivation"])
