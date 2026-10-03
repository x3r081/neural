import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_runtime import setup


def _args(tmp_path, **overrides):
    values = dict(data_dir=str(tmp_path / "data"), model_dir=str(tmp_path / "data" / "model"),
                  raw_store_dir=str(tmp_path / "data" / "raw"), store_dir=str(tmp_path / "data" / "ps4"),
                  config=str(tmp_path / "neural.local.json"), dependency_dir=None, dependency_cache_dir=None,
                  check=False, dry_run=False,
                  skip_dependencies=True)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_selected_files_only_includes_original_top_level_payload():
    rows = [
        SimpleNamespace(rfilename="config.json", size=10),
        SimpleNamespace(rfilename="model.safetensors.index.json", size=20),
        SimpleNamespace(rfilename="model-00001-of-00001.safetensors", size=30),
        SimpleNamespace(rfilename="original/model.safetensors", size=100),
        SimpleNamespace(rfilename="metal/model.safetensors", size=100),
        SimpleNamespace(rfilename="chat_template.jinja", size=5),
        SimpleNamespace(rfilename="USAGE_POLICY", size=6),
        SimpleNamespace(rfilename="../escape.safetensors", size=100),
        SimpleNamespace(rfilename="model.gguf", size=100),
        SimpleNamespace(rfilename="tokenizer.json", size=40),
    ]
    assert [row.rfilename for row in setup._selected_files(SimpleNamespace(siblings=rows))] == [
        "config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors",
        "chat_template.jinja", "USAGE_POLICY", "tokenizer.json"]


def test_disk_requirements_are_partitioned_by_volume_and_account_for_existing_data(tmp_path, monkeypatch):
    model, raw, ps4, repo = (tmp_path / name for name in ("M", "R", "P", "V"))
    monkeypatch.setattr(setup, "_same_volume_key", lambda p: {model: "M", raw: "R", ps4: "P", repo: "V"}[Path(p)])
    files = [SimpleNamespace(rfilename="config.json", size=101)]
    monkeypatch.setattr(setup, "_file_bytes", lambda p: 101 if Path(p).name == "config.json" else 0)
    got = setup.disk_requirements(files, model, raw, ps4, dependency_dir=repo, dependency_cache_dir=repo,
                                  include_dependencies=True, raw_complete=True)
    assert got == {"M": 0, "P": setup.N_LAYERS * setup.N_EXPERTS * setup.PS4_SLOT_BYTES + setup.STORE_BIAS_RESERVE_BYTES // 2,
                   "V": setup.DEPENDENCY_RESERVE_BYTES}


def test_non_gptoss_custom_config_is_preserved_with_instructions(tmp_path):
    custom_model = tmp_path / "custom"
    custom_model.mkdir()
    (custom_model / "config.json").write_text('{"model_type":"qwen3_moe"}', encoding="utf-8")
    config = tmp_path / "neural.local.json"
    config.write_text(json.dumps({"model_dir": str(custom_model), "store_dir": "somewhere"}), encoding="utf-8")
    before = config.read_bytes()
    with pytest.raises(setup.SetupError, match="preserved"):
        setup._select_paths(_args(tmp_path, data_dir=None, model_dir=None, store_dir=None, raw_store_dir=None), config)
    assert config.read_bytes() == before


@pytest.mark.parametrize(("field", "bad_value"), [
    ("model_dir", 42), ("store_dir", True), ("raw_store_dir", ["bad", "path"]),
])
def test_config_path_fields_must_be_nonempty_strings(tmp_path, field, bad_value):
    config = tmp_path / "neural.local.json"
    config.write_text(json.dumps({"model_dir": str(tmp_path / "model"), field: bad_value}), encoding="utf-8")
    with pytest.raises(setup.SetupError, match="non-empty path string"):
        setup._select_paths(_args(tmp_path), config)


def test_dry_run_does_not_download_build_or_write_config(tmp_path, monkeypatch):
    rows = [SimpleNamespace(rfilename=n, size=s) for n, s in (
        ("config.json", 1), ("model.safetensors.index.json", 1), ("model-00001.safetensors", 2))]
    api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha=setup.DEFAULT_MODEL_REVISION, siblings=rows))
    monkeypatch.setattr(setup, "_store_state", lambda *a: (False, False))
    monkeypatch.setattr(setup, "check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(setup, "_dependency_versions_ok", lambda: (True, "ok"))
    args = _args(tmp_path, dry_run=True)
    result = setup.run_setup(args, api=api, downloader=lambda *a, **kw: pytest.fail("download called"),
                             run_command=lambda *a, **kw: pytest.fail("store builder called"), hardware_check=False)
    assert result["status"] == "DRY_RUN_OK"
    assert not Path(args.config).exists()


def test_setup_invokes_existing_builder_and_verified_packer_then_writes_config(tmp_path, monkeypatch):
    rows = [SimpleNamespace(rfilename=n, size=s) for n, s in (
        ("config.json", 1), ("model.safetensors.index.json", 1), ("model-00001.safetensors", 2))]
    api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha=setup.DEFAULT_MODEL_REVISION, siblings=rows))
    monkeypatch.setattr(setup, "_store_state", lambda *a: (False, False))
    monkeypatch.setattr(setup, "check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(setup, "_dependency_versions_ok", lambda: (True, "ok"))
    monkeypatch.setattr(setup, "_validated_existing_store", lambda *a: False)
    monkeypatch.setattr(setup, "_preflight_host_capacity", lambda: {"hardware_preflight": {"passed": True}})
    monkeypatch.setattr(setup, "_checkpoint_model_type", lambda p: None)
    monkeypatch.setattr(setup, "_atomic_config", lambda path, values: Path(path).write_text(json.dumps(values)))
    monkeypatch.setattr(setup, "_validate_checkpoint_only", lambda path: None)
    from neural_runtime import model_spec
    monkeypatch.setattr(model_spec, "inspect_model", lambda *a, **kw: SimpleNamespace())
    calls = []

    def download(repo, *, revision, local_dir, allow_patterns, max_workers):
        assert repo == setup.DEFAULT_MODEL_REPO
        assert revision == setup.DEFAULT_MODEL_REVISION
        for name in allow_patterns:
            target = Path(local_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x" * next(row.size for row in rows if row.rfilename == name))

    def run(command, **kwargs):
        calls.append(command)

    monkeypatch.setattr(setup, "check_existing", lambda *a, **kw: {"model": {"validated": True}})
    result = setup.run_setup(_args(tmp_path), api=api, downloader=download, run_command=run,
                             hardware_check=False)
    assert result["status"] == "SETUP_COMPLETE"
    assert calls[0][1].endswith("build_store.py")
    assert calls[1][1].endswith("pack_store.py") and "--verify" in calls[1]
    assert json.loads(Path(_args(tmp_path).config).read_text())["store_dir"] == str(tmp_path / "data" / "ps4")


def test_config_write_is_atomic_and_preserves_other_fields(tmp_path):
    config = tmp_path / "neural.local.json"
    config.write_text('{"custom":true,"model_dir":"old"}', encoding="utf-8")
    setup._atomic_config(config, {"model_dir": "new", "store_dir": "ps4"})
    assert json.loads(config.read_text()) == {"custom": True, "model_dir": "new", "store_dir": "ps4"}
    assert not config.with_name(config.name + ".tmp").exists()


def test_checkpoint_only_validation_checks_pinned_hf_revision_without_a_store(tmp_path, monkeypatch):
    from neural_runtime import model_spec

    def read_json(path, label):
        if Path(path).name == "config.json":
            return {"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}}, "config-hash"
        return {"weight_map": {"tensor": "model-00001.safetensors"}}, "index-hash"
    monkeypatch.setattr(model_spec, "_read_json", read_json)
    monkeypatch.setattr(model_spec, "_geometry", lambda config: {"layers": 36, "experts": 128, "top_k": 4,
        "hidden": 2880, "intermediate": 2880, "num_heads": 64, "num_kv_heads": 8, "head_dim": 64, "window": 128})
    monkeypatch.setattr(model_spec, "_checkpoint_memory", lambda root: ({"expert_weight_dtypes": ["U8"],
        "expert_bias_dtypes": ["BF16"]}, {"safetensors_index": str(root / "model.safetensors.index.json")}))
    monkeypatch.setattr(model_spec, "_verify_download_revision", lambda *a: {"config.json": setup.DEFAULT_MODEL_REVISION})
    monkeypatch.setattr(model_spec, "_verified_source_dtype", lambda *a: "BF16")
    setup._validate_checkpoint_only(tmp_path)
    monkeypatch.setattr(model_spec, "_verify_download_revision", lambda *a: {"config.json": "wrong-revision"})
    with pytest.raises(setup.SetupError, match="pinned revision"):
        setup._validate_checkpoint_only(tmp_path)


def test_existing_valid_config_is_not_rewritten_without_explicit_path_choice(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt_oss"}', encoding="utf-8")
    config = tmp_path / "neural.local.json"
    original = {"model_dir": str(model), "store_dir": "existing-ps4", "other": "keep"}
    config.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(setup, "_checkpoint_model_type", lambda path: "gpt_oss")
    setup._save_config_if_needed(_args(tmp_path, data_dir=None, model_dir=None, store_dir=None,
                                       raw_store_dir=None), config, {"model_dir": "new", "store_dir": "new"})
    assert json.loads(config.read_text()) == original
