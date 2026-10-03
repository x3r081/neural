import json

import pytest

from neural_runtime import advisor as a


def hardware(**overrides):
    hw = {"platform": "Windows", "cuda_available": True, "bf16_supported": True, "gpu_is_integrated": False,
          "gpu_name": "Test GPU", "gpu_total_bytes": 12 * a.GIB, "gpu_free_bytes": 11 * a.GIB,
          "ram_total_bytes": 64 * a.GIB, "ram_available_bytes": 59 * a.GIB,
          "physical_cores": 8, "cpu_signature": "test", "native_kernel": {
              "available": True, "kernel_kind": "avx512-persistent", "sha256": "unmatched"}}
    hw.update(overrides)
    return hw


def bandwidth(cpu=40, gpu=900):
    return {"cpu": {"gbps_low": cpu, "gbps_high": cpu, "evidence": "ASSUMED", "note": "test"},
            "gpu": {"gbps_low": gpu, "gbps_high": gpu, "evidence": "ASSUMED", "note": "test"}}


def by_name(report, fragment):
    return next(row for row in report["recommendations"] + report["excluded"] if fragment in row["name"])


def test_dedicated_advice_does_not_claim_busy_machine_can_start_now():
    catalog = a.load_catalog()
    hw = hardware(gpu_free_bytes=a.GIB, ram_available_bytes=2 * a.GIB)
    report = a.suggest_models(catalog, hw, bandwidth())
    native = by_name(report, "120")
    assert native["status"] == "recommended"
    assert native["memory"]["fits_dedicated"] is True
    assert native["memory"]["fits_now"] is False
    assert any("other model servers" in note for note in native["notes"])
    current = a.suggest_models(catalog, hw, bandwidth(), scenario="current")
    assert current["recommendations"] == []
    assert all(row["speed"]["low_tps"] is None for row in current["excluded"])


@pytest.mark.parametrize("changes", [{"cuda_available": False}, {"bf16_supported": False}, {"gpu_is_integrated": True}])
def test_unsupported_hardware_never_recommends_or_fabricates_zero_speed(changes):
    report = a.suggest_models(a.load_catalog(), hardware(**changes), bandwidth())
    assert not report["recommendations"]
    assert all(row["speed"]["high_tps"] is None for row in report["excluded"])


def test_missing_kernel_excludes_native_but_preserves_reference_candidates():
    report = a.suggest_models(a.load_catalog(), hardware(native_kernel={"available": False}), bandwidth())
    assert by_name(report, "120")["status"] == "blocked"
    assert any(row["support_tier"] == "reference" for row in report["recommendations"])
    assert report["default_model"]["id"] == "gpt-oss-120b"
    assert report["default_model"]["status"] == "blocked"


def test_reference_capacity_only_counts_whole_resident_layers():
    for model in a.load_catalog()["models"]:
        if model["support_tier"] == "reference":
            placement = a._placement(model, hardware(), 16384, True)
            layer_bytes = model["storage_bytes"]["expert_all"] // model["geometry"]["expert_layers"]
            assert round(placement["gpu_expert_gib"] * a.GIB) % layer_bytes == 0
    report = a.suggest_models(a.load_catalog(), hardware(platform="Linux"), bandwidth())
    assert by_name(report, "120")["status"] == "blocked"


def test_missing_bandwidth_is_unknown_and_never_uses_another_models_measurement():
    report = a.suggest_models(a.load_catalog(), hardware(), bandwidth(gpu=None))
    assert report["recommendations"]
    for row in report["recommendations"]:
        assert row["speed"]["low_tps"] is None
        assert row["speed"]["measured"] is None
        assert row["speed"]["evidence"] == "INSUFFICIENT_BANDWIDTH_DATA"


def test_more_context_costs_memory_and_does_not_raise_bandwidth_only_throughput():
    catalog = a.load_catalog()
    short = a.suggest_models(catalog, hardware(), bandwidth(), context=8192)
    long = a.suggest_models(catalog, hardware(), bandwidth(), context=32768)
    for row in short["recommendations"]:
        later = next(x for x in long["recommendations"] + long["excluded"] if x["id"] == row["id"])
        assert later["memory"]["kv_gib"] >= row["memory"]["kv_gib"]
        if later["speed"]["low_tps"] is not None:
            assert later["speed"]["low_tps"] <= row["speed"]["low_tps"]


def test_task_ranking_prefers_verified_native_over_faster_unverified_theory():
    report = a.suggest_models(a.load_catalog(), hardware(), bandwidth())
    assert report["recommendations"][0]["support_tier"] == "optimized"
    assert all(row["speed"]["evidence"] == "CALCULATED_THEORETICAL" for row in report["recommendations"])
    assert all(row["speed"]["confidence"] == "low" for row in report["recommendations"])
    assert all(row["speed"]["measured"] is None for row in report["recommendations"])


def test_only_matching_host_and_kernel_can_show_historical_observations():
    data = json.loads((a.ROOT / "benchmarks/gptoss_agnostic_20261003/native_abba.json").read_text())
    hw = data["hardware"]
    report = a.suggest_models(a.load_catalog(), hw, bandwidth())
    native = by_name(report, "120")
    assert native["speed"]["measured"]["low_tps"] == pytest.approx(16.89084347818377)
    assert "2597" in native["speed"]["measured"]["label"]
    assert all(row["speed"]["measured"] is None for row in report["recommendations"] if row["support_tier"] == "reference")
    for altered in (dict(hw, cpu_signature="different"), dict(hw, gpu_name="different"),
                    dict(hw, native_kernel={**hw["native_kernel"], "sha256": "different"})):
        assert by_name(a.suggest_models(a.load_catalog(), altered, bandwidth()), "120")["speed"]["measured"] is None


@pytest.mark.parametrize("value", [0, 127, True, 1_048_577])
def test_invalid_context_is_rejected_before_probing(value):
    with pytest.raises(ValueError, match="context"):
        a.suggest_models(a.load_catalog(), hardware(), bandwidth(), context=value)


def test_cli_recommend_does_not_require_a_checkpoint_or_local_config(monkeypatch, capsys):
    from neural_runtime import __main__ as cli
    monkeypatch.setattr(cli, "model_paths", lambda args: pytest.fail("must not inspect a checkpoint"))
    report = a.suggest_models(a.load_catalog(), hardware(), bandwidth())
    monkeypatch.setattr(a, "build_report", lambda **kwargs: report)
    assert cli.main(["recommend", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1


def test_cli_open_without_server_is_rejected():
    from neural_runtime import __main__ as cli
    with pytest.raises(ValueError, match="requires --serve"):
        cli.main(["recommend", "--open"])
