from __future__ import annotations

import pytest

from neural_runtime import advisor_hardware as ah


def _hw(**overrides):
    result = {
        "cpu_signature": "Windows|x86_64|Test CPU",
        "physical_cores": 8,
        "logical_cores": 16,
        "ram_total_bytes": 32 * ah.GIB,
        "ram_available_bytes": 8 * ah.GIB,
        "gpu_name": "NVIDIA GeForce RTX 3080 Ti",
        "gpu_total_bytes": 12 * ah.GIB,
        "gpu_free_bytes": 10 * ah.GIB,
    }
    result.update(overrides)
    return result


def test_gpu_exact_known_name_reports_published_peak_not_measured_rate():
    profile = ah.bandwidth_profile(_hw(), cpu_bandwidth_gbps=80.0)
    gpu = profile["gpu"]
    assert gpu["gbps_low"] == gpu["gbps_high"] == 912.0
    assert gpu["evidence"] == "PUBLISHED_PEAK"
    assert "not expected" in gpu["note"]
    assert gpu["sources"] == [
        "https://www.gigabyte.com/us/Graphics-Card/GV-N308TVISION-OC-12GD/sp"
    ]


def test_gpu_device_properties_produce_calculated_peak():
    profile = ah.bandwidth_profile(
        _hw(gpu_name="Previously Unknown GPU", gpu_memory_clock_rate_khz=9_501_000,
            gpu_memory_bus_width_bits=384),
        cpu_bandwidth_gbps=80.0,
    )
    gpu = profile["gpu"]
    assert gpu["gbps_low"] == pytest.approx(912.096)
    assert gpu["gbps_high"] == pytest.approx(912.096)
    assert gpu["evidence"] == "CALCULATED_DEVICE_PEAK"
    assert "memory_clock_rate_kHz" in gpu["note"]
    assert len(gpu["sources"]) == 2


def test_unknown_gpu_does_not_match_nearby_model_by_substring():
    profile = ah.bandwidth_profile(_hw(gpu_name="NVIDIA GeForce RTX 3080"),
                                   cpu_bandwidth_gbps=80.0)
    assert profile["gpu"]["gbps_low"] is None
    assert profile["gpu"]["evidence"] == "UNKNOWN"


def test_user_overrides_are_assumptions_and_validate_bounds():
    profile = ah.bandwidth_profile(_hw(), cpu_bandwidth_gbps=80, gpu_bandwidth_gbps=700)
    assert profile["cpu"]["gbps_low"] == profile["cpu"]["gbps_high"] == 80.0
    assert profile["gpu"]["gbps_low"] == profile["gpu"]["gbps_high"] == 700.0
    assert profile["cpu"]["evidence"] == profile["gpu"]["evidence"] == "ASSUMED_USER_OVERRIDE"
    for bad in (True, 0, 0.01, float("nan"), float("inf"), 100_001, "80"):
        with pytest.raises(ValueError):
            ah.bandwidth_profile(_hw(), cpu_bandwidth_gbps=bad)


def test_cpu_copy_proxy_cached_for_sixty_seconds(monkeypatch):
    ah._cpu_cache.clear()
    calls = []

    def measure(hw):
        calls.append(hw["cpu_signature"])
        return {"gbps_low": 50.0, "gbps_high": 60.0, "evidence": "MEASURED_COPY_PROXY",
                "note": "proxy", "sources": []}

    tick = [10.0]
    monkeypatch.setattr(ah, "_measure_cpu_copy_proxy", measure)
    monkeypatch.setattr(ah.time, "monotonic", lambda: tick[0])
    first = ah.bandwidth_profile(_hw(), gpu_bandwidth_gbps=900)["cpu"]
    tick[0] += 59.9
    second = ah.bandwidth_profile(_hw(), gpu_bandwidth_gbps=900)["cpu"]
    assert first == second
    assert calls == ["Windows|x86_64|Test CPU"]
    tick[0] += 0.2
    ah.bandwidth_profile(_hw(), gpu_bandwidth_gbps=900)
    assert len(calls) == 2


def test_cpu_copy_proxy_guard_skips_when_ram_is_low():
    result = ah._measure_cpu_copy_proxy(_hw(ram_available_bytes=ah.GIB - 1))
    assert result["gbps_low"] is result["gbps_high"] is None
    assert result["evidence"] == "UNKNOWN"
    assert "below the 1 GiB" in result["note"]


def test_probe_uses_known_then_portable_manifest_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(ah.gptoss_platform, "ROOT", tmp_path)
    calls = []

    def probe(device, *, native_manifest):
        calls.append((device, native_manifest))
        return {"native_kernel": {"available": native_manifest.name == "portable_cpu_build.json"},
                "gpu_name": "Unlisted Device", "ram_available_bytes": None}

    monkeypatch.setattr(ah.gptoss_platform, "probe_hardware", probe)
    report = ah.probe_advisor_hardware("cuda:2")
    assert [p.name for _, p in calls] == ["gptoss_known_build.json", "portable_cpu_build.json"]
    assert all(device == "cuda:2" for device, _ in calls)
    assert report["native_kernel"]["available"] is True
    assert "bandwidth" not in report


def test_probe_does_not_use_portable_manifest_when_known_build_is_available(monkeypatch, tmp_path):
    monkeypatch.setattr(ah.gptoss_platform, "ROOT", tmp_path)
    calls = []

    def probe(device, *, native_manifest):
        calls.append(native_manifest)
        return {"native_kernel": {"available": True}, "gpu_name": None,
                "ram_available_bytes": None}

    monkeypatch.setattr(ah.gptoss_platform, "probe_hardware", probe)
    ah.probe_advisor_hardware()
    assert [p.name for p in calls] == ["gptoss_known_build.json"]


def test_platform_probe_includes_cuda_memory_properties_without_allocation(monkeypatch):
    from types import SimpleNamespace
    import neural_runtime.gptoss_platform as gp

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(index):
            assert index == 0
            return SimpleNamespace(name="Device", major=8, minor=6,
                                   memory_clock_rate=9_501_000,
                                   memory_bus_width=384, is_integrated=0)

        @staticmethod
        def mem_get_info(index):
            assert index == 0
            return 3 * ah.GIB, 12 * ah.GIB

    fake_torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test"), cuda=FakeCuda())
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(gp, "_native_library_status", lambda *args: {"available": False})
    report = gp.probe_hardware()
    assert report["gpu_memory_clock_rate_khz"] == 9_501_000
    assert report["gpu_memory_bus_width_bits"] == 384
    assert report["gpu_is_integrated"] is False
