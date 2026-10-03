import hashlib
import json

import pytest

from neural_runtime import gptoss_platform as gp


def _spec():
    return {
        "model_id": "gpt-oss-test-revision",
        "model_type": "gpt_oss",
        "source_format": {"quant_method": "mxfp4", "source_dtype": "bfloat16",
                          "store_format": "mxfp4_g32_ps4", "weight_preserving": True},
        "geometry": {
            "layers": 4, "experts": 8, "top_k": 2, "hidden": 64,
            "intermediate": 32, "num_heads": 4, "num_kv_heads": 2,
            "head_dim": 8, "layer_types": ["sliding_attention", "full_attention",
                                             "sliding_attention", "full_attention"],
            "window": 128,
        },
        "store": {
            "slot_bytes": 1024 * 1024, "format": "mxfp4_g32_ps4",
            "metadata": {"layout": {"n_layers": 4, "n_experts": 8, "top_k": 2}},
        },
        "context_limits": {"max_position_embeddings": 4096},
        "memory": {"source_core_bytes": 3 * gp.GIB, "gpu_core_bytes": gp.GIB,
                   "kv_bytes_per_element": 2},
    }


def _hardware():
    return {
        "platform": "Windows", "cpu_features": {k: True for k in gp.REQUIRED_GPTOSS_ISA},
        "physical_cores": 12, "logical_cores": 16, "ram_available_bytes": 16 * gp.GIB,
        "cuda_available": True, "bf16_supported": True,
        "gpu_free_bytes": 10 * gp.GIB,
        "native_kernel": {"available": True, "library": "trusted.dll",
                          "kernel_kind": "avx512-persistent",
                          "cpu_features": list(gp.REQUIRED_GPTOSS_ISA),
                          "model_id": "gpt-oss-test-revision"},
    }


def test_plan_derives_kv_pool_threads_and_caps_from_metadata_and_capacity():
    plan = gp.plan_gptoss(_spec(), _hardware(), context=1024, target_pool_gib=0.01,
                          scratch_slots=2, static_slots=3, capbufs=3)
    expected_kv = (2 * 256 + 2 * 1024) * 2 * 2 * 8 * 2
    assert plan["supported"] is True
    assert plan["kv_ring"] == 256
    assert plan["kv_bytes"] == expected_kv
    assert plan["pool_slots"] == 10
    assert plan["usable_slots"] == 8 and plan["adaptive_slots"] == 5
    assert plan["threads"] == 8
    assert plan["capbuf_bytes"] == 3 * 1024 * 1024
    assert plan["source_format"]["quant_method"] == "mxfp4"
    assert plan["store_format"] == "mxfp4_g32_ps4"


def test_unsupported_avx2_only_hardware_requires_explicit_gptoss_fallback():
    hw = _hardware()
    hw["cpu_features"]["AVX512F"] = False
    result = gp.plan_gptoss(_spec(), hw, context=1024)
    assert result["status"] == "unsupported"
    assert "AVX512F" in result["reason"]
    assert "no unvalidated fallback" in result["fallback_requirement"]
    assert "does not claim AVX2" not in result["fallback_requirement"]


def test_metadata_disagreement_and_capacity_overrides_fail_closed():
    bad = _spec()
    bad["store"]["metadata"]["layout"]["n_experts"] = 7
    with pytest.raises(ValueError, match="disagrees"):
        gp.plan_gptoss(bad, _hardware(), context=1024)
    with pytest.raises(MemoryError, match="exceeds free VRAM"):
        gp.plan_gptoss(_spec(), _hardware(), context=1024, pool_gib=20)


def test_native_dll_is_never_loaded_before_isa_and_cpu_signature_gates(tmp_path, monkeypatch):
    lib = tmp_path / "kernel.dll"
    lib.write_bytes(b"not a real DLL")
    manifest = tmp_path / "manifest.json"
    signature = "Windows|AMD64|test CPU"
    manifest.write_text(json.dumps({
        "cpu_features": list(gp.REQUIRED_GPTOSS_ISA), "cpu_signature": signature,
        "model_id": "gpt-oss-test-revision", "library": "kernel.dll",
        "prefill_library": "kernel.dll", "sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
        "prefill_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    calls = []

    def should_not_load(_path):
        calls.append(_path)
        raise AssertionError("must not load an incompatible DLL")

    monkeypatch.setattr(gp.ctypes, "WinDLL", should_not_load, raising=False)
    flags = {name: True for name in gp.REQUIRED_GPTOSS_ISA}
    flags["AVX512F"] = False
    result = gp._native_library_status("Windows", flags, manifest, signature)
    assert not result["available"] and "AVX512F" in result["reason"]
    assert calls == []

    flags["AVX512F"] = True
    result = gp._native_library_status("Windows", flags, manifest, "Windows|AMD64|other CPU")
    assert not result["available"] and "different CPU signature" in result["reason"]
    assert calls == []


def test_native_dll_load_failure_is_reported_without_invoking_symbols(tmp_path, monkeypatch):
    lib = tmp_path / "kernel.dll"
    lib.write_bytes(b"signed local test bytes")
    manifest = tmp_path / "manifest.json"
    signature = "Windows|AMD64|test CPU"
    manifest.write_text(json.dumps({
        "cpu_features": list(gp.REQUIRED_GPTOSS_ISA), "cpu_signature": signature,
        "model_id": "gpt-oss-test-revision", "library": "kernel.dll",
        "prefill_library": "kernel.dll", "sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
        "prefill_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    monkeypatch.setattr(gp, "ROOT", tmp_path)

    def missing_dependency(_path):
        raise OSError("missing OpenMP runtime")

    monkeypatch.setattr(gp.ctypes, "WinDLL", missing_dependency, raising=False)
    result = gp._native_library_status(
        "Windows", {name: True for name in gp.REQUIRED_GPTOSS_ISA}, manifest, signature)
    assert result["available"] is False
    assert "missing OpenMP runtime" in result["reason"]


def test_portable_kernel_requires_and_accepts_exact_parity_evidence(tmp_path, monkeypatch):
    lib = tmp_path / "portable.dll"
    lib.write_bytes(b"portable synthetic DLL")
    report = tmp_path / "parity.json"
    report.write_text(json.dumps({"result": "PASS", "mode": "full", "full_validation": True,
                                  "full_validation_passed": True,
                                  "portable_dll_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
                                  "openmp_transient_regions": True, "persistent_worker_pool": False,
                                  "thread_counts_tested": [1, 8]}),
                      encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "kernel_kind": "portable-avx2-fma", "cpu_features": ["AVX2", "FMA3"],
        "library": "portable.dll", "sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
        "prefill_library": "portable.dll", "prefill_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
        "exact_validation": {"status": "PASS", "report": "parity.json",
                             "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
        "symbols": [], "prefill_symbols": [],
    }), encoding="utf-8")
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    class FakeDll:
        def __getattr__(self, _name):
            return object()
    monkeypatch.setattr(gp.ctypes, "WinDLL", lambda _path: FakeDll(), raising=False)
    features = {name: name in {"AVX2", "FMA3"} for name in gp.REQUIRED_GPTOSS_ISA}
    status = gp._native_library_status("Windows", features, manifest, "different CPU is allowed")
    assert status["available"] is True
    assert status["kernel_kind"] == "portable-avx2-fma"
    assert status["parallel_supported"] is True

    # The planner chooses only this explicitly validated kernel and does not
    # impose the AVX-512 CPU restriction on it.
    hardware = _hardware()
    hardware["cpu_features"] = features
    hardware["native_kernel"] = status | {"model_id": None}
    plan = gp.plan_gptoss(_spec(), hardware, context=1024, target_pool_gib=0.01,
                          scratch_slots=2, static_slots=3, capbufs=3, kv_ring=128)
    assert plan["supported"] is True
    assert plan["kernel_persistent"] is False
    assert plan["threads"] == 8


def test_portable_parity_must_be_full_and_tied_to_exact_candidate(tmp_path, monkeypatch):
    lib = tmp_path / "portable.dll"
    lib.write_bytes(b"portable candidate")
    digest = hashlib.sha256(lib.read_bytes()).hexdigest()
    report = tmp_path / "parity.json"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    monkeypatch.setattr(gp.ctypes, "WinDLL", lambda _path: _FakeAllSymbols(), raising=False)

    def attempt(payload):
        report.write_text(json.dumps(payload), encoding="utf-8")
        manifest.write_text(json.dumps({
            "kernel_kind": "portable-avx2-fma", "cpu_features": ["AVX2", "FMA3"],
            "library": "portable.dll", "sha256": digest,
            "prefill_library": "portable.dll", "prefill_sha256": digest,
            "exact_validation": {"status": "PASS", "report": "parity.json",
                                 "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
        }), encoding="utf-8")
        return gp._native_library_status("Windows", {n: n in {"AVX2", "FMA3"}
                                                         for n in gp.REQUIRED_GPTOSS_ISA},
                                         manifest, "other CPU")

    quick = attempt({"result": "PASS", "mode": "quick", "full_validation": False,
                     "full_validation_passed": False,
                     "portable_dll_sha256": digest})
    assert not quick["available"] and "full validation" in quick["reason"]
    wrong_candidate = attempt({"result": "PASS", "mode": "full", "full_validation": True,
                               "full_validation_passed": True,
                               "portable_dll_sha256": "0" * 64})
    assert not wrong_candidate["available"] and "candidate DLL SHA-256" in wrong_candidate["reason"]


def test_portable_parallel_default_requires_full_report_thread_evidence(tmp_path, monkeypatch):
    lib = tmp_path / "portable.dll"
    lib.write_bytes(b"portable candidate without thread evidence")
    digest = hashlib.sha256(lib.read_bytes()).hexdigest()
    report = tmp_path / "parity.json"
    report.write_text(json.dumps({"result": "PASS", "mode": "full", "full_validation": True,
                                  "full_validation_passed": True, "portable_dll_sha256": digest,
                                  "openmp_transient_regions": True, "persistent_worker_pool": False,
                                  "thread_counts_tested": [1]}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "kernel_kind": "portable-avx2-fma", "cpu_features": ["AVX2", "FMA3"],
        "library": "portable.dll", "sha256": digest,
        "prefill_library": "portable.dll", "prefill_sha256": digest,
        "exact_validation": {"status": "PASS", "report": "parity.json",
                             "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
    }), encoding="utf-8")
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    monkeypatch.setattr(gp.ctypes, "WinDLL", lambda _path: _FakeAllSymbols(), raising=False)
    features = {name: name in {"AVX2", "FMA3"} for name in gp.REQUIRED_GPTOSS_ISA}
    status = gp._native_library_status("Windows", features, manifest, "other CPU")
    assert status["available"] and status["parallel_supported"] is False


class _FakeAllSymbols:
    def __getattr__(self, _name):
        return object()


def test_manifest_symbol_lists_can_add_but_not_remove_runtime_abi(tmp_path, monkeypatch):
    lib = tmp_path / "kernel.dll"
    lib.write_bytes(b"avx dll")
    digest = hashlib.sha256(lib.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "cpu_features": list(gp.REQUIRED_GPTOSS_ISA), "cpu_signature": "sig",
        "model_id": "gpt-oss-test-revision", "library": "kernel.dll", "sha256": digest,
        "prefill_library": "kernel.dll", "prefill_sha256": digest,
        "symbols": [], "prefill_symbols": [],
    }), encoding="utf-8")
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    loads = []

    class MissingPrefill:
        def __getattr__(self, name):
            if name == "gptoss_experts_multi":
                raise AttributeError(name)
            return object()

    def load(_path):
        loads.append(1)
        return _FakeAllSymbols() if len(loads) == 1 else MissingPrefill()

    monkeypatch.setattr(gp.ctypes, "WinDLL", load, raising=False)
    status = gp._native_library_status("Windows", {n: True for n in gp.REQUIRED_GPTOSS_ISA},
                                        manifest, "sig")
    assert not status["available"]
    assert "gptoss_experts_multi" in status["reason"]


def test_persistent_manifest_cannot_omit_persistent_lifecycle_or_fuse_symbols(tmp_path, monkeypatch):
    lib = tmp_path / "kernel.dll"
    lib.write_bytes(b"persistent dll")
    digest = hashlib.sha256(lib.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "cpu_features": list(gp.REQUIRED_GPTOSS_ISA), "cpu_signature": "sig",
        "model_id": "gpt-oss-test-revision", "library": "kernel.dll", "sha256": digest,
        "prefill_library": "kernel.dll", "prefill_sha256": digest,
        "symbols": [], "prefill_symbols": [],
    }), encoding="utf-8")
    monkeypatch.setattr(gp, "ROOT", tmp_path)

    class MissingPersistent:
        def __getattr__(self, name):
            if name == "gptoss_set_fuse":
                raise AttributeError(name)
            return object()

    monkeypatch.setattr(gp.ctypes, "WinDLL", lambda _path: MissingPersistent(), raising=False)
    status = gp._native_library_status("Windows", {n: True for n in gp.REQUIRED_GPTOSS_ISA},
                                        manifest, "sig")
    assert not status["available"]
    assert "gptoss_set_fuse" in status["reason"]
