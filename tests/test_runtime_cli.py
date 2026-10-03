from types import SimpleNamespace

from neural_runtime import __main__ as cli


def test_native_arguments_uses_plan_selected_both_libraries_and_manages_runtime_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    args = SimpleNamespace(engine_args=[], host="127.0.0.1", port=8012)
    spec = {"paths": {"model_dir": "M", "store_dir": "S"}}
    plan = {"pool_gib": 5.25, "context": 8192, "kv_ring": 256,
            "scratch_slots": 6, "threads": 7, "static_slots": 200, "capbufs": 12,
            "cpu_library": "C:/kernel.dll", "prefill_library": "C:/prefill.dll",
            "kernel_persistent": False, "kernel_kind": "portable-avx2-fma"}

    argv = cli.native_arguments(args, spec, plan)
    pairs = dict(zip(argv[1:-1:2], argv[2::2]))
    assert pairs["--kdll"] == "C:/kernel.dll"
    assert pairs["--cpu-multi-dll"] == "C:/prefill.dll"
    assert pairs["--kernel-persistent"] == "0"
    assert pairs["--smax"] == "8192"
    assert pairs["--threads"] == "7"
    assert pairs["--kernel-fuse"] == "0"
    assert pairs["--warm-method"] == "pages-parallel"
    assert argv[0] == str(tmp_path / "server.py")


def test_native_arguments_rejects_user_overrides_of_planned_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    args = SimpleNamespace(engine_args=["--threads", "4"], host="127.0.0.1", port=8001)
    spec = {"paths": {"model_dir": "M", "store_dir": "S"}}
    plan = {"pool_gib": 5, "context": 4096, "kv_ring": 256, "scratch_slots": 6,
            "threads": 8, "static_slots": 200, "capbufs": 12,
            "cpu_library": "K", "prefill_library": "P", "kernel_persistent": True}
    try:
        cli.native_arguments(args, spec, plan)
    except ValueError as exc:
        assert "managed by the model/platform plan" in str(exc)
    else:
        raise AssertionError("the command must not override the metadata-derived thread plan")


def test_portable_native_arguments_rejects_actual_cold_prefetch_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    args = SimpleNamespace(engine_args=["--kernel-cold-prefetch", "1"],
                           host="127.0.0.1", port=8001)
    spec = {"paths": {"model_dir": "M", "store_dir": "S"}}
    plan = {"pool_gib": 5, "context": 4096, "kv_ring": 256, "scratch_slots": 6,
            "threads": 8, "static_slots": 200, "capbufs": 12,
            "cpu_library": "K", "prefill_library": "P", "kernel_persistent": False,
            "kernel_kind": "portable-avx2-fma"}
    try:
        cli.native_arguments(args, spec, plan)
    except ValueError as exc:
        assert "portable CPU fallback" in str(exc)
    else:
        raise AssertionError("portable profile must reject the server's actual cold-prefetch option")


def test_persistent_native_arguments_keep_kernel_warmup_default(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    args = SimpleNamespace(engine_args=[], host="127.0.0.1", port=8001)
    spec = {"paths": {"model_dir": "M", "store_dir": "S"}}
    plan = {"pool_gib": 5, "context": 4096, "kv_ring": 256, "scratch_slots": 6,
            "threads": 8, "static_slots": 200, "capbufs": 12,
            "cpu_library": "K", "prefill_library": "P", "kernel_persistent": True,
            "kernel_kind": "avx512-persistent"}
    argv = cli.native_arguments(args, spec, plan)
    pairs = dict(zip(argv[1:-1:2], argv[2::2]))
    assert pairs["--kernel-fuse"] == "1"
    assert pairs["--kernel-persistent"] == "1"
    assert pairs["--warm-method"] == "kernel"


def test_native_plan_uses_validated_portable_manifest_only_when_default_is_unavailable(tmp_path, monkeypatch):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "portable_cpu_build.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    calls = []

    def probe(_device, *, native_manifest=None):
        calls.append(native_manifest)
        available = len(calls) == 2
        return {"native_kernel": {"available": available, "selected": str(native_manifest)}}

    monkeypatch.setattr("neural_runtime.gptoss_platform.probe_hardware", probe)
    monkeypatch.setattr("neural_runtime.gptoss_platform.plan_gptoss",
                        lambda _spec, hw, **_kw: {"supported": True, "selected": hw["native_kernel"]["selected"]})
    args = SimpleNamespace(device="cuda:0", native_manifest=None, context=4096, threads=None,
                           pool_gib=None, static_slots=None, capbufs=None, gpu_headroom_gib=1.5)
    hw, plan = cli.native_plan(args, {"model_type": "gpt_oss"})
    assert len(calls) == 2
    assert calls[0] == tmp_path / "artifacts" / "gptoss_known_build.json"
    assert calls[1] == tmp_path / "artifacts" / "portable_cpu_build.json"
    assert plan["supported"] is True
    assert plan["selected"] == hw["native_kernel"]["selected"]

    calls.clear()
    args.native_manifest = "explicit.json"
    cli.native_plan(args, {"model_type": "gpt_oss"})
    assert calls == ["explicit.json"]
