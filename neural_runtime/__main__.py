"""Model-aware entry point retaining Neural's optimized GPT-OSS engine."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

def parser():
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = ap.add_subparsers(dest="command", required=True)
    for name in ("inspect", "serve"):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument("--model-dir")
        sub.add_argument("--store-dir", help="GPT-OSS original MXFP4 or lossless PS4 store")
        sub.add_argument("--mode", choices=("auto", "fast", "reference"), default="auto")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--context", type=int, default=16384)
        sub.add_argument("--threads", type=int)
        sub.add_argument("--native-manifest", help="verified CPU kernel manifest")
        sub.add_argument("--pool-gib", type=float)
        sub.add_argument("--static-slots", type=int)
        sub.add_argument("--capbufs", type=int)
        sub.add_argument("--gpu-headroom-gib", type=float, default=1.0)
        sub.add_argument("--expert-backend", choices=("auto", "torch-cpu", "staged"), default="auto",
                         help="reference adapter only")
        sub.add_argument("--gpu-expert-budget-gib", type=float)
        sub.add_argument("--host-expert-cache-gib", type=float)
    commands.choices["inspect"].add_argument("--metadata-only", action="store_true")
    serve = commands.choices["serve"]
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument("--dry-run", action="store_true")
    serve.add_argument("engine_args", nargs=argparse.REMAINDER,
                       help="GPT-OSS tuning flags after --; capacity/library/model flags are managed here")
    recommend = commands.add_parser("recommend", allow_abbrev=False,
                                    help="suggest original-format MoE models and theoretical output speeds")
    recommend.add_argument("--device", default="cuda:0")
    recommend.add_argument("--context", type=int, default=16384)
    recommend.add_argument("--workload", choices=("coding", "general"), default="coding")
    recommend.add_argument("--scenario", choices=("dedicated", "current"), default="dedicated")
    recommend.add_argument("--cpu-bandwidth-gbps", type=float, help="explicit bandwidth assumption; skips CPU copy probe")
    recommend.add_argument("--gpu-bandwidth-gbps", type=float, help="explicit bandwidth assumption instead of device peak")
    recommend.add_argument("--json", action="store_true", help="print machine-readable assumptions and source evidence")
    recommend.add_argument("--output", help="also save the report as JSON")
    recommend.add_argument("--serve", action="store_true", help="open the local interactive model advisor")
    recommend.add_argument("--open", action="store_true", help="open the browser with --serve")
    recommend.add_argument("--port", type=int, default=8002)
    start = commands.add_parser("start", allow_abbrev=False,
                                help="start GPT-OSS and open a terminal chat (recommended)")
    start.add_argument("--model-dir")
    start.add_argument("--store-dir")
    start.add_argument("--device", default="cuda:0")
    start.add_argument("--context", type=int, default=16384)
    start.add_argument("--port", type=int, default=8001)
    start.add_argument("--check", action="store_true", help="check setup without loading the model")
    return ap

def model_paths(args):
    path = ROOT / "neural.local.json"
    local = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    if not isinstance(local, dict):
        raise ValueError("neural.local.json must contain an object")
    def configured_path(value, environment, key):
        explicit = value or os.environ.get(environment)
        if explicit:
            return explicit
        configured = local.get(key)
        if configured is None:
            return None
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError(f"neural.local.json {key} must be a nonempty path")
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_absolute() else ROOT / candidate

    model = configured_path(args.model_dir, "NEURAL_MODEL_DIR", "model_dir")
    store = configured_path(args.store_dir, "NEURAL_STORE_DIR", "store_dir")
    if not model:
        from .defaults import default_paths
        defaults = default_paths(ROOT / "data")
        model = defaults["model_dir"]
        store = store or defaults["store_dir"]
        if not (Path(model) / "config.json").is_file():
            raise ValueError("GPT-OSS 120B is the default. Run install_neural.bat once to download and prepare it. "
                             "For an existing installation, set the paths in neural.local.json.")
    return str(Path(model).expanduser().resolve()), str(Path(store).expanduser().resolve()) if store else None

def native_arguments(args, spec, plan):
    """Select the inherited fast profile without changing model arithmetic."""
    extra = list(args.engine_args)
    if extra[:1] == ["--"]:
        extra.pop(0)
    managed = {"--model-dir", "--store-dir", "--pool", "--smax", "--kv-ring", "--scratch",
               "--threads", "--static", "--capbufs", "--kdll", "--cpu-multi-dll",
               "--kernel-persistent", "--host", "--port"}
    for token in extra:
        if token.split("=", 1)[0] in managed:
            raise ValueError(f"{token} is managed by the model/platform plan; use launcher options")
    portable = plan["kernel_kind"].startswith("portable-")
    unsupported_tuning = {"--kernel-fuse", "--kernel-prefetch", "--kernel-pair", "--kernel-affinity",
                          "--cold-prefetch", "--kernel-cold-prefetch"}
    if portable and any(token.split("=", 1)[0] in unsupported_tuning for token in extra):
        raise ValueError("the portable CPU fallback does not implement AVX-512 tuning controls")
    flags = {
        "model-dir": spec["paths"]["model_dir"], "store-dir": spec["paths"]["store_dir"],
        "pool": plan["pool_gib"], "smax": plan["context"], "kv-ring": plan["kv_ring"],
        "scratch": plan["scratch_slots"], "threads": plan["threads"],
        "static": plan["static_slots"], "capbufs": plan["capbufs"],
        "kdll": plan["cpu_library"], "cpu-multi-dll": plan["prefill_library"],
        "kernel-persistent": int(plan["kernel_persistent"]),
        "kernel-fuse": 0 if portable else 1, "prefill-order": "layer", "hotset": str(ROOT / "hotset_code.json"),
        "refresh-every": 8, "refresh-m": 16, "prefill-m": 64,
        "early-every": 4, "early-tokens": 32,
        "warm-method": "pages-parallel" if portable else "kernel",
        "grouped-prefill-gemm": 1, "grouped-prefill-min-tokens": 1024,
        "grouped-prefill-max-tokens": 4096, "trim-prefill-cache": 1, "masked-gemv": 0,
        "host": args.host, "port": args.port,
    }
    result = [str(ROOT / "server.py")]
    for key, value in flags.items():
        result.extend(("--" + key, str(value)))
    return result + ["--skip-quality"] + extra

def native_plan(args, spec):
    from .gptoss_platform import probe_hardware, plan_gptoss
    manifest = args.native_manifest
    if manifest is None and not (ROOT / "artifacts" / "gptoss_native_build.json").is_file():
        manifest = ROOT / "artifacts" / "gptoss_known_build.json"
    hw = probe_hardware(args.device, native_manifest=manifest)
    if not hw["native_kernel"]["available"] and args.native_manifest is None:
        portable = ROOT / "artifacts" / "portable_cpu_build.json"
        if portable.is_file():
            fallback = probe_hardware(args.device, native_manifest=portable)
            if fallback["native_kernel"]["available"]:
                hw = fallback
    plan = plan_gptoss(spec, hw, context=args.context, threads=args.threads,
                       pool_gib=args.pool_gib, static_slots=args.static_slots, capbufs=args.capbufs,
                       headroom_gib=args.gpu_headroom_gib)
    return hw, plan

def reference_plan(args, model_dir):
    from neural_reference.backend import inspect_checkpoint
    from neural_reference.hardware import probe_hardware, plan_runtime
    profile = inspect_checkpoint(model_dir)
    hw = probe_hardware(args.device)
    plan = plan_runtime(profile, hw, context=args.context, threads=args.threads,
                        expert_backend=args.expert_backend,
                        gpu_budget_gib=args.gpu_expert_budget_gib,
                        host_cache_gib=args.host_expert_cache_gib,
                        headroom_gib=args.gpu_headroom_gib)
    return profile, hw, plan

def serve_reference(args, spec, profile, hw, plan):
    if args.engine_args:
        raise ValueError("GPT-OSS engine flags cannot be used with a reference adapter")
    import torch
    from neural_reference.backend import NeuralBackend
    from neural_reference.generation import GenerationEngine
    from neural_reference.server import make_server
    torch.cuda.set_device(args.device)
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(plan["threads"])
    if plan.get("native_library"):
        os.environ["QWEN_BF16_EXPERTS_DLL"] = plan["native_library"]
    backend = NeuralBackend(spec["paths"]["model_dir"], device=args.device,
                            expert_backend=plan["expert_backend"],
                            gpu_expert_budget_gib=plan["gpu_expert_budget_gib"],
                            native_gpu_layers=plan["native_gpu_layers"], native_threads=plan["threads"],
                            host_expert_cache_gib=plan["host_expert_cache_gib"], recurrent_graph=False)
    engine = server = None
    try:
        engine = GenerationEngine(backend, context=args.context,
                                  prefix_cache_max_bytes=plan["prefix_cache_max_bytes"])
        server = make_server(backend, engine, host=args.host, port=args.port,
                             model_id=Path(spec["paths"]["model_dir"]).name + "-neural",
                             runtime_info={"model": spec, "checkpoint": profile, "hardware": hw, "plan": plan})
        print(f"READY: reference adapter at http://{args.host}:{server.server_address[1]}/v1", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        if engine:
            engine.prefix_cache.clear("shutdown")
        backend.close()

def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "start":
        from .launcher import start_neural
        return start_neural(args)
    if args.command == "recommend":
        from .advisor import build_report, format_text, _validate_inputs
        _validate_inputs(args.context, args.workload, args.scenario)
        options = dict(context=args.context, workload=args.workload, scenario=args.scenario,
                       cpu_bandwidth_gbps=args.cpu_bandwidth_gbps, gpu_bandwidth_gbps=args.gpu_bandwidth_gbps)
        if args.serve:
            if args.json or args.output:
                raise ValueError("--json/--output are for a single report; use them without --serve")
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            from functools import partial
            from .advisor_ui import serve_advisor
            serve_advisor(partial(build_report, device=args.device), port=args.port,
                          open_browser=args.open, initial_settings=options)
            return 0
        if args.open:
            raise ValueError("--open requires --serve")
        report = build_report(device=args.device, **options)
        payload = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload + "\n", encoding="utf-8")
        print(payload if args.json else format_text(report))
        return 0
    from .model_spec import inspect_model
    model_dir, store_dir = model_paths(args)
    spec = inspect_model(model_dir, store_dir, mode=args.mode).to_dict()
    if args.command == "inspect" and args.metadata_only:
        print(json.dumps({"model": spec}, indent=2))
        return 0
    native = spec["backend"] == "gptoss-native-mxfp4"
    if native:
        hw, plan = native_plan(args, spec)
    else:
        profile, hw, plan = reference_plan(args, model_dir)
    report = {"model": spec, "hardware": hw, "plan": plan}
    if args.command == "inspect":
        print(json.dumps(report, indent=2))
        return 0 if plan.get("supported", True) else 2
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not plan.get("supported", True):
        raise RuntimeError(plan["reason"] + "; " + hw["native_kernel"].get("reason", ""))
    if native:
        command = native_arguments(args, spec, plan)
        report["engine_argv"] = command
    elif args.engine_args:
        raise ValueError("GPT-OSS engine flags cannot be used with a reference adapter")
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return 0
    if native:
        import torch
        torch.cuda.set_device(args.device)
        os.environ.setdefault("NEURAL_PREFILL_GROUP", "0")
        os.environ.setdefault("NEURAL_PREFILL_GROUP_EXPERTS", "3")
        os.environ.setdefault("NEURAL_PREFILL_GROUP_ROWS", "1024")
        old_argv = sys.argv
        try:
            sys.argv = command
            runpy.run_path(command[0], run_name="__main__")
        finally:
            sys.argv = old_argv
    else:
        serve_reference(args, spec, profile, hw, plan)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as exc:
        print(f"Neural dependencies are incomplete. Run install_neural.bat to check or repair setup. Details: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except (ValueError, RuntimeError, MemoryError, OSError, KeyError) as exc:
        print(f"Neural: {exc}", file=sys.stderr)
        raise SystemExit(2)
