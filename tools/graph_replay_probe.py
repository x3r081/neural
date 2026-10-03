"""Measure Python, C-extension, and direct CUDA graph-launch overhead.

This is a GPU microbenchmark, not a model or serving benchmark. It captures
stable-address deterministic graphs once, retains all graph owners/operands,
and then compares ``CUDAGraph.replay()``, the base C-extension method, and
``cudaGraphLaunch`` through ctypes. Includes one graph, 36-graph chains at a
small and hidden-state-sized width, and a three-graph chain whose middle graph
waits on an external CUDA event. Launch order and host-gap conditions are
interleaved. It uses no random tensor values or model weights.

Do not run while another GPU job is active. A measured microsecond saving only
matters if a matched, same-residency serving trial later confirms it.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback


METHODS = ("python_replay", "base_c_replay", "ctypes_c_replay", "ctypes_python_replay")


def atomic_json(path: str, value: object) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_gaps(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals or any(x < 0 for x in vals):
        raise argparse.ArgumentTypeError("host gaps must be a comma-separated list of nonnegative microseconds")
    return vals


def spin_gap_us(gap_us: float) -> float:
    """Apply a deliberate host scheduling gap without timer-resolution sleeps."""
    start = time.perf_counter_ns()
    deadline = start + int(gap_us * 1000)
    while time.perf_counter_ns() < deadline:
        pass
    return (time.perf_counter_ns() - start) / 1000.0


def check_cuda_status(status: int, get_error_string, label: str) -> None:
    if status:
        try:
            message = get_error_string(status)
            detail = message.decode("utf-8", errors="replace") if message else "unknown CUDA error"
        except Exception:
            detail = "could not read cudaGetErrorString"
        raise RuntimeError(f"{label} returned CUDA error {status}: {detail}")


def load_cudart(torch_module):
    """Load this PyTorch install's CUDA Runtime and expose GIL/free vs held calls."""
    libdir = Path(torch_module.__file__).resolve().parent / "lib"
    candidates = sorted(libdir.glob("cudart64_*.dll"))
    if not candidates:
        raise FileNotFoundError(f"no cudart64_*.dll under {libdir}")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("the PyDLL/WinDLL comparison requires 64-bit Windows")
    path = candidates[-1]
    dll_directory = os.add_dll_directory(str(libdir))
    libs = {}
    for name, loader in (("ctypes_c_replay", ctypes.CDLL),
                         ("ctypes_python_replay", ctypes.PyDLL)):
        lib = loader(str(path), use_last_error=True)
        launch = lib.cudaGraphLaunch
        launch.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        launch.restype = ctypes.c_int
        get_error = lib.cudaGetErrorString
        get_error.argtypes = (ctypes.c_int,)
        get_error.restype = ctypes.c_char_p
        libs[name] = {"library": lib, "launch": launch, "get_error": get_error}
    return path, libs, dll_directory


def make_launchers(torch_module, libs, stream):
    base_type = torch_module._C._CUDAGraph
    stream_handle = int(stream.cuda_stream)
    if stream_handle == 0:
        raise RuntimeError("expected a non-default explicit stream for replay probe")

    def public(g):
        g.replay()

    def base_c(g):
        base_type.replay(g)

    def direct_factory(libname):
        api = libs[libname]

        def direct(g):
            handle = g._probe_exec_handle
            if not handle:
                raise RuntimeError("missing retained cudaGraphExec_t")
            status = api["launch"](ctypes.c_void_p(handle), ctypes.c_void_p(stream_handle))
            check_cuda_status(status, api["get_error"], "cudaGraphLaunch")
        return direct

    return {"python_replay": public, "base_c_replay": base_c,
            "ctypes_c_replay": direct_factory("ctypes_c_replay"),
            "ctypes_python_replay": direct_factory("ctypes_python_replay")}


def capture_graph(torch, stream, fn, pool=None):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=pool):
        fn()
    handle = int(graph.raw_cuda_graph_exec())
    if not handle:
        raise RuntimeError("CUDAGraph returned a null cudaGraphExec_t")
    # The function object is attached to the owner, and the handle is valid
    # only while this graph remains alive and is not re-instantiated/reset.
    graph._probe_exec_handle = handle
    return graph


def build_scenario(torch, stream, name: str, width: int, chain_length: int, pool):
    owners = []
    retained = []
    one = torch.ones((width,), device="cuda", dtype=torch.float32)
    buffers = [torch.empty_like(one) for _ in range(chain_length)]
    seed = torch.zeros_like(one)
    retained.extend((one, buffers, seed))
    for i in range(chain_length):
        src = seed if i == 0 else buffers[i - 1]
        dst = buffers[i]
        fn = lambda src=src, dst=dst, one=one: torch.add(src, one, out=dst)
        owners.append(capture_graph(torch, stream, fn, pool=pool))
    def reset_outputs():
        for buf in buffers:
            buf.fill_(-12345.25)
    return {"name": name, "owners": owners, "retained": retained,
            "output": buffers[-1], "expected": float(chain_length),
            "launches_per_iteration": chain_length, "external_event": None,
            "reset_outputs": reset_outputs}


def build_single(torch, stream, pool):
    x = torch.arange(256, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    g = capture_graph(torch, stream, lambda: torch.add(x, 1.0, out=y), pool=pool)
    def reset_outputs():
        y.fill_(-12345.25)
    return {"name": "ordinary_single", "owners": [g], "retained": [x, y],
            "output": y, "expected_tensor": x + 1.0, "launches_per_iteration": 1,
            "external_event": None, "reset_outputs": reset_outputs}


def build_event_chain(torch, stream, pool):
    """Record an external event in graph 0 and consume it in graph 1."""
    src = torch.full((2880,), 2.0, dtype=torch.float32, device="cuda")
    mid = torch.empty_like(src)
    out = torch.empty_like(src)
    event = torch.cuda.Event(external=True)

    def record_source():
        src.fill_(2.0)
        event.record(stream)

    def wait_and_transform():
        event.wait(stream)
        torch.add(src, 3.0, out=mid)

    def copy_tail():
        torch.mul(mid, 2.0, out=out)

    def reset_outputs():
        src.fill_(-12345.25)
        mid.fill_(-23456.5)
        out.fill_(-34567.75)

    owners = [capture_graph(torch, stream, record_source, pool=pool),
              capture_graph(torch, stream, wait_and_transform, pool=pool),
              capture_graph(torch, stream, copy_tail, pool=pool)]
    return {"name": "external_event_middle", "owners": owners,
            "retained": [src, mid, out, event], "output": out, "expected": 10.0,
            "launches_per_iteration": len(owners), "external_event": event,
            "reset_outputs": reset_outputs}


def call_scenario(scenario, launch, gap_us):
    actual_gap = 0.0
    owners = scenario["owners"]
    for index, g in enumerate(owners):
        launch(g)
        if gap_us and index + 1 < len(owners):
            actual_gap += spin_gap_us(gap_us)
    return actual_gap


def verify_scenario(torch, stream, scenario, launchers):
    expected_tensor = scenario.get("expected_tensor")
    expected_scalar = scenario.get("expected")
    checks = {}
    for name in METHODS:
        launch = launchers[name]
        with torch.cuda.stream(stream):
            scenario["reset_outputs"]()
            for _ in range(3):
                call_scenario(scenario, launch, 0)
            stream.synchronize()
        got = scenario["output"].detach().contiguous()
        if expected_tensor is not None:
            expected = expected_tensor.detach().contiguous()
            equal = torch.equal(got.view(torch.int32), expected.view(torch.int32))
        else:
            expected = torch.full_like(got, expected_scalar)
            equal = torch.equal(got.view(torch.int32), expected.view(torch.int32))
        checks[name] = {"bit_equal": bool(equal), "numel": got.numel(),
                        "output_sha256": hashlib.sha256(got.cpu().numpy().tobytes()).hexdigest()}
        if not equal:
            raise AssertionError(f"{scenario['name']} output mismatch for {name}")
    # A deliberate skipped launch must be observable after the per-method poison
    # reset. This prevents stale output from a preceding method masking a no-op.
    with torch.cuda.stream(stream):
        scenario["reset_outputs"]()
        stream.synchronize()
        skip_index = len(scenario["owners"]) // 2
        skipped_graph = scenario["owners"][skip_index]
        launch = launchers["base_c_replay"]
        for graph in scenario["owners"]:
            if graph is not skipped_graph:
                launch(graph)
        stream.synchronize()
    got = scenario["output"].detach().contiguous()
    if expected_tensor is not None:
        expected = expected_tensor.detach().contiguous()
    else:
        expected = torch.full_like(got, expected_scalar)
    negative_detected = not torch.equal(got.view(torch.int32), expected.view(torch.int32))
    checks["skip_launch_negative_control"] = {"skipped_graph_index": skip_index,
                                               "failure_detected": bool(negative_detected)}
    if not negative_detected:
        raise AssertionError(f"{scenario['name']} skip-launch negative control unexpectedly passed")
    return checks


def measure_leg(torch, stream, scenario, launch, iterations, warmup, gap_us):
    with torch.cuda.stream(stream):
        scenario["reset_outputs"]()
        stream.synchronize()
        for _ in range(warmup):
            call_scenario(scenario, launch, 0)
        stream.synchronize()
        scenario["reset_outputs"]()
        stream.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        t0 = time.perf_counter_ns()
        gap_actual = 0.0
        owners = scenario["owners"]
        for iteration in range(iterations):
            for graph_index, graph in enumerate(owners):
                launch(graph)
                # Apply gaps only between launches, including chain boundaries;
                # do not charge an idle interval after the final launch.
                is_final = iteration + 1 == iterations and graph_index + 1 == len(owners)
                if gap_us and not is_final:
                    gap_actual += spin_gap_us(gap_us)
        host_ns = time.perf_counter_ns() - t0
        end_event.record(stream)
        end_event.synchronize()
        gpu_ms = start_event.elapsed_time(end_event)
        stream.synchronize()
    launches = iterations * scenario["launches_per_iteration"]
    requested_gap_us = gap_us * max(0, launches - 1)
    return {"iterations": iterations, "launches": launches,
            "requested_gap_us_per_launch": gap_us,
            "actual_gap_us_total": round(gap_actual, 2),
            "host_submit_ms_including_gaps": round(host_ns / 1e6, 4),
            "host_submit_us_per_launch_including_gaps": round(host_ns / launches / 1e3, 4),
            "host_submit_us_per_launch_minus_requested_gap": round(
                (host_ns / 1000 - requested_gap_us) / launches, 4),
            "host_submit_us_per_launch_minus_actual_gap": round(
                (host_ns / 1000 - gap_actual) / launches, 4),
            "gpu_event_ms_per_iteration": round(gpu_ms / iterations, 4),
            "gpu_event_us_per_launch": round(gpu_ms * 1000 / launches, 4)}


def measure_scenario(torch, stream, scenario, launchers, args, rng, results, save):
    record = {"scenario": scenario["name"], "launches_per_iteration": scenario["launches_per_iteration"],
              "correctness": {}, "legs": [], "medians": {}}
    # Publish the active scenario before any check/leg so errors and progress
    # saves include the current work rather than only already-finished scenarios.
    results.append(record)
    save()
    correctness = verify_scenario(torch, stream, scenario, launchers)
    record["correctness"] = correctness
    save()
    legs = record["legs"]
    conditions = [(method, gap) for gap in args.host_gaps_us for method in METHODS]
    for rep in range(args.repeats):
        order = list(conditions)
        rng.shuffle(order)
        for method, gap in order:
            leg = measure_leg(torch, stream, scenario, launchers[method],
                              args.iterations, args.warmup, gap)
            leg.update({"repeat": rep, "method": method, "scenario": scenario["name"]})
            legs.append(leg)
            save()
    medians = {}
    for method in METHODS:
        for gap in args.host_gaps_us:
            samples = [x for x in legs if x["method"] == method and
                       x["requested_gap_us_per_launch"] == gap]
            medians[f"{method}|gap_us={gap:g}"] = {
                key: round(statistics.median(x[key] for x in samples), 4)
                for key in ("host_submit_us_per_launch_including_gaps",
                            "host_submit_us_per_launch_minus_requested_gap",
                            "host_submit_us_per_launch_minus_actual_gap",
                            "gpu_event_us_per_launch")}
    record["medians"] = medians
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="JSON output path; atomically updated after every leg")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chain-length", type=int, default=36)
    ap.add_argument("--small-width", type=int, default=128)
    ap.add_argument("--model-width", type=int, default=2880)
    ap.add_argument("--iterations", type=int, default=200, help="timed iterations per leg")
    ap.add_argument("--warmup", type=int, default=10, help="untimed scenario replays before each leg")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--host-gaps-us", type=parse_gaps, default=parse_gaps("0,10,50"),
                    help="busy-wait between graph launches; gap work is reported separately")
    ap.add_argument("--seed", type=int, default=20261001,
                    help="Python treatment-order shuffle seed; device graph math uses no RNG")
    args = ap.parse_args()
    if min(args.chain_length, args.small_width, args.model_width, args.iterations, args.repeats) <= 0:
        ap.error("chain, widths, iterations and repeats must be positive")
    if args.warmup < 0:
        ap.error("warmup must be nonnegative")
    if os.path.exists(args.output):
        ap.error(f"refusing to overwrite existing output artifact: {args.output}")

    import torch
    import torch.cuda.graphs as torch_graphs
    result = {"schema": "neuralserver-graph-replay-probe-v1", "label": "UNMEASURED",
              "scope": "graph-launch microbenchmark only; no model inference or serving-speed claim",
              "config": vars(args), "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
              "results": [], "status": "running"}

    def save():
        atomic_json(args.output, result)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("this probe requires a CUDA device")
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        cudart_path, libs, dll_directory = load_cudart(torch)
        lib_names = {name: str(Path(lib["library"]._name).resolve())
                     for name, lib in libs.items()}
        launchers = make_launchers(torch, libs, stream)
        result["runtime"] = {"device": str(device), "device_name": torch.cuda.get_device_name(device),
                             "cudart": str(cudart_path), "ctypes_libraries": lib_names,
                             "cudart_dll_directory_handle_retained": dll_directory is not None,
                             "graph_replay_methods": list(METHODS),
                             "current_stream": int(stream.cuda_stream),
                             "torch_graphs_py": str(Path(torch_graphs.__file__).resolve()),
                             "torch_graphs_py_sha256": sha256_file(torch_graphs.__file__)}
        with torch.cuda.stream(stream):
            pool = torch.cuda.graph_pool_handle()
            scenarios = [build_single(torch, stream, pool),
                         build_scenario(torch, stream, "chain_small", args.small_width, args.chain_length, pool),
                         build_scenario(torch, stream, "chain_model_width", args.model_width, args.chain_length, pool),
                         build_event_chain(torch, stream, pool)]
        stream.synchronize()
        # Graphs, tensors, event and runtime DLLs remain reachable until all
        # checks and measurements finish; no graph is re-instantiated/reset.
        result["owners_retained"] = True
        rng = random.Random(args.seed)
        for scenario in scenarios:
            measure_scenario(torch, stream, scenario, launchers, args, rng, result["results"], save)
            save()
        result["label"] = "MEASURED"
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": repr(exc),
                           "traceback": traceback.format_exc()}
        save()
        raise
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
