"""CPU-only no-op rendezvous probe; no weights or inference.

Native QPC times a region with one phase barrier, sfence and per-thread marks.
The gap is unmeasured busy caller work. Results do not predict inference gain.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paths as NP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=160)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()
    if args.calls < 20 or args.rounds < 2:
        ap.error("calls>=20 and rounds>=2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dll = args.output.parent / "gptoss_cpu_persistent_probe.dll"
    source = ROOT / "kernels" / "gptoss_cpu_persistent.c"
    compiler = Path(NP.DEVKIT) / "gcc.exe"
    command = [str(compiler), "-O3", "-march=native", "-fopenmp", "-shared", "-o", str(dll), str(source)]
    build_env = dict(os.environ, PATH=NP.DEVKIT + os.pathsep + os.environ.get("PATH", ""))
    build = subprocess.run(command, capture_output=True, text=True, env=build_env)
    if build.returncode:
        raise RuntimeError(f"native build failed:\n{build.stdout}\n{build.stderr}")
    # Match server.py's setting, whose effect is reported as inert in LEVERS.
    os.environ.setdefault("GOMP_SPINCOUNT", "20000000")
    dll_dir = os.add_dll_directory(NP.DEVKIT)
    lib = ctypes.CDLL(str(dll.resolve()))
    VP = ctypes.c_void_p
    lib.gptoss_team_probe_batch.argtypes = [ctypes.c_int]*4 + [VP, VP]
    lib.gptoss_team_probe_batch.restype = ctypes.c_int
    lib.gptoss_set_persistent.argtypes = [ctypes.c_int]
    lib.gptoss_set_persistent.restype = ctypes.c_int
    lib.gptoss_persistent_shutdown.restype = ctypes.c_int
    lib.gptoss_persistent_status.argtypes = [VP]
    result = {"label": "MEASURED CPU_ONLY NO_OP_NOT_INFERENCE", "pid": os.getpid(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "dll_sha256": hashlib.sha256(dll.read_bytes()).hexdigest(),
              "compiler": subprocess.run([str(compiler), "--version"], capture_output=True, text=True).stdout.splitlines()[0],
              "build_command": command, "build_stdout": build.stdout, "build_stderr": build.stderr,
              "GOMP_SPINCOUNT": os.environ.get("GOMP_SPINCOUNT"),
              "wait_policy": "pause<150000TSC, yield<3000000TSC, WaitOnAddress thereafter",
              "calls_per_leg": args.calls, "rounds": args.rounds, "cases": []}
    def batch(mode, threads, calls, gap):
        times = (ctypes.c_double*calls)()
        checksum = ctypes.c_ulonglong()
        status = lib.gptoss_team_probe_batch(mode, threads, calls, gap, times, ctypes.byref(checksum))
        if status:
            raise RuntimeError(f"probe status{status}, mode{mode}, threads{threads}")
        return list(times), checksum.value
    try:
        assert lib.gptoss_set_persistent(1) == 0
        for threads in (1, 2, 4, 8):
            for gap in (0, 200, 1000):
                batch(0, threads, 24, gap)
                batch(1, threads, 24, gap)
                samples = {0: [], 1: []}
                legs = []
                for repeat in range(args.rounds):
                    for mode in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        times, checksum = batch(mode, threads, args.calls, gap)
                        # Keep every call; warm-up was separate.
                        samples[mode].extend(times)
                        legs.append({"round": repeat, "mode": mode, "median_us": statistics.median(times),
                                     "checksum": checksum})
                med = {"omp_us": statistics.median(samples[0]), "persistent_us": statistics.median(samples[1])}
                case = {"threads": threads, "workers": threads-1, "gap_us": gap, "medians": med,
                        "saving_us": med["omp_us"]-med["persistent_us"], "legs": legs,
                        "samples_us": {str(mode): samples[mode] for mode in (0, 1)}}
                result["cases"].append(case)
                print(f"threads{threads} gap{gap}us: omp{med['omp_us']:.2f}us persistent{med['persistent_us']:.2f}us saving{case['saving_us']:.2f}us", flush=True)
        # Explicit lifecycle bound and repeated restart/stop checks.
        for threads in (8, 1, 4, 2, 8):
            assert lib.gptoss_set_persistent(0) == 0
            assert lib.gptoss_set_persistent(1) == 0
            batch(1, threads, 8, 0)
        assert lib.gptoss_persistent_shutdown() == 0
        status = (ctypes.c_int*4)()
        lib.gptoss_persistent_status(status)
        result["shutdown_status"] = list(status)
        result["lifecycle_pass"] = status[0] == status[1] == status[2] == 0
        assert result["lifecycle_pass"]
    finally:
        lib.gptoss_persistent_shutdown()
        result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        dll_dir.close()
    print("PASS native no-op marks and lifecycle; no inference was run", flush=True)

if __name__ == "__main__":
    main()
