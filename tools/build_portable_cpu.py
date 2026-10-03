#!/usr/bin/env python3
"""Build the GPT-OSS portable CPU kernel without targeting AVX-512 or host ISA."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kernels" / "gptoss_cpu_portable.c"


def compiler_path(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    found = shutil.which("gcc")
    if found:
        return Path(found).resolve()
    devkit = os.environ.get("NEURAL_DEVKIT") or r"F:\AI\Neural\third_party\tools\w64devkit\bin"
    candidate = Path(devkit) / "gcc.exe"
    if candidate.is_file():
        return candidate.resolve()
    raise SystemExit("gcc not found; pass --compiler or set NEURAL_DEVKIT to a MinGW-w64 bin directory")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiler", help="MinGW-w64 GCC executable")
    ap.add_argument("--output", default=str(ROOT / ".tools" / "gptoss_cpu_portable_avx2.dll"))
    ap.add_argument("--scalar", action="store_true", help="compile the scalar row-dot fallback")
    args = ap.parse_args()

    cc = compiler_path(args.compiler)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Statically include libgomp/libgcc so the DLL needs only system Windows
    # imports. This is a transient OpenMP region per call, never a persistent
    # worker pool.
    flags = ["-O3", "-shared", "-ffp-contract=off", "-fopenmp", "-static"]
    if not args.scalar:
        flags += ["-mavx2", "-mfma"]
    cmd = [str(cc), *flags, "-o", str(out), str(SOURCE)]
    print("Building portable kernel:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PATH"] = str(cc.parent) + os.pathsep + env.get("PATH", "")
    subprocess.run(cmd, check=True, env=env)
    print(f"Built {out}; OpenMP is transient and AVX-512 was not targeted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
