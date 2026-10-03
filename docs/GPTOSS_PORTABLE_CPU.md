# GPT-OSS portable CPU kernel

`kernels/gptoss_cpu_portable.c` is a portable CPU implementation of the GPT-OSS
expert ABI. It reads existing Neural expert-slot bytes without modifying them,
supports raw MXFP4/E8M0 and PS4 packed scales, and exports the single-token
decode and grouped-token prefill entry points used by `server.py` and
`cpu_prefill.py`.

The AVX2+FMA build uses two 8-lane halves to preserve the reference's 16-lane
accumulator and reduction order. `--scalar` retains the scalar row-dot path.
Transient OpenMP regions parallelize independent rows for decode and independent
experts for prefill, without claiming a persistent worker pool. Grouped prefill remains a simple per-pair
fallback rather than the weight-reusing AVX-512 prefill implementation.

Build a candidate DLL under `.tools/` with the repository's MinGW-w64 compiler:

```powershell
.venv\Scripts\python.exe tools\build_portable_cpu.py --output .tools\gptoss_cpu_portable_avx2.dll
```

The build uses explicit `-mavx2 -mfma -fopenmp -static` flags and never uses
`-march=native` or AVX-512 flags. The ordinary AVX-512 DLL remains a separate,
machine-bound artifact.

Before enabling a candidate, validate it against both shipped reference DLLs.
The quick check uses one full-shape synthetic expert; the full suite is the
required selection gate:

```powershell
.venv\Scripts\python.exe tools\validate_portable_cpu.py --portable .tools\gptoss_cpu_portable_avx2.dll --quick
.venv\Scripts\python.exe tools\validate_portable_cpu.py --portable .tools\gptoss_cpu_portable_avx2.dll --full
```

The validator creates synthetic full-dimension expert slots and biases; it
does not open a checkpoint or run model inference. Full validation requires
bitwise equality of decoded outputs, captured slot bytes, grouped prefill
results, masked combine output, and auxiliary ABI behavior for raw and PS4
layouts and varied expert counts. It writes a JSON report. A passing report
applies only to the exact DLL hash it records. The validator executes bundled
reference kernels and therefore requires a compatible AVX-512 validation host;
an AVX2-only deployment must use an already gated DLL without running those
reference kernels locally.

The AVX2/OpenMP build passed the full synthetic gate on 2026-10-03: 20 cases,
with every decode and prefill case checked at both one and eight threads.
See [the exact report](../benchmarks/gptoss_agnostic_20261003/portable_avx2_omp_parity.json).
The selected DLL hash is
`cab1f6da1c5143a8ac4ec3a0c7ba0632c4fd183e6e747cabfde625fc6af1b66b`.
Inspection found no ZMM/opmask instructions and only KERNEL32/msvcrt imports;
OpenMP is statically included. A scalar configuration also compiled, but that
new scalar build is not the selected or model-benchmarked binary.

The hashed report enables up to eight physical CPU threads by default. The
portable backend is selected automatically when the inherited native profile
is unavailable, or explicitly with `--native-manifest artifacts/portable_cpu_build.json`.
Native remains preferred on the original host. Portable startup uses bounded
parallel page reads instead of discarded expert computation. See
[full-model validation](GPTOSS_AGNOSTIC_VALIDATION.md) for measured performance
and limitations; the CPU gate itself does not load model weights.
