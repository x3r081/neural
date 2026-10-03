# Neural technical overview

Neural's default is the official [OpenAI GPT-OSS 120B checkpoint](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a), pinned to revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. The model has 117 billion parameters, uses original MXFP4 expert weights, and selects four experts per token from 36 layers. Neural preserves those weights, the expert-selection rule, and the checkpoint. It does not prune experts or requantize the model.

This repository adds model and hardware inspection, memory planning, and safe backend selection around Neural's optimized GPT-OSS runtime. For this model, the runtime stores frequently used experts in GPU memory and computes other experts on the CPU from a memory-mapped expert store. The CPU and GPU work concurrently. The local server offers an OpenAI-compatible chat API and a terminal chat client. A deeper, illustrated account of this design is on the [Neural project site](https://x3r081.github.io/neural-site/) and its [technical page](https://x3r081.github.io/neural-site/technical.html).

## Requirements and storage

For the intended GPT-OSS setup, use 64-bit Windows 10 or 11, Python 3.12 64-bit, a driver that supports CUDA 13, a BF16-capable NVIDIA GPU (Ampere or newer), a CPU with AVX2+FMA, and an NVMe SSD. The tested host had an RTX 3080 Ti with 12 GiB of graphics memory and 64 GB of RAM; 64 GB of RAM is recommended, and the installer checks whether your actual hardware is compatible. The site’s published performance was measured on that GPU and RAM configuration with an Intel i7-11700K using AVX-512. The fastest CPU binary is restricted to its verified host signature; another CPU is not automatically supported just because it has AVX-512. Other compatible systems use the validated AVX2+FMA fallback, with lower decode performance and substantially slower short-prompt processing.

The checkpoint is about 61 GiB (65 GB) on disk. The raw expert store is about 57 GiB, and the lossless packed PS4 store is about 55.1 GiB. The installer retains the checkpoint and both stores, so the documented initial preflight budget is about 195 GiB free. An NVMe drive and ample free RAM matter: the CPU reads experts from the memory-mapped store, and RAM pressure can cause disk faults that slow generation. Close memory-heavy applications before serving.

## What “lossless PS4” means

The checkpoint's experts are stored in their original MXFP4 representation. The raw store repacks the expert data; the PS4 packed-scale format losslessly re-encodes the scale rows while preserving expert code bytes. The installer verifies the resulting stores. PS4 reduces store size relative to the raw expert store without changing the mathematical scale values. It is not a new lower-bit quantization of the model.

The native GPT-OSS engine is validated for the 36-layer, 128-expert, top-4 GPT-OSS 120B geometry. The wrapper rejects incompatible GPT-OSS sizes, source quantizations, or store layouts. Mixtral, Qwen3 MoE, and Qwen3.5 MoE have reference adapters based on their original tensor formats and upstream model components; fixture checks for those adapters do not establish full-model performance or quality.

## Published site results

The website explains the separate NeuralServer production runtime, whose core server and benchmark results were developed outside this model-aware replacement. Its October 1 comparison replayed four saved API requests in alternating Neural / llama.cpp / llama.cpp / Neural order. On the RTX 3080 Ti host, its all-answer weighted decode result was 24.6910 tokens/s versus 14.3189 for the tested llama.cpp configuration (1.7244×, calculated from measured rates). The replay included a long coding context, an initial answer, and follow-ups. This is a result for that runtime and workload; it is not a measurement of this replacement. The comparison also did not exhaustively tune llama.cpp's prompt-processing settings. See the site's [plain-language overview](https://x3r081.github.io/neural-site/overview.html) and [technical page](https://x3r081.github.io/neural-site/technical.html) for method, measurements, and limits.

The site also records an October 2 experimental backend study. It measured one completed split-backend screen at 29.3246 aggregate tokens/s, versus two historical combined-backend runs at 23.5950 and 23.6227. The split screen was not replicated or run contemporaneously, so it does not establish a causal improvement. The site's strict coding-task quality gate was **not cleared**: a blinded static rubric review found paired regressions, and the generated code was not executed during that review. This does not establish general model-quality degradation, but it does mean the experiment is not a quality-cleared production improvement. The change was not promoted. The public site publishes a [summary with methods and hashes](https://x3r081.github.io/neural-site/benchmark-summary-2026-10-02.json); its raw research evidence is in a private repository.

## Measurements for this repository

The model-aware replacement has its own [validation report](GPTOSS_AGNOSTIC_VALIDATION.md) and archived [full-model reports](../benchmarks/gptoss_agnostic_20261003/). On the RTX 3080 Ti validation host, a matched comparison of the inherited native path and the new launcher measured 16.89 and 17.15 generated tokens/s. All 1,536 corresponding generated token IDs matched. The small rate difference is treated as run variation; this check shows that the launcher preserved the established path, not a new performance gain.

The portable AVX2+FMA path measured 15.06 tokens/s against 18.24 for the native path in a separate repeated comparison, with all 768 generated token IDs matching. It is a compatibility path. Its 2,597-token cold prompt prefill measured 10.63–10.74 seconds versus 9.51–9.57 seconds native; a short prompt measured 5.19–5.35 seconds versus 1.12 seconds. Decode rate and prompt-processing time are separate measurements.

These checks compare execution paths and generated tokens for fixed requests. They are not broad benchmark scores or a guarantee of answer quality across tasks. Full-model numbers are from one Windows GPU host. Capacity estimates are calculated, and tiny architecture fixtures do not stand in for running other full models.

On the developer's existing installation, `install_neural.bat -CheckOnly` passed without changing the setup. Existing-store checks validate model/store metadata and file sizes; they do not reread and cryptographically rehash every payload. The fresh install path uses the store packer's full verification when it builds PS4. The setup and launch paths also have component and subprocess coverage. Final validation did not repeat a fresh 61 GiB checkpoint download and full raw/PS4 store build; the first installation runs that work locally.

## Supported use and limits

- The normal launch uses GPT-OSS 120B and serves on `127.0.0.1:8001`. The binding is local to the PC; a client must be able to reach that address from the same computer. The included starter provides terminal chat; it does not open a browser chat page.
- API clients use base URL `http://127.0.0.1:8001/v1`, model id `gpt-oss-120b-neural`, and a bearer key. The server does not check the key by default; `local` is the included terminal client's default. If `NEURAL_API_KEY` is set in the environment, the client must send that same value.
- Full-model optimized serving is validated on Windows with CUDA. The original AVX-512 binaries are limited to the verified CPU signature. The hash-verified AVX2+FMA fallback broadens CPU compatibility with slower execution.
- The reference adapters support experimental use of listed architecture families. Other model architectures, arbitrary community checkpoints, multi-GPU execution, and non-Windows full-model performance have not been validated.
- The installed project site concerns the separate NeuralServer runtime. Its editor integration examples and old manual install commands do not configure this repository's installer or launcher.

For exact setup steps, see the [Windows quick start](QUICK_START.md). For CPU parity details, see [portable CPU validation](GPTOSS_PORTABLE_CPU.md).
