# Neural

Neural runs OpenAI's original GPT-OSS 120B model locally on a Windows PC with an NVIDIA GPU. Its default model is the official, revision-pinned GPT-OSS 120B checkpoint in its original MXFP4 format. Neural keeps the model weights and routing intact and builds a lossless PS4 expert store for serving.

**Requirements:** 64-bit Windows 10 or 11, Python 3.12 64-bit, an NVIDIA GPU with BF16 support (Ampere or newer) and a driver that supports CUDA 13, and a CPU with AVX2 and FMA. **64 GB of RAM is recommended.** The measured host used an RTX 3080 Ti with 12 GiB of VRAM and 64 GB of RAM; compatibility is checked on your PC during installation. AVX-512 is used only with the verified native CPU build; other compatible CPUs use the slower AVX2 path. Allow **about 195 GiB free** on the data drive: the checkpoint is about **61 GiB (65 GB)**, and the raw and PS4 stores are kept alongside it.

## Get started

1. Download the [Neural ZIP](https://github.com/x3r081/neural/archive/refs/heads/main.zip), then extract it and open the `neural-main` folder. Git users can clone the repository instead.
2. Install [Python 3.12 for Windows (64-bit)](https://www.python.org/downloads/release/python-31210/) if it is not already installed.
3. Double-click `install_neural.bat` and let it finish. It creates Neural's private `.venv`, downloads the pinned GPT-OSS checkpoint, and builds and verifies its raw MXFP4 and lossless PS4 expert stores. This is a large one-time download and can take a while.
4. Double-click `start_neural.bat`. It starts the local API at `http://127.0.0.1:8001/v1` and opens terminal chat in the same window. Type `/quit` to leave chat; the launcher then stops the server it started.

The installer needs about **195 GiB free** on the data drive. The model, raw store and final PS4 store are kept; Neural does not remove the raw copies to reclaim space. The default data directory is `data`. To install them somewhere else, run `install_neural.bat -DataDir D:\NeuralData` from Command Prompt. If `neural.local.json` already contains GPT-OSS paths whose metadata and sizes validate, the installer reuses them; it does not replace a configuration for another model automatically. Existing-store checks do not reread and rehash every payload.

For requirements, first-run behavior, and troubleshooting, see the [beginner quick start](docs/QUICK_START.md). For how the engine works, what was measured, and which results belong to the separate production server, see the [technical overview](docs/TECHNICAL_OVERVIEW.md).

## What Neural does

GPT-OSS 120B is larger than the graphics memory on a typical consumer GPU. Neural stores frequently used experts in GPU memory and computes other experts on the CPU from a memory-mapped store in system RAM. Its CUDA kernels, CPU kernels, and cache preserve the checkpoint's architecture and expert-selection rule. The local server exposes an OpenAI-compatible chat API; see the technical overview for its connection settings.

The optimized GPT-OSS path is the default and the only path with full-model performance validation. Mixtral, Qwen3 MoE and Qwen3.5 MoE have reference adapters for functional experiments. Their small architecture fixtures do not establish full-model speed or quality. Other GPT-OSS sizes, arbitrary quantizations, multi-GPU serving, and non-Windows native serving are not claimed as supported.

## Site and evidence

The [Neural project site](https://x3r081.github.io/neural-site/) explains the original single-PC design and its production measurements. That site covers the separate NeuralServer runtime. Its performance figures are not benchmarks of this model-aware replacement. The site also records that the October 2 batch-backend experiment did not pass its task-quality gate and was not promoted to production; see [how to read the site results](docs/TECHNICAL_OVERVIEW.md#published-site-results).

This repository's replacement measurements and limitations are in [validation](docs/GPTOSS_AGNOSTIC_VALIDATION.md), with raw reports in `benchmarks/gptoss_agnostic_20261003/`. On the tested RTX 3080 Ti, the direct native path measured 16.89 generated tokens/s and the new launcher measured 17.15; all 1,536 corresponding generated token IDs matched. The small difference is run variation. The validated AVX2 CPU fallback matched all 768 tokens in its comparison and measured 15.06 versus 18.24 tokens/s for the native CPU path; short-prompt prefill was slower. These are parity and compatibility checks, not a new comparison with llama.cpp.

Run `start_neural_advisor.bat` for an optional browser report about model fit and hardware capacity. It uses catalog evidence and bandwidth estimates; it does not benchmark model quality, and it does not automatically switch Neural away from GPT-OSS. Details are in [model advisor](docs/MODEL_ADVISOR.md). [Portable CPU validation](docs/GPTOSS_PORTABLE_CPU.md) and [research notes](docs/GPTOSS_AGNOSTIC_RESEARCH.md) describe other advanced details.
