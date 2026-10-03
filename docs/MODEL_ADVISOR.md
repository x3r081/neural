# Local model advisor

The advisor scans the selected CUDA GPU, system RAM, CPU capabilities and verified Neural kernel availability. It suggests models from a small, bundled catalog and shows theoretical output tokens/second beside the support and memory evidence. It never downloads checkpoints or runs model inference.

**GPT-OSS 120B is the default.** New users can run `install_neural.bat` once and `start_neural.bat` to open chat without choosing a model. The advisor always shows the default model's status first, even if a requirement is missing. It does not replace an unavailable GPT-OSS model with an experimental alternative; those options are under the advanced disclosures. The JSON report includes `default_model` separately from its runnable `recommendations` list.

## Start

```bat
start_neural_advisor.bat
```

This opens `http://127.0.0.1:8002/`. The advisor is independent of the inference server and needs no configured checkpoint. Use **Dedicated machine** to plan for running a model after other inference servers have stopped, or **Current machine** to use currently available memory. A dedicated recommendation is not a claim that the model can start while another model is loaded.

The command line supports the same report:

```bat
.venv\Scripts\python.exe -m neural_runtime recommend
.venv\Scripts\python.exe -m neural_runtime recommend --context 32768 --workload general --scenario current
.venv\Scripts\python.exe -m neural_runtime recommend --json
.venv\Scripts\python.exe -m neural_runtime recommend --output report.json
.venv\Scripts\python.exe -m neural_runtime recommend --serve --open --port 8002
```

Optional `--cpu-bandwidth-gbps` and `--gpu-bandwidth-gbps` supply explicit assumptions. These values are labeled **ASSUMED_USER_OVERRIDE**, not measurements. CPU overrides also skip the short memory-copy probe. The browser accepts the same overrides and starts with the command-line settings.

## What a suggestion means

The initial catalog contains original-format GPT-OSS 120B, Qwen3-30B-A3B-Instruct-2507, Qwen3.6-35B-A3B and Mixtral-8x7B-Instruct-v0.1. Every entry pins an official checkpoint revision and links its configuration, weight metadata and model card. Source facts and byte derivations live in `neural_runtime/model_catalog.json`.

GPT-OSS is ranked first when its optimized path and memory requirements are compatible. The other entries are **reference experiments**: architecture fixtures were validated, but that is not full-checkpoint performance or quality validation. They are not advertised as faster replacements. The catalog is curated and offline, not an exhaustive or automatically refreshed model leaderboard. Coding/general tags are suitability hints, not intelligence scores.

Models are excluded when CUDA/BF16 support, the required native kernel, context limit or memory budget fails. The optimized GPT-OSS path requires Windows. Integrated GPUs with shared RAM are excluded because this estimator assumes separate memory pools. Unsupported architectures and unlisted variants receive no invented estimate.

## Memory calculation

- Dedicated mode uses total capacity, reserving 4 GiB RAM for the OS, 2 GiB RAM for runtime use, and 1 GiB VRAM for workspace. Current mode uses available memory with the same runtime/workspace reserve and no additional OS reserve.
- Core weights, full/sliding attention state, recurrent state and transient expert buffers are accounted for. Reference prefix caching reserves two states. Reference GPU residency is rounded down to complete expert layers, matching default CPU-expert execution.
- GPT-OSS uses its existing lossless PS4 slot size, with the proven default 6.05 GiB pool ceiling, whole slots and six scratch slots. CPU token embeddings are charged to RAM.
- Nonresident experts must fit RAM, including a bounded reference host tensor cache. GPU-resident file-backed weight pages are assumed evictable from the OS cache. Disk paging is not assigned a fictitious throughput.
- Catalog residual core bytes can include unused vision/auxiliary weights; this is a conservative capacity estimate. Available-memory snapshots, fragmentation and other activity still affect actual loading. The serving preflight remains authoritative.

## Understanding theoretical tokens/second

These numbers are **CALCULATED_THEORETICAL bandwidth scenarios**, with low confidence. They are neither expected throughput nor guaranteed bounds. Actual output speed can fall below both displayed values, particularly with an unoptimized reference adapter.

For one sequence at the selected context length, let `A` be active expert weight bytes per generated token, `r` the resident expert fraction, `C` resident core bytes, and `S` attention/recurrent state bytes. The model uses:

```text
CPU traffic = A * (1 - r)
GPU traffic = C + A * r + S
serial token time = CPU traffic / CPU bandwidth + GPU traffic / GPU bandwidth
theoretical output tokens/second = 1 / token time
```

Reference dispatch uses serial CPU/GPU time. For the optimized engine, the optimistic endpoint permits ideal overlap and uses the larger of the two times. CPU timing samples supply the bandwidth spread; this is not a statistical confidence interval. Memory units are GiB (binary); bandwidth is GB/s (decimal).

The assumptions intentionally remain visible:

- Uniform expert selection; resident capacity is not a measured routing hit rate.
- Original source precision and no expert pruning or further quantization.
- No disk faults or competing inference. Compute, dequantization, launch/framework overhead, activation transfers, synchronization and adaptive expert uploads are omitted.
- Reference estimates describe default CPU-expert execution with fixed GPU layers. They do not estimate the optional staged backend's PCIe weight streaming.
- Core weights and attention state are charged once per output token. This is a simplified traffic model, not an execution trace. Conservative catalog core totals may include bytes the text-only path does not read.
- Prompt processing, batching, speculative decoding and task success rate are outside the estimate.

Automatic CPU bandwidth comes from three copies over two private buffers totaling 256 MiB, with warmed pages and up to eight workers. It is labeled **MEASURED_COPY_PROXY**: expert matrix operations can behave very differently. Results are cached for 60 seconds; the probe is skipped when available RAM is below 1 GiB. Automatic GPU bandwidth is the memory-interface peak calculated from CUDA clock/bus properties, or an exact-name published specification. It is not measured application bandwidth. Missing inputs produce an unavailable speed instead of a made-up number.

## Historical measurements

A separate **MEASURED_HISTORICAL** line appears only for matching GPT-OSS host identity, CPU core count, GPU memory/name and verified kernel hash. The bundled native comparison observed 16.89–17.15 output tok/s; the portable AVX2 comparison observed 15.06 tok/s in its own run. These were frozen-placement, single-sequence tests with 87–2597 prompt tokens. They are not fresh measurements at the selected context or under current contention. No GPT-OSS measurement is transferred to a different model.

## Refreshing the catalog

Verify a new entry against official, revision-pinned configuration and Safetensors metadata. Account separately for routed experts, always-active shared experts, core weights and attention/recurrent state. Confirm architecture, dtype and source-format compatibility with Neural's registry; do not infer runnable support from the MoE label alone. Add validation provenance and byte derivations, then run the catalog, advisor and relevant adapter tests. Only full-model evidence can promote an experimental candidate to a validated recommendation.

The HTTP service binds to IPv4 loopback, validates Host/Origin and does not expose model-serving or download actions. Its page renders catalog text without interpreting it as HTML.
