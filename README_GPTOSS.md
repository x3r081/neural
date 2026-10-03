# Neural server - gpt-oss-120B on one consumer GPU

A local, OpenAI-compatible server for **gpt-oss-120B** (117B-parameter MoE, MXFP4) that runs
on a 12 GiB GPU plus 64 GB of RAM. Experts are split between VRAM and the CPU: the hottest
experts live on the GPU, the rest are computed on the CPU straight from a memory-mapped
expert store, and the per-layer attention/router core runs as fused Triton kernels in CUDA
graphs. Built for agent harnesses (Continue, Cline, OpenAI SDK...) via
`/v1/chat/completions` with streaming and tool calls.

It grew out of the Neural research project (heterogeneous-memory MoE inference); this repo
holds only what the server needs to run.

- **How it works, technically:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), with diagrams
  of the memory topology, the per-token decode, expert admission, prompt processing and the
  request lifecycle.
- **Plain-language overview:** [`site/index.html`](site/index.html), a single self-contained
  page, published at https://x3r081.github.io/neural-site/ (public repo `x3r081/neural-site`;
  to update it, copy `site/index.html` there as `index.html` and push).

## Requirements

- Windows 10/11, NVIDIA GPU with >= 12 GiB VRAM (developed on an RTX 3080 Ti), CUDA 13 driver
- CPU with **AVX-512**: Intel 11th-gen Core (Rocket Lake), Ice Lake / Sapphire Rapids Xeon, AMD Zen 4+
  (Intel Core 12th gen and later lack it), for the CPU expert kernels
- 64 GB RAM (the expert store is ~57 GiB and should mostly stay in the page cache;
  ~55 GiB free when starting gives the best speed)
- ~65 GB disk for the checkpoint + ~57 GiB for the expert store (NVMe recommended)
- Python 3.12

## Setup (fresh machine)

In Command Prompt (not PowerShell), all in one window:

```bat
git clone https://github.com/x3r081/NeuralServer.git
cd NeuralServer
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

rem 1. download the model (~65 GB; the repo also holds original/ and metal/ copies we skip)
.venv\Scripts\hf download openai/gpt-oss-120b --exclude "original/*" --exclude "metal/*" --local-dir D:\Models\gpt-oss-120b

rem 2. point the server at it and build the expert store (one time, a few minutes)
set NEURAL_MODEL_DIR=D:\Models\gpt-oss-120b
set NEURAL_STORE_DIR=D:\NeuralStores\gptoss120b_prepacked
.venv\Scripts\python tools\build_store.py
```

Set `NEURAL_MODEL_DIR` / `NEURAL_STORE_DIR` permanently (System Properties -> Environment
Variables), or edit the defaults in `paths.py`. The store is a byte-exact repack of the
checkpoint's MXFP4 experts (no requantization).

The prebuilt CPU kernels (`*.dll`) need nothing else at runtime. To rebuild them after
editing `kernels\*.c`, put a MinGW-w64 gcc on PATH and run `tools\build_kernels.bat`.

## Run

```bat
start_neural_server.bat
```

About 2 minutes to start (model load + ~20-30 s RAM warm-up). Then:

| setting | value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` (this PC only) |
| API key | any value; set `NEURAL_API_KEY` before starting to require one |
| Model | `gpt-oss-120b-neural` |
| Context | 16,384 tokens |

- **Continue** (VS Code / Cursor extension): copy `continue\config.yaml` to
  `%USERPROFILE%\.continue\config.yaml`, open the Continue panel, pick Agent mode.
  (Cursor's built-in agent cannot use local models: it calls them from Cursor's cloud.)
- **Terminal chat**: `start_neural_chat.bat` (talks to the running server).
- **Any OpenAI client**: base URL above, any key.

### Packed expert store (optional, 2.88% smaller experts)

`tools\pack_store.py` converts the expert store into a **packed-scale** form. Every expert keeps its
code bytes exactly; each 90-byte row of E8M0 scales is stored as one base byte (the row minimum) plus
4-bit deltas, 46 B instead of 90. The kernels rebuild the identical scale value (base + delta), so all
outputs are **bit-identical** to the raw store's (checked on the CPU kernels, the Triton kernels on the GPU,
and per-turn in the server). An expert is 12,839,040 B instead of 13,219,200 B (-2.88%): the whole store is
55.1 GiB instead of 56.7 GiB, and at the launcher's `--pool 6.05` the same VRAM holds 14 more expert slots (CALCULATED). Measured on the dev PC
(same slot count, temperature 0): follow-up answers 20.63 vs 19.88 tok/s (+3.8%), first answers 19.3 vs
16.8 tok/s (+15%), MEASURED.

```bat
rem read-only go/no-go scan of every scale row of the raw store (a row whose span exceeds 15 cannot be packed)
.venv\Scripts\python tools\pack_store.py --check-only

rem convert (resumable, ~2 minutes, needs ~55.2 GiB free in <dir>) and verify byte for byte; the source is never modified
.venv\Scripts\python tools\pack_store.py --dst <dir> --verify

rem use it (the variable or the flag; same launcher, same kernel DLLs)
set NEURAL_STORE_DIR=<dir>
start_neural_server.bat
rem   or: start_neural_server.bat --store-dir <dir>
```

The server reads the layout from the store's `metadata.json`, so raw and packed stores both just work; keep
the raw store if you want to go back. Older checkouts and tools that predate this change (and any
`gptoss_cpu_*.dll` built before it) are raw-only: they refuse a packed store instead of misreading it.

## What it supports

- `/v1/chat/completions` (streaming and non-streaming), `/v1/models`
- tool calls in gpt-oss's native (harmony) format, one per response
- `reasoning_effort` low / medium / high (default medium; reasoning streams as
  `reasoning_content`), `temperature` (default 1.0, 0 = greedy), `max_tokens`, `stop`
- **prompt cache**: the unchanged start of a conversation is reused between requests
- **tool loops**: the model's exact tool-call output is re-used verbatim, so each step only
  processes the new tool result, and the model keeps its reasoning between calls
- **idle work**: after a final answer the conversation is pre-processed for the next turn
- **side requests** (e.g. Continue's apply / title requests) don't evict the conversation;
  a new chat that continues takes over cleanly
- per-request timings in `logs\requests.jsonl` (no text unless `NEURAL_LOG_BODIES=1`)
- runtime knobs: `POST /neural/config` e.g. `{"cpu_prefill_max": 16}` or `{"prefill_weight": 16}`; routing trace for studies:
  `POST /neural/trace {"on": true}`, then `GET /neural/trace`

Not supported: parallel tool calls, forced `tool_choice` / `response_format`, images,
the Responses API, legacy `/v1/completions`.

## Performance

On the development machine (see `benchmarks\RESULTS.md`):
- **Speed:** code generation runs at ~13-19 tok/s, and prose is faster.
- **Agent steps:** a step inside a tool loop spends ~0.4-0.6 s processing new input.
- **Against llama.cpp b10361** (decode-tuned MoE offload `-ncmoe 31`, same PC, fresh server each
  run, alternating; `tools\bench_vs_llama.py`, `tools\bench_prompt_warm.py`):

  | workload | Neural vs llama.cpp |
  |---|---|
  | reading a 13,000 / 3,000-token prompt (both servers warm) | 3.1x / 3.0x faster |
  | 13,000-token coding conversation: first answer | 1.03x (about level; 0.93-1.18x per run) |
  | same: follow-up answers | 1.24x |
  | rewriting a file just read: writing, same answer length | 1.06x |

Code generation is bounded by RAM bandwidth (~29 GB/s). Code uses experts spread across the
whole model, so most are read from RAM every token. The expert store only just fits in
64 GB next to other programs, so close memory-hungry apps before starting the server. Lower
reasoning effort is the biggest remaining lever for agent wall time.

### Experimental decode levers (off by default)

`docs/LEVERS.md` describes flag-gated optimizations that keep outputs unchanged: kernel
memory-level-parallelism knobs (`--kernel-prefetch`, `--kernel-pair`, `--kernel-affinity`, bit-identical),
a rolling K/V cache for the sliding-window layers (`--kv-ring 256`, frees ~45 expert slots), a locked
expert arena (`--arena pinned`), zero-surcharge GPU admission (`--admit-gpu 1`, the lever the offline
policy study points at), near-miss candidates (`--near-miss 1`), and a near-future demand probe.
It also has the measurement runbook (`tools\measure_host_dram.py`, `tools\kernel_tuning_ab.py`) and
the order to test things in. Measured 2026-09-29 (tables at the top of `docs/LEVERS.md`): the rolling
K/V cache (+2-3%) and the two-phase CPU kernel (+6%) are bit-exact and are now the launcher defaults, as is `--prefill-order layer` (prompt processing runs each layer over all
blocks of the prompt so a non-resident expert is copied into the GPU once per layer instead of once per 4096-token block:
a 13k-token prompt 15.0 -> 10.7 s, bit-identical, `--prefill-order-check` proves it in one process), the prefill GEMM tile
`BLOCK_M=64` (10.7 -> 7.7 s, bit-identical), a staging thread for the expert copies and a cached Triton launch path, and
`--refresh-m 16` (follow-up answers +3-5%: half the admission captures at the same hit rate);
DMA admission from a pinned arena adds +1.5% more but needs 40 GiB of locked RAM; the GPU read pipe
and kernel thread pinning are losses in the server.

## Layout

| path | what |
|---|---|
| `server.py` | runtime setup + OpenAI HTTP server |
| `fused_core.py` | Triton kernels: fused decode core, split-K attention, prompt attention |
| `harmony_render.py` | chat-format rendering, tool-call token splicing |
| `cpu_prefill.py` | wrapper for the multi-token CPU expert kernel |
| `neural/` | the runtime modules from the Neural research project (slot pool, store, MXFP4 GEMV/GEMM kernels, gpt-oss adapter) |
| `kernels/` + `*.dll` | CPU expert kernels (C, AVX-512, OpenMP) and a threaded copy helper |
| `hotset_code.json` / `hotset_freq.json` | which experts start in VRAM (coding-calibrated / general) |
| `data/calib_prompts.json` | prompts for self-tests and for recalibrating a hot set |
| `tools/` | `build_store.py`, `pack_store.py` (raw store -> packed-scale store, `--check-only` / `--verify`), `build_kernels.bat`, `bench_agent.py`, `bench_vs_llama.py`, `bench_prompt_warm.py`, `calibrate_hotset.py`, `tune_prefill.py`; studies: `cache_policy_sim.py`, `admission_replay.py`, `lookup_spec_estimate.py`, `decode_kernel_ab.py` |
| `tests/` | kernel, rendering and server tests (see below) |
| `benchmarks/` | measured results |

## Tests

```bat
rem no model needed (tokenizer only / synthetic GPU data / a few store rows):
.venv\Scripts\python tests\test_splice.py
.venv\Scripts\python tests\test_attention.py --no-timing
.venv\Scripts\python tests\test_cpu_multi.py --no-timing
rem against a running server:
.venv\Scripts\python tests\test_protect.py
.venv\Scripts\python tools\bench_agent.py benchmarks\my_run.json
rem prompt-processing numerics vs the reference path (loads the model):
start_neural_server.bat --ppl-check
```
