# GPT-OSS-based replacement: validation, 2026-10-03

## What was replaced

The Qwen-specific launchers in NeuralQwen36 now select a model-aware front door
around the optimized GPT-OSS engine inherited from NeuralServer `772399c`.
The separate GPT-OSS production checkout and its environment were not changed.
The earlier Qwen-based generalization experiment was archived locally; its
Qwen timings are not evidence for this replacement.

All full-model measurements below use the already downloaded GPT-OSS 120B at
revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, original MXFP4 codes and
the existing lossless PS4 store. No requantization, pruning, changed router,
reduced expert fanout or replacement checkpoint was used.

## Matched native-path measurement — MEASURED

[`native_abba.json`](../benchmarks/gptoss_agnostic_20261003/native_abba.json)
records four independent server sessions in direct / launcher / launcher /
direct order. Each session generated 128 tokens for each of three requests:
a 2,597-token code review prompt, its exact repeat, and an 87-token coding
prompt. All 1,536 generated token IDs matched across the corresponding requests;
prompt hashes also matched. The repeated long prompt reused 2,596 cached tokens.

Aggregated decode rates were **16.89 tok/s direct** and **17.15 tok/s through
the new launcher**. This 1.5% difference is treated as run variation, not a new
speedup. The purpose of this comparison was to verify that generalization kept
the established native performance and output. It is not a new llama.cpp run.

The host was the existing RTX 3080 Ti / AVX-512-capable Windows PC. Both paths
used the same 6.05 GiB expert pool, 16,384-token context, 256-token sliding
rings, static expert placement and kernels. Admission refresh was frozen
(`--refresh-every 0 --prefill-m 0`) so changing CPU/GPU placement could not
confound arithmetic comparisons. The default serving policy retains inherited
adaptive admission. Exact commands, requests, output hashes and timings are
in the raw reports. Decode throughput excludes the first generated token;
startup, prefill and request wall time are recorded separately.

## Startup warming experiments — MEASURED, not promoted for native use

The inherited warm-up runs CPU experts with zero input to fault mapped weight
pages into RAM. A byte-only alternative avoids this discarded computation.
Neither variant changes weights, routing, scratch state or returned output.

- [`page_warm_abba.json`](../benchmarks/gptoss_agnostic_20261003/page_warm_abba.json):
  serial page reads took **41.24–43.40 seconds**, versus **23.66–23.77 seconds**
  for legacy kernel warming over 49.1 GiB. Decode was 17.80 versus 18.37 tok/s.
  All 768 generated token IDs matched. Serial warming regressed startup and
  was rejected as a native default.
- [`page_parallel_abba.json`](../benchmarks/gptoss_agnostic_20261003/page_parallel_abba.json):
  bounded parallel page reads took **22.97–28.36 seconds**, versus
  **23.61–23.88 seconds** for kernel warming. Decode was 17.97 versus 18.04
  tok/s; all 768 token IDs matched. This was inconclusive for startup and
  neutral for decode. Native serving keeps `--warm-method kernel`.

The byte-only methods remain explicit experimental options. Parallel warming
also avoids making a slower CPU compatibility kernel perform discarded model
arithmetic during startup; that composite path is tested separately.

## CPU portability

The inherited binaries were compiled for the original host. Checking AVX-512
alone does not make those `-march=native` binaries safe on another CPU.
The launcher restricts them to their verified CPU signature and model profile.
A separate explicit AVX2/FMA build provides the same MXFP4/PS4 expert ABI.
It has no persistent-team claim and does not alter the model representation.

The initial scalar implementation passed full synthetic tests and a short
full-model check, but was too slow to be a useful default on this host.
[`portable_model.json`](../benchmarks/gptoss_agnostic_20261003/portable_model.json)
records **0.834 tok/s** for that version versus **15.57 tok/s** native across
three eight-token requests, with all 24 corresponding token IDs matching.
These are short diagnostic timings, not a stable headline benchmark.

The final vectorized AVX2/FMA kernel uses two eight-lane vectors to reproduce
the reference's sixteen-lane accumulation, with transient OpenMP teams working
on independent rows. In
[`portable_avx2_abba.json`](../benchmarks/gptoss_agnostic_20261003/portable_avx2_abba.json),
direct / portable / portable / direct sessions generated 64 tokens per request.
All **768 generated token IDs matched**. Aggregate decode was **15.06 tok/s
portable** versus **18.24 tok/s native**, about 82.6% of native throughput on
this host. This is a usable compatibility path, not an improvement over the
existing AVX-512 engine. The different generation lengths mean the scalar
eight-token diagnostic should not be used to claim a precise speedup factor.

Portable startup completed in **44.56–44.59 seconds**, including 23.07–23.26
seconds of parallel page warming; native startup was 45.14–45.62 seconds.
The 2,597-token cold prompt took 10.63–10.74 seconds to prefill on portable
versus 9.51–9.57 native. The 87-token prompt took 5.19–5.35 versus 1.12 seconds:
the portable prefill implementation still lacks native weight reuse. The
exact repeated prompt reused the cache on both. These costs are recorded
separately so decode throughput does not hide the prefill limitation.

The final CPU build and its exact validation are described in
[`GPTOSS_PORTABLE_CPU.md`](GPTOSS_PORTABLE_CPU.md). Selection requires a full
passing parity report tied to the exact binary hash; a smoke report is
insufficient. Synthetic parity compares raw and PS4 source bytes, intermediate
scratch, decode, multi-token prefill, masked combine and auxiliary ABI behavior.
The final validator also corrects an earlier down-scratch offset in the test
itself. Historical scalar reports remain execution records; the final selected
binary is gated by the corrected AVX2 report, not those earlier reports.

## Other architectures and support limits

Reference adapters cover Mixtral, Qwen3 MoE and Qwen3.5 MoE architecture families.
Tiny generated CPU fixtures test sparse expert dispatch, cache ownership,
protocol formatting and serving. Three tiny CUDA fixtures compared their
outputs with upstream Transformers and checked prefix-cache parity. These
fixtures are not pretrained models and establish no full-model quality or
throughput claim. No downloaded Qwen model was used for further optimization.

The final focused Python suite passed **103 tests**, covering metadata refusal,
hardware/ISA and ABI gates, CLI controls, byte-only warming, and the reference
adapters/cache/protocol/server. The inherited targeted decode/store tests also
passed during development. The full repository includes scripts requiring
live services; no claim is made that every such script was run as a unit test.

The optimized path remains specific to the validated GPT-OSS 120B geometry.
GPT-OSS 20B, arbitrary MoE families, different source quantizations, non-Windows
native serving and multi-GPU execution are not silently accepted as supported.
Adding a family requires an adapter describing its original expert arithmetic,
router, attention/cache and protocol semantics, plus parity evidence. Reusing
GPT-OSS's exact expert kernels for a BF16 SiLU model would not be correct.

Hardware plans are **CALCULATED capacity estimates**, not measured throughput
predictions or automatic scheduling calibration. Source revision checks use
Hugging Face download metadata; `source_payload_hash_verified` remains false
because startup does not cryptographically rehash every checkpoint tensor.
`store_payload_hash_verified` is also false: inspection checks the store's
hash-field schema and file sizes, while payload verification remains in the
existing build/pack tools. Neither metadata-only check proves tensor integrity.

## Reproduction

Use the target repository's `.venv\Scripts\python.exe`. Run full-model checks
with exclusive GPU access; the validation driver owns and stops only its
created server processes. Requests and metrics are archived after every turn.

```bat
.venv\Scripts\python.exe tools\validate_gptoss_runtime.py
.venv\Scripts\python.exe tools\validate_portable_cpu.py --full
```

The CPU validator executes the inherited reference DLLs, so running it requires
the verified AVX-512 validation host. Deployment on an AVX2-only CPU uses the
already validated binary and its report rather than executing AVX-512 locally.
No broad hardware claim is inferred from tests on this single machine.
