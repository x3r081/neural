# Qwen3.6 validation record — 2026-10-02

The separate Qwen runtime works for the bounded text, streaming, tool-call and
coding checks below. It is currently slower than llama.cpp on this host. This
is an initial implementation and measurement, not a performance win or a
qualification for long autonomous coding sessions.

This private repository was cloned from NeuralServer at `772399c`. It targets
the official `Qwen/Qwen3.6-35B-A3B` checkpoint at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. The original runtime is unchanged.
Evidence files below are in [benchmarks/qwen36_20261002](../benchmarks/qwen36_20261002/).

## Weights and precision

All 26 original checkpoint shards were downloaded and their SHA-256 hashes
verified against official LFS digests. All 1,045 source tensors are BF16.
The text-only GGUF contains 733 tensors: 432 BF16 and 301 F32. All 733 payloads
were compared exactly with the source after the pinned converter's transforms,
covering 34,660,610,688 elements. The converter widens selected tensors and
performs architecture transformations; this is not quantization.

Vision and optional MTP tensors are excluded from ordinary text generation in
both paths. No added quantization, expert pruning or checkpoint substitution
was used. Preserved weights do not imply bit-identical floating-point execution.
See `checkpoint_verification.json`, `gguf_verification.json` and
[the conversion record](LLAMA_QWEN_BASELINE.md).

## Functional and numerical checks

- **23 focused tests passed**: storage, native expert components, protocol,
  optional fast delta attention and grouped dispatch. See `focused_tests_final.log`.
- **10/10 live HTTP checks passed on the final default Torch runtime**:
  health/models, malformed-input rejection, plain/streamed arithmetic, streamed
  weather-function arguments, consumption of a local tool result and generated
  Python. No external tool was called.
- The generated `merge_intervals` function passed **five behavior cases**,
  including empty input, overlap, touching intervals and nesting, plus input
  preservation checks. This is a bounded function test, not a coding benchmark.
  Evidence: `neural_torch_final02/http_validation*.json`.
- The optional FLA path also passed those HTTP and generated-code checks.
- A teacher-forced comparison against staged CUDA/Torch checked two prompts
  and 34 next-token positions. Both native Torch variants matched the reference
  argmax at **34/34** positions; native FLA matched **33/34**, with one near-tie.
  All checked states were finite. Attention/conv caches were BF16 and recurrent
  state was FP32.
- Native Torch logits were not identical to the reference: the largest observed
  absolute difference was 1.53125. Different FP32 reduction orders are expected,
  but this small test does not establish general quality equivalence. See
  `numerics01.json`. Torch remains the default; FLA is opt-in.

The first Torch benchmark completed, but its later coding validator sent a
wrapper object instead of its nested chat request and received HTTP 400 before
generation. That client bug was fixed, and the final run passed. The original
failed validation record is retained.

## Measured throughput

The common client clock includes prompt processing, generation, serialization
and transport. Weighted rates divide total tokens by total elapsed time.
Each full run generated 1,536 tokens across three frozen requests.

- **Final Neural Torch:** 3.368 output tokens/s by client wall time;
  3.594 subsequent decode steps/s by its internal timer.
- **llama.cpp, three GPU expert layers:** 8.961 output tokens/s by client wall
  time; 10.467 subsequent decode steps/s. This was the fastest tested baseline.
- **llama.cpp, two GPU expert layers:** 8.687 output tokens/s by client wall time;
  10.014 subsequent decode steps/s by its internal timer.
- **Neural FLA experiment:** 3.465 output tokens/s by client wall time;
  3.828 subsequent decode steps/s. This used the earlier input-validation code.
- **Initial Neural Torch:** 3.284 output tokens/s by client wall time;
  3.657 subsequent decode steps/s. Retained as an initial observation, not a
  statistically established optimization gain.

The final Neural default achieved 37.6% of the fastest tested llama.cpp run's
end-to-end rate (llama was 2.66 times faster), and about 39% of the two-layer
baseline. All five cohorts passed the comparison metadata checks in
[`comparison.json`](../benchmarks/qwen36_20261002/comparison.json).
Internal decode timers have slightly different boundaries:
Neural includes incremental detokenization and sampling work that llama.cpp's
timer does not fully include. Use the client clock for the primary comparison.

Runs were serial on an i7-11700K, RTX 3080 Ti 12 GiB and 64 GiB RAM, using eight
CPU threads, a 4,096-token configured context, BF16 K/V, greedy sampling, an
eight-token warmup and a 512-token cap per request. Prompt lengths were 77, 73
and 163 tokens. The first llama placement used `-ngl 99 -ncmoe 38`; Neural used
native/grouped experts, Torch GDN, two final expert layers on CUDA and a 3 GiB
expert cache. The final llama run used `-ncmoe 37`, retaining one additional
expert layer on GPU; it completed without an out-of-memory error. The two-layer
llama run remains the closest placement comparison. No `GOMP_SPINCOUNT` override
was used in the throughput runs.
Startup was measured separately and excluded.

Every prompt's token IDs and evaluated count matched. Conversation state was
fresh for each request (`cache_prompt=false`; llama reported zero cached prompt
tokens). Expert and OS file-cache warmth may vary. All answers reached the
512-token cap: these thinking workloads measure throughput, not completed-answer
quality. Configured context capacity is not the actual prompt length.

Original run/result files are retained unchanged. Early
`comparison_metadata.json` files add explicitly labeled post-run precision
annotations and evidence hashes. `setup.json` describes the historical bootstrap
stage, not the current validation status.

## Overhead fix and exploratory profiles

Input validation previously recomputed `len(tokenizer)` for every prompt token.
For 77 tokens, a CPU-only comparison measured 3.48–3.77 seconds before caching
the vocabulary size and about 9–10 microseconds afterward, with identical
validity checks. See `input_validation_overhead.json`.

All 1,536 output token IDs matched between initial and final Torch runs
(`input_validation_equivalence.json`). This verifies those outputs after the
validation/protocol changes, not equivalence with llama.cpp. It does not prove
a decode gain: final Torch decode was slightly slower in this run.

Instrumented profiles identify native CPU experts and gated-delta work as
substantial costs. Component synchronization changes timing behavior, so profile
rates are not normal generation throughput. The `GOMP_SPINCOUNT=20000000`
profile is exploratory and is not a shipped setting.

## Memory and practical limits

The BF16 expert pool puts substantial pressure on this host's RAM. Initial
runs recorded global available-memory minima below 0.3 GiB; the model does not
fit comfortably in otherwise free RAM.

Old Neural `max_server_rss_bytes` values measured the small Windows venv
redirector, not its inference child, and must not be used for model-memory
comparisons. `neural_torch_final02/process_tree_snapshot.json` separately
observed roughly 53.5 GiB RSS in the inference child. This is a snapshot, not a
high-water mark. Commit `818c014` fixes owned-process-tree monitoring and cleanup.
The initial native llama executable's roughly 54.1 GiB RSS measurement is
unaffected. The final llama process-tree peak was about 53.9 GiB RSS, with
0.286 GiB minimum available system RAM. Earlier raw reports are not rewritten.

These short serial runs have no confidence intervals. They do not establish
long-context performance, extended agent reliability or cross-engine output
identity. The server's 16,384-token default is configured capacity; full-context
inference has not been validated here. Each HTTP request currently reprocesses
the supplied conversation history instead of reusing prefix KV.

## Reproduction

From the new project directory, run one engine at a time with a fresh name:

```powershell
.\.venv\Scripts\python.exe tools\run_qwen_benchmark.py `
  --engine neural --name neural_torch_repro `
  --requests benchmarks\qwen36_20261002\requests_v1.json `
  --expert-backend native --gdn-backend torch `
  --native-gpu-layers 2 --gpu-cache 3 --context 4096 --threads 8 `
  --validate-http

.\.venv\Scripts\python.exe tools\run_qwen_benchmark.py `
  --engine llama --name llama_bf16_repro `
  --requests benchmarks\qwen36_20261002\requests_v1.json `
  --cpu-moe-layers 37 --context 4096 --threads 8
```

Use `--cpu-moe-layers 38` to reproduce the first llama placement.
Run the five focused test files listed in the [README](../README.md). Optional
FLA uses `--gdn-backend fla` after installing the pinned optional dependencies
into the new project environment.

`tools/summarize_qwen_benchmarks.py` checks full cohort coverage, prompt IDs,
generation settings, model revision, context and precision metadata before
calculating relative rates. Partial, repeated and validation-only runs are
labeled separately and are not canonical single-pass comparison cohorts.
