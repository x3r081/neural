# Decode follow-up — October 1, 2026

This follow-up addresses the distinction between faster prompt processing and
faster generated tokens. The first review found a repeatable warm 3k prefill
improvement and lower allocator reservation, but no reliable decode gain.
That result did not establish that software decode optimization was exhausted.

The follow-up starts from reviewed upstream `a5d3c95` plus the first review's
isolated changes (`2f16a36`). The new opt-in implementation is `9aa766f`.
The checkpoint, native MXFP4 precision, packed expert store, expert-selection
rule and context capacity are unchanged. Performance workloads run serially.

## Remaining cost and selected experiment

In the first review's contemporaneous stock follow-up measurements, median
per-token components were approximately 31.31 ms CPU expert calls, 4.84 ms GPU
wait, 1.45 ms graph replay submissions, 0.90 ms admission bookkeeping and
3.04 ms other work. These are separate categories; the last already excludes
the preceding four. Their marginal medians need not add to the median wall time.

The shipped kernel already uses dynamic row scheduling and a custom yielding
barrier between its two phases. It still enters and leaves an OpenMP parallel
region for each nonempty expert call, up to 36 times per decoded token. A
persistent team tests that remaining lifecycle cost while retaining the same
row chunks, arithmetic, scalar activation, scratch materialization, capture
bytes and expert accumulation order. The calling thread remains a worker;
seven native threads complete the eight-thread configuration.

The separate `gptoss_cpu_persistent.dll` defaults to the original path. Enabling
it requires `--kdll gptoss_cpu_persistent.dll --kernel-persistent 1` and
`--kernel-fuse 1`. The original DLL and included kernel source remain unchanged.
Jobs, settings and shutdown are serialized; capture stores are fenced before
workers publish completion. Unsupported configurations use the reference path
and are counted. Request results expose actual persistent jobs and fallbacks.

## Completed evidence

MEASURED, CPU-only: the no-op rendezvous probe measured 8-thread OpenMP versus
persistent overhead of 45.45/0.80 microseconds with no host gap, 36.40/1.10
with a 200-microsecond gap, and 100.95/45.35 with a 1-ms gap. These are not
model speeds. CALCULATED at 36 calls, the differences represent roughly
1.27–2.00 ms per token worth investigating, without proving that inference can
save all of it. The exact initial probe source and tool are archived with it.

MEASURED, synthetic expert calls: 432 strict bit comparisons passed across raw
and packed scales, 1/2/4/8 threads, 1–4 experts, varied signed inputs and capture
masks. Outputs, materialized scratch and captured bytes matched the original
DLL. The tests detected an intentional output corruption, exercised partial
thread-creation failures, restart, concurrent-call serialization and shutdown,
and verified 432 actual persistent jobs with zero fallbacks.

Packed, non-capturing call medians, original versus persistent, were:

- One expert: 0.3227 versus 0.2935 ms.
- Two experts: 0.6705 versus 0.6226 ms.
- Three experts: 1.0667 versus 1.0353 ms.
- Four experts: 1.2163 versus 1.1353 ms.

These are synthetic CPU results; capture-heavy timings varied. They do not
replace the serving comparison.

An initial prototype failed its FP-environment check before arithmetic tests:
the caller used x87 precision control `0x027f`, while OpenMP workers used
`0x037f`. Broadcasting caller settings to every native worker changed that
environment. The corrected implementation samples the actual OpenMP team's
controls when creating the pool, reproduces them for native workers and restores
thread controls after work. Nonstandard rounding/DAZ/FTZ callers fall back.
Dynamic assignment to a caller and workers with different FP environments is
an existing runtime characteristic; exactness claims remain tied to the tested
environments and cases.

MEASURED, full model: the frozen-residency check passed at 128, 3,000 and 13,000
prompt tokens with 64 decode steps per implementation and length. Every step's
full logits and final hidden state, all generated IDs, routing counts and the
complete K/V buffers matched. Both prefills used the original DLL. The candidate
executed 2,300/2,261/2,259 persistent jobs respectively, with zero fallbacks.
Correctness-run timings include hashing and synchronization and are not serving
speed measurements.

## Graph-launch alternative

MEASURED, GPU microbenchmark: public PyTorch replay, the underlying C-extension
method, and direct CUDA Runtime launches with the GIL released or held all
passed deterministic output checks. Each method began with poisoned buffers;
deliberately skipping a graph failed the negative control. The probe included
36-graph chains and external CUDA events on the same explicit stream.

At an 800-microsecond host gap, the hidden-width chain's adjusted submission
cost was 10.827 microseconds per launch for ordinary replay versus 9.023 for the
best direct path. CALCULATED over 36 launches, that is about 0.065 ms per token,
roughly 0.16% of a 42-ms token. The no-gap results varied by scenario/method.
This did not justify integrating a raw-handle path with extra ownership/RNG
constraints. No inference improvement is claimed. The initial probe attempt
failed on a CLI attribute typo before timing; its failed artifact is retained.

Primary API references: [PyTorch CUDA graph semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html),
[CUDAGraph ownership](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graphs.CUDAGraph.html),
[CUDA graph launch API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html).

## Serving comparison

The six-session mirrored trial completed: stock / persistent CPU /
combined profile / combined profile / persistent CPU / stock. It used the
same fixed source snapshot, existing native model/store and exact archived
requests, with a 13k coding prompt and four answers capped at 512 tokens.
The combined profile enables the already-tested bounded grouped prefill and
CUDA cache release. This workload has 13,108 new tokens initially and
518–521 on follow-ups, outside the grouped dispatch range; its recorded
group counts are zero. It measures persistent CPU plus CUDA cache release.
All 24 request/response archive links and request/message hashes verified.
Every enabled request reported actual persistent work, eight team threads
and zero fallbacks. The original DLL remained unchanged.

MEASURED, all four answers per session: stock weighted decode throughput was
23.8649 tokens/s, persistent CPU 24.2807 (+1.74%), and the combined profile
24.5437 (+2.84%). The CPU-only and stock sessions matched all corresponding
prompt and generated token IDs. The last CPU-only answer in the second session
was slower (22.9723 tokens/s, 1.16 system-wide disk MB/token); it remains in the aggregate.
The disk counter cannot uniquely attribute those reads to the kernel.
Both CPU-only sessions were faster overall than the controls.

The combined profile's second session changed generated IDs on its last two
answers. The strict all-session identity audit therefore correctly reports
`success: false`, with the generated-ID gate as its only failed prerequisite.
All internal prompt IDs still matched. This is consistent with the adaptive
placement issue also seen in the earlier stock control; it is not evidence of
full serving equivalence, nor a reason to remove those observations.

Restricting comparison explicitly to the first two answers, whose full prompt
and output token sequences matched across every session, gives stock 23.2288
tokens/s versus persistent CPU 23.9223 (+2.99%); the combined profile gained
2.89%. This subset was selected after observing the divergence. With only two
sessions per setting and possible time/resource drift, it supports a modest
workload-specific improvement rather than a precise universal effect. It is
not a claim that every answer is 3% faster.

## Frozen-placement serving repeat

MEASURED, a separate stock / persistent / persistent / stock run replayed the
same exact API requests with `--refresh-every 0`. All four sessions completed
four 512-token answers. Every corresponding prompt and generated-token hash
matched across all four sessions; request/response archive integrity passed.

Weighted decode improved from **19.8659 to 20.4391 tokens/s (+2.885%)**.
Each setting timed 4,088 tokens: 205.7795 seconds stock versus 200.0092 seconds
persistent. That is 1.41 ms less per timed token. Marginal CPU-call medians
were 40.0843 versus 39.1637 ms/token; graph-submission medians were 1.7113
versus 1.4735 ms/token. The latter also runs on the CPU and can benefit from
less thread scheduling contention; it does not establish a CUDA-kernel speedup.

This tighter repeat supports the new CPU implementation's modest gain without
the adaptive output divergence. It is deliberately slower overall because it
disables expert admission. Keep normal adaptive placement for use; the frozen
setting is an experimental control, not a recommended optimization.

Artifacts: `persistent_cpu_frozen.json`, its replay archive and
`persistent_cpu_frozen_summary.json`; command: `run_cpu_frozen.ps1`.

## Fresh llama.cpp comparison

MEASURED, opt / llama / llama / opt, using the installed llama.cpp build
10361 (`14e78ddef`, Clang 20.1.8 Windows x86_64), existing native MXFP4 GGUF,
and the exact same four saved API request bodies as both CPU studies.
The llama arguments were `-ngl 99 -ncmoe 31 -t 8 -c 16384 -fa on --jinja`.

Weighted all-answer decode was **24.6910 tokens/s Neural versus 14.3189
llama.cpp, a 1.7244x ratio**. First-answer weighted rates were 23.6047 versus
13.0566; follow-up weighted rates were 25.0756 versus 14.7957. First-request
client-wall medians were 30.38 versus 176.82 seconds, and follow-up medians
22.835 versus 39.405 seconds. Prompt-processing medians were 8.666 versus
137.552 seconds initially, and 2.258 versus 5.061 on follow-ups.

This is a new matched-API-input comparison against the installed decode-tuned
configuration, not an exhaustive search for llama.cpp's best prompt settings.
It is **not** a 72% gain from the new CPU change. That change's measured
increment is 1.74% in the adaptive study and 2.885% in the frozen repeat.
The native Neural and existing GGUF backends need not have identical numerical
outputs. llama.cpp does not expose internal prompt-token hashes in this runner;
API-body equality and matching token counts do not prove rendered-token equality.
Each engine's reported decode timing convention is retained: Neural times 511
tokens per 512-token answer; llama reports 512. The aggregates use those counts.

All archive hashes, shared source/context fingerprints, candidate/control DLL
hashes and per-request persistent-job checks passed. Both Neural repeats
matched each other's full prompt and generated-token hashes on all four
answers. This does not erase the earlier combined-run divergence.
See `persistent_vs_llama.json`, its replay archive, `serving_verified.json`,
`llama_version.log` and `run_cpu_vs_llama.ps1`.

## Selected configuration

The isolated [`start_neural_optimized.bat`](../start_neural_optimized.bat)
now enables the separate persistent DLL alongside the first pass's bounded
grouped prefill and CUDA cache release. Normal adaptive placement stays on;
masked GEMV stays off. The original launcher and DLL remain unchanged.
The earlier 7.03% warm-3k prefill result measures the first-pass profile, not
an additional new full-profile timing experiment at 3k.

To disable only the new CPU change while retaining the earlier profile:

```bat
start_neural_optimized.bat --kdll gptoss_cpu_cap2.dll --kernel-persistent 0
```

The new default applies only to this isolated launcher. The server CLI itself
continues to default to the original DLL with persistence disabled.

## Validation and remaining limits

The selected repository suite passed **103 tests**, with 23 existing warnings:
`test_kv_ring.py`, `test_splice.py`, `test_scale_layout.py`,
`test_doorbell_mock.py`, `test_grouped_gemm_plan.py` and `test_review_bench.py`.
The standalone 432 native cases and full-model exactness checks are additional.
Python compilation checks passed. Whole-directory pytest remains unsuitable
because existing standalone scripts parse CLI arguments during collection;
this is not a claim that every repository test was run.

These are small repeated studies on one Windows host and one coding workload.
They show further software optimization is possible, with a modest decode gain
and no added quantization. They do not prove a universal percentage, exact
adaptive-serving determinism or optimality across every workload. FP exactness
is bounded by the tested environments described above. No slow-client or
multi-request throughput test was performed.

CPU memory traffic still dominates. Further kernel scheduling and host/GPU
coordination changes need new controlled measurements. The earlier hypotheses
around idle canonicalization, side-request snapshots and streaming latency
remain workload-dependent opportunities, not delivered performance claims.
The direct graph-launch path was not adopted. Original model files, precision
and expert-selection policy are preserved.

## Reproduction and binary scope

Evidence directory: `benchmarks/codex_decode_followup_20261001/`. The complete
serving command is saved in `run_cpu_mirrored.ps1`. Native microchecks use
`tools/persistent_decode_ab.py`; the no-op probe uses
`tools/persistent_team_probe.py`; model equality uses
`--review-cpu-decode-check <fresh.json> --refresh-every 0`, with the standard
model/store/launch settings and the original reference DLL.

The new DLL targets this Windows/AVX-512 host and was built with GCC 16.1,
`-O3 -march=native -fopenmp -shared`; it is not a portable cross-platform binary.
The exact build command, source/include/DLL hashes and checked FP environment
are preserved in `persistent_decode_ab.json`. The native checker rebuilds its
candidate DLL and must only run with serving processes stopped.
