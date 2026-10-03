# NeuralServer optimization review — October 1, 2026

This document records the first prefill/cache review. The subsequent
[decode follow-up](CODEX_DECODE_FOLLOWUP_2026-10-01.md) investigates persistent
CPU workers and reports additional measurements. The figures below remain
the original first-pass results.

The selected profile reduces warm 3k prompt-processing latency by **7.03%**
(3.697 to 3.437 seconds) and GPU allocator reservation by about **1.11 GiB**
after a long prompt. It uses the existing model and precision. Two earlier
grouped-kernel studies independently measured 7.04% and 7.43% lower warm 3k
latency. No reliable decode-speed improvement was found in this first pass; the integrated cold
full-answer trial was essentially unchanged in total latency and decode speed.

Launch this isolated, validated configuration with
[`start_neural_optimized.bat`](../start_neural_optimized.bat). It enables bounded
grouped prefill only for 1,024–4,096 new rendered tokens, releases unused CUDA
allocator cache at the prefill boundary, and leaves masked decode disabled.
The launcher now also includes the subsequent persistent CPU improvement;
see the follow-up for its separate decode measurements and quality scope.
The original launcher and existing research worktrees remain untouched.
No benchmark server is left running at delivery.

## Version and scope

The starting point is NeuralServer `a5d3c95e17323e9b78f83976da583e31f73f188e`,
the remote heads of both `main` and `claude/decode-levers` when checked on
2026-09-30 and again at final delivery on 2026-10-01. The supplied technical page describes this runtime as Sep 30 (4),
or v9. The site checkout is `2f30846`. The older Neural research checkout
`214b937` is not the latest server; its closure conclusions do not describe the
current CPU/GPU hybrid implementation.

Sources: [reviewed upstream commit](https://github.com/x3r081/NeuralServer/commit/a5d3c95e17323e9b78f83976da583e31f73f188e),
[supplied technical documentation](https://x3r081.github.io/neural-site/technical.html).
The latter is also archived as `published_technical.html` in the evidence folder.

The control checkout is `../neuralserver-base`, detached at `a5d3c95`; the
candidate is this checkout, branch `codex/optimization-review-20260930`.
Existing worktrees and local research have not been reset or overwritten.

Model: the existing `G:\Models\gpt-oss-120b` checkpoint and losslessly packed
`G:\NeuralStores\gptoss120b_ps4` expert store (`gpt-oss-120b-b5c939de`, native
MXFP4, packed scale deltas). No model download, checkpoint modification,
requantization, routing approximation or hardware change is part of this work.
The Q80 and GLM research stores remain untouched.

Measured host: RTX 3080 Ti (12 GiB), i7-11700K, 64 GiB RAM, Windows;
project venv with PyTorch `2.10.0+cu130` and Triton `3.7.1`. Performance jobs ran
serially. Available RAM and desktop GPU occupancy varied and are material
limitations of a shared interactive host.

## Reproduced control

MEASURED: `../neuralserver-base/benchmarks/codex_baseline_20260930.json`,
fresh processes in Neural / llama.cpp / llama.cpp / Neural order, 13,000-token
code context and four answers capped at 512 tokens, temperature zero and low
reasoning effort. Neural uses the current launcher flags and packed store;
llama.cpp b10361 uses the existing native MXFP4 GGUF and the previously published
decode-oriented `-ngl 99 -ncmoe 31 -t 8 -c 16384 -fa on --jinja` configuration.

The original harness reports follow-up medians of 22.58255 tok/s for Neural
and 14.98790 tok/s for llama.cpp. First-answer medians are 19.9415 and
13.78595 tok/s. These reproduce a useful current baseline, not an optimization
gain. Initial prompt processing is cold-start behavior, not the published warm
prompt benchmark. The run crossed midnight; the chat template inserts the date,
and each engine's own answers become its follow-up prompts. These are explicit
limitations for exact A/B interpretation. New tests must retain full prompt and
generated-token identities, not only hashes of visible answer text.

The first Neural process started with 51.58 GiB available RAM (published step27
started with 56.65 GiB). After its first prompt, available RAM was 0.26 GiB and
decode incurred 5.88 MB of system-wide disk reads per token. Prompt processing
took 28.28 seconds, including 17.49 seconds waiting for staging, despite only
4.28 seconds of host GEMM launch time. This is evidence of a different resource
regime, not proof of a code regression. GPU allocator reserved memory was
10.826 GiB versus 9.602 GiB live allocation. Windows desktop GPU occupancy also
varies. The attribution of the excess wait to GPU residency versus host memory
pressure is a hypothesis requiring experiment.

## Work performed

1. Release unused prompt-processing allocations at the synchronized prompt/decode
   boundary. `--trim-prefill-cache 0` is the control, `1` releases unused CUDA
   allocator blocks, `2` additionally releases cached CPU-prefill tensors and
   inactive pinned-host allocations. Live weights, expert slots, K/V and CUDA
   graph storage retain their addresses. Release cost is included in prompt
   latency. CUDA-only release is enabled in the new launcher for the measured
   memory saving; no decode speedup is claimed.
   Raw experiment: `benchmarks/codex_review_20261001/trim_study.json`.
2. Group independent prefill expert GEMMs into fewer GPU launches while preserving
   each expert/block pair's row boundaries, tiling and accumulation order. Previous
   work grouped pointwise epilogues only, leaving one matrix-kernel launch per
   pair. Standalone bit equality and full-model hidden/logit/K/V equality passed;
   the selected size guard excludes the lengths that did not benefit.
3. Strengthen measurement provenance, output identity and matched-prompt replay.
   Newly recorded `prompt_ids_sha256` and `generated_ids_sha256` include hidden
   reasoning and control tokens, unlike the old content-only SHA-1 check.

## Existing conclusions retained

Packed scales, dynamic CPU work sharing, the yielding barrier, layer-order
prefill and BLOCK_M=64 are existing shipped gains, not discoveries of this review.
The latest robust doorbell is off by default after a measured regression.
Repeatedly unsuccessful prefetch, thread affinity and GPU miss-streaming paths
are not being presented as new opportunities.

The published 13.9x warm-prompt ratio is one run per engine against decode-tuned
llama.cpp; it does not establish superiority over a prompt-tuned llama.cpp
configuration. The 1.96x first-answer headline also used unusually slow llama.cpp
sessions. Published follow-up decode evidence is stronger. The old rewrite
headline is not a fresh measurement of v9.

## New evidence collected on October 1

The core kernels are locally committed as `67ad1869e599ca7ed90525c041413e0752909cb8`;
the final runtime size guard is in `06d05ae`, and the combined warm runner is
in `171801a`. The runtime source has not changed since `06d05ae`.
New kernels and cache release remain opt-in at the server CLI; the added launcher
selects the tested profile. The implementation does not alter checkpoint
bytes, expert placement policy, the stock kernels, or their arithmetic precision.

### Unused allocator memory

MEASURED: the six-session `off,cuda,both,both,cuda,off` experiment completed,
four answers capped at 256 tokens per session. Every corresponding generated-token
hash, including hidden reasoning and control tokens, matched across all six
sessions. CUDA-only release lowered reserved memory from 10.826–10.828 GiB to
9.715 GiB. The first release freed 1.1113 GiB in about 6 ms, charged to prefill.
Live allocation stayed 9.602 GiB.

Follow-up medians were 23.54455 / 23.72825 / 23.69510 tok/s for off / CUDA /
CUDA+host. The CUDA-only median difference is 0.78%, within the observed run
variation. A 17.01 tok/s paging-affected control turn inflates the improvement in
the aggregate weighted rate; it must not be advertised as a repeatable speed gain.
One CUDA-only initial prompt also took 12.68 seconds, versus about 8.2 seconds for
the others. Host-cache release added no clear benefit and depends on a private
PyTorch API. CUDA-only release is the useful optional memory lever.

Raw evidence: `trim_study.json`, `trim_summary.json` in the review benchmark folder.

### Grouped expert GEMMs

The new ragged kernel groups independent expert/block pairs without combining
rows into a different tensor-core tile. Each pair retains BLOCK_M=64,
BLOCK_N=128 and the original ordered dot accumulation. This differs from the
previously rejected grouped-pointwise-epilogue experiment.

The first implementation passed full-model identity but could be slower.
Compiler inspection found that indirect descriptor pointers lost their alignment
information: scalar BF16 loads/stores replaced the stock vectorized path.
An integer-address hint alone did not fix it. A guarded hint on the pointer
*after* its cast restored 16-byte asynchronous loads and vector stores. The guard
checks the actual addresses and row pitches; unaligned inputs keep a tested
fallback. See `GROUPED_GEMM_PTX_AUDIT_2026-10-01.md` for compiler evidence.

MEASURED: the final pointer-hint kernel passed 120 synthetic GPU checks covering
raw/packed scales, aligned/unaligned pointers, ragged/empty pairs, guards,
determinism, graph replay and a deliberate output-corruption negative control.
Full-model identity then passed for group sizes 1, 3 and 6 at 128 / 3,000 /
5,000 / 13,000 tokens: every final hidden-state bit, every prompt position's
logit bit, and every layer's full K/V buffer matched production. Routing counts
and residency also matched. This is 12 full-model candidate/reference comparisons.

In that correctness run, group size 3 took 7.214 seconds on 13k tokens against
8.014 seconds for production. That is a screening result, not the final speed
claim: hashing, fixed order, compiler/cache state and other check overhead make
the repeated fresh-server warm benchmark necessary. The exploratory
`warm_prefill_final.json` sweep was stopped after two completed sessions: stock
warm long prompts took 7.484 / 7.211 seconds, while group3 took 10.499 / 10.339.
Reserved GPU memory rose from 10.830 to 11.361 GiB. This is a real rejection of
the large-workspace configuration, despite its correct output and faster kernels.
The bounded test uses a 1,024-row target instead of 16,384; this is a soft cap
because each original pair and its tiles remain intact. An individual pair can
exceed the target. The source also exposes a staging dependency: a grouped GEMM
waits for all of its experts' copies, whereas the stock path can compute one
expert while transferring the next. Host timing counters alone do not isolate
that effect from allocator/residency pressure.

MEASURED: the complete `warm_prefill_bounded_abba.json` sweep ran
off / group1 / group3 / group3 / group1 / off, with fresh servers, one discarded
warm-up and three distinct measured prompts per session. Group3 reduced the
3,000-token file prompt median from **3.79965 to 3.53230 seconds** (7.04% lower
latency; samples 3.8591/3.7402 versus 3.5209/3.5437). The two long-prompt medians
instead rose from 7.32785/7.24675 to 7.49485/7.40995 seconds, about 2.3% slower.
Group1 was also slower on long prompts. Reserved memory stayed about 10.83 GiB
with the bounded configuration. This selects **group3 for short prompts only**,
not universal grouping.

MEASURED: `warm_prefill_short_abba.json` completed off / group3 / group3 / off,
with both a discarded 13k warm-up and discarded 3k warm-up. Contexts of
128/512/1,024/3,000 tokens rendered to 239/686/1,138/3,122 total prompt tokens.
Median stock versus grouped prompt times were:

- 128 context: 1.83985 versus 1.85830 seconds; no useful gain.
- 512 context: 2.30645 versus 2.31030 seconds; no useful gain.
- 1,024 context: 2.62775 versus 2.56275 seconds; 2.47% lower median latency,
  a small effect with only two sessions per setting.
- 3,000 context: 3.81540 versus 3.53185 seconds; 7.43% lower median latency.
  Both grouped samples were faster than both controls, as in the earlier file
  prompt. The controls varied more than the grouped samples.

The selected guard is therefore `--grouped-prefill-min-tokens 1024
--grouped-prefill-max-tokens 4096`, requiring a single block in that range of
**new rendered tokens**. Tiny and multi-block inputs retain the stock path.
Maximum `0` disables the range guard for unbounded research. No process-global
switch is changed while a request runs. The first maximum-only guard passed
all hidden/logit/K/V hashes at 128/512/1,024/3,000/5,000/13,000 tokens and
checked actual group counts (`full_prefill_guarded_identity.json`). The final
selected range passed the same six-length proof in
`full_prefill_selected_identity.json`: 594/1,036 groups at 1,024/3,000 tokens,
zero at 128/512/5,000/13,000, with every dispatch assertion and bit comparison
passing. The earlier unbounded checks
independently prove the large-input kernel's arithmetic, even though the selected
configuration does not use it there.

`prefill_summary.json` independently recomputes both completed studies and
verifies every saved request/response hash. Every corresponding API request,
internal prompt-token hash and generated-token hash matches across all sessions.

### Final combined profile

MEASURED: `warm_profile_abba.json` completed off / selected / selected / off,
with fresh servers, discarded 13k and 3k warm-ups, the final 1,024–4,096 guard,
group size 3, 1,024-row workspace target and CUDA-only cache release. The
combined settings preserve the warm 3k improvement:

- 3k context, 3,122 rendered tokens: **3.69655 to 3.43665 seconds**, 7.03% lower
  prefill latency. Control samples were 3.6834/3.7097; selected samples
  3.4203/3.4530. Client wall median fell from 3.76 to 3.51 seconds.
- 1k context: 2.59705 to 2.56175 seconds, a small 1.36% difference.
- 128/512 contexts: effectively unchanged; the grouped kernel did not run.
- Reserved GPU memory during measured requests: 10.830 to 9.715 GiB.

All 24 archived requests/responses passed hash verification; corresponding
internal prompt and generated IDs matched, including the discarded warm-ups.
Every measured request had zero cached prefix tokens. The audit can be rerun
without inference using `benchmarks/codex_review_20261001/audit_warm_profiles.py`;
`warm_profiles_verified.json` records its results for all three warm studies.
Each study has only two sessions per setting. These measurements establish
the tested code-prompt improvement, not a universal 7% gain on arbitrary prompts.

MEASURED: `integrated_profile_abba.json` separately tested the selected settings
on **cold first requests with full 512-token answers**, off / selected /
selected / off. The 3k rewrite context rendered to 3,220 tokens. All four
complete generated-token sequences and internal prompt identities matched;
the replay archive and result links passed integrity checks. Median prefill
was 4.0858 versus 4.0114 seconds, a small difference within variation. Weighted
decode throughput was 24.5115 versus 24.4532 tok/s, and median client wall time
24.990 versus 24.965 seconds: essentially unchanged. Reserved memory was
10.248 versus 9.715 GiB, with 9.601 GiB live allocation. The reported first
release was 0.5273 GiB; the roughly 1.11 GiB saving belongs to long-prompt trials.
See `integrated_summary.json` and `integrated_profile_abba.json.replay.json`.

The launcher therefore offers a measured **warm prefill and memory benefit**.
It does not promise faster cold first answers or higher decode throughput.
Pass `--grouped-prefill-gemm 0 --trim-prefill-cache 0` to that launcher to
disable both new behaviors while retaining the same model/store and launch settings.

The initial `warm_prefill_bounded.json` attempt failed in its runner before
starting a server; it is not a timing result. A later test-harness stub accidentally
started a server, which was stopped; it overlapped no accepted performance or
quality measurement and contributes no result to this report.

### Skip GPU work with zero contribution

The current decoder sends all four selected experts through GPU GEMVs, including
CPU misses whose GPU weight is zero. The opt-in masked kernel skips those GPU
rows; it retains the active experts' arithmetic, biases, activation and original
four-row weighted reduction. It does not offload additional experts or move
weights across PCIe.

MEASURED: 512 finite synthetic cases passed across all 16 hit masks, signed/zero
inputs and biases, raw/packed scales and CUDA graph replay. The full model then
passed at 128 / 3,000 / 13,000 prompt tokens with 64 decode steps per path: full
logits and final residuals at every step, generated IDs, routing counts and full
K/V buffers matched bit-for-bit with residency frozen. See `masked_gemv_micro.json`
and `full_decode_identity.json`.

The finite-result condition matters: skipping a zero-weight NaN is not generic
IEEE equivalence. A deliberate NaN test documents this limitation rather than
hiding it. No non-finite result occurred in the accepted model checks. Kernel
microtimings are not token-speed claims; much of this work overlaps CPU work.

MEASURED: the four-session `masked_gemv_abba.json` trial completed in off / on /
on / off order, 13k code context and four 512-token answers per session, with
the first session's exact request bodies replayed thereafter. All request,
response and result/archive link hashes verify; internal prompt IDs match on
every turn. First-answer medians were 22.6894 versus 21.3730 tok/s; follow-up
medians 24.0232 versus 23.4691; total timed-token throughput 23.8298 versus
23.0605 tok/s (off versus on). **Rejected for the selected profile.**

There is a material identity caveat: both masked sessions match every generated
token of the first control, but the last control differs on answers 3 and 4.
This directly observes nondeterminism in the unchanged adaptive path and means
the complete aggregate is not an output-identical comparison. Restricting the
comparison to answers 1 and 2, which match across all four sessions, gives
23.20562 versus 22.13367 timed tok/s. Paging and GPU-wait time also increased in
the first masked session; these counters do not establish a unique cause.
The experiment provides no reason to enable the kernel, despite its strong
isolated-kernel result. See `masked_summary.json` and the full replay archive.

### Tested lossless codec: rejected

A bounded, read-only, stratified sample of 12 existing packed experts contained
154,068,480 bytes. All timed codec roundtrips were verified byte-for-byte and by
SHA-256. MEASURED raw DEFLATE level 1 reduced bytes by 4.6407% but decompressed at
only 0.299 GB/s on one thread. CALCULATED zero-order code entropy was 3.868 bits
per nibble; this is a sample-based memoryless coding estimate, not a universal
compression lower bound. The measured codec is nowhere near the decode path's
bandwidth requirement and was not integrated. Source stores remain unchanged.

## Critical review beyond the kernels

This was a subsystem review of the current runtime, its published measurements
and the older research lineage, rather than a claim to have proved every source
line. Coverage includes:

- **Loading and storage:** `paths.py`, the store loader, `tools/pack_store.py`,
  memory mapping and working-set warming. The existing lossless scale layout
  is valuable; retaining two complete model/store copies would not help this
  memory-bound workload. Startup and page-fault behavior depend on available
  desktop memory. This review leaves all model/store bytes intact.
- **CPU execution and admission:** the native expert kernels, `cpu_prefill.py`,
  hot-set calibration, staging and adaptive residency. The dated latest lever
  results supersede older timing paragraphs in `docs/ARCHITECTURE.md`. The CPU
  expert reads are already near the measured DRAM limit; policy simulations
  using future routing knowledge are not executable speedups. No router or
  expert selection approximation is proposed.
- **GPU decode and prompt processing:** `fused_core.py`, the MXFP4 kernels,
  graph capture, layer-order prefill and scratch-slot staging. Both arithmetic
  identity and whole-server behavior were checked for the new kernel paths.
  Isolated kernel timing is insufficient when work overlaps CPU execution or
  changes the staging pipeline.
- **Cache and API behavior:** `kv_ring.py`, `harmony_render.py`, rollback,
  tool-token splicing, side-request protection, idle canonicalization and the
  streaming request handler. These are correctness-sensitive service paths;
  the remaining latency opportunities below are hypotheses until exercised
  with their own workload and rollback/cancellation checks.
- **Evidence and delivery:** launcher settings, dated benchmark records,
  comparison tooling and selected/standalone tests. This is a Windows
  single-request-host study, not a production concurrency or slow-client load
  test. Published performance claims need their workload, cache state, engine
  settings and measurement date attached.

1. **Measure service latency as well as model phases.** The default idle
   canonicalization job can re-prefill while holding the request lock. Incoming
   requests can wait behind it; rendering also precedes the old generation clock,
   and decode timing excludes final admission/side-K/V restoration. Client wall
   time is essential. New `queue_wait_s`, `render_s` and `generation_wall_s`
   diagnostics expose those gaps without changing scheduling. Canceling or
   deferring speculative cache preparation once demand arrives is a remaining
   hypothesis, not a tested speedup.
2. **Side requests are a different workload.** The protection path can back up
   hundreds of MiB of K/V to CPU and restore it while holding the lock. Ordinary
   conversation decode benchmarks do not measure that editor/title/apply latency.
   Reusable pinned snapshots or modified-range backups require dedicated rollback
   identity tests before replacing the current correctness-preserving path.
3. **Streaming has client backpressure.** Template assembly/tokenization runs
   before model prefill; streaming JSON writes and flushes execute inline with
   generation. Nonstream throughput does not characterize slow-client streaming.
   An exact render cache or bounded writer queue needs measured service traces
   and cancellation tests, rather than assumptions about token speed.
4. **Quality claims have a scope.** The existing fused core is not bit-exact with
   eager PyTorch; its published quality acceptance uses a small null-envelope
   experiment. Adaptive residency already changes CPU/GPU assignments with
   timing. The new checks preserve the current runtime at fixed residency;
   they do not prove universal semantic equivalence between Neural, Transformers
   and llama.cpp. No precision, checkpoint, routing rule or context reduction was
   introduced here.
5. **Measurement tooling needed repair.** The old benchmark retained visible
   content hashes but not complete reasoning/output identities or exact requests.
   It also defaulted to older launch settings. The new runner records exact
   requests/responses, failed in-flight requests, source/DLL/tokenizer/store
   metadata fingerprints and internal Neural token hashes; it can replay the
   first session for all later variants. The reviewer separates missing evidence
   from a mismatch and calculates throughput as timed tokens / timed seconds.
   GGUF identity is explicitly metadata-only; no 60 GB hash pass was performed.
   Identical API requests across engines still allow different chat templates.

Read-only review found no concrete corruption path in ring rollback or the
defensive Harmony/tool-call canonicalization. Existing prompt-lookup speculation
was already analyzed: the repository estimates 0.88× on rewrites and 0.79× on new
code with its verifier costs; it was not built. It is not a new discovery here.
Persistent CPU worker teams and vector activation remain low-single-digit
headroom hypotheses near the measured memory floor, not validated improvements.

## Final validation status

MEASURED: the final selected pytest-compatible suite passed **102 tests**, with
23 existing return-value warnings. The command names the ring, splice,
scale-layout, doorbell-mock, grouped-plan and benchmark-review files explicitly.
This does not silently convert the standalone GPU/CPU scripts into pytest tests.
Plain collection of the entire directory fails because some standalone GPU
scripts parse their own CLI at import; no whole-directory pass is claimed.
The new standalone GPU and full-model checks above were executed separately.
The final run is recorded in `selected_tests_delivery.log`. The interactive
review artifact also passes TypeScript checking.

## Reproduction

Run from this isolated checkout with the project venv. Keep performance runs
serial and use new result filenames; the original raw evidence should remain
intact. The benchmark source is the fixed `../neuralserver-base` checkout so
edits to the candidate do not silently change its code prompts.

```powershell
$py = 'F:\AI\Neural\.venv\Scripts\python.exe'
& $py tools/review_warm_prefill.py benchmarks/repeat_selected.json --schedule off,g3,g3,off --group-rows 1024 --group-min-tokens 1024 --group-max-tokens 4096 --group-trim-cache 1 --short-sweep --ctx-root F:\AI\Neural\experiments\neuralserver-base
& $py tools/review_warm_prefill.py benchmarks/repeat_short.json --schedule off,g3,g3,off --group-rows 1024 --group-min-tokens 0 --group-max-tokens 4096 --short-sweep --ctx-root F:\AI\Neural\experiments\neuralserver-base
& $py tools/review_warm_prefill.py benchmarks/repeat_unbounded.json --schedule off,g1,g3,g3,g1,off --group-rows 1024 --group-max-tokens 0 --ctx-root F:\AI\Neural\experiments\neuralserver-base
& $py tools/grouped_gemm_ab.py --real-shape --time --output benchmarks/repeat_grouped_kernel.json
& $py tools/masked_gemv_ab.py --real-shape --time --output benchmarks/repeat_masked_kernel.json
& $py -m pytest tests/test_kv_ring.py tests/test_splice.py tests/test_scale_layout.py tests/test_doorbell_mock.py tests/test_grouped_gemm_plan.py tests/test_review_bench.py -q
```

The warm runner pins the packed store and latest launcher settings. Its explicit
minimum `0` intentionally tests the tiny prompts too; the selected serving
configuration uses minimum `1024`. Raw files record the effective settings and
source fingerprints. The original unbounded results predate the range guard.

For the model-level gates, use the same server settings, `NEURAL_PREFILL_GROUP_ROWS=1024`,
and `--review-prefill-check <new.json> --review-prefill-groups 3
--review-prefill-lengths 128,512,1024,3000,5000,13000`. Set the minimum/maximum to
`1024/4096` for the selected dispatch proof, or maximum `0` for the unbounded
kernel proof. The decode gate uses `--refresh-every 0 --review-decode-check
<new.json>`; its default lengths are 128/3,000/13,000 with 64 decode steps.
These correctness timings include different cache/history conditions and must
not replace the serving benchmarks.

`run_profile_abba.ps1` and `run_masked_abba.ps1` in the evidence folder preserve
the complete commands for their trials. Use fresh output/log paths when adapting
them; the saved scripts deliberately refuse to overwrite their original JSON.
Raw evidence and rejected experiments are indexed in the
[evidence README](../benchmarks/codex_review_20261001/README.md).
