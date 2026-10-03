# Benchmark results

All numbers MEASURED on the development machine (RTX 3080 Ti 12 GiB, i7-11700K, 64 GB
DDR4-3000, NVMe) with `tools/bench_agent.py`, temperature 0. JSON files in this folder hold
every request's stats.

| file | server version |
|---|---|
| `bench_baseline.json` | first OpenAI-compatible server (general VRAM hot set, no split attention, GPU-only prompt experts) |
| `bench_v2.json` | + split-K decode attention, CPU multi-token prompt experts (threshold 8), tool-call splicing, idle re-processing, code hot set |
| `bench_v2t12.json` | v2 with 12 CPU threads (no gain; not adopted) |
| `bench_v3.json` | + review fixes, threshold 16, side-request protection, +16 VRAM slots (current) |
| `bench_v3g.json` | v3 with GOMP_SPINCOUNT set (no clear gain; not adopted) |
| `tune_prefill*.json` | CPU-route threshold sweep (0-128); 16 best for agent steps |

## Headline (baseline -> current)

| probe | baseline | current |
|---|---|---|
| code generation, ~0.1k context | 12.6 tok/s | 15.9 tok/s |
| code generation, ~4k context | 11.6 tok/s | 14.9 tok/s |
| code generation, ~9k context | 12.6 tok/s | 14.6 tok/s |
| agent step, +~300 new tokens (cached prefix) | 3.8 s | 1.7 s |
| agent step inside a tool loop | 1.6-4.7 s | 0.4-0.6 s |

v4 (prompt-weighted admission, RAM residency): see *Long coding conversation vs llama.cpp* below.

Whole agent sessions are not directly comparable across runs: routing near-ties and
timing-dependent VRAM residency make trajectories diverge even at temperature 0 (the same
task produced 1,817-6,155 generated tokens). Per generated token, sessions were ~12-27%
faster than the baseline's 71.5 ms/token.

Note: in `bench_v3.json` the `+~50 tok` probe (7.1 s) hit a since-fixed case (a new
conversation's second request after side-request protection); `tests/test_protect.py`
covers it.

## Measured dead ends (kept for the record)

- A code-calibrated VRAM hot set: covers 39% of coding routing in-sample, but only +1-3
  points of hit rate out-of-sample (code routing is spread across the whole model).
- GPU reading experts straight from the page cache (cudaHostRegister + zero-copy GEMV):
  works and is bit-exact, but CPU + GPU together read 28.5 GB/s vs 29.2 GB/s for the CPU
  alone - RAM bandwidth is the wall.
- 12 CPU threads, GOMP_SPINCOUNT: no measurable in-server gain.
- The multi-token CPU kernel used for single-token decode (`tools/decode_kernel_ab.py`):
  bit-identical, 0.85-1.04x - no gain. RAM runs at its rated speed already (XMP on).

## VRAM cache policy for code (`tools/cache_policy_sim.py`, SIMULATED on a MEASURED trace)

`code_trace.json`: 1,398 code-generation tokens recorded with `POST /neural/trace`. Code
touched 4,164 of the 4,608 experts. The server's measured VRAM hit rate on it was 0.35-0.44.

| slots / static core | LRU, admit every miss | Belady (knows the future) |
|---|---|---|
| 430 / 250 (current) | 0.406 | 0.632 |
| 430 / 0 | 0.468 | 0.715 |
| 472 / 250 (+ring KV) | 0.448 | 0.668 |

The oracle headroom looks large, but only while copying experts into VRAM is free. Counting
each admission as a RAM read (capture + upload), CPU expert reads per token are:

| policy (430 slots) | hit | reads/token |
|---|---|---|
| current (measured, ~4.8 admissions/token) | ~0.40 | ~96 |
| static set only, never admit | 0.297 | 101 |
| best realizable rule (admit on >= 3 uses in 64 tokens, no static core) | 0.492 | 168 |
| ORACLE rule (admit only if reused soon, farthest-next-use eviction) | 0.699 | 79 |

Every rule that looks only at the past moves more bytes than today's rate-limited
copy-on-compute. Only the oracle beats it, by ~18%, and that needs knowledge of the future
(predictive paging was already falsified in the Neural project). Verdict: no realizable
policy gain. Config lever only: more free VRAM means more slots (+42 slots ~ +4 points hit).

## Speculative decoding (`tools/lookup_spec_estimate.py`)

Verifying T drafted tokens at once reads the union of their experts: 1.73x the bytes of one
token for T=2, 2.92x for T=4, 4.72x for T=8 (from the trace). Even with every draft
accepted, T=8 yields only 1.69x tokens per byte. Processing n new tokens through the prompt
path costs (MEASURED) 236 ms for 5, 323 ms for 8, 472 ms for 15 tokens, against ~50-63 ms
per decoded token. Prompt-lookup drafting replayed on real outputs (estimated speedup):

| task | acceptance | est. speedup |
|---|---|---|
| full-file rewrite (Continue apply-style), 3-gram, <= 8 drafts | 0.61 | 0.88x |
| new code, 3-gram, <= 4 drafts | 0.34 | 0.79x |

A dedicated graphed verify path could reach at most ~1.1x on rewrites (CALCULATED from the
expert-union bound), and nothing on new code. Verdict: not built.

## Decode levers (branch `claude/decode-levers`, MEASURED 2026-09-29)

28 interleaved sessions on an identical 13k-token conversation: the branch as shipped against a clean
v4 checkout, then each lever against the branch as shipped; raw sessions in `levers_2026-09-29/`. Verdict table and method: `docs/LEVERS.md` (top).
Headline (second pass, checkpoint fixed so the ring is byte-exact across turns): ring 18.06, ring +
`--kernel-fuse 1` **19.22** tok/s on follow-ups (bit-exact; both now launcher defaults) vs v4 17.6-17.8;
DMA admission from a 40 GiB arena 19.50 (device-class); `--zc-misses 1` 18.29 with first answers -11%;
`--kernel-affinity 2` 16.4 (-8%). The first pass's ring figure (19.0, +7%) was measured before the
checkpoint fix, on diverged follow-up text. Against llama.cpp with the new defaults (same harness as
the section below): first answer **1.16x** (was 1.03x), follow-ups **1.34x** (was 1.24x), rewrite at
equal span **1.25x** (was 1.06x; same time-weighted method); prompt processing 6.8x / 6.3x on this run,
but the unchanged v4 build re-run the same night also read prompts 2x faster than in its published sessions
(6.5x / 6.0x), so that change is the PC's, not the version's (`docs/LEVERS.md`).

Third pass (night of Sep 29/30, `levers_2026-09-29/step7*`, `step8*`): 4-bit packed scale rows in the
expert store (bit-identical, -2.88% bytes) are +3.8% on follow-ups (20.63 vs 19.88 tok/s) and +15% on
first answers (19.4 vs 16.8); fused kernel + DMA admission +3% (device-class, 40 GiB pinned); `--wait
poll`, the host-residual patch and the "doorbell" graph pre-launch measured no gain (details and the
doorbell's failure mode in `docs/LEVERS.md`). Against llama.cpp with the packed store (Sep 30, same harness,
`levers_2026-09-29/step9*`): first answer **1.36x** (was 1.16x), follow-ups **1.38x** (1.34x), rewrite at equal
span **1.42x** (1.25x); prompt reading 6.5x / 6.4x (13k / 3k), unchanged by the store.

Fourth pass (Sep 30 morning, `levers_2026-09-29/step10*`): the fused CPU kernel's OpenMP scheduling
(`schedule(dynamic)` chunks + one yield-spin barrier instead of static loops with sleeping barriers) is
**+10.9% on follow-ups in-server (19.91 -> 22.07 tok/s, CPU phase 38.7 -> 34.1 ms, bit-identical answers in
all six mirrored sessions and a frozen-residency pair)**; dynamic chunks alone +5.7%. Standalone the kernel
goes from 33-34 to 37-38 GB/s, the private-memory wall, and stays there with 4-6 busy background threads.
Shipped in the tracked DLLs (`docs/LEVERS.md`, fourth pass). Against llama.cpp with that build (`step11*`, same
harness, same morning): first answer **1.49x** (was 1.36x), follow-ups **1.54x** (1.38x); prompt reading
6.4x / 6.4x, unchanged by the kernel; the rewrite could not be measured cleanly that morning (hard page-fault
stall windows on experts pushed out of the page cache, on every kernel; steady-state rewrite windows +7-8%), see `docs/LEVERS.md`.

Fifth pass (Sep 30 midday, `levers_2026-09-29/step14*`, `step16*`, `step17*`): prompt processing in LAYER order
(`--prefill-order layer`, now the launcher default) copies each non-resident expert into the GPU once per layer instead
of once per 4096-token block: **13k-token prompt 15.0 -> 10.7 s warm (-29%), 15.1 -> 11.3 s inside the conversation**,
3k prompt unchanged, decode unchanged, bit-identical (in-process check: logits and all K/V equal; a larger --prefill-chunk
is faster too but changes the outputs and stays at 4096). Details in `docs/LEVERS.md` (fifth pass).

Sixth pass (Sep 30 afternoon, `levers_2026-09-29/step19*`-`step26*`): prefill GEMM `BLOCK_M` 16 -> 64 is bit-identical
and takes the **13k prompt 10.8 -> 7.7 s**; a staging producer thread and a cached Triton launch path ship as defaults
(exact; neutral-to-useful); a grouped epilogue is exact but measured neutral (the prompt is now balanced host/GPU at
~7.2 s); `--refresh-m 16` halves admission captures at an unchanged hit rate, **follow-ups +3.1% median / +5.1% mean**
(launcher default); the robust doorbell is exact (incl. injected timeouts) but -1.6% now that the kernel barrier removed
the slack; the multi-token CPU kernel's scheduling fix is hidden behind the GPU (not shipped); a cold-expert prefetch
knob ships off by default. Details in `docs/LEVERS.md` (sixth pass). Against llama.cpp with that build (`step27*`):
reading the 13k prompt **13.9x** (was 9.3x), first answer **1.96x**, follow-ups **1.57x**. Against llama.cpp
with that build (`step18*`): reading the 13k prompt **9.3x** (was 6.4x), first answer **1.64x**, follow-ups **1.54x**. VRAM slot value: 0.7 ms/token per hit point.

## Long coding conversation vs llama.cpp (`tools/bench_vs_llama.py`, MEASURED)

**Workload.** A 13,000-token code context plus a question, then 3 follow-up questions in
the same conversation. Each turn is 512 tokens, temperature 0, reasoning effort low.
- Each session starts a fresh server; Neural and `llama-server` b10361 (`-ngl 99 -ncmoe 31
  -t 8 -c 16384 -fa on`, the decode-tuned MoE offload) take turns.
- Each JSON records its session order (`config.schedule` or ABBA rounds). The order matters:
  a Neural session that follows llama.cpp starts with llama's weights, not the expert store,
  in the page cache.
- Decode speed is each server's own timing.
- `--workload rewrite` sends one file (3,000 tokens) and asks for the whole file back with
  type hints, so the answer continues the prompt, like an editor's apply step.

**v3 server** (`vs_llama_v3.json`): first prompt 47.3 vs 153.8 s, follow-ups 15.8 vs 14.2 tok/s
(**1.11x**), first answer 10.9 vs 13.1 tok/s (**0.83x**). The first answer was behind for two
reasons, fixed as follows.

**1. Admission chased the prompt's experts.** The prompt's routing counts entered the
admission counts at full weight. After a 13k-token prompt they dominated the whole first
answer, and the experts the answer actually used were evicted first:
- VRAM hit was 0.16-0.18 on the first answer, against 0.31-0.35 on follow-ups.
- An offline replay of the policy (`tools/admission_replay.py` on
  `benchmarks/admission_trace_13k.json`, SIMULATED) reproduces this: 0.162 / 0.331, against
  0.157 / 0.329 measured.
- The same replay shows that reading code and writing about it use different experts.
  Bulk-loading the prompt's top experts into VRAM makes things worse (0.145-0.156).

Fix `--prefill-weight 16`: the prompt counts as at most 16 generated tokens. Counts are
rounded down per expert, so after a long prompt most prompt counts drop to zero. Prompts of
16 tokens or fewer are unchanged. Replayed hit: 0.305 / 0.356. MEASURED
(`pw_conversation.json` 3 rounds, `pw_rewrite.json`):

| | old | new |
|---|---|---|
| VRAM hit, first answer | 0.177 | 0.311 |
| VRAM hit, follow-ups | 0.322 | 0.351 |
| VRAM hit, rewrite (where the old weight was expected to do best) | 0.31 | 0.378 |

**2. The expert store sits at the edge of free RAM.** The CPU-side store is 51-54 GiB,
against about 54 GiB available next to ~10 GiB of other programs. Evidence (in the
`neural` stats of the JSONs below):
- First answers took ~1,000-1,500 page faults per token; follow-ups took 10-350.
- Timing per 64 tokens (`decode_tok_s_per_64`) showed the first 128 tokens at 5-13 tok/s,
  then 15-17.
- A long prompt makes Windows trim the expert pages, and the answer faults them back one
  4 KB page at a time.
- Cold 13 MB reads measured about 1 GB/s through faults with 8 threads. Re-mapping trimmed
  pages costs about as much.

Fixes:
- **Release RAM copies.** The RAM copies of VRAM-resident experts are released at startup
  (`--release-resident 1`). That gives 2.2 GiB for the CPU's experts, and prompt disk reads
  fell from ~6.5 to ~2 GB (`release_conversation.json`, system-wide counter).
- **Better warm-up.** The startup warm-up ranks by the hot-set counts plus the general
  calibration counts (about 0.8 coding + 1.2 general), so answers' plain-language reasoning is
  covered. It touches the most likely experts last, which makes them the last pages Windows
  trims, and keeps a 1.5 GiB margin. Prompt disk reads fell to 0.1-0.9 GB
  (`warm_conversation.json`).
- **Soft minimum working set** (`--ws-min 1`). Windows trims other processes' idle pages before
  the server's expert pages. First answers were 12.8 / 15.6 vs 11.4 / 11.9 tok/s without it
  (n=2, `wsmin_conversation.json`).

Measured dead ends:
- **Re-warming the warm set in bulk after long prompts** (`rewarm_conversation.json`): 10.7 /
  12.3 vs 13.3 / 12.6 tok/s without.
- **Prefetching each cold expert in one large read before the CPU computes it**
  (`cold_conversation.json`): 15.5 / 12.4 vs 14.4 / 16.2 tok/s without. Its residency check
  cost 1.5-2.7 ms/token in the server.

**Prompt processing, both servers warm** (`tools/bench_prompt_warm.py`, `prompt_warm.json`).
A server's first prompt is not a fair comparison: llama-server (mmap) loads its weights from
disk inside it, while Neural warms RAM at startup. Each session therefore sends a discarded
warm-up prompt first, then prompts with no shared prefix:

| prompt | Neural | llama.cpp | ratio |
|---|---|---|---|
| 13,000 tokens of code | 29.8 s (29.8 / 29.8) | 93.2 s (94.8 / 91.6) | **3.1x faster** |
| 3,000 tokens (one file) | 8.1 s (8.0 / 8.2) | 24.1 s (24.5 / 23.8) | **3.0x faster** |

The first prompt after a fresh start, which includes llama.cpp loading its weights, was 41.7
vs 137.8 s (3.3x) in the final conversation run.

**Final decode (current defaults).** In `final_conversation.json` the shipped configuration is
the kind `relall`. The kind `neural` there is `--release-resident 0`, which was the default
at the time.

| probe | Neural | llama.cpp | ratio |
|---|---|---|---|
| first answer after the 13k prompt (final run) | 14.08 tok/s (13.97 / 14.12 / 14.08) | 13.70 (13.71 / 13.70 / 13.58) | **1.03x** |
| first answer, all 5 runs of the shipped configuration (final + `wsmin_conversation.json` `neural`) | median 14.08 (12.79-15.64) | median 13.60 (13.22-13.71) | **1.035x**; per run 0.93-1.18x |
| follow-up answers (9 each, final run) | 18.07 tok/s | 14.52 | **1.24x** |
| rewrite a file just read: writing, equal length (first ~192 tokens; `final_rewrite.json`) | 15.2 tok/s (14.9 / 15.4) | 14.3 (14.2 / 14.4) | **1.06x** |

The first answer is about level: ahead on the median, within run-to-run spread. The `neural`
kind (`--release-resident 0`) gave 12.63 / 17.04 / 14.38 on the first answer and 17.74 tok/s
on follow-ups. It was less steady, so it is not the default.

On the rewrite, llama.cpp stopped after 196 tokens while Neural wrote to its 1,536-token cap.
The equal-length figure above compares the same stretch of answer; over its whole answer,
Neural averaged 18.9 tok/s.
