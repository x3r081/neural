# Experimental decode levers (branch `claude/decode-levers`)

Everything here is **off by default**: the launcher flags reproduce server v4 exactly. Each lever
is a flag, has a way to verify it does not change outputs, and reports its own numbers in the
`neural` block of every response (and `logs/requests.jsonl`). The reasoning and the offline
evidence live in the research repo: `Neural/docs/NEURALSERVER_DECODE_LEVERS.md`.

## Measured on the development machine, 2026-09-29

All at commit 1b44ccb plus the thread-pinning fix; the `--admit-gpu` rows below measured the SM zero-copy
admission path that 2146ffb has since replaced with a copy-engine DMA path (`arena.row_view`), so they
characterise the admission economics, not the current code path. 28 sessions (112 turns) of
`tools\bench_vs_llama.py --workload conversation` (13k-token prompt + 3
follow-ups, 512 tokens each, temperature 0), one server per session, interleaved: the branch as shipped
against a clean v4 checkout (step 1, ~46 GiB free), then every lever against the branch as shipped on an
identical prompt (~54-56 GiB free). `--server-roots`, `--ctx-root` and `content_sha1` were added to the
harness for this. Raw sessions: `benchmarks\levers_2026-09-29\` (answer text stripped, SHA-1 kept; two
more sessions were killed or refused, see its README). Medians; n = follow-up turns (3 per session).
Even v4 does not reproduce itself byte for byte at temperature 0: one of its three sessions diverged on
turn 3, and one default-flag branch session diverged on all turns (VRAM residency is timing-dependent,
as RESULTS.md notes), so answer identity below means "identical in the sessions where residency agreed".

| lever | claimed here before | MEASURED follow-up tok/s (n) | first answer | verdict |
|---|---|---|---|---|
| branch as shipped vs v4 | reproduces v4 exactly | 17.58 vs 17.60 (9 / 9) | 14.6 vs 14.2 | confirmed: answers byte-identical in 5 of 6 sessions (the 6th is v4 diverging from v4) |
| `--kv-ring 256 --scratch 6 --pool 6.05` | +4-5 hit points, ~3 ms/token | **19.0 (15), +7%** vs the contemporaneous branch (17.79), +8% vs v4; hit 0.33 -> 0.38; turn 2 19.7-20.0 | 13.5-16.0 (median 15.3) | **ship.** The ring path is byte-exact: with `--kv-ring 256` alone (pool 5.5, one session) the first answer is identical to v4's. At pool 6.05 the first answer differs because 45 more slots change which experts the GPU computes (device-class). Follow-ups differ in both cases: the checkpoint is one token early (`cached_tokens` 13107 vs 13108) and the re-prefilled token's K/V differs at bf16 |
| `--kernel-affinity 2` | +7% isolated | 16.4 (3), **-8%**; first answer -19 to -24% | 11.8 | **loss.** The shipped pin was a silent no-op (fixed in this commit, sessions ran the same build as `gptoss_cpu_cap2_tls.dll`); the working pin also lands on the Python thread that drives the GPU and HTTP. The pin covers only the decode kernel (`gptoss_experts_cap`); the prefill kernel in `gptoss_cpu_multi.c` never calls it |
| `--kernel-affinity 2 --kernel-pair 1 --kernel-prefetch 2048` | +10% isolated | 16.4 (3) | 12.0 | loss |
| `--pool 4.5` (slope point) | - | 16.2 (6) at hit 0.26 | 14.7 | **0.7 ms/token per VRAM-hit point** (the ring gives the same: +5 points = -3.8 ms). `--pool 4.0` refuses (`STATIC_N <= N_USABLE - 64`) |
| ring + `--arena pinned` (default budget) | no page trimming | 3,640 faults/token, 0.4 GiB free after the prompt | 13.1 (1) | killed after one session: 45.9 GiB pinned starves the rest |
| ring + `--arena pinned --arena-max-gib 40` | - | 18.7 (6) | 13.6 | -1% vs ring; the 874-expert tail faults through ~12 GiB of page cache |
| + `--admit-gpu 1 --admit-rule server` | up to +15-18% | **19.8 (6), +4% vs ring**; turn 2 20.2-20.7 | 13.5 / 15.4 (within the ring's own spread) | self-test passed (slot bytes == store row; zero-copy == pool GEMV bit-equal; CPU kernel 3.6e-3); 3.9 GPU admissions/token, 0 captures; device-class |
| + `--admit-gpu 1 --admit-rule window --admit-k 2 --admit-window 4 --victim lru --static 0` | RAM reads 96 -> 73 | 14.9 (6); **72 CPU experts/token, as simulated** | 11.8 | **loss:** `setup_arena` excludes VRAM-resident experts and GPU admission needs an arena row, so the residents evicted under `--static 0` are read from mmap: 7.7k and 9.1k faults/token on the two first answers, 1.2-3.3k on follow-ups |
| same, residents ranked into the arena | - | 17.2 (3); 65 CPU experts/token | 15.5, prompt 27.8 s | tail grows to 1,359 experts; `admission_ms` 4.3; reverted |
| server rule, residents ranked in | - | 16.6 (3) | 14.7, prompt 40.9 s | reverted |

Reading: the arena does not fit 64 GB next to the store and the server's ~13 GiB private commit, whichever
part of the store it leaves out. The window rule's byte cut is real; CALCULATED with the whole store pinned
(about 80 GB usable RAM or more) it is ~26 tok/s (65 x 13.2 MB / 29 GB/s + ~8 ms). Step 1's DRAM tool is
1 ms-quantized (`omp_get_wtime`): it reported 52-55 GB/s on a 48 GB/s bus; a `perf_counter` cross-check gives
~40 GB/s, private pages == file view, software prefetch +2-10%. Not run: `--wait poll`, `--near-miss`, the
demand probe, DDR4 retune.


### Second pass, same evening: two fixes and the 2146ffb levers (MEASURED, 78f73df + fixes)

Two bugs found by review and fixed first: (1) the ring checkpoint was taken at the end of the prompt,
before the first generated token was decoded, so every follow-up re-processed that token through the
prefill path (`cached_tokens` 13107 instead of 13108) and its bf16 K/V differed; a second checkpoint at
P+1 after the first decode step (superseding the first) makes a `--kv-ring 256` session's four answers
byte-identical to v4's. (2) `arena.py` (2146ffb) crashed on any arena bigger than one chunk
(`list.index` on dicts holding tensors); with that fixed the DMA admission path and `--zc-misses`
ran for the first time. Sessions in `benchmarks\levers_2026-09-29\step4*`: 10 interleaved, n = 2
each, same 13k conversation, the branch's own answers byte-identical across the ring and fuse sessions.

| config (all with `--kv-ring 256 --scratch 6 --pool 6.05`) | follow-up tok/s (n=6) | first answer (n=2) | verdict |
|---|---|---|---|
| ring alone, checkpoint fixed (control) | 18.06 | 14.33 | with identical text the ring is worth +2-3% over v4's 17.6-17.8, not the +7% measured above: the pre-fix ring sessions were generating different (diverged) follow-up text that happened to run faster |
| **+ `--kernel-fuse 1`** | **19.22 (+6.4%)** | **15.23 (+6%)** | **ship (now the launcher default with the ring): bit-exact (answers identical to the control in 4/4 sessions), CPU path 40 -> 38.7 ms at 30 GB/s** |
| + `--arena pinned --arena-max-gib 40 --admit-gpu 1 --admit-rule server` (DMA path) | 19.50 (+8%) | 14.04 (-2%) | works (self-test: arena row == store row, graph output bit-equal to the eager pool path; 4.0 GPU admissions/token, 0 captures); +1.5% over fuse for 40 GiB of pinned RAM and device-class numerics |
| + arena40 `--zc-misses 1` | 18.29 (+1%) | 12.73 (-11%) | 25.6 misses/token move to the GPU pipe and `cpu_experts_per_tok` falls to 69.5, but the CPU kernel drops to 21 GB/s while the copy engine shares the bus (the recorded two-pipe kill, now from the arena) |
| + arena40, DMA admission + `--zc-misses 1` | 18.74 | 14.30 | no better than either alone |


### Against llama.cpp with the new defaults (MEASURED 2026-09-29, `step5*`)

`tools\bench_vs_llama.py` and `tools\bench_prompt_warm.py` exactly as for the published v4 figures
(llama.cpp b10361 `-ngl 99 -ncmoe 31 -t 8 -c 16384 -fa on`, fresh server per session, alternating),
Neural = `--kv-ring 256 --scratch 6 --pool 6.05 --kernel-fuse 1` (the launcher defaults now).

| workload | published (v4, 2026-09-24) | now | llama.cpp now (then) |
|---|---|---|---|
| 13k-token coding conversation, first answer (median of 3 fresh servers) | 1.03x (14.08 vs 13.60 tok/s) | **1.16x** (15.6 vs 13.5; per round 1.27 / 1.16 / 1.02) | 13.5 (13.6) |
| same, follow-up answers (9 turns) | 1.24x (18.07 vs 14.52) | **1.34x** (19.5 vs 14.6; 16.4-20.6) | 14.6 (14.5) |
| rewriting a file just read, equal token span (llama stops at 171 tokens; Neural's first three 64-token windows, time-weighted as for the published figure) | 1.06x (15.2 vs 14.3) | **1.25x** (17.9 vs 14.3; the plain mean of the three windows gives 18.3 = 1.27x, not the published method; Neural's whole 1,536-token answer 20.9) | 14.3 (14.3) |
| reading a 13k-token prompt, both warm | 3.1x (29.8 s vs 93.2 s) | 6.8x (14.8 s vs 101.7 s) | 101.7 s (93.2 s) |
| reading a 3k-token prompt, both warm | 3.0x (8.1 s vs 24.1 s) | 6.3x (3.9 s vs 24.7 s) | 24.7 s (24.1 s) |

llama.cpp reproduced its published decode and prompt numbers within a few percent, so the decode
ratios are the levers. The prompt ratios are not: nothing in these commits touches prefill, and the
clean v4 checkout also processed the 13k prompt in ~16 s tonight (35-50 s in the published
sessions at the same page-fault counts). Re-run of the unchanged Sep 24 build through
`tools\bench_prompt_warm.py` the same night: 15.6 s (13k) and 4.1 s (3k), i.e. 6.5x and 6.0x against that
night's llama.cpp (`levers_2026-09-29/step6_prompt_warm_v4.json`), so the doubling comes from the PC, not from
these commits. Cause not established; the site quotes prompt reading as 3-7x with both dates shown.
For the conversation benchmark the first published server (Sep 23, `vs_llama_v3.json`) measured the same way:
first answer 0.83x (2 runs: 0.67x and 0.99x), follow-ups 1.11x, so follow-ups ran 1.11x -> 1.24x -> 1.34x.


### Third pass (night of 2026-09-29/30): the remaining software levers, each with a verdict

Same harness and conversation; controls interleaved in the same run (`benchmarks\levers_2026-09-29\step7*`,
`step8*`). Sessions that crossed midnight are compared only within their date group, because the chat
template writes the date into the system message (one token changes, so a conversation spanning midnight
re-processes its prompt once: a real-world cost, not a bug of any lever).

| lever | follow-up tok/s vs control (n) | first answer | verdict |
|---|---|---|---|
| `--wait poll` (busy-poll the router event) | 19.0 / 18.7 vs 18.5 / 19.6 (6) | - | **no gain**; `gpu_wait_ms_per_tok` unchanged at 5.6 |
| fused kernel + DMA admission (`--arena pinned --arena-max-gib 40 --admit-gpu 1 --admit-rule server`) | 19.1 / 20.3 vs 18.5 / 19.6 (6); with `--wait poll` 19.8 / 19.7 | ~level | **+3%**, device-class, 40 GiB pinned; the two levers stack |
| host-residual patch (one `.tolist()` pack parse, pre-built table views, reusable copy event, hoisted invariants) | 18.4 / 17.7 (6) | - | **no measurable gain**: `admission_ms_per_tok` fell 0.2 ms as predicted (1.75 -> 1.55) but both sessions ran under agent compile activity; expected +0.6% is below the run-to-run noise; not shipped |
| "doorbell" (launch layer L+1's graph before the CPU kernel; a spin node in the graph waits on a pinned sequence flag the host rings) | one clean session 20.7 (3) at `gpu_wait` 4.4 ms; the other aborted | - | **NO_GAIN by its microbenchmark, not shippable** (corrected in the fourth pass below: the in-server timers say +2-4%, n=1): its own reorder microbenchmark is -0.25 ms/token (launch latency here is 3-9 us/layer, the spin node costs 7 us); bit-exact (identity run byte-identical); but a >500 ms CPU stall in the first-answer fault storm made the spin give up and the request failed (HTTP 500); a token re-run recovery was designed and mock-tested. Code kept on branch `work-host` as the record |
| **4-bit packed scale rows (`mxfp4_g32_ps4` store, 12,839,040-B slots, -2.88% bytes)** | **20.63 vs 19.88 (6), +3.8% at equal slot count**; new DLLs on the raw store 20.06 (neutral) | **19.3 / 19.5 vs 16.8 / 16.7 (+15%)** | **ship.** Bit-identical everywhere: C decode kernel 1104 checks, prefill kernel 281, Triton pool kernels 134 on the GPU incl. real slots, converter full verify (unpack(dst) == src, SHA-256), in-server answers byte-identical with residency frozen. Kernel-side saving equals the byte saving (0.971-0.978 of raw at fuse=1, ideal 0.9712). +14 slots at `--pool 6.05` add nothing measurable (20.61). Prompt time unchanged (14.8-15.2 s) |

Not a software item: **DDR4 3200 Gear 1 (BIOS)**, CALCULATED +4-6% on the CPU-read part. Owner procedure: read the
current timings (CPU-Z / Thaiphoon; 4x Corsair CM4X16GC3000C16K4D, 2 DIMMs per channel, XMP at 3001 MT/s
MEASURED), set DRAM 3200 with Gear 1 in BIOS keeping the XMP timings, run an overnight memory test
(TestMem5/y-cruncher) and `python tools\pack_store.py --verify-only --verify-mode sha` on the store (silent bit
flips would break bit-exactness), then one interleaved A/B of `cpu_ms_per_tok`. Keep only if raw read
bandwidth (`tools\measure_host_dram.py`) rises >= 5% and the tests are clean. Not attempted here.

Where the token stands after all of it (MEASURED, follow-ups, packed store, `--kernel-fuse 1 --kv-ring 256`):
**20.6 tok/s = 48.5 ms = 37.2 ms CPU expert reads (30 GB/s) + 5.4 ms GPU wait + ~6 ms host.** The remaining
software-addressable part is the ~6 ms host residual (every cheap piece of it was tried tonight); the rest is
the RAM bus. Hardware (VRAM slots, RAM capacity for the window rule, DDR5) is where the next 30%+ lives.


### Against llama.cpp with the packed store (MEASURED 2026-09-30, `step9*`)

Same harness and settings as the Sep 29 table (llama.cpp b10361 `-ngl 99 -ncmoe 31 -t 8 -c 16384 -fa on`, fresh
server per session, alternating: conversation llama/neural x3, rewrite llama/neural/neural/llama, then
`tools\bench_prompt_warm.py`), Neural from the merged `main` tree (15dd3e7) with the launcher defaults and the packed
store: `--kv-ring 256 --scratch 6 --pool 6.05 --kernel-fuse 1 --store-dir G:\NeuralStores\gptoss120b_ps4`
(the tracked `gptoss_cpu_*.dll` rebuilt from the new source, mode 0 default). Ratios by the published method.

| workload | Sep 24 (v4) | Sep 29 (ring + fuse) | **Sep 30 (+ packed store)** | llama.cpp Sep 30 (Sep 29) |
|---|---|---|---|---|
| 13k-token coding conversation, first answer (median of 3 fresh servers) | 1.03x (14.08 vs 13.60) | 1.16x (15.6 vs 13.5) | **1.36x** (19.17 vs 14.06; per round 1.37 / 1.37 / 1.36) | 14.06 (13.5) |
| same, follow-up answers (9 turns) | 1.24x (18.07 vs 14.52) | 1.34x (19.5 vs 14.6) | **1.38x** (20.93 vs 15.15; Neural 20.0-21.6; per round 1.39 / 1.38 / 1.37) | 15.15 (14.6) |
| rewriting a file just read, equal span (Neural's first N tokens, time-weighted over its 64-token windows, vs llama.cpp's whole N-token answer: N = 196 / 171 / 291; the Sep 24 and Sep 29 columns used Neural's first 192 tokens) | 1.06x (15.2 vs 14.3) | 1.25x (17.9 vs 14.3) | **1.42x** (20.71 vs 14.57; per run 1.44 / 1.41; Neural's whole 1,536-token answer 21.9 tok/s) | 14.57 (14.3) |
| reading a 13k-token prompt, both warm | 3.1x (29.8 s vs 93.2 s) | 6.8x (14.8 s vs 101.7 s) | 6.5x (14.3 s vs 93.5 s) | 93.5 s (101.7 s) |
| reading a 3k-token prompt, both warm | 3.0x (8.1 s vs 24.1 s) | 6.3x (3.9 s vs 24.7 s) | 6.4x (3.8 s vs 24.2 s) | 24.2 s (24.7 s) |

Every Neural session gave the same answer at every turn (one `content_sha1` per turn across the three conversation sessions and across both rewrites; residency identical: hit 0.358 / 0.389 / 0.390 / 0.381), and llama.cpp likewise. The answers differ from Sep 29's because the chat template writes the date into the system message; the packed store's bit-identity to the raw store was established the same night with residency frozen (`step8id_*`). llama.cpp ran ~4% faster tonight than on Sep 29 (first 14.06 vs 13.46, follow-ups 15.15 vs 14.58 tok/s), so the ratios are contemporaneous, not inherited. Neural follow-up token: 36.4-38.0 ms CPU expert reads + 5.0-5.9 ms GPU wait. The rewrite ratio over Neural's first 192 tokens (the Sep 24 / Sep 29 span) is 1.43x (20.86 vs 14.57); the table uses llama.cpp's whole 291-token answer as the span, i.e. the same stretch of text.


### Fourth pass (2026-09-30, morning): the CPU kernel's scheduling, found by a headroom audit

After the Sep 30 comparison the question "is there any room left?" was put to an audit (three candidates read against
the code and measured standalone, two refuters each). Verdicts: **(1) kernel scheduling: REAL, shipped (below)**;
(2) a whole-token native loop: refuted as the doorbell under another name (its in-server timers in `step8_05_bellnf`
do show -2.3 ms/token, +4.1% n=1, cross-tree and confounded with the host patch; the frozen-residency pair `step8id`
gives +2.4%; so the doorbell row above is better labelled "+2-4% MEASURED, n=1, unshippable" than NO_GAIN, and the
mechanism it addresses is now the only host-side item left); (3) first-answer page faults: not a RAM-capacity problem
but compulsory first touches of the ~249 adaptive-slot experts released at startup (`release_rows`), worth at most
+1-2% on first answers via `--release-resident 0`, which was already measured and rejected on Sep 24.

**The kernel gap.** The fused decode kernel achieved 30-33 GB/s in-server against a 38.6 GB/s private-memory ceiling
(MEASURED, `tools\measure_host_dram.py`, 8 threads). The audit decomposed it per thread with rdtsc: compute is ~36%
duty (not the limiter), pages/TLB and in-server contention are not measurable, and the loss is the OpenMP structure:
`schedule(static)` leaves one slow thread streaming alone at the end of each phase (13-40% max/min spread), and the
two libgomp barriers plus fork/join sleep and wake (~40 us each; `GOMP_SPINCOUNT` from `server.py` is inert with this
libgomp). Fix: `schedule(dynamic, 64)` / `schedule(dynamic, 32)` with `nowait`, joined by one spin barrier that pauses
for ~50 us and then yields (`SwitchToThread`). Every output element is still computed by exactly one thread with the
same instructions, so outputs are bit-identical; the yield matters because the same barrier without it collapsed
2.2-2.8x with 4-6 busy background threads (MEASURED by the refuters).

| build (all bit-identical) | standalone kernel, ms per token-equivalent (idle / 4 busy / 6 busy) | in-server follow-ups, 6 turns, mirrored (tok/s; CPU phase; gpu_wait) |
|---|---|---|
| tracked (static, previous) | 34.9 / 34.1 / 33.7 | 19.91 (18.7-20.8); 38.7 ms; 5.29 |
| dynamic chunks only | 32.9 / 32.3 / 31.9 | 21.03 (+5.7%); 36.2 ms; 5.02 |
| **dynamic + yield-spin barrier (shipped)** | **30.5 / 30.3 / 29.8** (37-38 GB/s = the private-memory wall) | **22.07 (21.7-22.6), +10.9% median / +11.5% mean; 34.1 ms (-12%); 4.82** |

Identity: `tools\kernel_tuning_ab.py --ref <previous DLL>` (16 cases x 9 settings, raw layout), 1,320 standalone
comparisons on the packed layout including under load, `tools\scale_pack_multi_ab.py --ref` for the prefill DLL
(281 checks; it `#include`s this file), and in-server: a frozen-residency pair (`step10id_*`, `--refresh-every 0`)
and all six performance sessions (`step10_*`) gave one `content_sha1` per turn, the same hashes as the Sep 30
comparison. The performance leg ran with ~3 GiB less free RAM than that comparison (other programs), so its absolute
numbers are ~5% below the Sep 30 row; the ratio is contemporaneous. Both tracked DLLs are rebuilt from the changed
source (`tools\build_kernels.bat`); `--kernel-fuse 0` paths are unchanged (still static).

Where the token stands now (MEASURED, follow-ups, that morning's environment): **22.1 tok/s = 45.3 ms = 34.1 ms CPU
expert reads (~36 GB/s) + 4.8 ms GPU wait + ~6 ms host.** The CPU phase is within ~1.5 ms of its bandwidth floor
(88 experts x 12.84 MB at 38.6 GB/s = 29.3 ms, plus ~2.4 ms of capture/DMA bus tax); what remains addressable in
software is the host residual, i.e. the doorbell's mechanism made robust.


### Against llama.cpp with the shipped kernel (MEASURED 2026-09-30 morning, `step11*`)

Same legs, schedule and flags as the Sep 30 (packed store) table; Neural = `main` 2798e84 (tracked DLLs rebuilt with the
scheduling change), packed store, launcher defaults. Run right after `step10`, in the same RAM-tight environment
(other programs held ~3 GiB more than during `step9`); llama.cpp was ~3-4% slower than in `step9` for the same reason,
so the ratios are contemporaneous.

| workload | Sep 30 (1): packed store | **Sep 30 (2): + kernel scheduling** | llama.cpp (2) / (1) |
|---|---|---|---|
| 13k-token coding conversation, first answer (median of 3 fresh servers) | 1.36x (19.17 vs 14.06) | **1.49x** (20.26 vs 13.62; per round 1.46 / 1.50 / 1.48) | 13.62 / 14.06 |
| same, follow-up answers (9 turns) | 1.38x (20.93 vs 15.15) | **1.54x** (22.46 vs 14.55; Neural 22.0-23.2; per round 1.53 / 1.55 / 1.50) | 14.55 / 15.15 |
| rewriting a file just read, equal span (llama.cpp's whole 291-token answer) | 1.42x (20.71 vs 14.57) | **inconclusive** (three legs, every pair spoiled by hard-fault stall windows or a llama.cpp outlier; see the note; steady state +7-8%) | 14.4-16.9 / 14.57 |
| reading a 13k-token prompt, both warm | 6.5x (14.3 s vs 93.5 s) | 6.4x (14.7 s vs 94.8 s) | 94.8 s / 93.5 s |
| reading a 3k-token prompt, both warm | 6.4x (3.8 s vs 24.2 s) | 6.4x (3.8 s vs 24.1 s) | 24.1 s / 24.2 s |

Every Neural answer was byte-identical across the three conversation sessions and identical to the Sep 30 (1) hashes (the kernel change is bit-exact; residency was identical too). Neural = 2798e84 for the conversation and prompt legs. The rewrite row could not be measured cleanly that morning, in three attempts, and is reported as inconclusive rather than as a number: `step11rw_*` (2798e84): Neural 15.53 (one stall window inside the 291-token span) / 20.50 vs llama.cpp 14.43 / 11.30 (llama.cpp's own stall); `step11rw2_*` (2798e84, free RAM recovered): 21.33 / 16.35 (stall) vs 14.61 / 14.51, i.e. one clean pair at 1.46x; `step13rw_*` (157882d): 15.82 (stall) / 16.92 (stall) vs 16.73 / 16.93, llama.cpp itself 15% faster than in any other run that day (free RAM rose to 55.8 GiB during the leg). The mirrored kernel A/B `step12_*` shows what the stalls are: a rewrite is not residency-frozen, so admission timing picks one of two greedy paths (answers 42f83167 / 3c280373), and the first ~300 tokens of either path touch released adaptive-slot experts whose standby pages the morning's RAM pressure (other programs, memory compression) and each llama.cpp session's 61 GB mmap had repurposed; those come back as hard NVMe faults, ~5 ms per expert (the headroom audit's first-touch cost in its cold form), and one such burst costs a 64-token window ~5 s. They hit the previous kernel and both new barriers alike (step12: 3c280373 stalled in 2 of 3 sessions, 42f83167 in 0 of 3 there and in 1 of 1 in step13), and the night's rewrites ran the same paths with the pages still in standby and never stalled. Steady-state rewrite windows (after the fault-heavy start) are unambiguous: previous kernel 23.0 / 22.9, shipped 24.5-24.8 tok/s (+7-8%). Follow-up token now: 22.5 tok/s = 44.5 ms = 33.5 ms CPU expert reads (~36 GB/s) + 4.7 ms GPU wait + ~6 ms host.


### Fifth pass (2026-09-30, midday): prompt processing in layer order (MEASURED, `step14*`, `step16*`, `step17*`)

The first four passes were about decode; prompt processing had never been profiled. A second headroom audit found the
lever before the box froze under its concurrent measurements (see the safety note in `benchmarks\levers_2026-09-29\README.md`):
`prefill()` walked the prompt in blocks of `--prefill-chunk` (4096) tokens and ran every block through all 36 layers, so a
non-resident expert that several blocks route to was staged into the GPU once PER BLOCK (8-thread memcpy 0.72 ms + PCIe
0.52 ms per 12.8 MB slot, MEASURED standalone). A 13k-token prompt staged 8,460 (layer, block, expert) triples.

`--prefill-order layer` (server.py `_prefill_layer_major` / `moe_prefill_layer`) runs each LAYER over all blocks: attention
per block in order (same calls, same K/V and ring writes), then the layer's MoE with every staged expert copied once and
its GEMMs run for every block that routes to it. Per block nothing changes: the same sort, the same CPU/GPU split on that
block's own token counts (never merged across blocks), the same GEMM calls on the same rows, the same combine. It is
therefore bit-identical, and the server's self-check proves it in one process: `start_neural_server.bat --prefill-order-check`
prints `PREFILL_ORDER_CHECK logits_equal=True kv_equal=True max_abs_logit_diff=0` on a 13,000-token prompt with the ring
on (all 36 layers' K/V and ring buffers compared whole; 9,194 stagings block vs 3,047 layer; 5,079 CPU expert calls both ways).
The same check shows why a bigger `--prefill-chunk` is NOT the way to get the gain: 4096 vs 16384 differs from position 0
(`max_abs_logit_diff` 19.75; 24.7 even with every expert on the GPU), because the blocking changes which experts take
the CPU kernel and how attention and the GEMMs tile, so chunk size is a quality-affecting knob and stays at 4096.

| 13k-token code prompt, warm (`tools\bench_prompt_warm.py`, one server per run) | 3k prompt | in the 13k conversation (`step17conv`) |
|---|---|---|
| block (0cc8563 launcher): 15.03 / 15.05 s | 3.93 / 4.12 s | 15.11 / 15.13 s |
| **layer (shipped launcher default): 10.67 / 10.68 s (-29%)** | 3.89 / 3.90 s (one block: same path) | **11.47 / 11.23 s (-25%)**, 2,717 staging copies for 8,460 pairs |

Decode is untouched (first answer 21.9 vs 21.9 tok/s, follow-ups 22.7 vs 22.8 in the same mirrored run); the first
answer after the prefill is the same hash in all four sessions. Zero-code alternative measured for the record
(`step14_chunk_pw.json`): `--prefill-chunk 8192` 11.84 / 11.82 s and 16384 11.48 / 13.22 s on the 13k prompt, but +0.3 s on
the 3k prompt and not bit-identical (above). Costs of layer order: every block's residual stream and the layer's sorted
expert inputs/outputs stay in VRAM (~46 KB per prompt token, ~600 MB at 13k; fits next to `--pool 6.05` at `--smax 16384`)
and one pinned buffer set per block for the CPU jobs. Request stats gain `prefill_order`, `prefill_stagings`,
`prefill_stg_pairs`, `prefill_cpu_experts`.

Where the prompt time goes now (13k, CALCULATED from the counts): ~2,700 stagings x ~1.2 ms = 3.3 s of copies, ~5,000 CPU
expert calls (<= 16 tokens each) on the worker thread overlapped with the GPU, the MoE GEMMs (~93 TFLOP of MXFP4 GEMM at a
fraction of the card's peak), attention, and ~150k small kernel launches. The next prompt-side items would be the CPU
kernel's own scheduling (`gptoss_cpu_multi.c` still uses `schedule(static)`) and the launch count per expert; not measured.


### Against llama.cpp with layer-order prompt processing (MEASURED 2026-09-30 midday, `step18*`)

Same legs and flags as the Sep 30 (2) table plus `--prefill-order layer` (the launcher default since c1d993e); conversation
x3 pairs and warm prompts (the rewrite leg is left out: after llama.cpp sessions it measures the file cache, see the Sep 30 (2)
note). Free RAM ~55 GiB throughout (the morning's RAM pressure was gone), which is why the first answers are faster than in
`step11` on both sides of the comparison.

| workload | Sep 30 (2): kernel scheduling | **Sep 30 (3): + layer-order prefill** | llama.cpp (3) / (2) |
|---|---|---|---|
| 13k-token coding conversation, first answer (median of 3 fresh servers) | 1.49x (20.26 vs 13.62) | **1.64x** (21.75 vs 13.25; per round 1.63 / 1.60 / 1.66) | 13.25 / 13.62 |
| same, follow-up answers (9 turns) | 1.54x (22.46 vs 14.55) | **1.54x** (22.66 vs 14.73; Neural 21.7-23.5; per round 1.55 / 1.55 / 1.53) | 14.73 / 14.55 |
| reading a 13k-token prompt, both warm | 6.4x (14.7 s vs 94.8 s) | **9.3x** (10.7 s vs 99.2 s) | 99.2 s / 94.8 s |
| reading a 3k-token prompt, both warm | 6.4x (3.8 s vs 24.1 s) | 6.6x (3.9 s vs 25.5 s) | 25.5 s / 24.1 s |

Neural's first answer after the 13k prefill is the same hash in all three sessions (f9cf79b1, the canonical one); the follow-ups took the two known timing-dependent residency paths (device-class, both seen all day). llama.cpp's first answers were 12.7 / 13.6 / 13.3 this round against 13.5-14.1 earlier in the day: its first answer after loading weights through mmap is sensitive to the file cache the preceding Neural session leaves behind, so the follow-up ratio (1.54x, unchanged from Sep 30 (2)) is the steadier decode number; the first-answer ratio moved from 1.49x to 1.64x mostly on llama.cpp's side. Prompt reading: Neural 10.71 s vs llama.cpp 99.17 s (llama.cpp 93-102 s across the week), 3k: 3.87 s vs 25.47 s. Neural's own prompt time 10.7-12.0 s across the six 13k prefills of the day on this build.


### Sixth pass (2026-09-30, afternoon): the prompt path instrumented and taken to its balance point; decode knobs (MEASURED, `step19*`-`step26*`)

Method note: after the morning's freeze (README of the sessions), every agent in this pass only read and wrote code; every
number below was measured by one process at a time through the same harnesses.

**Instrumentation.** `prefill_*` timers in the request stats and the console line (`PTIME`: memcpy / stage thread / stage wait /
ring wait / routing sync / CPU-job run and wait / attention+router / GEMM launch / tail; the accounting closes on `prefill_s`).
A 13k prompt in layer order at BLOCK_M=16 (12.3 s): host GEMM-launch time 4.0 s over ~12,500 `_expert_gemms` calls (~10
pointwise torch launches each), synchronous `memcpy_mt` staging 2.45 s, copy-ring wait 1.3 s, routing sync (GPU) 3.45 s;
the CPU multi-token kernel ran 3.9 s on its worker thread and was never waited on.

| lever (all bit-identical unless stated; identity = `--prefill-order-check` in one process: logits, all K/V and ring buffers) | 13k prompt, warm, mirrored runs | verdict |
|---|---|---|
| prefill GEMM `BLOCK_M` 16 -> 64 (`NEURAL_PREFILL_BM`, default now 64) | **10.78 / 12.22 -> 7.67 / 7.73 s** (`step22`); 3k 3.9 -> 3.7 | **ship** (16-row tensor-core tiles wasted rows; `PREFILL_BM_CHECK` equal) |
| staging producer thread (`--stage-thread 1`, `--stage-ring 6`) + cached Triton launch (`NEURAL_FAST_LAUNCH`) | neutral at BM16 (the GPU was the wall: main thread waited 4.0 s for staged copies), useful at BM64 (stage wait 0.3 s, no memcpy on the main thread) | ship, defaults on (`PREFILL_FAST_LAUNCH_CHECK` equal) |
| grouped bias/SwiGLU epilogue per slot group (`NEURAL_PREFILL_GROUP=1`, `NEURAL_PREFILL_GROUP_EXPERTS`) | groups of 6: 10.57 / 10.59 s (the copy pipeline stalls: stage wait 3.7 s); groups of 3: 7.21 / 7.29 vs 7.21 / 7.22 ungrouped: host launch time 4.4 -> 2.7-3.4 s but the wait for staging/H2D grew by the same amount (`step25`) | measured neutral: **the prompt is now balanced between host and GPU at ~7.2 s**; kept as an opt-in |
| multi-token CPU kernel scheduling (dynamic + spin barrier, like decode) | +1-2.5% per call standalone, bit-identical (281 checks x 8 and 3 threads); in-server `cpu_job_wait` is 0.0-0.2 s, the jobs are hidden behind the GPU | not shipped (no in-server effect) |
| `--prefill-chunk` 8192 / 16384 | 11.8 / 12.4 s but NOT bit-identical (`PREFILL_CHUNK_CHECK`, even all-GPU) and +0.3 s on 3k prompts | rejected |

**Decode.** `--refresh-m 16` (admission candidates per refresh halved): captures per 512-token turn 2,060 -> 1,030, VRAM hit
rate unchanged (0.387 vs 0.384), **follow-ups 23.47 vs 22.77 tok/s (+3.1% median, +5.1% mean)**, first answers +2-4%
(`step20`, mirrored, 6 sessions per kind; `--refresh-every 16` +2.3% / +3.9%). The admitted set differs, so answers take a
different (internally consistent) residency path: device-class, as any admission change. Launcher default.

**Doorbell, robust port** (merged into main as e32cb5c, `--doorbell 1`, off by default; branch `work-bell` is the record): whole-token re-run on a spin timeout, mode
switch to the classic loop for the first 32 tokens and after a >300-fault token. MEASURED (`step23`): bit-identical at frozen
residency (four turns, same hashes), an injected timeout every 50 early tokens recovered 9 times per turn with the same
hashes and no failed request; performance mirrored 4 sessions: **22.32 vs 22.70 tok/s (-1.6%), gpu_wait 4.82 vs 4.76 ms**.
The launch slack it targeted is gone since the kernel scheduling change; NO_GAIN, kept as the record. Merged into main on the
evening of 2026-09-30 at the owner's request, off by default: the default path (`_decode_token_classic`) is AST-identical to the
pre-merge `decode_token` (`tests/test_doorbell_mock.py` against 9dd3fe5) and the merged build reproduced the pre-merge build's
answer hashes turn for turn at frozen residency (`step28`, four turns, MEASURED).

**Cold-expert prefetch** (`--kernel-cold-prefetch 1`, kernel knob `gptoss_set_cold_prefetch`, off by default; default path
bit-identical to the previous build): before each call a one-byte probe per expert flags non-resident experts and one
`PrefetchVirtualMemory` call reads them at queue depth. MEASURED in the intended regime (`step26`: Neural rewrites right after llama.cpp sessions, mirrored off/on, 4 sessions): the hard-fault storms of the morning did not recur with ~56 GiB free (no stall window, ~460 soft faults/token), and the knob was neutral (25.4 / 25.7 tok/s off vs 25.7 / 25.1 on over the whole answer; identical hashes and residency in all four). Its target regime needs memory pressure that must not be manufactured on this box, so it ships OFF and untested there; the default path is bit-identical to the previous build (kernel_tuning_ab --ref).

Where the 13k prompt stands (MEASURED, `step25` g0): **7.2 s = host launch 4.4 s ‖ GPU (routing sync 1.2 + staging/H2D
waits 0.7) with the CPU multi-kernel 5.0 s hidden on its worker.** The two sides are balanced; the next step on either side
alone does not move the wall (the grouped epilogue proved it for the host). llama.cpp reads the same prompt in 93-102 s.
The runbook below is kept as written; the tables above record what each step gave.

### Against llama.cpp with the sixth-pass build (MEASURED 2026-09-30 afternoon, `step27*`)

Same legs and flags as the Sep 30 (3) table on the merged sixth-pass build (8636f7b: `BLOCK_M` 64 prefill tiles, staging
thread + cached launch, `--refresh-m 16`, `--prefill-order layer`, packed store); conversation x3 pairs and warm prompts
(no rewrite leg, as before). Free RAM 56.6-56.9 GiB at every session start. llama.cpp's first answers ran low this round (11.6 / 10.9 / 12.8 tok/s against 13.3-13.6 in the two previous rounds; its first prompt took 125-131 s, paging the 61 GB mmap in next to Neural's store in the file cache), so the first-answer ratio is the least steady of the four rows; the follow-ups (llama.cpp 14.3-16.6 tok/s) are the number to compare across versions.

| workload | Sep 30 (3): layer-order prefill | **Sep 30 (4): + BM64 tiles, staging thread, refresh-m 16** | llama.cpp (4) / (3) |
|---|---|---|---|
| 13k-token coding conversation, first answer (median of 3 fresh servers) | 1.64x (21.75 vs 13.25) | **1.96x** (22.74 vs 11.60; per round 1.95 / 2.08 / 1.84) | 11.60 / 13.25 |
| same, follow-up answers (9 turns) | 1.54x (22.66 vs 14.73) | **1.57x** (24.13 vs 15.33; Neural 23.4-25.2; per round 1.63 / 1.56 / 1.61) | 15.33 / 14.73 |
| reading a 13k-token prompt, both warm | 9.3x (10.7 s vs 99.2 s) | **13.9x** (6.8 s vs 95.2 s) | 95.2 s / 99.2 s |
| reading a 3k-token prompt, both warm | 6.6x (3.9 s vs 25.5 s) | 7.4x (3.4 s vs 25.0 s) | 25.0 s / 25.5 s |

Neural's answers are the same hash in all three sessions at every turn (734a944b / a18e28a9 / fe994f13 / f3264eb1) with identical VRAM hit rates turn for turn (0.3535 / 0.3894 / 0.4003 / 0.3701): one residency path, reproduced three times. The first answer differs from the previous build's canonical f9cf79b1 because `--refresh-m 16` admits a different set (device-class, sixth pass); the prompt-side changes are bit-identical by the in-process checks. In-conversation 13k prompt 7.8-8.5 s (was 11.3 s), follow-up prompts 2.2-2.4 s; the warm 13k prompt alone 6.8 s (10.7 s on the previous build), 3k 3.4 s (3.9 s).

## Measured so far (development machine, 2026-09-29; MEASURED unless noted)

| what | result |
|---|---|
| host DRAM read ceiling, 8 threads, 4 KiB pages (probe, corrected timer, 12 passes) | **~40 GB/s** (36.0 plain, 39.8 with prefetch; best pass 40.6) = 83% of the CALCULATED 48 GB/s DDR4-3000 dual-channel peak |
| file view of the store vs private memory | no difference (39.1 vs 36.0-39.8) |
| large (2 MiB) pages | not tested: the account lacks "Lock pages in memory" |
| old kernel, standalone, synthetic experts, E=4 | 31.8-32.2 GB/s |
| new kernel, settings off | 31.7-31.9 (same code path, as designed) |
| prefetch / paired rows alone | 30.7-32.3: **nothing** |
| pinned one thread per core (`--kernel-affinity 2`) | 34.3 (1.07x); + pair + prefetch 2048: 35.1-35.3 (**1.10x**) |
| every setting vs the old build | bit-identical |
| in-server kernel rate (v4 request stats) | 24-29 GB/s (25.8 on the code trace) |
| server decode with any lever | see the in-server table above (measured the same evening, at 1b44ccb) |

What it says: the memory side is the wall at ~40 GB/s, the standalone kernel is at 80% of it, and
the in-server kernel at 60-70%. The kernel-side levers are therefore worth at most ~1.25x on the
77% of a token they cover, and the standalone A/B already shows most of that is thread placement,
not access pattern. Everything larger has to come from (a) the in-server gap: per-call fixed costs
at 1-3 misses per layer (barriers, fork/join, scalar activation), the Python master thread being one
of the 8 workers, capture writes; (b) fewer bytes per token (more slots, zero-surcharge admission);
(c) a second read pipe (the GPU's copy engine streaming misses from the arena, step 6b). The
first probe run reported 52-55 GB/s because MinGW's `omp_get_wtime` ticks in whole milliseconds;
fixed (`QueryPerformanceCounter`). `kernel_tuning_ab.py` now loads the DLL by absolute path and
measures the pinned rows in a child process (pinning cannot be undone in a process).

## 0. Rebuild the kernels (once)

```bat
tools\build_kernels.bat
```

builds `gptoss_cpu_cap2.dll` / `gptoss_cpu_multi.dll` with the new `gptoss_set_tuning` export and
the new `membw_probe.dll`. The server prints a warning and ignores the kernel flags if the DLL
predates them.

## 1. Measure the DRAM ceiling (30 min, no server)

```bat
.venv\Scripts\python tools\measure_host_dram.py
```

Prints the DIMM configuration (rated vs configured clock, channel population) and measures 8-thread
read bandwidth over a private 4 KiB-page buffer, a 2 MiB large-page buffer (needs "Lock pages in
memory" for the account) and a read-only view of `layer_0.slots`, each with and without software
prefetch and with one or two streams per thread. Writes `reports/host_dram_bandwidth.json` with a
verdict. The server's own kernel has never been compared with this: it reports 24-29 GB/s and the
docs call 29 "the ceiling", but that number is the kernel timing itself.

- Plain 4 KiB ≈ large pages ≈ file view, prefetch does nothing, everything within 10%: the wall is
  the memory side; expect steps 2-3 to give ≤10%.
- Prefetch or large pages lift the rate by 15% or more: steps 2-3 carry that much on 77% of a
  code token.

## 2. Kernel memory-level parallelism (10 min) — bit-identical

```bat
.venv\Scripts\python tools\kernel_tuning_ab.py --lib .\gptoss_cpu_cap2.dll --ref .\old_cap2.dll --affinity 2
```

Checks that every (prefetch, pair, fuse) setting gives bit-identical outputs and capture bytes on
synthetic experts, then prints ms/call and GB/s per setting for E=4 and E=2 experts per call
(the server averages ~2.4 misses per layer, where fixed per-call costs matter); the pinned rows
come from a child process. `--kernel-fuse 1` is new: a two-phase kernel with three fewer barriers
per call (gate/up rows of an expert are streamed as a pair and activated on the spot; the down
rows of all experts are summed directly). It keeps the four-phase kernel's memory-materialized
expression forms, which is what makes it bit-identical (a register-held version was 1 ulp off in
1 of 16 synthetic cases through compiler contraction). Then run the server with the best setting:

```bat
start_neural_server.bat --kernel-affinity 2 --kernel-fuse 1 --kernel-pair 1 --kernel-prefetch 2048
```

and compare `cpu_GBps` / `decode_tok_s` in the `neural` block on the same prompt (interleave runs;
`tools\bench_vs_llama.py --workload rewrite` is the fairest fixed workload). Note `--kernel-affinity`
also pins the server's Python thread (it is OpenMP thread 0) to logical CPU 0; if that hurts,
try stride 2 with `--threads 7` so the master's core is not also a full worker.

## 3. Serial time: `--wait poll` (5 min)

Busy-polls the per-layer event instead of blocking. Compare `gpu_wait_ms_per_tok` and
`other_ms_per_tok`, which the server now reports alongside `replay_ms_per_tok` and
`admission_ms_per_tok` (the ~7-8 ms/token that used to be an unexplained residual).

## 4. More slots for free: `--kv-ring 256` and `--scratch 6` (30 min)

The 18 sliding-window layers only ever read the last 128 positions, but allocated K/V for all 16,384.
`--kv-ring 256` keeps a 256-position ring per sliding layer (fused_core's ring path, bit-identical to
the linear cache) and frees ~0.55 GiB. Raise the pool to use it:

```bat
start_neural_server.bat --kv-ring 256 --pool 6.05
```

Prompt-cache rollback uses ring checkpoints (9 MiB host each, `--kv-checkpoints 8`), taken at the end
of every prompt and, replacing that one, right after the first generated token (a follow-up shares it,
so it reuses prompt + 1 tokens like the linear cache; the first version stopped one token early):
continuing a conversation costs nothing; re-rendering the last turn resumes from
that checkpoint; editing an earlier turn resumes from that turn's checkpoint; a side request that
diverges early re-processes from 0 (it did from the divergence point before). `kv_ring.py` holds the
policy; `tests\test_kv_ring.py` covers the cases. `--scratch 6` frees 10 more slots (prefill staging
only; check `tools\tune_prefill.py` did not regress). Expect roughly +4-5 hit points per 45 slots
(`vram_hit` in the `neural` block), ~3 ms/token on code.

Not compatible with `--selftest` and with hot-set calibration (both use the eager reference
prefill); the server refuses those combinations.

## 5. Locked expert arena: `--arena pinned` (1 hour)

Holds the CPU-side experts in `cudaHostAlloc`'d (pinned) memory instead of the page cache: no page
trimming after long prompts (the measured 5-13 tok/s over a first answer's first 128 tokens), and
the same pointer is valid for GPU zero-copy reads (step 6). Allocated in 1.58 GiB chunks until the
budget (`free RAM - --warm-margin`, or `--arena-max-gib`) or the first failure; experts that do not
fit stay on the mmap path as before. Watch the startup lines `arena: ...` and the first answer after
a 13k-token prompt (`decode_page_faults_per_tok` should drop to ~0; `decode_tok_s_per_64` should be
flat from the first window).

`--arena large` uses 2 MiB pages (`VirtualAlloc(MEM_LARGE_PAGES)`, needs the account right "Lock
pages in memory" in secpol.msc and a recently booted machine) and registers the memory with CUDA;
if registration fails the arena still serves the CPU and step 6 is disabled with a message.

Risk: 50 GiB of non-pageable memory on a 64 GB machine. Close other programs first, as the README
already asks; if allocation stops early the startup line says how many rows landed.

## 6. Zero-surcharge admission: `--admit-gpu 1` (the lever the policy study points at)

```bat
start_neural_server.bat --arena pinned --admit-gpu 1 --admit-rule window --admit-k 3 --admit-window 64 --victim lru --static 0
```

An admitted miss is no longer computed by the CPU and copied twice. The copy engine streams the
expert from the pinned arena straight into the victim slot (the same `cudaMemcpyAsync` path the
shipped admission upload uses, at the MEASURED 21.4 GiB/s pinned rate; no CPU, no capture buffer)
and the pool GEMV computes it there, bit-identical to a VRAM hit. Admission cost drops from two
extra RAM traversals to zero, which is what makes generous policies pay: SIMULATED on
`benchmarks/code_trace.json`, RAM expert reads per token 96 → 73 with the window rule above
(Belady 41, next-16-token demand oracle 45).

At startup the server runs a self-test (`--selftest-admit 1`): it streams one arena expert into a
scratch slot and checks (a) the arena row equals the store row, (b) the graph's output equals the
eager pool path bit for bit, (c) it agrees with the CPU kernel. If any check fails the feature
turns itself off and the server continues on copy-on-compute. Look at `gpu_admissions_per_tok`,
`vram_hit`, `cpu_experts_per_tok` and `cpu_GBps`. `--admit-gpu-max` caps GPU admissions per layer
(PCIe budget; 1 is the safe default).

## 6b. The GPU as a second read pipe: `--zc-misses 1`

```bat
start_neural_server.bat --arena pinned --zc-misses 1
```

With the DRAM ceiling at ~40 GB/s and the in-server CPU kernel at 24-29, a quarter to a third of
the memory bandwidth goes unused every token. This flag routes up to N plain misses per layer
(never the last one, so the CPU always has work) through the same DMA-into-a-scratch-slot +
pool-GEMV path, without admitting them. The copy engine and the CPU then read DRAM
concurrently. Exact (same numerics as a hit). Whether it adds bandwidth is the open question:
`RESULTS.md` measured CPU + GPU zero-copy reads at 28.5 vs 29.2 GB/s CPU-alone, but that test
used SM zero-copy reads with a strided access pattern against a CPU believed to be at its ceiling;
DMA reads are sequential and the CPU is now known to sit well under the ceiling. If it works, the
kernel term shrinks by the fraction of misses moved (CALCULATED: 1 per layer ≈ 18 of ~96 reads/token,
≈1.15x; with the CPU also pinned ≈1.3x). If DRAM contention makes it a wash, `cpu_GBps` drops by
the same amount the GPU takes and `decode_tok_s` does not move: turn it off. Needs the arena; can
be combined with `--admit-gpu 1`. Does not need `--static 0`.

The `--static 0` matters: under zero-surcharge admission every simulated policy is better without
a protected static core (the core was a defence against admission cost).

## 7. Near-miss candidates: `--near-miss 2` (free)

The router's ranks 5-8 (already computed, never used). `--near-miss 1` computes them into the
per-layer pack (and into traces) without changing admission; `--near-miss 2` also marks them as
admission candidates. Exact: the top-4 and their weights are untouched; at 0 the kernel is the
previous one. Works with either admission path.

Numerics note for step 6: a GPU-admitted expert follows the pool path's bf16 rounding points (like
every hit) plus one more bf16 add into the hit sum, where the CPU miss path is fp32 throughout.
This is the same class of difference as which device computes an expert today (`RESULTS.md`
notes residency is already timing-dependent); it is not bit-exact against the CPU path.

## 8. Demand probe (research; offline first)

1. Record traces with hidden states: `POST /neural/trace {"on": true, "hidden": true}`, run agent
   traffic (tens of thousands of tokens), `GET /neural/trace > reports/trace_hidden.json`.
2. `python tools\fit_demand_probe.py reports\trace_hidden.json --l2 1e-4` fits a linear probe from
   the final residual to the next-16-token expert demand and reports AUC / hit@430 against the
   past-16 histogram baseline and the oracle. Sweep `--l2` 1e-5..1e-3.
3. Feed `reports/demand_probe_pred.npz` into the offline cache study (Neural repo,
   `scripts/neuralserver_admission_study.py`) before touching the runtime.
4. Runtime: `--demand-probe reports\demand_probe.npz --admit-rule demand` (with `--admit-gpu`):
   one 2880x4608 GEMV per token on the GPU; admission = predicted demand beats the least-demanded
   adaptive resident; eviction = lowest predicted demand.

## Flag summary

| flag | default | effect |
|---|---|---|
| `--kernel-prefetch B` | 0 | software prefetch B bytes ahead in the CPU expert kernel (bit-identical) |
| `--kernel-pair 0/1` | 0 | two rows per thread iteration (bit-identical) |
| `--kernel-affinity S` | 0 | pin kernel thread t to logical CPU t*S (1.07-1.10x standalone; MEASURED in-server: -8%, see top) |
| `--kernel-fuse 0/1` | 0 | two-phase kernel, 3 fewer barriers per call (bit-identical; MEASURED in-server +6.4%, launcher default) |
| `--zc-misses N` | 0 | GPU computes up to N plain misses per layer via DMA into scratch slots (needs arena; MEASURED: +1% follow-ups, -11% first answers) |
| `--wait event/poll` | event | host wait on the per-layer router event (MEASURED: poll = no gain) |
| `--kv-ring R` | 0 | rolling K/V for sliding layers (R ≥ 128); frees ~0.55 GiB at R=256 |
| `--kv-checkpoints N` | 8 | ring snapshots for prompt-cache rollback |
| `--scratch N` | 16 | prefill scratch slots |
| `--near-miss 0/1` | 0 | router ranks 5-8 as admission candidates |
| `--arena off/pinned/large` | off | locked host arena for the CPU-side experts |
| `--arena-max-gib G` | 0 (auto) | arena budget |
| `--admit-gpu 0/1` | 0 | zero-surcharge GPU admission (needs arena) |
| `--admit-gpu-max N` | 1 | GPU admissions per layer |
| `--admit-rule server/window/demand` | server | who is admitted |
| `--admit-k K --admit-window W` | 3 / 64 | window rule |
| `--victim count/lru` | count | eviction key |
| `--demand-probe FILE` | none | probe weights for `--admit-rule demand` |
| `--selftest-admit 0/1` | 1 | verify the zero-copy kernel at startup |

New response fields: `replay_ms_per_tok`, `admission_ms_per_tok`, `other_ms_per_tok`,
`gpu_admissions_per_tok`, `gpu_misses_per_tok`, `cpu_experts_per_tok`, `vram_reserved_gib`,
`vram_allocated_gib`.
`GET /neural/trace` now also returns `weights` (top-4 softmax), `near_miss` (ranks 5-8) and, when
recording was started with `"hidden": true`, `hidden` (final residual per token).
