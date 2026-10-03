# Decode-lever sessions, 2026-09-29 (MEASURED)

`tools\bench_vs_llama.py --workload conversation` (13k-token prompt + 3 follow-ups, 512 tokens each,
temperature 0), one server per session, all on this branch at 1b44ccb plus the kernel-affinity fix
(`kernels\gptoss_cpu_cap2.c`; the `aff*` sessions passed `--kdll gptoss_cpu_cap2_tls.dll`, a build of the
same source that this commit ships as `gptoss_cpu_cap2.dll`). Step 1 ran with ~46 GiB free, the rest ~54-56. Answer text is stripped; each turn
keeps `content_sha1` so answers can be compared across sessions. Verdict table: `docs\LEVERS.md` (top).

| file prefix | sessions | what |
|---|---|---|
| `step1_baseline` | main, branch x3 interleaved | `main` = clean v4 checkout (`--server-roots main=...`), `branch` = this tree, default flags |
| `step2_*`, `step2b_*` | branch, ring, aff, affpp, ringaff, pool4, pool45 | kv-ring / thread pinning / pool-size slope; `pool4` (`--pool 4.0`) refused to start and has no file |
| `step3_*`, `step3b_*` | ringonly, ring, arena, arena40, admit_srv40, admit_win40 | arena and zero-surcharge GPU admission; `step3_02_arena` (default 45.9 GiB budget) was stopped after one session and is not included |
| `step4id_*` | ringonly | `--kv-ring 256` at pool 5.5 AFTER the checkpoint fix: all four answers byte-identical to v4, `cached_tokens` 13108 |
| `step4_*` | ring, fuse, dma, zc1, dmazc | after the checkpoint and arena fixes: `--kernel-fuse 1`, the DMA admission path (`--admit-gpu 1 --admit-rule server` on a 40 GiB arena), `--zc-misses 1`, both |
| `step5conv_*`, `step5rw_*`, `step5_prompt_warm.json` | llama, neural | the comparison against llama.cpp b10361 (`-ncmoe 31`) with the new launcher defaults (ring + fuse): conversation, rewrite, warm prompt |
| `step6_prompt_warm_v4.json` | neural (v4) | the unchanged Sep 24 build (clean 72d3d7b checkout) through `tools\bench_prompt_warm.py` on Sep 29: 15.6 s / 4.1 s |
| `step7_*` | base, poll, fusedma, fusedmapoll, hostpatch | third pass: `--wait poll`, fuse + DMA admission, the host-residual patch (a separate worktree copy; not shipped). Sessions 5-9 ran after midnight (date token changed): compare within date groups |
| `step8id_*` | base0, bell0, rawnew0, ps4eq0 | identity run with `--refresh-every 0` (residency frozen): all four byte-identical per turn |
| `step8_*` | base, bell, bellnf, rawnew, ps4eq, ps4 | performance: doorbell (`bell` aborted on a spin timeout), new DLLs on the raw store, packed store at equal slots (`--pool 5.872`) and at `--pool 6.05` |
| `step9conv_*`, `step9rw_*`, `step9_prompt_warm.json` | llama, neural | Sep 30: the comparison against llama.cpp b10361 with the packed store (`--store-dir G:\NeuralStores\gptoss120b_ps4`) and the launcher defaults, from the merged `main` tree (15dd3e7): conversation (3 pairs), rewrite (2 pairs), warm prompt |
| `step10id_*` | base0, dysy0 | kernel scheduling, identity leg with `--refresh-every 0`: the previous tracked `gptoss_cpu_cap2.dll` vs the dynamic + yield-spin build (`--kdll cap2_dysy.dll`), byte-identical per turn |
| `step10_*` | base, dyn2, dysy | kernel scheduling, mirrored performance leg (base, dyn2, dysy, dysy, dyn2, base): `dyn2` = dynamic chunks only, `dysy` = dynamic + yield-spin barrier (shipped as the tracked DLL). Ran with ~3 GiB less free RAM than `step9` |
| `step11conv_*`, `step11rw_*`, `step11_prompt_warm.json` | llama, neural | Sep 30 (2): the comparison against llama.cpp b10361 with the shipped kernel-scheduling build (2798e84), packed store, launcher defaults; same legs as `step9`, same RAM-tight environment as `step10`. The rewrite leg was spoiled on both sides (a Neural stall window, a llama.cpp run at 11.3 tok/s) |
| `step11rw2_*` | llama, neural | the rewrite leg re-run after free RAM recovered: llama.cpp 14.6 / 14.5; Neural 21.33 (answer 42f83167, clean) and 16.35 (answer 3c280373, one stall window) over the 291-token span |
| `step12_*` | prev, ship, dysw | rewrite-regime kernel A/B, mirrored: `prev` = the pre-scheduling DLL (`--kdll cap2_prev.dll`), `ship` = dynamic + yield-spin (2798e84), `dysw` = dynamic + yield-then-WaitOnAddress (157882d, shipped). Stall windows follow the answer (3c280373: 2 of 3 stalled; 42f83167: 0 of 3), not the kernel; steady-state windows 23.0 / 24.7 / 24.5 tok/s |
| `step13rw_*` | llama, neural | the rewrite leg vs llama.cpp with the final build (157882d): the Sep 30 (2) rewrite row |
| `step14_chunk_pw.json` | neural, neural8k, neural16k | `tools\bench_prompt_warm.py`, two mirrored rounds: `--prefill-chunk` 4096 / 8192 / 16384 (zero-code upper bound for the prompt lever; 8192 and 16384 are faster on 13k but change outputs, see `step16`) |
| `step16_order_check.log` | (one process) | `--prefill-order-check`: block vs layer order bit-identical (logits, all K/V, ring) on a 13k prompt; chunk 4096 vs 16384 NOT identical, with or without the CPU kernel |
| `step17_pw_*.json`, `step17conv_*` | base, lm | layer-major prefill: warm prompts block/layer/layer/block (one server each; `lm` from the work-lm tree) and the 13k conversation base,lm,lm,base (first answer identical hash across all four) |
| `step18conv_*`, `step18_prompt_warm.json` | llama, neural | Sep 30 (3): the comparison against llama.cpp b10361 with the layer-order prefill build (c1d993e), packed store, launcher defaults; conversation (3 pairs) and warm prompts |
| `step19_pw_split.json` | neural | the instrumented layer-order prefill (BM16): the time split that pointed at the GEMM launch cost and the staging copies |
| `step20_*` | base, m16, e16 | admission rate, mirrored: `--refresh-m 16` and `--refresh-every 16` vs the launcher's 32 / 8 |
| `step21_order_check.log`, `step24_order_check.log` | (one process) | the five in-process identity lines with the new prefill code (order, chunk x2, BM 16 vs 64, stock vs fast launch); step24 with the grouped epilogue and BM64 defaults |
| `step22_pw_*.json` | base, new, bm64 | mirrored warm prompts: main c1d993e vs work-lm (staging thread + fast launch) vs + `NEURAL_PREFILL_BM=64` |
| `step23id_*`, `step23_*` | base0, bell0, inj0, base, bell | the robust doorbell: identity at frozen residency, injected timeouts (`--doorbell-inject 50`), mirrored performance |
| `step25_pw_*.json` | g0, g6, g3 | grouped epilogue: off, groups of 6, groups of 3 (mirrored) |
| `step26rw_*` | llama, off, on | cold-expert prefetch in the cold regime (Neural rewrites after llama.cpp sessions), knob off vs on, mirrored |
| `step27conv_*`, `step27_prompt_warm.json` | llama, neural | Sep 30 (4): the comparison against llama.cpp b10361 with the sixth-pass build (8636f7b: BM64 prefill tiles, staging thread, `--refresh-m 16`), packed store, launcher defaults; conversation (3 pairs) and warm prompts |
| `step28_*` | pre0, post0 | the doorbell merge (e32cb5c) against the pre-merge main 9dd3fe5 with residency frozen (`--refresh-every 0`), doorbell off: answer hashes and hit rates equal on all four turns |

**Safety note (2026-09-30 11:31):** the PC hard-froze (Kernel-Power 41, no bugcheck) while three audit agents ran heavy
standalone measurements at the same time (a GPU prefill replica + an 8-thread kernel hard-faulting a memory-mapped file purged
with unbuffered writes + cudaHostRegister loops). No benchmark ever combined those. Rule since then: agents read, analyze and
edit; every measurement runs alone, one server or one script at a time, through these harnesses.
| `step3c_*` | admit_win40, admit_srv40 | **same flags, different code:** `setup_arena` ranked VRAM-resident experts into the arena as well (the `if TAB[g] < 0` filter removed). Reverted after these two sessions; the args recorded in the JSON do not show this |

`kind` names: `ring` = `--kv-ring 256 --scratch 6 --pool 6.05`; `aff` = `--kernel-affinity 2` (fixed DLL); `affpp` =
aff + `--kernel-pair 1 --kernel-prefetch 2048`; `arena40` = ring + `--arena pinned --arena-max-gib 40`; `admit_srv40` =
arena40 + `--admit-gpu 1 --admit-rule server`; `admit_win40` = arena40 + `--admit-gpu 1 --admit-rule window --admit-k 2
--admit-window 4 --victim lru --static 0`. Exact strings are in each file's `sessions[].args`.
