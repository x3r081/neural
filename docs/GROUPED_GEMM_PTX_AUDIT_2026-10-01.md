# Grouped GEMM pointer-alignment audit

Read-only compiler evidence collected on 2026-10-01, before adding guarded
alignment hints. GPU: RTX 3080 Ti (sm86), Triton 3.7.1, torch 2.10.0+cu130.
Both kernels use BLOCK_M=64, BLOCK_N=128, GK=90, 4 warps and 3 stages.
This is compiler evidence, not a measured claim of inference improvement.

The CUDA cache root is `C:\Users\USER\.triton\cache`. Relevant entries:

- Stock packed gate GEMM, N=5760:
  `2HA2ZSKX2ESRFKFNV3333KF6WF25AW5D3DYTNOLRHDX3477W6KLA`.
  TTIR signature marks x/output pointers `tt.divisibility = 16`.
  PTX uses 16-byte `cp.async.cg.shared.global` input copies, and eight
  `st.global.v4.b32` output instructions. Shared memory: 20,480 bytes.
- Grouped packed gate GEMM, N=5760:
  `C4C4GZRS3FEINM7BXOZPRSKDBWYCTV2VPEQZF2UBI3PWTKYHKKYQ`.
  Descriptor-derived x/output pointers have no alignment information.
  PTX has sixteen scalar `ld.global.b16` input instructions and sixty-four
  scalar `st.global.b16` output instructions. The code-weight input still
  uses cp.async, but the activation-input pipeline is lost. Shared memory:
  16,384 bytes.
- Grouped packed down GEMM, N=2880:
  `OQUVGE4PCHWZ6OM4J6FF655FM7IXXZ4XUE7PFUHN3QFUNVVSWDNA`.
  Same scalar activation-load/output-store pattern.

The PTX declarations also expand virtual address registers (225 b64 in the
grouped gate variant versus 55 in the stock gate variant). These are virtual
register counts; they do not establish physical register pressure or occupancy.

To reproduce, inspect `_mxfp4_gemm.ttir/.ptx/.json` in the stock cache entry and
`_mxfp4_gemm_grouped.ttir/.ptx/.json` in the grouped entries. Search TTIR for
`tt.divisibility`, and PTX for `cp.async`, `ld.global.b16`, and `st.global`.
Kernel names and cache hashes pin the exact pre-hint compiler outputs.

The guarded change now records whether every nonempty pair's input/output address
and row pitch are 16-byte aligned. Only that specialization hints the loaded
integer byte addresses as multiples of 16 before converting them to pointers.
Unaligned inputs retain the original fallback. Arithmetic, tensor-core tiles,
pair boundaries and accumulation order remain unchanged. Both aligned and
unaligned variants must pass the bit tests before accepting any timing result.
The implementation is `GroupedGemmPlan.aligned` plus the kernel's `ALIGNED`
specialization. The first hint attempt attached metadata before the integer to
pointer conversion. The scheduled microbenchmark passed all 120 bit checks
(`benchmarks/codex_review_20261001/grouped_gemm_aligned_micro.json`), but compiler
inspection showed the vectorization regression remained.

For that first attempt, aligned real-shape packed cache entries are gate
`TQ46BVRSZONBPEYSRZ2TJX4IF2UMI3O6VXFY37EMEQ4PQU4W7WPA` and down
`YXZKB23VXRHXGP4G5MBZWFMT2IMB4GOTTNQFIDFYSBXYXSDDG4RQ`. Their TTIR integer
descriptor loads have `tt.divisibility = 16`; their `tt.int_to_ptr` results do
not. Both still have sixteen scalar activation loads, sixty-four scalar output
stores, and 16,384 bytes shared memory. Bit identity alone did not establish
that the intended compiler optimization occurred.

The minimal follow-up attaches the same guarded hint directly to the
converted pointers. The [Triton 3.7.1 AxisInfo implementation](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Analysis/AxisInfo.cpp)
omits an integer-to-pointer visitor; its pessimistic initialization accepts
explicit hints on defining operations. Pointer divisibility is measured in
bytes and divided by the element byte size when choosing vector width. This
supports the follow-up, but new PTX and bit checks are still required. No extra
contiguity assertion is proposed: the existing arange already defines adjacent
lanes, and the validation proves input/output addresses and row pitches.

The separately scheduled pointer-side microbenchmark exited successfully:
`benchmarks/codex_review_20261001/grouped_gemm_pointer_aligned_micro.json` has
120 passing checks, bit equality, finite outputs, intact guards, deterministic
repeats and CUDA graph equality. Its deliberate output-bit corruption was
detected. Peak allocated memory was 50.068 MiB. Root's CPU plan tests passed
16 tests before this pointer-side change; their validation logic is unchanged.

Aligned packed gate PTX is now pinned by cache
`HIW3KV52RJAXNNDGUT4HPN6FWDBTCKGM4UXEZBLUPMTJUCIJQY3A`, and down by
`X7CZKVIALGGELI5GYFOD5NCLEWGVKZIBXKOLF64JR3S67MK6QEUA`. Both have nine
16-byte cp.async instructions, zero scalar BF16 global loads, eight
`st.global.v4.b32` stores, zero scalar BF16 global stores, and 20,480 bytes
shared memory. TTIR places divisibility metadata on `tt.int_to_ptr` itself.
This confirms that the intended memory-operation vectorization was restored.

For aligned pairs with M=[0,1,63,64,65,129], BM=64 and BN=128, measured packed
gate stock/reused-plan/fresh-plan medians were 0.743/0.404/0.460 ms; packed down
were 0.656/0.243/0.311 ms. Raw gate medians were 0.719/0.369/0.441 ms, raw down
0.619/0.234/0.281 ms. These are synthetic CUDA event timelines including host
submission gaps, four alternating rounds of 30 iterations; they do not measure
inference. Full-model hidden/logit/KV identity and warm prompt timing remain
separately scheduled after this pointer-side kernel change.

The initial synthetic microbenchmark intentionally used unaligned pointer
offsets, so its result alone cannot establish the production aligned-path
opportunity. The test tool now adds aligned pairs with a 24-element guard gap,
keeping all output-row origins aligned while preserving the unaligned cases.
