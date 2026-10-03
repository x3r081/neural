# October 1 optimization evidence

Reviewed upstream: `a5d3c95e17323e9b78f83976da583e31f73f188e`.
Core experimental kernels: `67ad186`; explicit workspace runner: `c145fce`;
maximum-only dispatch guard: `db68eba`; selected 1,024–4,096-token range:
`06d05ae`; final warm-runner combined-cache setting: `171801a`. Later
report/launcher commits do not imply that old measurements used new defaults.
The source fingerprints and effective settings in each result are authoritative.

All model runs use the existing native gpt-oss-120B and lossless packed-scale
store. No extra quantization, checkpoint change or fabricated routing trace.

## Completed performance measurements

- [Verified warm-profile summary](warm_profiles_verified.json), recomputed by
  [audit_warm_profiles.py](audit_warm_profiles.py), covers all three warm
  prefill studies. It checks request/response archive hashes, corresponding
  request and internal prompt/generated-ID identities, and zero cached-prefix
  reuse.
- [Bounded sweep](warm_prefill_bounded_abba.json): six fresh sessions,
  off/g1/g3/g3/g1/off, 1,024-row target, unbounded prompt-length dispatch.
  Group3 lowers 3k file-prompt median latency by 7.04%, but raises the two
  long-prompt medians by about 2.3%.
- [Earlier short-size sweep](warm_prefill_short_abba.json): four fresh sessions,
  off/g3/g3/off, with 13k and short discarded warm-ups. Latency was roughly
  neutral at 128/512, 2.47% lower at 1k, and 7.43% lower at 3k; two sessions
  per setting, so this is preliminary.
- [Final short-profile sweep](warm_profile_abba.json): four fresh sessions,
  off/g3/g3/off with 1,024–4,096 dispatch, 1,024 row cap, and cache trim mode 1.
  Both arms received the same discarded 13k and 3k warm-ups. Median 3k prefill
  fell from 3.69655s to 3.43665s (7.03088%); 1k changed 2.59705s to 2.56175s,
  while 128/512 were effectively neutral. Each arm has two sessions; treat this
  as a narrow prompt-size result, not a general serving-speed gain.
- [Integrated cold full-answer trial](integrated_profile_abba.json) with its
  [request replay](integrated_profile_abba.json.replay.json) and
  [audit](integrated_summary.json): four off/opt/opt/off
  sessions, 3,220 rendered prompt tokens and 512 output tokens. All four output
  ID hashes matched. Cold prefill medians were 4.0858s/4.0114s, decode
  24.5115/24.4532 tok/s, and client wall 24.99/24.965s (off/combined option):
  effectively flat. Reserved memory fell 10.248→9.715 GiB. This supports the
  memory effect, not an end-to-end latency claim.
- [Allocator trial](trim_study.json) and [summary](trim_summary.json): six
  sessions, all corresponding generated IDs equal. CUDA-only release frees
  1.1113 GiB reserved memory in about 6 ms. No reliable decode throughput gain.
- [Unchanged Neural versus llama.cpp](control_baseline.json): copied byte-for-byte
  from the control checkout's `benchmarks/codex_baseline_20260930.json`.
  Follow-up medians 22.58255 versus 14.98790 tok/s; this is existing performance,
  with the midnight/template/history limitations explained in the review.

## Correctness evidence

- [Selected dispatch](full_prefill_selected_identity.json): final 1,024–4,096
  range; six tested lengths. Grouped kernels actually execute at 1,024/3,000;
  stock executes at 128/512/5,000/13,000. All dispatch assertions and full
  hidden-state, every-position logits and full K/V hashes pass.
- [Unbounded final-kernel proof](full_prefill_pointer_identity.json): group sizes
  1/3/6 at 128/3,000/5,000/13,000; all 12 comparisons exact. This exercises the
  kernel at long lengths that selected serving dispatch deliberately excludes.
- [Final grouped microcheck](grouped_gemm_pointer_aligned_micro.json): 120 GPU
  checks, graph replay, alignment fallback, guards and corruption negative
  control. See the [PTX audit](../../docs/GROUPED_GEMM_PTX_AUDIT_2026-10-01.md).
- [Masked microcheck](masked_gemv_micro.json): 512 finite cases and an explicit
  NaN limitation; [full decode proof](full_decode_identity.json): 64 steps at
  each of three prompt lengths, fixed residency, every-step logits/residuals,
  token IDs and full K/V equal.
- [Delivery pytest log](selected_tests_delivery.log): 102 tests passed in 33.09
  seconds, with 23 existing metric-return warnings. The earlier
  [selected run log](selected_tests_final.log) also passed. Neither is a
  whole-directory pytest pass; standalone model/GPU checks are listed separately.

## Rejected and development artifacts

- [Masked decode ABBA](masked_gemv_abba.json) and [audit](masked_summary.json):
  complete matched-request trial, off/on/on/off. Follow-up medians 24.0232 versus
  23.4691 tok/s; no speed gain. Both masked runs match the first control's
  generated IDs, but the last control diverges on its last two answers. The
  first two answers match everywhere and are also slower with masking.
- [Large-workspace exploration](warm_prefill_final.json) and
  [earlier kernel exploration](warm_prefill_g1g3.json) were stopped; their
  `.stop.json` files explain why. They are not completed balanced trials.
- [Failed bounded attempt](warm_prefill_bounded.json.error.json) failed in the
  runner before server start; it supplies no performance result.
- [Lossless compression sample](lossless_entropy_probe.json) verified all
  roundtrips, but the tested DEFLATE codec saved only 4.64% and decompressed at
  0.299 GB/s on one thread. It was rejected for the inference path.
- Earlier `grouped_gemm_*micro` and `full_prefill_*identity` files retain the
  compiler/alignment/dispatch development record. Their correctness timings
  are not serving-speed measurements.
- `published_technical.html` is an archived external documentation snapshot,
  not a runtime measurement.

The combined-profile trial is complete and shows a small short-prompt prefill
gain with flat full-answer latency; the selected combination should remain
workload-scoped. See [the full review](../../docs/CODEX_OPTIMIZATION_REVIEW_2026-10-01.md)
for decisions, limitations and reproduction commands. Keep this raw evidence
intact; use new result filenames for repetitions.
