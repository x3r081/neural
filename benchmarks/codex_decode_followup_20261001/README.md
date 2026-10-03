# Decode follow-up evidence index — October 1, 2026

Candidate runtime is `9aa766f`, based on upstream `a5d3c95` plus the first
review's isolated changes. All server variants use that same runtime with
feature flags selecting the original or persistent DLL. Checkpoint, packed
MXFP4 store, expert-selection policy and context limit are unchanged. Runtime
source and DLL stayed frozen throughout performance testing. See the
[follow-up review](../../docs/CODEX_DECODE_FOLLOWUP_2026-10-01.md)
for method and limitations.

## Completed evidence

- [Persistent-team no-op probe](persistent_team_probe.json), its [C source](persistent_team_probe_source.c)
  and [runner](persistent_team_probe_tool.py): synthetic rendezvous test. The
  persistent team reduced measured per-call overhead by about 35–56 μs in the
  tested eight-thread scenarios. This is a candidate lifecycle saving, not inference speed.
- [Persistent decode kernel checks](persistent_decode_ab.json): 432 bitwise
  checks over raw/packed scales, 1–4 experts and 1/2/4/8 threads. Captured bytes,
  scratch outputs, failure injection, restart, concurrent serialization,
  shutdown and floating-point-control checks passed. Synthetic CPU medians
  improved modestly; capture-heavy cases varied.
- [Full-model identity check](full_cpu_decode_identity.json): all comparisons
  equal at 128, 3,000 and 13,000 prompt tokens over 64 decode steps. Full
  per-step logits/hidden states, generated IDs, routing and K/V matched; the
  candidate ran persistent jobs with no fallbacks. Its correctness timings are
  not serving-performance measurements.
- [Corrected graph replay probe](graph_replay_probe_fixed.json): deterministic
  output checks passed (16 positive, 4 negative cases). At an 800 μs host gap,
  the best direct launch path saved about 1.8 μs per graph submission, or
  0.065 ms/token CALCULATED across 36 launches. This is a microbenchmark result;
  no inference improvement is claimed and the raw-handle path was not adopted.
- [Mirrored serving comparison](persistent_cpu_mirrored.json), its
  [request/response archive](persistent_cpu_mirrored.json.replay.json) and
  [audited summary](persistent_cpu_mirrored_summary.json): six fresh sessions,
  stock / persistent CPU / combined / combined / persistent CPU / stock.
  CPU-only weighted decode improved 23.8649 to 24.2807 tokens/s (+1.74%),
  with all corresponding stock/CPU output IDs equal. The slow CPU-only answer
  is retained. The combined variant measured +2.84%, but one session changed
  outputs on its final two answers, so its strict all-session identity gate
  failed. The posthoc shared first-two-answer subset gives CPU +2.99% and
  combined +2.89%. There are two sessions per variant. Grouped prefill did
  not execute in this 13k/518–521-new-token workload.
- [Frozen-placement repeat](persistent_cpu_frozen.json), its
  [archive](persistent_cpu_frozen.json.replay.json) and
  [summary](persistent_cpu_frozen_summary.json): stock / CPU / CPU / stock,
  four 512-token answers each, with `--refresh-every 0` for both variants.
  Weighted decode improved 19.8659 to 20.4391 tokens/s (+2.885%). All 16
  corresponding prompt and output token sequences matched, and enabled
  requests reported persistent work with no fallbacks. The frozen setting is
  a diagnostic control; normal adaptive placement is faster overall.
- [Fresh llama.cpp comparison](persistent_vs_llama.json), its
  [archive](persistent_vs_llama.json.replay.json), and
  [final cross-trial verification](serving_verified.json): opt / llama /
  llama / opt, exact saved API requests. Neural 24.6910 versus installed
  llama.cpp 14.3189 tokens/s weighted across all answers (1.7244x). This
  measures the engines on the workload; the new CPU improvement is the
  separate 1.74% / 2.885% result above. Both Neural repeats' prompt/output
  hashes match. Cross-backend internal token identity and quality equivalence
  are not established. Decode timing uses each backend's reported counts.
- [Selected tests](selected_tests_delivery.log): 103 passed, 23 existing
  warnings. Standalone native/GPU/full-model checks above are separate.

## Failed evidence and scope

- [Initial graph probe](graph_replay_probe.json) failed on a CLI attribute error
  before timing; use the corrected probe above for results.
- The mirrored trial's strict all-session identity gate remains failed for
  the two divergent combined-profile answers. No observations were removed.
- The first native prototype failed its FP-control check before arithmetic
  testing; the corrected implementation samples the reference worker controls.
  That initial JSON was overwritten during development; the final synthetic
  artifact and the explanation in the review are retained.

Commands: [mirrored](run_cpu_mirrored.ps1), [frozen](run_cpu_frozen.ps1),
[llama comparison](run_cpu_vs_llama.ps1). Recompute the final comparison with
`F:\AI\Neural\.venv\Scripts\python.exe benchmarks/codex_decode_followup_20261001/summarize_final.py`.
The mirrored audit is `tools/review_persistent_trial.py` and intentionally
returns failure for its preserved output-identity mismatch.
