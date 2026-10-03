# October 2 research closeout

The validated gpt-oss runtime remains at the October 1 implementation (`7c0ce9c`).
October 2 research is preserved on separate branches; it has not cleared the
task-quality and replicated production-performance gates for promotion.

## Measured results

- One completed split-graph backend cohort: 16 independent requests, 42,639
  generated tokens (including 16 prefill seeds), 1,454.0354679 seconds,
  **29.324594 aggregate tokens/s**.
- Historical combined-graph cohorts with identical complete prompt/output IDs:
  **23.595025** and **23.622681 aggregate tokens/s**. The calculated throughput
  difference is 24.14–24.28%; this is not a contemporary alternating experiment
  or a replicated causal improvement.
- The grouped numerical audit passed 42,623 decode checks and 16 natural finals.
- Final grouped timing was interrupted by a Windows reboot. An identical retry
  failed before generation at the explicit host RAM-reserve guard. The split
  repeat was deferred. Neither attempt supplies a new timing rate.
- The frozen static coding review found paired rubric regressions; generated
  code was not executed in that review. **Task-quality gate: NOT_CLEARED.**

These backend cohort clocks include prefill and K/V copying, but exclude startup,
graph capture, diagnostics and HTTP. They are not single-conversation decode
rates. The earlier approximate 11% comparison used different outputs and does not
establish a quality-equivalent production gain. Model and precision were unchanged.

## Preserved evidence

- [Completed timing and quality evidence](https://github.com/x3r081/NeuralServer/tree/5f3ac01644beb8136c497b099752c839b3f578e2/benchmarks)
- [Final audit and incomplete timing archive](https://github.com/x3r081/NeuralServer/blob/fed0590dbc6eff2e9b061de49cf346df0952b566/benchmarks/streamed_final_evidence_20261002/archive_manifest.json)
- [Published chart and method](https://x3r081.github.io/neural-site/#oct-2-research)

All 52 committed raw files in the final archive were checked against their
manifest SHA-256 values. Research was paused at the user's request. New Qwen3.6
runtime development is to be isolated in a separate repository rather than
changing this gpt-oss runtime.
