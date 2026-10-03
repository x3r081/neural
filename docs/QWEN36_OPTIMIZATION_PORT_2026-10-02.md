# Exact-preserving runtime optimization port — 2026-10-02

This port changes how the existing Neural Qwen3.6 path schedules and reuses
work. It does not change the model revision, checkpoint weights, or arithmetic
precision: the model remains `Qwen/Qwen3.6-35B-A3B` at
`995ad96eacd98c81ed38be0c5b274b04031597b0`, with original BF16 weights, BF16
attention K/V cache, and FP32 gated-delta recurrent state. No quantization or
weight rewrite was introduced.

## What changed

- **Native BF16 expert kernel:** the C kernel fuses the gate/up dot products
  with the SiLU and multiply phase, then uses a barrier before the down
  projection. It retains the reference BF16 rounding boundaries and switches
  static row assignment to dynamic OpenMP scheduling. Barrier state belongs to
  each call, keeping concurrent teams independent.
- **Reusable dispatch and compute workspaces:** grouped native dispatch reuses
  host staging for hidden states, routes, route weights, and output, plus the
  expert-kernel scratch buffers. Buffers grow only when the observed shape or
  dtype requires it.
- **Safetensors mmap expert-view cache:** selected expert slices are retained
  as references into the read-only store mapping. The workspace does not copy
  expert weights. The numerical-validation run had 7,273 entries (45.76 GB of
  logical tensor lengths), zero workspace copy bytes, and no evictions. Those
  logical lengths are not a claim about resident RAM.
- **Shared recurrent CUDA Graph:** the Torch one-token recurrent GDN operator
  uses one captured graph with fixed staging buffers across compatible linear
  attention layers. The captured operator is unchanged; projections,
  convolution updates, cache writes, expert dispatch, and sampling remain on
  their existing paths. The full model and its growing attention cache are not
  captured.

## Verification evidence

The comparison reference is the immutable runtime snapshot at commit
`7a46a8605e89dfdaf7541107bdc901774648f9ce`, recorded in
`.tools/qwen36-baseline-7a46a86/snapshot_manifest.json`. The reference and
ported candidate used the same checkpoint revision and the same frozen request
cohort.

- `benchmarks/qwen36_20261002/port_focused_tests02.log`: 31 focused tests
  passed (14 upstream PyTorch deprecation warnings).
- `benchmarks/qwen36_20261002/port_kernel_exact_final.json`: the C kernel
  matched its reference exactly in the random multithreaded and single-thread
  model-shape cases and sign/tie-extreme case (zero mismatches and zero maximum
  absolute error); the concurrent reentrant case passed.
- `benchmarks/qwen36_20261002/port_exact_comparison.json`: measured teacher-
  forced comparison passed. Across three prompts, all 75 checked full-vocabulary
  logit positions were bit-identical (maximum absolute difference 0), and the
  80 tensors in each prefill/final cache snapshot matched exactly for all
  three prompts (480 tensor comparisons). The evidence is bounded
  numerical identity, not a general answer-quality score.
- `benchmarks/qwen36_20261002/port_candidate_numerics.json`: the recurrent graph
  captured once, replayed 2,160 times across 30 linear-attention layers, and had
  zero fallbacks in the comparison run.
- `benchmarks/qwen36_20261002/port_generation_identity.json`: all 1,536 tokens
  and all three response texts from the full benchmark matched the fresh
  original-runtime run exactly.
- `benchmarks/qwen36_20261002/port_candidate01/http_validation.json`: all 10
  live API checks passed, including streaming, function calls and consuming a
  tool result. The generated Python function passed five behavior and
  input-preservation cases in the linked coding report.

## Reproduce the Neural-only candidate benchmark

Run one engine at a time. The command below runs only the ported Neural runtime;
it uses the frozen cohort, context 4,096, eight CPU threads, two GPU expert
layers, and a 3 GiB GPU expert budget. Use a fresh name for each run:

```powershell
.\.venv\Scripts\python.exe tools\run_qwen_benchmark.py `
  --engine neural --name neural_port_repro `
  --requests benchmarks\qwen36_20261002\requests_v1.json `
  --expert-backend native --gdn-backend torch `
  --native-gpu-layers 2 --gpu-cache 3 --context 4096 --threads 8 `
  --native-workspace --recurrent-graph
```

The comparison uses short cold conversations with cache prompting disabled and
a 512-token output cap. Expert/file pages may be warm after earlier requests.
Prefix/prompt-cache reuse is not implemented. These checks do not measure long
agent sessions, long-context quality, or completed-answer quality.

## Measured performance

The combined port at `b568a01` improved client throughput from **3.34255 to
4.15979 output tokens/s (+24.45%)**. The same 1,536-token cohort took 459.53
seconds with the original runtime and 369.25 seconds with the port, saving
90.28 seconds. Decode improved from **3.55582 to 4.51103 steps/s (+26.86%)**.
All three cases improved end to end: 158.35 to 117.33 seconds for
`reasoning_short`, 162.64 to 137.95 seconds for `coding_short`, and 138.54 to
113.97 seconds for `coding_context`.

These are weighted aggregates of one fresh, serial three-prompt run per
configuration, not an average of per-prompt rates or a confidence interval.
The earlier original result was 3.36822 output tokens/s. Both original runs
produced identical tokens. No concurrent model or benchmark job ran during
the timed port cohort. Model loading and the eight-token warmup are excluded
from the timed cohort. Each request starts a new conversation, while graph,
expert-view and OS file caches may remain warm.

This is a decode-led gain: combined prompt-processing time was **28.35 seconds
before and 29.36 seconds after**. No prefill improvement is established, and
the combined benchmark does not isolate the contribution of each port.
Further work on prompt-cache reuse and expert execution remains separate.

The existing llama.cpp results remain **8.96071 output tokens/s** with three
GPU expert layers and **8.68651** with two layers. Neural remains slower.
llama.cpp was **not rerun**. The two-GPU-layer run uses its existing, explicitly
labeled `comparison_metadata.json` supplement for precision provenance; raw
run artifacts were not rewritten. The common client clock is the primary
cross-runtime measure: internal prompt/decode timers have different boundaries.
Decode counts output tokens minus the first token supplied by prefill logits.

Raw before/after records are under `port_reference01/` and `port_candidate01/`
in `benchmarks/qwen36_20261002/`. The derived
[`port_performance_comparison.json`](../benchmarks/qwen36_20261002/port_performance_comparison.json)
passed all model-revision, precision, context, prompt-token and cohort gates.
The preserved reference snapshot manifest and runtime hashes are embedded in
the reference run record; candidate source and DLL hashes are in its run
record. The original DLL SHA-256 is `11076a97409428a22765425e7bdeda05091114dec1256665a9d4b4e897b3bf`;
the tested port DLL is `9ef19adae539cd02d3bd25bacb3c297aa10aa4ba41c7c905375524fbbecbfcd9`.

The graph reported **46,200 replays with zero fallbacks** after warmup and the
timed cohort. Workspace reporting showed 8,979 cached expert views, 487,730
hits, zero copied weight bytes and zero evictions. Dispatch scratch used
529,408 bytes and native-kernel scratch 3,612,672 bytes. The graph's retained
input/output/state tensors used 4,227,264 bytes.

Host memory remained tight: minimum available RAM was about 302 MiB in the
reference lifecycle and 80 MiB in the candidate lifecycle. The latter also
includes post-benchmark API/coding validation, so these minima are not an
isolated memory-overhead comparison. Owned-process RSS sums peaked at about
53.79 and 53.94 GiB respectively; shared pages can be counted more than once.
GPU memory at the post-throughput sample was 8,574 versus 8,630 MiB. Both
owned server process trees stopped cleanly after completion.

## Start the measured port

```powershell
.\start_qwen_server.bat --recurrent-graph
```

Native workspace reuse is on by default. The recurrent graph is explicit
because this is still bounded validation; omitting the flag uses eager Torch
recurrence. It requires `--gdn-backend torch`. Rebuild the local DLL with
`tools/build_qwen_cpu.ps1` after pulling source on another checkout. The normal
launcher retains its 16,384-token capacity; the measurements above used 4,096
and short actual prompts, not a full-context workload.
