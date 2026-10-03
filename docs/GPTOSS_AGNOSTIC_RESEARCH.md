# GPT-OSS-based generalization and optimization review — 2026-10-03

## Scope and evidence

The reference is the optimized GPT-OSS runtime at NeuralServer commit
`772399c380274eec4fe00ba081dba7549a1a8a7f`, already present in this repository's
history. Its production checkout and environment are unchanged. The replacement
is developed in NeuralQwen36. No Qwen checkpoint is used for further tuning.

The same downloaded GPT-OSS 120B, original MXFP4 codes, and existing lossless PS4
store are retained. A different checkpoint, lower precision, reduced expert
fanout, altered router, reduced context, or hardware purchase is not a proposed
speedup. The local Strata checkout and current public source were reviewed as
implementation references, not as an equal-quality benchmark competitor.

Labels: **MEASURED** means an archived execution on this host; **CALCULATED** is
metadata arithmetic; **HYPOTHESIS** is an unproven optimization candidate. A
paper's reported speedup is not a Neural measurement.

## What the research actually changes

The core lesson from heterogeneous MoE work is that architecture semantics,
weight representation, execution kernels, and hardware placement need separate
contracts. A generic Transformers wrapper alone does not transfer GPT-OSS's
performance to other models. Conversely, applying its clamped SwiGLU, biased
experts, attention sinks, top-4 routing kernels, or Harmony format to another
architecture would change that model.

The replacement therefore keeps GPT-OSS's native path and adds metadata-only
model/store validation, explicit backend capabilities, ISA-safe kernel dispatch,
and memory planning. Reference adapters preserve the source dtype and upstream
attention/router for other supported MoE architectures. They are functional
extension points, not a claim that all MoEs have a fused implementation.

The AVX2/FMA CPU fallback is a portability result. It preserves the original
MXFP4/PS4 representation and reproduces the existing CPU kernel's arithmetic;
it is not a speed optimization for this AVX-512 host. Its full synthetic gate
compares intermediate scratch, output bits and copied source bytes. The first
implementation failed multi-expert reduction by 1–2 ULP; reproducing the native
rounded-product groups and FMA tail resolved the failure.

## Ideas already implemented or ruled out here

- Layer-major prefill already loads each required expert once per layer across
  prompt chunks. `docs/LEVERS.md` records identical checked logits/KV/rings and
  fewer copies. Calling this a new discovery would be incorrect.
- Persistent CPU workers, grouped prefill, rolling sliding-window KV, exact PS4
  scale packing, prompt-cache/tool splicing, and bounded admission are inherited.
- Earlier zero-copy miss splitting competed with CPU DRAM traffic and regressed
  performance. Copying Strata's PCIe split fraction directly would repeat that
  risk. Bigger prefill chunks previously failed the exactness gate.
- The later experimental split-graph closeout did not clear its quality gate.
  Its headline speed is not the baseline or a production feature in this branch.

## Ranked candidates for further exact-quality work

### 1. Multiple exact-prefix KV snapshots for interleaved coding sessions

**HYPOTHESIS:** generalize the existing active-conversation plus last-side-request
protection into a bounded bank for three or more interleaved conversations. The
native engine already has `--protect-side`, `SIDE_KV`, save/restore and takeover
logic (`server.py`, `tests/test_protect.py`); simple A/B/A protection is not a new
opportunity. This extension targets
time-to-first-token in autonomous coding/tool workflows, not steady-state decode.
The reference adapter implements bounded whole-cache snapshots; the GPT-OSS
engine retains its inherited active/side caches and ring checkpoints.

The native extension must key entries by exact rendered tokens, model/store
identity, position/cache geometry and execution profile. Its 18 sliding-window
rings need complete snapshots, not an assumed `crop()` operation. Divergent
continuations need independent writable state. Cache budgeting must include
copy overlap and the cost of displacing expert pages from RAM. Frozen placement
is required for a strict fresh-computation comparison because CPU/GPU reduction
order can otherwise differ.

Gate: interleave A/B/C/B requests beyond the existing protection policy; require complete KV/ring/state and token equality
against cold recomputation, then measure end-to-end TTFT, copied bytes, RAM/VRAM,
and subsequent decode speed. Reject a TTFT gain that causes a larger paging loss.

### 2. Weightless suffix drafts with an exact GPT-OSS verifier

**HYPOTHESIS:** repeated code spans can draft candidate token runs using an n-gram
or suffix index without another model. Strata has `SuffixDrafter`; the primary
[prompt lookup implementation](https://github.com/apoorvumang/prompt-lookup-decoding)
uses prompt matches as speculative candidates. This is particularly relevant to
edits that copy most of an input file.

The difficult part is the verifier. Neural's canonical decode is a one-token
hybrid graph. Verifying candidates one by one offers no inherent compute speedup;
batched verification can amortize expert reads but changes GEMM/reduction order.
[Draft & Verify](https://arxiv.org/abs/2309.08168) establishes the algorithmic
approach, not bit identity of a new Neural kernel. Require exact greedy-token
parity, correct rollback of every ring/cache, and a separate distribution-correct
accept/reject implementation before supporting sampling. Draft tokens are never
returned without target verification. No approximated expert replaces the target.

Gate: measure candidate coverage/acceptance on real code editing, then total
verification cost. Enable only when accepted tokens per verification cost beat
canonical decode, with a quick fallback on low-overlap text. This is a larger
architectural experiment and is not enabled in the replacement.

### 3. Predictive file-page warming, independent of routing and residency

**HYPOTHESIS:** use a bounded next-layer router lookahead to warm source pages
before naturally occurring disk faults. Strata's `RouterLookahead` and
`expert_source.cpp` provide a concrete implementation reference. Predictions
would affect page residency only: actual top-k routing, weights, execution
device and generated outputs remain authoritative and unchanged.

[HybriMoE](https://arxiv.org/abs/2504.05897) combines workload-aware placement,
prefetch and caching. [SP-MoE](https://arxiv.org/abs/2510.10302) shows why
speculative verification also needs a transfer budget. Neither establishes a
gain for Neural's warm, memory-bandwidth-limited 3080 Ti setup.

Gate: first establish real hard-fault stalls, without manufacturing memory
pressure. Then cap lookahead/wasted bytes, record accurate/wasted page touches,
disk latency and DRAM contention, and compare exact outputs and wall time in
alternating trials. Do not enable it merely because prediction accuracy is high.

### 4. Hardware-specific capacity and scheduling profiles

**IMPLEMENTED FOUNDATION / HYPOTHESIS FOR SPEED:** memory capacity, CPU ISA,
checkpoint geometry, CUDA BF16 capability and native ABI now determine whether
and how a backend can start. A future measured cost model should distinguish
CPU bandwidth, expert transfer cost, graph overhead and prefill reuse on each
host. One transfer-ratio setting is not portable performance tuning.

The current planner deliberately retains the proven pool ceiling by default
and permits an explicit larger pool after capacity checks. It does not claim an
autotuned optimum. A recent [llama.cpp expert-cache RFC](https://github.com/ggml-org/llama.cpp/discussions/28248)
also separates persistent caching from generic layer offload; Neural already has
the equivalent broad concept. The useful next comparison is admission cost and
state stability on matched workloads, not adoption of another headline speedup.

## Conclusion for this implementation

There is no evidence yet of another large, quality-preserving GPT-OSS decode
breakthrough. The delivered change protects the optimized engine while opening
explicit model and hardware adaptation points. New measurements are recorded in
`GPTOSS_AGNOSTIC_VALIDATION.md`; none of the hypotheses above is presented as a
measured speedup or silently enabled.
