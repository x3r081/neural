# llama.cpp Qwen3.6 BF16 baseline setup

This records the completed llama.cpp setup, original-checkpoint conversion,
payload verification, and initial BF16 inference run for the Qwen3.6 comparison.
The model is the official `Qwen/Qwen3.6-35B-A3B` revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`.

## Pinned upstream build

The stable GitHub release API pointed to `v0.5.0`, which publishes only a
nightly-tag marker. On 2026-10-02 the newest official Windows CUDA prerelease
was `b11349`, published at 2026-10-02 15:41:03 UTC. Source and prebuilt binary
are pinned to commit
`fb4b2737a808a3fb7c2117a498f43815dc9be53e`.

The setup script clones that tag into `.tools/llama.cpp`, downloads the
official `win-cuda-12.4-x64` binary and its CUDA runtime companion, verifies
each against the digest advertised by the official GitHub release API and
against the checked-in expected SHA-256, extracts both into
`.tools/llama-b11349-win-cuda-12.4`, then checks the CLI build version.

| Official asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `llama-b11349-bin-win-cuda-12.4-x64.zip` | 263288146 | `1084a0a4c6567511c7f67d8c4db71979f837170ed22b2fcb72c0d8637a4eaf14` |
| `cudart-llama-bin-win-cuda-12.4-x64.zip` | 391443627 | `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |

The setup has been completed. To check the existing installation without
downloading assets again, run:

```powershell
.\tools\setup_llama_qwen.ps1
.\tools\setup_llama_qwen.ps1 -VerifyOnly
```

The manifest at `.tools/llama-qwen-setup.json` records release provenance,
paths, binary version output, and asset hashes. The host has CUDA Toolkit 12.8
and NVIDIA driver 610.62; the CUDA 12.4 Windows bundle is selected for its
compatibility with that newer driver. This setup leaves the existing
`G:\Tools\llama-b10361` untouched.

The pinned converter registers `Qwen3_5MoeForConditionalGeneration` and
`Qwen3_5MoeForCausalLM` for the `qwen35moe` architecture. The official
Qwen3.6 config uses `model_type: qwen3_5_moe`; the pinned source has the
required architecture mapping. The installation manifest is
`benchmarks/qwen36_20261002/llama_setup.json`.

## Converter environment and command

The converter requirements at this pin are declared in
`.tools/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt`; they
include the CPU PyTorch 2.11.0 wheel, Transformers 4.57.6, NumPy 2.2.x,
SentencePiece, `gguf`, and protobuf. They are not installed into the legacy
`F:\AI\Neural\.venv` by this setup. Use the new project `.venv` after it has
been created, or a dedicated isolated converter venv, and do not upgrade the
legacy environment. Conversion succeeded using the project `.venv` with
Torch 2.10.0 and Transformers 5.14.1; no converter dependency upgrade was
needed. Do not install the pinned requirements wholesale unless a future
reproduction actually needs an unavailable import or incompatible dependency.
If needed, install only into the isolated project environment:

```powershell
F:\AI\NeuralQwen36\.venv\Scripts\python.exe -m pip install `
  -r F:\AI\NeuralQwen36\.tools\llama.cpp\requirements\requirements-convert_hf_to_gguf.txt
```

The conversion used this unquantized BF16 command (the verified output already
exists at the target path; rerunning it is a full conversion job):

```powershell
F:\AI\NeuralQwen36\.venv\Scripts\python.exe `
  F:\AI\NeuralQwen36\.tools\llama.cpp\convert_hf_to_gguf.py `
  F:\Models\Qwen3.6-35B-A3B `
  --outfile F:\Models\Qwen3.6-35B-A3B-bf16.gguf `
  --outtype bf16 `
  --no-nextn
```

`--no-nextn` matters: pinned llama.cpp exports the checkpoint's MTP block by
default. This source has 19 MTP tensors and one configured MTP layer; omit them
for a matched base-model comparison. TextModel conversion skips vision/audio
tensors automatically. The expected text-only plan is 733 GGUF tensors from
693 non-MTP text source tensors; the gate/up expert split increases the output
tensor count. A header-only source inventory and meta-tensor conversion plan
were checked without reading any weight payloads.

This wrote a second model copy of 69,376,637,440 bytes (~64.6 GiB). Both source
and output were on F:. BF16 conversion retains the requested weight precision
but reorders/adapts tensors for GGUF/runtime layout; do not describe the whole
file as raw source-byte identical to the source checkpoint.
At the pinned revision, the converter widens source BF16 to FP32 before the
architecture transforms, then writes only BF16 or F32 tensors with `--outtype
bf16`. It reorders linear-attention V heads; squeezes/reorders the conv kernel;
splits aggregated expert gate/up tensors; renames `dt_bias` to `dt_proj.bias`;
maps `A_log` to `-exp(A_log)`; and adds 1 to norm weights other than
`linear_attn.norm.weight`. One-dimensional tensors, `_norm.weight`, recurrent
convolution tensors, and several special tensors are deliberately F32. These
are converter/runtime adaptations, not quantization. Do not run `llama-quantize`
for this baseline.

After conversion, the GGUF payload was compared against the source after
applying those exact pinned transforms. The verifier streamed source tensors
and rejected any GGUF type other than BF16/F32; it did not compute whole-file
weight hashes. The completed evidence is in
`benchmarks/qwen36_20261002/gguf_verification.json` (status
`PASS_EXACT_BF16_OR_F32_PAYLOADS`):

```powershell
F:\AI\NeuralQwen36\.venv\Scripts\python.exe `
  F:\AI\NeuralQwen36\tools\verify_qwen_gguf.py `
  --model-dir F:\Models\Qwen3.6-35B-A3B `
  --gguf F:\Models\Qwen3.6-35B-A3B-bf16.gguf `
  --report F:\AI\NeuralQwen36\benchmarks\qwen36_20261002\gguf_verification.json
```

The verifier's `--plan-only` mode uses SafeTensors headers and meta tensors
only; it can be used to inspect a future converter revision without scanning
weight payloads:

```powershell
F:\AI\NeuralQwen36\.venv\Scripts\python.exe `
  F:\AI\NeuralQwen36\tools\verify_qwen_gguf.py `
  --model-dir F:\Models\Qwen3.6-35B-A3B --plan-only `
  --report F:\AI\NeuralQwen36\benchmarks\qwen36_20261002\qwen36_converter_plan.json
```

## Initial inference configuration and resource constraints

The official config describes 40 hybrid layers: a full-attention block every
fourth layer and linear/recurrent blocks in between, with 256 experts and top-8
routing. The recurrent state has per-layer fixed storage, while full-attention
KV grows with context. Use BF16 K/V caches on llama.cpp if matching the HF
BF16 runtime cache, and record the actual recurrent-state policy independently.

The initial llama.cpp run used the RTX 3080 Ti (12 GiB VRAM), 64 GiB system RAM,
and these settings: `-ngl 99 -ncmoe 38 -t 8 -tb 8 -c 4096 -np 1 -b 512
-ub 128 -fa on -ctk bf16 -ctv bf16 --jinja`. Thus it requested CPU placement
for 38 MoE layers and GPU placement for the remaining MoE layers while keeping
the non-MoE core on CUDA. The exact invocation and captured resources are in
`benchmarks/qwen36_20261002/llama_bf16_01/run.json`; startup output is in that
run's `server.log`.

The process reached a minimum of 241,766,400 bytes available system RAM
(~0.225 GiB) and a maximum recorded RSS of 58,078,511,104 bytes (~54.1 GiB).
This is a near-exhausted-memory run. The log warns that CPU tensor overrides
are used with mmap and suggests considering `--load-mode none`; treat that as
a separate experiment, not an automatic optimization. Disabling mmap can make
more of the ~64.6 GiB GGUF resident at once and leaves little headroom on a
64 GiB machine. Do not claim this placement has safe RAM margin. The active 3B
parameter count is not the total loaded/mapped model size.

For the matched throughput workload, the harness renders frozen prompts with
the official HF tokenizer/chat template and replays the same serialized prompt
bytes through llama.cpp's completion endpoint. It checks prompt token IDs and
evaluated prompt-token count before recording timings. Generation is capped at
512 tokens per request, temperature is zero, and cache prompting is disabled;
this is a throughput workload, not a coding-quality or full-context evaluation.
Report each engine's own generated token count and throughput. Do not assume
generated token IDs or answer quality match across backends. The initial run is
recorded in `benchmarks/qwen36_20261002/llama_bf16_01/`; the dated validation
report owns the current matched-results status.

## Upstream references

The completed additional placement run used `-ncmoe 37` (three final expert
layers on GPU) with all other benchmark settings unchanged. It measured 8.961
output tokens/s by the common client wall clock and 10.467 subsequent decode
steps/s by llama.cpp's timer. Its owned-process-tree RSS peaked at
57,922,285,568 bytes, and minimum available system RAM was 306,827,264 bytes.
See `benchmarks/qwen36_20261002/llama_bf16_02_ncmoe37/` and the
[final validation record](QWEN36_VALIDATION_2026-10-02.md). No further placement
search is implied by these two tested configurations.

- [Official releases API](https://api.github.com/repos/ggml-org/llama.cpp/releases)
- [Pinned b11349 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11349)
- [Pinned Qwen converter registration](https://github.com/ggml-org/llama.cpp/blob/fb4b2737a808a3fb7c2117a498f43815dc9be53e/conversion/qwen.py)
- [Pinned converter requirements](https://github.com/ggml-org/llama.cpp/blob/fb4b2737a808a3fb7c2117a498f43815dc9be53e/requirements/requirements-convert_hf_to_gguf.txt)
- [Official Qwen3.6 config at the requested revision](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json)
