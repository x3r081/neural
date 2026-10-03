# Optional Qwen Gated DeltaNet acceleration

The Qwen runtime defaults to the Transformers Torch implementation. An
explicit `gdn_backend="fla"` mode may bind FLA's `chunk_gated_delta_rule` and
`fused_recurrent_gated_delta_rule` to each `Qwen3_5MoeGatedDeltaNet` instance;
the adapter is in `qwen_neural.fast_delta`. Binding is opt-in and reversible.
It does not replace convolutions, cast weights, or alter recurrent-cache
dtypes. HF continues to create and carry the recurrent state (normally FP32).

The optional package set is listed in `requirements-qwen-fast.txt`. It is not
part of the default requirements. `fla-core` is pure Python packaging around
Triton kernels; this does not establish that those kernels work with the
community `triton-windows` build. The current Windows/Triton/torch combination
must be validated independently before selecting FLA. `causal-conv1d` is a
separate optional acceleration and is not installed or selected by this
adapter.

The Qwen project venv now has `fla-core==0.5.2` and `einops==0.8.2` installed
without resolving or changing any shared/base environment dependencies.
Importing both FLA GDN ops succeeds on CPU. This does not validate kernel
execution: the environment uses community `triton-windows` rather than
official upstream Triton support. Keep `gdn_backend="torch"` as the
correctness/reference path until a platform-specific smoke test establishes
numerical agreement for prefill chunks and single-token decode, including
recurrent-cache updates.

For a separate Qwen venv, install only the packages named here with
`python -m pip install --no-deps -r requirements-qwen-fast.txt`; do not add
these optional packages to the shared Neural environment.
