"""CAPACITY-4 — gpt-oss-120b adapter for the generic Neural MoE runtime.

Everything heavy is REUSED from the common runtime (slot pool, host tiers,
prepacked store, admission v4, expert graphs), parameterized by GPTOSS_LAYOUT.
Model-specific content only:

  - exact MXFP4 dequantization (fp4-e2m1 LUT x 2^(E8M0-127); byte-verified
    against transformers' reference converter)
  - the gate/up DE-INTERLEAVE row permutation (checkpoint alternates gate/up
    output rows; the store keeps chunk order [gate | up] so the generic
    runtime's chunk(2) convention holds)
  - MXFP4 -> INT4-g32 requantization (stacked quantization; candidate chosen
    by the pre-registered stage-1 sample gate) + streaming store builder
  - per-expert bias extraction (biases live GPU-resident via
    runtime.attach_expert_biases — outside the slots)
  - core loader that NEVER materializes the 234-GiB bf16 expert tensors
  - the sparse-block installer (transformers' GptOssExperts has the same
    ``experts(hidden, indices, weights)`` seam as Qwen3-Next/OLMoE)
  - naive streaming references for the machinery gate (same-INT4) and the
    quality gate (exact-MXFP4 bf16)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neural.moe.layout import GPTOSS_LAYOUT, GPTOSS_LAYOUT_PS4, gptoss_layout_for_repr, mxfp4_pool_views

GPTOSS_MODEL_ID = "gpt-oss-120b-b5c939de"


def layout_for_store(pstore_root: str):
    """The ExpertLayout a gpt-oss prepacked store directory holds: GPTOSS_LAYOUT (raw scales, weight_repr
    "mxfp4_g32") or GPTOSS_LAYOUT_PS4 (packed 4-bit scale deltas, "mxfp4_g32_ps4"), read from the store's
    metadata.json. A missing/unknown descriptor is an error, never a guess: the two layouts differ in slot size
    and every consumer (pool, CPU kernels, arena, capture buffers) must use the store's."""
    from neural.q80.prepacked_store import Q80PrepackedStore

    desc = Q80PrepackedStore.peek_layout(pstore_root)
    if desc is None:
        raise RuntimeError(f"{pstore_root}: no readable metadata.json (not a prepacked store, or an incomplete one)")
    try:
        lay = gptoss_layout_for_repr(desc.get("weight_repr"))
    except KeyError:
        raise RuntimeError(f"{pstore_root}: weight_repr {desc.get('weight_repr')!r} is not a gpt-oss layout "
                           f"(known: mxfp4_g32, mxfp4_g32_ps4)") from None
    if desc.get("slot_bytes") != lay.slot_bytes:
        raise RuntimeError(f"{pstore_root}: descriptor slot_bytes {desc.get('slot_bytes')} != "
                           f"{lay.slot_bytes} for weight_repr {lay.weight_repr}")
    return lay

# fp4 e2m1 value table (transformers.integrations.mxfp4.FP4_VALUES)
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---------------------------------------------------------------------------
# exact MXFP4 dequant + row de-interleave
# ---------------------------------------------------------------------------

def mxfp4_dequant_nk(blocks, scales, dtype=None):
    """[N, K/32, 16] u8 blocks + [N, K/32] u8 scales -> [N, K] float.

    Exact port of transformers' _convert_moe_packed_tensors WITHOUT the final
    transpose: low nibble -> even positions, high nibble -> odd, value =
    lut[code] * 2^(scale-127). fp32 by default (reference-grade)."""
    import torch

    dtype = dtype or torch.float32
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    n, g, b = blocks.shape
    exp = (scales.to(torch.int32) - 127).reshape(n, g, 1)
    blk = blocks.reshape(n, g, b)
    out = torch.empty(n, g, b * 2, dtype=dtype, device=blocks.device)
    out[..., 0::2] = lut[(blk & 0x0F).to(torch.int)]
    out[..., 1::2] = lut[(blk >> 4).to(torch.int)]
    torch.ldexp(out, exp, out=out)
    return out.reshape(n, g * b * 2)


def deinterleave_perm(n_rows: int):
    """Checkpoint gate_up rows alternate gate/up; return the permutation that
    reorders them to [gate | up] chunk layout."""
    import torch

    return torch.cat([torch.arange(0, n_rows, 2), torch.arange(1, n_rows, 2)])


# ---------------------------------------------------------------------------
# checkpoint access
# ---------------------------------------------------------------------------

class GptOssCheckpoint:
    """Read-only view over the sharded MXFP4 checkpoint (mmap via safetensors)."""

    def __init__(self, ckpt_dir: str) -> None:
        self.dir = Path(ckpt_dir)
        self.weight_map = json.loads(
            (self.dir / "model.safetensors.index.json").read_text("utf-8")
        )["weight_map"]
        self._handles: dict[str, Any] = {}

    def tensor(self, name: str):
        from safetensors import safe_open

        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.dir / shard),
                                             framework="pt", device="cpu")
        return self._handles[shard].get_tensor(name)

    def close(self) -> None:
        self._handles.clear()

    def expert_mxfp4(self, li: int, e: int, which: str):
        """blocks [N, 90, 16] u8, scales [N, 90] u8 for one expert."""
        base = f"model.layers.{li}.mlp.experts.{which}"
        return (self.tensor(f"{base}_blocks")[e], self.tensor(f"{base}_scales")[e])

    def expert_bf16_exact(self, li: int, e: int, device):
        """Exact dequant of one expert on `device`, de-interleaved:
        (gate_up [5760, 2880], down [2880, 2880], b_gu [5760], b_dn [2880]),
        bf16 weights (the dtype the reference forward consumes)."""
        import torch

        L = GPTOSS_LAYOUT
        gb, gs = self.expert_mxfp4(li, e, "gate_up_proj")
        db, ds = self.expert_mxfp4(li, e, "down_proj")
        perm = deinterleave_perm(L.gate_up_n).to(device)
        gu = mxfp4_dequant_nk(gb.to(device), gs.to(device)).index_select(0, perm)
        dn = mxfp4_dequant_nk(db.to(device), ds.to(device))
        b_gu = self.tensor(
            f"model.layers.{li}.mlp.experts.gate_up_proj_bias")[e].to(device)
        b_dn = self.tensor(
            f"model.layers.{li}.mlp.experts.down_proj_bias")[e].to(device)
        b_gu = b_gu.index_select(0, perm)
        return (gu.to(torch.bfloat16), dn.to(torch.bfloat16),
                b_gu.to(torch.bfloat16), b_dn.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# store builder: MXFP4 shards -> prepacked INT4 GEMM-ready rows (streaming)
# ---------------------------------------------------------------------------

def build_gptoss_prepacked_store(ckpt_dir: str, root: str, *,
                                 progress=None) -> dict[str, Any]:
    """CODE-PRESERVING repack (the stage-1 gate rejected INT4 requant):
    store rows carry the source fp4 codes + E8M0 scales byte-exactly; the
    only transform is the gate/up row DE-INTERLEAVE (a permutation of
    independent output rows — numerically exact). Streaming, bounded RAM
    (one layer of tensors + one slot buffer), zero temp disk, no GPU."""
    import time

    import torch

    from neural.q80.prepacked_store import Q80PrepackedStore

    L = GPTOSS_LAYOUT
    ck = GptOssCheckpoint(ckpt_dir)
    rootp = Path(root)
    rootp.mkdir(parents=True, exist_ok=True)
    perm = deinterleave_perm(L.gate_up_n)
    buf = torch.empty(L.slot_bytes, dtype=torch.uint8)
    bias_gu = torch.empty(L.n_layers * L.n_experts, L.gate_up_n,
                          dtype=torch.bfloat16)
    bias_dn = torch.empty(L.n_layers * L.n_experts, L.down_n,
                          dtype=torch.bfloat16)
    cache: dict[str, Any] = {}

    def layer_tensors(li: int):
        if cache.get("li") != li:
            base = f"model.layers.{li}.mlp.experts."
            cache.clear()
            cache["li"] = li
            cache["gb"] = ck.tensor(base + "gate_up_proj_blocks")
            cache["gs"] = ck.tensor(base + "gate_up_proj_scales")
            cache["db"] = ck.tensor(base + "down_proj_blocks")
            cache["ds"] = ck.tensor(base + "down_proj_scales")
            bias_gu[li * L.n_experts:(li + 1) * L.n_experts] = (
                ck.tensor(base + "gate_up_proj_bias")
                .index_select(1, perm).to(torch.bfloat16))
            bias_dn[li * L.n_experts:(li + 1) * L.n_experts] = (
                ck.tensor(base + "down_proj_bias").to(torch.bfloat16))
        return cache["gb"], cache["gs"], cache["db"], cache["ds"]

    def builder(li: int, e: int):
        gb, gs, db, ds = layer_tensors(li)
        o = 0
        for t in (gb[e].index_select(0, perm), db[e],
                  gs[e].index_select(0, perm), ds[e]):
            b = t.contiguous().view(-1)
            buf[o:o + b.numel()].copy_(b)
            o += b.numel()
        assert o == L.slot_bytes
        return buf

    t0 = time.time()
    meta = Q80PrepackedStore.build(root, builder, model_id=GPTOSS_MODEL_ID,
                                   layout=L, progress=progress)
    torch.save({"bias_gu": bias_gu, "bias_dn": bias_dn},
               rootp / "expert_biases.pt")
    meta["build_seconds_total"] = round(time.time() - t0, 1)
    meta["conversion"] = "byte-exact repack (de-interleave only); NO requantization"
    ck.close()
    return meta


# ---------------------------------------------------------------------------
# core loader — no expert materialization, ever
# ---------------------------------------------------------------------------

def load_gptoss_core(ckpt_dir: str):
    """GptOssForCausalLM skeleton on meta; materialize ONLY non-expert tensors
    (~4.3 GiB bf16) on CPU; expert params stay 0-byte stubs. Quantization
    config stripped (the MXFP4 payload is consumed by the store builder, not
    by transformers)."""
    import torch
    from transformers import GptOssConfig, GptOssForCausalLM
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

    cfg = GptOssConfig.from_pretrained(ckpt_dir)
    if hasattr(cfg, "quantization_config"):
        cfg.quantization_config = None
    cfg._attn_implementation = "eager"      # sinks-exact reference path
    with torch.device("meta"):
        model = GptOssForCausalLM(cfg)
    model.eval()

    ck = GptOssCheckpoint(ckpt_dir)
    for name, param in list(model.named_parameters()):
        mod = model.get_submodule(name.rsplit(".", 1)[0])
        leaf = name.rsplit(".", 1)[1]
        if ".mlp.experts." in name:
            mod._parameters[leaf] = torch.nn.Parameter(
                torch.empty(0), requires_grad=False)
            continue
        t = ck.tensor(name).to(torch.bfloat16)
        mod._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
    # buffers computed at init live on meta -> rebuild rotary on CPU
    model.model.rotary_emb = GptOssRotaryEmbedding(cfg, device="cpu")
    ck.close()
    for name, p in model.named_parameters():
        if ".mlp.experts." not in name:
            assert p.device.type != "meta", name
    return model, cfg


def place_core_on_gpu(model, device) -> None:
    """Move every non-expert module to GPU (expert stubs are already 0-byte)."""
    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.lm_head.to(device)
    model.model.rotary_emb.to(device)
    for layer in model.model.layers:
        layer.self_attn.to(device)
        layer.input_layernorm.to(device)
        layer.post_attention_layernorm.to(device)
        layer.mlp.router.to(device)


# ---------------------------------------------------------------------------
# runtime installer (sibling of install_q80/olmoe_expert_runtime)
# ---------------------------------------------------------------------------

def install_gptoss_expert_runtime(runtime, model) -> list:
    installed = []
    layers = model.model.layers

    def make(layer_id: int):
        def forward(hidden_states, router_indices=None, routing_weights=None):
            if hidden_states.shape[0] == 1:
                return runtime.forward_decode(layer_id, hidden_states,
                                              router_indices, routing_weights)
            return runtime.forward_prefill(layer_id, hidden_states,
                                           router_indices, routing_weights)
        return forward

    for li in range(GPTOSS_LAYOUT.n_layers):
        experts = layers[li].mlp.experts
        installed.append((experts, experts.forward))
        experts.forward = make(li)
    return installed


def build_gptoss_neural(model, device, pstore_root: str, *,
                        budget_gib: float = 6.0, pinned_gib: float = 2.0,
                        pageable_gib: float = 2.0,
                        admission_v4: bool = True, layout=None) -> dict[str, Any]:
    """Assemble the gpt-oss Neural path from the COMMON runtime. `layout` None = whatever the store holds
    (layout_for_store: raw or packed scales); the runtime's slot pool, views and kernels follow it."""
    import torch

    from neural.q80.expert_runtime import (
        Q80SlotPoolRuntimeV2,
        enable_admission_v4,
    )
    from neural.q80.hot_path import make_host_pool_v3
    from neural.q80.prepacked_store import Q80PrepackedStore, attach_prepacked

    L = layout if layout is not None else layout_for_store(pstore_root)
    pstore = Q80PrepackedStore(pstore_root, layout=L)
    st = pstore.open(expect_model_id=GPTOSS_MODEL_ID)
    assert st["status"] == "ok", st
    host = make_host_pool_v3(None, device, pinned_gib, pageable_gib, layout=L)
    rt = Q80SlotPoolRuntimeV2(host, device, budget_gib=budget_gib, layout=L)
    attach_prepacked(host, pstore)
    host._prepacked_active = True
    biases = torch.load(Path(pstore_root) / "expert_biases.pt",
                        weights_only=True)
    rt.attach_expert_biases(biases["bias_gu"], biases["bias_dn"])
    if admission_v4:
        enable_admission_v4(rt)
        rt.admission_v4 = True
    rt.cold_io_decode = False
    installed = install_gptoss_expert_runtime(rt, model)
    place_core_on_gpu(model, device)
    return {"runtime": rt, "host": host, "pstore": pstore,
            "installed": installed, "layout": L}


# ---------------------------------------------------------------------------
# naive streaming references (machinery STRICT anchor / quality reference)
# ---------------------------------------------------------------------------

class StreamingStoreReference:
    """Machinery-gate reference: SAME store rows and SAME GEMM kernels as the
    runtime, but computed with the naive eager path — no LRU, no graphs, no
    admission, no host pools. Each call H2Ds the needed rows straight from
    the mmap store. Slow by design."""

    def __init__(self, pstore, biases, device) -> None:
        import torch

        L = pstore.layout or GPTOSS_LAYOUT          # the layout the store was opened with (raw or packed scales)
        assert L.is_mxfp4, L.weight_repr
        self.L = L
        self.pstore = pstore
        self.device = device
        self.bias_gu = biases["bias_gu"].to(device)
        self.bias_dn = biases["bias_dn"].to(device)
        self._row = torch.empty(L.slot_bytes, dtype=torch.uint8,
                                pin_memory=True)
        self._slot = torch.empty(1, L.slot_bytes, dtype=torch.uint8,
                                 device=device)
        (self.gate_blocks, self.down_blocks,
         self.gate_scales, self.down_scales) = mxfp4_pool_views(self._slot, L)
        self._zero = torch.zeros(1, dtype=torch.long, device=device)
        from neural.q80.expert_runtime import resolve_activation
        self._act = resolve_activation(L)

    def _load(self, layer: int, e: int) -> None:
        self.pstore.read_into(layer, e, self._row)
        self._slot[0].copy_(self._row)

    def forward(self, layer, x, top_k_index, top_k_weights):
        import torch

        from neural.q80.mxfp4_kernels import mxfp4_gemm, mxfp4_gemv

        L = self.L
        rows = top_k_index.tolist()
        per_expert: dict[int, list[tuple[int, int]]] = {}
        for t, ids in enumerate(rows):
            for j, e in enumerate(ids):
                per_expert.setdefault(e, []).append((t, j))
        final = torch.zeros_like(x)
        for e in sorted(per_expert):
            self._load(layer, e)
            tj = per_expert[e]
            t_idx = torch.tensor([t for t, _ in tj], dtype=torch.long,
                                 device=x.device)
            j_idx = torch.tensor([j for _, j in tj], dtype=torch.long,
                                 device=x.device)
            state = x.index_select(0, t_idx)
            g = layer * L.n_experts + e
            # decode (full-call T==1) uses the SAME GEMV kernels/config as
            # the runtime so the STRICT anchor compares identical arithmetic.
            # NOTE: keyed on x (the full call), NOT the per-expert group -
            # prefill groups of one token still use the GEMM kernel, like the
            # runtime's forward_prefill.
            if x.shape[0] == 1:
                gate_up = mxfp4_gemv(state, self.gate_blocks, self.gate_scales,
                                     self._zero, block_n=32, block_g=4,
                                     num_warps=8).view(1, L.gate_up_n)
            else:
                gate_up = mxfp4_gemm(state, self.gate_blocks, self.gate_scales,
                                     self._zero).view(state.shape[0], L.gate_up_n)
            gate_up = gate_up + self.bias_gu[g]
            h = self._act(gate_up)
            if x.shape[0] == 1:
                out = mxfp4_gemv(h, self.down_blocks, self.down_scales,
                                 self._zero, per_expert_x=True, block_n=32,
                                 block_g=4, num_warps=8).view(1, L.down_n)
            else:
                out = mxfp4_gemm(h, self.down_blocks, self.down_scales,
                                 self._zero).view(state.shape[0], L.down_n)
            out = out + self.bias_dn[g]
            w = top_k_weights[t_idx, j_idx, None].to(out.dtype)
            final.index_add_(0, t_idx, (out * w).to(final.dtype))
        return final


class StreamingMxfp4Reference:
    """Quality-gate reference: experts computed from the EXACT MXFP4 dequant
    in bf16 (model-exact semantics; dequant itself is lossless). Streams from
    the source checkpoint per call."""

    def __init__(self, ckpt_dir: str, device) -> None:
        self.ck = GptOssCheckpoint(ckpt_dir)
        self.device = device
        self.L = GPTOSS_LAYOUT
        from neural.q80.expert_runtime import resolve_activation
        self._act = resolve_activation(self.L)

    def forward(self, layer, x, top_k_index, top_k_weights):
        import torch

        L = self.L
        rows = top_k_index.tolist()
        per_expert: dict[int, list[tuple[int, int]]] = {}
        for t, ids in enumerate(rows):
            for j, e in enumerate(ids):
                per_expert.setdefault(e, []).append((t, j))
        final = torch.zeros_like(x)
        for e in sorted(per_expert):
            gu_w, dn_w, b_gu, b_dn = self.ck.expert_bf16_exact(
                layer, e, self.device)
            tj = per_expert[e]
            t_idx = torch.tensor([t for t, _ in tj], dtype=torch.long,
                                 device=x.device)
            j_idx = torch.tensor([j for _, j in tj], dtype=torch.long,
                                 device=x.device)
            state = x.index_select(0, t_idx)
            gate_up = state @ gu_w.t() + b_gu
            h = self._act(gate_up)
            out = h @ dn_w.t() + b_dn
            w = top_k_weights[t_idx, j_idx, None].to(out.dtype)
            final.index_add_(0, t_idx, (out * w).to(final.dtype))
        return final
