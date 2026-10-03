"""OVERNIGHT-3 — per-layer CUDA graphs for the gpt-oss core (decode T=1).

Each layer's [input_norm -> qkv(+bias) -> rope -> static-KV write ->
sinks-attention over a fixed-length window -> o_proj -> residual ->
post_norm -> router topk/softmax] is captured as ONE CUDA graph; a tail
graph covers [final_norm -> lm_head]. The token position lives in a device
tensor consumed inside the graphs (one 8-byte H2D per token). Expert
compute stays on the existing v4 expert-graph path between layer replays.

Numerics: op-for-op replication of transformers' eager gpt-oss forward,
EXCEPT attention reduces over a fixed S_MAX-length static KV with an
additive mask (-inf beyond the causal/sliding window; padded V rows are
zeros). Padded columns contribute exactly zero probability; the reduction
ORDER differs from variable-length eager, so decode outputs differ from
the ungraphed path at fp-summation-order scale (pre-declared in
reports/overnight3_gate.json). Prefill is untouched (HF eager).
"""

from __future__ import annotations

import torch

from neural.moe.layout import GPTOSS_LAYOUT as L


class _LayerGraph:
    def __init__(self, layer, pos_dev, smax: int, dev) -> None:
        self.layer = layer
        self.pos = pos_dev                       # int64 [1], shared
        self.smax = smax
        attn = layer.self_attn
        self.sliding = attn.sliding_window       # None for full layers
        self.n_kv = attn.k_proj.out_features // attn.head_dim
        self.n_q = attn.q_proj.out_features // attn.head_dim
        self.hd = attn.head_dim
        self.scaling = attn.scaling
        self.k = torch.zeros(1, self.n_kv, smax, self.hd,
                             dtype=torch.bfloat16, device=dev)
        self.v = torch.zeros_like(self.k)
        self.x_in = torch.zeros(1, L.hidden, dtype=torch.bfloat16, device=dev)
        self.h_mid = torch.zeros_like(self.x_in)     # residual + attn_out
        self.h_norm = torch.zeros_like(self.x_in)    # post_norm out (expert in)
        self.r_scores = torch.zeros(1, L.top_k, dtype=torch.bfloat16, device=dev)
        self.r_idx = torch.zeros(1, L.top_k, dtype=torch.int64, device=dev)
        self.ar = torch.arange(smax, device=dev)
        self.graph = None

    def _rms(self, norm, x):
        dt = x.dtype
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + norm.variance_epsilon)
        return (norm.weight * xf).to(dt)

    def _compute(self, rotary_inv_freq, attention_scaling) -> None:
        import torch.nn.functional as F

        a = self.layer.self_attn
        x = self.x_in
        h = self._rms(self.layer.input_layernorm, x)
        q = F.linear(h, a.q_proj.weight, a.q_proj.bias).view(
            1, 1, self.n_q, self.hd).transpose(1, 2)
        k = F.linear(h, a.k_proj.weight, a.k_proj.bias).view(
            1, 1, self.n_kv, self.hd).transpose(1, 2)
        v = F.linear(h, a.v_proj.weight, a.v_proj.bias).view(
            1, 1, self.n_kv, self.hd).transpose(1, 2)
        # rope (fp32, HF op order: cos/sin scaled then bf16 cast)
        pos_f = self.pos.to(torch.float32).view(1, 1)
        freqs = rotary_inv_freq.view(-1, 1).float() @ pos_f      # [hd/2, 1]
        emb = freqs.t()                                          # [1, hd/2]
        cos = (emb.cos() * attention_scaling).to(q.dtype)
        sin = (emb.sin() * attention_scaling).to(q.dtype)
        q1, q2 = q.chunk(2, dim=-1)
        q = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        k = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
        # static KV write at pos
        self.k.index_copy_(2, self.pos, k)
        self.v.index_copy_(2, self.pos, v)
        # sinks attention over smax with additive mask — GROUPED (no repeat_kv
        # materialization: [1,64,smax,64] expand/reshape would pin ~34 MB per
        # graph pool and spill VRAM across 36 graphs)
        rep = self.n_q // self.n_kv
        qg = q.view(1, self.n_kv, rep, self.hd)                    # [1,kv,rep,hd]
        attn = torch.matmul(qg, self.k.transpose(2, 3)) * self.scaling
        # [1,kv,rep,smax]
        ok = self.ar <= self.pos
        if self.sliding is not None:
            ok = ok & (self.ar > self.pos - self.sliding)
        mask = torch.where(ok, 0.0, float("-inf")).to(attn.dtype).view(
            1, 1, 1, self.smax)
        attn = attn + mask
        sinks = a.sinks.reshape(1, self.n_kv, rep, 1)
        combined = torch.cat([attn, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        out = torch.matmul(probs[..., :-1], self.v)                # [1,kv,rep,hd]
        out = out.reshape(1, -1)
        out = F.linear(out, a.o_proj.weight, a.o_proj.bias)
        h_mid = x + out
        h_norm = self._rms(self.layer.post_attention_layernorm, h_mid)
        r = self.layer.mlp.router
        logits = F.linear(h_norm, r.weight, r.bias)
        top_v, top_i = torch.topk(logits, L.top_k, dim=-1)
        scores = F.softmax(top_v, dim=1, dtype=top_v.dtype)
        self.h_mid.copy_(h_mid)
        self.h_norm.copy_(h_norm)
        self.r_scores.copy_(scores)
        self.r_idx.copy_(top_i)

    def capture(self, rotary_inv_freq, attention_scaling, pool=None) -> None:
        self._compute(rotary_inv_freq, attention_scaling)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self._compute(rotary_inv_freq, attention_scaling)


class GptOssCoreGraphs:
    """Graphed decode loop: embed (CPU) -> 36 layer replays with the expert
    path between -> tail graph (final norm + lm_head)."""

    def __init__(self, model, runtime, dev, smax: int = 2048) -> None:
        self.model = model
        self.rt = runtime
        self.dev = dev
        self.smax = smax
        self.pos_pin = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self.pos = torch.zeros(1, dtype=torch.int64, device=dev)
        rot = model.model.rotary_emb
        self.inv_freq = rot.inv_freq.to(dev)
        self.att_scale = rot.attention_scaling
        self.layers = [_LayerGraph(model.model.layers[li], self.pos, smax, dev)
                       for li in range(L.n_layers)]
        # tail: final norm + lm_head
        self.t_in = torch.zeros(1, L.hidden, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(1, model.lm_head.out_features,
                                  dtype=torch.bfloat16, device=dev)
        self.t_graph = None

    def _tail_compute(self):
        import torch.nn.functional as F

        m = self.model
        h = self.layers[0]._rms(m.model.norm, self.t_in)
        self.logits.copy_(F.linear(h, m.lm_head.weight))

    def capture(self) -> None:
        # ONE shared memory pool: replays are strictly sequential, so all 36
        # layer graphs + tail can reuse the same intermediate arena instead of
        # pinning ~per-graph private pools (the VRAM blowup found on first try)
        pool = torch.cuda.graph_pool_handle()
        for lg in self.layers:
            lg.capture(self.inv_freq, self.att_scale, pool=pool)
        self._tail_compute()
        torch.cuda.synchronize()
        self.t_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.t_graph, pool=pool):
            self._tail_compute()

    def load_prefill_kv(self, past, prefill_len: int) -> None:
        """Copy HF DynamicCache contents into the static buffers at absolute
        positions (sliding layers may retain only the last `window` entries)."""
        for li, lg in enumerate(self.layers):
            kl = past.layers[li].keys if hasattr(past, "layers") else past[li][0]
            vl = past.layers[li].values if hasattr(past, "layers") else past[li][1]
            n = kl.shape[2]
            start = prefill_len - n
            lg.k.zero_()
            lg.v.zero_()
            lg.k[:, :, start:prefill_len] = kl.to(lg.k.dtype)
            lg.v[:, :, start:prefill_len] = vl.to(lg.v.dtype)

    @torch.inference_mode()
    def decode_token(self, token_id: int, position: int) -> int:
        m = self.model
        self.pos_pin[0] = position
        self.pos.copy_(self.pos_pin, non_blocking=True)
        tid = torch.tensor([[token_id]])
        h = m.model.embed_tokens.forward(tid)
        if h.device != self.pos.device:               # CPU-resident embed
            h = h.to(self.pos.device, torch.bfloat16)
        h = h.view(1, L.hidden)
        for li, lg in enumerate(self.layers):
            lg.x_in.copy_(h)
            lg.graph.replay()
            eo = self.rt.forward_decode(li, lg.h_norm, lg.r_idx, lg.r_scores)
            h = lg.h_mid + eo
        self.t_in.copy_(h)
        self.t_graph.replay()
        return int(self.logits.argmax())
