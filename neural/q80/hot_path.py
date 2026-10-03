"""Q80-15 — v13 hot-path collapse runtime.

v13 = v12 + three measured fixes (semantics identical):

  B1. Host pool V3 — decode-path gets NEVER promote pageable->pinned (the
      Q80-15 profile showed ~22 ms/token of host memcpy in promotions); H2D
      runs directly from whichever tier holds the expert. Pinned tier default
      raised to 14 GiB so first-touch builds land pinned.
  B2. Expert graph V3 — slot/order metadata fused into ONE pinned H2D copy;
      the static output is returned without a defensive clone (safe: the
      consumer op is enqueued before the next replay on the same stream).
  A.  Attention split-compile — per attention layer: compiled PRE region
      (norm -> q/k/v projections + q/k norms + rotary + gate), EAGER middle
      (DynamicCache update + SDPA — the only dynamic-shape math), compiled
      POST region (gate-multiply -> o_proj -> residual -> post-norm -> shared
      expert -> router), then the opaque expert boundary and the compiled
      combine region. Same split pattern proven for GDN layers in Q80-13/14.

No policy, quantization, sampling, or architecture change. LRU untouched
(measured cost of the Python LRU dict: 0.66 ms/token — a native extension is
NOT justified by the pre-registered >=5 ms bar).
"""

from __future__ import annotations

import time
import types
from typing import Any

GIB = 1024**3
TOP_K = 10
_perf = time.perf_counter

# Q80-FLOOR-1 A/B switch. The meta-fill optimization is bit-exact, so the only
# way to price it honestly is to run both halves in the SAME session with
# interleaved controls; comparing against numbers taken hours earlier is the
# stale-control error this project has already been burned by. Default False =
# the shipped fast path. Set only by measurement harnesses.
LEGACY_META_FILL = False


def _distinct(fn, name):
    return types.FunctionType(fn.__code__.replace(co_name=name), fn.__globals__,
                              name, fn.__defaults__, fn.__closure__)


# ---------------------------------------------------------------------------
# B1 — host pool without decode-path promotion
# ---------------------------------------------------------------------------

def make_host_pool_v3(store, device, pinned_gib: float, pageable_gib: float,
                      layout=None):
    from neural.q80.expert_runtime import PrepackedHostPoolV2

    class PrepackedHostPoolV3(PrepackedHostPoolV2):
        pageable_serves = 0      # H2D sourced from an UNPINNED buffer: the
        # copy is staged and synchronous (~half PCIe rate) and stalls
        # submission. Q80-18 measured ~12-15 ms/token when ~30% of misses
        # came from this tier — a silent-degradation channel, so it is
        # counted and surfaced in dispatch_report().
        evict_sync_waits = 0     # evicting a pinned buffer whose H2D is still
        evict_sync_ms = 0.0      # in flight BLOCKS THE CPU on a GPU event. The
        # code called this "rare" and never counted it; the conversation regime
        # evicts ~13 pinned rows per decode token, so "rare" was an assumption,
        # not a measurement. Counted here so it can be one.
        build_ms = 0.0           # store read + host-side assembly
        demote_ms = 0.0          # pinned -> pageable transfer on eviction

        def get(self, layer: int, expert: int):
            if self.deferred_demote:
                self._drain_demotes()        # main thread owns all tier mutation
            key = (layer, expert)
            buf = self._resident.get(key)
            if buf is not None:
                # THE refresh that makes prefill a scan: a prefill touch of a
                # row already pinned moves it to the MRU end of the very LRU
                # decode depends on, and (below) a prefill miss evicts from
                # that same LRU's cold end.
                self.note("pinned_hit")
                self.note("pinned_refresh")
                self._resident.move_to_end(key)
                return buf
            pg = self._pageable.get(key)
            if pg is not None:                       # serve pageable DIRECTLY
                self.note("pageable_hit_promote")
                self._pageable.move_to_end(key)      # (no promote memcpy)
                self.pageable_serves += 1
                return pg
            if self._demote_pending:
                # deferred demotion in flight: the row's PINNED buffer is still
                # ours and still holds the bytes, so serve from it. Both uses
                # are reads, and skipping this would re-read the row from the
                # store AND leave two copies of it in two tiers.
                inflight = self._demote_pending.get(key)
                if inflight is not None:
                    self.note("pinned_hit")
                    return inflight
            # true first touch: build into pinned (evictions demote as in V2)
            while len(self._resident) >= self.capacity:
                vk, vbuf = self._resident.popitem(last=False)
                ev = self._events.pop(vk, None)
                if ev is not None and not ev.query():
                    t_s = _perf()
                    ev.synchronize()
                    PrepackedHostPoolV3.evict_sync_waits += 1
                    PrepackedHostPoolV3.evict_sync_ms += (_perf() - t_s) * 1000
                self.note("evict_pinned")
                t_d = _perf()
                self._demote(vk, vbuf)
                PrepackedHostPoolV3.demote_ms += (_perf() - t_d) * 1000
                if self.c is not None:
                    self.c.host_pinned_evictions += 1
            self.note("build")
            t_b = _perf()
            buf = self._build(layer, expert)
            PrepackedHostPoolV3.build_ms += (_perf() - t_b) * 1000
            self.note("insert_pinned")
            self._resident[key] = buf
            return buf

    return PrepackedHostPoolV3(store, device, pinned_budget_gib=pinned_gib,
                               pageable_budget_gib=pageable_gib, layout=layout)


# ---------------------------------------------------------------------------
# B2 — slimmed expert graph (fused metadata, no output clone)
# ---------------------------------------------------------------------------

def slim_expert_graph(rt) -> None:
    """Patch the runtime's graph-run host path: one fused pinned copy for
    slots+order, output returned without clone."""
    import torch

    from neural.q80.expert_runtime import _ExpertGraph

    # top_k comes from the LAYOUT, and gid_list is accepted: review flagged the
    # previous 4-arg / TOP_K=10 hardcoding as a latent trap — any bias-carrying
    # or non-top-10 layout routed through build_v13 would have died on its
    # second decode. Behaviour for bias-free top-10 Q80 is unchanged.
    def run_v3(self, x, slot_list, order, top_k_weights, gid_list=None):
        K = self.rt.L.top_k
        # Q80-FLOOR-1: the meta buffer used to be filled with 2*K individual
        # Tensor.__setitem__ calls. Measured at 73.2 us per expert call - 3.5 ms
        # per token across 48 layers, 7% of the whole token - against 0.8 us for
        # the same values written through a numpy view of the SAME pinned
        # buffer. Identical bytes, one C-level store instead of 20 dispatches.
        # PATH ENGAGEMENT (Q80-FLOOR-1 review F8): dispatch_report derives the
        # variant from hasattr(_meta_pin), which is true for BOTH fills, so
        # every artifact said "v3_slim" whichever branch ran. This project has
        # already lost two milestones to a number whose artifact could not say
        # which path produced it, so the branch counts itself.
        rt = self.rt
        if LEGACY_META_FILL:                       # A/B arm: pre-optimization
            rt.meta_fill_setitem = getattr(rt, "meta_fill_setitem", 0) + 1
            mp = self._meta_pin
            for i in range(K):
                mp[i] = slot_list[i]
                mp[K + i] = order[i]
            if gid_list is not None:
                for i in range(K):
                    mp[2 * K + i] = gid_list[i]
        else:
            rt.meta_fill_numpy = getattr(rt, "meta_fill_numpy", 0) + 1
            mn = self._meta_np
            mn[:K] = slot_list
            mn[K:2 * K] = order
            if gid_list is not None:
                mn[2 * K:3 * K] = gid_list
        self._meta_dev.copy_(self._meta_pin, non_blocking=True)
        self.slots.copy_(self._meta_dev[:K])
        self._order_dev.copy_(self._meta_dev[K:2 * K])
        if gid_list is not None:
            self.gids.copy_(self._meta_dev[2 * K:3 * K])
        self.x.copy_(x[0:1])
        # copy_ performs the dtype conversion itself, so the explicit .to()
        # kernel is redundant; the gather and the cast are unchanged in value
        # and order, which keeps this bit-exact.
        self.w.copy_(top_k_weights[0].index_select(0, self._order_dev).to(torch.bfloat16)
                     if LEGACY_META_FILL else
                     top_k_weights[0].index_select(0, self._order_dev))
        self.graph.replay()
        return self.out                                # no clone (see module doc)

    def ensure_v3(g):
        if not hasattr(g, "_meta_pin"):
            k = g.rt.L.top_k
            n = 3 * k if g.rt.bias_gu is not None else 2 * k
            g._meta_pin = torch.zeros(n, dtype=torch.long, pin_memory=True)
            g._meta_dev = torch.zeros(n, dtype=torch.long, device=g.x.device)
            # numpy view over the SAME pinned memory - no copy, no allocation
            g._meta_np = g._meta_pin.numpy()
        g.run = types.MethodType(run_v3, g)

    orig_fd = rt.forward_decode

    def fd(layer, x, tki, tkw):
        out = orig_fd(layer, x, tki, tkw)
        # _ExpertGraphV4 already owns a leaner host path (run_v4) and also has
        # _meta_pin, so the hasattr test alone would both skip it AND, if it
        # ever ran, rebind .run to the v3 preamble it was built to avoid.
        from neural.q80.expert_runtime import _ExpertGraphV4
        if (rt._graph is not None and not isinstance(rt._graph, _ExpertGraphV4)
                and not hasattr(rt._graph, "_meta_pin")):
            ensure_v3(rt._graph)
        return out
    # patch lazily on first call (graph created on first decode)
    rt.forward_decode = fd


# ---------------------------------------------------------------------------
# A — attention split-compile + v13 decode step
# ---------------------------------------------------------------------------

class DecodeStepV13:
    def __init__(self, model, runtime, cfg, compile_attention: bool = False,
                 attn_graph: bool = False) -> None:
        import torch
        import torch.nn.functional as F
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            apply_rotary_pos_emb,
        )

        self.model = model
        self.runtime = runtime
        self.layers = list(model.model.layers)
        self.types = [getattr(l, "block_type", "full_attention") for l in self.layers]
        self.embed = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary = model.model.rotary_emb
        hidden = cfg.hidden_size
        self.hidden = hidden
        self.compile_attention = compile_attention
        self._attn_paths = {}
        self._ag = None
        if attn_graph and not compile_attention:
            from neural.q80.attn_graph import AttnGraphRunner
            self._ag = AttnGraphRunner(model)

        for li, layer in enumerate(self.layers):
            if self.types[li] != "full_attention" or not compile_attention:
                continue
            attn = layer.self_attn
            inorm, pnorm, mlp = (layer.input_layernorm,
                                 layer.post_attention_layernorm, layer.mlp)
            head_dim = attn.head_dim

            def make(li, attn, inorm, pnorm, mlp, head_dim):
                def attn_pre(x, cos, sin):
                    h = inorm(x)
                    ishape = h.shape[:-1]
                    hshape = (*ishape, -1, head_dim)
                    q, gate = torch.chunk(
                        attn.q_proj(h).view(*ishape, -1, head_dim * 2), 2, dim=-1)
                    gate = gate.reshape(*ishape, -1)
                    q = attn.q_norm(q.reshape(*hshape)).transpose(1, 2)
                    k = attn.k_norm(attn.k_proj(h).view(hshape)).transpose(1, 2)
                    v = attn.v_proj(h).view(hshape).transpose(1, 2)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    return q, k, v, gate

                def attn_post(attn_out, gate, x):
                    o = attn_out * torch.sigmoid(gate)
                    o = attn.o_proj(o)
                    x1 = x + o
                    hs = pnorm(x1).view(-1, hidden)
                    shared = mlp.shared_expert(hs)
                    shared = torch.sigmoid(mlp.shared_expert_gate(hs)) * shared
                    _, w, idx = mlp.gate(hs)
                    return x1, hs, shared, w, idx

                def combine(x1, eo, shared):
                    return x1 + (eo + shared).view(1, 1, hidden)

                pre = torch.compile(_distinct(attn_pre, f"q80_attnpre_L{li}"),
                                    mode="reduce-overhead", dynamic=False)
                post = torch.compile(_distinct(attn_post, f"q80_attnpost_L{li}"),
                                     mode="reduce-overhead", dynamic=False)
                comb = torch.compile(_distinct(combine, f"q80_attncomb_L{li}"),
                                     mode="reduce-overhead", dynamic=False)
                return pre, post, comb
            self._attn_paths[li] = make(li, attn, inorm, pnorm, mlp, head_dim)
        self._sdpa = F.scaled_dot_product_attention
        self.python_transitions_per_token = 3 + len(self.layers) + 2 * len(self._attn_paths)

    def _attn_layer(self, li, layer, h, cache, pe):
        import torch

        pre, post, comb = self._attn_paths[li]
        cos, sin = pe
        q, k, v, gate = pre(h, cos, sin)
        attn = layer.self_attn
        k, v = cache.update(k, v, attn.layer_idx)          # DynamicCache append
        n_rep = q.shape[1] // k.shape[1]
        ke = k.repeat_interleave(n_rep, dim=1)
        ve = v.repeat_interleave(n_rep, dim=1)
        ao = self._sdpa(q, ke, ve, attn_mask=None, dropout_p=0.0,
                        scale=attn.scaling)
        ao = ao.transpose(1, 2).reshape(1, 1, -1).contiguous()
        x1, hs, shared, w, idx = post(ao, gate, h)
        eo = self.runtime.forward_decode(li, hs, idx, w)
        return comb(x1, eo, shared)

    def __call__(self, step_ids, cache, pos: int):
        import torch

        torch.compiler.cudagraph_mark_step_begin()
        h = self.embed(step_ids)
        pos_t = torch.tensor([[pos]], device=h.device)
        pe = self.rotary(h, pos_t)
        if self._ag is not None:
            self._ag.begin_token(pe)
        for li, (layer, t) in enumerate(zip(self.layers, self.types)):
            if t == "linear_attention":
                h = layer.forward(h, past_key_values=cache)
            elif self._ag is not None:
                # Q80-17: graph-captured EAGER attention (same kernels/order)
                h = self._ag.attn_sublayer(li, h, cache, pe)
                r = h
                x = layer.post_attention_layernorm(h)
                x = layer.mlp(x)
                h = r + x
            elif self.compile_attention:
                h = self._attn_layer(li, layer, h, cache, pe)
            else:
                # bit-exact v12 attention path (same module calls, same order)
                r = h
                x = layer.input_layernorm(h)
                x, _ = layer.self_attn(hidden_states=x, position_embeddings=pe,
                                       attention_mask=None, past_key_values=cache)
                h = r + x
                r = h
                x = layer.post_attention_layernorm(h)
                x = layer.mlp(x)
                h = r + x
        h = self.norm(h)
        return self.lm_head(h[:, -1, :])


def build_v13(model, device, *, budget_gib: float = 4.0, pinned_gib: float = 14.0,
              pageable_gib: float = 22.0, compile_attention: bool = False,
              attn_graph: bool = False, admission_v4: bool = False,
              layout=None, prepacked_root=None,
              expect_model_id=None, cold_io: bool = True,
              deferred_demote: bool = False) -> dict[str, Any]:
    import torch

    from neural.q80.compiled_layer import install_compiled_gdn_layers, triton_available
    from neural.q80.expert_runtime import Q80SlotPoolRuntimeV2, install_q80_expert_runtime
    from neural.q80.path2_quantized import INT4_STORE_ROOT
    from neural.q80.quant_store import Q80Int4DiskStore

    if not triton_available():
        raise RuntimeError("runtime v13 requires the Triton substrate (triton-windows)")
    if prepacked_root is not None:
        # Q80-QUALITY-PARETO: alternate expert store/representation (e.g. the
        # int8_pc store) — rows served directly from the prepacked store
        from neural.q80.prepacked_store import Q80PrepackedStore, attach_prepacked

        pstore = Q80PrepackedStore(prepacked_root, layout=layout)
        st = pstore.open(expect_model_id=expect_model_id)
        assert st["status"] == "ok", st
        host = make_host_pool_v3(None, device, pinned_gib, pageable_gib,
                                 layout=layout)
        runtime = Q80SlotPoolRuntimeV2(host, device, budget_gib=budget_gib,
                                       layout=layout)
        attach_prepacked(host, pstore)
        host._prepacked_active = True
    else:
        store = Q80Int4DiskStore(root=INT4_STORE_ROOT)
        assert store.open()["status"] == "ok"
        host = make_host_pool_v3(store, device, pinned_gib, pageable_gib)
        runtime = Q80SlotPoolRuntimeV2(host, device, budget_gib=budget_gib)
    # Q80-INT6-PRODUCTION: engage the Q80-22 within-layer prefill compute/IO
    # overlap for prepacked stores. Byte-identical (same rows, same host tier,
    # same eviction order — only WHO reads them and at what queue depth
    # changes), so numerics and STRICT are untouched; it exists because a
    # long-context prefill touches most of the expert set and is bounded by
    # serialized page-faulting, not by the SSD. The DECODE hook stays off
    # (Q80-21 measured it as a regression on page-cache-warm rows).
    cold_pool = None
    if prepacked_root is not None and cold_io:
        from neural.q80.cold_io import ColdReadPool, install_cold_io

        cold_pool = ColdReadPool(prepacked_root, n_workers=8, n_bufs=64,
                                 slot_bytes=runtime.L.slot_bytes)
        install_cold_io(runtime, cold_pool)
        runtime.cold_io_overlap = True
        runtime.cold_io_decode = False
    installed, backend = install_q80_expert_runtime(runtime, model)
    gdn_installed, gdn_stats = install_compiled_gdn_layers(model, runtime, model.config)
    slim_expert_graph(runtime)
    if deferred_demote:
        # Move the pinned->pageable eviction memcpy off the decode critical
        # path. Same bytes, same tier, same order - only WHO copies and WHEN.
        host.enable_deferred_demote()
    if admission_v4:
        from neural.q80.expert_runtime import enable_admission_v4
        enable_admission_v4(runtime)
        runtime.admission_v4 = True
    ds = DecodeStepV13(model, runtime, model.config, compile_attention=compile_attention,
                       attn_graph=attn_graph)
    # Review finding (Q80-INT6-PRODUCTION Part 10): these stats objects existed
    # but were bound and discarded, so a compiled-GDN or attention-graph
    # fallback — each worth 10-20% — was invisible to every measurement. Attach
    # them to the runtime so dispatch_report()/assert_graphed() can gate on them.
    runtime._gdn_stats = gdn_stats
    runtime._gdn_layers = gdn_stats.get("layers", 0)
    runtime._attn_stats = ds._ag.stats if ds._ag is not None else None
    backend["runtime"] = ("Q80-15 v13 hot-path collapse (no-promote host V3 + slim "
                          "expert graph + compiled GDN"
                          + (" + split-compiled attention" if compile_attention else "")
                          + (" + graph-captured eager attention [Q80-17]" if attn_graph else "")
                          + (" + V4 admission [Q80-18]" if admission_v4 else "")
                          + ")")

    def counters():
        c = runtime.c.as_dict()
        # the dispatch block is part of the standard contract: a harness that
        # only calls counters() must still be unable to report a number without
        # seeing which path produced it
        return {"vram_hit_rate": c["vram_hit_rate"], "h2d_gib": c["h2d_gib"],
                "h2d_ops": c["h2d_ops"], "ssd_bytes": c["ssd_bytes"],
                "runtime_int4_converts": 0, "forced_syncs": c["forced_syncs"],
                "host_pinned_evictions": c["host_pinned_evictions"],
                "prepack_converts_buildtime": c["prepack_converts"],
                "misses": c["misses"], "ensure_calls": c["ensure_calls"],
                **runtime.dispatch_snapshot(),
                "require_graph": runtime.require_graph}

    def reset():
        runtime.c.reset()

    def vram_reset():
        runtime.relocate_all_to_cpu()
        runtime.free = [s for s in runtime.free if s not in runtime.scratch_slots]

    def teardown():
        from neural.q80.compiled_layer import uninstall_compiled_gdn_layers
        from neural.q80.expert_runtime import uninstall_q80_expert_runtime
        uninstall_compiled_gdn_layers(gdn_installed)
        uninstall_q80_expert_runtime(installed)
        if cold_pool is not None:
            cold_pool.shutdown()
        runtime.relocate_all_to_cpu()

    return {"name": "NEW_q80_15_v13", "backend": backend, "counters": counters,
            "reset": reset, "vram_reset": vram_reset, "teardown": teardown,
            "runtime": runtime, "decode_step": ds, "cold_io": cold_pool,
            "gdn_stats": gdn_stats,
            "attn_stats": ds._ag.stats if ds._ag is not None else None}
