"""SPARSE-2 — ExpertLayout: the single description of an MoE model's expert
geometry that the common runtime (slot pool, host tiers, prepacked store,
cold I/O, expert graphs) is parameterized by.

Byte formulas verified against the Q80 constants they generalize:
  gate_pack = gate_up_n * hidden / 2            (int4 nibbles, tinygemm-packed)
  down_pack = down_n * down_k / 2
  gate_saz  = (hidden / group) * gate_up_n * 2 * 2 bytes (bf16 scale+zero)
  down_saz  = (down_k / group) * down_n * 2 * 2 bytes
"""

from __future__ import annotations

from dataclasses import dataclass

# fp4-e2m1 codes + E8M0 group scales; "_ps4" = packed 4-bit scale deltas (see ExpertLayout.weight_repr)
MXFP4_REPRS = ("mxfp4_g32", "mxfp4_g32_ps4")


def packed_scale_row_bytes(n_groups: int) -> int:
    """Bytes of one packed scale row: 1 base byte + one nibble per group (n_groups even)."""
    assert n_groups % 2 == 0
    return 1 + n_groups // 2


@dataclass(frozen=True)
class ExpertLayout:
    name: str
    n_layers: int
    n_experts: int
    top_k: int
    hidden: int          # GEMM-1 K (input dim of gate_up)
    gate_up_n: int       # GEMM-1 N (= 2 * moe intermediate)
    down_n: int          # GEMM-2 N (= hidden)
    down_k: int          # GEMM-2 K (= moe intermediate)
    int4_group: int = 32
    inner_k_tiles: int = 2
    # CAPACITY-4: activation family + per-expert-bias feature flags. Keys are
    # FEATURE names resolved by expert_runtime.resolve_activation — never model
    # names. Defaults preserve every pre-existing layout byte-for-byte.
    act: str = "swiglu"
    act_alpha: float = 1.0
    act_limit: float = 0.0
    has_expert_bias: bool = False
    # weight representation feature key: "int4_g32" (tinygemm) or "mxfp4_g32"
    # (source-exact fp4-e2m1 codes + E8M0 group scales, Triton kernel family).
    # "mxfp4_g32_ps4": the SAME logical data as mxfp4_g32, with each 90-scale row
    # stored as 46 B (base byte = the row's minimum scale, then 45 bytes of 4-bit
    # deltas, low nibble = even group): scale = base + delta, an exact integer, so
    # every kernel's arithmetic is unchanged; the slot is 2.88% smaller. Lossless
    # only where every row's max-min <= 15 (tools/pack_store.py refuses otherwise).
    weight_repr: str = "int4_g32"

    def __post_init__(self):
        kt = self.inner_k_tiles * 16
        assert self.hidden % kt == 0 and self.down_k % kt == 0, "tinygemm K tiling"
        assert self.gate_up_n % 8 == 0 and self.down_n % 8 == 0, "tinygemm N tiling"
        assert self.hidden % self.int4_group == 0 and self.down_k % self.int4_group == 0
        assert self.gate_up_n == 2 * self.down_k, "gate_up stacks gate|up over the intermediate"
        if self.weight_repr == "mxfp4_g32_ps4":
            # nibble pairs: byte 1+k of a packed row holds groups 2k (low) and 2k+1 (high)
            assert (self.hidden // self.int4_group) % 2 == 0 and (self.down_k // self.int4_group) % 2 == 0, \
                "mxfp4_g32_ps4 needs an even number of scale groups per row"

    @property
    def gate_pack_bytes(self) -> int:
        if self.weight_repr == "int8_pc":
            return self.gate_up_n * self.hidden
        if self.weight_repr == "int6_g32":
            return self.gate_up_n * self.hidden * 3 // 4
        return self.gate_up_n * self.hidden // 2

    @property
    def down_pack_bytes(self) -> int:
        if self.weight_repr == "int8_pc":
            return self.down_n * self.down_k
        if self.weight_repr == "int6_g32":
            return self.down_n * self.down_k * 3 // 4
        return self.down_n * self.down_k // 2

    @property
    def gate_saz_bytes(self) -> int:
        # int4_g32: bf16 scale+zero per group; mxfp4_g32: one E8M0 byte per
        # group; int8_pc: one bf16 scale per output channel
        if self.weight_repr == "mxfp4_g32":
            return (self.hidden // self.int4_group) * self.gate_up_n
        if self.weight_repr == "mxfp4_g32_ps4":
            return packed_scale_row_bytes(self.hidden // self.int4_group) * self.gate_up_n
        if self.weight_repr == "int8_pc":
            return self.gate_up_n * 2
        return (self.hidden // self.int4_group) * self.gate_up_n * 2 * 2

    @property
    def down_saz_bytes(self) -> int:
        if self.weight_repr == "mxfp4_g32":
            return (self.down_k // self.int4_group) * self.down_n
        if self.weight_repr == "mxfp4_g32_ps4":
            return packed_scale_row_bytes(self.down_k // self.int4_group) * self.down_n
        if self.weight_repr == "int8_pc":
            return self.down_n * 2
        return (self.down_k // self.int4_group) * self.down_n * 2 * 2

    @property
    def slot_bytes(self) -> int:
        return (self.gate_pack_bytes + self.down_pack_bytes
                + self.gate_saz_bytes + self.down_saz_bytes)

    @property
    def is_mxfp4(self) -> bool:
        """fp4-e2m1 codes + E8M0 group scales (either scale layout): the Triton mxfp4 kernel family."""
        return self.weight_repr in MXFP4_REPRS

    @property
    def scale_layout_mode(self) -> int:
        """Scale layout code shared with the CPU kernels' gptoss_set_scale_layout: 0 = one raw E8M0 byte
        per group, 1 = packed (base byte + 4-bit deltas). Only meaningful for mxfp4 representations."""
        return 1 if self.weight_repr == "mxfp4_g32_ps4" else 0

    def descriptor(self) -> dict:
        return {"layout_name": self.name, "n_layers": self.n_layers,
                "n_experts": self.n_experts, "top_k": self.top_k,
                "hidden": self.hidden, "gate_up_n": self.gate_up_n,
                "down_n": self.down_n, "down_k": self.down_k,
                "int4_group": self.int4_group, "inner_k_tiles": self.inner_k_tiles,
                "slot_bytes": self.slot_bytes}


Q80_LAYOUT = ExpertLayout("qwen3-next-80b", n_layers=48, n_experts=512, top_k=10,
                          hidden=2048, gate_up_n=1024, down_n=2048, down_k=512)

OLMOE_LAYOUT = ExpertLayout("olmoe-1b-7b", n_layers=16, n_experts=64, top_k=8,
                            hidden=2048, gate_up_n=2048, down_n=2048, down_k=1024)

PHI_TINY_LAYOUT = ExpertLayout("phi-tiny-moe", n_layers=32, n_experts=16, top_k=2,
                               hidden=4096, gate_up_n=896, down_n=4096, down_k=448)

# gpt-oss-120b: clamped-swiglu ((up+1)*gate*sigmoid(alpha*gate), clamps at
# ±limit) with per-expert biases (biases live GPU-resident OUTSIDE the slot —
# slot_bytes stays pure GEMM payload). Store rows are de-interleaved at build
# (checkpoint gate/up rows alternate; store = [gate | up] chunk order).
# weight_repr mxfp4_g32: the CAPACITY-4 stage-1 gate REJECTED INT4 stacked
# requantization; store rows keep the source fp4 codes byte-exactly.
GPTOSS_LAYOUT = ExpertLayout("gpt-oss-120b", n_layers=36, n_experts=128, top_k=4,
                             hidden=2880, gate_up_n=5760, down_n=2880, down_k=2880,
                             act="clamped_swiglu", act_alpha=1.702, act_limit=7.0,
                             has_expert_bias=True, weight_repr="mxfp4_g32")
assert GPTOSS_LAYOUT.slot_bytes == 13_219_200  # CAPACITY-4 addendum-verified

# The same experts with packed 4-bit scale deltas (tools/pack_store.py converts a raw store into this).
# 12,441,600 B of codes + 8,640 scale rows x 46 B; kernels/gptoss_cpu_cap2.c SLOT_PACKED.
GPTOSS_LAYOUT_PS4 = ExpertLayout("gpt-oss-120b-ps4", n_layers=36, n_experts=128, top_k=4,
                                 hidden=2880, gate_up_n=5760, down_n=2880, down_k=2880,
                                 act="clamped_swiglu", act_alpha=1.702, act_limit=7.0,
                                 has_expert_bias=True, weight_repr="mxfp4_g32_ps4")
assert GPTOSS_LAYOUT_PS4.slot_bytes == 12_839_040
assert GPTOSS_LAYOUT_PS4.gate_pack_bytes == GPTOSS_LAYOUT.gate_pack_bytes
assert GPTOSS_LAYOUT_PS4.down_pack_bytes == GPTOSS_LAYOUT.down_pack_bytes
assert (GPTOSS_LAYOUT.scale_layout_mode, GPTOSS_LAYOUT_PS4.scale_layout_mode) == (0, 1)

_GPTOSS_STORE_LAYOUTS = {"mxfp4_g32": GPTOSS_LAYOUT, "mxfp4_g32_ps4": GPTOSS_LAYOUT_PS4}


def gptoss_layout_for_repr(weight_repr: str) -> ExpertLayout:
    """The gpt-oss ExpertLayout of a store whose descriptor says `weight_repr` (KeyError: not a gpt-oss store)."""
    return _GPTOSS_STORE_LAYOUTS[weight_repr]


def mxfp4_pool_views(pool, L: "ExpertLayout"):
    """(gate_blocks, down_blocks, gate_scales, down_scales): u8 views over a slot pool `pool` [cap, L.slot_bytes]
    (or any [n, slot_bytes] uint8 tensor) for an mxfp4 layout, in the shapes neural.q80.mxfp4_kernels consumes:
      blocks [cap, N, GK, 16]; scales [cap, N, GK] (raw) or [cap, N, GK//2 + 1] (weight_repr "mxfp4_g32_ps4").
    The single place that knows where the scale regions sit, shared by the runtime, the streaming reference and
    tools/scale_pack_gpu_test.py. Slices are slot-strided (stride(0) == slot_bytes)."""
    assert L.is_mxfp4, L.weight_repr
    assert pool.dim() == 2 and pool.shape[1] == L.slot_bytes, (tuple(pool.shape), L.slot_bytes)
    cap = pool.shape[0]
    g_end = L.gate_pack_bytes
    d_end = g_end + L.down_pack_bytes
    gs_end = d_end + L.gate_saz_bytes
    gg, dg = L.hidden // L.int4_group, L.down_k // L.int4_group
    gw, dw = ((packed_scale_row_bytes(gg), packed_scale_row_bytes(dg))
              if L.weight_repr == "mxfp4_g32_ps4" else (gg, dg))
    return (pool[:, :g_end].view(cap, L.gate_up_n, gg, 16),
            pool[:, g_end:d_end].view(cap, L.down_n, dg, 16),
            pool[:, d_end:gs_end].view(cap, L.gate_up_n, gw),
            pool[:, gs_end:].view(cap, L.down_n, dw))


def prepacked_descriptor(layout: "ExpertLayout") -> dict:
    """The tensor-layout descriptor a prepacked store's metadata.json carries for `layout` (the constants the
    runtime asserts against at open()). Torch-free on purpose: tools/pack_store.py writes it too."""
    d = {"slot_bytes": layout.slot_bytes, "n_layers": layout.n_layers,
         "n_experts": layout.n_experts, "hidden": layout.hidden,
         "gate_up_n": layout.gate_up_n, "down_n": layout.down_n,
         "down_k": layout.down_k, "int4_group": layout.int4_group,
         "inner_k_tiles": layout.inner_k_tiles,
         "row": "gate_pack|down_pack|gate_saz|down_saz (tinygemm prepacked)",
         "kernel": "aten::_weight_int4pack_mm (sm_86-class tinygemm)"}
    # non-default representations are identity-relevant; key added only for
    # them so every pre-existing store's metadata stays byte-compatible
    repr_ = getattr(layout, "weight_repr", "int4_g32")
    if repr_ != "int4_g32":
        d["weight_repr"] = repr_
        d["row"] = "gate_codes|down_codes|gate_scales|down_scales"
        d["kernel"] = ("neural.q80.mxfp4_kernels (Triton, fp32-accum)"
                       if repr_ == "mxfp4_g32" else
                       "neural.q80.mxfp4_kernels (Triton, fp32-accum, packed 4-bit scale deltas)"
                       if repr_ == "mxfp4_g32_ps4" else
                       "neural.q80.int8_kernels (Triton, fp32-accum)")
        if repr_ == "mxfp4_g32_ps4":
            d["row"] = "gate_codes|down_codes|gate_scales_ps4|down_scales_ps4"
    return d

# invariants against the historical Q80 constants (regression tripwire)
assert Q80_LAYOUT.gate_pack_bytes == 1_048_576
assert Q80_LAYOUT.down_pack_bytes == 524_288
assert Q80_LAYOUT.gate_saz_bytes == 262_144
assert Q80_LAYOUT.down_saz_bytes == 131_072
assert Q80_LAYOUT.slot_bytes == 1_966_080


# Q80-QUALITY-PARETO: the quality-restoring Q80 expert representation
# (INT4 store QUALITY_REJECTED; int6 passed but failed the dominance bar).
Q80_INT8_LAYOUT = ExpertLayout("qwen3-next-80b-int8", n_layers=48,
                               n_experts=512, top_k=10, hidden=2048,
                               gate_up_n=1024, down_n=2048, down_k=512,
                               weight_repr="int8_pc")
assert Q80_INT8_LAYOUT.slot_bytes == 3_151_872


# Q80-INT6-PROBE: the minimum quality-clean representation, measured.
Q80_INT6_LAYOUT = ExpertLayout("qwen3-next-80b-int6", n_layers=48,
                               n_experts=512, top_k=10, hidden=2048,
                               gate_up_n=1024, down_n=2048, down_k=512,
                               weight_repr="int6_g32")
assert Q80_INT6_LAYOUT.slot_bytes == 2_752_512
