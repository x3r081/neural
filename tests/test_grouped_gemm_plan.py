"""CPU-only validation of ragged plans; no GPU queries, launches or model load."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("triton")
from neural.q80.mxfp4_kernels import GroupedGemmPlan, _grouped_tile_rows


def test_pair_local_tiles_at_every_bm_boundary():
    descs = [(100, 200, i, m) for i, m in enumerate((0, 1, 15, 16, 17, 63, 64, 65, 128, 129))]
    for bm in (16, 32, 64, 128):
        rows = _grouped_tile_rows(descs, bm)
        for desc in descs:
            actual = [r for r in rows if r[2] == desc[2]]
            assert [r[4] for r in actual] == list(range((desc[3] + bm - 1) // bm))
            assert all(r[:4] == desc for r in actual)
            positions = [p for r in actual for p in range(r[4] * bm, min((r[4] + 1) * bm, r[3]))]
            assert positions == list(range(desc[3]))


def test_plan_keeps_pointer_offsets_and_original_pair_boundaries():
    xs = torch.empty(140 * 64 + 9, dtype=torch.bfloat16)
    ys = torch.empty(140 * 33 + 13, dtype=torch.bfloat16)
    pairs = [(xs[1:1 + 65 * 64].view(65, 64), 2, ys[3:3 + 65 * 33].view(65, 33)),
             (xs[1 + 65 * 64:1 + 130 * 64].view(65, 64), 2,
              ys[3 + 65 * 33:3 + 130 * 33].view(65, 33))]
    plan = GroupedGemmPlan(pairs, block_m=64)
    assert plan.metadata.device.type == "cpu"
    assert (plan.K, plan.N, plan.ntiles, plan.slots) == (64, 33, 4, (2, 2))
    assert plan.metadata.tolist() == [
        [pairs[0][0].data_ptr(), pairs[0][2].data_ptr(), 2, 65, 0],
        [pairs[0][0].data_ptr(), pairs[0][2].data_ptr(), 2, 65, 1],
        [pairs[1][0].data_ptr(), pairs[1][2].data_ptr(), 2, 65, 0],
        [pairs[1][0].data_ptr(), pairs[1][2].data_ptr(), 2, 65, 1]]
    assert plan.pairs[0][0] is pairs[0][0]


@pytest.mark.parametrize("bm", [0, 1, 15, 24, 63])
def test_invalid_tile_height(bm):
    with pytest.raises(ValueError, match="power of two"):
        _grouped_tile_rows((), bm)


@pytest.mark.parametrize("desc", [(1, 2, -1, 3), (1, 2, 0, -3), (1, 2, 0, 2**31),
                                  (-1, 2, 0, 3), (1, 2, 0)])
def test_invalid_descriptor(desc):
    with pytest.raises(ValueError):
        _grouped_tile_rows([desc], 64)


def test_empty_pairs_are_safe():
    assert GroupedGemmPlan([]).metadata.shape == (0, 5)
    plan = GroupedGemmPlan([(torch.empty(0, 64, dtype=torch.bfloat16), 0,
                            torch.empty(0, 16, dtype=torch.bfloat16))])
    assert plan.ntiles == 0


def test_alignment_hint_requires_both_pointers_and_row_pitches():
    x = torch.empty(4, 64, dtype=torch.bfloat16)
    y = torch.empty(4, 32, dtype=torch.bfloat16)
    assert x.data_ptr() % 16 == y.data_ptr() % 16 == 0
    assert GroupedGemmPlan([(x, 0, y)]).aligned
    xp = torch.empty(4 * 64 + 1, dtype=torch.bfloat16)
    yp = torch.empty(4 * 32 + 1, dtype=torch.bfloat16)
    assert not GroupedGemmPlan([(xp[1:].view(4, 64), 0, y)]).aligned
    assert not GroupedGemmPlan([(x, 0, yp[1:].view(4, 32))]).aligned
    # Aligned first row alone is insufficient: following output rows shift by
    # 66 bytes, so a pointer-wide alignment hint would be false for those rows.
    odd_pitch = torch.empty(4, 33, dtype=torch.bfloat16)
    assert odd_pitch.data_ptr() % 16 == 0
    assert not GroupedGemmPlan([(x, 0, odd_pitch)]).aligned
    # An empty pair introduces no tile or pointer access and does not disable
    # a safe aligned specialization of the non-empty pairs.
    assert GroupedGemmPlan([(x, 0, y), (x[:0], 1, y[:0])]).aligned


def test_reject_overlapping_outputs_and_input_output_alias():
    x = torch.empty(32, 64, dtype=torch.bfloat16)
    y = torch.empty(32, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="output slices overlap"):
        GroupedGemmPlan([(x[:16], 0, y[:16]), (x[16:], 1, y[8:24])])
    with pytest.raises(ValueError, match="overlap an input"):
        GroupedGemmPlan([(x[:16], 0, x[16:]), (x[16:], 1, y[:16])])
    with pytest.raises(ValueError, match="overlap an input"):
        GroupedGemmPlan([(x, 0, x)])


def test_reject_bad_shapes_dtypes_and_strides():
    x = torch.empty(3, 64, dtype=torch.bfloat16)
    y = torch.empty(3, 16, dtype=torch.bfloat16)
    for pairs in ([(x, 0, y[:2])], [(x.float(), 0, y)], [(x[:, ::2], 0, y)],
                  [(x, 0, y), (torch.empty(3, 32, dtype=torch.bfloat16), 1, y.clone())]):
        with pytest.raises(ValueError):
            GroupedGemmPlan(pairs)
