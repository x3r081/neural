import pytest
import torch

from neural_reference.prefix_cache import BoundedPrefixCache, cache_nbytes, clone_cache


def test_cache_budget_counts_unique_backing_storage_not_views():
    base = torch.arange(64, dtype=torch.float32)
    cache = {"full": base, "view": base[16:32]}
    assert cache_nbytes(cache) == base.untyped_storage().nbytes()
    assert cache_nbytes({"view": base[16:32]}) == base.untyped_storage().nbytes()


def test_clone_preserves_shared_storage_across_distinct_overlapping_views():
    base = torch.arange(16, dtype=torch.float32)
    cache = {"left": base[:12], "right": base[4:]}
    cloned = clone_cache(cache)
    assert cache_nbytes(cloned) == cache_nbytes(cache)
    assert cloned["left"].untyped_storage().data_ptr() == cloned["right"].untyped_storage().data_ptr()
    assert cloned["left"].untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
    cloned["right"][0] = -1
    assert cloned["left"][4].item() == -1
    assert base[4].item() == 4


def test_rejects_unaligned_and_over_budget_snapshots():
    prefix = BoundedPrefixCache(max_bytes=16, chunk_size=4)
    assert not prefix.save([1, 2, 3], torch.ones(1), "key")
    assert prefix.last_reason == "insufficient_prefix_or_disabled"
    assert not prefix.save([1, 2, 3, 4], torch.ones(8), "key")
    assert prefix.last_reason == "snapshot_over_budget"
    assert prefix.bytes_used == 0


def test_replacement_drops_old_snapshot_before_allocating_new_one():
    prefix = BoundedPrefixCache(max_bytes=1024, chunk_size=4)
    old = torch.ones(4)
    new = torch.full((4,), 2.0)
    assert prefix.save([1, 2, 3, 4], old, "key")
    old_snapshot = prefix._snapshot
    assert prefix.save([5, 6, 7, 8], new, "key")
    assert prefix._snapshot is not old_snapshot
    assert prefix.longest_hit([5, 6, 7, 8, 9, 1, 2, 3], "key")[0].tolist() == [2.0] * 4


def test_transformers_mixed_attention_and_recurrent_cache_clone():
    cache_utils = pytest.importorskip("transformers.cache_utils")
    attention = cache_utils.DynamicLayer()
    attention.keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    attention.values = attention.keys + 1
    attention.is_initialized = True
    linear = cache_utils.LinearAttentionLayer(number_of_states=1)
    linear.conv_states[0] = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    linear.recurrent_states[0] = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    linear.is_conv_states_initialized[0] = True
    linear.is_recurrent_states_initialized[0] = True
    cache = cache_utils.Cache(layers=[attention, linear])

    cloned = clone_cache(cache)
    assert isinstance(cloned, cache_utils.Cache)
    assert isinstance(cloned.layers[0], cache_utils.DynamicLayer)
    assert isinstance(cloned.layers[1], cache_utils.LinearAttentionLayer)
    assert cloned.layers[0].keys.data_ptr() != attention.keys.data_ptr()
    assert cloned.layers[1].recurrent_states[0].data_ptr() != linear.recurrent_states[0].data_ptr()
    assert torch.equal(cloned.layers[0].keys, attention.keys)
    assert torch.equal(cloned.layers[1].recurrent_states[0], linear.recurrent_states[0])
    cloned.layers[1].recurrent_states[0].add_(1)
    assert not torch.equal(cloned.layers[1].recurrent_states[0], linear.recurrent_states[0])
