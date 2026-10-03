import json

import pytest


def _split_state(torch, family="qwen3_moe"):
    if family == "mixtral":
        owner = "block_sparse_moe"
        aliases = {"gate_proj": "w1.weight", "up_proj": "w3.weight", "down_proj": "w2.weight"}
    else:
        owner = "mlp"
        aliases = {"gate_proj": "gate_proj.weight", "up_proj": "up_proj.weight", "down_proj": "down_proj.weight"}
    packed_base = f"model.layers.0.{owner}.experts"
    gate_up = torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4)
    down = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    state = {}
    for expert in range(2):
        state[f"{packed_base}.{expert}.{aliases['gate_proj']}"] = gate_up[expert, :6].clone()
        state[f"{packed_base}.{expert}.{aliases['up_proj']}"] = gate_up[expert, 6:].clone()
        state[f"{packed_base}.{expert}.{aliases['down_proj']}"] = down[expert].clone()
    return state, gate_up, down


@pytest.mark.parametrize("family", ["qwen3_moe", "mixtral"])
def test_split_expert_projection_mapping_and_cache(tmp_path, family):
    import torch
    from safetensors.torch import save_file
    from neural_reference.store import ExpertStore

    state, expected_gu, expected_down = _split_state(torch, family)
    # Exercise index/shard discovery: each expert is in a different file.
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())}, "weight_map": {}}
    for i, (key, tensor) in enumerate(state.items()):
        shard = f"part-{i:05d}.safetensors"
        save_file({key: tensor}, str(tmp_path / shard))
        index["weight_map"][key] = shard
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    bytes_limit = (expected_gu[0].numel() + expected_down[0].numel()) * 4
    with ExpertStore(tmp_path, host_cache_bytes=bytes_limit) as store:
        gu, down = store.fetch_expert(0, 1)
        assert torch.equal(gu, expected_gu[1])
        assert torch.equal(down, expected_down[1])
        gu_again, down_again = store.fetch_expert(0, 1)
        assert gu_again.data_ptr() == gu.data_ptr()
        assert down_again.data_ptr() == down.data_ptr()
        report = store.report()
        assert report["host_expert_cache_bytes"] == bytes_limit
        assert report["host_expert_cache_hits"] == 1


def test_packed_expert_slice_preserves_dtype_and_reports_conservative_bytes(tmp_path):
    import torch
    from safetensors.torch import save_file
    from neural_reference.store import ExpertStore

    gate_up = torch.arange(2 * 12 * 4, dtype=torch.bfloat16).reshape(2, 12, 4)
    down = torch.arange(2 * 4 * 6, dtype=torch.bfloat16).reshape(2, 4, 6)
    save_file({
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.down_proj": down,
    }, str(tmp_path / "model.safetensors"))
    with ExpertStore(tmp_path, host_cache_bytes=1024) as store:
        gu, dn = store.fetch_expert(0, 1)
        assert gu.dtype is torch.bfloat16
        assert dn.dtype is torch.bfloat16
        assert torch.equal(gu, gate_up[1])
        assert torch.equal(dn, down[1])
        assert store.report()["host_expert_cache_bytes"] == (gu.numel() + dn.numel()) * 2


def test_quantized_expert_dtype_is_rejected(tmp_path):
    import torch
    from safetensors.torch import save_file
    from neural_reference.store import ExpertStore

    save_file({
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros((2, 8, 4), dtype=torch.int8),
        "model.layers.0.mlp.experts.down_proj": torch.zeros((2, 4, 4), dtype=torch.int8),
    }, str(tmp_path / "model.safetensors"))
    store = ExpertStore(tmp_path)
    try:
        with pytest.raises(TypeError, match="Unsupported source dtype"):
            store.fetch_expert(0, 0)
    finally:
        store.close()


def test_packed_projection_cache_reuses_source_storage_without_weight_copies(tmp_path):
    import torch
    from safetensors.torch import save_file
    from neural_reference.store import ExpertStore
    gu = torch.arange(96, dtype=torch.bfloat16).reshape(2, 12, 4)
    dn = torch.arange(48, dtype=torch.bfloat16).reshape(2, 4, 6)
    save_file({'model.layers.0.mlp.experts.gate_up_proj': gu,
               'model.layers.0.mlp.experts.down_proj': dn}, str(tmp_path/'model.safetensors'))
    store = ExpertStore(tmp_path, host_cache_bytes=0, packed_projection_cache=False)
    lazy = store.fetch_expert(0, 1)
    store.packed_projection_cache = True
    cached = store.fetch_expert(0, 1)
    for a, b in zip(lazy, cached):
        assert torch.equal(a, b)
        assert a.data_ptr() == b.data_ptr()
        assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    store.fetch_expert(0, 0)
    assert store.report()['packed_projection_view_count'] == 2
    assert store.report()['host_expert_cache_bytes'] == 0
    store.close()
    assert store.report()['packed_projection_view_count'] == 0


def test_ambiguous_expert_aliases_fail_closed(tmp_path):
    import torch
    from safetensors.torch import save_file
    from neural_reference.store import ExpertStore

    save_file({
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros((2, 8, 4)),
        "model.layers.0.block_sparse_moe.experts.gate_up_proj": torch.ones((2, 8, 4)),
    }, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Duplicate packed expert alias"):
        ExpertStore(tmp_path)


def test_mixed_expert_dtypes_are_rejected_during_preflight(tmp_path):
    import torch
    from safetensors.torch import save_file
    from neural_reference.store import ExpertStore

    save_file({
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros((2, 8, 4), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.down_proj": torch.zeros((2, 4, 4), dtype=torch.bfloat16),
        "model.layers.1.mlp.experts.gate_up_proj": torch.zeros((2, 8, 4), dtype=torch.float32),
        "model.layers.1.mlp.experts.down_proj": torch.zeros((2, 4, 4), dtype=torch.float32),
    }, str(tmp_path / "model.safetensors"))
    with ExpertStore(tmp_path) as store:
        with pytest.raises(TypeError, match="Mixed expert dtypes"):
            store.validate_layout(hidden_size=4, intermediate_size=4, expert_count=2)
