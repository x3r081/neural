import types

import pytest
import torch

from neural_reference.generation import GenerationEngine


class FakeTokenizer:
    eos_token_id = 9

    def __len__(self):
        return 10

    def encode(self, text, add_special_tokens=False):
        return [int(part) for part in text.split()]

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(map(str, tokens))


class MutableHybridCache:
    """Attention-like and recurrent state fields both mutate during forward."""

    def __init__(self, attention=None, recurrent=0):
        self.attention = list(attention or [])
        self.recurrent = torch.tensor(recurrent, dtype=torch.int64)


class FakeBackend:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.config = types.SimpleNamespace(model_type="fake-moe", eos_token_id=9)
        self.generation_config = types.SimpleNamespace(eos_token_id=[8, 9])
        self.cache_identity = ("fake", 1)
        self.calls = []

    def forward(self, input_ids, past_key_values=None, **kwargs):
        ids = input_ids[0].tolist()
        cache = past_key_values if past_key_values is not None else MutableHybridCache()
        # Deliberately mutate passed state as a real decoder may mutate Cache.
        cache.attention.extend(ids)
        cache.recurrent.add_(sum(ids))
        self.calls.append((list(ids), list(cache.attention), cache.recurrent.item()))
        logits = torch.zeros((1, len(ids), 10))
        logits[..., 1] = 5
        return types.SimpleNamespace(logits=logits, past_key_values=cache)


def make_engine(backend=None, **kwargs):
    return GenerationEngine(backend or FakeBackend(), context=4096,
                            prefill_chunk=64, prefix_cache_max_bytes=4096, **kwargs)


def test_exact_aligned_prefix_reuse_clones_attention_and_recurrent_state():
    backend = FakeBackend()
    engine = make_engine(backend)
    prompt = [i % 7 + 1 for i in range(200)]
    first = engine.generate(prompt, max_tokens=2)
    assert first["prefix_cache"]["hit"] is False
    assert first["prefix_cache"]["cached_tokens"] == 0
    snapshot = engine.prefix_cache._snapshot
    assert snapshot is not None and len(snapshot.tokens) == 128
    before = (list(snapshot.past_key_values.attention), snapshot.past_key_values.recurrent.item())

    calls_before = len(backend.calls)
    second = engine.generate(prompt, max_tokens=2)
    assert second["prefix_cache"]["hit"] is True
    assert second["prefix_cache"]["cached_tokens"] == 128
    assert second["prefix_cache"]["new_prefill_tokens"] == 72
    assert (snapshot.past_key_values.attention, snapshot.past_key_values.recurrent.item()) == before
    prefill = [call[0] for call in backend.calls[calls_before:] if len(call[0]) > 1]
    assert prefill == [prompt[128:192], prompt[192:]]


def test_longer_exact_prefix_reuses_only_unchanged_aligned_prefix():
    engine = make_engine()
    prompt = [i % 7 + 1 for i in range(200)]
    engine.generate(prompt, max_tokens=1)
    extended = prompt + [3, 4, 5, 6, 7, 1, 2, 3]
    result = engine.generate(extended, max_tokens=1)
    assert result["prefix_cache"]["hit"] is True
    assert result["prefix_cache"]["cached_tokens"] == 128
    assert result["prefix_cache"]["new_prefill_tokens"] == len(extended) - 128


def test_disjoint_prompt_does_not_reuse_cached_state():
    engine = make_engine()
    prompt = [i % 7 + 1 for i in range(200)]
    engine.generate(prompt, max_tokens=1)
    changed = [2] + prompt[1:]
    result = engine.generate(changed, max_tokens=1)
    assert result["prefix_cache"]["hit"] is False
    assert result["prefix_cache"]["cached_tokens"] == 0


def test_request_can_bypass_an_existing_snapshot():
    engine = make_engine()
    prompt = [i % 7 + 1 for i in range(200)]
    engine.generate(prompt, max_tokens=1)
    snapshot = engine.prefix_cache._snapshot
    result = engine.generate(prompt, max_tokens=1, cache_prompt=False)
    assert not result['prefix_cache']['hit']
    assert result['prefix_cache']['new_prefill_tokens'] == len(prompt)
    assert result['prefix_cache']['reason'] == 'bypassed_for_request'
    assert engine.prefix_cache._snapshot is snapshot


def test_eos_ids_come_from_backend_metadata_not_checkpoint_constants():
    backend = FakeBackend()
    engine = make_engine(backend, sample_fn=lambda *args: torch.tensor([[8]]))
    result = engine.generate([1, 2], max_tokens=4)
    assert result["tokens"] == [8]
    assert result["finish_reason"] == "stop"


def test_callback_failure_invalidates_cached_snapshot():
    engine = make_engine()
    prompt = [i % 7 + 1 for i in range(200)]
    engine.generate(prompt, max_tokens=1)
    assert engine.prefix_cache.bytes_used > 0
    try:
        engine.generate(prompt, max_tokens=2,
                        on_text=lambda _text: (_ for _ in ()).throw(RuntimeError("callback")))
    except RuntimeError as exc:
        assert str(exc) == "callback"
    else:
        raise AssertionError("expected callback exception")
    assert engine.prefix_cache.bytes_used == 0
    assert engine.prefix_cache.last_reason == "request_error"


@pytest.mark.parametrize("family", ["qwen3_moe", "mixtral"])
def test_tiny_transformers_moe_cached_and_uncached_logits_match(family):
    transformers = pytest.importorskip("transformers")
    if family == "qwen3_moe":
        config = transformers.Qwen3MoeConfig(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=8, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            num_experts=4, num_experts_per_tok=2, eos_token_id=[30, 31],
        )
        model = transformers.Qwen3MoeForCausalLM(config)
    else:
        config = transformers.MixtralConfig(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, num_local_experts=4, num_experts_per_tok=2,
            eos_token_id=[30, 31],
        )
        model = transformers.MixtralForCausalLM(config)
    model.eval()

    class HFTokenizer(FakeTokenizer):
        eos_token_id = [30, 31]

        def __len__(self):
            return 32

    class HFBackend:
        def __init__(self):
            self.model = model
            self.tokenizer = HFTokenizer()
            self.config = config
            self.generation_config = types.SimpleNamespace(eos_token_id=[30, 31])
            self.cache_identity = (family, "tiny-transformers-reference")
            self.prefill_logits = []

        def forward(self, input_ids, **kwargs):
            out = self.model(input_ids=input_ids, **kwargs)
            if input_ids.shape[-1] > 1:
                self.prefill_logits.append(out.logits[:, -1, :].detach().cpu().clone())
            return out

    backend = HFBackend()
    engine = GenerationEngine(backend, context=512, prefill_chunk=64,
                              prefix_cache_max_bytes=2_000_000)
    prompt = [i % 27 + 1 for i in range(200)]
    uncached = engine.generate(prompt, max_tokens=3)
    uncached_logits = backend.prefill_logits[-1]
    cached = engine.generate(prompt, max_tokens=3)
    cached_logits = backend.prefill_logits[-1]
    assert cached["prefix_cache"]["hit"] is True
    assert cached["prefix_cache"]["cached_tokens"] == 128
    assert cached["tokens"] == uncached["tokens"]
    assert torch.allclose(cached_logits, uncached_logits, rtol=1e-5, atol=1e-6)
