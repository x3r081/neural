"""Model-independent autoregressive generation over a small backend contract."""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch

from .prefix_cache import BoundedPrefixCache, stable_backend_key


def sample(logits, temperature=0.0, top_k=20, top_p=0.95, generator=None):
    """Greedy or top-k/top-p sampling shared by the reference generation loop."""
    scores = logits.float()
    if temperature <= 0:
        return torch.argmax(scores, dim=-1, keepdim=True)
    scores = scores / temperature
    if top_k > 0:
        threshold = torch.topk(scores, min(top_k, scores.shape[-1])).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if 0 < top_p < 1:
        ordered, indices = torch.sort(scores, descending=True)
        excluded = torch.softmax(ordered, -1).cumsum(-1) > top_p
        excluded[..., 1:] = excluded[..., :-1].clone()
        excluded[..., 0] = False
        ordered.masked_fill_(excluded, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, indices, ordered)
    return torch.multinomial(torch.softmax(scores, -1), 1, generator=generator)


def _backend_device(backend):
    device = getattr(backend, "device", None)
    if device is not None:
        return torch.device(device)
    model = getattr(backend, "model", None)
    if model is not None:
        for parameter in model.parameters():
            if not getattr(parameter, "is_meta", False):
                return parameter.device
    return torch.device("cpu")


def _eos_ids(backend) -> set[int]:
    """Respect the model's generation config, with config/tokenizer fallback."""
    model = getattr(backend, "model", None)
    candidates = [getattr(backend, "generation_config", None),
                  getattr(model, "generation_config", None),
                  getattr(backend, "config", None), getattr(model, "config", None),
                  getattr(backend, "tokenizer", None)]
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "eos_token_id"):
            continue
        value = candidate.eos_token_id
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        return {token_id for token_id in values
                if isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0}
    return set()


class GenerationEngine:
    """Generate from `backend.forward(input_ids, past_key_values, use_cache, ...)`.

    The optional prefix cache is one immutable exact-token snapshot. The cached
    prefix is always chunk-aligned and leaves at least one prefill chunk at the
    end of a prompt. Cache classes are cloned whole, including recurrent state.
    """

    def __init__(self, backend, context=16384, prefill_chunk=64, *,
                 prefix_cache=True, prefix_cache_max_bytes=0,
                 sample_fn: Callable[..., Any] = sample):
        if isinstance(context, bool) or not isinstance(context, int) or context < 1:
            raise ValueError("context must be a positive integer")
        if isinstance(prefill_chunk, bool) or not isinstance(prefill_chunk, int) or prefill_chunk < 1:
            raise ValueError("prefill_chunk must be a positive integer")
        if not isinstance(prefix_cache, bool):
            raise TypeError("prefix_cache must be a boolean")
        if not callable(sample_fn):
            raise TypeError("sample_fn must be callable")
        self.backend = backend
        self.context = context
        self.prefill_chunk = prefill_chunk
        self.vocab_size = len(backend.tokenizer)
        budget = prefix_cache_max_bytes if prefix_cache else 0
        self.prefix_cache = BoundedPrefixCache(max_bytes=budget, chunk_size=prefill_chunk)
        self.sample_fn = sample_fn

    def _sync(self, device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def generate(self, prompt, *, max_tokens=256, temperature=0.0, top_k=20,
                 top_p=0.95, seed=0, ignore_eos=False, on_text=None, cache_prompt=True):
        backend = self.backend
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature < 0
                or isinstance(top_p, bool) or not isinstance(top_p, (int, float))
                or not math.isfinite(top_p) or not 0 < top_p <= 1
                or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("invalid temperature, top_p or top_k")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean")
        if not isinstance(cache_prompt, bool):
            raise ValueError("cache_prompt must be a boolean")
        if on_text is not None and not callable(on_text):
            raise ValueError("on_text must be callable")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        ids = prompt if isinstance(prompt, list) else backend.tokenizer.encode(prompt, add_special_tokens=False)
        ids = list(ids)
        if (not ids or any(not isinstance(i, int) or isinstance(i, bool)
                            or i < 0 or i >= self.vocab_size for i in ids)):
            raise ValueError("invalid or empty prompt token IDs")
        if max_tokens < 1 or len(ids) + max_tokens > self.context:
            raise ValueError(f"prompt plus output must fit context={self.context}")
        device = _backend_device(backend)
        generator = torch.Generator(device=device).manual_seed(seed)
        started = time.perf_counter()
        prefill_started = started
        cached_tokens = 0
        clone_ms = 0.0
        past = None
        start_at = 0
        try:
            self._sync(device)
            key = stable_backend_key(backend, self.context, device)
            hit_started = time.perf_counter()
            hit = self.prefix_cache.longest_hit(ids, key) if cache_prompt else None
            if hit is not None:
                self._sync(device)
                clone_ms = (time.perf_counter() - hit_started) * 1000
                past, cached_tokens, _host_clone_ms = hit
                start_at = cached_tokens
        except Exception:
            self.prefix_cache.clear("request_error")
            raise
        candidate = ((len(ids) - self.prefill_chunk) // self.prefill_chunk) * self.prefill_chunk
        snapshot_saved = False
        new_prefill_tokens = 0
        generated: list[int] = []
        decode_calls = 0
        first_token_s = None
        finish_reason = "length"
        raw = ""
        try:
            out = None
            cursor = start_at
            while cursor < len(ids):
                end = min(cursor + self.prefill_chunk, len(ids))
                batch = torch.tensor([ids[cursor:end]], dtype=torch.long, device=device)
                out = backend.forward(batch, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = out.past_key_values
                cursor = end
                # Count new tokens directly; cursor has advanced exactly one chunk.
                new_prefill_tokens += batch.shape[-1]
                if cache_prompt and not snapshot_saved and candidate > cached_tokens and cursor == candidate:
                    self._sync(device)
                    copy_started = time.perf_counter()
                    saved = self.prefix_cache.save(ids[:candidate], past, key)
                    if saved:
                        self._sync(device)
                        clone_ms += (time.perf_counter() - copy_started) * 1000
                        snapshot_saved = True
            self._sync(device)
            prefill_s = time.perf_counter() - prefill_started
            if out is None:
                raise RuntimeError("prompt prefill produced no model output")
            for step in range(max_tokens):
                token = self.sample_fn(out.logits[:, -1, :], temperature, top_k, top_p, generator)
                token_id = int(token.item())
                generated.append(token_id)
                if first_token_s is None:
                    first_token_s = time.perf_counter() - started
                if token_id in _eos_ids(backend) and not ignore_eos:
                    finish_reason = "stop"
                    break
                raw = backend.tokenizer.decode(generated, skip_special_tokens=False).rstrip("\ufffd")
                if on_text is not None:
                    on_text(raw)
                if step + 1 < max_tokens:
                    out = backend.forward(token, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
                    decode_calls += 1
            self._sync(device)
            wall_s = time.perf_counter() - started
            decode_s = wall_s - prefill_s
            result = {"content": raw, "tokens": generated, "prompt_tokens": ids,
                      "finish_reason": finish_reason, "timings": {
                          "prompt_n": len(ids), "computed_prompt_n": new_prefill_tokens,
                          "cached_prompt_n": cached_tokens,
                          "prompt_ms": prefill_s * 1000,
                          "prompt_per_second": new_prefill_tokens / prefill_s if prefill_s else 0,
                          "effective_prompt_per_second": len(ids) / prefill_s if prefill_s else 0,
                          "predicted_n": len(generated), "decode_calls": decode_calls,
                          "predicted_ms": decode_s * 1000,
                          "predicted_per_second": decode_calls / decode_s if decode_s else 0,
                          "first_token_s": first_token_s, "wall_s": wall_s,
                          "generation_tokens_per_wall_s": len(generated) / wall_s if wall_s else 0,
                          "clock": "request prefill through generation, including sampling, detokenization and streaming callback; excludes startup"},
                      "prefix_cache": {"hit": hit is not None, "cached_tokens": cached_tokens,
                          "new_prefill_tokens": new_prefill_tokens, "copy_ms": clone_ms,
                          "bytes": self.prefix_cache.bytes_used,
                          "reason": self.prefix_cache.last_reason if cache_prompt else "bypassed_for_request"}}
            del past, out
            return result
        except Exception:
            self.prefix_cache.clear("request_error")
            raise
