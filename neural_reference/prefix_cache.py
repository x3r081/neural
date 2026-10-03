"""Bounded immutable snapshots for exact, chunk-aligned prompt prefixes.

The cache stores a cloned full Transformers cache object and its exact input
token IDs. It deliberately knows nothing about individual cache tensor layouts
(including hybrid/recurrent caches); cropping or partial state surgery is not
safe for those cache types.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import time
import types
from typing import Any


def clone_cache(value: Any, _memo=None, _storage_memo=None) -> Any:
    """Clone full cache state while preserving object and tensor-storage aliases."""
    if _memo is None:
        _memo = {}
    if _storage_memo is None:
        _storage_memo = {}
    ident = id(value)
    if ident in _memo:
        return _memo[ident]
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"Unsupported non-strided tensor in cache: {value.layout}")
        storage = value.untyped_storage()
        storage_key = (str(value.device), storage.data_ptr(), storage.nbytes())
        cloned_storage = _storage_memo.get(storage_key)
        if cloned_storage is None:
            nbytes = storage.nbytes()
            raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                storage, 0, (nbytes,), (1,))
            cloned_storage = raw.clone().untyped_storage()
            _storage_memo[storage_key] = cloned_storage
        clone = torch.empty(0, dtype=value.dtype, device=value.device).set_(
            cloned_storage, value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        _memo[ident] = clone
        return clone
    if isinstance(value, tuple):
        # Preserve named tuples used by some Transformers cache wrappers.
        if hasattr(value, "_fields"):
            clone = type(value)(*(clone_cache(item, _memo, _storage_memo) for item in value))
        else:
            clone = tuple(clone_cache(item, _memo, _storage_memo) for item in value)
        _memo[ident] = clone
        return clone
    if isinstance(value, list):
        clone = []
        _memo[ident] = clone
        clone.extend(clone_cache(item, _memo, _storage_memo) for item in value)
        return clone
    if isinstance(value, dict):
        clone = copy.copy(value)
        _memo[ident] = clone
        clone.clear()
        for key, item in value.items():
            clone[clone_cache(key, _memo, _storage_memo)] = clone_cache(item, _memo, _storage_memo)
        return clone
    if value is None or isinstance(value, (str, bytes, int, float, bool, complex, type)):
        return value
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return value
    attrs = getattr(value, "__dict__", None)
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    if isinstance(attrs, dict) or slots:
        # Copy the wrapper shell, then independently clone its state. This avoids
        # Tensor.__deepcopy__ failures on inference tensors and preserves HF Cache
        # subclasses without interpreting attention or recurrent-state fields.
        cloned = copy.copy(value)
        _memo[ident] = cloned
        if isinstance(attrs, dict):
            cloned_attrs = {key: clone_cache(item, _memo, _storage_memo) for key, item in attrs.items()}
            try:
                cloned.__dict__ = cloned_attrs
            except (AttributeError, TypeError):
                for key, item in cloned_attrs.items():
                    setattr(cloned, key, item)
        for name in slots:
            if name in {"__dict__", "__weakref__"} or not hasattr(value, name):
                continue
            setattr(cloned, name, clone_cache(getattr(value, name), _memo, _storage_memo))
        return cloned
    # Opaque immutable/helper objects are left to their deepcopy contract.
    clone = copy.deepcopy(value, _memo)
    _memo[ident] = clone
    return clone


def cache_nbytes(value: Any) -> int:
    """Count unique backing storage, including full storage for tensor views."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    seen: set[int] = set()
    seen_storage: set[tuple[str, int]] = set()

    def visit(item: Any) -> int:
        ident = id(item)
        if ident in seen:
            return 0
        seen.add(ident)
        if torch is not None and isinstance(item, torch.Tensor):
            try:
                storage = item.untyped_storage()
                key = (str(item.device), storage.data_ptr())
                if key in seen_storage:
                    return 0
                seen_storage.add(key)
                return storage.nbytes()
            except (RuntimeError, NotImplementedError):
                return item.numel() * item.element_size()
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return sum(visit(getattr(item, f.name)) for f in dataclasses.fields(item))
        if isinstance(item, dict):
            return sum(visit(k) + visit(v) for k, v in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return sum(visit(child) for child in item)
        attrs = getattr(item, "__dict__", None)
        if isinstance(attrs, dict):
            return visit(attrs)
        slots = getattr(type(item), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        return sum(visit(getattr(item, name)) for name in slots if hasattr(item, name))

    return visit(value)


@dataclasses.dataclass(frozen=True)
class PrefixSnapshot:
    tokens: tuple[int, ...]
    past_key_values: Any
    nbytes: int
    backend_key: Any


def stable_backend_key(backend, context: int, device: Any) -> tuple[Any, ...]:
    """Fingerprint the model/configuration without hashing full model weights."""
    explicit = getattr(backend, "cache_identity", None)
    if callable(explicit):
        explicit = explicit()
    epoch = getattr(backend, "cache_epoch", None)
    model = getattr(backend, "model", None)
    tokenizer = getattr(backend, "tokenizer", None)
    config = getattr(backend, "config", None) or getattr(model, "config", None)
    if config is not None and callable(getattr(config, "to_dict", None)):
        config_value = config.to_dict()
    elif config is not None:
        config_value = vars(config) if hasattr(config, "__dict__") else repr(config)
    else:
        config_value = None
    encoded = json.dumps(config_value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    flags = tuple((name, repr(getattr(backend, name, None))) for name in (
        "model_id", "model_revision", "adapter_name", "expert_backend", "gdn_backend",
        "device_name", "dtype", "cache_dtype", "compute_dtype"))
    return (explicit, epoch, id(model), id(tokenizer), digest, flags, str(device), context)


class BoundedPrefixCache:
    """One bounded immutable prefill checkpoint; all cache operations are exact."""

    def __init__(self, *, max_bytes: int = 0, chunk_size: int = 64,
                 tail_margin: int | None = None) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        margin = chunk_size if tail_margin is None else tail_margin
        if isinstance(margin, bool) or not isinstance(margin, int) or margin < chunk_size:
            raise ValueError("tail_margin must be at least one chunk")
        self.max_bytes = max_bytes
        self.chunk_size = chunk_size
        self.tail_margin = margin
        self._snapshot: PrefixSnapshot | None = None
        self.last_reason = "disabled_by_budget" if max_bytes == 0 else "empty"
        self.last_copy_ms = 0.0

    @property
    def bytes_used(self) -> int:
        return self._snapshot.nbytes if self._snapshot is not None else 0

    @property
    def snapshot_tokens(self) -> int:
        return len(self._snapshot.tokens) if self._snapshot is not None else 0

    def clear(self, reason: str = "invalidated") -> None:
        # Drop the old state before any replacement allocation is attempted.
        self._snapshot = None
        self.last_reason = reason

    def longest_hit(self, tokens: list[int], backend_key: Any):
        snapshot = self._snapshot
        if snapshot is None:
            return None
        if snapshot.backend_key != backend_key:
            self.clear("backend_mismatch")
            return None
        n = len(snapshot.tokens)
        if len(tokens) < n + self.tail_margin or tuple(tokens[:n]) != snapshot.tokens:
            return None
        if n % self.chunk_size:
            self.clear("unaligned_snapshot")
            return None
        started = time.perf_counter()
        try:
            past = clone_cache(snapshot.past_key_values)
        except Exception:
            self.clear("clone_failed")
            return None
        return past, n, (time.perf_counter() - started) * 1000

    def save(self, tokens: list[int], past_key_values: Any, backend_key: Any) -> bool:
        # `tokens` describe exactly the prefix represented by the provided cache.
        # The generation caller selects a boundary with the required prompt tail.
        boundary = len(tokens)
        if self.max_bytes == 0 or boundary < self.chunk_size or boundary % self.chunk_size:
            self.clear("insufficient_prefix_or_disabled")
            return False
        # Capture must be provided by the caller at this exact processed-token
        # boundary. This method never crops or rewrites a hybrid cache.
        size = cache_nbytes(past_key_values)
        if size > self.max_bytes:
            self.clear("snapshot_over_budget")
            return False
        # Never overlap old and replacement snapshots; the active request's
        # working cache remains separate and is accounted for by the planner.
        self.clear("replacing")
        started = time.perf_counter()
        try:
            cloned = clone_cache(past_key_values)
        except Exception:
            self.clear("clone_failed")
            return False
        self.last_copy_ms = (time.perf_counter() - started) * 1000
        self._snapshot = PrefixSnapshot(tuple(tokens[:boundary]), cloned, size, backend_key)
        self.last_reason = "saved"
        return True
