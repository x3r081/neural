from __future__ import annotations

import ctypes

import pytest

from neural_runtime import page_warm
from neural_runtime.page_warm import touch_pages, touch_page_ranges


def _buffer(data: bytes):
    storage = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return storage, ctypes.addressof(storage)


def test_unaligned_multpage_range_samples_only_page_edges_without_writing():
    page = 16
    storage, base = _buffer(bytes([0xEE] * 80))
    start = base + ((3 - base % page) % page)
    length = 2 * page + 5
    # In range-relative coordinates the three page intersections are
    # [0, 12], [13, 28], and [29, 36]. Put distinct values at their edges and
    # sentinels in the interior to detect accidental broad reads in the sum.
    data = (ctypes.c_ubyte * length).from_address(start)
    for i in range(length):
        data[i] = 0
    for i, value in ((0, 1), (12, 2), (13, 3), (28, 4), (29, 5), (length - 1, 6)):
        data[i] = value
    expected_storage = bytes(storage)

    assert touch_pages(start, length, page_size=page) == 21
    assert bytes(storage) == expected_storage
    assert len(storage) == 80


def test_short_unaligned_range_samples_each_edge_once():
    storage, base = _buffer(bytes([0xA5, 7, 11, 0xA5]))
    assert touch_pages(base + 1, 2, page_size=16) == 18
    assert bytes(storage) == bytes([0xA5, 7, 11, 0xA5])


def test_single_byte_range_is_not_double_counted():
    storage, base = _buffer(bytes([0x40]))
    assert touch_pages(base, 1, page_size=16) == 0x40
    assert bytes(storage) == b"@"


def test_empty_range_does_not_dereference_pointer():
    assert touch_pages(1, 0, page_size=4096) == 0


@pytest.mark.parametrize("args", [
    (0, 1, 4096),
    (True, 1, 4096),
    (1, -1, 4096),
    (1, True, 4096),
    (1, 1, 0),
    (1, 1, False),
])
def test_invalid_arguments_are_rejected(args):
    with pytest.raises(ValueError):
        touch_pages(*args)


def test_pointer_width_overflow_is_rejected_before_read():
    max_address = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
    with pytest.raises(ValueError, match="overflows"):
        touch_pages(max_address, 2)


def test_page_range_batching_is_bounded_and_xors_region_checksums(monkeypatch):
    seen = []

    def fake_batch(ranges, page_size):
        seen.append(tuple(ranges))
        return sum(address for address, _length in ranges)

    monkeypatch.setattr(page_warm, "_touch_batch", fake_batch)
    ranges = [(n, 1) for n in range(1, 11)]
    expected = 0
    for start in range(0, len(ranges), 3):
        expected ^= sum(address for address, _ in ranges[start:start + 3])

    assert touch_page_ranges(iter(ranges), workers=2, batch=3, page_size=16) == expected
    assert sorted(len(group) for group in seen) == [1, 3, 3, 3]
    assert sorted(address for group in seen for address, _ in group) == list(range(1, 11))


def test_empty_page_ranges_and_invalid_concurrency_options():
    assert touch_page_ranges([], workers=2, batch=4) == 0
    with pytest.raises(ValueError, match="workers"):
        touch_page_ranges([], workers=0)
    with pytest.raises(ValueError, match="batch"):
        touch_page_ranges([], batch=True)
