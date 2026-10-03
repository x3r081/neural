"""Read-only warm-up of pages in an already-owned contiguous address range."""
from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from itertools import islice


def touch_pages(address: int, length: int, page_size: int = 4096) -> int:
    """Read the first and last in-range byte of each OS page intersecting a range.

    ``address`` must identify a readable contiguous region of at least ``length``
    bytes whose lifetime covers this call. No memory is modified. For a page
    intersecting the range in only one byte, that byte is read once. The return
    value is the unsigned sum of the sampled bytes, making the reads observable.
    """
    if isinstance(address, bool) or not isinstance(address, int) or address <= 0:
        raise ValueError("address must be a positive integer pointer")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    max_index = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8 - 1)) - 1
    if page_size > max_index:
        raise ValueError("page_size exceeds the maximum representable NumPy index")
    if length == 0:
        return 0
    max_address = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
    if address > max_address or length - 1 > max_address - address:
        raise ValueError("address range overflows the process pointer width")
    if length > max_index:
        raise ValueError("length exceeds the maximum representable NumPy view size")

    import numpy as np

    # The first page can start partway through this buffer. Subsequent starts
    # are exactly page_size bytes apart; the view is constrained to [address,
    # address + length), so neither sample can touch a guard/out-of-range byte.
    first_boundary = page_size - (address % page_size)
    starts = np.concatenate((np.array([0], dtype=np.int64),
                             np.arange(first_boundary, length, page_size, dtype=np.int64)))
    ends = np.concatenate((starts[1:] - 1,
                           np.array([length - 1], dtype=np.int64)))
    samples = np.concatenate((starts, ends[ends != starts]))

    raw = (ctypes.c_ubyte * length).from_address(address)
    view = np.ctypeslib.as_array(raw)
    view.flags.writeable = False
    return int(view[samples].sum(dtype=np.uint64))


def _touch_batch(ranges, page_size):
    checksum = 0
    for address, length in ranges:
        checksum ^= touch_pages(address, length, page_size=page_size)
    return checksum


def touch_page_ranges(ranges, workers: int = 8, batch: int = 4,
                      page_size: int = 4096) -> int:
    """Warm multiple (address, length) regions with bounded parallel waves.

    Each worker handles at most ``batch`` regions serially. At most
    ``workers * batch`` range tuples are pulled from the input iterator at one
    time, and no more than ``workers`` tasks are submitted per wave. Returns
    the XOR of each region's :func:`touch_pages` checksum. Empty input returns
    zero without starting worker threads.
    """
    for value, name in ((workers, "workers"), (batch, "batch")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    iterator = iter(ranges)
    checksum = 0
    wave_size = workers * batch
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while wave := list(islice(iterator, wave_size)):
            jobs = [pool.submit(_touch_batch, wave[i:i + batch], page_size)
                    for i in range(0, len(wave), batch)]
            for job in jobs:
                checksum ^= job.result()
    return checksum


__all__ = ["touch_pages", "touch_page_ranges"]
