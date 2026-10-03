"""Locked host memory for the CPU-side expert store (--arena pinned|large).

The shipped server reads experts from a memory-mapped store in the OS page cache: 4 KiB pages,
trimmable (a long prompt makes Windows drop expert pages; the next answer faults them back
one page at a time, MEASURED 5-13 tok/s over its first 128 tokens), and invisible to the GPU.
An arena holds the same bytes, in the same row layout, in memory that cannot be trimmed:

  pinned  torch pinned host memory (cudaHostAlloc). Non-pageable, and the same pointer is valid
          on the GPU (UVA; the server already reads pinned buffers zero-copy from Triton kernels
          this way), which is what --admit-gpu needs. 4 KiB-backed.
  large   VirtualAlloc(MEM_LARGE_PAGES): 2 MiB pages (no page walk per 4 KiB, no hardware-
          prefetcher restart per 4 KiB, physically contiguous frames). Needs the account right
          "Lock pages in memory" (secpol.msc -> User Rights Assignment) and enough contiguous
          physical memory (allocate soon after boot). Then cudaHostRegister'ed so the GPU can
          read it (device pointer from cudaHostGetDevicePointer); if registration fails the
          arena still serves the CPU and --admit-gpu is disabled.

Rows are allocated in chunks (default 128 rows = 1.58 GiB) until the requested count or the
first failure, so a partial arena is possible: rows that did not fit stay on the mmap path, and
the caller keeps their pointers as before. Nothing here is Windows-only except the large-page
allocation; on Linux `pinned` works unchanged and `large` falls back to pinned with a note.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import torch

MEM_COMMIT, MEM_RESERVE, MEM_LARGE_PAGES, PAGE_READWRITE = 0x1000, 0x2000, 0x20000000, 0x04


def _enable_lock_pages_privilege() -> tuple[bool, int]:
    """SeLockMemoryPrivilege for the current process token (Windows). Returns (ok, winerror)."""
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", ctypes.c_ulong)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", ctypes.c_ulong), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    k32.GetCurrentProcess.restype = ctypes.c_void_p
    adv.OpenProcessToken.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]
    adv.LookupPrivilegeValueW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.POINTER(LUID)]
    adv.AdjustTokenPrivileges.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.POINTER(TOKEN_PRIVILEGES),
                                          ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]
    tok = ctypes.c_void_p()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x20 | 0x8, ctypes.byref(tok)):
        return False, ctypes.get_last_error()
    luid = LUID()
    if not adv.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege", ctypes.byref(luid)):
        return False, ctypes.get_last_error()
    tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES * 1)(LUID_AND_ATTRIBUTES(luid, 0x2)))
    ok = adv.AdjustTokenPrivileges(tok, False, ctypes.byref(tp), 0, None, None)
    err = ctypes.get_last_error()            # ERROR_NOT_ALL_ASSIGNED (1300): the account lacks the right
    return bool(ok) and err == 0, err


def _cudart():
    """ctypes handle to the CUDA runtime torch was built against (for cudaHostGetDevicePointer)."""
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    names = []
    if sys.platform == "win32":
        names = sorted((n for n in os.listdir(lib_dir) if n.lower().startswith("cudart64_") and n.lower().endswith(".dll")),
                       reverse=True)
        names = [os.path.join(lib_dir, n) for n in names]
    else:
        names = [os.path.join(lib_dir, n) for n in os.listdir(lib_dir) if n.startswith("libcudart.so")]
        names += ["libcudart.so"]
    for n in names:
        try:
            L = ctypes.CDLL(n)
            L.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
            L.cudaHostGetDevicePointer.restype = ctypes.c_int
            return L
        except (OSError, AttributeError):
            continue
    return None


class ExpertArena:
    def __init__(self, mode: str, slot_bytes: int, rows_per_chunk: int | None = None) -> None:
        # Default chunk = the most rows that fit in 2 GiB: 162 rows x 13,219,200 B = 2,141,510,400 B (raw store),
        # 167 rows x 12,839,040 B = 2,144,119,680 B (packed-scale store). torch's pinned-host allocator rounds
        # requests up to a power of two, so a chunk just under 2 GiB wastes ~6 MB instead of ~450 MB.
        assert mode in ("pinned", "large")
        if rows_per_chunk is None:
            rows_per_chunk = max(1, (1 << 31) // int(slot_bytes))
        self.mode, self.slot_bytes, self.rows_per_chunk = mode, int(slot_bytes), int(rows_per_chunk)
        self.chunks: list[dict] = []
        self.row_ptr: dict[int, int] = {}       # gid -> host pointer (CPU kernels)
        self.row_dev: dict[int, int] = {}       # gid -> device-visible pointer (GPU zero-copy), if available
        self.row_loc: dict[int, tuple[int, int]] = {}   # gid -> (chunk index, byte offset), for row_view()
        self.gpu_visible = False
        self.notes: list[str] = []
        self._rows = 0

    # ------------------------------------------------------------------ allocation
    def _alloc_chunk(self, rows: int) -> dict | None:
        nbytes = rows * self.slot_bytes
        if self.mode == "pinned" or sys.platform != "win32":
            if self.mode == "large" and sys.platform != "win32":
                self.notes.append("large pages: Windows only; using pinned memory")
            try:
                t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            except (RuntimeError, MemoryError) as e:
                self.notes.append(f"pinned allocation of {nbytes / 2**30:.2f} GiB failed: {str(e)[:120]}")
                return None
            return {"t": t, "ptr": t.data_ptr(), "dev": t.data_ptr(), "rows": rows, "kind": "pinned"}
        # Windows large pages
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
        k32.GetLargePageMinimum.restype = ctypes.c_size_t
        if not getattr(self, "_priv", False):
            ok, err = _enable_lock_pages_privilege()
            self._priv = True
            if not ok:
                self.notes.append(f"SeLockMemoryPrivilege not granted (error {err}): give this account 'Lock pages in "
                                  "memory' in secpol.msc and log in again; falling back to pinned memory")
                self.mode = "pinned"
                return self._alloc_chunk(rows)
        lp = int(k32.GetLargePageMinimum()) or (2 << 20)
        size = (nbytes + lp - 1) // lp * lp
        ptr = k32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE)
        if not ptr:
            self.notes.append(f"VirtualAlloc(MEM_LARGE_PAGES, {size / 2**30:.2f} GiB) failed (error "
                              f"{ctypes.get_last_error()}): not enough contiguous physical memory")
            return None
        buf = (ctypes.c_uint8 * size).from_address(ptr)
        t = torch.frombuffer(buf, dtype=torch.uint8)
        dev = None
        try:
            rc = torch.cuda.cudart().cudaHostRegister(ptr, size, 3)   # Portable | Mapped
            if int(rc) == 0:
                rt = _cudart()
                if rt is not None:
                    dp = ctypes.c_void_p()
                    if rt.cudaHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(ptr), 0) == 0 and dp.value:
                        dev = int(dp.value)
                    else:
                        self.notes.append("cudaHostGetDevicePointer failed; arena is CPU-only")
                else:
                    self.notes.append("cudart not found for cudaHostGetDevicePointer; arena is CPU-only")
            else:
                self.notes.append(f"cudaHostRegister returned {rc}; arena is CPU-only (large pages still in effect)")
        except Exception as e:                                           # noqa: BLE001
            self.notes.append(f"cudaHostRegister raised {type(e).__name__}: {str(e)[:100]}; arena is CPU-only")
        return {"t": t, "ptr": ptr, "dev": dev, "rows": rows, "kind": "large", "buf": buf}

    def alloc(self, n_rows: int) -> int:
        """Allocate up to n_rows rows in chunks; returns the number of rows available."""
        left = int(n_rows)
        while left > 0:
            rows = min(left, self.rows_per_chunk)
            ch = self._alloc_chunk(rows)
            if ch is None:
                break
            ch["idx"] = len(self.chunks)          # row_loc needs the index; list.index(c) would compare the tensors
            self.chunks.append(ch)
            left -= rows
        self._rows = sum(c["rows"] for c in self.chunks)
        self.gpu_visible = bool(self.chunks) and all(c["dev"] is not None for c in self.chunks)
        return self._rows

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def gib(self) -> float:
        return self._rows * self.slot_bytes / 2**30

    # ------------------------------------------------------------------ rows
    def _locate(self, i: int) -> tuple[dict, int]:
        for c in self.chunks:
            if i < c["rows"]:
                return c, i
            i -= c["rows"]
        raise IndexError(i)

    def assign(self, gids: list[int]) -> list[int]:
        """Map gids (in order) to arena rows; returns the gids that got a row."""
        got = []
        for i, g in enumerate(gids[:self._rows]):
            c, r = self._locate(i)
            off = r * self.slot_bytes
            self.row_ptr[int(g)] = c["ptr"] + off
            if c["dev"] is not None:
                self.row_dev[int(g)] = c["dev"] + off
            self.row_loc[int(g)] = (c["idx"], off)
            got.append(int(g))
        return got

    def row_view(self, gid: int) -> torch.Tensor:
        """u8 [slot_bytes] view of gid's arena row (host tensor over pinned / locked memory): the source of a
        cudaMemcpyAsync H2D that runs on the copy engine at the pinned rate."""
        ci, off = self.row_loc[int(gid)]
        return self.chunks[ci]["t"][off:off + self.slot_bytes]

    def fill(self, gids: list[int], src_row, after_row=None, log_every_s: float = 5.0) -> float:
        """Copy each gid's row from `src_row(gid)` (any u8 tensor of slot_bytes) into its arena row.
        `after_row(gid)` is called after each copy (e.g. to drop the source's pages from the working
        set). Returns the elapsed seconds."""
        t0 = t_log = time.perf_counter()
        n = 0
        for i, g in enumerate(gids):
            c, r = self._locate(i)
            off = r * self.slot_bytes
            c["t"][off:off + self.slot_bytes].copy_(src_row(g))
            if after_row is not None:
                after_row(g)
            n += 1
            now = time.perf_counter()
            if now - t_log > log_every_s:
                done = n * self.slot_bytes / 2**30
                print(f"  arena fill: {done:.1f} / {len(gids) * self.slot_bytes / 2**30:.1f} GiB "
                      f"({done / (now - t0):.2f} GiB/s)", flush=True)
                t_log = now
        return time.perf_counter() - t0

    def summary(self) -> str:
        kinds = sorted({c["kind"] for c in self.chunks})
        return (f"arena: {self._rows} rows = {self.gib:.1f} GiB in {len(self.chunks)} chunks ({', '.join(kinds) or 'none'}); "
                f"GPU-visible: {self.gpu_visible}" + (f"; notes: {' | '.join(self.notes)}" if self.notes else ""))
