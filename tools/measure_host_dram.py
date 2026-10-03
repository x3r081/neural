"""Measure the host's RAW DRAM read bandwidth (label: MEASURED, decimal GB = 1e9 B; Windows-only).

Why: the CPU expert kernel (kernels/gptoss_cpu_cap2.c) streams rows at a measured 24-29 GB/s through a
4 KiB-paged file view with no software prefetch; the raw read ceiling was never measured. This drives
kernels/membw_probe.c (same 1440 B row / 16-byte-load shape) over three kinds of 4 GiB memory to separate:
(1) raw read bandwidth = private_4k, prefetch 0, 1 stream; (2) software prefetch (0/512/1024/2048 B
ahead) and a 2nd stream; (3) page size / file mapping: private_4k = VirtualAlloc; large_2m = VirtualAlloc +
MEM_LARGE_PAGES (2 MiB pages); filemap_4k = read-only view of <NEURAL_STORE_DIR>/layer_0.slots (the kernel's real memory type).
Also records the DIMM configuration and a thread sweep at each buffer's best setting. Build first
(build_kernels.bat lacks it): gcc -O3 -march=native -fopenmp -shared -o membw_probe.dll kernels\\membw_probe.c
Output: reports/host_dram_bandwidth.json + a table on stdout. Close other heavy apps first.
"""
import argparse, contextlib, ctypes, json, os, pathlib, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import paths as NP                                   # STORE_DIR (env NEURAL_STORE_DIR), add_devkit_dll_dir()

GIB, ROW = 1 << 30, 1440                             # ROW = RB in gptoss_cpu_cap2.c
PREFETCH, STREAMS, SWEEP = (0, 512, 1024, 2048), (1, 2), (1, 2, 4, 8, 16)
BUILD_CMD = r"cd /d %s && gcc -O3 -march=native -fopenmp -shared -o membw_probe.dll kernels\membw_probe.c"
PS_CMD = ("Get-CimInstance Win32_PhysicalMemory | Select-Object DeviceLocator,Capacity,Speed,"
          "ConfiguredClockSpeed,Manufacturer,PartNumber | ConvertTo-Json")

# ---- 1. DIMM configuration (CIM query, nothing Windows-API specific) -------------------------------
def dimm_report():
    out = {"count": 0, "modules": [], "warnings": [], "ddr_peak_GBps_CALCULATED": None}
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", PS_CMD], capture_output=True,
                           text=True, errors="replace", timeout=90, check=True)
        j = json.loads(r.stdout)
    except Exception as e:                           # report it, do not abort the bandwidth run
        out["warnings"].append("could not read DIMM info: %s" % e)
        return out
    mods = j if isinstance(j, list) else [j]         # ConvertTo-Json emits a bare object for 1 DIMM
    out["modules"], out["count"] = mods, len(mods)
    for m in mods:
        rated, conf = int(m.get("Speed") or 0), int(m.get("ConfiguredClockSpeed") or 0)
        if rated and conf and conf < rated:
            out["warnings"].append("%s runs at %d MT/s but is rated %d MT/s (XMP/DOCP profile off?)"
                                   % (m.get("DeviceLocator"), conf, rated))
    if len(mods) < 2:
        out["warnings"].append("only %d populated DIMM(s): single-channel caps read bandwidth" % len(mods))
    if len({m.get("Capacity") for m in mods}) > 1:
        out["warnings"].append("DIMM capacities differ: part of the memory may run single-channel")
    confs = [int(m["ConfiguredClockSpeed"]) for m in mods if m.get("ConfiguredClockSpeed")]
    if confs:      # DDR: 8 B/transfer/channel; assumes the DIMMs sit in different channels
        out["ddr_peak_GBps_CALCULATED"] = round(min(confs) * 8 * min(len(mods), 2) / 1000, 1)
    return out

# ---- 2. the probe DLL ------------------------------------------------------------------------------
def load_probe():
    path = os.path.join(REPO, "membw_probe.dll")
    if not os.path.exists(path):
        sys.exit("membw_probe.dll not found in %s\nBuild it (MinGW-w64 gcc with OpenMP on PATH, flags as in "
                 "tools\\build_kernels.bat):\n  %s" % (REPO, BUILD_CMD % REPO))
    NP.add_devkit_dll_dir()
    lib = ctypes.CDLL(path)
    lib.membw_read.restype = ctypes.c_double
    lib.membw_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_int] * 5
    lib.membw_checksum.restype = ctypes.c_uint64
    return lib

# ---- 3. ALL Windows-only calls live in this class (never constructed off Windows) ------------------
class Win32:
    COMMIT_RESERVE, LARGE, RELEASE = 0x1000 | 0x2000, 0x20000000, 0x8000          # VirtualAlloc/Free flags
    RW, RO, MAP_READ, GENERIC_READ, SHARE_RW, OPEN_EXISTING, ATTR_NORMAL = 4, 2, 4, 0x80000000, 3, 3, 0x80
    SE_ENABLED, NOT_ALL_ASSIGNED = 2, 1300

    def __init__(self):
        from ctypes import wintypes as wt
        V, SZ, D = ctypes.c_void_p, ctypes.c_size_t, wt.DWORD
        k = self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        a = self.a = ctypes.WinDLL("advapi32", use_last_error=True)

        class LUID(ctypes.Structure): _fields_ = [("LowPart", D), ("HighPart", wt.LONG)]
        class LAA(ctypes.Structure): _fields_ = [("Luid", LUID), ("Attributes", D)]     # LUID_AND_ATTRIBUTES
        class TP(ctypes.Structure): _fields_ = [("PrivilegeCount", D), ("Privileges", LAA * 1)]  # TOKEN_PRIVILEGES
        self._t = (LUID, LAA, TP)
        for fn, res, args in [                       # explicit prototypes: 64-bit pointers must not truncate
            (k.VirtualAlloc, V, [V, SZ, D, D]), (k.VirtualFree, wt.BOOL, [V, SZ, D]),
            (k.GetLargePageMinimum, SZ, []), (k.GetCurrentProcess, V, []), (k.CloseHandle, wt.BOOL, [V]),
            (k.CreateFileW, V, [wt.LPCWSTR, D, D, V, D, D, V]),
            (k.CreateFileMappingW, V, [V, V, D, D, D, wt.LPCWSTR]),
            (k.MapViewOfFile, V, [V, D, D, D, SZ]), (k.UnmapViewOfFile, wt.BOOL, [V]),
            (a.OpenProcessToken, wt.BOOL, [V, D, ctypes.POINTER(V)]),
            (a.LookupPrivilegeValueW, wt.BOOL, [wt.LPCWSTR, wt.LPCWSTR, ctypes.POINTER(LUID)]),
            (a.AdjustTokenPrivileges, wt.BOOL, [V, wt.BOOL, ctypes.POINTER(TP), D, V, V])]:
            fn.restype, fn.argtypes = res, args

    def ok(self, v, what):
        """Return v, or raise with the Windows error text if the call failed (NULL / FALSE / INVALID_HANDLE_VALUE)."""
        if v in (None, 0, ctypes.c_void_p(-1).value):
            e = ctypes.get_last_error()
            raise OSError("%s failed: %s (code %d)" % (what, ctypes.WinError(e).strerror, e))
        return v

    def enable_lock_pages(self):
        """Enable SeLockMemoryPrivilege in this process token (required for MEM_LARGE_PAGES)."""
        LUID, LAA, TP = self._t
        tok, luid = ctypes.c_void_p(), LUID()
        self.ok(self.a.OpenProcessToken(self.k.GetCurrentProcess(), 0x20 | 8, ctypes.byref(tok)), "OpenProcessToken")
        try:
            ctypes.set_last_error(0)
            ok = self.a.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege", ctypes.byref(luid)) and \
                self.a.AdjustTokenPrivileges(tok, False, ctypes.byref(TP(1, (LAA * 1)(LAA(luid, self.SE_ENABLED)))), 0, None, None)
            if not ok or ctypes.get_last_error() == self.NOT_ALL_ASSIGNED:   # can "succeed" yet grant nothing
                raise PermissionError("SeLockMemoryPrivilege not granted (%s): the account needs 'Lock pages in "
                                      "memory' (secpol.msc > Local Policies > User Rights Assignment), then log "
                                      "off/on" % ctypes.WinError(ctypes.get_last_error()).strerror)
        finally:
            self.k.CloseHandle(tok)

# ---- 4. the three buffer kinds; each yields (address, nbytes) already touched once -----------------
@contextlib.contextmanager
def virt_buf(W, n, large=False):                     # (a) 4 KiB pages, or (b) 2 MiB pages if large=True
    flags = W.COMMIT_RESERVE
    if large:
        W.enable_lock_pages()
        lp = W.k.GetLargePageMinimum()
        if not lp:
            raise OSError("this CPU/OS reports no large-page support")
        n, flags = n // lp * lp, flags | W.LARGE     # size must be a multiple of the large-page size
    p = W.ok(W.k.VirtualAlloc(None, n, flags, W.RW), "VirtualAlloc(%d B%s)" % (
        n, ", MEM_LARGE_PAGES; needs contiguous free RAM, retry after a reboot" if large else ""))
    try:
        ctypes.memset(p, 0x5A, n)                    # touch once: commit-on-touch, so write every page
        yield p, n
    finally:
        W.k.VirtualFree(p, 0, W.RELEASE)

@contextlib.contextmanager
def file_buf(W, path, want, warm):                   # (c) read-only view of an existing store file
    n = min(want, os.path.getsize(path)) >> 20 << 20  # layer_0.slots is ~1.6 GiB: map what exists
    if n < (256 << 20):
        raise OSError("%s is too small for a bandwidth test (%d B)" % (path, n))
    with contextlib.ExitStack() as st:               # LIFO teardown: unmap, close mapping, close file
        h = W.ok(W.k.CreateFileW(path, W.GENERIC_READ, W.SHARE_RW, None, W.OPEN_EXISTING, W.ATTR_NORMAL, None), "CreateFileW")
        st.callback(W.k.CloseHandle, h)
        m = W.ok(W.k.CreateFileMappingW(h, None, W.RO, 0, 0, None), "CreateFileMappingW")
        st.callback(W.k.CloseHandle, m)
        p = W.ok(W.k.MapViewOfFile(m, W.MAP_READ, 0, 0, n), "MapViewOfFile(%d B)" % n)
        st.callback(W.k.UnmapViewOfFile, p)
        warm(p, n)                                   # touch once: page cache warm, view PTEs populated
        yield p, n

# ---- 5. measurement + verdict (pure Python, no Windows calls) --------------------------------------
def measure(lib, p, n, threads, reps):
    """Prefetch x streams grid at `threads`, then the thread sweep at the best cell."""
    def rd(t, pf, s):
        v = lib.membw_read(p, n, t, ROW, pf, s, reps)
        assert v >= 0, "membw_read rejected its arguments"
        return round(v, 3)
    grid, sums = {}, set()
    for pf in PREFETCH:
        for s in STREAMS:
            grid.setdefault(str(pf), {})[str(s)] = rd(threads, pf, s)
            sums.add(lib.membw_checksum())           # must be identical: every setting reads the same bytes
    bpf, bs = max(((pf, s) for pf in PREFETCH for s in STREAMS), key=lambda x: grid[str(x[0])][str(x[1])])
    sweep = {"prefetch": bpf, "streams": bs, "GBps_by_threads": {str(t): rd(t, bpf, bs) for t in SWEEP}}
    return grid, sweep, len(sums) == 1, rd(threads, 0, 1)   # last = plain baseline re-run (drift check)

def make_verdict(R):
    if "private_4k" not in R:
        return "INCONCLUSIVE: the plain private 4 KiB baseline (a) could not be measured."
    cs = [(c, pf, s, v) for c, d in R.items() for pf, sd in d.items() for s, v in sd.items()]
    a0 = R["private_4k"]["0"]["1"]
    pct = lambda x, y: "%+.1f%%" % ((x / y - 1) * 100)
    # Lift is judged against plain (a), except file-view cells: those are judged against the file
    # view's own plain read, so (c)'s prefetch effect is not confused with the file-vs-private delta.
    base = lambda c: R["filemap_4k"]["0"]["1"] if c == "filemap_4k" else a0
    lift, c, pf, s, v = max((v / base(c) - 1, c, pf, s, v) for c, pf, s, v in cs
                            if not (pf == "0" and s == "1" and c != "large_2m"))
    hi, lo = max(x[3] for x in cs), min(x[3] for x in cs)
    if lift > 0.15:
        out = ("kernel-side headroom: large pages / software prefetch lift the read rate (%.1f GB/s at %s, "
               "prefetch %s, streams %s, %s vs plain %.1f GB/s)" % (v, c, pf, s, pct(v, base(c)), base(c)))
    elif hi / lo - 1 <= 0.10:
        out = ("memory-side wall: the platform read ceiling is ~%.1f GB/s regardless of page size or prefetch "
               "(all %d measured cells within %.1f%%)" % (hi, len(cs), (hi / lo - 1) * 100))
    else:
        out = ("mixed: no setting lifts >15%% over plain (best %s at %s, prefetch %s, streams %s) but cells spread "
               "%.1f%%; read the table" % (pct(v, base(c)), c, pf, s, (hi / lo - 1) * 100))
    best = lambda cfg: max(x[3] for x in cs if x[0] == cfg)
    if "filemap_4k" in R:
        f0, fb, ab = R["filemap_4k"]["0"]["1"], best("filemap_4k"), best("private_4k")
        out += ("; file-view (c) vs private (a): plain %.1f vs %.1f GB/s (%s), best %.1f vs %.1f GB/s (%s)"
                % (f0, a0, pct(f0, a0), fb, ab, pct(fb, ab)))
    else:
        out += "; file-view (c) was not measured"
    return out + ("" if "large_2m" in R else "; large-page (b) was not measured")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gib", type=float, default=4.0, help="buffer size in GiB (default 4)")
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if os.name != "nt":
        sys.exit("measure_host_dram.py uses VirtualAlloc / MapViewOfFile: run it on the Windows host.")

    dimms = dimm_report()
    print("DIMMs: %d populated" % dimms["count"])
    for m in dimms["modules"]:
        print("  %(DeviceLocator)-14s %(Capacity)s B  rated %(Speed)s / configured %(ConfiguredClockSpeed)s MT/s  "
              "%(Manufacturer)s %(PartNumber)s" % m)
    for w in dimms["warnings"]:
        print("  WARNING:", w)
    lib, W, n = load_probe(), Win32(), int(a.gib * GIB)
    warm = lambda p, nb: lib.membw_read(p, nb, a.threads, ROW, 0, 1, 1)   # one full read pass = touch
    kinds = {"private_4k": lambda: virt_buf(W, n), "large_2m": lambda: virt_buf(W, n, large=True),
             "filemap_4k": lambda: file_buf(W, os.path.join(NP.STORE_DIR, "layer_0.slots"), n, warm)}
    R, sweeps, skipped, info = {}, {}, {}, {}
    for cfg, make in kinds.items():
        try:
            with make() as (p, nb):
                print("\n[%s] %.2f GiB touched; measuring (%d threads, best of %d)..." % (cfg, nb / GIB, a.threads, a.reps))
                R[cfg], sweeps[cfg], ok, rr = measure(lib, p, nb, a.threads, a.reps)
                info[cfg] = {"bytes": nb, "checksum_consistent_across_settings": ok, "baseline_rerun_GBps": rr}
        except Exception as e:                       # one failed buffer kind must not lose the others
            skipped[cfg] = str(e)
            print("\n[%s] SKIPPED: %s" % (cfg, e))
    print("\nread bandwidth, GB/s (decimal, MEASURED), %d threads\n%-11s %8s %10s %10s"
          % (a.threads, "config", "prefetch", "streams=1", "streams=2"))
    for cfg, d in R.items():
        for pf, sd in d.items():
            print("%-11s %8s %10.2f %10.2f" % (cfg, pf, sd["1"], sd["2"]))
    print("\nthread sweep at each config's best setting")
    for cfg, t in sweeps.items():
        print("%-11s pf=%-5s streams=%s  %s" % (cfg, t["prefetch"], t["streams"],
              "  ".join("%st=%.1f" % kv for kv in t["GBps_by_threads"].items())))
    hi = max((v for d in R.values() for sd in d.values() for v in sd.values()), default=None)
    verdict = make_verdict(R)
    print("\nVERDICT:", verdict)
    print("best raw read %s GB/s vs server kernel 24-29 GB/s; calculated DDR peak %s GB/s"
          % (hi, dimms["ddr_peak_GBps_CALCULATED"]))
    report = {"kind": "MEASURED_host_dram_read_bandwidth", "neural_metrics_version": 2, "byte_unit": "decimal_GB_1e9",
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "params": {"threads": a.threads, "reps": a.reps, "row_bytes": ROW, "prefetch_dist": PREFETCH},
              "dimms": dimms, "buffers": info, "results": R, "thread_sweep": sweeps, "skipped": skipped,
              "reference": {"server_kernel_GBps_MEASURED_elsewhere": [24, 29], "best_measured_GBps": hi,
                            "ddr_peak_GBps_CALCULATED": dimms["ddr_peak_GBps_CALCULATED"]},
              "verdict": verdict}
    out = os.path.join(REPO, "reports", "host_dram_bandwidth.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(report, indent=2))
    print("wrote", out)

if __name__ == "__main__":
    main()
