"""HYBRID-2 Stage F: fused-graph hybrid decode for gpt-oss-120B.

Per layer L the host does only:
    replay A_L  -> D2H pack -> event -> replay B -> wait event -> CPU misses -> H2D c_out
A_L (captured, per layer):
    combine(prev layer: mid + g_out + c_out) -> attention core -> router
    -> GPU residency lookup (static slot table) -> write B's inputs + host pack
B (captured once): the resident ("hit") experts on the GPU, miss weights zeroed,
    running CONCURRENTLY with the CPU computing the misses.
T (captured): final combine -> norm -> lm_head -> argmax (token id on device).

--doorbell 1 (doorbell.py, fused_core.k_bell) reorders that, in steady state, to: launch A_{L+1} -> CPU misses of layer L
-> ring. A_{L+1} starts with a spin node that waits, in pinned host memory, for the ring, so the GPU resumes microseconds
after the CPU kernel's output is complete instead of after the host's bookkeeping + graph launch. Same kernels, same
stream order, same data: bit-identical outputs (residency frozen). The spin is bounded (--doorbell-timeout-ms); a spin
that gives up makes the host re-run the token in the classic order, so it never fails a request, and a per-token mode
switch (--doorbell-warm-tokens, --doorbell-faults-max) keeps the page-fault-storm regime on the classic loop.
0 (default) = the sequence above, decode_token's classic loop unchanged.

The VRAM hot set is frequency-ranked on calibration prompts disjoint from the
test prompts and frozen. Prefill uses Neural's V2 path, which streams misses
through scratch slots and never changes residency (so the slot table stays valid).
"""
import argparse, ctypes, hashlib, json, os, statistics, sys, time
os.environ.setdefault("GOMP_SPINCOUNT", "20000000")
import numpy as np
import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the vendored runtime in ./neural
import paths as NP                                                   # noqa: E402
ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--model-dir", default=NP.MODEL_DIR)
ap.add_argument("--store-dir", default=NP.STORE_DIR,
                help="expert store directory (tools/build_store.py; default NEURAL_STORE_DIR). A raw store (13,219,200 B/slot) "
                     "or a packed-scale store (tools/pack_store.py, 12,839,040 B/slot) - the layout is read from the store's "
                     "metadata.json and every consumer follows it")
ap.add_argument("--out", default=None)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--model-name", default="gpt-oss-120b-neural")
ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--identity", default=None)
ap.add_argument("--api-key-env", default="NEURAL_API_KEY")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--prewarm", type=int, default=1)
ap.add_argument("--fast-prefill", type=int, default=1)
ap.add_argument("--prefill-chunk", type=int, default=4096)
ap.add_argument("--trim-prefill-cache", type=int, choices=(0, 1, 2), default=0,
                help="experimental: release unused allocations after prompt processing and its GPU synchronization. "
                     "0 keeps the allocator caches; 1 releases unused CUDA blocks; 2 also releases the reusable "
                     "CPU-prefill buffers and unused pinned-host blocks. Model, K/V and graph storage stay live. "
                     "The release cost is charged to prefill_s; no tensor arithmetic changes")
ap.add_argument("--grouped-prefill-gemm", type=int, choices=(0, 1), default=None,
                help="experimental: group independent prefill expert GEMMs into one launch without merging their "
                     "row tiles or changing accumulation order (default: NEURAL_PREFILL_GROUPED_GEMM, normally 0)")
ap.add_argument("--grouped-prefill-max-tokens", type=int, default=4096,
                help="limit grouped GEMMs to one prefill block with at most this many new tokens; "
                     "0 disables the size guard for research (default 4096)")
ap.add_argument("--grouped-prefill-min-tokens", type=int, default=1024,
                help="minimum new tokens for grouped GEMMs when the maximum is positive; "
                     "smaller prompts retain the stock path (default 1024)")
ap.add_argument("--masked-gemv", type=int, choices=(0, 1), default=0,
                help="experimental: skip zero-weight GPU experts during decode, preserving active expert arithmetic "
                     "and the four-row reduction. Requires finite original expert results; default 0")
ap.add_argument("--prefill-order", choices=("block", "layer"), default="block",
                help="order of prompt processing when the prompt is longer than --prefill-chunk. block: each block runs "
                     "all layers before the next block starts, so a non-resident expert that several blocks route to "
                     "crosses PCIe once per block. layer: each layer runs over all blocks and every non-resident expert "
                     "the GPU computes is copied ONCE per layer and serves every block that routes to it. Outputs are "
                     "bit-identical (the per-block CPU/GPU split and GEMM calls are unchanged; --prefill-order-check "
                     "verifies it). Costs VRAM proportional to the whole prompt instead of one block "
                     "(~50 KB per prompt token: the layer's sorted expert inputs and outputs, plus the residuals)")
ap.add_argument("--prefill-order-check", action="store_true",
                help="validation entry point (like --ppl-check): prefill a long text in both orders, compare the final "
                     "hidden states, every position's logits and every layer's K/V bit for bit, print "
                     "PREFILL_ORDER_CHECK (with the timings and expert stagings of both); then the same comparison "
                     "for block order at --prefill-chunk 4096 vs 16384 (PREFILL_CHUNK_CHECK, and "
                     "PREFILL_CHUNK_CHECK_GPUONLY with --cpu-prefill-max 0 for both) and exit")
ap.add_argument("--prefill-order-check-tokens", type=int, default=13000,
                help="with --prefill-order-check: prompt length in tokens (capped at --smax - 17; more than "
                     "--prefill-chunk, or there is only one block and nothing to compare)")
ap.add_argument("--review-prefill-check", default=None, metavar="JSON",
                help="run the optimization review's frozen-residency prefill bit checks and write their JSON report")
ap.add_argument("--review-prefill-lengths", default="128,3000,5000,13000",
                help="comma-separated prompt lengths for --review-prefill-check")
ap.add_argument("--review-prefill-groups", default="6",
                help="comma-separated staged-expert group sizes to validate against production")
ap.add_argument("--review-decode-check", default=None, metavar="JSON",
                help="validate masked GEMV against full decode logits, hidden states, routing and K/V at frozen residency")
ap.add_argument("--review-decode-lengths", default="128,3000,13000")
ap.add_argument("--review-decode-tokens", type=int, default=64)
ap.add_argument("--review-cpu-decode-check", default=None, metavar="JSON",
                help="compare reference and candidate CPU DLLs with frozen residency; exits without serving")
ap.add_argument("--review-cpu-dll", default="gptoss_cpu_persistent.dll",
                help="candidate DLL for --review-cpu-decode-check")
ap.add_argument("--count-routing", action="store_true")
ap.add_argument("--cpu-prefill-max", type=int, default=16)
ap.add_argument("--idle-canon", type=int, default=1)
ap.add_argument("--split-k", type=int, default=16)
ap.add_argument("--ppl-check", action="store_true")
ap.add_argument("--protect-side", type=int, default=1)
ap.add_argument("--pool", type=float, default=6.0)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--calib-tokens", type=int, default=64)
ap.add_argument("--smax", type=int, default=1024)
ap.add_argument("--hotset", default=None, help="json with chosen (layer, expert) list; calibrate if absent")
ap.add_argument("--kdll", default="gptoss_cpu_cap2.dll")
ap.add_argument("--cpu-multi-dll", default="gptoss_cpu_multi.dll",
                help="ABI-compatible grouped CPU expert kernel selected by the platform launcher")
ap.add_argument("--kernel-persistent", type=int, choices=(0, 1), default=0,
                help="opt-in persistent CPU team; requires a compatible --kdll (default off)")
ap.add_argument("--skip-quality", action="store_true")
ap.add_argument("--refresh-every", type=int, default=8)
ap.add_argument("--refresh-m", type=int, default=32)
ap.add_argument("--capbufs", type=int, default=24)
ap.add_argument("--prefill-m", type=int, default=64)
ap.add_argument("--prefill-weight", type=int, default=16,
                help="the prompt's expert routing counts as this many generated tokens for VRAM admission "
                     "(0 = ignore the prompt; a huge value = the old full-weight behaviour)")
ap.add_argument("--early-every", type=int, default=4)
ap.add_argument("--early-tokens", type=int, default=32)
ap.add_argument("--static", type=int, default=0)
ap.add_argument("--no-unlock", action="store_true")
ap.add_argument("--warm-margin", type=float, default=1.5,
                help="GiB of free RAM the startup warm-up leaves untouched")
ap.add_argument("--warm-general", type=int, default=1,
                help="rank the warm-up by coding + general calibration counts (0 = coding only)")
ap.add_argument("--warm-order", choices=("likely-last", "likely-first"), default="likely-last",
                help="touch order of the warm-up; the most recently touched pages are trimmed last")
ap.add_argument("--warm-method", choices=("kernel", "pages", "pages-parallel"), default="kernel",
                help="kernel = legacy zero-input expert computation; pages = read each mapped page without model arithmetic")
ap.add_argument("--warm-workers", type=int, default=8, choices=range(1, 65),
                help="bounded page-warm workers (independent of expert compute threads)")
ap.add_argument("--ws-min", type=int, default=1,
                help="after the warm-up, ask Windows (soft minimum working set) to trim other processes' idle "
                     "pages before this server's expert pages (0 = off)")
ap.add_argument("--release-resident", type=int, default=1,
                help="at startup, drop the RAM copies of all VRAM-resident experts (0 = the static core only)")
ap.add_argument("--no-fold", action="store_true")
ap.add_argument("--eager-core", action="store_true")
ap.add_argument("--no-zerocopy", action="store_true")
ap.add_argument("--direct-prefill", type=int, default=1)
ap.add_argument("--quality-keys", default="all")
ap.add_argument("--mmap-embed", type=int, default=1)
ap.add_argument("--splice-memory", type=int, default=512,
                help="tool-call responses whose exact tokens are remembered for re-rendering (0 = off)")
ap.add_argument("--splice-memory-tokens", type=int, default=1 << 22,
                help="total tokens the splice memory may hold (4 bytes each; LRU-evicted beyond it)")
ap.add_argument("--splice-after-final", choices=("drop", "keep"), default="drop",
                help="drop (harmony-canonical): once a final answer follows an agent loop, its tool-call turns "
                     "render without their reasoning again, so the next user turn re-prefills that loop once. "
                     "keep: keep splicing them (prefix cache kept across user turns; earlier reasoning stays in "
                     "context, which is not the rendering the model was trained on)")
ap.add_argument("--splice-reserve", type=int, default=1024,
                help="a spliced prompt must leave this many tokens (or the request's max_tokens if smaller; "
                     "at most a quarter of the context) below smax-16 for generation; otherwise the shorter "
                     "unspliced prompt is used")
# ---- experimental decode levers (docs/LEVERS.md). Every default reproduces the v4 server exactly.
ap.add_argument("--kernel-prefetch", type=int, default=0,
                help="CPU expert kernel: software prefetch distance in bytes (0 = off; try 1024). Bit-identical.")
ap.add_argument("--kernel-pair", type=int, default=0,
                help="CPU expert kernel: 1 = stream two rows per thread iteration (more misses in flight). Bit-identical.")
ap.add_argument("--kernel-affinity", type=int, default=0,
                help="CPU expert kernel: pin OpenMP thread t to logical CPU t*stride (0 = off; 2 = one per physical core)")
ap.add_argument("--kernel-fuse", type=int, default=0,
                help="CPU expert kernel: 1 = two-phase kernel (3 fewer barriers per call). Bit-identical.")
ap.add_argument("--kernel-cold-prefetch", type=int, default=0,
                help="CPU expert kernel: 1 = before each call, probe one byte per expert and PrefetchVirtualMemory the "
                     "experts whose pages are not resident, so the OS reads them at queue depth instead of 8 threads "
                     "faulting page by page (first answers, rewrites after the file cache was churned). Outputs untouched.")
ap.add_argument("--zc-misses", type=int, default=0,
                help="per layer, up to N plain misses (not admitted) are streamed by DMA from the arena into scratch "
                     "slots and computed by the GPU, as an extra DRAM read pipe next to the CPU (needs --arena; "
                     "always leaves at least one miss to the CPU). Exact: same numerics as a VRAM hit.")
ap.add_argument("--wait", choices=("event", "poll"), default="event",
                help="how the host waits for a layer's router: cudaEventSynchronize, or busy-poll the event")
ap.add_argument("--doorbell", type=int, default=0, choices=(0, 1),
                help="1 = in steady state launch each layer's graph BEFORE the CPU expert kernel that feeds it; the graph's "
                     "first node spins on a pinned-host doorbell word (doorbell.py) that the host rings once the kernel's "
                     "output is complete. Same kernels in the same stream order: outputs are bit-identical (residency "
                     "frozen). A spin that gives up (host stalled) is recovered by re-running the token in the classic "
                     "order: it never fails a request. 0 = the classic sequence (no spin node is captured).")
ap.add_argument("--doorbell-timeout-ms", type=int, default=800,
                help="a spin node gives up after this long (WDDM's TDR kills a kernel that runs ~2 s, so 1800 is the "
                     "enforced maximum). Not fatal: the host then re-runs the token in the classic order")
ap.add_argument("--doorbell-max-timeouts", type=int, default=3,
                help="after this many timeout recoveries in one process the early launch is switched off for good "
                     "(every token then runs the classic order); 0 = never switch it off")
ap.add_argument("--doorbell-faults-max", type=int, default=300,
                help="a token that follows one with more than this many process page faults runs the classic order "
                     "(first answers after a cold start ~1,200 faults/token, first turns ~130, later ~20: MEASURED)")
ap.add_argument("--doorbell-warm-tokens", type=int, default=32,
                help="the first N decode tokens of every request run the classic order (the fault-storm regime)")
ap.add_argument("--doorbell-flush", type=int, default=1, choices=(0, 1),
                help="1 = cudaStreamQuery right after each early launch (WDDM batches launches; this submits the spin graph "
                     "to the GPU now instead of at the next sync). A/B knob.")
ap.add_argument("--doorbell-poll-cap", type=int, default=1 << 20,
                help="second bound of a spin (iterations), in case the GPU timer (globaltimer) misbehaves: ~1.1 polls/us "
                     "MEASURED, so the default is ~1 s; at most 2**20 (below WDDM's TDR)")
ap.add_argument("--doorbell-inject", type=int, default=0,
                help="TEST ONLY: every N-th early token withholds the ring of one hand-off (after layer 17): its spin must "
                     "time out and the token must be recovered (0 = off). Exercises the recovery on the real GPU; with "
                     "residency frozen the answers must equal the classic run's")
ap.add_argument("--kv-ring", type=int, default=0,
                help="rolling K/V cache of this many positions for the 18 sliding-window layers (0 = full smax; "
                     "256 frees ~0.55 GiB for ~45 more expert slots; needs --pool raised to use it)")
ap.add_argument("--kv-checkpoints", type=int, default=8,
                help="with --kv-ring: ring snapshots kept (9 MiB each at R=256) for prompt-cache rollback")
ap.add_argument("--scratch", type=int, default=16,
                help="prefill scratch slots carved out of the pool (16 = v4; 4-6 frees 10-12 decode slots)")
ap.add_argument("--stage-ring", type=int, default=6,
                help="pinned staging buffers (12.8 MB each, 2..8) between the mmap'd store and the prompt-processing H2D. "
                     "With --stage-thread 1 this is the copy look-ahead (up to N experts copied and not yet issued to the "
                     "GPU); 4 = the old ring size")
ap.add_argument("--stage-thread", type=int, default=1, choices=(0, 1),
                help="1 = a producer thread does the prompt-processing expert copies (mmap -> pinned ring) ahead of the "
                     "launching thread; 0 = the old inline copy on the launching thread. Same bytes, same order, same "
                     "events either way")
ap.add_argument("--near-miss", type=int, default=0, choices=(0, 1, 2),
                help="router ranks 5-8 (near misses): 0 = off, 1 = compute them (into traces), 2 = also mark them "
                     "as admission candidates. Exact: the top-4 and their weights are unchanged")
ap.add_argument("--trace-max-tokens", type=int, default=200_000)
ap.add_argument("--arena", choices=("off", "pinned", "large"), default="off",
                help="hold the CPU-side expert store in a locked arena instead of the page cache: pinned = "
                     "cudaHostAlloc'd (also zero-copy readable by the GPU), large = VirtualAlloc MEM_LARGE_PAGES + "
                     "cudaHostRegister (needs 'Lock pages in memory'); partial fallback to the mmap on failure")
ap.add_argument("--arena-max-gib", type=float, default=0.0,
                help="cap the arena size (0 = free RAM at start minus --warm-margin)")
ap.add_argument("--admit-gpu", type=int, default=0,
                help="1 = zero-surcharge admission: an admitted miss is computed by the GPU straight from the arena "
                     "and the streamed bytes land in its VRAM slot (needs --arena)")
ap.add_argument("--admit-gpu-max", type=int, default=1, help="GPU admissions per layer at most (PCIe budget)")
ap.add_argument("--admit-rule", choices=("server", "window", "demand"), default="server",
                help="server = v4 candidates (>=2 uses in the request, refreshed every 8 tokens); window = admit a miss "
                     "used >= --admit-k times in the last --admit-window tokens; demand = --demand-probe scores")
ap.add_argument("--admit-k", type=int, default=3)
ap.add_argument("--admit-window", type=int, default=64)
ap.add_argument("--victim", choices=("count", "lru"), default="count",
                help="eviction key among adaptive slots: count = v4 (uses in request, then calibration), lru = last use")
ap.add_argument("--demand-probe", default=None,
                help="reports/demand_probe.npz from tools/fit_demand_probe.py: per-token near-future expert demand "
                     "scores from the final hidden state (used by --admit-rule demand)")
ap.add_argument("--selftest-admit", type=int, default=1,
                help="with --admit-gpu: verify the zero-copy compute-and-store kernel against the CPU kernel at startup")
A = ap.parse_args()
if A.grouped_prefill_max_tokens < 0:
    ap.error("--grouped-prefill-max-tokens must be >= 0")
if (A.grouped_prefill_min_tokens < 0 or
        (A.grouped_prefill_max_tokens and A.grouped_prefill_min_tokens > A.grouped_prefill_max_tokens)):
    ap.error("--grouped-prefill-min-tokens must be >= 0 and not exceed a positive maximum")
if A.trim_prefill_cache == 2 and not callable(getattr(torch._C, "_host_emptyCache", None)):
    ap.error("--trim-prefill-cache 2 requires this PyTorch build's host cache release API; use 1 for CUDA only")
if A.prefill_weight < 0:
    ap.error("--prefill-weight must be >= 0")
if A.kv_ring and A.kv_ring < 128:
    ap.error("--kv-ring must be 0 or >= 128 (the sliding window)")
if (A.admit_gpu or A.zc_misses) and A.arena == "off":
    ap.error("--admit-gpu / --zc-misses need --arena pinned|large (the GPU streams experts from locked host memory)")
if A.admit_rule == "demand" and not A.demand_probe:
    ap.error("--admit-rule demand needs --demand-probe")
if not 2 <= A.stage_ring <= 8:
    ap.error("--stage-ring must be 2..8 (each buffer is one pinned expert row)")
if A.no_fold and (A.kv_ring or A.near_miss):
    ap.error("--no-fold (eager per-layer graphs) cannot be combined with --kv-ring or --near-miss")
if A.doorbell:
    if A.no_fold or A.eager_core or A.no_zerocopy:
        ap.error("--doorbell needs the default fused zero-copy folded path (no --no-fold / --eager-core / --no-zerocopy)")
    if not 1 <= A.doorbell_timeout_ms <= 1800:
        ap.error("--doorbell-timeout-ms must be in 1..1800 (WDDM's TDR kills a kernel that runs ~2 s)")
    if not 1024 <= A.doorbell_poll_cap <= 1 << 20:
        ap.error("--doorbell-poll-cap must be in 1024..1048576 (~1 s of polls: below WDDM's TDR)")
    if min(A.doorbell_max_timeouts, A.doorbell_faults_max, A.doorbell_warm_tokens, A.doorbell_inject) < 0:
        ap.error("--doorbell-max-timeouts / -faults-max / -warm-tokens / -inject must be >= 0")

import neural.moe.gptoss_adapter as _ga                                        # noqa: E402
from neural.moe.gptoss_adapter import (StreamingStoreReference,                 # noqa: E402
                                       build_gptoss_neural, load_gptoss_core)
from neural.moe.gptoss_core_graphs import GptOssCoreGraphs                      # noqa: E402
from neural.q80.bench_hygiene import require_idle, snapshot, stamp               # noqa: E402
from neural.q80.mxfp4_kernels import mxfp4_gemv, mxfp4_gemv_masked               # noqa: E402

NP.add_devkit_dll_dir()                     # kernels are self-contained; harmless if absent
S = os.path.dirname(os.path.abspath(__file__))
CKPT, STORE = A.model_dir, A.store_dir
# The store decides the slot layout: raw E8M0 scales (weight_repr mxfp4_g32, 13,219,200 B/slot) or packed 4-bit scale
# deltas (mxfp4_g32_ps4, 12,839,040 B/slot). Every consumer follows it: the slot pool and Triton views (build_gptoss_neural),
# both CPU kernel DLLs (bind_scale_layout below), the capture buffers / prompt ring / arena rows (SLOTB).
SLAYOUT = _ga.layout_for_store(STORE)
SLOTB = SLAYOUT.slot_bytes
SCALE_MODE = SLAYOUT.scale_layout_mode
import cpu_prefill as _CP                                                    # noqa: E402
lib = ctypes.CDLL(os.path.join(S, A.kdll))
VP = ctypes.c_void_p
try:        # the mode is process-global inside each DLL: set it on both; a build without the switch + a packed store is refused
    _CP.bind_scale_layout(lib, SCALE_MODE, SLOTB, A.kdll)
    _CP.bind_scale_layout(ctypes.CDLL(os.path.join(S, A.cpu_multi_dll)), SCALE_MODE, SLOTB, A.cpu_multi_dll)
except RuntimeError as e:
    raise SystemExit(f"{e} [store {STORE}: {SLAYOUT.weight_repr}]")
lib.gptoss_experts.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int]
lib.gptoss_experts_cap.argtypes = [ctypes.c_int, VP, VP, VP, VP, VP, VP, VP, ctypes.c_int, VP]
if hasattr(lib, "gptoss_set_tuning"):
    lib.gptoss_set_tuning.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.gptoss_set_tuning(A.kernel_prefetch, A.kernel_pair, A.kernel_affinity)
    if hasattr(lib, "gptoss_set_fuse"):
        lib.gptoss_set_fuse.argtypes = [ctypes.c_int]
        lib.gptoss_set_fuse(A.kernel_fuse)
    elif A.kernel_fuse:
        print(f"WARNING: {A.kdll} predates gptoss_set_fuse; rebuild it to use --kernel-fuse (ignored)", flush=True)
    if hasattr(lib, "gptoss_set_cold_prefetch"):
        lib.gptoss_set_cold_prefetch.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.gptoss_set_cold_prefetch(A.kernel_cold_prefetch, 0)
    elif A.kernel_cold_prefetch:
        print(f"WARNING: {A.kdll} predates gptoss_set_cold_prefetch; rebuild it to use --kernel-cold-prefetch (ignored)", flush=True)
    print(f"CPU kernel tuning: prefetch {A.kernel_prefetch} B, pair {A.kernel_pair}, affinity stride {A.kernel_affinity}, "
          f"fuse {A.kernel_fuse}, cold-prefetch {A.kernel_cold_prefetch}", flush=True)
elif A.kernel_prefetch or A.kernel_pair or A.kernel_affinity or A.kernel_fuse:
    print(f"WARNING: {A.kdll} predates gptoss_set_tuning; rebuild it with tools\\build_kernels.bat to use "
          "--kernel-prefetch/--kernel-pair/--kernel-affinity/--kernel-fuse (ignored)", flush=True)
import neural as _neural_pkg                                         # noqa: E402
print(f"runtime package: {os.path.dirname(_neural_pkg.__file__)}", flush=True)
print(f"expert store: {STORE} ({SLAYOUT.weight_repr}, {SLOTB:,} B/slot, scale layout {SCALE_MODE}); decode kernel {A.kdll}", flush=True)
if A.kernel_persistent:
    if not A.kernel_fuse or not hasattr(lib, "gptoss_set_persistent"):
        raise SystemExit("--kernel-persistent 1 requires --kernel-fuse 1 and a compatible --kdll")
    lib.gptoss_set_persistent.argtypes = [ctypes.c_int]
    lib.gptoss_set_persistent.restype = ctypes.c_int
    lib.gptoss_persistent_shutdown.argtypes = []
    lib.gptoss_persistent_shutdown.restype = ctypes.c_int
    lib.gptoss_persistent_status.argtypes = [ctypes.c_void_p]
    lib.gptoss_persistent_status.restype = None
    if lib.gptoss_set_persistent(1):
        raise SystemExit("failed to enable persistent CPU team")
    import atexit
    atexit.register(lib.gptoss_persistent_shutdown)
    print("persistent CPU team: enabled", flush=True)


def _persistent_snapshot():
    if not A.kernel_persistent:
        return None
    values = (ctypes.c_uint32 * 8)()
    lib.gptoss_persistent_status(values)
    return (int(values[4]) | (int(values[5]) << 32),
            int(values[6]) | (int(values[7]) << 32), int(values[1]))


def _persistent_request_stats(before):
    if before is None:
        return {}
    after = _persistent_snapshot()
    return {"persistent_cpu_jobs": after[0] - before[0],
            "persistent_cpu_fallback_calls": after[1] - before[1],
            "persistent_cpu_team_threads": after[2]}


H, K, NE, NL, GUN = 2880, 4, 128, 36, 5760
_orig_place = _ga.place_core_on_gpu


def _place(model, device):
    _orig_place(model, device)
    model.model.embed_tokens.to("cpu")


_ga.place_core_on_gpu = _place


def free_ram_gib():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.ullAvailPhys / 2**30, 2)


FREE0 = free_ram_gib()
print(f"free RAM at start: {FREE0} GiB", flush=True)
HY0 = None
dev = torch.device("cuda")
model, cfg = load_gptoss_core(CKPT)
from neural.q80.expert_runtime import Q80SlotPoolRuntimeV2 as _Q80RT                # noqa: E402
_Q80RT.N_SCRATCH = max(1, A.scratch)          # prefill staging slots carved out of the pool (v4: 16)
path = build_gptoss_neural(model, dev, STORE, budget_gib=A.pool, pinned_gib=0.25,
                           pageable_gib=0.25, admission_v4=False, layout=SLAYOUT)
rt, ps = path["runtime"], path["pstore"]
assert rt.L.slot_bytes == ps.slot_bytes == SLOTB == rt.pool.shape[1], (rt.L.slot_bytes, ps.slot_bytes, SLOTB)
if A.direct_prefill:
    path["host"]._prepacked_direct = True        # prefill H2D straight from the mmap row view
    path["host"].pageable_capacity = 0           # no pageable demotion copies
rt.require_graph = True
if A.kv_ring:
    # Sliding-window layers get a ring of --kv-ring positions instead of smax: allocate that size
    # from the start so the full-length K/V of those 18 layers never exists (FusedCore re-points
    # lg.k / lg.v at ring tensors anyway). lg.smax keeps the real context length for the full layers
    # and the split-K attention. The eager reference path (_LayerGraph._compute, load_prefill_kv)
    # writes absolute positions and is therefore invalid for sliding layers: guarded below.
    assert not A.eager_core, "--kv-ring needs the fused core"
    import neural.moe.gptoss_core_graphs as _gcg
    _LG_init = _gcg._LayerGraph.__init__

    def _lg_init(self, layer, pos_dev, smax, dev):
        sl = getattr(layer.self_attn, "sliding_window", None)
        _LG_init(self, layer, pos_dev, A.kv_ring if sl else smax, dev)
        self.smax = smax
    _gcg._LayerGraph.__init__ = _lg_init
cg = GptOssCoreGraphs(model, rt, dev, smax=A.smax)          # buffers only; we capture our own graphs
sys.path.insert(0, S)
from fused_core import FusedCore, fetch, bell_wait                              # noqa: E402
from doorbell import Doorbell, DoorbellTimeout                                  # noqa: E402
FC = None if A.eager_core else FusedCore(cg, dev, split_k=A.split_k, ring=A.kv_ring,
                                         near_miss=bool(A.near_miss))   # before ANY capture (re-points q/k/v weights)
LAY = cg.layers
N_USABLE = rt.capacity                  # V2 runtime already excludes its scratch slots from capacity
bias = torch.load(os.path.join(STORE, "expert_biases.pt"), weights_only=True)
BGU = np.ascontiguousarray(bias["bias_gu"].float().numpy())
BDN = np.ascontiguousarray(bias["bias_dn"].float().numpy())
ROWP = [ps.row(g // NE, g % NE).data_ptr() for g in range(NL * NE)]
BGP = [BGU.ctypes.data + g * GUN * 4 for g in range(NL * NE)]
BDP = [BDN.ctypes.data + g * H * 4 for g in range(NL * NE)]


def _embed_memmap():
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    p = os.path.join(CKPT, idx["weight_map"]["model.embed_tokens.weight"])
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    m = hdr["model.embed_tokens.weight"]
    assert m["dtype"] == "BF16", m["dtype"]
    return np.memmap(p, dtype=np.uint16, mode="r", offset=8 + n + m["data_offsets"][0],
                     shape=tuple(m["shape"]))


if A.mmap_embed:
    EMM = _embed_memmap()
    _w = model.model.embed_tokens.weight
    for _t in (0, 1, 17, 1234, 123456, 200000):
        assert torch.equal(torch.from_numpy(np.array(EMM[_t])).view(torch.bfloat16), _w[_t]), _t
    model.model.embed_tokens.weight.data = torch.empty(0, H, dtype=torch.bfloat16)
    del _w
    import gc
    gc.collect()

    def EMB_ROWS(ids):
        return torch.from_numpy(np.array(EMM[np.asarray(ids, dtype=np.int64)])).view(torch.bfloat16)
    print("embedding: read-only memmap of the checkpoint tensor (6 rows verified bit-equal); CPU copy freed", flush=True)
else:
    _EW = model.model.embed_tokens.weight

    def EMB_ROWS(ids):
        return _EW[torch.as_tensor(np.asarray(ids, dtype=np.int64))]

# ------------------------------------------------------------ static buffers
bf = torch.bfloat16
mid = torch.zeros(1, H, dtype=bf, device=dev)
g_out = torch.zeros(1, H, dtype=bf, device=dev)
c_out = torch.zeros(1, H, dtype=torch.float32, device=dev)
bx = torch.zeros(1, H, dtype=bf, device=dev)
bslots = torch.zeros(K, dtype=torch.long, device=dev)
bgids = torch.zeros(K, dtype=torch.long, device=dev)
bw = torch.zeros(K, dtype=bf, device=dev)
PACKN = H + 3 * K + K            # h_norm | 4 weights | 4 ids | 4 hit flags | 4 near-miss ids (ranks 5-8, fused core only)
NM_OFF = H + 3 * K
pack_dev = torch.zeros(PACKN, dtype=torch.float32, device=dev)
pack_pin = torch.zeros(PACKN, dtype=torch.float32).pin_memory()
pack_np = pack_pin.numpy()
out_pin = torch.zeros(1, H, dtype=torch.float32).pin_memory()
out_np = out_pin.numpy()
slot_tab = torch.full((NL * NE,), -1, dtype=torch.long, device=dev)
id_dev = torch.zeros(1, dtype=torch.long, device=dev)
id_pin = torch.zeros(1, dtype=torch.long).pin_memory()
emb_pin = torch.zeros(1, H, dtype=bf).pin_memory()
BELL = (Doorbell(torch.zeros(8, dtype=torch.int32).pin_memory(), timeout_ms=A.doorbell_timeout_ms,
                 faults_max=A.doorbell_faults_max, warm_tokens=A.doorbell_warm_tokens,
                 max_timeouts=A.doorbell_max_timeouts, inject=A.doorbell_inject) if A.doorbell else None)


def a_body(L):
    lg = LAY[L]
    if L == 0:
        lg.x_in.copy_(mid)
    else:
        lg.x_in.copy_(mid + (g_out + c_out.to(bf)))
    lg._compute(cg.inv_freq, cg.att_scale)
    ids = lg.r_idx[0]
    sl = slot_tab.index_select(0, ids + L * NE)
    m = sl >= 0
    bslots.copy_(torch.where(m, sl, torch.zeros_like(sl)))
    bgids.copy_(ids + L * NE)
    bw.copy_(lg.r_scores[0] * m.to(bf))
    bx.copy_(lg.h_norm)
    pack_dev[:H].copy_(lg.h_norm[0])
    pack_dev[H:H + K].copy_(lg.r_scores[0])
    pack_dev[H + K:H + 2 * K].copy_(ids)
    pack_dev[H + 2 * K:H + 3 * K].copy_(m)        # (eager path: no near-miss ranks; pack[NM_OFF:] stays 0)
    mid.copy_(lg.h_mid)


def b_body():
    gemv = mxfp4_gemv_masked if A.masked_gemv else mxfp4_gemv
    mask_args = (bw,) if A.masked_gemv else ()
    gu = gemv(bx, rt.gate_blocks, rt.gate_scales, bslots, *mask_args,
              block_n=32, block_g=4, num_warps=8).view(K, GUN)
    gu = gu + rt.bias_gu.index_select(0, bgids)
    h = rt._act(gu)
    y = gemv(h, rt.down_blocks, rt.down_scales, bslots, *mask_args, per_expert_x=True,
             block_n=32, block_g=4, num_warps=8)
    y = y + rt.bias_dn.index_select(0, bgids)
    g_out.copy_((y * bw.view(K, 1)).sum(0, keepdim=True))


def t_body():
    h = mid + (g_out + c_out.to(bf))
    hn = LAY[0]._rms(model.model.norm, h)
    logits = F.linear(hn, model.lm_head.weight)
    id_dev.copy_(logits.argmax(dim=-1))


gpool = torch.cuda.graph_pool_handle()


def capture(fn):
    if BELL is not None:
        BELL.open()                     # the eager warm-up run below must not wait for a host ring (nothing is in flight)
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=gpool):
        fn()
    return g


EVX = [torch.cuda.Event(external=True) for _ in range(NL)]


ZC = FC is not None and not A.no_zerocopy
NM_ON = bool(A.near_miss and ZC)                 # k_topk writes ranks 5-8 into the pack (fused core only)
NEAR_MISS = bool(A.near_miss >= 2 and ZC)        # ...and they mark admission candidates
if A.near_miss and not ZC:
    print("WARNING: --near-miss needs the fused zero-copy core; ignored", flush=True)


def _c_fetch():
    """c_out <- the CPU experts' output of the previous layer (pinned out_pin): the first work of every graph that
    consumes it. With --doorbell that is preceded by the spin node, so the graph may be launched before the CPU
    kernel has finished (doorbell.py). Without it: exactly the fetch of the previous build."""
    if BELL is not None:
        bell_wait(BELL.pin, A.doorbell_poll_cap)
    if ZC:
        fetch(out_pin, c_out)
    else:
        c_out.copy_(out_pin, non_blocking=True)


def a_full(L):
    if L > 0:
        _c_fetch()
    if FC is None:
        a_body(L)
    elif ZC:
        FC.run(L, mid, g_out, c_out, slot_tab, bx, bslots, bgids, bw, pack_pin)
    else:
        FC.run(L, mid, g_out, c_out, slot_tab, bx, bslots, bgids, bw, pack_dev)
    if not ZC:
        pack_pin.copy_(pack_dev, non_blocking=True)
    EVX[L].record()
    b_body()


def t_full():
    _c_fetch()
    t_body()
    id_pin.copy_(id_dev, non_blocking=True)


with torch.inference_mode():
    if A.no_fold:
        GA = [capture(lambda L=L: a_body(L)) for L in range(NL)]
        GB = capture(b_body)
        GT = capture(t_body)
    else:
        GA = [capture(lambda L=L: a_full(L)) for L in range(NL)]
        GB = None
        GT = capture(t_full)
    GA_REF = [capture(lambda L=L: a_body(L)) for L in range(NL)] if not (A.skip_quality or A.kv_ring) else None
print(f"fused graphs captured; usable slots={N_USABLE}; vram_reserved="
      f"{torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
if BELL is not None:
    print(f"DOORBELL ON: the {NL - 1} layer graphs GA[1:] and the tail graph(s) start with a spin node (k_bell); in steady "
          f"state the host launches the next graph BEFORE the CPU kernel, then rings; timeout {A.doorbell_timeout_ms} ms "
          f"(a timeout re-runs the token in the classic order; early launch off after {A.doorbell_max_timeouts or 'never'}), "
          f"classic order for the first {A.doorbell_warm_tokens} tokens of a request and after a token with more than "
          f"{A.doorbell_faults_max} page faults, poll cap {A.doorbell_poll_cap}, flush-after-launch {A.doorbell_flush}"
          + (f", TEST INJECTION every {A.doorbell_inject} early tokens" if A.doorbell_inject else ""), flush=True)
else:
    print("doorbell: off (classic sequence: CPU kernel -> bookkeeping -> launch next graph)", flush=True)

P = (VP * K)()
BG = (VP * K)()
BD = (VP * K)()
WV = np.zeros(K, np.float32)
XP, OP, WP = pack_pin.data_ptr(), out_pin.data_ptr(), WV.ctypes.data
SCR = np.zeros(K * (GUN + 3 * H) + H + 64, np.float32)
SP = SCR.ctypes.data
EV = torch.cuda.Event()
C = {}
REF = None


def reset():
    C.update(cpu=0, hit=0, gpuadm=0, gpumiss=0, wait_s=0.0, cpu_s=0.0, replay_s=0.0, adm_s=0.0)


reset()
TRACE = []                       # per token: ([36, 4] ids int16, [36, 4] weights f16, [36, 4] near-miss ids int16 | None, hidden f16[2880] | None)
COUNTS = np.zeros((NL, NE), np.int64)
MODE = {"counting": False, "trace": False, "trace_hidden": False, "ref": False}



# ---------------------------------------------------------------- copy-on-compute admission
from collections import OrderedDict, deque


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


_psapi = ctypes.WinDLL("psapi")
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = ctypes.c_void_p
_psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]


class _MRE(ctypes.Structure):
    _fields_ = [("VirtualAddress", ctypes.c_void_p), ("NumberOfBytes", ctypes.c_size_t)]


_k32.PrefetchVirtualMemory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulong]
_HPROC = _k32.GetCurrentProcess()


def disk_read_bytes():
    """System-wide disk bytes read (all processes); 0 if Windows exposes no disk counters."""
    try:
        d = psutil.disk_io_counters()
        return d.read_bytes if d else 0
    except Exception:
        return 0


def pmem():
    m = PMC()
    m.cb = ctypes.sizeof(PMC)
    _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
    return m.PageFaultCount, m.WorkingSetSize / 2**30, m.PrivateUsage / 2**30


CAPB = [torch.empty(SLOTB, dtype=torch.uint8).pin_memory() for _ in range(A.capbufs)]   # SLOTB: from the store (top of file)
CAPEV = [torch.cuda.Event() for _ in range(A.capbufs)]
CAPSTATE = [0] * A.capbufs
PEND = []
PENDG = set()
COPYS = torch.cuda.Stream()
CP = (VP * K)()
UPD = 64
upd_idx_pin = torch.zeros(UPD, dtype=torch.long).pin_memory(); upd_idx_np = upd_idx_pin.numpy()
upd_val_pin = torch.zeros(UPD, dtype=torch.long).pin_memory(); upd_val_np = upd_val_pin.numpy()
upd_idx_dev = torch.zeros(UPD, dtype=torch.long, device=dev)
upd_val_dev = torch.zeros(UPD, dtype=torch.long, device=dev)
TAB = np.full(NL * NE, -1, np.int64)
OWNER = None
GC = None
GEN = np.zeros(NL * NE, np.int64)
PREF = np.zeros(NL * NE, np.int64)      # prompt routing counts of the current request; folded into GEN after prefill
CAND = set()
RC = {"captures": 0, "cand_marked": 0}
BIG = np.int64(1) << 40
# ---- experimental decode levers: state read by decode_token / pick_victim (defaults = v4 behaviour;
# setup_levers() flips ADMIT_GPU / POLICY_STATE / PROBE / ARENA_* after their self-tests)
GPU_MISS = False                                 # slot-GEMV graphs ready: misses can be computed by the GPU
ADMIT_GPU = False                                # ...and admitted (kept in a victim slot) per --admit-rule
ZC_MISSES = 0                                    # ...and up to this many extra misses per layer go to scratch slots
POLICY_STATE = False
ARENA = None
ARENA_ROWS = set()
PROBE = None
TOKI = [0]
LAST = np.full(NL * NE, -1, np.int64)            # token index of last use
WCNT = np.zeros(NL * NE, np.int64)               # uses in the last --admit-window tokens
WHIST = deque()
DEMAND = np.zeros(NL * NE, np.float32)           # predicted near-future demand (probe)
VMIN = [0.0]
ADM_RING = 64
ADMI = [0]
ZCI = [0]
ADMEV = ADMEV2 = ADM_PIN = ADM_WPIN = ADM_DEV = ADM_WDEV = GADM = None


def apply_tab_updates(pairs):
    if not pairs:
        return
    last = {}
    for g, s in pairs:
        last[g] = s
    pairs = list(last.items())
    n = len(pairs)
    upd_idx_np[:n] = [g for g, _ in pairs]
    upd_val_np[:n] = [s for _, s in pairs]
    upd_idx_dev[:n].copy_(upd_idx_pin[:n], non_blocking=True)
    upd_val_dev[:n].copy_(upd_val_pin[:n], non_blocking=True)
    slot_tab.index_copy_(0, upd_idx_dev[:n], upd_val_dev[:n])


def free_capbuf():
    for b in range(A.capbufs):
        if CAPSTATE[b] == 0 or (CAPSTATE[b] == 2 and CAPEV[b].query()):
            CAPSTATE[b] = 1
            return b
    return -1


def refresh_point(m=None):
    m = A.refresh_m if m is None else m
    marked = 0
    for g in np.argsort(-GEN, kind="stable"):         # deterministic among ties (small GEN values tie often)
        if GEN[g] < 2 or marked >= m:
            break
        g = int(g)
        if TAB[g] < 0 and g not in CAND and g not in PENDG:
            CAND.add(g)
            marked += 1
    RC["cand_marked"] += marked


def victim_keys():
    """Eviction key per adaptive slot (lower = evict first): v4 = uses in this request then calibration
    count; --victim lru = token of last use; --admit-rule demand = predicted near-future demand."""
    own = OWNER
    o = np.maximum(own, 0)
    if A.admit_rule == "demand":
        keys = np.where(own >= 0, (DEMAND[o] * 1e6).astype(np.int64), np.iinfo(np.int64).max)
    elif A.victim == "lru":
        keys = np.where(own >= 0, LAST[o], np.iinfo(np.int64).max)
    else:
        keys = np.where(own >= 0, GEN[o] * BIG + GC[o], np.iinfo(np.int64).max)
    return keys


def pick_victim(exclude, touched=()):
    global STATIC_N
    keys = victim_keys()
    for s in touched:
        keys[s] = np.iinfo(np.int64).max
    keys[:STATIC_N] = np.iinfo(np.int64).max
    for g in exclude:
        s = TAB[g]
        if s >= 0:
            keys[s] = np.iinfo(np.int64).max
    v = int(np.argmin(keys))
    assert v >= STATIC_N and keys[v] < np.iinfo(np.int64).max, v
    return v




VERIFY = []


def verify_residency(tag, nsample=16):
    torch.cuda.synchronize()
    dev_tab = slot_tab.cpu().numpy()
    bad_tab = int((dev_tab != TAB).sum())
    own_ok = all(TAB[OWNER[s]] == s for s in range(N_USABLE) if OWNER[s] >= 0)
    rng = np.random.default_rng(len(VERIFY))
    res_slots = [s for s in range(N_USABLE) if OWNER[s] >= 0]
    bad_bytes = 0
    for s in rng.choice(res_slots, size=min(nsample, len(res_slots)), replace=False):
        g = int(OWNER[s])
        if not torch.equal(rt._rows[int(s)].cpu(), ps.row(g // NE, g % NE)):
            bad_bytes += 1
        if int(s) < STATIC_N:
            release_rows([g])                  # the check touched a released static row
    v = {"tag": tag, "device_vs_host_table_mismatches": bad_tab, "owner_table_consistent": bool(own_ok),
         "sampled_slots": min(nsample, len(res_slots)), "sampled_slot_byte_mismatches": bad_bytes}
    VERIFY.append(v)
    print("VERIFY", json.dumps(v), flush=True)
    return v


_orig_prefill = rt.forward_prefill


def _counting_prefill(layer, x, top_k_index, top_k_weights):
    if COUNT_PREFILL[0]:
        np.add.at(PREF, layer * NE + top_k_index.reshape(-1).cpu().numpy().astype(np.int64), 1)
    return _orig_prefill(layer, x, top_k_index, top_k_weights)


rt.forward_prefill = _counting_prefill
COUNT_PREFILL = [False]


def _decode_token_classic(tok, pos):
    cg.pos_pin[0] = pos
    cg.pos.copy_(cg.pos_pin, non_blocking=True)
    emb_pin.copy_(EMB_ROWS([tok]).view(1, H))
    mid.copy_(emb_pin, non_blocking=True)
    tr_i = np.zeros((NL, K), np.int16) if MODE["trace"] else None
    tr_w = np.zeros((NL, K), np.float16) if MODE["trace"] else None
    tr_nm = np.zeros((NL, K), np.int16) if (MODE["trace"] and NM_ON) else None
    poll = A.wait == "poll"
    tok_gids = []
    for L in range(NL):
        _tr = time.perf_counter()
        (GA_REF if MODE["ref"] else GA)[L].replay()
        C["replay_s"] += time.perf_counter() - _tr
        if MODE["ref"]:                                      # quality reference: eager store path
            lg = LAY[L]
            g_out.copy_(REF.forward(L, lg.h_norm, lg.r_idx, lg.r_scores))
            c_out.zero_()
            continue
        if GB is not None:
            pack_pin.copy_(pack_dev, non_blocking=True)
            EV.record()
            GB.replay()
            _ev = EV
        else:
            _ev = EVX[L]
        t0 = time.perf_counter()
        if poll:
            while not _ev.query():          # busy-poll: no blocking-sync wake-up latency
                pass
        else:
            _ev.synchronize()
        C["wait_s"] += time.perf_counter() - t0
        ids = pack_np[H + K:H + 2 * K]
        msk = pack_np[H + 2 * K:NM_OFF]
        if MODE["counting"]:
            COUNTS[L, ids.astype(np.int64)] += 1
        GEN[L * NE + ids.astype(np.int64)] += 1
        if tr_i is not None:
            tr_i[L] = ids
            tr_w[L] = pack_np[H:H + K]
            if tr_nm is not None:
                tr_nm[L] = pack_np[NM_OFF:NM_OFF + K]
        if NEAR_MISS:
            # the router's ranks 5-8: an expert just under the top-4 cut is likely to be selected soon.
            # Marked as admission candidates (server rule); the top-4 and their weights are untouched.
            for v in pack_np[NM_OFF:NM_OFF + K]:
                g = L * NE + int(v)
                if TAB[g] < 0 and g not in CAND and g not in PENDG:
                    CAND.add(g)
                    RC["cand_marked"] += 1
        E = 0
        caps = []
        gadm = []                                     # (g, j, admit): misses the GPU computes this layer
        base = L * NE
        cur = None
        if POLICY_STATE:
            tok_gids.append(base + ids.astype(np.int64))
        misses = [(base + int(ids[j]), j) for j in range(K) if msk[j] == 0.0]
        if GPU_MISS and misses:
            rest = []
            for g, j in misses:                       # 1. admissions: kept in a victim slot
                if (ADMIT_GPU and len(gadm) < A.admit_gpu_max and g in ARENA_ROWS and g not in PENDG
                        and admit_ok(g)):
                    gadm.append((g, j, True))
                else:
                    rest.append((g, j))
            n_extra = 0                               # 2. extra read pipe: scratch slots, never the last CPU miss
            while ZC_MISSES and n_extra < ZC_MISSES and len(rest) >= 2 and rest[-1][0] in ARENA_ROWS:
                g, j = rest.pop()
                gadm.append((g, j, False))
                n_extra += 1
            misses = rest
        for g, j in misses:
            P[E] = ROWP[g]
            BG[E] = BGP[g]
            BD[E] = BDP[g]
            WV[E] = pack_np[H + j]
            CP[E] = None
            if not ADMIT_GPU and g in CAND:
                bb = free_capbuf()
                if bb >= 0:
                    CP[E] = CAPB[bb].data_ptr()
                    caps.append((g, bb))
            E += 1
        C["hit"] += K - E - len(gadm)
        upd = []
        touched = set()
        if gadm:
            # GPU-computed misses, enqueued before the CPU kernel blocks the host: the copy engine streams the
            # expert's bytes from the pinned arena into a slot (DMA, no CPU, no capture buffer) and the pool GEMV
            # computes it there, concurrently with the CPU's misses. Admitted experts land in a victim slot and
            # flip when done (zero-surcharge admission); extra misses land in a scratch slot.
            _ta = time.perf_counter()
            cur = [base + int(v) for v in ids]
            main = torch.cuda.current_stream()
            for k_, (g, j, admit) in enumerate(gadm):
                if admit:
                    s = pick_victim(cur, touched)
                    touched.add(s)
                    vg = int(OWNER[s])
                    upd.append((vg, -1))
                    TAB[vg] = -1
                    OWNER[s] = -1
                    CAND.discard(g)
                    PENDG.add(g)
                    LAST[g] = TOKI[0] + 1              # counts as used now (LRU victims would otherwise evict it at once)
                else:
                    s = SCR_SLOTS[ZCI[0] % len(SCR_SLOTS)]
                    ZCI[0] += 1
                i = ADMI[0] % ADM_RING
                ADMI[0] += 1
                ADMEV2[i].synchronize()                # ring entry i (pinned params + events) is free again
                COPYS.wait_stream(main)
                if not admit and SLOT_FREE[s] is not None:
                    COPYS.wait_event(SLOT_FREE[s])     # the last GEMV that read this scratch slot
                with torch.cuda.stream(COPYS):
                    rt._rows[s].copy_(ARENA.row_view(g), non_blocking=True)
                    ADMEV[i].record(COPYS)
                main.wait_event(ADMEV[i])
                ADM_PIN[i][0] = s
                ADM_PIN[i][1] = g
                ADM_WPIN[i][0] = float(pack_np[H + j])
                ADM_DEV[k_].copy_(ADM_PIN[i], non_blocking=True)
                ADM_WDEV[k_].copy_(ADM_WPIN[i], non_blocking=True)
                GADM[k_].replay()
                ADMEV2[i].record(main)
                if admit:
                    PEND.append((g, s, -1 - i))        # negative = GPU admission, flips when ADMEV2[-1 - bb] lands
                    C["gpuadm"] += 1
                else:
                    SLOT_FREE[s] = ADMEV2[i]
                    C["gpumiss"] += 1
            # the victims' -1 entries ride the layer's single apply_tab_updates below (before the next
            # replay); a second apply here would reuse its pinned staging buffers while their copy is in flight
            C["adm_s"] += time.perf_counter() - _ta
        if E:
            C["cpu"] += E
            t1 = time.perf_counter()
            lib.gptoss_experts_cap(E, P, XP, BG, BD, WP, OP, SP, A.threads, CP)
            C["cpu_s"] += time.perf_counter() - t1
        else:
            out_np[:] = 0.0
        _ta = time.perf_counter()
        if PEND:
            keep = []
            for g, s, bb in PEND:
                done = CAPEV[bb].query() if bb >= 0 else ADMEV2[-1 - bb].query()
                if done:
                    upd.append((g, s))
                    touched.add(s)
                    PENDG.discard(g)
                    TAB[g] = s
                    OWNER[s] = g
                    if bb >= 0:
                        CAPSTATE[bb] = 0
                else:
                    keep.append((g, s, bb))
            PEND[:] = keep
        newc = []
        if caps:
            if cur is None:
                cur = [base + int(v) for v in ids]
            for g, bb in caps:
                s = pick_victim(cur, touched)
                touched.add(s)
                vg = int(OWNER[s])
                upd.append((vg, -1))
                TAB[vg] = -1
                OWNER[s] = -1
                CAND.discard(g)
                PENDG.add(g)
                newc.append((g, s, bb))
                RC['captures'] += 1
        apply_tab_updates(upd)
        if newc:
            COPYS.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(COPYS):
                for g, s, bb in newc:
                    rt._rows[s].copy_(CAPB[bb], non_blocking=True)
                    CAPEV[bb].record(COPYS)
                    CAPSTATE[bb] = 2
                    PEND.append((g, s, bb))
        C["adm_s"] += time.perf_counter() - _ta
        if GB is not None:
            c_out.copy_(out_pin, non_blocking=True)
    GT.replay()
    if GB is not None:
        id_pin.copy_(id_dev, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    if POLICY_STATE and tok_gids:
        # per-token policy state: last use (LRU victims), windowed counts (--admit-rule window),
        # predicted near-future demand from the final residual (--admit-rule demand)
        gids = np.concatenate(tok_gids)
        TOKI[0] += 1
        LAST[gids] = TOKI[0]
        if A.admit_rule == "window":
            np.add.at(WCNT, gids, 1)
            WHIST.append(gids)
            if len(WHIST) > A.admit_window:
                np.subtract.at(WCNT, WHIST.popleft(), 1)
        if PROBE is not None:
            hfin = (mid + (g_out + c_out.to(bf))).float().view(-1)
            sc = torch.sigmoid((((hfin - PROBE["mu"]) / PROBE["sd"]).to(PROBE["W"].dtype) @ PROBE["W"]).float() + PROBE["b"])
            DEMAND[:] = sc.cpu().numpy()
            own = OWNER[STATIC_N:]
            own = own[own >= 0]
            VMIN[0] = float(DEMAND[own].min()) if own.size else 0.0
    if tr_i is not None:
        hid = None
        if MODE["trace_hidden"]:
            # final residual (the tail graph's input, before the final norm): mid + g_out + c_out of layer 35
            hid = (mid + (g_out + c_out.to(bf))).to(torch.float16).cpu().numpy().reshape(H)
        if len(TRACE) < A.trace_max_tokens:
            TRACE.append((tr_i, tr_w, tr_nm, hid))
    return int(id_pin[0])


def _book_tab(ids, base, cur, caps, upd, touched):
    """Per-layer slot-table bookkeeping that follows the CPU kernel in the classic order (the same statements as
    _decode_token_classic): flip finished admissions (PEND scan), pick victims for this layer's captures, enqueue ONE tab
    update on the main stream. Returns the captures whose H2D copies _copies_issue() enqueues. Nothing here reads the CPU
    kernel's output, so with --doorbell it runs BEFORE the kernel (the next graph must see the update and is launched
    before the kernel). ids is only read here, before that launch."""
    if PEND:
        keep = []
        for g, s, bb in PEND:
            done = CAPEV[bb].query() if bb >= 0 else ADMEV2[-1 - bb].query()
            if done:
                upd.append((g, s))
                touched.add(s)
                PENDG.discard(g)
                TAB[g] = s
                OWNER[s] = g
                if bb >= 0:
                    CAPSTATE[bb] = 0
            else:
                keep.append((g, s, bb))
        PEND[:] = keep
    newc = []
    if caps:
        if cur is None:
            cur = [base + int(v) for v in ids]
        for g, bb in caps:
            s = pick_victim(cur, touched)
            touched.add(s)
            vg = int(OWNER[s])
            upd.append((vg, -1))
            TAB[vg] = -1
            OWNER[s] = -1
            CAND.discard(g)
            PENDG.add(g)
            newc.append((g, s, bb))
            RC['captures'] += 1
    apply_tab_updates(upd)
    return newc


def _copies_issue(newc):
    """H2D copies of the captured experts on the copy stream (after COPYS.wait_stream(main)). They read CAPB[bb], which the
    CPU kernel of THIS layer has just filled, so they are enqueued after the kernel returned (also with --doorbell)."""
    with torch.cuda.stream(COPYS):
        for g, s, bb in newc:
            rt._rows[s].copy_(CAPB[bb], non_blocking=True)
            CAPEV[bb].record(COPYS)
            CAPSTATE[bb] = 2
            PEND.append((g, s, bb))


def _bell_abort(where):
    """A spin node gave up (the host did not ring within --doorbell-timeout-ms): the data its graph then read is garbage.
    Wait for the queued graphs to finish and hand the token to decode_token(), which re-runs it."""
    torch.cuda.synchronize()
    raise DoorbellTimeout(f"doorbell: the spin before layer {where} gave up ({BELL.describe()})")


def _bell_recover(e, snap):
    """Recovery from a timeout (doorbell.py, TIMEOUT AND RECOVERY, has the argument). The graph that gave up (GA[L+1], or the
    tail) ran on stale data and overwrote mid and g_out, which it also reads: it cannot be repaired or replayed in place. What
    it wrote is otherwise per-token output that a re-run rewrites (pack_pin, out_pin, c_out, router buffers, K/V at `pos`, the
    token id) and slot-table / capture state it did not touch. So the whole token is re-run from layer 0 in the classic order
    (a ring before every launch: a spin exits at once). The host counters the failed attempt already advanced are put back;
    residency bookkeeping (tab updates, captures, PEND) is real device state and stays."""
    BELL.armed = False                                  # (nothing is armed at this point: ring() runs in a finally)
    BELL.open()
    GEN[:] = snap[0]
    C["cpu"], C["hit"] = snap[1], snap[2]
    if snap[3] is not None:
        COUNTS[:] = snap[3]
    last = BELL.recovered()
    print(f"{e}; token re-run in the classic order (recovery {BELL.recoveries}"
          f"{'; early launch now OFF' if last else ''})", flush=True)


def decode_token(tok, pos, doorbell=False):
    """One decode step.
    BELL is None (--doorbell 0, the default): _decode_token_classic, the previous build's loop, unchanged.
    --doorbell 1: doorbell=True (the serving generate()) lets BELL.choose() pick the order of THIS token (classic during the
    first --doorbell-warm-tokens of a request and after a token with more than --doorbell-faults-max page faults, early
    launch otherwise); a timeout is not an error: the token is re-run in the classic order. doorbell=False (calibration,
    warm-up) is always the classic order on the spin graphs."""
    if BELL is None:
        return _decode_token_classic(tok, pos)
    if not doorbell:
        return _decode_token_bell(tok, pos, False)
    early, _why = BELL.choose(pmem()[0])
    if not early:
        return _decode_token_bell(tok, pos, False)
    snap = (GEN.copy(), C["cpu"], C["hit"], COUNTS.copy() if MODE["counting"] else None)
    try:
        r = _decode_token_bell(tok, pos, True)
    except DoorbellTimeout as e:
        _bell_recover(e, snap)
    else:
        BELL.req["tokens"] += 1
        return r
    return _decode_token_bell(tok, pos, False)             # a timeout inside the classic-order retry is a hardware fault: it propagates


def _bell_stats(n):
    """The per-request doorbell keys of generate()'s stats (zeros when --doorbell 0)."""
    if BELL is None:
        return {"doorbell": 0, "doorbell_tokens": 0, "doorbell_classic_tokens": 0, "doorbell_timeouts": 0}
    return BELL.stats(n)


def _decode_token_bell(tok, pos, early):
    """decode_token for a --doorbell 1 build: the same computation as _decode_token_classic (the previous build's loop,
    above, textually unchanged; tests/test_doorbell_mock.py proves the two issue the same operations), for graphs GA[1:] and
    GT that start with a spin node (fused_core.k_bell).
      early=True   launch each of them BEFORE the CPU expert kernel that feeds it, ring the doorbell (doorbell.py) once the
                   kernel's output is complete
      early=False  the classic order, with a ring before every launch: the spin exits at once
    A spin that gave up is noticed at the router event of its graph (BELL.read()): _bell_abort raises DoorbellTimeout and
    decode_token() re-runs the token."""
    cg.pos_pin[0] = pos
    cg.pos.copy_(cg.pos_pin, non_blocking=True)
    emb_pin.copy_(EMB_ROWS([tok]).view(1, H))
    mid.copy_(emb_pin, non_blocking=True)
    tr_i = np.zeros((NL, K), np.int16) if MODE["trace"] else None
    tr_w = np.zeros((NL, K), np.float16) if MODE["trace"] else None
    tr_nm = np.zeros((NL, K), np.int16) if (MODE["trace"] and NM_ON) else None
    poll = A.wait == "poll"
    tok_gids = []
    early = early and not MODE["ref"]                        # (GA_REF, the quality-reference graphs, have no spin node)
    if early:
        main = torch.cuda.current_stream()
        drop_at = BELL.drop_layer()                          # -1 unless --doorbell-inject (test only)
        _tr = time.perf_counter()
        GA[0].replay()                                       # GA[0] has no CPU dependency (no spin node): launch it now
        C["replay_s"] += time.perf_counter() - _tr
    for L in range(NL):
        if not early:
            if L > 0 and not MODE["ref"]:
                BELL.arm()                                   # classic order on spin graphs: the CPU output is already
                BELL.ring()                                  # complete, so the spin exits at once
            _tr = time.perf_counter()
            (GA_REF if MODE["ref"] else GA)[L].replay()
            C["replay_s"] += time.perf_counter() - _tr
        if MODE["ref"]:                                      # quality reference: eager store path
            lg = LAY[L]
            g_out.copy_(REF.forward(L, lg.h_norm, lg.r_idx, lg.r_scores))
            c_out.zero_()
            continue
        if GB is not None:
            pack_pin.copy_(pack_dev, non_blocking=True)
            EV.record()
            GB.replay()
            _ev = EV
        else:
            _ev = EVX[L]
        t0 = time.perf_counter()
        if poll:
            while not _ev.query():          # busy-poll: no blocking-sync wake-up latency
                pass
        else:
            _ev.synchronize()
        C["wait_s"] += time.perf_counter() - t0
        if L > 0 and BELL.read()[0]:                         # the spin node of GA[L] has finished (its router event fired)
            _bell_abort(L)
        ids = pack_np[H + K:H + 2 * K]
        msk = pack_np[H + 2 * K:NM_OFF]
        if MODE["counting"]:
            COUNTS[L, ids.astype(np.int64)] += 1
        GEN[L * NE + ids.astype(np.int64)] += 1
        if tr_i is not None:
            tr_i[L] = ids
            tr_w[L] = pack_np[H:H + K]
            if tr_nm is not None:
                tr_nm[L] = pack_np[NM_OFF:NM_OFF + K]
        if NEAR_MISS:
            # the router's ranks 5-8: an expert just under the top-4 cut is likely to be selected soon.
            # Marked as admission candidates (server rule); the top-4 and their weights are untouched.
            for v in pack_np[NM_OFF:NM_OFF + K]:
                g = L * NE + int(v)
                if TAB[g] < 0 and g not in CAND and g not in PENDG:
                    CAND.add(g)
                    RC["cand_marked"] += 1
        E = 0
        caps = []
        gadm = []                                     # (g, j, admit): misses the GPU computes this layer
        base = L * NE
        cur = None
        if POLICY_STATE:
            tok_gids.append(base + ids.astype(np.int64))
        misses = [(base + int(ids[j]), j) for j in range(K) if msk[j] == 0.0]
        if GPU_MISS and misses:
            rest = []
            for g, j in misses:                       # 1. admissions: kept in a victim slot
                if (ADMIT_GPU and len(gadm) < A.admit_gpu_max and g in ARENA_ROWS and g not in PENDG
                        and admit_ok(g)):
                    gadm.append((g, j, True))
                else:
                    rest.append((g, j))
            n_extra = 0                               # 2. extra read pipe: scratch slots, never the last CPU miss
            while ZC_MISSES and n_extra < ZC_MISSES and len(rest) >= 2 and rest[-1][0] in ARENA_ROWS:
                g, j = rest.pop()
                gadm.append((g, j, False))
                n_extra += 1
            misses = rest
        for g, j in misses:
            P[E] = ROWP[g]
            BG[E] = BGP[g]
            BD[E] = BDP[g]
            WV[E] = pack_np[H + j]
            CP[E] = None
            if not ADMIT_GPU and g in CAND:
                bb = free_capbuf()
                if bb >= 0:
                    CP[E] = CAPB[bb].data_ptr()
                    caps.append((g, bb))
            E += 1
        C["hit"] += K - E - len(gadm)
        upd = []
        touched = set()
        if gadm:
            # GPU-computed misses, enqueued before the CPU kernel blocks the host: the copy engine streams the
            # expert's bytes from the pinned arena into a slot (DMA, no CPU, no capture buffer) and the pool GEMV
            # computes it there, concurrently with the CPU's misses. Admitted experts land in a victim slot and
            # flip when done (zero-surcharge admission); extra misses land in a scratch slot.
            _ta = time.perf_counter()
            cur = [base + int(v) for v in ids]
            main = torch.cuda.current_stream()
            for k_, (g, j, admit) in enumerate(gadm):
                if admit:
                    s = pick_victim(cur, touched)
                    touched.add(s)
                    vg = int(OWNER[s])
                    upd.append((vg, -1))
                    TAB[vg] = -1
                    OWNER[s] = -1
                    CAND.discard(g)
                    PENDG.add(g)
                    LAST[g] = TOKI[0] + 1              # counts as used now (LRU victims would otherwise evict it at once)
                else:
                    s = SCR_SLOTS[ZCI[0] % len(SCR_SLOTS)]
                    ZCI[0] += 1
                i = ADMI[0] % ADM_RING
                ADMI[0] += 1
                ADMEV2[i].synchronize()                # ring entry i (pinned params + events) is free again
                COPYS.wait_stream(main)
                if not admit and SLOT_FREE[s] is not None:
                    COPYS.wait_event(SLOT_FREE[s])     # the last GEMV that read this scratch slot
                with torch.cuda.stream(COPYS):
                    rt._rows[s].copy_(ARENA.row_view(g), non_blocking=True)
                    ADMEV[i].record(COPYS)
                main.wait_event(ADMEV[i])
                ADM_PIN[i][0] = s
                ADM_PIN[i][1] = g
                ADM_WPIN[i][0] = float(pack_np[H + j])
                ADM_DEV[k_].copy_(ADM_PIN[i], non_blocking=True)
                ADM_WDEV[k_].copy_(ADM_WPIN[i], non_blocking=True)
                GADM[k_].replay()
                ADMEV2[i].record(main)
                if admit:
                    PEND.append((g, s, -1 - i))        # negative = GPU admission, flips when ADMEV2[-1 - bb] lands
                    C["gpuadm"] += 1
                else:
                    SLOT_FREE[s] = ADMEV2[i]
                    C["gpumiss"] += 1
            # the victims' -1 entries ride the layer's single apply_tab_updates below (before the next
            # replay); a second apply here would reuse its pinned staging buffers while their copy is in flight
            C["adm_s"] += time.perf_counter() - _ta
        if early:
            # DOORBELL: everything the next graph must see is enqueued on the main stream BEFORE it: the slot-table updates
            # (_book_tab: nothing there reads the CPU kernel's output) and the copy-stream fence. Only the capture H2D
            # copies wait: they read CAPB, which the CPU kernel is about to fill. Everything that reads pack_pin (ids, masks,
            # weights, the kernel's own x) was read or is read by the kernel before the ring: the next graph overwrites
            # pack_pin only after its spin node, i.e. after the ring.
            _ta = time.perf_counter()
            newc = _book_tab(ids, base, cur, caps, upd, touched)
            if newc:
                COPYS.wait_stream(main)
            C["adm_s"] += time.perf_counter() - _ta
            _tl = _kd = 0.0
            _tarm = time.perf_counter()
            BELL.arm()                                # publish the sequence number this launch waits for
        try:
            if early:
                _tr = time.perf_counter()
                (GA[L + 1] if L + 1 < NL else GT).replay()      # its first node spins until BELL.ring()
                if A.doorbell_flush:
                    main.query()                      # WDDM batches launches: submit the spin graph to the GPU now
                _tl = time.perf_counter() - _tr
                C["replay_s"] += _tl
            if E:
                C["cpu"] += E
                t1 = time.perf_counter()
                lib.gptoss_experts_cap(E, P, XP, BG, BD, WP, OP, SP, A.threads, CP)
                _kd = time.perf_counter() - t1
                C["cpu_s"] += _kd
            else:
                out_np[:] = 0.0
        finally:
            if early:
                BELL.stamp = (L, E, _tl, time.perf_counter() - _tarm, _kd)     # what the host did while the GPU spun
                if L == drop_at:
                    BELL.drop()                       # TEST ONLY (--doorbell-inject): this ring is lost, the spin must time out
                else:
                    BELL.ring()                       # the kernel ended with an sfence: out_pin is complete, release the spin
        _ta = time.perf_counter()
        if early:
            if newc:
                _copies_issue(newc)
        else:
            newc = _book_tab(ids, base, cur, caps, upd, touched)
            if newc:
                COPYS.wait_stream(torch.cuda.current_stream())
                _copies_issue(newc)
        C["adm_s"] += time.perf_counter() - _ta
        if GB is not None:
            c_out.copy_(out_pin, non_blocking=True)
    if not early:
        BELL.arm()                                        # the tail graph starts with a spin node too
        BELL.ring()
        GT.replay()
    if GB is not None:
        id_pin.copy_(id_dev, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    if BELL.read()[0]:                                    # ...the tail graph's spin node has finished
        _bell_abort(NL)
    if POLICY_STATE and tok_gids:
        # per-token policy state: last use (LRU victims), windowed counts (--admit-rule window),
        # predicted near-future demand from the final residual (--admit-rule demand)
        gids = np.concatenate(tok_gids)
        TOKI[0] += 1
        LAST[gids] = TOKI[0]
        if A.admit_rule == "window":
            np.add.at(WCNT, gids, 1)
            WHIST.append(gids)
            if len(WHIST) > A.admit_window:
                np.subtract.at(WCNT, WHIST.popleft(), 1)
        if PROBE is not None:
            hfin = (mid + (g_out + c_out.to(bf))).float().view(-1)
            sc = torch.sigmoid((((hfin - PROBE["mu"]) / PROBE["sd"]).to(PROBE["W"].dtype) @ PROBE["W"]).float() + PROBE["b"])
            DEMAND[:] = sc.cpu().numpy()
            own = OWNER[STATIC_N:]
            own = own[own >= 0]
            VMIN[0] = float(DEMAND[own].min()) if own.size else 0.0
    if tr_i is not None:
        hid = None
        if MODE["trace_hidden"]:
            # final residual (the tail graph's input, before the final norm): mid + g_out + c_out of layer 35
            hid = (mid + (g_out + c_out.to(bf))).to(torch.float16).cpu().numpy().reshape(H)
        if len(TRACE) < A.trace_max_tokens:
            TRACE.append((tr_i, tr_w, tr_nm, hid))
    return int(id_pin[0])



from transformers import AutoTokenizer                                          # noqa: E402
tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
emb = model.model.embed_tokens


@torch.inference_mode()
def generate(prompt, n, stream=None):
    ids = tok(prompt, return_tensors="pt").input_ids
    GEN[:] = 0
    PREF[:] = 0
    CAND.clear()
    COUNT_PREFILL[0] = not MODE["ref"]
    out = model(inputs_embeds=EMB_ROWS(ids[0].tolist()).unsqueeze(0).to(dev), use_cache=True)
    COUNT_PREFILL[0] = False
    GEN[:] += PREF                           # hot-set calibration path: prompt counts at full weight (unchanged)
    PREF[:] = 0
    Pn = ids.shape[1]
    t = int(out.logits[0, -1].argmax())
    cg.load_prefill_kv(out.past_key_values, Pn)
    del out
    toks, argmx = [t], []
    reset()
    RC.update(captures=0, cand_marked=0)
    if A.refresh_every and not MODE["ref"] and A.prefill_m:
        refresh_point(A.prefill_m)
    pf0, _, _ = pmem()
    dk0 = disk_read_bytes()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(n - 1):
        cur = stream[i] if stream is not None else t
        t = decode_token(cur, Pn + i)
        argmx.append(t)
        toks.append(t)
        _every = A.early_every if (i + 1) <= A.early_tokens else A.refresh_every
        if A.refresh_every and not MODE["ref"] and (i + 1) % _every == 0:
            refresh_point()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1
    per = dt / (n - 1) * 1e3
    tot = C["cpu"] + C["hit"] + C["gpuadm"] + C["gpumiss"]
    r = {"tok_s": round((n - 1) / dt, 3), "ms": round(per, 2),
         "hit_rate": round(C["hit"] / max(tot, 1), 4),
         "cpu_experts_per_tok": round(C["cpu"] / (n - 1), 2),
         "cpu_ms": round(C["cpu_s"] / (n - 1) * 1e3, 2),
         "gpu_wait_ms": round(C["wait_s"] / (n - 1) * 1e3, 2),
         "other_ms": round(per - (C["cpu_s"] + C["wait_s"]) / (n - 1) * 1e3, 2),
         "cpu_GBps": round(C["cpu"] * SLOTB / max(C["cpu_s"], 1e-9) / 1e9, 2),
         "replay_ms": round(C["replay_s"] / (n - 1) * 1e3, 2), "admission_ms": round(C["adm_s"] / (n - 1) * 1e3, 2),
         "captures": RC["captures"], "cand_marked": RC["cand_marked"]}
    pf1, ws, pv = pmem()
    dk1 = disk_read_bytes()
    r.update(disk_read_mb_per_tok=round((dk1 - dk0) / (n - 1) / 1e6, 2), free_ram_gib=free_ram_gib())
    r.update(page_faults_per_tok=round((pf1 - pf0) / (n - 1)), ws_gib=round(ws, 2), private_gib=round(pv, 2))
    torch.cuda.synchronize()
    if PEND:
        upd = []
        for g, s, bb in PEND:
            upd.append((g, s)); TAB[g] = s; OWNER[s] = g
            if bb >= 0:
                CAPSTATE[bb] = 0
        PEND.clear()
        PENDG.clear()
        apply_tab_updates(upd)
        torch.cuda.synchronize()
    if OWNER is not None:                    # None while calibrating a hot set (no slots assigned yet)
        so = rt.slot_of
        so.clear()
        for s in range(N_USABLE):
            if OWNER[s] >= 0:
                so[(int(OWNER[s]) // NE, int(OWNER[s]) % NE)] = s
    return r, toks, argmx


# Calibration prompts (only used to build a VRAM hot set when no --hotset file exists) and
# held-out prompts, embedded from the research corpus so the server does not need it.
_CAL = json.load(open(os.path.join(S, "data", "calib_prompts.json"), encoding="utf-8"))
WARM = ("The three most important properties of a well-designed memory hierarchy for "
        "sparse mixture-of-experts inference are")
TEST = {
    "f1": "Explain, step by step, how a compiler decides to inline a function and what can go wrong when it does:",
    "f2": "A short history of the transatlantic telegraph cable, beginning with",
    "f3": "List the practical trade-offs between renting and buying a home in a high-interest-rate environment:",
}
TEST.update(_CAL["test_extra"])
CALIB = list(_CAL["calib"])
assert not (set(CALIB) & set(TEST.values()))

res = {"config": {"doorbell": A.doorbell, "admission": "copy_on_compute_v5_static_core", "core": "eager" if A.eager_core else "triton_fused_v1", "zerocopy_small_transfers": not (A.eager_core or A.no_zerocopy), "direct_prefill": bool(A.direct_prefill), "prefill_m": A.prefill_m, "early_every": A.early_every, "early_tokens": A.early_tokens, "admission_v4_ring": False, "warmup": "12 calib @64 + warm", "refresh_every": A.refresh_every, "refresh_m": A.refresh_m, "capbufs": A.capbufs, "pool_gib": A.pool, "usable_slots": N_USABLE, "threads": A.threads, "smax": A.smax,
                  "tokens": A.tokens, "kdll": A.kdll, "store": STORE, "store_weight_repr": SLAYOUT.weight_repr,
                  "slot_bytes": SLOTB, "free_ram_start_gib": FREE0}, "runs": []}
N = A.tokens

# ---------------------------------------------------------------- hot set
if A.hotset and os.path.exists(A.hotset):
    _hs = json.load(open(A.hotset))
    _cnt = np.array(_hs["counts"], dtype=np.int64).ravel()
    chosen = [(int(g // NE), int(g % NE)) for g in np.argsort(-_cnt)[:N_USABLE]]
    GC = _cnt
    print(f"hot set loaded from {A.hotset}: {len(chosen)}", flush=True)
else:
    if A.kv_ring:
        raise SystemExit("--kv-ring needs a --hotset file: hot-set calibration uses the eager reference "
                         "prefill (absolute K/V positions), which the rolling cache does not support")
    MODE["counting"] = True                          # pool empty -> every expert on the CPU
    _refresh = A.refresh_every
    A.refresh_every = 0                              # no admission: slot bookkeeping is not set up yet
    for i, p in enumerate(CALIB):
        r, _, _ = generate(p, A.calib_tokens)
        print(f"calib {i} (all-CPU experts): {json.dumps(r)}", flush=True)
        res["runs"].append(dict(r, workload="calib_all_cpu", idx=i))
    MODE["counting"] = False
    A.refresh_every = _refresh
    GC = COUNTS.ravel().copy()
    flat = np.argsort(-COUNTS.ravel())
    chosen = [(int(g // NE), int(g % NE)) for g in flat[:N_USABLE]]
    cov = float(COUNTS.ravel()[flat[:N_USABLE]].sum() / COUNTS.sum())
    res["calibration_coverage"] = round(cov, 4)
    json.dump({"chosen": chosen, "coverage": cov, "counts": COUNTS.tolist()},
              open(os.path.join(S, "hotset_freq.json"), "w"))
    print(f"hot set: {len(chosen)} experts cover {cov:.3f} of calibration routing (saved)", flush=True)
so = rt.slot_of
so.clear()
tab = torch.full((NL * NE,), -1, dtype=torch.long)
for s, (l, e) in enumerate(chosen):
    rt._rows[s].copy_(ps.row(l, e))
    so[(l, e)] = s
    tab[l * NE + e] = s
rt.free = []
slot_tab.copy_(tab.to(dev))
OWNER = np.full(N_USABLE, -1, np.int64)
for s, (l, e) in enumerate(chosen):
    OWNER[s] = l * NE + e
    TAB[l * NE + e] = s
torch.cuda.synchronize()
_k32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def release_rows(gids):
    # VirtualUnlock on an unlocked range removes those pages from the working set
    # (verified: WS drops by exactly the range; returns ERROR_NOT_LOCKED by design)
    if A.no_unlock:
        return 0
    for g in gids:
        _k32.VirtualUnlock(ctypes.c_void_p(ROWP[g]), SLOTB)
    return len(gids)


STATIC_N = min(A.static, N_USABLE)
assert STATIC_N <= N_USABLE - 64, "leave at least 64 adaptive slots"
_t0 = time.perf_counter()
# VRAM-resident experts are computed on the GPU: their RAM copies (read once to load the slots) crowd
# out experts the CPU computes, and the store is at the edge of free RAM. Measured: releasing all
# residents (default) gave steadier first answers than releasing the static core only.
_nrel = release_rows([int(OWNER[s]) for s in range(N_USABLE if A.release_resident else STATIC_N) if OWNER[s] >= 0])
_, _ws, _pv = pmem()
print(f"static core: {STATIC_N} slots protected; {_nrel} VRAM-resident experts' store pages released from RAM in "
      f"{time.perf_counter() - _t0:.2f}s; WS {_ws:.2f} GiB, private commit {_pv:.2f} GiB, "
      f"free RAM {free_ram_gib()} GiB", flush=True)
res["config"].update(static=STATIC_N, unlock=not A.no_unlock, fold=not A.no_fold,
                     mmap_embed=bool(A.mmap_embed), kv_ring=A.kv_ring, scratch=A.scratch, wait=A.wait,
                     kernel_tuning=[A.kernel_prefetch, A.kernel_pair, A.kernel_affinity], arena=A.arena,
                     admit_gpu=bool(A.admit_gpu), admit_rule=A.admit_rule, victim=A.victim, near_miss=bool(A.near_miss))


# =============================================================================
# OpenAI-compatible server on the H8 decode path (appended to the stage_h8 prefix
# by make_server_h8.py; every name used below that is not defined here comes from
# that prefix: model, rt, cg, LAY, FC, tok, EMB_ROWS, decode_token, ...).
# =============================================================================
import http.server, re, secrets, signal, threading, uuid
from fused_core import prefill_attention, ring_write
from kv_ring import RingCheckpoints
import queue
import harmony_render as HR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TKN = {n: tok.convert_tokens_to_ids(n) for n in
       ("<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|start|>", "<|call|>", "<|constrain|>")}
STOPS = {TKN["<|return|>"], TKN["<|call|>"]}
VOCAB = model.lm_head.weight.shape[0]
SPECIAL_TEXT = sorted({t for t in tok.all_special_tokens} | {t for t in tok.get_added_vocab()}, key=len, reverse=True)
SPECIAL_TEXT = [t for t in SPECIAL_TEXT if t.startswith("<|") and t.endswith("|>")]


# ---------------------------------------------------------------- sampling tail graph
temp_dev = torch.ones(1, device=dev)
u_dev = torch.rand(VOCAB, device=dev)


def t_body_sample():
    h = mid + (g_out + c_out.to(bf))
    hn = LAY[0]._rms(model.model.norm, h)
    logits = F.linear(hn, model.lm_head.weight).float()
    gum = -torch.log(-torch.log(u_dev.clamp(1e-12, 1.0 - 1e-7)))
    id_dev.copy_((logits / temp_dev + gum).argmax(dim=-1))


def t_full_sample():
    _c_fetch()
    t_body_sample()
    id_pin.copy_(id_dev, non_blocking=True)


with torch.inference_mode():
    GT_GREEDY = GT
    GT_SAMPLE = capture(t_full_sample)


# ---------------------------------------------------------------- fast expert staging for prompt processing
# Neural's prefill streams each non-resident expert to a scratch slot. From the mmap'd
# store that is a pageable H2D (~7 GB/s: the driver stages it single-threaded). Here the
# row is copied into a pinned ring buffer by 8 threads and the H2D is asynchronous, so
# the next expert's copy overlaps the previous transfer. Same bytes, same destinations.
_mt = ctypes.CDLL(os.path.join(S, "memcpy_mt.dll"))
_mt.memcpy_mt.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
RING = [torch.empty(SLOTB, dtype=torch.uint8).pin_memory() for _ in range(A.stage_ring)]    # --stage-ring (was 4)
RING_PTR = [t_.data_ptr() for t_ in RING]         # plain ints: the producer thread never touches a tensor
RING_EV = [None] * len(RING)                      # per buffer: the event of the H2D that last read it (main thread only)
_ring = {"i": 0, "last": 0}


def _fast_get(layer, expert):
    b = _ring["i"]
    _ring["i"] = (b + 1) % len(RING)
    if RING_EV[b] is not None:
        RING_EV[b].synchronize()                  # the H2D that used this buffer must be done
        RING_EV[b] = None
    _mt.memcpy_mt(RING[b].data_ptr(), ROWP[layer * NE + expert], SLOTB, A.threads)
    _ring["last"] = b
    return RING[b]


def _fast_note(key, ev):
    RING_EV[_ring["last"]] = ev


if A.fast_prefill:
    path["host"].get = _fast_get
    path["host"].note_h2d = _fast_note


# ---- staging copies off the launching thread (--stage-thread). moe_prefill / moe_prefill_layer create one _Stager per
# layer over the experts they will stage, in staging order; the loop then calls stager.stage(slot, main) per expert.
_STAGE_SWITCH_S = 0.0005        # GIL switch interval while the producer runs (default 5 ms; the producer needs the GIL
                                # for a few bytecodes after each blocking call, and the launching thread is pure Python)


class _Stager:
    """The mmap -> pinned-ring copies of one layer's staged experts, done by a producer thread AHEAD of the thread that
    issues the H2Ds. keys = global expert ids (layer * NE + expert) in staging order. Expert i uses ring buffer
    (r0 + i) % n: the same round robin as the old inline loop, so every buffer's chain of users is unchanged.
    Producer (STAGE_WORKER), per expert i:  (rb, ev) = free_q.get()   # buffer + the H2D event that last read it
                                            ev.synchronize()           # previous H2D of that buffer has landed
                                            memcpy_mt(RING_PTR[rb], ROWP[keys[i]], ...)
                                            ready_q.put(rb)
      It touches no tensor and makes no CUDA call but Event.synchronize (thread-safe, GIL released; the event was
      recorded on the launching thread BEFORE it was put on free_q, so it is a plain wait on a recorded event).
    Consumer (the launching thread), stage(s_, main), per expert i:
                                            rb = ready_q.get()         # blocks only if the producer is behind
                                            COPY_STREAM.wait_event(SLOT_FREE[s_]); H2D RING[rb] -> scratch slot s_
                                            ev.record(COPY_STREAM); RING_EV[rb] = ev
                                            free_q.put((rb, ev))       # buffer's next user is expert i + n
                                            main.wait_event(ev)
      then the caller runs the GEMMs and records SLOT_FREE[s_], as before.
    close() (always, in the caller's finally) stops and JOINS the producer: no copy is in flight afterwards. A producer
    error is posted to the consumer, which re-raises it from stage(); RING_EV always holds the last H2D issued from
    each buffer, so an abort leaves nothing the next user does not already wait for. Without the thread
    (--stage-thread 0, or a single expert) stage() runs the old inline sequence on the calling thread."""

    def __init__(self, keys):
        self.keys, self.K, self.n = keys, len(keys), len(RING)
        self.r0 = _ring["i"]
        _ring["i"] = (self.r0 + self.K) % self.n         # as the old loop: one step per staged expert
        self.i = 0
        self.fut = None
        self.threaded = bool(A.stage_thread) and self.K >= 2
        if self.threaded:
            self.free_q, self.ready_q = queue.SimpleQueue(), queue.SimpleQueue()
            self.stop, self.err = threading.Event(), None
            for j in range(min(self.n, self.K)):         # the first n experts wait on whatever last read their buffer
                b = (self.r0 + j) % self.n
                self.free_q.put((b, RING_EV[b]))
            self._sw = sys.getswitchinterval()
            sys.setswitchinterval(min(self._sw, _STAGE_SWITCH_S))
            self.fut = STAGE_WORKER.submit(self._produce)

    def _produce(self):
        try:
            for i in range(self.K):
                tok = self.free_q.get()
                if tok is None or self.stop.is_set():
                    return
                b, ev = tok
                if ev is not None:
                    t_w = _pc()
                    ev.synchronize()                     # the H2D that last read this buffer has landed
                    PTIME["stage_thread_ring_wait_s"] += _pc() - t_w
                    if self.stop.is_set():
                        return
                t_m = _pc()
                _mt.memcpy_mt(RING_PTR[b], ROWP[self.keys[i]], SLOTB, A.threads)
                PTIME["stage_thread_s"] += _pc() - t_m
                self.ready_q.put(b)
        except BaseException as e_:                      # -> the consumer, which re-raises it
            self.err = e_
            self.ready_q.put(-1)

    def take(self):
        """The ring index that holds the next expert's bytes (threaded: waits for the producer)."""
        i = self.i
        assert i < self.K
        if self.threaded:
            t_q = _pc()
            rb = self.ready_q.get()
            PTIME["stage_wait_s"] += _pc() - t_q         # the consumer's blocked time: the critical-path cost
            if rb < 0:
                raise self.err
        else:                                            # the old inline sequence
            rb = (self.r0 + i) % self.n
            if RING_EV[rb] is not None:
                t_w = _pc()
                RING_EV[rb].synchronize()                # the H2D that used this buffer must be done
                PTIME["ring_wait_s"] += _pc() - t_w
                RING_EV[rb] = None
            t_m = _pc()
            _mt.memcpy_mt(RING_PTR[rb], ROWP[self.keys[i]], SLOTB, A.threads)
            PTIME["memcpy_s"] += _pc() - t_m
        self.i = i + 1
        PTIME["n_stage"] += 1
        return rb

    def stage(self, s_, main):
        """The next expert's bytes into scratch slot s_ on COPY_STREAM; main waits for the copy."""
        rb = self.take()
        t_h = _pc()
        if SLOT_FREE[s_] is not None:
            COPY_STREAM.wait_event(SLOT_FREE[s_])
        with torch.cuda.stream(COPY_STREAM):
            rt._rows[s_].copy_(RING[rb], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(COPY_STREAM)
        RING_EV[rb] = ev
        if self.threaded and self.i - 1 + self.n < self.K:   # expert (i - 1) + n uses this buffer next
            self.free_q.put((rb, ev))
        main.wait_event(ev)
        PTIME["stage_h2d_launch_s"] += _pc() - t_h

    def close(self):
        """Stop and join the producer (idempotent). Nothing is copying into the ring once this returns."""
        if self.fut is None:
            return
        self.stop.set()
        self.free_q.put(None)                            # wakes a producer parked on a free buffer
        self.fut.result()                                # _produce never raises: it posts its error instead
        self.fut = None
        sys.setswitchinterval(self._sw)


# ---------------------------------------------------------------- sync-free MoE for prompt processing
from neural.q80.mxfp4_kernels import mxfp4_gemm
SLOT_IDS = torch.arange(rt.gate_blocks.shape[0], dtype=torch.long, device=dev)   # every physical slot incl. scratch
COPY_STREAM = torch.cuda.Stream()
SCR_SLOTS = list(rt.scratch_slots)
SLOT_FREE = {sl: None for sl in SCR_SLOTS}      # event: last compute that read this scratch slot


import cpu_prefill as _CP
import concurrent.futures as _CF
CPUX = _CP.CpuMultiExperts(threads=A.threads, dll=os.path.join(S, A.cpu_multi_dll), scale_layout=SCALE_MODE)
assert CPUX.slot_bytes() == SLOTB, (CPUX.slot_bytes(), SLOTB)
CPU_WORKER = _CF.ThreadPoolExecutor(max_workers=1)   # one persistent thread (and one OpenMP team)
STAGE_WORKER = _CF.ThreadPoolExecutor(max_workers=1)  # the staging producer (_Stager): one memcpy team at a time
_PIN = {}


def _pinned(name, n, dtype=torch.float32):
    """Growing pinned host buffers reused across calls (pinning is slow)."""
    t = _PIN.get(name)
    if t is None or t.numel() < n:
        t = torch.empty(max(n, int((t.numel() if t is not None else 0) * 1.5), 1 << 16), dtype=dtype).pin_memory()
        _PIN[name] = t
    return t[:n]


# ---- prompt-processing time split (MEASUREMENT ONLY: perf_counter around existing calls, no new sync) ----
# Seconds of HOST wall time, accumulated per prefill() call (reset at its start, copied into the request stats
# by the caller). Timers around ASYNC CUDA launches (attn_router, gemm_launch, h2d_launch) measure the host cost
# of issuing the work (Python + Triton/driver launch overhead), NOT GPU time. The GPU shows up only where the
# host already blocks: counts_sync (the .cpu() after bincount waits for every kernel queued so far: that block's
# attention/router, and at layer start the previous layer's MoE tail), cpu_in_sync, ring_wait (the pinned ring
# buffer's previous H2D), and tail_wait (final drain at the first sampled token). So GPU-bound time ~ the blocked
# waits, wall - host_busy. moe_host_s (the whole moe_prefill / moe_prefill_layer call) CONTAINS memcpy, ring_wait,
# cpu_job_wait, counts_sync, cpu_in_sync, h2d_launch and gemm_launch; the rest of it is prefill_moe_other_s.
# cpu_job_run_s is the CPU multi-kernel's busy time measured on its worker thread (not host time).
# Staging (--stage-thread 1): the expert copies run on the producer thread. stage_thread_s = its memcpy time,
# stage_thread_ring_wait_s = its wait for a buffer's previous H2D (both worker-thread time, NOT host time of the
# launching thread), stage_wait_s = the launching thread's wait for the next copied expert (the critical-path cost;
# part of moe_host_s). memcpy_s / ring_wait_s stay the launching thread's own copy / ring wait (--stage-thread 0).
# Grouped epilogue (layer order, NEURAL_PREFILL_GROUP): gemm_launch_s is the host time of the whole group work (both GEMM
# launches per pair AND the once-per-group bias / SwiGLU / scale ops / write-back), excluding the stager calls;
# n_pairs = (expert, block) GEMM pairs and n_epi_groups = epilogue groups of the request (0 on the per-pair path), so
# gemm_launch_s / n_pairs is the host cost per pair in either mode.
_pc = time.perf_counter
PTIME = {"memcpy_s": 0.0, "ring_wait_s": 0.0, "cpu_job_wait_s": 0.0, "cpu_job_run_s": 0.0, "counts_sync_s": 0.0,
         "cpu_in_sync_s": 0.0, "attn_router_s": 0.0, "moe_host_s": 0.0, "stage_h2d_launch_s": 0.0,
         "gemm_launch_s": 0.0, "stage_wait_s": 0.0, "stage_thread_s": 0.0, "stage_thread_ring_wait_s": 0.0,
         "n_stage": 0, "n_cpu_jobs": 0, "n_pairs": 0, "n_epi_groups": 0}


def ptime_reset():
    for k_ in PTIME:
        PTIME[k_] = 0 if k_.startswith("n_") else 0.0


def ptime_stats(pt, tail_wait_s, wall_s):
    """Request-stats keys (seconds, 3 decimals) from a PTIME snapshot, the tail wait and the request's prefill wall time."""
    moe_other = pt["moe_host_s"] - (pt["memcpy_s"] + pt["ring_wait_s"] + pt["stage_wait_s"] + pt["cpu_job_wait_s"]
                                    + pt["counts_sync_s"] + pt["cpu_in_sync_s"] + pt["stage_h2d_launch_s"]
                                    + pt["gemm_launch_s"])
    blocked = (pt["ring_wait_s"] + pt["stage_wait_s"] + pt["cpu_job_wait_s"] + pt["counts_sync_s"]
               + pt["cpu_in_sync_s"] + tail_wait_s)
    unattr = wall_s - (pt["attn_router_s"] + pt["moe_host_s"] + tail_wait_s)
    r3 = lambda v: round(v, 3)
    return {"prefill_memcpy_s": r3(pt["memcpy_s"]), "prefill_ring_wait_s": r3(pt["ring_wait_s"]),
            "prefill_cpu_job_wait_s": r3(pt["cpu_job_wait_s"]), "prefill_cpu_job_run_s": r3(pt["cpu_job_run_s"]),
            "prefill_counts_sync_s": r3(pt["counts_sync_s"]), "prefill_cpu_in_sync_s": r3(pt["cpu_in_sync_s"]),
            "prefill_attn_router_s": r3(pt["attn_router_s"]), "prefill_moe_host_s": r3(pt["moe_host_s"]),
            "prefill_h2d_launch_s": r3(pt["stage_h2d_launch_s"]), "prefill_gemm_launch_s": r3(pt["gemm_launch_s"]),
            "prefill_moe_other_s": r3(moe_other), "prefill_tail_wait_s": r3(tail_wait_s),
            "prefill_blocked_s": r3(blocked), "prefill_unattributed_s": r3(unattr),
            "prefill_stage_wait_s": r3(pt["stage_wait_s"]), "prefill_stage_thread_s": r3(pt["stage_thread_s"]),
            "prefill_stage_thread_ring_wait_s": r3(pt["stage_thread_ring_wait_s"]),
            "prefill_cpu_jobs": pt["n_cpu_jobs"], "prefill_gemm_pairs": pt["n_pairs"],
            "prefill_epi_groups": pt["n_epi_groups"]}


def ptime_note(r):
    """Console suffix for the per-request line ('' when the request carried no split)."""
    if "prefill_memcpy_s" not in r or not r.get("prefill_new_tokens"):
        return ""
    return (f" | prefill: memcpy {r['prefill_memcpy_s']:.1f} s, ring-wait {r['prefill_ring_wait_s']:.1f} s, "
            f"stage-wait {r['prefill_stage_wait_s']:.1f} s (thread: copy {r['prefill_stage_thread_s']:.1f} s, "
            f"ring-wait {r['prefill_stage_thread_ring_wait_s']:.1f} s), "
            f"cpu-wait {r['prefill_cpu_job_wait_s']:.1f} s, count-sync {r['prefill_counts_sync_s']:.1f} s, "
            f"attn {r['prefill_attn_router_s']:.1f} s, moe-host {r['prefill_moe_host_s']:.1f} s, "
            f"gemm-launch {r['prefill_gemm_launch_s']:.1f} s, tail {r['prefill_tail_wait_s']:.1f} s")


def moe_prefill(L, hn, top_i, sc):
    """Experts for T prompt tokens of layer L. Tokens are grouped by expert on the GPU
    (one host sync per layer, for the counts). Three routes per expert:
      resident in VRAM                     -> GPU from its slot (first, overlapping the CPU)
      not resident, <= cpu_prefill_max tok -> CPU multi-token kernel (reads the expert ONCE
                                              from RAM for all its tokens), background thread
      not resident, more tokens            -> mmap -> pinned ring (8 threads) -> scratch slot
                                              on a copy stream, GEMM on the GPU
    Output = per-token sum of its 4 weighted experts (fp32 accumulation in top-k order,
    one bf16 rounding)."""
    t_moe0 = _pc()
    T = hn.shape[0]
    flat = top_i.reshape(-1)
    order = torch.argsort(flat, stable=True)
    t_c = _pc()
    counts = torch.bincount(flat, minlength=NE).cpu().numpy()       # host sync (existing): waits for the queued GPU work
    PTIME["counts_sync_s"] += _pc() - t_c
    if COUNT_PREFILL[0]:
        PREF[L * NE:(L + 1) * NE] += counts
    if A.count_routing:
        PCOUNTS[L] += counts
    xs = hn.index_select(0, torch.div(order, K, rounding_mode="floor"))
    ws = sc.reshape(-1).index_select(0, order)
    ys = torch.empty_like(xs)
    main = torch.cuda.current_stream()
    offs = np.zeros(NE + 1, np.int64)
    np.cumsum(counts, out=offs[1:])
    used = [int(e) for e in np.nonzero(counts)[0]]
    res = [e for e in used if (L, e) in rt.slot_of]
    cpu = [e for e in used if (L, e) not in rt.slot_of and counts[e] <= A.cpu_prefill_max]
    stg = [e for e in used if (L, e) not in rt.slot_of and counts[e] > A.cpu_prefill_max]
    job = None
    if cpu:
        rows = torch.from_numpy(np.concatenate([np.arange(offs[e], offs[e + 1]) for e in cpu])).to(dev)
        Pc = int(rows.numel())
        xh = _pinned("x", Pc * H).view(Pc, H)
        wh = _pinned("w", Pc)
        yh = _pinned("y", Pc * H).view(Pc, H)
        xh.copy_(xs.index_select(0, rows).float(), non_blocking=True)
        wh.copy_(ws.index_select(0, rows).float(), non_blocking=True)
        t_s = _pc()
        torch.cuda.current_stream().synchronize()                   # (existing) pinned inputs complete
        PTIME["cpu_in_sync_s"] += _pc() - t_s
        coff = np.zeros(len(cpu) + 1, np.int64)
        np.cumsum([counts[e] for e in cpu], out=coff[1:])
        gids = [L * NE + e for e in cpu]
        xn, wn, yn = xh.numpy(), wh.numpy(), yh.numpy()

        def _run():
            t_r = _pc()
            CPUX.experts([ROWP[g] for g in gids], coff, xn, wn, [BGP[g] for g in gids], [BDP[g] for g in gids], out=yn)
            PTIME["cpu_job_run_s"] += _pc() - t_r                   # worker thread; main reads it only after the join
        job = CPU_WORKER.submit(_run)
        PTIME["n_cpu_jobs"] += 1
    si = 0
    stager = None
    try:
      if stg:                                       # the staged experts, in the order the loop reaches them (residents first)
          stager = _Stager([L * NE + e for e in stg])
      for e in res + stg:
          c = int(counts[e])
          off = int(offs[e])
          s_ = rt.slot_of.get((L, e))
          scratch = s_ is None
          if scratch:
              s_ = SCR_SLOTS[si]
              si = (si + 1) % len(SCR_SLOTS)
              stager.stage(s_, main)
          t_g = _pc()
          slt = SLOT_IDS[s_:s_ + 1]
          x = xs[off:off + c]
          _g = mxfp4_gemm(x, rt.gate_blocks, rt.gate_scales, slt)
          gu = _g.view(c, -1) + rt.bias_gu[L * NE + e]
          y = mxfp4_gemm(rt._act(gu), rt.down_blocks, rt.down_scales, slt).view(c, -1) + rt.bias_dn[L * NE + e]
          ys[off:off + c] = y * ws[off:off + c, None]
          PTIME["gemm_launch_s"] += _pc() - t_g
          if scratch:
              fe = torch.cuda.Event()
              fe.record(main)
              SLOT_FREE[s_] = fe
    finally:
        try:
            if stager is not None:
                stager.close()                      # joins the staging producer: it never outlives this call
        finally:
            if job is not None:
                t_j = _pc()
                job_err = job.exception()           # waits: the CPU job never outlives this call
                PTIME["cpu_job_wait_s"] += _pc() - t_j
    if job is not None:
        if job_err is not None:
            raise job_err
        ys.index_copy_(0, rows, yh.to(dev, non_blocking=True).to(ys.dtype))
    PSTAT["res"] += len(res); PSTAT["cpu"] += len(cpu); PSTAT["stg"] += len(stg); PSTAT["stg_pairs"] += len(stg)
    back = torch.empty_like(order)
    back[order] = torch.arange(order.numel(), device=dev)
    out = ys.index_select(0, back).view(T, K, -1).float().sum(1).to(hn.dtype)
    PTIME["moe_host_s"] += _pc() - t_moe0
    return out


PSTAT = {"res": 0, "cpu": 0, "stg": 0, "stg_pairs": 0}
# res/cpu/stg_pairs count (layer, block, expert) triples per route; stg counts the staging COPIES
# (mmap -> ring -> scratch slot). Block order: stg == stg_pairs. Layer order: stg <= stg_pairs (a copy serves every block).


class _PBlk:
    """One prompt block's routing at one layer, alive for the duration of moe_prefill_layer."""
    __slots__ = ("i", "T", "dt", "order", "counts", "offs", "xs", "ws", "ys", "res", "cpu", "stg", "stg_set",
                 "rows", "xh", "wh", "yh", "rbi")


import neural.q80.mxfp4_kernels as _MXK      # FAST_LAUNCH / PREFILL_BLOCK_M are read from the module at every call
_MXK.PREFILL_GROUPED_GEMM = getattr(_MXK, "PREFILL_GROUPED_GEMM", False)
if A.grouped_prefill_gemm is not None:
    _MXK.PREFILL_GROUPED_GEMM = bool(A.grouped_prefill_gemm)
from neural.q80.expert_runtime import _make_clamped_swiglu as _MCS
_EG = {}                                        # _expert_gemms' tables (see _eg_build)


def _eg_build():
    """Tables for _expert_gemms, built once (rebuilt if rt's tensors are replaced): a length-1 view of SLOT_IDS per
    physical slot and one row view per (layer, expert) bias, i.e. the tensors SLOT_IDS[s:s + 1] and rt.bias_*[i] would
    create on every call (same storage, same pointers); the two output widths; and (alpha, limit) when rt._act is
    expert_runtime's clamped SwiGLU closure (recognised by its code object), else None -> rt._act is called as before.
    The fast branch needs bf16 biases (attach_expert_biases guarantees it); anything else takes the stock branch."""
    f = rt._act
    act = None
    if getattr(f, "__code__", None) is _MCS(1.0, 1.0).__code__ and f.__closure__:
        cv = dict(zip(f.__code__.co_freevars, (c_.cell_contents for c_ in f.__closure__)))
        if set(cv) == {"alpha", "limit"}:
            act = (cv["alpha"], cv["limit"])
    ok = rt.bias_gu.dtype is bf and rt.bias_dn.dtype is bf
    st = (rt.gate_blocks, rt.down_blocks, rt.bias_gu, rt.bias_dn, f, ok,
          tuple(SLOT_IDS.view(-1, 1).unbind(0)), rt.bias_gu.unbind(0), rt.bias_dn.unbind(0),
          rt.gate_blocks.shape[1], rt.down_blocks.shape[1], act)
    _EG[0] = st
    return st


def _expert_gemms(L, e, s_, xs, ws, ys, c, off):
    """The two GEMMs of moe_prefill for the c rows at [off, off + c) of one expert, from physical slot s_.
    Fast branch (mxfp4_kernels.FAST_LAUNCH; NEURAL_FAST_LAUNCH=0 restores the stock branch): the same kernels and the
    same torch elementwise ops in the same order and dtypes, with less host work per call. Bit-exact because
    (1) the GEMMs run the same compiled kernel with the same arguments (mxfp4_kernels._launch_gemm) into buffers
    allocated here; (2) the bf16 bias adds and the clamped-SwiGLU ops run in place: an in-place elementwise op is the
    same kernel with the output aliasing an input, every element is rounded exactly like the out-of-place result, and
    none of those buffers is read again; (3) the last SwiGLU multiply writes into the dead sigmoid buffer, a fresh
    contiguous [c, K2] tensor, which is what the down GEMM needs; (4) y * ws goes straight into the ys slice (out=)
    instead of through a temporary and a copy_: bf16 * bf16 is rounded once either way (other dtypes: the stock
    expression). Everything runs on the current stream, so a buffer freed here is only reused by work enqueued after
    the kernels that read it. BLOCK_M of both GEMMs is mxfp4_kernels.PREFILL_BLOCK_M (NEURAL_PREFILL_BM, default 16)."""
    t_g = _pc()
    bm = _MXK.PREFILL_BLOCK_M
    st = None
    if _MXK.FAST_LAUNCH:
        st = _EG.get(0)
        if (st is None or st[0] is not rt.gate_blocks or st[1] is not rt.down_blocks or st[2] is not rt.bias_gu
                or st[3] is not rt.bias_dn or st[4] is not rt._act):
            st = _eg_build()
    if st is None or not st[5]:                         # stock branch
        slt = SLOT_IDS[s_:s_ + 1]
        x = xs[off:off + c]
        _g = mxfp4_gemm(x, rt.gate_blocks, rt.gate_scales, slt, block_m=bm)
        gu = _g.view(c, -1) + rt.bias_gu[L * NE + e]
        y = mxfp4_gemm(rt._act(gu), rt.down_blocks, rt.down_scales, slt, block_m=bm).view(c, -1) + rt.bias_dn[L * NE + e]
        ys[off:off + c] = y * ws[off:off + c, None]
    else:
        i = L * NE + e
        slt = st[6][s_]
        gu = torch.empty((c, st[9]), dtype=bf, device=dev)
        _MXK.mxfp4_gemm_into(xs.narrow(0, off, c), rt.gate_blocks, rt.gate_scales, slt, gu, block_m=bm)
        gu.add_(st[7][i])
        if st[11] is None:
            h = rt._act(gu)
        else:
            alpha, limit = st[11]                       # _make_clamped_swiglu's act(), op for op, in place
            gate, up = gu.chunk(2, dim=-1)
            gate.clamp_(max=limit)
            up.clamp_(min=-limit, max=limit)
            h = gate * alpha
            h.sigmoid_()
            gate.mul_(h)                                # glu = gate * sigmoid(gate * alpha)
            up.add_(1)
            torch.mul(up, gate, out=h)                  # (up + 1) * glu
            if not h.is_contiguous():
                h = h.contiguous()
        y = torch.empty((c, st[10]), dtype=bf, device=dev)
        _MXK.mxfp4_gemm_into(h, rt.down_blocks, rt.down_scales, slt, y, block_m=bm)
        y.add_(st[8][i])
        w = ws.narrow(0, off, c).unsqueeze(1)
        yo = ys.narrow(0, off, c)
        if ws.dtype is bf and ys.dtype is bf:
            torch.mul(y, w, out=yo)
        else:
            yo.copy_(y * w)
    PTIME["gemm_launch_s"] += _pc() - t_g                  # host launch time only (async kernels)


# ---------------------------------------------------------------- grouped epilogue of the layer-order prefill
# moe_prefill_layer's per-(expert, block) call spent ~0.35 ms of HOST time, ~10 pointwise torch launches of which sit
# between and after the two GEMM launches. The GEMMs stay one launch per (expert, block) pair; the pointwise work (bias
# add, clamped SwiGLU, down bias add, routing-weight scale) now runs once per GROUP of pairs on one concatenated buffer.
# NEURAL_PREFILL_GROUP=0 restores _expert_gemms per pair (A/B); NEURAL_PREFILL_GROUP_ROWS caps a group's rows (default
# 16384: transient buffers ~ rows * 28.8 KB); NEURAL_PREFILL_GROUP_EXPERTS caps the staged experts per group (default
# 0 = one per scratch slot). moe_prefill (block order) is unchanged and does not use any of this.
def _pf_tables():
    """_eg_build's tables when the grouped path applies, else None (-> _expert_gemms per pair, as before): fast launch
    on (NEURAL_FAST_LAUNCH), bf16 biases (st[5]), rt._act recognised as expert_runtime's clamped SwiGLU (st[11]: the
    op chain below is that function's; any other activation is not known to be row-wise, so it keeps the per-pair
    path), NEURAL_PREFILL_GROUP != 0."""
    if not (_MXK.FAST_LAUNCH and (_MXK.PREFILL_GROUP or _MXK.PREFILL_GROUPED_GEMM)):
        return None
    st = _EG.get(0)
    if (st is None or st[0] is not rt.gate_blocks or st[1] is not rt.down_blocks or st[2] is not rt.bias_gu
            or st[3] is not rt.bias_dn or st[4] is not rt._act):
        st = _eg_build()
    return st if (st[5] and st[11] is not None) else None


def _pf_split(pairs, cap):
    """pairs [(pb, e, off, c)] in issue order -> consecutive chunks of at most `cap` rows. A pair is never split (its
    GEMM is one launch on its own rows), so a chunk holds at least one pair even when that pair alone exceeds cap."""
    out, cur, rows = [], [], 0
    for p in pairs:
        if cur and rows + p[3] > cap:
            out.append(cur)
            cur, rows = [], 0
        cur.append(p)
        rows += p[3]
    if cur:
        out.append(cur)
    return out


def _pf_runs(chunk):
    """chunk [(pb, e, off, c)] -> ([(pb, off, n, r)], P). A run is a maximal stretch of consecutive pairs of ONE block
    whose sorted source rows are adjacent (the next expert's rows start where the previous one's end); r is the run's
    first row in the chunk's buffer, whose pairs are laid out back to back in chunk order (P rows in all). Runs only
    shorten the index / weight gathers and the write-back; the GEMMs stay per pair."""
    runs, r = [], 0
    for pb, e, off, c in chunk:
        if runs and runs[-1][0] is pb and runs[-1][1] + runs[-1][2] == off:
            b_, o_, n_, r_ = runs[-1]
            runs[-1] = (b_, o_, n_ + c, r_)
        else:
            runs.append((pb, off, c, r))
        r += c
    return runs, r


def _pf_cat(parts):
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def _pf_run_chunk(L, chunk, tb, slot_of, stage_cb=None, done_cb=None, left=None):
    """One epilogue group = chunk [(pb, e, off, c)], its pairs laid out back to back in ONE gate_up buffer [P, N1].
      1. GEMM1 per pair (same kernel, x rows, slot, BLOCK_M as _expert_gemms) into that pair's rows of the buffer;
      2. ONE bias add and ONE clamped-SwiGLU chain over all P rows; 3. GEMM2 per pair from the pair's rows of the
      activation into its rows of y [P, N2]; 4. ONE bias add and ONE weight scale, then each run of rows is copied
      to its place in the block's ys (a plain bf16 copy).
    Exactness: bias add, clamp, sigmoid, mul are POINTWISE, so element (row, col) sees the same inputs, the same
    kernel functor and the same rounding whether its expert's rows are alone or among other experts' rows; only the
    launch count changes. The bias operand is index_select of the same bf16 bias rows by the row's own (layer, expert)
    index (pb.rbi), so every element adds the same bf16 value as before. The op sequence, dtypes and in-place-ness are
    _expert_gemms' (see its docstring). A pair's GEMM still sees exactly its own c rows (same M, same tiling, same
    pointer alignment: a row is 11520 / 5760 bytes, both multiples of 16), so its output is bit-identical.
    Slots: slot_of {e: scratch or resident slot}. An expert missing from it is staged first (stage_cb(e) copies it into
    a scratch slot and fills slot_of); after the LAST GEMM2 of an expert (left[e] counts its remaining pairs in the
    whole group of experts) done_cb(e) records SLOT_FREE, i.e. after the last kernel that reads that slot, as before.
    gemm_launch_s covers this function's host time except stage_cb (which times itself)."""
    alpha, limit = tb[11]
    bm = _MXK.PREFILL_BLOCK_M
    t_g = _pc()
    runs, P = _pf_runs(chunk)
    gu = torch.empty((P, tb[9]), dtype=bf, device=dev)
    grouped = _MXK.PREFILL_GROUPED_GEMM
    gate_pairs = []
    r = 0
    for pb, e, off, c in chunk:
        s_ = slot_of.get(e)
        if s_ is None:                                  # first pair of a staged expert: its H2D, main waits for it
            PTIME["gemm_launch_s"] += _pc() - t_g
            s_ = stage_cb(e)
            t_g = _pc()
        x_part, gu_part = pb.xs.narrow(0, off, c), gu.narrow(0, r, c)
        if grouped:
            gate_pairs.append((x_part, s_, gu_part))
        else:
            _MXK.mxfp4_gemm_into(x_part, rt.gate_blocks, rt.gate_scales, tb[6][s_], gu_part, block_m=bm)
        r += c
    gate_plan = (_MXK.mxfp4_gemm_grouped_into(gate_pairs, rt.gate_blocks, rt.gate_scales, block_m=bm)
                 if grouped else None)
    # Work and descriptor upload were enqueued on this stream; allocator reuse follows their last read.
    # Do not keep slice views alive through the down GEMM: each view owns the entire gate-up allocation.
    del gate_plan, gate_pairs, x_part, gu_part
    idx = _pf_cat([pb.rbi.narrow(0, o, n) for pb, o, n, _ in runs])     # per row: L * NE + expert
    gu.add_(tb[2].index_select(0, idx))
    gate, up = gu.chunk(2, dim=-1)                      # _make_clamped_swiglu's act(), op for op, in place
    gate.clamp_(max=limit)
    up.clamp_(min=-limit, max=limit)
    h = gate * alpha
    h.sigmoid_()
    gate.mul_(h)                                        # glu = gate * sigmoid(gate * alpha)
    up.add_(1)
    torch.mul(up, gate, out=h)                          # (up + 1) * glu
    if not h.is_contiguous():
        h = h.contiguous()
    del gate, up, gu
    y = torch.empty((P, tb[10]), dtype=bf, device=dev)
    down_pairs = []
    r = 0
    for pb, e, off, c in chunk:
        h_part, y_part = h.narrow(0, r, c), y.narrow(0, r, c)
        if grouped:
            down_pairs.append((h_part, slot_of[e], y_part))
        else:
            _MXK.mxfp4_gemm_into(h_part, rt.down_blocks, rt.down_scales, tb[6][slot_of[e]], y_part, block_m=bm)
        r += c
        if done_cb is not None and not grouped:
            left[e] -= 1
            if left[e] == 0:
                done_cb(e)
    down_plan = (_MXK.mxfp4_gemm_grouped_into(down_pairs, rt.down_blocks, rt.down_scales, block_m=bm)
                 if grouped else None)
    del down_plan, down_pairs, h_part, y_part
    if done_cb is not None and grouped:
        # A scratch slot is reusable only AFTER the grouped kernel containing its final read was queued.
        for pb, e, off, c in chunk:
            left[e] -= 1
            if left[e] == 0:
                done_cb(e)
    del h
    y.add_(tb[3].index_select(0, idx))
    w = _pf_cat([pb.ws.narrow(0, o, n) for pb, o, n, _ in runs]).unsqueeze(1)
    pb0 = chunk[0][0]
    if pb0.ws.dtype is bf and pb0.ys.dtype is bf:
        y.mul_(w)                                       # bf16 * bf16 rounded once, as torch.mul(y, w, out=ys slice)
        res = y
    else:
        res = y * w                                     # other dtypes: the stock expression, cast by copy_ below
    for pb, o, n, r0 in runs:
        pb.ys.narrow(0, o, n).copy_(res.narrow(0, r0, n))
    PTIME["gemm_launch_s"] += _pc() - t_g
    PTIME["n_epi_groups"] += 1


def _pf_layer_grouped(L, blks, stg_all, stager, main, tb):
    """The GEMM part of moe_prefill_layer with grouped epilogues. Pair order is the per-pair path's: resident pairs
    block-major, then staged experts in stg_all order, each with every block that has it. Resident pairs are chunked
    by rows only (their slots are always valid). Staged experts go in GROUPS of up to len(SCR_SLOTS) (rows-capped
    chunks inside a group): the group's experts are staged one by one, each followed by its GEMM1s (stager.stage
    makes main wait for that expert's H2D only), then the epilogue, then the GEMM2s; SLOT_FREE of an expert is
    recorded after its last GEMM2. A group is fully issued (all events recorded) before the next group stages into
    the slots it frees, and slots are handed out round robin exactly as before, so within a group they are distinct."""
    cap = _MXK.PREFILL_GROUP_ROWS
    nscr = len(SCR_SLOTS)
    gsz = max(1, min(_MXK.PREFILL_GROUP_EXPERTS or nscr, nscr))
    res_pairs = [(pb, e, int(pb.offs[e]), int(pb.counts[e])) for pb in blks for e in pb.res]
    for chunk in _pf_split(res_pairs, cap):             # resident: no staging, overlaps the CPU jobs
        _pf_run_chunk(L, chunk, tb, {e: rt.slot_of[(L, e)] for _, e, _, _ in chunk})
    si = [0]
    for g0 in range(0, len(stg_all), gsz):
        grp = stg_all[g0:g0 + gsz]
        pairs = [(pb, e, int(pb.offs[e]), int(pb.counts[e])) for e in grp for pb in blks if e in pb.stg_set]
        left, slot_of = {}, {}
        for p in pairs:
            left[p[1]] = left.get(p[1], 0) + 1

        def stage_cb(e, slot_of=slot_of):
            s_ = SCR_SLOTS[si[0]]
            si[0] = (si[0] + 1) % nscr
            stager.stage(s_, main)
            slot_of[e] = s_
            return s_

        def done_cb(e, slot_of=slot_of):
            fe = torch.cuda.Event()
            fe.record(main)                             # after this expert's LAST GEMM2 (the last reader of its slot)
            SLOT_FREE[slot_of[e]] = fe

        for chunk in _pf_split(pairs, cap):
            _pf_run_chunk(L, chunk, tb, slot_of, stage_cb, done_cb, left)


def moe_prefill_layer(L, hns, top_is, scs, allow_grouped=True):
    """moe_prefill for ALL blocks of a prompt at layer L (--prefill-order layer). Per block nothing changes:
    same sort, same res/cpu/stg split (an expert with <= --cpu-prefill-max tokens IN THAT BLOCK goes to the CPU
    kernel for that block; counts are never merged across blocks), same GEMM calls on that block's rows, same
    combine. Only the staging differs: an expert that any block sends down the stg route is copied ONCE
    (mmap -> pinned ring -> scratch slot) and every block that routes to it runs its GEMMs on that slot before
    the slot is released (SLOT_FREE is recorded after the LAST of them).
    Sequence: per-block CPU jobs are submitted first (one worker runs them in block order, each block with its own
    pinned buffers so queued jobs never share one), then the resident experts of every block, then the staged
    experts expert-major, then the CPU jobs are joined and their rows scattered. hns/top_is/scs are emptied as
    they are consumed. Returns the per-block MoE outputs [T_b, H].
    GEMM launches: with the grouped epilogue (_pf_tables() not None, the default) the same (expert, block) GEMMs are
    issued, but bias / SwiGLU / weight scale run once per group of pairs on a concatenated buffer (_pf_layer_grouped);
    otherwise _expert_gemms runs per pair exactly as before."""
    t_moe0 = _pc()
    main = torch.cuda.current_stream()
    tb = _pf_tables() if allow_grouped else None          # None -> the per-pair _expert_gemms path
    blks = []
    for b in range(len(hns)):
        hn, top_i, sc = hns[b], top_is[b], scs[b]
        hns[b] = top_is[b] = scs[b] = None              # the routing tensors die with this iteration
        pb = _PBlk()
        pb.i, pb.T, pb.dt = b, hn.shape[0], hn.dtype
        flat = top_i.reshape(-1)
        pb.order = torch.argsort(flat, stable=True)
        t_c = _pc()
        pb.counts = counts = torch.bincount(flat, minlength=NE).cpu().numpy()   # host sync (existing)
        PTIME["counts_sync_s"] += _pc() - t_c
        if COUNT_PREFILL[0]:
            PREF[L * NE:(L + 1) * NE] += counts
        if A.count_routing:
            PCOUNTS[L] += counts
        pb.xs = hn.index_select(0, torch.div(pb.order, K, rounding_mode="floor"))
        pb.ws = sc.reshape(-1).index_select(0, pb.order)
        pb.ys = torch.empty_like(pb.xs)
        pb.offs = np.zeros(NE + 1, np.int64)
        np.cumsum(counts, out=pb.offs[1:])
        used = [int(e) for e in np.nonzero(counts)[0]]
        pb.res = [e for e in used if (L, e) in rt.slot_of]
        pb.cpu = [e for e in used if (L, e) not in rt.slot_of and counts[e] <= A.cpu_prefill_max]
        pb.stg = [e for e in used if (L, e) not in rt.slot_of and counts[e] > A.cpu_prefill_max]
        pb.stg_set = set(pb.stg)
        pb.rows = pb.xh = pb.wh = pb.yh = None
        pb.rbi = None
        if tb is not None:                              # sorted row -> its bias row L * NE + expert (top_i sorted by order)
            pb.rbi = flat.index_select(0, pb.order)
            pb.rbi.add_(L * NE)
        blks.append(pb)
        del hn, top_i, sc, flat
    cpu_blks = [pb for pb in blks if pb.cpu]
    for pb in cpu_blks:
        pb.rows = torch.from_numpy(np.concatenate([np.arange(pb.offs[e], pb.offs[e + 1]) for e in pb.cpu])).to(dev)
        Pc = int(pb.rows.numel())
        pb.xh = _pinned(f"x{pb.i}", Pc * H).view(Pc, H)
        pb.wh = _pinned(f"w{pb.i}", Pc)
        pb.yh = _pinned(f"y{pb.i}", Pc * H).view(Pc, H)
        pb.xh.copy_(pb.xs.index_select(0, pb.rows).float(), non_blocking=True)
        pb.wh.copy_(pb.ws.index_select(0, pb.rows).float(), non_blocking=True)
    if cpu_blks:
        t_s = _pc()
        main.synchronize()                              # every block's pinned inputs are complete
        PTIME["cpu_in_sync_s"] += _pc() - t_s
    stg_all = sorted({e for pb in blks for e in pb.stg})
    jobs = []
    stager = None
    try:
        for pb in cpu_blks:
            coff = np.zeros(len(pb.cpu) + 1, np.int64)
            np.cumsum([pb.counts[e] for e in pb.cpu], out=coff[1:])
            gids = [L * NE + e for e in pb.cpu]
            xn, wn, yn = pb.xh.numpy(), pb.wh.numpy(), pb.yh.numpy()

            def _run(gids=gids, coff=coff, xn=xn, wn=wn, yn=yn):       # bound per block, not per name
                t_r = _pc()
                CPUX.experts([ROWP[g] for g in gids], coff, xn, wn, [BGP[g] for g in gids], [BDP[g] for g in gids], out=yn)
                PTIME["cpu_job_run_s"] += _pc() - t_r               # worker thread; main reads it only after the join
            jobs.append(CPU_WORKER.submit(_run))
            PTIME["n_cpu_jobs"] += 1
        if stg_all:                                     # the copies start now: they overlap the resident launches below
            stager = _Stager([L * NE + e for e in stg_all])
        if tb is not None:
            _pf_layer_grouped(L, blks, stg_all, stager, main, tb)
        else:
            for pb in blks:                             # resident: no staging, overlaps the CPU jobs
                for e in pb.res:
                    _expert_gemms(L, e, rt.slot_of[(L, e)], pb.xs, pb.ws, pb.ys, int(pb.counts[e]), int(pb.offs[e]))
            si = 0
            for e in stg_all:                           # expert-major: one copy, then every block that has it
                s_ = SCR_SLOTS[si]
                si = (si + 1) % len(SCR_SLOTS)
                stager.stage(s_, main)
                for pb in blks:
                    if e in pb.stg_set:
                        _expert_gemms(L, e, s_, pb.xs, pb.ws, pb.ys, int(pb.counts[e]), int(pb.offs[e]))
                fe = torch.cuda.Event()
                fe.record(main)                         # after the LAST block's GEMMs on this slot
                SLOT_FREE[s_] = fe
    finally:
        try:
            if stager is not None:
                stager.close()                          # joins the staging producer: it never outlives this call
        finally:
            t_j = _pc()
            job_errs = [j.exception() for j in jobs]    # waits: no CPU job outlives this call
            PTIME["cpu_job_wait_s"] += _pc() - t_j
    for err in job_errs:
        if err is not None:
            raise err
    for pb in cpu_blks:
        pb.ys.index_copy_(0, pb.rows, pb.yh.to(dev, non_blocking=True).to(pb.ys.dtype))
    PSTAT["res"] += sum(len(pb.res) for pb in blks)
    PSTAT["cpu"] += sum(len(pb.cpu) for pb in blks)
    PSTAT["stg"] += len(stg_all)
    PSTAT["stg_pairs"] += sum(len(pb.stg) for pb in blks)
    PTIME["n_pairs"] += sum(len(pb.res) + len(pb.stg) for pb in blks)      # GEMM (expert, block) pairs, either path
    outs = []
    for pb in blks:
        back = torch.empty_like(pb.order)
        back[pb.order] = torch.arange(pb.order.numel(), device=dev)
        outs.append(pb.ys.index_select(0, back).view(pb.T, K, -1).float().sum(1).to(pb.dt))
        pb.xs = pb.ws = pb.ys = pb.rbi = None           # freed block by block
    PTIME["moe_host_s"] += _pc() - t_moe0
    return outs


# ---------------------------------------------------------------- prompt processing
# Two orders (--prefill-order). Both process the prompt in blocks of --prefill-chunk tokens and are layer-major
# WITHIN a block; they differ over the prompt:
#   block: block 0 through all layers, then block 1 ... (one block's activations alive; an expert used by
#          several blocks is staged once per block)
#   layer: layer 0 over all blocks, then layer 1 ... (all blocks' residuals alive; a staged expert is copied
#          once per layer, moe_prefill_layer)
def _rope(t, cos, sin):
    a_, b_ = t.chunk(2, dim=-1)
    return torch.cat((a_ * cos - b_ * sin, b_ * cos + a_ * sin), dim=-1)


def _rope_tables(p0, T):
    """cos/sin [T, 1, 32] (attention scale folded in) for positions p0..p0+T-1."""
    pos = torch.arange(p0, p0 + T, device=dev, dtype=torch.float32)
    freqs = pos[:, None] * cg.inv_freq.float()[None, :]              # [T, 32]
    cos = (freqs.cos() * cg.att_scale).to(bf)[:, None, :]
    sin = (freqs.sin() * cg.att_scale).to(bf)[:, None, :]
    return cos, sin


def _attn_router(L, x, p0, cs):
    """Layer L's attention and router for one block x [T, H] at positions p0..p0+T-1 (cs = _rope_tables). K/V go
    straight into the static decode cache (or the rolling ring). Shared by both prefill orders, so the
    attention calls are the same in both. Returns (x after the attention residual, hn, top_i, sc)."""
    t_ar = _pc()                                    # host launch time of the async attention/router kernels
    T = x.shape[0]
    cos, sin = cs
    lg = LAY[L]
    ly = lg.layer
    at = ly.self_attn
    h = lg._rms(ly.input_layernorm, x)
    qkv = F.linear(h, FC.W[L], FC.B[L])
    q = _rope(qkv[:, :4096].reshape(T, 64, 64), cos, sin).contiguous()
    k = _rope(qkv[:, 4096:4608].reshape(T, 8, 64), cos, sin)
    v = qkv[:, 4608:].reshape(T, 8, 64)
    del qkv, h
    R_ = getattr(lg, "ring", 0)
    if R_:
        # rolling cache: keys < p0 come from the ring, the block's own K/V from k/v; then the
        # last min(T, R) positions are written into their ring slots (fused_core, bit-identical)
        o = prefill_attention(q, lg.k, lg.v, at.sinks, p0, lg.scaling, lg.sliding, lg.smax,
                              k_new=k, v_new=v, ring=R_)
        ring_write(lg.k, lg.v, k, v, p0, R_)
    else:
        lg.k[0, :, p0:p0 + T, :] = k.transpose(0, 1)
        lg.v[0, :, p0:p0 + T, :] = v.transpose(0, 1)
        o = prefill_attention(q, lg.k, lg.v, at.sinks, p0, lg.scaling, lg.sliding, lg.smax)
    x = x + F.linear(o.reshape(T, 4096), at.o_proj.weight, at.o_proj.bias)
    del q, k, v, o
    hn = lg._rms(ly.post_attention_layernorm, x)
    r = ly.mlp.router
    top_v, top_i = torch.topk(F.linear(hn, r.weight, r.bias), K, dim=-1)
    sc = F.softmax(top_v, dim=1, dtype=top_v.dtype)
    PTIME["attn_router_s"] += _pc() - t_ar
    return x, hn, top_i, sc


def _prefill_block(ids, p0):
    """Block order (--prefill-order block): ONE block of prompt tokens at positions p0..p0+T-1 through all
    layers. Layer-major within the block, block-major over the prompt: each non-resident expert crosses PCIe
    once per block, and again for every further block that routes to it. K/V go straight into the static decode
    cache. Returns final hidden [T, H]."""
    T = len(ids)
    x = EMB_ROWS(ids).to(dev)                                       # [T, H] bf16
    cs = _rope_tables(p0, T)
    for L in range(NL):
        x, hn, top_i, sc = _attn_router(L, x, p0, cs)
        x = x + (moe_prefill(L, hn, top_i, sc) if A.fast_prefill else rt.forward_prefill(L, hn, top_i, sc))
    return x


def _prefill_layer_major(ids, p0, allow_grouped=True):
    """Layer order (--prefill-order layer): every LAYER runs over all blocks of the prompt before the next layer
    starts. Per layer: attention of each block in order (block b-1's K/V of this layer are in the cache/ring
    before block b attends, as in block order), then moe_prefill_layer for all blocks at once. Keeps every block's
    residual stream alive (T*H*2 B, 75 MB at 13k tokens) and, inside the MoE, the sorted expert inputs and outputs
    of all blocks (~46 KB/token). Returns the final hidden [T_b, H] of every block."""
    C = A.prefill_chunk
    starts = list(range(0, len(ids), C))
    xs = [EMB_ROWS(ids[a:a + C]).to(dev) for a in starts]
    css = [_rope_tables(p0 + a, len(ids[a:a + C])) for a in starts]
    for L in range(NL):
        hns, tis, scs = [], [], []
        for b, a in enumerate(starts):
            xs[b], hn, ti, sc = _attn_router(L, xs[b], p0 + a, css[b])
            hns.append(hn); tis.append(ti); scs.append(sc)
        del hn, ti, sc
        if A.fast_prefill:
            ms = moe_prefill_layer(L, hns, tis, scs, allow_grouped=allow_grouped)
        else:                                            # reference MoE: per block, no staging to share
            ms = [rt.forward_prefill(L, h_, t_, s_) for h_, t_, s_ in zip(hns, tis, scs)]
        for b in range(len(xs)):
            xs[b] = xs[b] + ms[b]
            ms[b] = None
    return xs


def _logits(xb):
    return F.linear(LAY[0]._rms(model.model.norm, xb), model.lm_head.weight).float()


@torch.inference_mode()
def prefill(ids, p0, all_logits=False, order=None, hidden_out=None):
    """Prompt processing in blocks of --prefill-chunk tokens (bounds temporary VRAM); later blocks attend to
    earlier ones through the cache. order (default --prefill-order): 'block' = each block through all layers
    (_prefill_block), 'layer' = each layer over all blocks (_prefill_layer_major; a prompt of one block is
    processed the block way, the two are the same computation there). Bit-identical results either way.
    Returns last-token logits (or all positions' logits for validation). hidden_out: a list that receives
    every block's final hidden state (--prefill-order-check)."""
    order = order or A.prefill_order
    ptime_reset()                                   # PTIME covers exactly this call (read by the caller afterwards)
    COUNT_PREFILL[0] = True
    outs = []
    # Grouped kernels help medium prompts; tiny prompts showed no gain and
    # multi-block prompts regressed. Decide once per call without changing globals;
    # the same decision controls entry and every layer's expert dispatch.
    allow_grouped = (not _MXK.PREFILL_GROUPED_GEMM or A.grouped_prefill_max_tokens == 0
                     or (0 < len(ids) <= min(A.prefill_chunk, A.grouped_prefill_max_tokens)
                         and len(ids) >= A.grouped_prefill_min_tokens))
    try:
        if order == "layer" and (len(ids) > A.prefill_chunk or (_MXK.PREFILL_GROUPED_GEMM and allow_grouped)):
            blocks = _prefill_layer_major(ids, p0, allow_grouped=allow_grouped)
        else:
            blocks = (_prefill_block(ids[a:a + A.prefill_chunk], p0 + a) for a in range(0, len(ids), A.prefill_chunk))
        for xb in blocks:
            if hidden_out is not None:
                hidden_out.append(xb)
            if all_logits:
                outs.append(_logits(xb))
            last = xb[-1:]
            del xb
    finally:
        COUNT_PREFILL[0] = False
    if all_logits:
        return torch.cat(outs)
    return _logits(last)[0]


# ---------------------------------------------------------------- engine state
LOCK = threading.Lock()
PCOUNTS = np.zeros((NL, NE), np.int64)          # prompt-processing routing counts (--count-routing)
if A.count_routing:
    MODE["counting"] = True                     # decode_token then counts into COUNTS[L, ids]
CACHE = []                     # token ids whose K/V are valid in the static cache (positions 0..len-1)
REQ_SEQ = [0]                  # bumped by every request; an idle job aborts if it changed
START_ASSIST = tok("<|start|>assistant", add_special_tokens=False).input_ids


def canon_job(seq, messages, content, tools, effort):
    """Process [messages + this final answer] exactly as the next request will render it."""
    with LOCK:
        if REQ_SEQ[0] != seq:
            return                                   # a newer request already ran
        try:
            t0 = time.perf_counter()
            ids = render(list(messages) + [{"role": "assistant", "content": content}], tools, effort, info={})
            if ids[-len(START_ASSIST):] != START_ASSIST:
                return
            ids = ids[:-len(START_ASSIST)]           # the next request continues right here
            if len(ids) >= A.smax - 16:
                return
            c = 0
            for a_, b_ in zip(CACHE, ids):
                if a_ != b_:
                    break
                c += 1
            if c >= len(ids):
                return
            if KVC is not None:
                c = KVC.rewind(c, len(CACHE))
                KVC.begin_write(c)
            del CACHE[c:]             # K/V past c are about to be overwritten
            prefill(ids[c:], c)
            CACHE[:] = ids
            if KVC is not None:
                KVC.take(len(ids), ring_snapshot())
            torch.cuda.synchronize()
            print(f"[{time.strftime('%H:%M:%S')}] idle: pre-processed {len(ids) - c} tokens of the "
                  f"conversation for the next turn in {time.perf_counter() - t0:.1f} s", flush=True)
        except Exception as e:                       # never let the idle job break the server
            print(f"idle job skipped: {e}", flush=True)
STOP_REQ = [False]


def flush_admission():
    torch.cuda.synchronize()
    if PEND:
        upd = []
        for g, s, bb in PEND:
            upd.append((g, s)); TAB[g] = s; OWNER[s] = g
            if bb >= 0:
                CAPSTATE[bb] = 0
        PEND.clear()
        PENDG.clear()
        apply_tab_updates(upd)
        torch.cuda.synchronize()
    so = rt.slot_of
    so.clear()
    for s in range(N_USABLE):
        if OWNER[s] >= 0:
            so[(int(OWNER[s]) // NE, int(OWNER[s]) % NE)] = s


class Harmony:
    """Incremental parser of the model's harmony output. Calls emit(kind, text) with
    kind in {reasoning, content}; collects at most one tool call (the model stops
    at <|call|>). Detokenization is incremental (a small token window per step)."""

    def __init__(self, emit):
        self.emit = emit
        self.state, self.header, self.buf = "header", [], []
        self.channel, self.recipient = None, None
        self.tool = None                    # (name, arguments_text)
        self.final_done = False
        self._reset_msg()

    def _reset_msg(self):
        self.buf, self.p_off, self.r_off, self.sent = [], 0, 0, ""

    def _kind(self):
        if self.recipient:
            return "tool"
        return "reasoning" if self.channel == "analysis" else "content"

    def _flush(self, end=False):
        kind = self._kind()
        if end:                                          # reconcile with one full decode
            full = tok.decode(self.buf, skip_special_tokens=False)
            if kind != "tool" and full.startswith(self.sent) and len(full) > len(self.sent):
                self.emit(kind, full[len(self.sent):])
            self.sent = full
            return
        if kind == "tool":                               # arguments are decoded once, at the end
            return
        prefix = tok.decode(self.buf[self.p_off:self.r_off], skip_special_tokens=False)
        new = tok.decode(self.buf[self.p_off:], skip_special_tokens=False)
        if len(new) > len(prefix) and not new.endswith("\ufffd"):
            delta = new[len(prefix):]
            self.emit(kind, delta)
            self.sent += delta
            self.p_off, self.r_off = self.r_off, len(self.buf)

    def feed(self, t):
        if t == TKN["<|start|>"]:
            self.state, self.header = "header", []
            return
        if t == TKN["<|message|>"] and self.state == "header":
            htxt = tok.decode(self.header, skip_special_tokens=False)
            m = re.search(r"<\|channel\|>\s*([A-Za-z_]+)", htxt)
            self.channel = m.group(1) if m else "final"
            m = re.search(r"to=([^\s<]+)", htxt)
            self.recipient = m.group(1) if m and m.group(1) != "assistant" else None
            self.state = "content"
            self._reset_msg()
            return
        if t in (TKN["<|end|>"], TKN["<|return|>"], TKN["<|call|>"]):
            if self.state == "content":
                self._flush(end=True)
                if self.recipient:
                    name = self.recipient
                    name = name[len("functions."):] if name.startswith("functions.") else name
                    self.tool = (name, self.sent.strip())
                elif self.channel == "final":
                    self.final_done = True
            self.state = "between"
            return
        if self.state == "header":
            self.header.append(t)
            return
        if self.state == "between":                    # stray text outside a message
            self.state, self.channel, self.recipient = "content", "final", None
            self._reset_msg()
        self.buf.append(t)
        self._flush()


# ---------------------------------------------------------------- request -> prompt
# The rendering logic lives in harmony_render.py (pure, testable without the model:
# dev/test_splice.py). These are thin wrappers bound to this server's tokenizer.
# SPLICE remembers, per returned tool_call id, the exact tokens the model generated
# for that response (reasoning + call through <|call|>); render() splices them back
# in place of the template's re-rendering of that call, so the model keeps its
# reasoning across consecutive tool calls and the prefix KV cache keeps matching.
SPLICE = HR.SpliceMemory(A.splice_memory, call_token=TKN["<|call|>"], max_tokens=A.splice_memory_tokens)


def _text(c):
    return HR.text_of(c)


def _defang(s):
    """Stop text that spells a control token (e.g. a file containing '<|end|>') from
    being tokenized as that control token."""
    return HR.defang(s, SPECIAL_TEXT)


def render(messages, tools, effort, info=None, reserve=0):
    """max_tokens = the bound do_POST enforces (len(ids) >= smax-16 -> 400): a spliced
    prompt that would not fit (with `reserve` tokens left for generation) falls back to
    the unspliced one, so splicing never rejects a request the pre-splice server served."""
    return HR.render(tok, messages, tools, effort, identity=A.identity, special_text=SPECIAL_TEXT,
                     memory=SPLICE, info=info, after_final=A.splice_after_final,
                     max_tokens=A.smax - 16, reserve=reserve)


def splice_reserve(max_new):
    return HR.splice_reserve(max_new, A.splice_reserve, A.smax - 16)


# ---------------------------------------------------------------- generation
# ---- side-request protection: a request that diverges early from the cached conversation
# (e.g. Continue's apply / title requests) would overwrite its K/V; save the rows it will
# touch and restore them afterwards. A request that continues the previous side request is a
# new conversation taking over, and is not protected.
SIDE_LAST = [None]              # token ids of the last protected side request
SIDE_KV = [None]                # (tokens, start, rows): that side request's own K/V, kept in host RAM
SIDE_MAX_BYTES = 1 << 30        # never back up more than 1 GiB of K/V


def kv_save(a, b):
    """Copy K/V rows [a, b) of every layer to host memory (ring layers: the whole ring)."""
    out = []
    for L in range(NL):
        lg = LAY[L]
        if getattr(lg, "ring", 0):
            out.append(("ring", lg.k.cpu(), lg.v.cpu()))
        else:
            out.append(("lin", lg.k[0, :, a:b].cpu(), lg.v[0, :, a:b].cpu()))
    return out


def kv_load(rows, a):
    for L, (kind, k_, v_) in enumerate(rows):
        lg = LAY[L]
        if kind == "ring":
            lg.k.copy_(k_)
            lg.v.copy_(v_)
        else:
            n_ = k_.shape[1]
            lg.k[0, :, a:a + n_].copy_(k_)
            lg.v[0, :, a:a + n_].copy_(v_)


def ring_snapshot():
    """Host copies of the sliding layers' rings (18 x 2 x 8 x R x 64 x 2 B = 9 MiB at R=256)."""
    return [(lg.k.cpu(), lg.v.cpu()) if getattr(lg, "ring", 0) else None for lg in LAY]


def ring_restore(snap):
    for lg, kv in zip(LAY, snap):
        if kv is not None:
            lg.k.copy_(kv[0])
            lg.v.copy_(kv[1])


# ring checkpoints for prompt-cache rollback (kv_ring.py); None without --kv-ring
KVC = RingCheckpoints(A.kv_ring, 128, A.kv_checkpoints, restore=ring_restore) if A.kv_ring else None


def _lcp(x, y):
    n_ = 0
    for p_, q_ in zip(x, y):
        if p_ != q_:
            break
        n_ += 1
    return n_


class ContextFull(Exception):
    pass


def fold_prefill_counts(n_new):
    """Admission counts (GEN) start from the prompt's routing, but a prompt is worth at most
    --prefill-weight generated tokens. A long prompt (a file being read) routes differently from
    the answer written about it; at full weight its counts made admission chase the prompt's
    experts and evict the answer's for the whole first answer (VRAM hit 0.16 vs 0.30 replayed)."""
    w = min(1.0, A.prefill_weight / max(n_new, 1))
    if w >= 1.0:
        GEN[:] += PREF
    elif w > 0.0:
        GEN[:] += (PREF * w).astype(np.int64)
    PREF[:] = 0
    return w


def generate(ids, max_new, temperature, emit, stop_strings=(), ignore_eos=False):
    """Prefill (reusing the cached prefix) + decode. emit(kind, text) streams deltas.
    Returns dict with tool call / finish reason / usage / timings."""
    global GT
    P = len(ids)
    if P >= A.smax - 16:
        raise ContextFull(P)
    budget = max(1, min(max_new or A.smax, A.smax - P - 1))
    c = 0
    for a, b in zip(CACHE, ids):
        if a != b:
            break
        c += 1
    c = min(c, P - 1)
    protect = None
    if A.protect_side and len(CACHE) >= 1024 and c < 0.5 * len(CACHE):
        continues_side = SIDE_LAST[0] is not None and _lcp(SIDE_LAST[0], ids) >= len(SIDE_LAST[0]) - 8
        sk = SIDE_KV[0]
        if continues_side and sk is not None and _lcp(CACHE, sk[0]) >= sk[1]:
            # the last side request was the start of a new conversation: swap its K/V back in
            kv_load(sk[2], sk[1])
            CACHE[:] = sk[0]
            if KVC is not None:
                KVC.import_list(sk[3])
            c = min(_lcp(CACHE, ids), P - 1)
        elif not continues_side:
            c_from = KVC.plan(c, len(CACHE))[0] if KVC is not None else c   # where prefill will really start
            hi = min(len(CACHE), P + budget + 1)
            if hi > c_from and (hi - c_from) * NL * 2 * 8 * 64 * 2 <= SIDE_MAX_BYTES:
                protect = (list(CACHE), c_from, kv_save(c_from, hi), KVC.export_list() if KVC is not None else None)
        SIDE_LAST[0] = list(ids) if protect is not None else None
        SIDE_KV[0] = None
    else:
        SIDE_LAST[0] = None
        SIDE_KV[0] = None
    GEN[:] = 0
    PREF[:] = 0
    CAND.clear()
    pfp0, _, _ = pmem()
    dkp0 = disk_read_bytes()
    pst0 = dict(PSTAT)
    t0 = time.perf_counter()
    if KVC is not None:
        c = KVC.rewind(c, len(CACHE))   # sliding rings: resume where their window exists (may restore a checkpoint)
        KVC.begin_write(c)              # rings are dirty until this pass ends (a failed pass leaves partial writes)
    del CACHE[c:]                 # K/V past c are about to be overwritten: never trust them if prefill fails
    try:
        logits = prefill(ids[c:], c)
    except BaseException:
        if protect is not None:
            kv_load(protect[2], protect[1])
            CACHE[:] = protect[0]
            if KVC is not None:
                KVC.import_list(protect[3])   # whole rings were restored above: state as before this request
        raise
    ptm = dict(PTIME)                 # prompt-processing time split of this prefill (host wall times, see PTIME)
    fold_w = fold_prefill_counts(P - c)
    pst = {k_: PSTAT[k_] - pst0[k_] for k_ in PSTAT}
    pf_pre = (pmem()[0] - pfp0) & 0xFFFFFFFF
    dk_pre = disk_read_bytes() - dkp0
    t_tail0 = time.perf_counter()     # prefill() only queued its last kernels: this first sampled-token sync drains them
    if temperature > 0:
        t = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
        temp_dev.fill_(temperature)
        GT = GT_SAMPLE
    else:
        t = int(logits.argmax())
        GT = GT_GREEDY
    t_tail = time.perf_counter() - t_tail0
    CACHE[:] = ids
    if KVC is not None:
        KVC.take(P, ring_snapshot())      # the next request most likely resumes here (idle re-render, next turn)
    torch.cuda.synchronize()
    trim_stats = {}
    if A.trim_prefill_cache:
        # All prefill CPU jobs were joined by moe_prefill[_layer] and all GPU reads have completed above.
        # Only inactive allocator blocks are released; live graph/KV/model storage is never replaced.
        trim_t0 = time.perf_counter()
        trim_before = torch.cuda.memory_reserved()
        if A.trim_prefill_cache == 2:
            _PIN.clear()
            torch._C._host_emptyCache()
        torch.cuda.empty_cache()
        trim_stats = {"prefill_cache_trim_s": time.perf_counter() - trim_t0,
                      "prefill_cache_released_gib": (trim_before - torch.cuda.memory_reserved()) / 2**30}
    t_prefill = time.perf_counter() - t0
    pts = ptime_stats(ptm, t_tail, t_prefill)
    persistent_before = _persistent_snapshot()
    reset()
    if BELL is not None:
        BELL.begin_request()            # the warm-up window and the fault baseline restart with every request
    RC.update(captures=0, cand_marked=0)
    if A.refresh_every and A.prefill_m:
        refresh_point(A.prefill_m)
    cand_start = RC["cand_marked"]
    content_acc = []

    def _emit(kind, text):
        if kind == "content":
            content_acc.append(text)
        emit(kind, text)

    hp = Harmony(_emit)
    hp.feed(t)
    gen = [t]
    n = 0
    pf0, _, _ = pmem()
    dk0 = disk_read_bytes()
    tick = []
    t1 = time.perf_counter()
    finish = "length"
    try:
        while n < budget - 1 and not STOP_REQ[0]:
            if not ignore_eos and (t in STOPS or hp.final_done or hp.tool):
                break
            if stop_strings and any(s in "".join(content_acc) for s in stop_strings):
                finish = "stop"
                break
            if temperature > 0:
                u_dev.uniform_()
            t = decode_token(t, P + n, True)
            if n % 64 == 63:
                tick.append(time.perf_counter())
            CACHE.append(gen[-1])
            n += 1
            if n == 1 and KVC is not None:
                # position P (the first generated token) is in the rings now. A follow-up re-renders the
                # answer, so it shares this token but rarely more: its K/V is decode's and the linear cache
                # reuses it (cached_tokens P + 1), so the checkpoint it resumes from is taken here, not at P
                KVC.take(P + 1, ring_snapshot(), replaces=P)
            gen.append(t)
            hp.feed(t)
            every = A.early_every if n <= A.early_tokens else A.refresh_every
            if A.refresh_every and n % every == 0:
                refresh_point()
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        pf1, ws1, _ = pmem()
        dk1 = disk_read_bytes()
    finally:
        flush_admission()
        if protect is not None:                        # give the conversation its K/V back
            saved_tokens, a0, rows, cps_saved = protect
            if len(CACHE) > a0 and (len(CACHE) - a0) * NL * 2 * 8 * 64 * 2 <= SIDE_MAX_BYTES:
                SIDE_KV[0] = (list(CACHE), a0, kv_save(a0, len(CACHE)),       # keep the side request's own K/V
                              KVC.export_list() if KVC is not None else None)
            kv_load(rows, a0)
            CACHE[:] = saved_tokens
            if KVC is not None:
                KVC.import_list(cps_saved)
            torch.cuda.synchronize()
    dt = t_end - t1                 # decode only: excludes the side-request K/V restore in finally
    win = [round(64 / (b - a), 2) for a, b in zip([t1] + tick[:-1], tick)]
    aborted = bool(STOP_REQ[0])
    if hp.tool:
        finish = "tool_calls"
    elif t in STOPS or hp.final_done:
        finish = "stop"
    elif aborted:
        finish = "stop"
    natural_final = bool(not hp.tool and not aborted and (hp.final_done or t == TKN["<|return|>"]))
    tot = C["cpu"] + C["hit"] + C["gpuadm"] + C["gpumiss"]
    n_ = max(n, 1)
    return {"tool": hp.tool, "gen_ids": gen,         # gen_ids: popped by do_POST (splice memory), not reported
            "aborted": aborted, "natural_final": natural_final, "side_protected": protect is not None,
            "finish": finish, "prompt_tokens": P, "cached_tokens": c, "completion_tokens": len(gen),
            "prefill_s": t_prefill, "decode_s": dt, "decode_tok_s": (n / dt) if n else 0.0,
            "vram_hit": C["hit"] / max(tot, 1), "cpu_GBps": C["cpu"] * SLOTB / max(C["cpu_s"], 1e-9) / 1e9,
            "cpu_ms_per_tok": C["cpu_s"] / n_ * 1e3, "gpu_wait_ms_per_tok": C["wait_s"] / n_ * 1e3,
            # the serial split that used to be a residual: graph launches, admission bookkeeping, and the rest
            "replay_ms_per_tok": C["replay_s"] / n_ * 1e3, "admission_ms_per_tok": C["adm_s"] / n_ * 1e3,
            "other_ms_per_tok": max(0.0, dt / n_ * 1e3 - (C["cpu_s"] + C["wait_s"] + C["replay_s"] + C["adm_s"]) / n_ * 1e3),
            "gpu_admissions_per_tok": C["gpuadm"] / n_, "gpu_misses_per_tok": C["gpumiss"] / n_,
            "cpu_experts_per_tok": C["cpu"] / n_,
            # --doorbell: tokens run early / classic (mode switch, warm-up, timeout retries) and spins that gave up, this request
            **_bell_stats(n_),
            **_persistent_request_stats(persistent_before),
            "vram_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
            "vram_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
            "prefill_weight": A.prefill_weight, "prefill_fold_w": round(fold_w, 5), "prefill_new_tokens": P - c,
            "cand_marked_start": cand_start, "captures": RC["captures"],
            "decode_page_faults_per_tok": round(((pf1 - pf0) & 0xFFFFFFFF) / max(n, 1)),
            "decode_disk_mb_per_tok": round((dk1 - dk0) / max(n, 1) / 1e6, 2),
            "prefill_page_faults": pf_pre, "prefill_disk_mb": round(dk_pre / 1e6),
            # prompt-processing expert routes: staging copies (PCIe), the (layer, block, expert) pairs they served,
            # and the pairs computed by the CPU kernel; --prefill-order layer lowers the first, not the others
            "prefill_order": A.prefill_order, "prefill_stagings": pst["stg"], "prefill_stg_pairs": pst["stg_pairs"],
            "prefill_cpu_experts": pst["cpu"],
            **trim_stats,
            **pts,     # host-time split of prompt processing (ptime_stats): memcpy / ring wait / CPU-job wait / attn / moe
            "decode_tok_s_per_64": win, "ws_gib_end": round(ws1, 2), "free_ram_gib_end": free_ram_gib()}


# ---------------------------------------------------------------- HTTP
MODEL_ID = A.model_name
LOG_DIR = os.path.join(S, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_BODIES = os.environ.get("NEURAL_LOG_BODIES") == "1"


def log_request(req, r, tc, wall_s):
    """One JSON line per request: timings + the SHAPE of the conversation (roles, which
    assistant messages carry reasoning / tool calls) - not its text, unless NEURAL_LOG_BODIES=1."""
    shape = []
    for m in req.get("messages") or []:
        e = {"role": m.get("role"), "chars": len(_text(m.get("content")))}
        for k in ("reasoning_content", "reasoning", "thinking"):
            if m.get(k):
                e[k] = len(_text(m.get(k)))
        if m.get("tool_calls"):
            e["tool_calls"] = [tc_.get("function", {}).get("name") for tc_ in m["tool_calls"]]
        shape.append(e)
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(wall_s, 3),
           "stream": bool(req.get("stream")), "n_tools": len(req.get("tools") or []),
           "effort": req.get("reasoning_effort"), "temperature": req.get("temperature"),
           "max_tokens": req.get("max_completion_tokens") or req.get("max_tokens"),
           "tool_call": tc["function"]["name"] if tc else None, "messages": shape}
    rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "tool"})
    if LOG_BODIES:
        rec["body"] = req
    try:
        with open(os.path.join(LOG_DIR, "requests.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
API_KEY = os.environ.get(A.api_key_env) or None


def _chunk(cid, created, delta, finish=None):
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    timeout = 120                      # a stalled client can no longer hold LOCK forever

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg, typ="invalid_request_error", err_code=None):
        self._json(code, {"error": {"message": msg, "type": typ, "code": err_code}})

    def _auth(self):
        if not API_KEY:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {API_KEY}":
            return True
        self._err(401, "invalid API key", "authentication_error", "invalid_api_key")
        return False

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            if self._auth():
                self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0,
                                                              "owned_by": "neural-local", "context_length": A.smax}]})
        elif self.path.rstrip("/") == "/neural/trace":         # per-token decode routing (diagnostic)
            snap = list(TRACE)                                  # atomic under the GIL; never wait on a running request
            body = {"enabled": bool(MODE["trace"]), "hidden_enabled": bool(MODE["trace_hidden"]), "tokens": len(snap),
                    "ids": [t[0].tolist() for t in snap],
                    "weights": [t[1].astype(np.float32).tolist() for t in snap],
                    "static_slots": int(STATIC_N), "usable_slots": int(N_USABLE),
                    "static_gids": [int(OWNER[s_]) for s_ in range(STATIC_N)]}
            if snap and snap[0][2] is not None:
                body["near_miss"] = [t[2].tolist() for t in snap]          # router ranks 5-8 per layer (fused core)
            if snap and snap[0][3] is not None:
                body["hidden"] = [t[3].astype(np.float32).tolist() for t in snap]   # final residual per token, f16 values
            self._json(200, body)
        elif self.path.rstrip("/") == "/neural/modules":       # packaging diagnostic
            self._json(200, sorted((n, getattr(m, "__file__", None)) for n, m in list(sys.modules.items())
                                   if n == "neural" or n.startswith("neural.")))
        elif self.path.rstrip("/") == "/neural/routing":
            self._json(200, {"decode_counts": COUNTS.tolist(), "prefill_counts": PCOUNTS.tolist(),
                             "decode_tokens": int(COUNTS.sum() // K), "enabled": bool(A.count_routing)})
        elif self.path.rstrip("/") in ("/health", "/v1/health", ""):
            self._json(200, {"status": "ok", "model": MODEL_ID, "context_length": A.smax})
        else:
            self._err(404, f"unknown path {self.path}")

    def do_POST(self):
        if self.path.rstrip("/") == "/neural/trace":           # {"on": true|false}: start/stop decode tracing
            if not self._auth():
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                on = bool(body.get("on"))
                hidden = bool(body.get("hidden", False))
            except (ValueError, AttributeError):
                return self._err(400, "request body is not valid JSON")
            with LOCK:
                MODE["trace"] = on
                MODE["trace_hidden"] = on and hidden      # also record the final residual per token (5.7 KB each)
                if on:
                    TRACE.clear()
            return self._json(200, {"enabled": on, "hidden": MODE["trace_hidden"]})
        if self.path.rstrip("/") == "/neural/config":          # runtime tuning knobs (localhost; key-checked if set)
            if not self._auth():
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._err(400, "request body is not valid JSON")
            lim = {"cpu_prefill_max": (0, 4096), "prefill_chunk": (16, A.smax), "idle_canon": (0, 1),
                   "protect_side": (0, 1), "prefill_weight": (0, 1 << 30)}
            try:
                vals = {k: int(body[k]) for k in lim if k in body}
            except (TypeError, ValueError):
                return self._err(400, "config values must be integers")
            bad = [k for k, v in vals.items() if not lim[k][0] <= v <= lim[k][1]]
            if bad:
                return self._err(400, f"out of range: {bad}; limits {lim}")
            with LOCK:
                for k, v in vals.items():
                    setattr(A, k, v)
                return self._json(200, {k: getattr(A, k) for k in lim})
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._err(404, f"unknown path {self.path} (supported: /v1/chat/completions, /v1/models)")
        if not self._auth():
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._err(400, "request body is not valid JSON")
        tools = req.get("tools") if req.get("tool_choice") != "none" else None
        effort = req.get("reasoning_effort") or (req.get("reasoning") or {}).get("effort") or A.effort
        if effort not in ("low", "medium", "high"):
            effort = A.effort
        temperature = req.get("temperature")
        temperature = A.temperature if temperature is None else float(temperature)
        max_new = req.get("max_completion_tokens") or req.get("max_tokens")
        stop = req.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        stream = bool(req.get("stream"))
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        _t_queue = time.perf_counter()
        with LOCK:
            _queue_wait_s = time.perf_counter() - _t_queue
            STOP_REQ[0] = False
            REQ_SEQ[0] += 1
            my_seq = REQ_SEQ[0]
            final_text = []
            splice = {}
            _t_render = time.perf_counter()
            try:
                ids = render(req.get("messages") or [], tools, effort, info=splice, reserve=splice_reserve(max_new))
            except Exception as e:                        # template rejected the conversation
                return self._err(400, f"could not render messages: {e}")
            _render_s = time.perf_counter() - _t_render
            if len(ids) >= A.smax - 16:
                return self._err(400, f"This model's maximum context length is {A.smax} tokens. However, your "
                                      f"messages resulted in {len(ids)} tokens.", err_code="context_length_exceeded")
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def send(obj):
                    try:
                        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                        self.wfile.flush()
                    except OSError:
                        STOP_REQ[0] = True                # client went away: stop generating

                send(_chunk(cid, created, {"role": "assistant", "content": ""}))

                def emit(kind, text):
                    if kind == "content":
                        final_text.append(text)
                    send(_chunk(cid, created, {"reasoning_content": text} if kind == "reasoning" else {"content": text}))
            else:
                parts = {"reasoning": [], "content": []}

                def emit(kind, text):
                    parts[kind].append(text)
                    if kind == "content":
                        final_text.append(text)
            _t_req = time.perf_counter()
            try:
                r = generate(ids, max_new, temperature, emit, stop, bool(req.get("ignore_eos")))
            except Exception as e:
                import traceback
                traceback.print_exc()
                msg_ = f"generation failed: {type(e).__name__}: {e}"
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    msg_ += " (out of GPU memory - try a smaller --prefill-chunk or --pool)"
                if stream:
                    send({"error": {"message": msg_, "type": "server_error", "code": None}})
                    send(_chunk(cid, created, {}, "stop"))
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass
                    return
                return self._err(500, msg_, "server_error")
            gen_ids = r.pop("gen_ids")
            # Model-phase timings omit queueing, template/tokenization, and final restoration.
            r["queue_wait_s"] = _queue_wait_s
            r["render_s"] = _render_s
            r["generation_wall_s"] = time.perf_counter() - _t_req
            # Diagnostics include reasoning/control tokens, unlike a hash of visible answer text alone.
            r["prompt_ids_sha256"] = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
            r["generated_ids_sha256"] = hashlib.sha256(np.asarray(gen_ids, dtype="<u4").tobytes()).hexdigest()
            r.update(HR.report_fields(splice))           # splice outcome -> neural object, requests.jsonl, console
            tc = None
            if r["tool"]:
                tc = {"id": "call_" + secrets.token_hex(12), "type": "function",
                      "function": {"name": r["tool"][0], "arguments": r["tool"][1]}}
                # exact tokens of this turn, for re-rendering it later (kept only if they end in one <|call|>)
                SPLICE.remember(tc["id"], gen_ids, tc["function"]["name"], tc["function"]["arguments"])
            usage = {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                     "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
                     "prompt_tokens_details": {"cached_tokens": r["cached_tokens"]}}
            neural = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "tool"}
            log_request(req, r, tc, time.perf_counter() - _t_req)
            print(f"[{time.strftime('%H:%M:%S')}] prompt {r['prompt_tokens']} tok ({r['cached_tokens']} cached) "
                  f"processed in {r['prefill_s']:.1f} s | {r['completion_tokens']} tok at {r['decode_tok_s']:.1f} tok/s "
                  f"| VRAM hit {r['vram_hit']:.0%} | {r['finish']}" + (f" -> {tc['function']['name']}" if tc else "")
                  + HR.report_note(r) + ptime_note(r),
                  flush=True)
            if stream:
                if tc:
                    send(_chunk(cid, created, {"tool_calls": [{"index": 0, "id": tc["id"], "type": "function",
                                                               "function": {"name": tc["function"]["name"], "arguments": ""}}]}))
                    send(_chunk(cid, created, {"tool_calls": [{"index": 0, "function": {"arguments": tc["function"]["arguments"]}}]}))
                send(_chunk(cid, created, {}, r["finish"]))
                if (req.get("stream_options") or {}).get("include_usage"):
                    send({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                          "choices": [], "usage": usage, "neural": neural})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            else:
                msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if tc else "")}
                if parts["reasoning"]:
                    msg["reasoning_content"] = "".join(parts["reasoning"])
                if tc:
                    msg["tool_calls"] = [tc]
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": r["finish"]}], "usage": usage,
                                 "neural": neural})
            if A.idle_canon and not tc and r.get("natural_final") and not r.get("side_protected") and not req.get("ignore_eos"):
                threading.Thread(target=canon_job, daemon=True,
                                 args=(my_seq, req.get("messages") or [], "".join(final_text), tools, effort)).start()


def touch_experts(gids):
    """Bring experts' store pages into this process's working set: one large asynchronous
    prefetch (deep disk queue), then a pass of the CPU kernel with zero inputs to map them."""
    arr = (_MRE * len(gids))(*[_MRE(ROWP[g], SLOTB) for g in gids])
    _k32.PrefetchVirtualMemory(_HPROC, len(gids), arr, 0)
    if A.warm_method != "kernel":
        from neural_runtime.page_warm import touch_pages, touch_page_ranges
        # Only page residency changes. Keep the prefetch ranges and hotness
        # order, bounded to one worker wave; no model state is modified.
        if A.warm_method == "pages-parallel":
            checksum = touch_page_ranges(((ROWP[g], SLOTB) for g in gids),
                                         workers=min(A.warm_workers, os.cpu_count() or 1), batch=1)
        else:
            checksum = 0
            for g in gids:
                checksum ^= touch_pages(ROWP[g], SLOTB)
        res["config"]["warm_page_checksum"] = checksum
        if A.kernel_persistent:
            # The legacy dummy math initialized the team as a side effect. Keep
            # initialization in startup, using the existing no-op rendezvous.
            probe = lib.gptoss_team_probe_batch
            probe.argtypes = [ctypes.c_int] * 4 + [VP, VP]
            probe.restype = ctypes.c_int
            duration, mark = ctypes.c_double(), ctypes.c_ulonglong()
            status = probe(1, A.threads, 1, 0, ctypes.byref(duration), ctypes.byref(mark))
            if status:
                raise RuntimeError(f"persistent CPU team startup failed: {status}")
        return
    Pw, BGw, BDw, CPw = (VP * K)(), (VP * K)(), (VP * K)(), (VP * K)()
    xz, wz, oz = np.zeros(H, np.float32), np.zeros(K, np.float32), np.zeros(H, np.float32)
    for i in range(0, len(gids), K):
        ch = gids[i:i + K]
        for j, g in enumerate(ch):
            Pw[j], BGw[j], BDw[j], CPw[j] = ROWP[g], BGP[g], BDP[g], None
        lib.gptoss_experts_cap(len(ch), Pw, xz.ctypes.data, BGw, BDw, wz.ctypes.data, oz.ctypes.data, SP, A.threads, CPw)


def warm_scores():
    """Expected use of each expert: the hot-set counts (hotset_code.json = coding decode counts
    + 0.25 x general, see tools/calibrate_hotset.py) plus, by default, the general calibration
    counts again, i.e. about 0.8 coding + 1.2 general after normalising. Answers start with
    plain-language reasoning, whose experts a coding-heavy ranking puts last."""
    s = GC / max(float(GC.sum()), 1.0)
    fp = os.path.join(S, "hotset_freq.json")
    if A.warm_general and os.path.exists(fp):
        gf = np.array(json.load(open(fp))["counts"], dtype=np.float64).ravel()
        s = s + gf / max(float(gf.sum()), 1.0)
    return s


def prewarm():
    budget = int((free_ram_gib() - A.warm_margin) * 2**30 // SLOTB)
    skip = lambda g: TAB[g] >= 0 or g in ARENA_ROWS   # resident (GPU) or already locked in the arena
    gids = [int(g) for g in np.argsort(-warm_scores(), kind="stable") if not skip(g)][:max(budget, 0)]  # the CPU's experts
    if not gids:
        return 0
    if A.warm_order == "likely-last":
        gids = gids[::-1]              # most likely touched last = youngest pages = the last Windows trims
    t0 = time.perf_counter()
    touch_experts(gids)
    res["config"]["warm_method"] = A.warm_method
    res["config"]["warm_s"] = time.perf_counter() - t0
    print(f"RAM cache warmed: {len(gids) * SLOTB / 2**30:.1f} GiB of experts in {res['config']['warm_s']:.2f} s "
          f"(method {A.warm_method})", flush=True)
    return len(gids)


def hold_working_set():
    """Soft minimum working set = the size reached after the warm-up (+0.5 GiB). With the store
    at the edge of free RAM, Windows otherwise trims the expert pages while a long prompt is
    processed and the answer soft-faults them back, ~3,200 pages per expert contending for one
    lock (MEASURED: first answers 5-13 tok/s over their first 64-128 tokens, follow-ups ~17).
    Soft minimum (QUOTA_LIMITS_HARDWS_MIN_DISABLE): Windows trims other processes' idle pages
    first, but can still go below it if memory gets scarce. Nothing persists after exit."""
    _, ws, _ = pmem()
    lo = int((ws + 0.5) * 2**30)
    hi = int(psutil.virtual_memory().total)
    _k32.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
    ok = _k32.SetProcessWorkingSetSizeEx(_HPROC, lo, max(hi, lo + (1 << 30)), 0x2 | 0x8)   # min soft, max soft
    err = 0 if ok else ctypes.get_last_error()
    print(f"working set: soft minimum {lo / 2**30:.1f} GiB " + ("set" if ok else f"NOT set (error {err})"), flush=True)
    return bool(ok)


# ---------------------------------------------------------------- experimental decode levers
# (their module-level state is defined next to the admission tables, before decode_token, because the
# hot-set calibration path runs decode_token during import; setup_levers(), called from main, turns
# features on only after their preconditions and self-tests pass)


def admit_ok(g):
    """Should this miss be admitted (computed by the GPU into a slot)? Rule per --admit-rule."""
    if A.admit_rule == "server":
        return g in CAND
    if A.admit_rule == "window":
        return WCNT[g] + 1 >= A.admit_k
    return DEMAND[g] > VMIN[0]                   # demand: beats the least-demanded adaptive resident


def setup_arena():
    global ARENA, ARENA_ROWS
    import arena as _arena
    ARENA = _arena.ExpertArena(A.arena, SLOTB)
    cand = [int(g) for g in np.argsort(-warm_scores(), kind="stable") if TAB[g] < 0]   # non-resident, likeliest first
    budget_gib = A.arena_max_gib if A.arena_max_gib > 0 else max(0.0, free_ram_gib() - A.warm_margin)
    want = min(len(cand), int(budget_gib * 2**30 // SLOTB))
    t0 = time.perf_counter()
    got = ARENA.alloc(want)
    gids = ARENA.assign(cand[:got])
    print(f"arena: allocated {got} of {want} rows wanted ({len(cand)} non-resident experts) in "
          f"{time.perf_counter() - t0:.1f}s; filling from the store...", flush=True)
    dt = ARENA.fill(gids, lambda g: ps.row(g // NE, g % NE), after_row=lambda g: release_rows([g]))
    for g, p in ARENA.row_ptr.items():
        ROWP[g] = p
    ARENA_ROWS = set(ARENA.row_ptr)
    print(f"{ARENA.summary()}; filled in {dt:.0f}s ({ARENA.gib / max(dt, 1e-9):.2f} GiB/s); "
          f"free RAM {free_ram_gib()} GiB; {len(cand) - len(gids)} experts stay on the mmap path", flush=True)
    res["config"].update(arena_rows=len(gids), arena_gib=round(ARENA.gib, 2), arena_gpu_visible=ARENA.gpu_visible)


def load_probe(path):
    z = np.load(path)
    W = np.asarray(z["W"], dtype=np.float32)
    b = np.asarray(z["bias"], dtype=np.float32) if "bias" in z.files else None
    if W.shape[0] == H + 1 and b is None:
        W, b = W[:-1], W[-1]
    assert W.shape == (H, NL * NE), W.shape
    if b is None:
        b = np.zeros(NL * NE, np.float32)
    mu = np.asarray(z["mean"], dtype=np.float32).reshape(H)
    sd = np.asarray(z["std"], dtype=np.float32).reshape(H) + 1e-6
    return {"W": torch.from_numpy(W).to(dev, torch.float16), "b": torch.from_numpy(b).to(dev),
            "mu": torch.from_numpy(mu).to(dev), "sd": torch.from_numpy(sd).to(dev),
            "horizon": int(z["horizon"]) if "horizon" in z.files else None}


def setup_gpu_misses():
    """GPU-computed misses: the copy engine streams an expert from the pinned arena into a slot (the same
    cudaMemcpyAsync path the shipped admission upload uses, at the MEASURED 21.4 GiB/s pinned rate) and the
    pool GEMV computes it there, bit-identical to a VRAM hit. Captures one slot-GEMV graph per possible
    GPU miss of a layer and verifies the path on a real arena row. Returns True when the path is usable."""
    global ADMEV, ADMEV2, ADM_PIN, ADM_WPIN, ADM_DEV, ADM_WDEV, GADM
    if not ARENA_ROWS:
        print("gpu misses: the arena holds no rows; staying on the CPU-only miss path", flush=True)
        return False
    nmax = max(1, (A.admit_gpu_max if A.admit_gpu else 0) + A.zc_misses)
    ADMEV = [torch.cuda.Event() for _ in range(ADM_RING)]       # copy landed (copy stream)
    ADMEV2 = [torch.cuda.Event() for _ in range(ADM_RING)]      # GEMV done (main stream)
    ADM_PIN = [torch.zeros(2, dtype=torch.long).pin_memory() for _ in range(ADM_RING)]
    ADM_WPIN = [torch.zeros(1, dtype=bf).pin_memory() for _ in range(ADM_RING)]
    ADM_DEV = [torch.zeros(2, dtype=torch.long, device=dev) for _ in range(nmax)]
    ADM_WDEV = [torch.zeros(1, dtype=bf, device=dev) for _ in range(nmax)]
    g0 = min(ARENA_ROWS)
    s0 = rt.scratch_slots[0]
    rt._rows[s0].copy_(ARENA.row_view(g0))                      # valid slot contents for the capture warm-up
    for k_ in range(nmax):
        ADM_DEV[k_].copy_(torch.tensor([s0, g0], dtype=torch.long))
        ADM_WDEV[k_].fill_(1.0)
    torch.cuda.synchronize()

    def adm_body(k_):
        slot, gid = ADM_DEV[k_][0:1], ADM_DEV[k_][1:2]
        gu = mxfp4_gemv(bx, rt.gate_blocks, rt.gate_scales, slot, block_n=32, block_g=4, num_warps=8).view(1, GUN)
        gu = gu + rt.bias_gu.index_select(0, gid)
        h = rt._act(gu)
        y = mxfp4_gemv(h, rt.down_blocks, rt.down_scales, slot, per_expert_x=True, block_n=32, block_g=4, num_warps=8)
        y = y + rt.bias_dn.index_select(0, gid)
        g_out.add_(y.view(1, H) * ADM_WDEV[k_])
    with torch.inference_mode():
        GADM = [capture(lambda k_=k_: adm_body(k_)) for k_ in range(nmax)]
    if not A.selftest_admit:
        return True
    # --- self-test on the scratch slot: arena bytes == store row, graph == eager pool path, ~= CPU kernel
    rng = torch.Generator(device="cpu").manual_seed(1)
    x = (torch.randn(1, H, generator=rng) * 0.8).to(bf)
    with torch.inference_mode():
        bx.copy_(x.to(dev))
        g_out.zero_()
        GADM[0].replay()
        torch.cuda.synchronize()
        y_zc = g_out.clone()
        ok_bytes = bool(torch.equal(rt._rows[s0].cpu(), ps.row(g0 // NE, g0 % NE)))
        slots1 = torch.tensor([s0], device=dev)
        gid1 = torch.tensor([g0], device=dev)
        gu = mxfp4_gemv(bx, rt.gate_blocks, rt.gate_scales, slots1, block_n=32, block_g=4, num_warps=8).view(1, GUN)
        gu = gu + rt.bias_gu.index_select(0, gid1)
        h = rt._act(gu)
        y = mxfp4_gemv(h, rt.down_blocks, rt.down_scales, slots1, per_expert_x=True, block_n=32, block_g=4, num_warps=8)
        y_pool = (y + rt.bias_dn.index_select(0, gid1)).view(1, H)
        ok_pool = bool(torch.equal(y_pool, y_zc))
    Pw, BGw, BDw, CPw = (VP * K)(), (VP * K)(), (VP * K)(), (VP * K)()
    Pw[0], BGw[0], BDw[0], CPw[0] = ROWP[g0], BGP[g0], BDP[g0], None
    xz = np.ascontiguousarray(x.float().numpy().reshape(H))
    wz = np.ones(K, np.float32)
    oz = np.zeros(H, np.float32)
    lib.gptoss_experts_cap(1, Pw, xz.ctypes.data, BGw, BDw, wz.ctypes.data, oz.ctypes.data, SP, A.threads, CPw)
    d = np.abs(y_zc.float().cpu().numpy().reshape(H) - oz)
    rel = float(d.max() / (np.abs(oz).max() + 1e-6))
    # the GPU path rounds gu / h / y to bf16 like every pool hit does, the CPU path is fp32 throughout:
    # a few percent is the expected numerics gap; a layout or pointer error gives garbage (>> 100%)
    ok_cpu = rel < 0.1
    print(f"gpu-miss self-test on expert {g0} -> scratch slot {s0}: arena row == store row: {ok_bytes}; "
          f"graph output == eager pool path (bit): {ok_pool}; vs CPU kernel max|d|/max|y| = {rel:.2e} ({'ok' if ok_cpu else 'BAD'})",
          flush=True)
    res["config"].update(gpu_miss_selftest={"bytes": ok_bytes, "pool_bitequal": ok_pool, "cpu_rel": rel})
    if not (ok_bytes and ok_pool and ok_cpu):
        print("GPU miss path DISABLED: self-test failed", flush=True)
        return False
    return True


def setup_levers():
    """Arena, GPU-computed misses (admission / extra pipe) and demand probe: after the hot set is loaded,
    before serving."""
    global GPU_MISS, ADMIT_GPU, ZC_MISSES, POLICY_STATE, PROBE
    if A.arena != "off":
        setup_arena()
    if A.demand_probe:
        PROBE = load_probe(A.demand_probe)
        print(f"demand probe loaded from {A.demand_probe} (horizon {PROBE['horizon']})", flush=True)
    if A.admit_gpu or A.zc_misses:
        GPU_MISS = setup_gpu_misses()
        ADMIT_GPU = bool(GPU_MISS and A.admit_gpu)
        ZC_MISSES = int(A.zc_misses) if GPU_MISS else 0
    POLICY_STATE = bool(ADMIT_GPU or PROBE is not None or A.victim == "lru" or A.admit_rule != "server")
    res["config"].update(admit_gpu_active=bool(ADMIT_GPU), zc_misses_active=ZC_MISSES, policy_state=POLICY_STATE)
    if ADMIT_GPU:
        print(f"zero-surcharge GPU admission ON: rule {A.admit_rule} (k={A.admit_k}, window={A.admit_window}), "
              f"victim {A.victim}, at most {A.admit_gpu_max}/layer; copy-on-compute capture OFF", flush=True)
    if ZC_MISSES:
        print(f"GPU extra read pipe ON: up to {ZC_MISSES} plain misses per layer via DMA into scratch slots", flush=True)


def selftest():
    """Prompt-processing check: layer-major Triton prefill vs the HF-eager prefill on
    the same prompt (same expert path), plus prefix reuse vs a full prefill."""
    import torch.nn.functional as F_
    if A.kv_ring:
        raise SystemExit("--selftest compares against the HF eager prefill (absolute K/V positions); run it without --kv-ring")
    text = "\n\n".join(_CAL["selftest_first12"])
    ids = render([{"role": "system", "content": "You are a careful assistant."}, {"role": "user", "content": text}], None, "low")
    P = len(ids)
    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(inputs_embeds=EMB_ROWS(ids).unsqueeze(0).to(dev), use_cache=True)
        torch.cuda.synchronize(); t_hf = time.perf_counter() - t0
        lhf_all = out.logits[0].float()
        lhf = lhf_all[-1]
        cg.load_prefill_kv(out.past_key_values, P)
        def span(L):
            return (max(0, P - 128), P) if LAY[L].sliding else (0, P)   # HF keeps only the window for sliding layers
        kref = {L: (LAY[L].k[0, :, span(L)[0]:span(L)[1]].float().clone(), LAY[L].v[0, :, span(L)[0]:span(L)[1]].float().clone())
                for L in (0, 1, 17, 35)}
        del out
        prefill(ids, 0)                                   # warm-up pass (cold page cache / first-call costs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lmy = prefill(ids, 0)
        torch.cuda.synchronize(); t_my = time.perf_counter() - t0
        kv_rel = {}
        for L, (a, b) in kref.items():
            lo_, hi_ = span(L)
            keep = a.flatten(0, 0).abs().sum(dim=(0, 2)) > 0            # rows the reference really holds
            ka = LAY[L].k[0, :, lo_:hi_].float()[:, keep]; va = LAY[L].v[0, :, lo_:hi_].float()[:, keep]
            kv_rel[L] = round(max(((ka - a[:, keep]).norm() / a[:, keep].norm()).item(),
                                  ((va - b[:, keep]).norm() / b[:, keep].norm()).item()), 5)
        logit_max_diff = (lmy - lhf).abs().max().item()
        kl = F_.kl_div(F_.log_softmax(lmy, -1), F_.log_softmax(lhf, -1), log_target=True, reduction="sum").item()
        cut = P * 2 // 3
        prefill(ids[:cut], 0)
        lsplit = prefill(ids[cut:], cut)
        kl2 = F_.kl_div(F_.log_softmax(lsplit, -1), F_.log_softmax(lmy, -1), log_target=True, reduction="sum").item()
        from neural.quality.harness import tf_metrics
        nt = torch.tensor(ids[1:], device=dev)
        lmy_all = prefill(ids, 0, all_logits=True)
        tf_mine = tf_metrics(lmy_all[:-1], lhf_all[:-1], nt)
        import fused_core as _fc
        _orig = _fc.prefill_attention
        tf_alt = {}
        for bm, bn in ((32, 32), (128, 32)):
            globals()["prefill_attention"] = lambda *a, _bm=bm, _bn=bn, **k: _orig(*a, BM=_bm, BN=_bn, **k)
            la = prefill(ids, 0, all_logits=True)
            tf_alt[f"mine_BM{bm}_BN{bn}_vs_mine"] = tf_metrics(la[:-1], lmy_all[:-1], nt)
        globals()["prefill_attention"] = _orig
    r = {"prompt_tokens": P, "hf_eager_prefill_s": round(t_hf, 2), "layer_major_prefill_warm_s": round(t_my, 2), "direct_prefill": bool(A.direct_prefill),
         "argmax_equal": int(lmy.argmax()) == int(lhf.argmax()), "kl_vs_hf": kl, "logit_max_abs_diff": logit_max_diff, "max_rel_kv_err": kv_rel,
         "prefix_reuse_argmax_equal": int(lsplit.argmax()) == int(lmy.argmax()), "prefix_reuse_kl": kl2, "teacher_forced_all_positions_mine_vs_hf": tf_mine, "same_path_tiling_variants": tf_alt}
    print("SELFTEST", json.dumps(r), flush=True)
    return r



def ppl_check():
    """Prompt processing vs the HF reference path, teacher-forced on natural text and code:
    top-1 agreement, mean KL and perplexity of both (validation for prefill changes)."""
    from neural.quality.harness import tf_metrics
    texts = {
        "prose": ("The Industrial Revolution began in Great Britain in the late eighteenth century and spread to "
                  "continental Europe and North America over the following decades. It transformed economies that had "
                  "been based on agriculture and handicrafts into economies based on large-scale industry, mechanized "
                  "manufacturing, and the factory system. New machines, new power sources, and new ways of organizing "
                  "work made existing industries more productive and efficient."),
        "code": open(os.path.join(S, "neural", "quality", "harness.py"), encoding="utf-8").read()[:2600],
    }
    out = {}
    for name, text in texts.items():
        ids = tok(text, add_special_tokens=False).input_ids
        nt = torch.tensor(ids[1:], device=dev)
        with torch.inference_mode():
            o = model(inputs_embeds=EMB_ROWS(ids).unsqueeze(0).to(dev), use_cache=True)
            lhf = o.logits[0].float()
            del o
            PSTAT.update(res=0, cpu=0, stg=0, stg_pairs=0)
            lmy = prefill(ids, 0, all_logits=True)
        m = tf_metrics(lmy[:-1], lhf[:-1], nt)
        out[name] = {"tokens": len(ids), "top1_vs_hf": round(m["top1"], 4), "mean_kl": m["mean_kl"],
                     "ppl_mine": m["ppl"], "ppl_hf": tf_metrics(lhf[:-1], lhf[:-1], nt)["ppl"], "routes": dict(PSTAT)}
    print("PPLCHECK", json.dumps(out), flush=True)


def _bits_equal(a, b):
    """torch.equal on the raw bit patterns (== alone calls -0.0 equal to 0.0 and NaN unequal to itself)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    w = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return torch.equal(a.contiguous().view(w), b.contiguous().view(w))


@torch.inference_mode()
def prefill_order_check():
    """--prefill-order-check: the same long prompt through prefill() from position 0, K/V caches (and rings) zeroed
    before every run so a K/V write a run missed shows up as zeros, not as the previous run's values. Compares bit
    for bit: every position's final hidden state, the logits of EVERY position (computed from the hidden states
    in 512-row chunks: the full [P, 201k] fp32 logits of a 13k-token prompt would be 10 GB) and every layer's K/V
    (the whole buffer for a rolling ring layer, which holds the same final state whatever the blocking).
    Comparisons, one line each:
      PREFILL_ORDER_CHECK          block order vs layer order at the configured --prefill-chunk
      PREFILL_CHUNK_CHECK          block order at chunk 4096 vs 16384 (one block for a 13k prompt)
      PREFILL_CHUNK_CHECK_GPUONLY  the same with --cpu-prefill-max 0 for both runs (no CPU kernel)
      PREFILL_BM_CHECK             prefill GEMM tile height BLOCK_M 16 vs 64 (mxfp4_kernels.PREFILL_BLOCK_M), configured order
      PREFILL_FAST_LAUNCH_CHECK    stock launch + torch-op path vs the fast one (mxfp4_kernels.FAST_LAUNCH), configured order
    Residency is frozen during all of it (prefill admits nothing: PREF counts are folded and admission happens in
    generate()), so an expert's arithmetic depends on residency only through WHERE it runs: resident -> GPU from its
    slot; not resident and <= cpu_prefill_max tokens IN THAT BLOCK -> CPU kernel; else staged -> GPU from a scratch
    slot (same mxfp4_gemm, same bytes as a resident slot). The per-block token counts change with the chunk size, so
    the CPU/GPU split can differ between chunk sizes (cpu_experts_a/b show it); the GPUONLY line removes that source.
    The self-test's own sample is only ~250 tokens (one block), so it is followed by the repo's README, docs and
    sources to reach the requested length."""
    if not A.fast_prefill:
        raise SystemExit("--prefill-order-check needs --fast-prefill 1 (layer order only changes the sync-free MoE)")
    files = ["README.md"] + ["docs/" + n for n in sorted(os.listdir(os.path.join(S, "docs"))) if n.endswith(".md")] \
        + ["fused_core.py", "server.py"]
    parts = ["\n\n".join(_CAL["selftest_first12"])]
    parts += [open(os.path.join(S, f), encoding="utf-8").read() for f in files if os.path.exists(os.path.join(S, f))]
    ids = tok("\n\n".join(parts), add_special_tokens=False).input_ids
    P = min(len(ids), A.prefill_order_check_tokens, A.smax - 17)
    ids = ids[:P]
    nb = -(-P // A.prefill_chunk)
    if nb < 2:
        print(f"WARNING: {P} tokens is {nb} block of --prefill-chunk {A.prefill_chunk}; nothing to compare "
              f"(raise --smax / --prefill-order-check-tokens or lower --prefill-chunk)", flush=True)
    slots0 = dict(rt.slot_of)

    def run(order, chunk=None, cpu_max=None):
        """prefill from 0; --prefill-chunk / --cpu-prefill-max overridden for this run only (prefill() reads them per call)."""
        saved = (A.prefill_chunk, A.cpu_prefill_max)
        A.prefill_chunk = chunk or saved[0]
        A.cpu_prefill_max = saved[1] if cpu_max is None else cpu_max
        try:
            for lg in LAY:
                lg.k.zero_()
                lg.v.zero_()
            PSTAT.update(res=0, cpu=0, stg=0, stg_pairs=0)
            hid = []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            last = prefill(ids, 0, order=order, hidden_out=hid)
            torch.cuda.synchronize()
            return {"last": last, "hid": hid, "s": time.perf_counter() - t0, "ps": dict(PSTAT)}
        finally:
            A.prefill_chunk, A.cpu_prefill_max = saved

    def snap():
        return [(lg.k[..., :P, :].cpu(), lg.v[..., :P, :].cpu()) for lg in LAY]   # the whole ring for a rolling layer

    def compare(ra, kv_a, rb):
        """Run a (hidden states + K/V snapshot taken right after it) against run b (K/V still live in the cache)."""
        bad_kv = [L for L, lg in enumerate(LAY)
                  if not (_bits_equal(lg.k[..., :P, :].cpu(), kv_a[L][0]) and _bits_equal(lg.v[..., :P, :].cpu(), kv_a[L][1]))]
        ha, hb = torch.cat(ra["hid"]), torch.cat(rb["hid"])                    # [P, H] whatever the blocking
        bad_pos = (ha.view(torch.int16) != hb.view(torch.int16)).any(1).nonzero().flatten()
        logits_eq = _bits_equal(ra["last"], rb["last"])
        max_diff = (ra["last"] - rb["last"]).abs().max().item()
        for i in range(0, P, 512):
            la, lb = _logits(ha[i:i + 512]), _logits(hb[i:i + 512])
            logits_eq = logits_eq and _bits_equal(la, lb)
            max_diff = max(max_diff, (la - lb).abs().max().item())
        return {"logits_eq": logits_eq, "kv_eq": not bad_kv, "max_diff": max_diff, "hid_eq": not len(bad_pos),
                "first_bad": int(bad_pos[0]) if len(bad_pos) else -1, "bad_kv": bad_kv}

    frozen = lambda: dict(rt.slot_of) == slots0

    run("block")                                        # warm-up: page cache, first-call costs
    ra = run("block")
    kv = snap()
    rb = run("layer")
    c = compare(ra, kv, rb)
    del kv
    routes_eq = all(ra["ps"][k_] == rb["ps"][k_] for k_ in ("res", "cpu", "stg_pairs"))
    print(f"PREFILL_ORDER_CHECK logits_equal={c['logits_eq']} kv_equal={c['kv_eq']} max_abs_logit_diff={c['max_diff']:.6g} "
          f"block_s={ra['s']:.3f} layer_s={rb['s']:.3f} stagings_block={ra['ps']['stg']} stagings_layer={rb['ps']['stg']} "
          f"hidden_equal={c['hid_eq']} routes_equal={routes_eq} prompt_tokens={P} blocks={nb} chunk={A.prefill_chunk} "
          f"kv_ring={A.kv_ring} stg_pairs={ra['ps']['stg_pairs']} cpu_experts_block={ra['ps']['cpu']} "
          f"cpu_experts_layer={rb['ps']['cpu']} residency_frozen={frozen()}", flush=True)
    if c["bad_kv"] or not c["hid_eq"]:
        print(f"PREFILL_ORDER_CHECK mismatch: first differing position {c['first_bad']}, kv layers {c['bad_kv']}", flush=True)
    ca, cb = 4096, 16384
    for name, cpu_max in (("PREFILL_CHUNK_CHECK", None), ("PREFILL_CHUNK_CHECK_GPUONLY", 0)):
        ra = run("block", ca, cpu_max)
        kv = snap()
        rb = run("block", cb, cpu_max)
        c = compare(ra, kv, rb)
        del kv
        print(f"{name} chunk_a={ca} chunk_b={cb} logits_equal={c['logits_eq']} kv_equal={c['kv_eq']} "
              f"max_abs_logit_diff={c['max_diff']:.6g} a_s={ra['s']:.3f} b_s={rb['s']:.3f} hidden_equal={c['hid_eq']} "
              f"first_diff_pos={c['first_bad']} prompt_tokens={P} cpu_prefill_max={A.cpu_prefill_max if cpu_max is None else cpu_max} "
              f"cpu_experts_a={ra['ps']['cpu']} cpu_experts_b={rb['ps']['cpu']} stagings_a={ra['ps']['stg']} "
              f"stagings_b={rb['ps']['stg']} kv_ring={A.kv_ring} residency_frozen={frozen()}", flush=True)

    # The two host-side settings of the prefill GEMM path, each switched inside this process and restored. Same run() /
    # compare() as above: every position's final hidden state, the logits of every position (from the hidden states, in
    # 512-row chunks: prefill(all_logits=True) would hold a [P, 201k] fp32 tensor) and every layer's K/V, bit for bit.
    def ab(name, tag_a, tag_b, set_a, set_b, restore, warm_b):
        """Run the configured order under setting a, then under setting b, and compare; restore the setting."""
        try:
            if warm_b:
                set_b()
                run(A.prefill_order)                     # untimed: compiles/loads what setting b launches first
            set_a()
            ra_ = run(A.prefill_order)
            kv_ = snap()
            set_b()
            rb_ = run(A.prefill_order)
            c_ = compare(ra_, kv_, rb_)
            del kv_
        finally:
            restore()
        print(f"{name} {tag_a} {tag_b} logits_equal={c_['logits_eq']} kv_equal={c_['kv_eq']} "
              f"max_abs_logit_diff={c_['max_diff']:.6g} a_s={ra_['s']:.3f} b_s={rb_['s']:.3f}", flush=True)
        if c_["bad_kv"] or not c_["hid_eq"]:
            print(f"{name} mismatch: first differing position {c_['first_bad']}, kv layers {c_['bad_kv']}", flush=True)

    bm0, fl0 = _MXK.PREFILL_BLOCK_M, _MXK.FAST_LAUNCH
    _set = lambda **kw: (lambda: [setattr(_MXK, k_, v_) for k_, v_ in kw.items()])
    ab("PREFILL_BM_CHECK", "bm_a=16", "bm_b=64", _set(PREFILL_BLOCK_M=16), _set(PREFILL_BLOCK_M=64),
       _set(PREFILL_BLOCK_M=bm0), True)
    ab("PREFILL_FAST_LAUNCH_CHECK", "fast_a=0", "fast_b=1", _set(FAST_LAUNCH=False), _set(FAST_LAUNCH=True),
       _set(FAST_LAUNCH=fl0), False)


def main():
    if A.eager_core:
        raise SystemExit("--eager-core is not supported by the server (prompt processing needs the fused core)")
    if A.review_prefill_check:
        from tools.review_prefill_check import run
        run(globals())
        return
    if A.review_decode_check:
        from tools.review_decode_check import run
        run(globals())
        return
    if A.review_cpu_decode_check:
        from tools.review_cpu_decode_check import run
        run(globals())
        return
    if A.selftest:
        selftest()
        return
    if A.ppl_check:
        ppl_check()
        return
    if A.prefill_order_check:
        prefill_order_check()
        return
    setup_levers()
    if A.prewarm:
        prewarm()
    if A.ws_min:
        hold_working_set()
    srv = http.server.ThreadingHTTPServer((A.host, A.port), Handler)
    print("\n" + "=" * 78)
    print(f" Neural OpenAI-compatible server  |  model id: {MODEL_ID}")
    print(f" base URL : http://{A.host}:{A.port}/v1")
    print(f" API key  : " + ("required (from env " + A.api_key_env + ")" if API_KEY else "not checked - any value works (e.g. 'local')"))
    print(f" context  : {A.smax} tokens | reasoning effort default: {A.effort} | default temperature: {A.temperature}")
    print(f" runtime  : gpt-oss-120B MXFP4" + (" (packed scales)" if SCALE_MODE else "")
          + f", H8 hybrid (static core {STATIC_N}, VRAM pool {N_USABLE} experts)")
    print(f" splicing : " + (f"{A.splice_memory} calls / {A.splice_memory_tokens} tokens | after final answer: "
                             f"{A.splice_after_final} | reserve {A.splice_reserve} tokens" if A.splice_memory > 0 else "off"))
    print("=" * 78, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("shutting down")


main()
