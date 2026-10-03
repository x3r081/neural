/* HYBRID-1 Stage C: gpt-oss-120B MXFP4 expert on the CPU, batch 1.
 *
 * Row layout (Neural store, 13,219,200 B):
 *   gate_codes [5760][90][16] | down_codes [2880][90][16] | gate_scales [5760][90] | down_scales [2880][90]
 * codes: fp4 e2m1, 2 per byte, LOW nibble = even position. scale: E8M0, value 2^(s-127).
 * Store gate_up rows are de-interleaved: out[0:2880] = gate, out[2880:5760] = up.
 * act: gate = min(gate,7); up = clamp(up,-7,7); h = (up+1) * gate * sigmoid(1.702*gate)
 *
 * The 16 fp4 values are exactly one zmm register, so a nibble dequantizes with a
 * single vpermps. x is de-interleaved once per GEMV (xe = even, xo = odd lanes),
 * so the inner loop needs no shuffles: blk = lut[lo]*xe + lut[hi]*xo; acc += blk*2^(s-127).
 *
 * MEMORY-LEVEL-PARALLELISM KNOBS (gptoss_set_tuning; all default OFF = the shipped kernel):
 *   prefetch_bytes  software prefetch this many bytes ahead of each code stream (one
 *                   _mm_prefetch per 64 B). The out-of-order window of this loop covers only
 *                   ~4-7 cache lines per core; prefetch lets more misses be in flight.
 *   pair_rows       1 = each thread iteration streams TWO adjacent rows with independent
 *                   accumulators (two sequential streams per core, double the outstanding
 *                   misses). The per-row arithmetic and its order are exactly rowdot()'s, so
 *                   every output is bit-identical to pair_rows = 0.
 *   affinity_stride 0 = off; s > 0 pins OpenMP thread t to logical CPU t*s once (2 = one thread
 *                   per physical core on an 8C/16T Intel part). libgomp reuses its pool threads
 *                   across calls, so the pin persists.
 * Neither knob changes a single floating-point operation; see tools/kernel_tuning_ab.py.
 *
 * SCHEDULING (2026-09-30): the fused path runs its two row loops with schedule(dynamic) and joins them with one
 * spin barrier (pause; SwitchToThread after ~50 us; WaitOnAddress after ~1 ms) instead of two static loops with libgomp's sleeping barriers.
 * Static chunks left one slow thread streaming alone at the end of each phase and every barrier cost ~40 us of
 * sleep/wake; the kernel is bandwidth-bound, so those tails were the gap between ~33 and ~37 GB/s. Each output element
 * is still computed by exactly one thread with the same instruction sequence, so outputs are bit-identical
 * (tools/kernel_tuning_ab.py --ref, and in-server per-turn content_sha1 with residency frozen). MEASURED: kernel
 * -12% standalone, no collapse with 4-6 busy background threads (the yield fallback), in-server see docs/LEVERS.md.
 * The blocking fallback keeps fault-heavy regimes (first answers, rewrites: hundreds of page faults per token) free of
 * the stall windows a pure spin/yield wait showed there; it is never reached in the normal ~50 us wait.
 *
 * SCALE LAYOUT SWITCH (gptoss_set_scale_layout; default 0 = the raw layout above; server.py sets it from the
 * expert store's weight_repr: mxfp4_g32 -> 0, mxfp4_g32_ps4 -> 1):
 *   mode 1  packed scales. Codes are untouched; the scale region (8,640 rows, gate rows then down rows,
 *           at OFF_GS) stores each 90-scale row as 46 B: byte 0 = base (the row's minimum scale),
 *           bytes 1..45 = 4-bit deltas, byte 1+k holds groups 2k (LOW nibble) and 2k+1 (HIGH nibble).
 *           scale[g] = base + delta[g], an exact integer, so 2^(s-127) is formed exactly as before and every
 *           output is bit-identical to mode 0 on the same logical data. Needs row span (max-min) <= 15.
 *           Slot = 12,441,600 + 8,640 x 46 = 12,839,040 B (-2.88%). The capture path copies the packed bytes.
 *   The mode is read ONCE per call and selects a whole-function instantiation, so the inner loops carry no
 *   mode test. It is process-global PER DLL: gptoss_cpu_multi.c (the prefill kernel, which #includes this file)
 *   is a separate DLL with its own copy of the switch and reads it in gptoss_experts_multi too, so a process
 *   must set the mode on every kernel DLL it loads (cpu_prefill.bind_scale_layout does).
 *
 * COLD-PAGE PREFETCH (gptoss_set_cold_prefetch(on, probe_threshold_cycles); opt-in, default OFF; Windows only, a no-op
 * elsewhere; process-global PER DLL like the knobs above). When the slots point into an mmap'd store file, an expert
 * whose pages are not in the process working set makes the threads take page faults inside the row loops (soft: page on
 * the standby list, ~0.66 us/page; hard: page repurposed, read from NVMe, ~5 ms/expert), and page-by-page faulting from
 * 8 threads keeps the NVMe queue depth low. With the knob on, every call first PROBES each expert on the calling thread,
 * before sb_init() and the parallel region: four 1-byte reads at spread offsets (gate codes row 1000, up codes row
 * H+1000, down codes row 1000, one gate/up scale byte), each timed with rdtsc between lfences. A read slower than
 * probe_threshold_cycles TSC ticks (default 100000 = ~30 us = a hard fault; a soft fault is only ~2k ticks and a
 * DRAM + TLB miss ~0.3-1k, so a soft-fault gate needs a threshold of a few thousand) flags the expert, and ALL flagged
 * experts of the call go into ONE PrefetchVirtualMemory call (whole slot, page-rounded; resolved with GetProcAddress
 * like WaitOnAddress) so the OS reads their pages in parallel at queue depth before the threads start streaming.
 * Nothing else changes: no arithmetic is touched, outputs are bit-identical with the knob on or off, and a resident
 * expert costs 4 loads + 8 rdtsc + 8 lfence, no syscall (~1-2 us per call with cold caches: CALCULATED, not measured).
 * Limits: four samples find a partly cold expert with probability 1-(1-f)^4 for cold fraction f; the probe of a cold
 * expert stops at its first slow read, but that read is itself one synchronous fault; a probe thread preempted
 * mid-read can false-flag (the price is one useless PrefetchVirtualMemory). gptoss_cold_prefetch_stats(long long*
 * out5) reads {calls, experts probed, experts flagged cold, PVM calls, probe TSC ticks (probe loop only, PVM call
 * time excluded)}; gptoss_set_cold_prefetch zeroes them; gptoss_get_cold_prefetch(int* cfg) reports {effective on,
 * threshold, PrefetchVirtualMemory resolved}. threshold <= 0 selects the default.
 */
#ifndef _WIN32
  #define _GNU_SOURCE            /* CPU_SET / pthread_setaffinity_np on Linux builds (test harness only) */
#endif
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
  /* kernel32 only; declared by hand so that neither this DLL nor gptoss_cpu_multi.dll
   * (which #includes this file) pulls in <windows.h>. */
  __declspec(dllimport) void* __stdcall GetCurrentThread(void);
  __declspec(dllimport) unsigned long long __stdcall SetThreadAffinityMask(void*, unsigned long long);
  __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
  __declspec(dllimport) int __stdcall GetProcessAffinityMask(void*, unsigned long long*, unsigned long long*);
#else
  #define __declspec(x)
  #include <sched.h>
  #include <pthread.h>
#endif

#define H   2880
#define GU  5760
#define GK  90
#define RB  1440                              /* code bytes per row */
#define OFF_DC  (GU * RB)                     /* 8,294,400  */
#define OFF_GS  (OFF_DC + H * RB)             /* 12,441,600 */
#define OFF_DS  (OFF_GS + GU * GK)            /* 12,960,000 */
#define SR_P4   (1 + GK / 2)                  /* packed-scale row: base + 45 nibble bytes = 46 B */
#define SLOT_RAW    (OFF_DS + H * GK)         /* 13,219,200 */
#define SLOT_PACKED (OFF_GS + (GU + H) * SR_P4)   /* 12,839,040 */
/* one spin barrier per fused call (phase 1 -> phase 2). All team threads are live inside the region, so the usual wait
 * is the fork skew or the phase-1 tail (tens of us): spin with pause. Past ~50 us it yields the CPU (SwitchToThread),
 * and past ~1 ms it blocks (WaitOnAddress on the generation word, woken by the last arriver) so a teammate stuck in a
 * page-fault storm or descheduled on a busy box never has seven cores spinning against it. Measured: without the
 * yield the barrier collapsed 2-3x with 4-6 busy background threads; without the block, fault-heavy rewrites showed
 * stall windows that the sleeping libgomp barrier did not. */
#ifdef _WIN32
  __declspec(dllimport) int __stdcall SwitchToThread(void);
  __declspec(dllimport) void* __stdcall GetModuleHandleA(const char*);
  __declspec(dllimport) void* __stdcall GetProcAddress(void*, const char*);
  typedef int (__stdcall *sb_wait_t)(volatile void*, void*, size_t, unsigned long);
  typedef void (__stdcall *sb_wake_t)(void*);
  static sb_wait_t sb_wait; static sb_wake_t sb_wake; static int sb_inited;
  static void sb_init(void){
    void* h = GetModuleHandleA("kernelbase.dll");
    if (h){ sb_wait = (sb_wait_t)GetProcAddress(h, "WaitOnAddress"); sb_wake = (sb_wake_t)GetProcAddress(h, "WakeByAddressAll"); }
    sb_inited = 1;
  }
  #define SB_YIELD() SwitchToThread()
#else
  #include <sched.h>
  static void sb_init(void){}
  #define SB_YIELD() sched_yield()
#endif
typedef struct { volatile long cnt; char pad[60]; volatile long gen; char pad2[60]; } sbar_t;
static sbar_t g_sb;
static inline void spin_barrier(int n){
  long g = __atomic_load_n(&g_sb.gen, __ATOMIC_SEQ_CST);
  if (__atomic_add_fetch(&g_sb.cnt, 1, __ATOMIC_SEQ_CST) == n){
    __atomic_store_n(&g_sb.cnt, 0, __ATOMIC_SEQ_CST); __atomic_add_fetch(&g_sb.gen, 1, __ATOMIC_SEQ_CST);
#ifdef _WIN32
    if (sb_wake) sb_wake((void*)&g_sb.gen);
#endif
  } else {
    unsigned long long t0 = __rdtsc();
    while (__atomic_load_n(&g_sb.gen, __ATOMIC_SEQ_CST) == g){
      unsigned long long dt = __rdtsc() - t0;
      if (dt < 150000ULL) _mm_pause();                       /* ~50 us: the normal wait */
#ifdef _WIN32
      else if (dt > 3000000ULL && sb_wait){ long cmp = g; sb_wait(&g_sb.gen, &cmp, sizeof(long), 5); }   /* ~1 ms+: block */
#endif
      else SB_YIELD();
    }
  }
}

#define AINL static inline __attribute__((always_inline))

/* ---------------------------------------------------------------- tuning knobs (process-global) */
static int g_pf   = 0;      /* prefetch distance in bytes, 0 = off; clamped to [0, 16384], multiple of 64 */
static int g_pair = 0;      /* 1 = two rows per iteration */
static int g_aff  = 0;      /* affinity stride, 0 = off */
static int g_fuse = 0;      /* 1 = two-phase kernel (see gptoss_set_fuse) */
static int g_sl   = 0;      /* scale layout: 0 = raw 90 B/row, 1 = packed 4-bit deltas 46 B/row */
/* Pin state is per OS THREAD, not per OpenMP thread number: every OS thread that calls the kernel
 * (the main thread at startup, then each HTTP handler thread) becomes the master of its own libgomp
 * team with its own pool threads, so a process-global table indexed by omp_get_thread_num() pinned
 * only the first team and silently left every later team unpinned. Each thread compares its own
 * stamp with the generation of the current stride setting and re-pins once per change. */
static int g_aff_gen = 0;             /* bumped by every gptoss_set_tuning that changes the stride */
static __thread int t_aff_gen = 0;    /* this thread's stamp: pinned for generation t_aff_gen */

__declspec(dllexport) void gptoss_set_tuning(int prefetch_bytes, int pair_rows, int affinity_stride){
  if (prefetch_bytes < 0) prefetch_bytes = 0;
  if (prefetch_bytes > 16384) prefetch_bytes = 16384;
  g_pf = (prefetch_bytes / 64) * 64;
  g_pair = pair_rows ? 1 : 0;
  const int aff = affinity_stride > 0 ? affinity_stride : 0;
  if (aff != g_aff) g_aff_gen++;      /* every thread re-pins on its next call (stride 0 = no pin) */
  g_aff = aff;
}

/* cfg[0] prefetch_bytes, cfg[1] pair_rows, cfg[2] affinity_stride, cfg[3] fuse (effective values) */
__declspec(dllexport) void gptoss_get_tuning(int* cfg){ cfg[0] = g_pf; cfg[1] = g_pair; cfg[2] = g_aff; cfg[3] = g_fuse; }

/* fuse = 1: two worksharing phases instead of four. Phase 1 streams gate row j and up row H+j
 * of an expert together and applies the activation on the spot (the scalar expf hides under
 * the memory stalls of the next rows); phase 2 streams down row n of every expert and forms
 * the weighted sum directly. Same per-element float operations in the same order as the
 * four-phase kernel (bit-identical); three fewer barriers per call, which matters when a layer
 * has 1-3 misses (~1 ms of work) rather than the 4 experts the A/B tool times. */
__declspec(dllexport) void gptoss_set_fuse(int fuse){ g_fuse = fuse ? 1 : 0; }

/* Scale layout of the slots handed to gptoss_experts / gptoss_experts_cap (and of the capture slots they
 * fill). 0 = raw, 1 = packed 4-bit deltas. Returns 0 on success, -1 for an unsupported mode (unchanged).
 * Process-global and not thread-safe against calls in flight: set it once, before serving. */
__declspec(dllexport) int gptoss_set_scale_layout(int mode){
  if (mode != 0 && mode != 1) return -1;
  g_sl = mode;
  return 0;
}
__declspec(dllexport) int gptoss_get_scale_layout(void){ return g_sl; }
/* Bytes per slot in the given layout (-1 = unsupported). */
__declspec(dllexport) long long gptoss_slot_bytes(int mode){
  return mode == 0 ? (long long)SLOT_RAW : mode == 1 ? (long long)SLOT_PACKED : -1LL;
}

/* ---------------------------------------------------------------- cold-page prefetch (see the header comment) */
#define CP_DEFAULT_THRESH 100000     /* TSC ticks (~30 us): above any DRAM/TLB miss or soft fault, below an NVMe read */
#define CP_MAXR           64         /* ranges per PrefetchVirtualMemory call (stack array, flushed when full) */
static int g_cp     = 0;                       /* effective switch: requested AND (Windows with PrefetchVirtualMemory) */
static int g_cp_thr = CP_DEFAULT_THRESH;
static long long g_cp_stat[5];                 /* calls, experts probed, experts flagged, PVM calls, probe ticks */
#ifdef _WIN32
  typedef struct { void* addr; size_t bytes; } cp_range_t;                       /* WIN32_MEMORY_RANGE_ENTRY */
  /* BOOL PrefetchVirtualMemory(HANDLE hProcess, ULONG_PTR NumberOfEntries, PWIN32_MEMORY_RANGE_ENTRY, ULONG Flags) */
  typedef int (__stdcall *cp_pvm_t)(void*, size_t, cp_range_t*, unsigned long);
  static cp_pvm_t cp_pvm;
  static void cp_resolve(void){
    static const char* const mods[2] = { "kernelbase.dll", "kernel32.dll" };
    for (int i = 0; i < 2 && !cp_pvm; ++i){
      void* h = GetModuleHandleA(mods[i]);
      if (h) cp_pvm = (cp_pvm_t)GetProcAddress(h, "PrefetchVirtualMemory");
    }
  }
#endif

__declspec(dllexport) void gptoss_set_cold_prefetch(int on, int probe_threshold_cycles){
  g_cp_thr = probe_threshold_cycles > 0 ? probe_threshold_cycles : CP_DEFAULT_THRESH;
  for (int i = 0; i < 5; ++i) __atomic_store_n(&g_cp_stat[i], 0, __ATOMIC_RELAXED);
#ifdef _WIN32
  if (on) cp_resolve();
  g_cp = (on && cp_pvm) ? 1 : 0;
#else
  (void)on; g_cp = 0;
#endif
}

/* cfg[0] effective on (0 if PrefetchVirtualMemory could not be resolved or not Windows), cfg[1] threshold ticks,
 * cfg[2] PrefetchVirtualMemory resolved (1 only after the first gptoss_set_cold_prefetch(1, ...)) */
__declspec(dllexport) void gptoss_get_cold_prefetch(int* cfg){
  cfg[0] = g_cp; cfg[1] = g_cp_thr;
#ifdef _WIN32
  cfg[2] = cp_pvm ? 1 : 0;
#else
  cfg[2] = 0;
#endif
}

/* out5[0] calls, [1] experts probed, [2] experts flagged cold, [3] PrefetchVirtualMemory calls, [4] probe TSC ticks
 * (the probe loop, PVM call time excluded). Counters run only while the knob is on. */
__declspec(dllexport) void gptoss_cold_prefetch_stats(long long* out5){
  for (int i = 0; i < 5; ++i) out5[i] = __atomic_load_n(&g_cp_stat[i], __ATOMIC_RELAXED);
}

#ifdef _WIN32
/* One timed 1-byte read: lfence; rdtsc; lfence; load; lfence; rdtsc; lfence. The lfences keep the load out of the
 * timestamps' reorder window (each lfence waits for prior instructions, a load included, to complete). */
static inline unsigned long long cp_timed_read(const uint8_t* p){
  _mm_lfence();
  const unsigned long long t0 = __rdtsc();
  _mm_lfence();
  const unsigned v = *(const volatile uint8_t*)p;
  __asm__ __volatile__("" :: "r"(v));
  _mm_lfence();
  const unsigned long long t1 = __rdtsc();
  _mm_lfence();
  return t1 - t0;
}

/* One PrefetchVirtualMemory call for the ranges gathered so far; returns the TSC ticks it took. A pure hint: the result
 * is ignored (a failed or partial prefetch just leaves the pages to fault in the ordinary way). */
static unsigned long long cp_flush(cp_range_t* r, int* nr, long long* pvm_calls){
  if (*nr <= 0) return 0;
  const unsigned long long t0 = __rdtsc();
  cp_pvm(GetCurrentProcess(), (size_t)*nr, r, 0);
  const unsigned long long t1 = __rdtsc();
  ++*pvm_calls; *nr = 0;
  return t1 - t0;
}

/* Probe the E slots (single thread, before the parallel region); prefetch the ones whose probe read was slow. */
static __attribute__((noinline)) void cold_probe(const int E, const uint8_t** slots, const size_t sr, const size_t slot_bytes){
  const unsigned long long thr = (unsigned long long)g_cp_thr;
  const size_t off[4] = { (size_t)1000 * RB,                                   /* gate codes, row 1000 */
                          (size_t)(H + 1000) * RB,                             /* up codes, row H + 1000 */
                          (size_t)OFF_DC + (size_t)1000 * RB,                  /* down codes, row 1000 */
                          (size_t)OFF_GS + (size_t)(H + 1000) * sr + sr / 2 }; /* a gate/up scale byte */
  cp_range_t r[CP_MAXR]; int nr = 0;
  long long probed = 0, flagged = 0, pvm_calls = 0;
  unsigned long long pvm_ticks = 0;
  const unsigned long long c0 = __rdtsc();
  for (int e = 0; e < E; ++e){
    const uint8_t* sl = slots[e];
    if (!sl) continue;
    ++probed;
    int cold = 0;
    for (int k = 0; k < 4 && !cold; ++k) cold = cp_timed_read(sl + off[k]) > thr;   /* stop at the first slow read */
    if (!cold) continue;
    ++flagged;
    const uintptr_t a = (uintptr_t)sl & ~(uintptr_t)4095;                            /* whole slot, 4 KiB pages */
    const uintptr_t b = ((uintptr_t)sl + slot_bytes + 4095) & ~(uintptr_t)4095;
    if (nr == CP_MAXR) pvm_ticks += cp_flush(r, &nr, &pvm_calls);
    r[nr].addr = (void*)a; r[nr].bytes = (size_t)(b - a); ++nr;
  }
  pvm_ticks += cp_flush(r, &nr, &pvm_calls);                                          /* one call for all flagged experts */
  const unsigned long long c1 = __rdtsc();
  __atomic_fetch_add(&g_cp_stat[0], 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_cp_stat[1], probed, __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_cp_stat[2], flagged, __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_cp_stat[3], pvm_calls, __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_cp_stat[4], (long long)((c1 - c0) - pvm_ticks), __ATOMIC_RELAXED);
}
#else
static inline void cold_probe(const int E, const uint8_t** slots, const size_t sr, const size_t slot_bytes){
  (void)E; (void)slots; (void)sr; (void)slot_bytes;
}
#endif

static inline void pin_thread(void){
  if (!g_aff || t_aff_gen == g_aff_gen) return;
  t_aff_gen = g_aff_gen;
  const int tid = omp_get_thread_num();
  if (tid < 0) return;
  const int cpu = (tid * g_aff) % 64;
#ifdef _WIN32
  SetThreadAffinityMask(GetCurrentThread(), 1ull << cpu);
#else
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

/* Test hook: run a `threads`-wide team from the CALLING thread, pin exactly as the kernel would,
 * and report each member's resulting affinity mask (Windows reads it back by set-and-restore, since
 * kernel32 has no getter; other platforms report 0). Lets a test prove that a team started by a
 * second OS thread (an HTTP handler) is pinned too, which the old per-team-id table missed.
 * `masks` must hold `threads` entries and be zeroed by the caller: libgomp may hand out fewer threads
 * than requested, and those entries are then left untouched. Note that only this decode kernel pins;
 * the prefill kernel in gptoss_cpu_multi.c (which #includes this file) never calls pin_thread(). */
__declspec(dllexport) void gptoss_affinity_probe(int threads, unsigned long long* masks){
  #pragma omp parallel num_threads(threads)
  {
    pin_thread();
    const int tid = omp_get_thread_num();
#ifdef _WIN32
    unsigned long long pm = 0, sm = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm);
    const unsigned long long prev = SetThreadAffinityMask(GetCurrentThread(), pm);
    if (prev) SetThreadAffinityMask(GetCurrentThread(), prev);
    masks[tid] = prev;
#else
    masks[tid] = 0;
#endif
  }
}

/* sr = scale bytes per row (GK raw, SR_P4 packed); a compile-time constant at every call site. */
AINL void cap_row(uint8_t* d, const uint8_t* s, uint8_t* ds, const uint8_t* ss, const int sr){
  for (int k = 0; k < RB; k += 32)
    _mm256_stream_si256((__m256i*)(d + k), _mm256_loadu_si256((const __m256i*)(s + k)));
  memcpy(ds, ss, sr);
}

/* Scales of groups g and g+1 (g even) of one row as integers. sm = 0: s points at the 90 raw bytes.
 * sm = 1: s points at the 46-byte packed row (base, then nibbles; see the header comment); `base` is s[0],
 * hoisted by the caller. sm is a compile-time constant, so only one branch survives. */
AINL void scale_pair(const uint8_t* s, const int base, const int g, const int sm, int* e0, int* e1){
  if (sm){ const unsigned d = s[1 + (g >> 1)]; *e0 = base + (int)(d & 15u); *e1 = base + (int)(d >> 4); }
  else   { *e0 = s[g]; *e1 = s[g+1]; }
}

/* One row. pf > 0: one prefetch per 64 B of codes, pf bytes ahead (runs past the row end into
 * the next row of the same thread's chunk, which is adjacent in memory). */
AINL float rowdot_pf(const uint8_t* c, const uint8_t* s, const float* xe, const float* xo, const int pf, const int sm){
  const __m512 lut = _mm512_setr_ps(0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f);
  const __m512i m4 = _mm512_set1_epi32(0xF);
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  const int base = sm ? s[0] : 0;
  for (int g = 0; g < GK; g += 2){
    if (pf && !(g & 2)) _mm_prefetch((const char*)(c + g*16 + pf), _MM_HINT_T0);
    __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c + g*16)));
    __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c + g*16 + 16)));
    __m512 b0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(v0,m4),lut), _mm512_loadu_ps(xe + g*16));
    __m512 b1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(v1,m4),lut), _mm512_loadu_ps(xe + g*16 + 16));
    b0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(v0,4),lut), _mm512_loadu_ps(xo + g*16), b0);
    b1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(v1,4),lut), _mm512_loadu_ps(xo + g*16 + 16), b1);
    int e0, e1; scale_pair(s, base, g, sm, &e0, &e1);
    a0 = _mm512_fmadd_ps(b0, _mm512_castsi512_ps(_mm512_set1_epi32(e0 << 23)), a0);
    a1 = _mm512_fmadd_ps(b1, _mm512_castsi512_ps(_mm512_set1_epi32(e1 << 23)), a1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}

/* The shipped kernel's row dot product (kept: gptoss_cpu_multi.c documents its contract against it). */
static inline float rowdot(const uint8_t* c, const uint8_t* s, const float* xe, const float* xo){
  return rowdot_pf(c, s, xe, xo, 0, 0);
}

/* Two rows, interleaved. Row 0 uses A0/A1, row 1 uses B0/B1; each row's sequence of
 * mul / fmadd / fmadd / reduce is exactly rowdot_pf's, so r0 == rowdot(c0,...) and
 * r1 == rowdot(c1,...) bit for bit. Only the memory-level parallelism changes. */
AINL void rowdot2_pf(const uint8_t* c0, const uint8_t* s0, const uint8_t* c1, const uint8_t* s1,
                     const float* xe, const float* xo, float* r0, float* r1, const int pf, const int sm){
  const __m512 lut = _mm512_setr_ps(0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f);
  const __m512i m4 = _mm512_set1_epi32(0xF);
  __m512 A0 = _mm512_setzero_ps(), A1 = _mm512_setzero_ps();
  __m512 B0 = _mm512_setzero_ps(), B1 = _mm512_setzero_ps();
  const int base0 = sm ? s0[0] : 0, base1 = sm ? s1[0] : 0;
  for (int g = 0; g < GK; g += 2){
    if (pf && !(g & 2)){
      _mm_prefetch((const char*)(c0 + g*16 + pf), _MM_HINT_T0);
      _mm_prefetch((const char*)(c1 + g*16 + pf), _MM_HINT_T0);
    }
    const __m512 xe0 = _mm512_loadu_ps(xe + g*16), xe1 = _mm512_loadu_ps(xe + g*16 + 16);
    const __m512 xo0 = _mm512_loadu_ps(xo + g*16), xo1 = _mm512_loadu_ps(xo + g*16 + 16);
    __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c0 + g*16)));
    __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c0 + g*16 + 16)));
    __m512i u0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c1 + g*16)));
    __m512i u1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c1 + g*16 + 16)));
    __m512 p0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(v0,m4),lut), xe0);
    __m512 p1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(v1,m4),lut), xe1);
    __m512 q0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(u0,m4),lut), xe0);
    __m512 q1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(u1,m4),lut), xe1);
    p0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(v0,4),lut), xo0, p0);
    p1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(v1,4),lut), xo1, p1);
    q0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(u0,4),lut), xo0, q0);
    q1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(u1,4),lut), xo1, q1);
    int a, b, c, d;
    scale_pair(s0, base0, g, sm, &a, &b);
    scale_pair(s1, base1, g, sm, &c, &d);
    A0 = _mm512_fmadd_ps(p0, _mm512_castsi512_ps(_mm512_set1_epi32(a << 23)), A0);
    A1 = _mm512_fmadd_ps(p1, _mm512_castsi512_ps(_mm512_set1_epi32(b << 23)), A1);
    B0 = _mm512_fmadd_ps(q0, _mm512_castsi512_ps(_mm512_set1_epi32(c << 23)), B0);
    B1 = _mm512_fmadd_ps(q1, _mm512_castsi512_ps(_mm512_set1_epi32(d << 23)), B1);
  }
  *r0 = _mm512_reduce_add_ps(_mm512_add_ps(A0, A1));
  *r1 = _mm512_reduce_add_ps(_mm512_add_ps(B0, B1));
}

/* E experts of one layer. out[H] = sum_e w[e] * expert_e(x). cap == NULL or cap[e] == NULL: no
 * capture; otherwise the rows of expert e are also streamed into cap[e] (copy-on-compute).
 * scratch must hold E*(GU + H + H + H) + 2*(H/2) floats. */
AINL void experts_impl_m(int E, const uint8_t** slots, const float* x, const float** bgu, const float** bdn,
                         const float* w, float* out, float* scratch, int threads, uint8_t** cap, const int sm){
  const size_t SR = sm ? SR_P4 : GK;                    /* scale bytes per row: a constant in each instantiation */
  const size_t ODS = OFF_GS + (size_t)GU * SR;          /* down-scale region (OFF_DS when raw) */
  float* xe = scratch;            float* xo = xe + H/2;
  float* gu = xo + H/2;                                 /* E*GU */
  float* he = gu + (size_t)E*GU;  float* ho = he + (size_t)E*(H/2);   /* E*H/2 each */
  float* y  = ho + (size_t)E*(H/2);                     /* E*H */
  const int pf = g_pf, pair = g_pair;
  if (__builtin_expect(g_cp, 0)) cold_probe(E, slots, SR, sm ? SLOT_PACKED : SLOT_RAW);   /* opt-in; single thread, before the region */
  for (int i = 0; i < H/2; ++i){ xe[i] = x[2*i]; xo[i] = x[2*i+1]; }
  if (g_fuse){
    if (!sb_inited) sb_init();
    #pragma omp parallel num_threads(threads)
    {
      pin_thread();
      /* phase 1: gate row j + up row H+j of expert e -> h[j] right away (no gu[] round trip, no pass) */
      #pragma omp for schedule(dynamic, 64) nowait
      for (int i = 0; i < E*H; ++i){
        const int e = i / H, j = i % H; const uint8_t* sl = slots[e];
        float rg, ru;
        rowdot2_pf(sl + (size_t)j*RB, sl + OFF_GS + (size_t)j*SR,
                   sl + (size_t)(H + j)*RB, sl + OFF_GS + (size_t)(H + j)*SR, xe, xo, &rg, &ru, pf, sm);
        float* g = gu + (size_t)e*GU;                                     /* same stores/loads as the 4-phase kernel */
        g[j] = rg + bgu[e][j];
        g[H + j] = ru + bgu[e][H + j];
        float gt = fminf(g[j], 7.f), up = fminf(fmaxf(g[H + j], -7.f), 7.f);
        float h = (up + 1.f) * gt / (1.f + expf(-1.702f * gt));
        if (j & 1) ho[(size_t)e*(H/2) + j/2] = h; else he[(size_t)e*(H/2) + j/2] = h;
        if (cap && cap[e]){
          cap_row(cap[e] + (size_t)j*RB, sl + (size_t)j*RB, cap[e] + OFF_GS + (size_t)j*SR, sl + OFF_GS + (size_t)j*SR, SR);
          cap_row(cap[e] + (size_t)(H + j)*RB, sl + (size_t)(H + j)*RB, cap[e] + OFF_GS + (size_t)(H + j)*SR, sl + OFF_GS + (size_t)(H + j)*SR, SR);
        }
      }
      spin_barrier(omp_get_num_threads());
      /* phase 2: down row n of every expert -> out[n] = 0 + sum_e w[e] * (rowdot + bdn), same e order */
      #pragma omp for schedule(dynamic, 32) nowait
      for (int n = 0; n < H; ++n){
        for (int e = 0; e < E; ++e){
          const uint8_t* sl = slots[e];
          y[(size_t)e*H + n] = rowdot_pf(sl + OFF_DC + (size_t)n*RB, sl + ODS + (size_t)n*SR,
                                         he + (size_t)e*(H/2), ho + (size_t)e*(H/2), pf, sm) + bdn[e][n];
          if (cap && cap[e]) cap_row(cap[e] + OFF_DC + (size_t)n*RB, sl + OFF_DC + (size_t)n*RB, cap[e] + ODS + (size_t)n*SR, sl + ODS + (size_t)n*SR, SR);
        }
        float a = 0.f; for (int e = 0; e < E; ++e) a += w[e] * y[(size_t)e*H + n];
        out[n] = a;
      }
      _mm_sfence();
    }
    _mm_sfence();
    return;
  }
  #pragma omp parallel num_threads(threads)
  {
    pin_thread();
    if (pair){
      /* rows 2p, 2p+1 always belong to the same expert (GU is even) */
      #pragma omp for schedule(static)
      for (int p = 0; p < E*GU/2; ++p){
        const int i0 = 2*p, e = i0 / GU, n = i0 % GU; const uint8_t* sl = slots[e];
        float r0, r1;
        rowdot2_pf(sl + (size_t)n*RB, sl + OFF_GS + (size_t)n*SR,
                   sl + (size_t)(n+1)*RB, sl + OFF_GS + (size_t)(n+1)*SR, xe, xo, &r0, &r1, pf, sm);
        gu[i0] = r0 + bgu[e][n];
        gu[i0+1] = r1 + bgu[e][n+1];
        if (cap && cap[e]){
          cap_row(cap[e] + (size_t)n*RB, sl + (size_t)n*RB, cap[e] + OFF_GS + (size_t)n*SR, sl + OFF_GS + (size_t)n*SR, SR);
          cap_row(cap[e] + (size_t)(n+1)*RB, sl + (size_t)(n+1)*RB, cap[e] + OFF_GS + (size_t)(n+1)*SR, sl + OFF_GS + (size_t)(n+1)*SR, SR);
        }
      }
    } else {
      #pragma omp for schedule(static)
      for (int i = 0; i < E*GU; ++i){
        const int e = i / GU, n = i % GU; const uint8_t* sl = slots[e];
        gu[i] = rowdot_pf(sl + (size_t)n*RB, sl + OFF_GS + (size_t)n*SR, xe, xo, pf, sm) + bgu[e][n];
        if (cap && cap[e]) cap_row(cap[e] + (size_t)n*RB, sl + (size_t)n*RB, cap[e] + OFF_GS + (size_t)n*SR, sl + OFF_GS + (size_t)n*SR, SR);
      }
    }
    #pragma omp for schedule(static)
    for (int i = 0; i < E*H; ++i){
      const int e = i / H, j = i % H; const float* g = gu + (size_t)e*GU;
      float gt = fminf(g[j], 7.f), up = fminf(fmaxf(g[H + j], -7.f), 7.f);
      float h = (up + 1.f) * gt / (1.f + expf(-1.702f * gt));
      if (j & 1) ho[(size_t)e*(H/2) + j/2] = h; else he[(size_t)e*(H/2) + j/2] = h;
    }
    if (pair){
      #pragma omp for schedule(static)
      for (int p = 0; p < E*H/2; ++p){
        const int i0 = 2*p, e = i0 / H, n = i0 % H; const uint8_t* sl = slots[e];
        const float* hE = he + (size_t)e*(H/2); const float* hO = ho + (size_t)e*(H/2);
        float r0, r1;
        rowdot2_pf(sl + OFF_DC + (size_t)n*RB, sl + ODS + (size_t)n*SR,
                   sl + OFF_DC + (size_t)(n+1)*RB, sl + ODS + (size_t)(n+1)*SR, hE, hO, &r0, &r1, pf, sm);
        y[i0] = r0 + bdn[e][n];
        y[i0+1] = r1 + bdn[e][n+1];
        if (cap && cap[e]){
          cap_row(cap[e] + OFF_DC + (size_t)n*RB, sl + OFF_DC + (size_t)n*RB, cap[e] + ODS + (size_t)n*SR, sl + ODS + (size_t)n*SR, SR);
          cap_row(cap[e] + OFF_DC + (size_t)(n+1)*RB, sl + OFF_DC + (size_t)(n+1)*RB, cap[e] + ODS + (size_t)(n+1)*SR, sl + ODS + (size_t)(n+1)*SR, SR);
        }
      }
    } else {
      #pragma omp for schedule(static)
      for (int i = 0; i < E*H; ++i){
        const int e = i / H, n = i % H; const uint8_t* sl = slots[e];
        y[i] = rowdot_pf(sl + OFF_DC + (size_t)n*RB, sl + ODS + (size_t)n*SR,
                         he + (size_t)e*(H/2), ho + (size_t)e*(H/2), pf, sm) + bdn[e][n];
        if (cap && cap[e]) cap_row(cap[e] + OFF_DC + (size_t)n*RB, sl + OFF_DC + (size_t)n*RB, cap[e] + ODS + (size_t)n*SR, sl + ODS + (size_t)n*SR, SR);
      }
    }
    _mm_sfence();   /* each thread orders its own streaming stores before the barrier */
    #pragma omp for schedule(static)
    for (int n = 0; n < H; ++n){
      float a = 0.f; for (int e = 0; e < E; ++e) a += w[e] * y[(size_t)e*H + n];
      out[n] = a;
    }
  }
  _mm_sfence();
}

/* The layout is read once per call; each branch is a separate whole-function instantiation. */
static void experts_impl(int E, const uint8_t** slots, const float* x, const float** bgu, const float** bdn,
                         const float* w, float* out, float* scratch, int threads, uint8_t** cap){
  if (g_sl == 1) experts_impl_m(E, slots, x, bgu, bdn, w, out, scratch, threads, cap, 1);
  else           experts_impl_m(E, slots, x, bgu, bdn, w, out, scratch, threads, cap, 0);
}

__declspec(dllexport) void gptoss_experts(int E, const uint8_t** slots, const float* x,
                                          const float** bgu, const float** bdn, const float* w,
                                          float* out, float* scratch, int threads){
  experts_impl(E, slots, x, bgu, bdn, w, out, scratch, threads, (uint8_t**)0);
}

__declspec(dllexport) void gptoss_experts_cap(int E, const uint8_t** slots, const float* x,
                                          const float** bgu, const float** bdn, const float* w,
                                          float* out, float* scratch, int threads, uint8_t** cap){
  experts_impl(E, slots, x, bgu, bdn, w, out, scratch, threads, cap);
}
