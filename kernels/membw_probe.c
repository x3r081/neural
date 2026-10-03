/* membw_probe: raw DRAM read-bandwidth probe (measurement tool, no model data involved).
 *
 * Purpose: nobody has measured the host's raw read bandwidth independently of the expert
 * kernel (kernels/gptoss_cpu_cap2.c streams 1440-byte code rows at a MEASURED 24-29 GB/s
 * through a 4 KiB-paged memory-mapped file, with no software prefetch). This probe reads a
 * caller-provided buffer with the SAME access shape (1440 B "rows", 16-byte loads, static
 * contiguous per-thread ranges) and lets the caller vary only: threads, software-prefetch
 * distance, and 1 vs 2 independent sequential streams per thread. The caller (tools/
 * measure_host_dram.py) supplies the memory: 4 KiB private pages, 2 MiB large pages, or a
 * read-only file view, which isolates the page-size / file-mapping effect.
 *
 * Result label: MEASURED (decimal GB = 1e9 B). The buffer must already be touched/committed:
 * page-fault cost is not part of what this probe means to measure (best-of-`reps` hides it).
 * `base` should be at least 16-byte aligned (VirtualAlloc / MapViewOfFile / malloc-large all are).
 *
 * Build (tools/build_kernels.bat flags): gcc -O3 -march=native -fopenmp -shared -o membw_probe.dll kernels\membw_probe.c
 * Linux compile check:                   gcc -O2 -mavx512f -mavx512bw -fopenmp -shared -fPIC -o membw_probe.so kernels/membw_probe.c
 */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
/* __declspec(dllexport) is native under MinGW-w64. High-resolution timer: omp_get_wtime() in
 * MinGW's libgomp ticks in whole milliseconds, which made a 40 ms pass read 52-55 GB/s. */
__declspec(dllimport) int __stdcall QueryPerformanceCounter(long long*);
__declspec(dllimport) int __stdcall QueryPerformanceFrequency(long long*);
static double now_s(void){
  long long c, f;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c / (double)f;
}
#else
#define __declspec(x)   /* Linux compile-check only: symbols are exported by default */
#include <time.h>
static double now_s(void){
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#define RB_DEFAULT 1440                 /* one gpt-oss code row (RB in gptoss_cpu_cap2.c) */
#define INLINE static inline __attribute__((always_inline))

static uint64_t g_checksum;             /* sum of all bytes read in the last pass, mod 2^32 */

/* Four zmm accumulators (cap2's rowdot uses two). Every loaded byte is folded into one, and the
 * folded value is stored to g_checksum, so the compiler cannot elide or narrow the loads. */
typedef struct { __m512i a0, a1, a2, a3; } acc_t;

/* 16-byte load widened u8 -> u32 lanes: the same load shape rowdot() uses (_mm_loadu_si128 +
 * _mm512_cvtepu8_epi32); the widening add is cheap next to DRAM speed at 8+ threads. */
#define LD(q) _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(q)))

/* Software prefetch: issued ONCE PER 64-BYTE LINE, `pd` bytes ahead of the load stream. Rows are
 * 16-byte multiples, so exactly one 16-byte load per line starts 64-byte aligned; prefetching on
 * that load covers each line exactly once even when 1440 B rows start mid-line. `pf` is a
 * compile-time constant at every call site, so the no-prefetch path contains no test at all. */
#define PF(q) do { if (pf && !((uintptr_t)(q) & 63)) \
    _mm_prefetch((const char*)((uintptr_t)(q) + pd), _MM_HINT_T0); } while (0)
#define ST1(q, A) do { PF(q); A = _mm512_add_epi32(A, LD(q)); } while (0)

/* One row, one stream: n16 sixteen-byte loads. */
INLINE void row1(acc_t* s, const uint8_t* a, int n16, size_t pd, int pf){
  int k = 0;
  for (; k + 4 <= n16; k += 4){
    const uint8_t* q = a + 16 * (size_t)k;
    ST1(q, s->a0); ST1(q + 16, s->a1); ST1(q + 32, s->a2); ST1(q + 48, s->a3);
  }
  for (; k < n16; ++k) ST1(a + 16 * (size_t)k, s->a0);
}

/* Two rows from two independent sequential streams, loads interleaved one-for-one so both
 * streams are in flight at once (more memory-level parallelism than one stream). */
INLINE void row2(acc_t* s, const uint8_t* a, const uint8_t* b, int n16, size_t pd, int pf){
  int k = 0;
  for (; k + 4 <= n16; k += 4){
    const uint8_t* p = a + 16 * (size_t)k; const uint8_t* q = b + 16 * (size_t)k;
    ST1(p, s->a0); ST1(q, s->a1); ST1(p + 16, s->a2); ST1(q + 16, s->a3);
    ST1(p + 32, s->a0); ST1(q + 32, s->a1); ST1(p + 48, s->a2); ST1(q + 48, s->a3);
  }
  for (; k < n16; ++k){ ST1(a + 16 * (size_t)k, s->a0); ST1(b + 16 * (size_t)k, s->a1); }
}

/* Rows [r0, r1) of one thread. streams==2: stream A walks the first half of the range, stream B
 * the second half (two far-apart sequential streams); an odd leftover row is read as one stream. */
INLINE void range(acc_t* s, const uint8_t* base, size_t r0, size_t r1, size_t rb, int n16,
                  size_t pd, int pf, int two){
  size_t i, half = two ? (r1 - r0) / 2 : 0;
  for (i = 0; i < half; ++i)
    row2(s, base + (r0 + i) * rb, base + (r0 + half + i) * rb, n16, pd, pf);
  for (i = r0 + 2 * half; i < r1; ++i) row1(s, base + i * rb, n16, pd, pf);
}

/* Specialise on (pf, two) so each variant is compiled with constants and no dead branches. */
static void thread_pass(acc_t* s, const uint8_t* base, size_t r0, size_t r1, size_t rb, int n16,
                        size_t pd, int two){
  if (pd) { if (two) range(s, base, r0, r1, rb, n16, pd, 1, 1); else range(s, base, r0, r1, rb, n16, pd, 1, 0); }
  else    { if (two) range(s, base, r0, r1, rb, n16, 0, 0, 1);  else range(s, base, r0, r1, rb, n16, 0, 0, 0); }
}

/* Read [base, base+nbytes) `reps` times with `threads` OpenMP threads; return the BEST pass in
 * decimal GB/s (bytes actually read = floor(nbytes/row)*row, over QueryPerformanceCounter seconds).
 *   chunk_bytes   row size, rounded down to a multiple of 16 (<=0 -> 1440)
 *   prefetch_dist >0: _MM_HINT_T0 prefetch of (line + prefetch_dist) once per 64 B; 0 = none
 *   streams       1 or 2 (see range())
 * Returns -1.0 on invalid arguments. Timing wraps the parallel region (fork/join included, which
 * is microseconds against a multi-hundred-ms pass). */
__declspec(dllexport) double membw_read(const uint8_t* base, size_t nbytes, int threads,
                                        int chunk_bytes, int prefetch_dist, int streams, int reps){
  size_t rb = (size_t)(chunk_bytes > 0 ? chunk_bytes : RB_DEFAULT) & ~(size_t)15;
  if (!base || threads < 1 || reps < 1 || rb < 16 || (streams != 1 && streams != 2)) return -1.0;
  size_t nrows = nbytes / rb, pd = prefetch_dist > 0 ? (size_t)prefetch_dist : 0;
  if (nrows == 0) return -1.0;
  int n16 = (int)(rb / 16), two = (streams == 2);
  double best = 0.0;
  for (int rep = 0; rep < reps; ++rep){
    uint64_t sum = 0;
    double t0 = now_s();
    #pragma omp parallel num_threads(threads) reduction(+:sum)
    {
      /* same partition as `omp for schedule(static)`: contiguous, sizes differ by <= 1 row */
      size_t nth = (size_t)omp_get_num_threads(), tid = (size_t)omp_get_thread_num();
      size_t q = nrows / nth, rem = nrows % nth;
      size_t r0 = tid * q + (tid < rem ? tid : rem), r1 = r0 + q + (tid < rem ? 1 : 0);
      acc_t s = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
      thread_pass(&s, base, r0, r1, rb, n16, pd, two);
      __m512i t = _mm512_add_epi32(_mm512_add_epi32(s.a0, s.a1), _mm512_add_epi32(s.a2, s.a3));
      sum += (uint32_t)_mm512_reduce_add_epi32(t);
    }
    double dt = now_s() - t0;
    g_checksum = sum & 0xFFFFFFFFu;     /* partition-independent: sum of bytes mod 2^32 */
    if (dt > 0.0){ double bw = (double)(nrows * rb) / dt / 1e9; if (bw > best) best = bw; }
  }
  return best;
}

/* Sum (mod 2^32) of every byte read in the most recent pass. Identical for every
 * (threads, prefetch, streams) setting on the same buffer, so it doubles as a "same bytes
 * were read" check across configurations. */
__declspec(dllexport) uint64_t membw_checksum(void){ return g_checksum; }
