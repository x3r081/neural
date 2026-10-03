/* Native BF16 Qwen expert MLP for the Neural Qwen3.6 CPU path.
 *
 * Weights remain in their original uint16 BF16 representation. Callers may
 * pass pointers into read-only mmap-backed tensors. The operation order is:
 *   BF16 gate_up linear -> BF16 round
 *   BF16 SiLU(gate) -> BF16 round
 *   BF16 multiply with up -> BF16 round
 *   BF16 down linear -> BF16 round
 * Routing weights and route summation intentionally stay in Python.
 */
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#  define _WIN32_WINNT 0x0602
#endif

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <x86intrin.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sched.h>
#endif

#if defined(_WIN32)
#  define QWEN_EXPORT __declspec(dllexport)
#else
#  define QWEN_EXPORT __attribute__((visibility("default")))
#endif

static inline float bf16_to_float(uint16_t x) {
  uint32_t bits = ((uint32_t)x) << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

/* IEEE round-to-nearest-even, matching torch.float32 -> torch.bfloat16. */
static inline uint16_t float_to_bf16(float x) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  /* Keep NaNs NaNs (the ordinary rounding bias can erase a low payload). */
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
    return (uint16_t)((bits >> 16) | 0x0040u);
  const uint32_t lsb = (bits >> 16) & 1u;
  return (uint16_t)((bits + 0x7fffu + lsb) >> 16);
}

static inline __m512 load_bf16x16(const uint16_t* p) {
  const __m256i h = _mm256_loadu_si256((const __m256i*)p);
  const __m512i w = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
  return _mm512_castsi512_ps(w);
}

static inline float dot_bf16(const uint16_t* w, const uint16_t* x, int k) {
  __m512 a0 = _mm512_setzero_ps();
  __m512 a1 = _mm512_setzero_ps();
  int j = 0;
  for (; j + 31 < k; j += 32) {
    a0 = _mm512_fmadd_ps(load_bf16x16(w + j), load_bf16x16(x + j), a0);
    a1 = _mm512_fmadd_ps(load_bf16x16(w + j + 16), load_bf16x16(x + j + 16), a1);
  }
  __m512 a = _mm512_add_ps(a0, a1);
  for (; j + 15 < k; j += 16)
    a = _mm512_fmadd_ps(load_bf16x16(w + j), load_bf16x16(x + j), a);
  float sum = _mm512_reduce_add_ps(a);
  for (; j < k; ++j) sum += bf16_to_float(w[j]) * bf16_to_float(x[j]);
  return sum;
}

/* Per-call barrier state lives on the caller's stack: calls are reentrant and
 * independent OpenMP teams cannot corrupt each other's phase generations. */
typedef struct {
  volatile uint32_t arrived;
  volatile uint32_t generation;
} qwen_phase_barrier;

static inline void qwen_phase_wait(qwen_phase_barrier* barrier,
                                   uint32_t team_size) {
  const uint32_t generation =
      __atomic_load_n(&barrier->generation, __ATOMIC_ACQUIRE);
  if (__atomic_add_fetch(&barrier->arrived, 1u, __ATOMIC_ACQ_REL) == team_size) {
    __atomic_store_n(&barrier->arrived, 0u, __ATOMIC_RELAXED);
    __atomic_add_fetch(&barrier->generation, 1u, __ATOMIC_RELEASE);
    return;
  }

  const unsigned long long start = __rdtsc();
  while (__atomic_load_n(&barrier->generation, __ATOMIC_ACQUIRE) == generation) {
    const unsigned long long elapsed = __rdtsc() - start;
    if (elapsed < 150000ULL) {
      _mm_pause();
    } else if (elapsed < 3000000ULL) {
#if defined(_WIN32)
      SwitchToThread();
#else
      sched_yield();
#endif
    } else {
#if defined(_WIN32)
      /* After a long delay, actually sleep instead of burning cores if a
       * teammate is descheduled or blocked by a file-backed page fault. The
       * bounded sleep limits added tail latency once the phase completes. */
      Sleep(1);
#else
      sched_yield();
#endif
    }
  }
}

/*
 * Inputs:
 *   x:                 [tokens, hidden] BF16
 *   gate_up_experts:   [routes] pointers to [2*intermediate, hidden] BF16
 *   down_experts:      [routes] pointers to [hidden, intermediate] BF16
 * Outputs:
 *   y:                 [tokens, routes, hidden] BF16, route weights unapplied
 *
 * gu_tmp and act_tmp are caller-owned scratch arrays. One parallel iteration
 * computes one intermediate coordinate; the rounded BF16 gate/up values are
 * materialized before the same SiLU/multiply boundaries as the reference.
 */
QWEN_EXPORT int qwen_bf16_experts(
    const uint16_t* x,
    const uint16_t* const* gate_up_experts,
    const uint16_t* const* down_experts,
    uint16_t* y,
    uint16_t* gu_tmp,
    uint16_t* act_tmp,
    int tokens,
    int routes,
    int hidden,
    int intermediate,
    int threads) {
  if (!x || !gate_up_experts || !down_experts || !y || !gu_tmp || !act_tmp ||
      tokens <= 0 || routes <= 0 || hidden <= 0 || intermediate <= 0 || threads <= 0)
    return 1;
  for (int r = 0; r < routes; ++r)
    if (!gate_up_experts[r] || !down_experts[r]) return 2;

  const long long nrows_act = (long long)tokens * routes * intermediate;
  const long long nrows_dn = (long long)tokens * routes * hidden;

  qwen_phase_barrier phase = {0u, 0u};
  #pragma omp parallel num_threads(threads)
  {
    const uint32_t team_size = (uint32_t)omp_get_num_threads();

    /* Pair gate/up dots so the activation can immediately consume their BF16
     * results. Dot-product instructions and every BF16 rounding boundary are
     * unchanged; each coordinate is independent of the others. */
    #pragma omp for schedule(dynamic, 32) nowait
    for (long long row = 0; row < nrows_act; ++row) {
      const int n = (int)(row % intermediate);
      const long long tr = row / intermediate;
      const int r = (int)(tr % routes);
      const int t = (int)(tr / routes);
      const uint16_t* inp = x + (long long)t * hidden;
      const uint16_t* weights = gate_up_experts[r];
      const long long gu_row = tr * (2LL * intermediate);
      const uint16_t gate_bf16 = float_to_bf16(
          dot_bf16(weights + (long long)n * hidden, inp, hidden));
      const uint16_t up_bf16 = float_to_bf16(
          dot_bf16(weights + (long long)(n + intermediate) * hidden,
                   inp, hidden));
      gu_tmp[gu_row + n] = gate_bf16;
      gu_tmp[gu_row + n + intermediate] = up_bf16;
      const float gate = bf16_to_float(gate_bf16);
      const float up = bf16_to_float(up_bf16);
      /* torch.nn.functional.silu on BF16, then BF16 elementwise multiply. */
      const float sigmoid = 1.0f / (1.0f + expf(-gate));
      const uint16_t gate_silu = float_to_bf16(gate * sigmoid);
      act_tmp[row] = float_to_bf16(bf16_to_float(gate_silu) * up);
    }

    qwen_phase_wait(&phase, team_size);

    #pragma omp for schedule(dynamic, 32) nowait
    for (long long row = 0; row < nrows_dn; ++row) {
      const int n = (int)(row % hidden);
      const long long tr = row / hidden;
      const int r = (int)(tr % routes);
      const uint16_t* inp = act_tmp + tr * intermediate;
      const uint16_t* w = down_experts[r] + (long long)n * intermediate;
      y[row] = float_to_bf16(dot_bf16(w, inp, intermediate));
    }
  }
  return 0;
}
