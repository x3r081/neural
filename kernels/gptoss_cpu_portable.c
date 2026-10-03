/*
 * Portable scalar GPT-OSS MXFP4 CPU reference kernel.
 *
 * This is a deliberately conservative compatibility fallback. It consumes the
 * Neural expert slot bytes unchanged (raw E8M0 scales or exact PS4 packed
 * scales), preserves the reference's per-lane FMA accumulation and reduction
 * tree, and does not create a persistent worker pool or advertise one.
 *
 * Build with tools/build_portable_cpu.py. The default build explicitly targets
 * AVX2+FMA (not AVX-512 and not -march=native); --scalar retains the scalar
 * reference row-dot path.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

#define H 2880
#define GU 5760
#define GK 90
#define RB 1440
#define OFF_DC ((size_t)GU * RB)
#define OFF_GS (OFF_DC + (size_t)H * RB)
#define SR_P4 (1 + GK / 2)
#define SLOT_RAW (OFF_GS + (size_t)(GU + H) * GK)
#define SLOT_PACKED (OFF_GS + (size_t)(GU + H) * SR_P4)

static int g_scale_layout = 0;
static int g_prefetch = 0, g_pair = 0, g_affinity = 0, g_fuse = 0;
static int g_rows_per_tile = 16, g_tokens_per_block = 8;
static int g_dynamic = 1, g_rows_per_item = 64;

EXPORT int gptoss_set_scale_layout(int mode) {
    if (mode != 0 && mode != 1) return -1;
    g_scale_layout = mode;
    return 0;
}
EXPORT int gptoss_get_scale_layout(void) { return g_scale_layout; }
EXPORT long long gptoss_slot_bytes(int mode) {
    return mode == 0 ? (long long)SLOT_RAW : mode == 1 ? (long long)SLOT_PACKED : -1LL;
}
EXPORT void gptoss_set_tuning(int prefetch_bytes, int pair_rows, int affinity_stride) {
    g_prefetch = prefetch_bytes > 0 ? prefetch_bytes : 0;
    g_pair = pair_rows != 0;
    g_affinity = affinity_stride > 0 ? affinity_stride : 0;
}
EXPORT void gptoss_get_tuning(int *cfg) {
    cfg[0] = g_prefetch; cfg[1] = g_pair; cfg[2] = g_affinity; cfg[3] = g_fuse;
}
EXPORT void gptoss_set_fuse(int fuse) { g_fuse = fuse != 0; }

/* Cold-page prefetch is unsupported here; preserve the query ABI and report OFF. */
EXPORT void gptoss_set_cold_prefetch(int on, int threshold) { (void)on; (void)threshold; }
EXPORT void gptoss_get_cold_prefetch(int *cfg) { cfg[0] = 0; cfg[1] = 100000; cfg[2] = 0; }
EXPORT void gptoss_cold_prefetch_stats(long long *out) {
    for (int i = 0; i < 5; ++i) out[i] = 0;
}

static inline float lut4(unsigned v) {
    static const float lut[16] = {
        0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
    };
    return lut[v & 15u];
}

static inline unsigned scale_exp(const uint8_t *s, int group, int packed, int base) {
    if (!packed) return s[group];
    const uint8_t b = s[1 + group / 2];
    const unsigned d = (group & 1) ? (b >> 4) : (b & 15u);
    return (unsigned)base + d;
}

/* GCC's `_mm512_reduce_add_ps` reduction tree, expressed scalar: first join
 * corresponding low/high 256-bit halves, then 128-bit halves, then the two
 * cross-pairs. Keep each add as an FP32 operation (build disables contraction). */
static inline float reduce16(const float v[16]) {
    float t[8], u[4];
    for (int i = 0; i < 8; ++i) t[i] = v[i] + v[i + 8];
    for (int i = 0; i < 4; ++i) u[i] = t[i] + t[i + 4];
    const float a = u[0] + u[2];
    const float b = u[1] + u[3];
    return a + b;
}

/* Scalar compatibility row dot retained for baseline/scalar builds. */
static float rowdot_scalar(const uint8_t *codes, const uint8_t *scales,
                           const float *xe, const float *xo, int packed) {
    float a0[16] = {0}, a1[16] = {0};
    const int base = packed ? scales[0] : 0;
    for (int g = 0; g < GK; g += 2) {
        const unsigned e0 = scale_exp(scales, g, packed, base);
        const unsigned e1 = scale_exp(scales, g + 1, packed, base);
        const uint32_t b0 = (uint32_t)e0 << 23;
        const uint32_t b1 = (uint32_t)e1 << 23;
        float s0, s1;
        memcpy(&s0, &b0, sizeof(s0)); memcpy(&s1, &b1, sizeof(s1));
        for (int lane = 0; lane < 16; ++lane) {
            const unsigned c0 = codes[(size_t)g * 16 + lane];
            const unsigned c1 = codes[(size_t)(g + 1) * 16 + lane];
            const size_t i0 = (size_t)g * 16 + lane;
            const size_t i1 = (size_t)(g + 1) * 16 + lane;
            const float p0 = lut4(c0) * xe[i0];
            const float p1 = lut4(c1) * xe[i1];
            const float q0 = fmaf(lut4(c0 >> 4), xo[i0], p0);
            const float q1 = fmaf(lut4(c1 >> 4), xo[i1], p1);
            a0[lane] = fmaf(q0, s0, a0[lane]);
            a1[lane] = fmaf(q1, s1, a1[lane]);
        }
    }
    float sum[16];
    for (int i = 0; i < 16; ++i) sum[i] = a0[i] + a1[i];
    return reduce16(sum);
}

#if defined(__AVX2__) && defined(__FMA__)
/* Map an 8-lane vector of FP4 nibbles through the positive half of lut4.
 * XORing the sign bit handles negative zero as well as signed nonzero values. */
static inline __m256 lut4x8(__m256i code) {
    const __m256 table = _mm256_setr_ps(0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f);
    const __m256i index = _mm256_and_si256(code, _mm256_set1_epi32(7));
    const __m256i sign = _mm256_slli_epi32(
        _mm256_and_si256(code, _mm256_set1_epi32(8)), 28);
    return _mm256_xor_ps(_mm256_permutevar8x32_ps(table, index), _mm256_castsi256_ps(sign));
}

static inline void rowdot_group8(const uint8_t *code, const float *xe,
                                const float *xo, __m256 scale,
                                __m256 *acc_lo, __m256 *acc_hi) {
    const __m128i bytes = _mm_loadu_si128((const __m128i *)code);
    const __m128i low = _mm_and_si128(bytes, _mm_set1_epi8(0x0f));
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0f));
    const __m256i low_lo = _mm256_cvtepu8_epi32(low);
    const __m256i low_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(low, 8));
    const __m256i high_lo = _mm256_cvtepu8_epi32(high);
    const __m256i high_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(high, 8));
    const __m256 even_lo = _mm256_loadu_ps(xe);
    const __m256 even_hi = _mm256_loadu_ps(xe + 8);
    const __m256 odd_lo = _mm256_loadu_ps(xo);
    const __m256 odd_hi = _mm256_loadu_ps(xo + 8);
    const __m256 p_lo = _mm256_mul_ps(lut4x8(low_lo), even_lo);
    const __m256 p_hi = _mm256_mul_ps(lut4x8(low_hi), even_hi);
    const __m256 q_lo = _mm256_fmadd_ps(lut4x8(high_lo), odd_lo, p_lo);
    const __m256 q_hi = _mm256_fmadd_ps(lut4x8(high_hi), odd_hi, p_hi);
    *acc_lo = _mm256_fmadd_ps(q_lo, scale, *acc_lo);
    *acc_hi = _mm256_fmadd_ps(q_hi, scale, *acc_hi);
}

/* Two 8-lane halves reproduce the native 16 independent accumulators. The
 * g/g+1 pair feeds a0/a1 in the same order; each FMA is explicit. */
static float rowdot_avx2(const uint8_t *codes, const uint8_t *scales,
                         const float *xe, const float *xo, int packed) {
    __m256 a0_lo = _mm256_setzero_ps(), a0_hi = _mm256_setzero_ps();
    __m256 a1_lo = _mm256_setzero_ps(), a1_hi = _mm256_setzero_ps();
    const int base = packed ? scales[0] : 0;
    for (int g = 0; g < GK; g += 2) {
        const unsigned e0 = scale_exp(scales, g, packed, base);
        const unsigned e1 = scale_exp(scales, g + 1, packed, base);
        const uint32_t b0 = (uint32_t)e0 << 23;
        const uint32_t b1 = (uint32_t)e1 << 23;
        float s0, s1;
        memcpy(&s0, &b0, sizeof(s0)); memcpy(&s1, &b1, sizeof(s1));
        const size_t offset = (size_t)g * 16;
        rowdot_group8(codes + offset, xe + offset, xo + offset,
                      _mm256_set1_ps(s0), &a0_lo, &a0_hi);
        rowdot_group8(codes + offset + 16, xe + offset + 16, xo + offset + 16,
                      _mm256_set1_ps(s1), &a1_lo, &a1_hi);
    }
    const __m256 sum_lo = _mm256_add_ps(a0_lo, a1_lo);
    const __m256 sum_hi = _mm256_add_ps(a0_hi, a1_hi);
    float sum[16];
    _mm256_storeu_ps(sum, sum_lo);
    _mm256_storeu_ps(sum + 8, sum_hi);
    return reduce16(sum);
}
#endif

static inline float rowdot(const uint8_t *codes, const uint8_t *scales,
                           const float *xe, const float *xo, int packed) {
#if defined(__AVX2__) && defined(__FMA__)
    return rowdot_avx2(codes, scales, xe, xo, packed);
#else
    return rowdot_scalar(codes, scales, xe, xo, packed);
#endif
}

static inline void activation(float g, float u, float *h) {
    const float gt = fminf(g, 7.f);
    const float up = fminf(fmaxf(u, -7.f), 7.f);
    *h = (up + 1.f) * gt / (1.f + expf(-1.702f * gt));
}

static inline void copy_row(uint8_t *dst, const uint8_t *src, size_t n) {
    if (dst != NULL) memcpy(dst, src, n);
}

static inline float mul_rn(float a, float b) {
    volatile float p = a * b;
    return p;
}

/* Layout matches native experts_impl_m scratch: xe[H/2], xo[H/2],
 * gu[E*GU], he[E*H/2], ho[E*H/2], y[E*H]. */
static void experts_impl(int E, const uint8_t **slots, const float *x,
                         const float **bgu, const float **bdn, const float *w,
                         float *out, float *scratch, uint8_t **cap, int threads) {
    if (E <= 0) return;
    const int packed = g_scale_layout == 1;
    const size_t sr = packed ? SR_P4 : GK;
    const size_t ods = OFF_GS + (size_t)GU * sr;
    const size_t slot_bytes = packed ? SLOT_PACKED : SLOT_RAW;
    float *xe = scratch, *xo = xe + H/2;
    float *gu = xo + H/2;
    float *he = gu + (size_t)E * GU;
    float *ho = he + (size_t)E * (H/2);
    float *y = ho + (size_t)E * (H/2);
    for (int i = 0; i < H/2; ++i) { xe[i] = x[2*i]; xo[i] = x[2*i+1]; }

#if defined(_OPENMP)
    if (threads < 1) threads = 1;
#else
    (void)threads;
#endif
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static)
#endif
    for (int row = 0; row < E * GU; ++row) {
        const int e = row / GU, j = row % GU;
        const uint8_t *slot = slots[e];
        const uint8_t *sc = slot + OFF_GS + (size_t)j * sr;
        gu[(size_t)e * GU + j] = rowdot(slot + (size_t)j * RB, sc, xe, xo, packed) + bgu[e][j];
        if (cap != NULL && cap[e] != NULL) {
            copy_row(cap[e] + (size_t)j * RB, slot + (size_t)j * RB, RB);
            copy_row(cap[e] + OFF_GS + (size_t)j * sr, sc, sr);
        }
    }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int row = 0; row < E * H; ++row) {
        const int e = row / H, j = row % H;
        const float *g = gu + (size_t)e * GU;
        float h;
        activation(g[j], g[H + j], &h);
        if (j & 1) ho[(size_t)e * (H/2) + j/2] = h;
        else       he[(size_t)e * (H/2) + j/2] = h;
    }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int row = 0; row < E * H; ++row) {
        const int e = row / H, n = row % H;
        const uint8_t *slot = slots[e];
        const uint8_t *sc = slot + ods + (size_t)n * sr;
        y[(size_t)e * H + n] = rowdot(slot + OFF_DC + (size_t)n * RB, sc,
                                      he + (size_t)e*(H/2), ho + (size_t)e*(H/2), packed)
                                + bdn[e][n];
        if (cap != NULL && cap[e] != NULL) {
            copy_row(cap[e] + OFF_DC + (size_t)n * RB,
                     slot + OFF_DC + (size_t)n * RB, RB);
            copy_row(cap[e] + ods + (size_t)n * sr, sc, sr);
        }
    }
#if defined(_OPENMP)
    }
#endif
    /* The target decode path has E=4. This is the source expression from the
     * reference; exact target validation gates whether the DLL is selectable. */
    for (int n = 0; n < H; ++n) {
        /* The reference compiler emits rounded products in 8/4-wide SLP
         * groups and a fused scalar tail; reproduce this independent of target ISA. */
        const int unfused = (E / 8) * 8 + ((E % 8) >= 4 ? 4 : 0);
        float a = 0.f;
        for (int e = 0; e < E; ++e) {
            const float v = y[(size_t)e * H + n];
            if (e < unfused) a = a + mul_rn(w[e], v);
            else a = fmaf(w[e], v, a);
        }
        out[n] = a;
    }
    (void)slot_bytes;
}

EXPORT void gptoss_experts(int E, const uint8_t **slots, const float *x,
                           const float **bgu, const float **bdn, const float *w,
                           float *out, float *scratch, int threads) {
    experts_impl(E, slots, x, bgu, bdn, w, out, scratch, NULL, threads);
}
EXPORT void gptoss_experts_cap(int E, const uint8_t **slots, const float *x,
                               const float **bgu, const float **bdn, const float *w,
                               float *out, float *scratch, int threads, uint8_t **cap) {
    experts_impl(E, slots, x, bgu, bdn, w, out, scratch, cap, threads);
}

EXPORT int gptoss_multi_scratch_floats(int E, int P) {
    (void)E; if (P < 0) P = 0;
    const long long n = (long long)P * (H + GU) + 16;
    return n > 2147483647LL ? -1 : (int)n;
}

/* Correct but intentionally simple fallback: one expert-token pair at a time.
 * Scratch is the public P*(H+GU)+16 allocation; first H floats hold temporary
 * input/activation and the next GU hold gate/up rows. */
EXPORT void gptoss_experts_multi(int E, const uint8_t **slots, const int *off,
                                 const float *X, const float **bgu, const float **bdn,
                                 const float *W, float *Y, float *scratch, int threads) {
    if (E <= 0) return;
    const int P = off[E];
    if (P <= 0) return;
    const int packed = g_scale_layout == 1;
    const size_t sr = packed ? SR_P4 : GK;
    const size_t ods = OFF_GS + (size_t)GU * sr;
#if defined(_OPENMP)
    if (threads < 1) threads = 1;
#pragma omp parallel for schedule(static) num_threads(threads)
#else
    (void)threads;
#endif
    for (int e = 0; e < E; ++e) {
        const uint8_t *slot = slots[e];
        for (int p = off[e]; p < off[e+1]; ++p) {
            float *tmp = scratch + (size_t)p * (H + GU);
            float *g = tmp + H;
            const float *x = X + (size_t)p * H;
            float *xe = tmp, *xo = tmp + H/2;
            for (int i = 0; i < H/2; ++i) { xe[i] = x[2*i]; xo[i] = x[2*i+1]; }
            for (int j = 0; j < GU; ++j)
                g[j] = rowdot(slot + (size_t)j*RB, slot + OFF_GS + (size_t)j*sr,
                              xe, xo, packed) + bgu[e][j];
            /* Activation overwrites even/odd input positions only after all
             * gate/up dots have consumed x. It is stored as [even | odd]. */
            for (int j = 0; j < H; ++j) {
                float h; activation(g[j], g[H+j], &h);
                if (j & 1) tmp[H/2 + j/2] = h;
                else       tmp[j/2] = h;
            }
            for (int n = 0; n < H; ++n) {
                const float r = rowdot(slot + OFF_DC + (size_t)n*RB,
                                       slot + ods + (size_t)n*sr, tmp, tmp + H/2, packed);
                float a = 0.f;
                a += W[p] * (r + bdn[e][n]);
                Y[(size_t)p * H + n] = a;
            }
        }
    }
}

EXPORT void gptoss_combine_pairs(int N, int K, const int *idx, const float *w,
                                 const float *Yr, float *out, int threads) {
    (void)threads;
    for (int t = 0; t < N; ++t) {
        const int *ix = idx + (size_t)t*K;
        const float *wt = w + (size_t)t*K;
        int E = 0;
        for (int k = 0; k < K; ++k) E += ix[k] >= 0;
        const int unfused = (E / 8) * 8 + ((E % 8) >= 4 ? 4 : 0);
        for (int n = 0; n < H; ++n) {
            float a = 0.f; int j = 0;
            for (int k = 0; k < K; ++k) if (ix[k] >= 0) {
                const float y = Yr[(size_t)ix[k]*H+n];
                if (j < unfused) a = a + mul_rn(wt[k], y);
                else a = fmaf(wt[k], y, a);
                ++j;
            }
            out[(size_t)t*H+n] = a;
        }
    }
}

/* This scalar implementation has fixed iteration order, so scheduling controls
 * are ABI-compatible metadata knobs rather than performance controls. */
EXPORT void gptoss_multi_set_tiling(int rows_per_tile, int tokens_per_block) {
    if (rows_per_tile > 0) g_rows_per_tile = rows_per_tile < GU ? rows_per_tile : GU;
    if (tokens_per_block >= 1 && tokens_per_block <= 8) g_tokens_per_block = tokens_per_block;
}
EXPORT void gptoss_multi_set_schedule(int dynamic, int rows_per_item) {
    g_dynamic = dynamic != 0;
    if (rows_per_item > 0) g_rows_per_item = rows_per_item < GU ? rows_per_item : GU;
}
EXPORT void gptoss_multi_get_config(int *cfg) {
    cfg[0] = g_rows_per_tile; cfg[1] = g_tokens_per_block;
    cfg[2] = g_dynamic; cfg[3] = g_rows_per_item;
}

EXPORT void gptoss_multi_activation(int P, const float *GUin, float *Hout, int vec) {
    (void)vec;
    for (int p = 0; p < P; ++p) {
        const float *g = GUin + (size_t)p * GU;
        float *he = Hout + (size_t)p * H;
        float *ho = he + H/2;
        for (int j = 0; j < H; ++j) {
            float h; activation(g[j], g[H+j], &h);
            if (j & 1) ho[j/2] = h; else he[j/2] = h;
        }
    }
}

/* No approximate vector exponential exists in this DLL. The production
 * operation is expf for every lane, so it agrees with the scalar reference by
 * construction; report every checked lane as taking that exact fallback. */
EXPORT long long gptoss_multi_check_exp(long long begin, long long end, int threads,
                                        unsigned *first_bad, long long *n_fallback) {
    (void)threads;
    if (first_bad) *first_bad = 0xFFFFFFFFu;
    if (n_fallback) *n_fallback = end > begin ? end - begin : 0;
    return 0;
}
