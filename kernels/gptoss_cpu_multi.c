/* gpt-oss-120B MXFP4 experts on the CPU, MULTI-TOKEN (prompt processing).
 *
 * Build (w64devkit gcc 16, same flags as gptoss_cpu_cap2.dll):
 *   gcc -O3 -march=native -fopenmp -shared -o gptoss_cpu_multi.dll kernels/gptoss_cpu_multi.c
 *   (add -DMULTI_PROFILE for per-phase timers in the exported gptoss_multi_prof[])
 *
 * This file #includes gptoss_cpu_cap2.c UNCHANGED, so gptoss_cpu_multi.dll is a strict
 * superset of gptoss_cpu_cap2.dll: it also exports gptoss_experts / gptoss_experts_cap
 * compiled from the very same source. A process that loads only this DLL therefore has a
 * single OpenMP (libgomp, statically linked) thread pool instead of two spinning ones.
 *
 * During prompt processing one expert is routed several tokens (c) at once. Calling the
 * 1-token kernel c times streams the 13.2 MB expert c times from DRAM. Here each
 * 16-byte code block is dequantized (vpermps LUT, lo/hi nibble) and its E8M0 scale
 * broadcast ONCE per (row, token block) and reused for up to TB tokens, each token with its
 * own a0/a1 accumulators. Rows are processed in small tiles (16 rows) that all token blocks
 * sweep, so each row comes from DRAM once and each token block's input stays in L2.
 * Work distribution: dynamic items of 64 rows of one expert (default) or a cost-weighted static
 * contiguous split (gptoss_multi_set_schedule(0, ..)).
 *   DELIBERATE DEVIATION FROM THE SPEC ("OpenMP static schedule over rows"): the default is the
 *   dynamic split because the prefill CPU kernel shares the 8 cores with the server's Python /
 *   CUDA-launch threads; a static split waits for the slowest (preempted) thread. Indicative
 *   medians (8 threads, benchmark server running): dynamic 0.97 vs static 1.69 ms/expert at
 *   c=8, 7.01 vs 8.11 at c=64. Both splits are bit-identical (tested); the spec's static
 *   schedule stays one call away.
 * The activation is a vectorized, exhaustively verified bit-identical form of cap2's scalar
 * loop (see below).
 *
 * SCALE LAYOUT SWITCH: this file shares cap2's process-global g_sl (gptoss_set_scale_layout, exported from
 * this DLL too: a process must set it on EVERY kernel DLL it loads). 0 = raw slots (90 scale bytes per row);
 * 1 = packed slots (46 B per row: base byte + 45 bytes of 4-bit deltas, see gptoss_cpu_cap2.c). The layout is
 * read once per gptoss_experts_multi call and selects whole-function instantiations, so the inner loops carry no
 * mode test; scale = base + delta is exactly the raw E8M0 byte, so every result is bit-identical to mode 0 on the
 * same logical data (tools/scale_pack_multi_ab.py).
 *
 * BIT-EXACTNESS CONTRACT: for every (row, token) the float operations are exactly those
 * of cap2's rowdot(): two accumulators a0 (even g) / a1 (odd g), g stepping by 2,
 *   b = mul(lut(lo), xe); b = fmadd(lut(hi), xo, b); acc = fmadd(b, 2^(s-127), acc),
 * final _mm512_reduce_add_ps(add(a0, a1)), then + bias in fp32; the same clamped SwiGLU
 * (vectorized, bit-identical to cap2's scalar expf-based loop: see the activation section),
 * the same down projection, and 0.f + w*y. Only the
 * weight dequant is shared between tokens; no cross-token arithmetic exists. Results do
 * not depend on the thread count, the tiling, or how rows are split among threads.
 */
#include "gptoss_cpu_cap2.c"       /* H, GU, GK, RB, OFF_*, rowdot, gptoss_experts(_cap) */
#include <stdlib.h>

#define HH   (H / 2)               /* 1440: de-interleaved half vector                      */
#define TB   8                     /* max tokens per register block (16 zmm accumulators)   */
#define ROW_ALPHA 4                /* load-balance: a row costs (tokens + ROW_ALPHA) units;
                                      ROW_ALPHA ~ DRAM cost of one row in token-compute units */

/* Tuning knobs (process-global; change only between calls, a call reads them once at entry).
 * Row counts are CLAMPED to [1, GU]: no matrix has more than GU rows, so a larger tile / item
 * is exactly equivalent to GU (one tile / one item per matrix), and the clamp keeps every row
 * index expression (nc + rpt, n0 + irows, GU + irows - 1) far from int overflow. Values <= 0
 * (and tokens_per_block outside 1..TB) are ignored. */
static int g_rows_per_tile = 16;   /* 16 rows x 1530 B = 24 KB of weights: stays in L1/L2      */
static int g_tok_per_block = 8;    /* 8 tokens x 11.25 KB of x = 90 KB: L2-resident; 16 zmm acc */

__declspec(dllexport) void gptoss_multi_set_tiling(int rows_per_tile, int tokens_per_block){
  if (rows_per_tile > 0) g_rows_per_tile = (rows_per_tile < GU) ? rows_per_tile : GU;
  if (tokens_per_block >= 1 && tokens_per_block <= TB) g_tok_per_block = tokens_per_block;
}

/* work distribution of the two GEMV phases: 0 = cost-weighted static contiguous row ranges,
 * 1 = dynamic items of g_item_rows rows (robust when other processes steal cores; default,
 * see the header note). The result is bit-identical either way (per-(row, token) arithmetic
 * never changes). */
static int g_dynamic = 1;
static int g_item_rows = 64;
__declspec(dllexport) void gptoss_multi_set_schedule(int dynamic, int rows_per_item){
  g_dynamic = dynamic ? 1 : 0;
  if (rows_per_item > 0) g_item_rows = (rows_per_item < GU) ? rows_per_item : GU;
}

/* Effective knob values: cfg[0] rows_per_tile, cfg[1] tokens_per_block, cfg[2] dynamic,
 * cfg[3] rows_per_item (after clamping). */
__declspec(dllexport) void gptoss_multi_get_config(int* cfg){
  cfg[0] = g_rows_per_tile; cfg[1] = g_tok_per_block; cfg[2] = g_dynamic; cfg[3] = g_item_rows;
}

#ifdef MULTI_PROFILE
/* [0] de-interleave, [1] gate_up, [2] activation, [3] down (seconds, summed over calls), [4] calls */
__declspec(dllexport) double gptoss_multi_prof[5];
#endif

/* E8M0 scale s -> fp32 bit pattern s << 23 (exactly cap2's (int)s << 23), broadcast from memory */
#define E8_4(i)  ((uint32_t)(i) << 23), ((uint32_t)((i)+1) << 23), ((uint32_t)((i)+2) << 23), ((uint32_t)((i)+3) << 23)
#define E8_16(i) E8_4(i), E8_4((i)+4), E8_4((i)+8), E8_4((i)+12)
#define E8_64(i) E8_16(i), E8_16((i)+16), E8_16((i)+32), E8_16((i)+48)
static const uint32_t E8BITS[272] = { E8_64(0), E8_64(64), E8_64(128), E8_64(192), E8_16(256) };
       /* [256..271] exist only so a corrupted packed row (base + nibble > 255) reads garbage, not out of bounds */

/* Up to TB tokens against ONE row. x = first token's de-interleaved vector [xe(1440)|xo(1440)],
 * consecutive tokens H floats apart. r[t] = rowdot(c, s, xe_t, xo_t), bit-identical.
 * vpermps reads only index bits [3:0], so lut[v] == lut[v & 15] (cap2's AND is a no-op). */
static inline __attribute__((always_inline))
void rowdot_tok(const int T, const int sm, const uint8_t* c, const uint8_t* s, const float* x, float* r){
  const __m512 lut = _mm512_setr_ps(0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f);
  __m512 a0[TB], a1[TB];
  #pragma GCC unroll 8
  for (int t = 0; t < T; ++t){ a0[t] = _mm512_setzero_ps(); a1[t] = _mm512_setzero_ps(); }
  const int base = sm ? s[0] : 0;                       /* packed row: byte 0 = base, 1..45 = nibbles */
  for (int g = 0; g < GK; g += 2){
    const __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c + g*16)));
    const __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(c + g*16 + 16)));
    const __m512 l0 = _mm512_permutexvar_ps(v0, lut);
    const __m512 l1 = _mm512_permutexvar_ps(v1, lut);
    const __m512 h0 = _mm512_permutexvar_ps(_mm512_srli_epi32(v0,4), lut);
    const __m512 h1 = _mm512_permutexvar_ps(_mm512_srli_epi32(v1,4), lut);
    int e0, e1; scale_pair(s, base, g, sm, &e0, &e1);   /* the two E8M0 bytes of groups g, g+1, exactly */
    const __m512 s0 = _mm512_castsi512_ps(_mm512_set1_epi32((int)E8BITS[e0]));
    const __m512 s1 = _mm512_castsi512_ps(_mm512_set1_epi32((int)E8BITS[e1]));
    #pragma GCC unroll 8
    for (int t = 0; t < T; ++t){
      const float* xe = x + (size_t)t*H;
      const float* xo = xe + HH;
      __m512 b0 = _mm512_mul_ps(l0, _mm512_loadu_ps(xe + g*16));
      __m512 b1 = _mm512_mul_ps(l1, _mm512_loadu_ps(xe + g*16 + 16));
      b0 = _mm512_fmadd_ps(h0, _mm512_loadu_ps(xo + g*16), b0);
      b1 = _mm512_fmadd_ps(h1, _mm512_loadu_ps(xo + g*16 + 16), b1);
      a0[t] = _mm512_fmadd_ps(b0, s0, a0[t]);
      a1[t] = _mm512_fmadd_ps(b1, s1, a1[t]);
    }
  }
  #pragma GCC unroll 8
  for (int t = 0; t < T; ++t) r[t] = _mm512_reduce_add_ps(_mm512_add_ps(a0[t], a1[t]));
}

/* ---------------------------------------------------------------- activation
 * cap2 (scalar):  gt = fminf(g,7); up = fminf(fmaxf(u,-7),7);
 *                 h  = (up + 1) * gt / (1 + expf(-1.702f*gt))
 * where this toolchain's expf(a) is (float)exp((double)a) and exp() is mingw's x87 routine
 * (~28 ns). The vector path below is bit-identical:
 *  - vminps/vmaxps return the 2nd operand when either input is NaN, i.e. exactly
 *    fminf(x,7) / fmaxf(x,-7) (C99: NaN -> the other argument); +,*,/ are IEEE-exact ops
 *    in the same order;
 *  - e = (float)exp((double)a) is computed as (float)exp_pd((double)a), where exp_pd is a
 *    double exp with a relative error of a few double ulps. Rounding to float can only differ
 *    from mingw's if the double result lies near a float rounding MIDPOINT (low 29 mantissa
 *    bits ~ 0x10000000); every lane within EXP_GUARD double ulps of one, and every lane with
 *    a outside [-86, 88] (or NaN), is recomputed with the scalar expf. gptoss_multi_check_exp()
 *    verifies this against the scalar expf for ALL 2^32 float inputs. */
#define EXP_GUARD 4096LL             /* double ulps around a float midpoint -> scalar fallback */

static inline __m512d exp_pd(__m512d x){                  /* |rel err| ~ 1e-16, |x| <= 90 */
  const __m512d log2e = _mm512_set1_pd(1.4426950408889634074);
  const __m512d ln2hi = _mm512_set1_pd(6.93147180369123816490e-01);   /* fdlibm split of ln 2 */
  const __m512d ln2lo = _mm512_set1_pd(1.90821492927058770002e-10);
  const __m512d n = _mm512_roundscale_pd(_mm512_mul_pd(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512d r = _mm512_fnmadd_pd(n, ln2hi, x);
  r = _mm512_fnmadd_pd(n, ln2lo, r);                      /* |r| <= 0.347 */
  __m512d p = _mm512_set1_pd(1.0 / 6227020800.0);         /* Taylor to r^13: trunc. < 5e-18 */
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 479001600.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 39916800.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 3628800.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 362880.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 40320.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 5040.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 720.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 120.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 24.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0 / 6.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(0.5));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0));
  p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(1.0));
  return _mm512_scalef_pd(p, n);
}

static inline __mmask8 near_midpoint(__m512d d){
  const __m512i low = _mm512_and_si512(_mm512_castpd_si512(d), _mm512_set1_epi64(0x1FFFFFFFLL));
  const __m512i dist = _mm512_abs_epi64(_mm512_sub_epi64(low, _mm512_set1_epi64(0x10000000LL)));
  return _mm512_cmple_epi64_mask(dist, _mm512_set1_epi64(EXP_GUARD));
}

/* 16 x expf(a), bit-identical to the scalar expf; *nfb (if non-NULL) += scalar fallbacks */
static inline __m512 vexpf_x16(__m512 a, long long* nfb){
  const __m512d dlo = exp_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(a)));
  const __m512d dhi = exp_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(a, 1)));
  __m512 e = _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_cvtpd_ps(dlo)), _mm512_cvtpd_ps(dhi), 1);
  const __mmask16 ok = _mm512_cmp_ps_mask(a, _mm512_set1_ps(-86.f), _CMP_GE_OQ)
                     & _mm512_cmp_ps_mask(a, _mm512_set1_ps(88.f), _CMP_LE_OQ);   /* NaN -> not ok */
  const __mmask16 fb = (__mmask16)(~ok | (__mmask16)near_midpoint(dlo) | ((__mmask16)near_midpoint(dhi) << 8));
  if (__builtin_expect(fb != 0, 0)){
    float av[16], ev[16];
    _mm512_storeu_ps(av, a); _mm512_storeu_ps(ev, e);
    for (int i = 0; i < 16; ++i) if (fb & (1u << i)) ev[i] = expf(av[i]);
    if (nfb) *nfb += __builtin_popcount((unsigned)fb);
    e = _mm512_loadu_ps(ev);
  }
  return e;
}

/* 16 activations from gate g[0..15] and up u[0..15] */
static inline __m512 act16(const float* g, const float* u){
  const __m512 seven = _mm512_set1_ps(7.f), one = _mm512_set1_ps(1.f);
  const __m512 gt = _mm512_min_ps(_mm512_loadu_ps(g), seven);                       /* fminf(g, 7)          */
  const __m512 up = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(u), _mm512_set1_ps(-7.f)), seven);
  const __m512 e  = vexpf_x16(_mm512_mul_ps(_mm512_set1_ps(-1.702f), gt), NULL);      /* expf(-1.702f * gt)   */
  return _mm512_div_ps(_mm512_mul_ps(_mm512_add_ps(up, one), gt), _mm512_add_ps(one, e));
}

/* one pair: g = [gate(2880) | up(2880)] -> he[1440] (even j), ho[1440] (odd j) */
static void act_pair_vec(const float* g, float* he, float* ho){
  const __m512i ev = _mm512_setr_epi32(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
  const __m512i od = _mm512_setr_epi32(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
  for (int j = 0; j < H; j += 32){
    const __m512 h0 = act16(g + j, g + H + j), h1 = act16(g + j + 16, g + H + j + 16);
    _mm512_storeu_ps(he + j/2, _mm512_permutex2var_ps(h0, ev, h1));
    _mm512_storeu_ps(ho + j/2, _mm512_permutex2var_ps(h0, od, h1));
  }
}

/* cap2's scalar loop body, for one pair (reference / test hook) */
static void act_pair_ref(const float* g, float* he, float* ho){
  for (int j = 0; j < H; ++j){
    float gt = fminf(g[j], 7.f), up = fminf(fmaxf(g[H + j], -7.f), 7.f);
    float h = (up + 1.f) * gt / (1.f + expf(-1.702f * gt));
    if (j & 1) ho[j/2] = h; else he[j/2] = h;
  }
}

/* Test hook: activation of P pairs GUin[P][GU] -> Hout[P][H] ([he | ho]); vec = 1 production
 * vector path, vec = 0 cap2's scalar expression. */
__declspec(dllexport) void gptoss_multi_activation(int P, const float* GUin, float* Hout, int vec){
  for (int p = 0; p < P; ++p){
    const float* g = GUin + (size_t)p*GU; float* he = Hout + (size_t)p*H;
    if (vec) act_pair_vec(g, he, he + HH); else act_pair_ref(g, he, he + HH);
  }
}

/* Exhaustive check of vexpf_x16 vs the scalar expf over float bit patterns [begin, end)
 * (end <= 2^32). Returns the number of mismatching bit patterns (NaN results compared as
 * NaN == NaN); *first_bad = first mismatching pattern, *n_fallback = lanes that took the
 * scalar fallback. */
__declspec(dllexport) long long gptoss_multi_check_exp(long long begin, long long end, int threads,
                                                       unsigned* first_bad, long long* n_fallback){
  long long bad = 0, nfb = 0; unsigned first = 0xFFFFFFFFu;
  if (threads < 1) threads = 1;
  const long long nblk = (end - begin + 15) / 16;
  #pragma omp parallel for schedule(static, 4096) num_threads(threads) reduction(+:bad,nfb) reduction(min:first)
  for (long long b = 0; b < nblk; ++b){
    uint32_t bits[16]; float a[16], ev[16];
    for (int i = 0; i < 16; ++i){
      long long v = begin + b*16 + i; if (v >= end) v = end - 1;
      bits[i] = (uint32_t)v; memcpy(&a[i], &bits[i], 4);
    }
    long long f = 0;
    _mm512_storeu_ps(ev, vexpf_x16(_mm512_loadu_ps(a), &f));
    nfb += f;
    for (int i = 0; i < 16; ++i){
      const float r = expf(a[i]);
      uint32_t x, y; memcpy(&x, &ev[i], 4); memcpy(&y, &r, 4);
      if (x != y && !(r != r && ev[i] != ev[i])){ ++bad; if (bits[i] < first) first = bits[i]; }
    }
  }
  if (first_bad) *first_bad = first;
  if (n_fallback) *n_fallback = nfb;
  return bad;
}

/* Rows [n0, n1) of one matrix (codes at cb + n*RB, scales at sb + n*SR, SR = GK raw / SR_P4 packed) against the c tokens
 * whose de-interleaved vectors start at x (pair p0 .. p0+c-1, stride H).
 *   down == 0: GUS[p][n] = r + bias[n]                       (gate_up, pre-activation)
 *   down == 1: Y[p][n]   = 0.f + W[p] * (r + bias[n])        (down projection, weighted)
 * Tile of rpt rows (L2) x blocks of tpb tokens (x in L1): every row is fetched from DRAM once
 * per call; the per-(row, token) arithmetic is rowdot's regardless of the tile shape. */
AINL void tile_rows_m(const uint8_t* cb, const uint8_t* sb, const int n0, const int n1,
                      const float* x, const int c, const int p0, const int rpt, const int tpb,
                      const int down, const float* bias, float* GUS, const float* W, float* Y, const int sm){
  const size_t SR = sm ? SR_P4 : GK;                    /* scale bytes per row (constant per instantiation) */
  float r[TB];
  for (int nc = n0; nc < n1; nc += rpt){
    const int ne = (n1 - nc < rpt) ? n1 : nc + rpt;
    for (int q = 0; q < c; q += tpb){
      const int T = (c - q < tpb) ? c - q : tpb;
      const float* xb = x + (size_t)q*H;
      for (int n = nc; n < ne; ++n){
        const uint8_t* cr = cb + (size_t)n*RB; const uint8_t* sr = sb + (size_t)n*SR;
        switch (T){
          case 8: rowdot_tok(8, sm, cr, sr, xb, r); break;
          case 7: rowdot_tok(7, sm, cr, sr, xb, r); break;
          case 6: rowdot_tok(6, sm, cr, sr, xb, r); break;
          case 5: rowdot_tok(5, sm, cr, sr, xb, r); break;
          case 4: rowdot_tok(4, sm, cr, sr, xb, r); break;
          case 3: rowdot_tok(3, sm, cr, sr, xb, r); break;
          case 2: rowdot_tok(2, sm, cr, sr, xb, r); break;
          default: rowdot_tok(1, sm, cr, sr, xb, r); break;
        }
        if (!down){
          for (int k = 0; k < T; ++k) GUS[(size_t)(p0 + q + k)*GU + n] = r[k] + bias[n];
        } else {
          for (int k = 0; k < T; ++k){
            const int p = p0 + q + k;
            const float y = r[k] + bias[n];
            float a = 0.f; a += W[p] * y;          /* == cap2's combine for E = 1 */
            Y[(size_t)p*H + n] = a;
          }
        }
      }
    }
  }
}

static void tile_rows_raw(const uint8_t* cb, const uint8_t* sb, const int n0, const int n1,
                          const float* x, const int c, const int p0, const int rpt, const int tpb,
                          const int down, const float* bias, float* GUS, const float* W, float* Y){
  tile_rows_m(cb, sb, n0, n1, x, c, p0, rpt, tpb, down, bias, GUS, W, Y, 0);
}
static void tile_rows_pk(const uint8_t* cb, const uint8_t* sb, const int n0, const int n1,
                         const float* x, const int c, const int p0, const int rpt, const int tpb,
                         const int down, const float* bias, float* GUS, const float* W, float* Y){
  tile_rows_m(cb, sb, n0, n1, x, c, p0, rpt, tpb, down, bias, GUS, W, Y, 1);
}
/* sm = the scale layout of the slots (0 raw, 1 packed), read ONCE per gptoss_experts_multi call. */
static inline void tile_rows(const int sm, const uint8_t* cb, const uint8_t* sb, const int n0, const int n1,
                             const float* x, const int c, const int p0, const int rpt, const int tpb,
                             const int down, const float* bias, float* GUS, const float* W, float* Y){
  if (sm) tile_rows_pk(cb, sb, n0, n1, x, c, p0, rpt, tpb, down, bias, GUS, W, Y);
  else    tile_rows_raw(cb, sb, n0, n1, x, c, p0, rpt, tpb, down, bias, GUS, W, Y);
}

/* Cost-weighted contiguous ("static") partition of the flattened (expert, row) space:
 * a row of expert e costs cnt_e + ROW_ALPHA units (empty experts cost nothing). Thread t
 * owns every row whose cost interval STARTS in [total*t/nt, total*(t+1)/nt). Returns the
 * row range [*n0, *n1) of expert e for this thread (possibly empty). Deterministic. */
static inline void part_rows(long long total, long long base, long long w, int rows, int t, int nt,
                             int* n0, int* n1){
  const long long lo = total * t / nt, hi = total * (t + 1) / nt;
  long long a = (lo <= base) ? 0 : (lo - base + w - 1) / w;
  long long b = (hi <= base) ? 0 : (hi - base + w - 1) / w;
  if (a > rows) a = rows;
  if (b > rows) b = rows;
  *n0 = (int)a; *n1 = (int)b;
}

/* Scratch layout (floats), P = off[E] pairs, base aligned up to 64 B inside the kernel:
 *   XH [P][H]  : phase 1 de-interleaved input [xe(1440) | xo(1440)] per pair;
 *                phase 3 overwrites it with the de-interleaved activation [he | ho]
 *   GUS[P][GU] : gate_up pre-activation per pair ([gate(2880) | up(2880)], + bias)
 * total = P*(H + GU) + 16 floats (+16 = alignment slack). E is unused (API symmetry).
 * Returns -1 if that does not fit an int, i.e. unless P*(H+GU) + 16 <= INT_MAX
 * (P <= 248,551 = floor((2^31 - 1 - 16) / 8640)). */
__declspec(dllexport) int gptoss_multi_scratch_floats(int E, int P){
  (void)E;
  if (P < 0) P = 0;
  if ((long long)P * (H + GU) + 16 > 2147483647LL) return -1;
  return P * (H + GU) + 16;
}

/* E experts of one layer, pairs grouped by expert: pairs [off[e], off[e+1]) are routed to
 * slots[e]. Per pair p (NOT summed over experts):
 *   Y[p*H + n] = 0.f + W[p] * (down(act(gate_up(X[p]) + bgu[e])) + bdn[e])[n]
 * bit-identical to gptoss_experts(1, &slots[e], X+p*H, &bgu[e], &bdn[e], &W[p], Y+p*H, ...).
 * Pass W[p] = 1.f to get the raw expert output y (0.f + 1.f*y == y up to the sign of zero).
 * Pairs outside [off[0], off[E]) are not touched. X, Y, W are read/written only for those. */
__declspec(dllexport) void gptoss_experts_multi(int E, const uint8_t** slots, const int* off,
                                                const float* X, const float** bgu, const float** bdn,
                                                const float* W, float* Y, float* scratch, int threads){
  if (E <= 0) return;
  const int P = off[E];
  if (P <= 0) return;
  if (threads < 1) threads = 1;
  const int rpt = g_rows_per_tile, tpb = g_tok_per_block;
  const int sm = (g_sl == 1);                                          /* scale layout, read once per call */
  const size_t ods = OFF_GS + (size_t)GU * (sm ? SR_P4 : GK);          /* down-scale region of a slot (OFF_DS when raw) */
  float* XH  = (float*)(((uintptr_t)scratch + 63) & ~(uintptr_t)63);
  float* GUS = XH + (size_t)P*H;

  long long tot_gu = 0, tot_dn = 0;
  int nact = 0;
  int act_small[256];                                   /* active (non-empty) experts */
  int* act = (E <= 256) ? act_small : (int*)malloc(sizeof(int) * (size_t)E);
  const int dyn = g_dynamic && act != NULL, irows = g_item_rows;   /* no list -> static */
  for (int e = 0; e < E; ++e){
    const int c = off[e+1] - off[e];
    if (c > 0){
      tot_gu += (long long)GU * (c + ROW_ALPHA); tot_dn += (long long)H * (c + ROW_ALPHA);
      if (act) act[nact++] = e;
    }
  }
#ifdef MULTI_PROFILE
  double pt = omp_get_wtime();
  #define PROF(i) do { if (t == 0){ const double _n = omp_get_wtime(); gptoss_multi_prof[i] += _n - pt; pt = _n; } } while (0)
#else
  #define PROF(i) ((void)0)
#endif

  #pragma omp parallel num_threads(threads)
  {
    const int t = omp_get_thread_num(), nt = omp_get_num_threads();

    /* phase 1: de-interleave every pair's input (pairs off[0] .. P-1) */
    #pragma omp for schedule(static)
    for (int p = off[0]; p < P; ++p){
      const float* x = X + (size_t)p*H;
      float* xe = XH + (size_t)p*H; float* xo = xe + HH;
      for (int i = 0; i < HH; ++i){ xe[i] = x[2*i]; xo[i] = x[2*i+1]; }
    }
    PROF(0);

    /* phase 2: gate_up rows, each row read once for all of its expert's tokens */
    if (dyn){
      const long long per = ((long long)GU + irows - 1) / irows;   /* irows in [1, GU] */
      #pragma omp for schedule(dynamic, 1) nowait
      for (long long it = 0; it < (long long)nact * per; ++it){
        const int e = act[it / per], n0 = (int)(it % per) * irows;
        const int n1 = (n0 + irows < GU) ? n0 + irows : GU, p0 = off[e];
        tile_rows(sm, slots[e], slots[e] + OFF_GS, n0, n1, XH + (size_t)p0*H, off[e+1] - p0, p0, rpt, tpb,
                  0, bgu[e], GUS, NULL, NULL);
      }
    } else {
      long long base = 0;
      for (int e = 0; e < E; ++e){
        const int p0 = off[e], c = off[e+1] - off[e];
        if (c <= 0) continue;
        const long long w = c + ROW_ALPHA;
        int n0, n1; part_rows(tot_gu, base, w, GU, t, nt, &n0, &n1);
        base += (long long)GU * w;
        if (n0 < n1)
          tile_rows(sm, slots[e], slots[e] + OFF_GS, n0, n1, XH + (size_t)p0*H, c, p0, rpt, tpb,
                    0, bgu[e], GUS, NULL, NULL);
      }
    }
    #pragma omp barrier
    PROF(1);

    /* phase 3: clamped SwiGLU (bit-identical vector form of cap2's expression, see above),
     * de-interleaved into XH (the input is dead now) */
    #pragma omp for schedule(static)
    for (int p = off[0]; p < P; ++p)
      act_pair_vec(GUS + (size_t)p*GU, XH + (size_t)p*H, XH + (size_t)p*H + HH);
    PROF(2);

    /* phase 4: down rows, each row read once for all tokens; Y = 0 + w*(y + bias) */
    if (dyn){
      const long long per = ((long long)H + irows - 1) / irows;
      #pragma omp for schedule(dynamic, 1) nowait
      for (long long it = 0; it < (long long)nact * per; ++it){
        const int e = act[it / per], n0 = (int)(it % per) * irows;
        const int n1 = (n0 + irows < H) ? n0 + irows : H, p0 = off[e];
        tile_rows(sm, slots[e] + OFF_DC, slots[e] + ods, n0, n1, XH + (size_t)p0*H, off[e+1] - p0, p0, rpt, tpb,
                  1, bdn[e], NULL, W, Y);
      }
    } else {
      long long base = 0;
      for (int e = 0; e < E; ++e){
        const int p0 = off[e], c = off[e+1] - off[e];
        if (c <= 0) continue;
        const long long w = c + ROW_ALPHA;
        int n0, n1; part_rows(tot_dn, base, w, H, t, nt, &n0, &n1);
        base += (long long)H * w;
        if (n0 < n1)
          tile_rows(sm, slots[e] + OFF_DC, slots[e] + ods, n0, n1, XH + (size_t)p0*H, c, p0, rpt, tpb,
                    1, bdn[e], NULL, W, Y);
      }
    }
  }
#ifdef MULTI_PROFILE
  { const int t = 0; PROF(3); }
  gptoss_multi_prof[4] += 1.0;
#endif
  #undef PROF
  if (act && act != act_small) free(act);
}

/* w*y rounded to fp32 on its own: the empty asm makes the product opaque, so the compiler
 * cannot contract it with the following add into an FMA. */
static inline float mul_rn(float a, float b){ float p = a * b; __asm__("" : "+v"(p)); return p; }

/* Optional exact combine. out[t*H + n] = the E-expert combine of gptoss_experts() over the
 * valid entries (idx[t*K+k] >= 0, in k order) of token t, bit for bit, given
 * Yr = gptoss_experts_multi(..., W = 1.f) (raw expert outputs; 0 + 1*y == y).
 *
 * The C source of that combine is `a = 0.f; for e < E: a += w[e]*y[e]`, but its gcc-16
 * -O3 -march=native code (gptoss_cpu_cap2.dll, verified in the disassembly and by test)
 * is NOT one uniform FMA chain: it SLP-vectorizes the e-loop in blocks of 8 and then one
 * block of 4 (products rounded by vmulps, then added one by one in order), and only the
 * last E mod 4 (< 4) terms use fused vfmadd231ss. So E <= 3 -> pure FMA chain, E = 4 ->
 * four rounded products summed in order, E = 5..7 -> 4 unfused + FMA tail, etc.
 * This function reproduces exactly that. An entry with idx < 0 is skipped (e.g. an
 * expert computed on the GPU); a token with no valid entry gets 0. */
__declspec(dllexport) void gptoss_combine_pairs(int N, int K, const int* idx, const float* w,
                                                const float* Yr, float* out, int threads){
  if (N <= 0) return;
  if (threads < 1) threads = 1;
  #pragma omp parallel for schedule(static) num_threads(threads)
  for (long long i = 0; i < (long long)N*H; ++i){
    const int t = (int)(i / H), n = (int)(i % H);
    const int* ix = idx + (size_t)t*K; const float* wt = w + (size_t)t*K;
    int E = 0;
    for (int k = 0; k < K; ++k) E += (ix[k] >= 0);
    const int unfused = (E / 8) * 8 + ((E % 8) >= 4 ? 4 : 0);
    float a = 0.f;
    for (int k = 0, j = 0; k < K; ++k){
      if (ix[k] < 0) continue;
      const float y = Yr[(size_t)ix[k]*H + n];
      if (j < unfused) a = a + mul_rn(wt[k], y);
      else             a = fmaf(wt[k], y, a);
      ++j;
    }
    out[i] = a;
  }
}
