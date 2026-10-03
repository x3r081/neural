/* Opt-in persistent decode team and a bounded rendezvous probe. Build a
 * SEPARATE DLL; never replace gptoss_cpu_cap2.dll. Default OFF. The included
 * reference implementation and its original ABI are preserved under wrappers.
 * Math is the fused cap2 path, with the same gu/y stores and row/expert order.
 */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stddef.h>

#define gptoss_experts base_gptoss_experts
#define gptoss_experts_cap base_gptoss_experts_cap
#define gptoss_set_tuning base_gptoss_set_tuning
#define gptoss_get_tuning base_gptoss_get_tuning
#define gptoss_set_fuse base_gptoss_set_fuse
#define gptoss_set_scale_layout base_gptoss_set_scale_layout
#define gptoss_get_scale_layout base_gptoss_get_scale_layout
#define gptoss_set_cold_prefetch base_gptoss_set_cold_prefetch
#define gptoss_get_cold_prefetch base_gptoss_get_cold_prefetch
#define gptoss_cold_prefetch_stats base_gptoss_cold_prefetch_stats
#define gptoss_affinity_probe base_gptoss_affinity_probe
#include "gptoss_cpu_cap2.c"
#undef gptoss_experts
#undef gptoss_experts_cap
#undef gptoss_set_tuning
#undef gptoss_get_tuning
#undef gptoss_set_fuse
#undef gptoss_set_scale_layout
#undef gptoss_get_scale_layout
#undef gptoss_set_cold_prefetch
#undef gptoss_get_cold_prefetch
#undef gptoss_cold_prefetch_stats
#undef gptoss_affinity_probe

#ifndef _WIN32
#error This experimental native team currently targets the Windows host.
#endif
__declspec(dllimport) void* __stdcall GetModuleHandleA(const char*);
__declspec(dllimport) void* __stdcall GetProcAddress(void*, const char*);
__declspec(dllimport) int __stdcall SwitchToThread(void);
__declspec(dllimport) void* __stdcall CreateThread(void*, size_t,
    unsigned long (__stdcall *)(void*), void*, unsigned long, unsigned long*);
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void*, unsigned long);
__declspec(dllimport) int __stdcall CloseHandle(void*);
__declspec(dllimport) void __stdcall AcquireSRWLockExclusive(void*);
__declspec(dllimport) void __stdcall ReleaseSRWLockExclusive(void*);
__declspec(dllimport) int __stdcall QueryPerformanceCounter(long long*);
__declspec(dllimport) int __stdcall QueryPerformanceFrequency(long long*);
typedef int (__stdcall *wait_fn)(volatile void*, void*, size_t, unsigned long);
typedef void (__stdcall *wake_fn)(void*);

#define TEAM_MAX 16
typedef struct { volatile long count; char pad[60]; volatile long generation; char pad2[60]; } team_barrier;
typedef struct { volatile unsigned long long value; char pad[56]; } probe_mark;
static void* pool_lock; /* SRWLOCK_INIT, pointer-sized and zero initialized */
static wait_fn wait_address;
static wake_fn wake_address;
static void* workers[TEAM_MAX - 1];
static int pool_threads, pool_enabled;
static volatile long job_generation, job_done, worker_ready, stop_requested;
static team_barrier phase;
static probe_mark marks[TEAM_MAX] __attribute__((aligned(64)));
static void (*job_step)(int);
static unsigned long long probe_sequence;
static unsigned long long math_jobs, fallback_calls;
static int fail_create_after = -1; /* test hook, disabled in production */
static unsigned worker_mxcsr[TEAM_MAX];
static unsigned short worker_x87[TEAM_MAX];
typedef struct {
    int E, pf;
    const uint8_t** slots;
    const float** bgu;
    const float** bdn;
    const float* w;
    float* out;
    float *xe, *xo, *gu, *he, *ho, *y;
    uint8_t** cap;
} math_context;
static math_context current_math;
static volatile long row1, row2;
static int stop_pool(void);
static int start_pool(int threads);
static void dispatch(void (*step)(int));
static inline unsigned short x87_control(void) {
    unsigned short value; __asm__ volatile("fnstcw %0" : "=m"(value)); return value;
}
static inline void set_x87(unsigned short value) {
    __asm__ volatile("fldcw %0" : : "m"(value));
}

static void wake(volatile long* address) {
    if (wake_address) wake_address((void*)address);
}
static void wait_change(volatile long* address, long old) {
    const unsigned long long begin = __rdtsc();
    while (__atomic_load_n(address, __ATOMIC_ACQUIRE) == old) {
        const unsigned long long elapsed = __rdtsc() - begin;
        if (elapsed < 150000ULL) _mm_pause();
        else if (elapsed > 3000000ULL && wait_address)
            wait_address(address, &old, sizeof(old), 5);
        else SwitchToThread();
    }
}
static void phase_barrier(int n) {
    const long old = __atomic_load_n(&phase.generation, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(&phase.count, 1, __ATOMIC_SEQ_CST) == n) {
        __atomic_store_n(&phase.count, 0, __ATOMIC_RELAXED);
        __atomic_add_fetch(&phase.generation, 1, __ATOMIC_RELEASE);
        wake(&phase.generation);
    } else wait_change(&phase.generation, old);
}
static void probe_step(int tid) {
    marks[tid].value = probe_sequence;
    phase_barrier(pool_threads);
    marks[tid].value = probe_sequence + 1;
    _mm_sfence();
}
static unsigned long __stdcall worker_main(void* argument) {
    const int tid = (int)(intptr_t)argument;
    long seen = __atomic_load_n(&job_generation, __ATOMIC_ACQUIRE);
    __atomic_add_fetch(&worker_ready, 1, __ATOMIC_RELEASE);
    wake(&worker_ready);
    for (;;) {
        /* Startup/partial-CreateThread failure may publish stop before this
         * worker snapshots generation. Check before waiting, not only after. */
        if (__atomic_load_n(&stop_requested, __ATOMIC_ACQUIRE)) break;
        wait_change(&job_generation, seen);
        seen = __atomic_load_n(&job_generation, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&stop_requested, __ATOMIC_ACQUIRE)) break;
        const unsigned old_mxcsr = _mm_getcsr();
        const unsigned short old_x87 = x87_control();
        _mm_setcsr(worker_mxcsr[tid]); set_x87(worker_x87[tid]);
        job_step(tid);
        _mm_setcsr(old_mxcsr); set_x87(old_x87);
        __atomic_add_fetch(&job_done, 1, __ATOMIC_RELEASE);
        wake(&job_done);
    }
    return 0;
}

/* Two phases copied from the fused reference. Only work assignment and team
 * lifetime differ. Do not remove gu/y materialization or combine/activate math.
 * sm is a constant in the raw/packed callback wrappers below. */
AINL void persistent_math_step(int tid, const int sm) {
    (void)tid;
    const math_context* m = &current_math;
    const size_t SR = sm ? SR_P4 : GK;
    const size_t ODS = OFF_GS + (size_t)GU * SR;
    const int E = m->E, pf = m->pf;
    for (;;) {
        const long first = __atomic_fetch_add(&row1, 64, __ATOMIC_RELAXED);
        if (first >= E*H) break;
        const long end = first + 64 < E*H ? first + 64 : E*H;
        for (long i = first; i < end; ++i) {
            const int e = i / H, j = i % H; const uint8_t* sl = m->slots[e];
            float rg, ru;
            rowdot2_pf(sl + (size_t)j*RB, sl + OFF_GS + (size_t)j*SR,
                       sl + (size_t)(H + j)*RB, sl + OFF_GS + (size_t)(H + j)*SR,
                       m->xe, m->xo, &rg, &ru, pf, sm);
            float* g = m->gu + (size_t)e*GU;
            g[j] = rg + m->bgu[e][j];
            g[H + j] = ru + m->bgu[e][H + j];
            float gt = fminf(g[j], 7.f), up = fminf(fmaxf(g[H + j], -7.f), 7.f);
            float h = (up + 1.f) * gt / (1.f + expf(-1.702f * gt));
            if (j & 1) m->ho[(size_t)e*(H/2) + j/2] = h; else m->he[(size_t)e*(H/2) + j/2] = h;
            if (m->cap && m->cap[e]) {
                cap_row(m->cap[e] + (size_t)j*RB, sl + (size_t)j*RB,
                        m->cap[e] + OFF_GS + (size_t)j*SR, sl + OFF_GS + (size_t)j*SR, SR);
                cap_row(m->cap[e] + (size_t)(H + j)*RB, sl + (size_t)(H + j)*RB,
                        m->cap[e] + OFF_GS + (size_t)(H + j)*SR, sl + OFF_GS + (size_t)(H + j)*SR, SR);
            }
        }
    }
    phase_barrier(pool_threads);
    for (;;) {
        const long first = __atomic_fetch_add(&row2, 32, __ATOMIC_RELAXED);
        if (first >= H) break;
        const long end = first + 32 < H ? first + 32 : H;
        for (long n = first; n < end; ++n) {
            for (int e = 0; e < E; ++e) {
                const uint8_t* sl = m->slots[e];
                m->y[(size_t)e*H + n] = rowdot_pf(sl + OFF_DC + (size_t)n*RB,
                    sl + ODS + (size_t)n*SR, m->he + (size_t)e*(H/2),
                    m->ho + (size_t)e*(H/2), pf, sm) + m->bdn[e][n];
                if (m->cap && m->cap[e])
                    cap_row(m->cap[e] + OFF_DC + (size_t)n*RB, sl + OFF_DC + (size_t)n*RB,
                        m->cap[e] + ODS + (size_t)n*SR, sl + ODS + (size_t)n*SR, SR);
            }
            float a = 0.f;
            for (int e = 0; e < E; ++e) a += m->w[e] * m->y[(size_t)e*H + n];
            m->out[n] = a;
        }
    }
    /* Required before release-publishing completion: capture stores are NT. */
    _mm_sfence();
}
static void math_raw(int tid) { persistent_math_step(tid, 0); }
static void math_packed(int tid) { persistent_math_step(tid, 1); }

__declspec(dllexport) void gptoss_experts_cap(int E, const uint8_t** slots,
    const float* x, const float** bgu, const float** bdn, const float* w,
    float* out, float* scratch, int threads, uint8_t** cap) {
    AcquireSRWLockExclusive(&pool_lock);
    /* Affinity uses OpenMP thread ids in the reference. Preserve that ABI by
     * falling back rather than pinning all native workers as tid=0. The opt-in
     * path supports the production masked-exception/nearest FP environment;
     * non-default rounding/DAZ/FTZ callers retain the reference behavior. */
    const unsigned mxcsr = _mm_getcsr();
    const unsigned short cw = x87_control();
    if (!pool_enabled || !g_fuse || g_aff || threads < 1 || threads > TEAM_MAX
            || (mxcsr & 0xffc0U) != 0x1f80U || (cw & 0x0c3fU) != 0x003fU
            || start_pool(threads) != 0) {
        fallback_calls++;
        experts_impl(E, slots, x, bgu, bdn, w, out, scratch, threads, cap);
        ReleaseSRWLockExclusive(&pool_lock);
        return;
    }
    const size_t SR = g_sl ? SR_P4 : GK;
    if (__builtin_expect(g_cp, 0)) cold_probe(E, slots, SR, g_sl ? SLOT_PACKED : SLOT_RAW);
    current_math = (math_context){.E=E, .pf=g_pf, .slots=slots, .bgu=bgu, .bdn=bdn,
        .w=w, .out=out, .xe=scratch, .xo=scratch+H/2, .gu=scratch+H,
        .cap=cap};
    current_math.he = current_math.gu + (size_t)E*GU;
    current_math.ho = current_math.he + (size_t)E*(H/2);
    current_math.y = current_math.ho + (size_t)E*(H/2);
    for (int i = 0; i < H/2; ++i) { current_math.xe[i] = x[2*i]; current_math.xo[i] = x[2*i+1]; }
    __atomic_store_n(&row1, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&row2, 0, __ATOMIC_RELAXED);
    dispatch(g_sl ? math_packed : math_raw);
    _mm_sfence();
    math_jobs++;
    ReleaseSRWLockExclusive(&pool_lock);
}
__declspec(dllexport) void gptoss_experts(int E, const uint8_t** slots, const float* x,
    const float** bgu, const float** bdn, const float* w, float* out, float* scratch, int threads) {
    gptoss_experts_cap(E, slots, x, bgu, bdn, w, out, scratch, threads, 0);
}

/* All mutable configuration and probes share the job/lifecycle lock. */
__declspec(dllexport) void gptoss_set_tuning(int pf, int pair, int aff) {
    AcquireSRWLockExclusive(&pool_lock); base_gptoss_set_tuning(pf, pair, aff); ReleaseSRWLockExclusive(&pool_lock);
}
__declspec(dllexport) void gptoss_get_tuning(int* cfg) {
    AcquireSRWLockExclusive(&pool_lock); base_gptoss_get_tuning(cfg); ReleaseSRWLockExclusive(&pool_lock);
}
__declspec(dllexport) void gptoss_set_fuse(int fuse) {
    AcquireSRWLockExclusive(&pool_lock); base_gptoss_set_fuse(fuse); ReleaseSRWLockExclusive(&pool_lock);
}
__declspec(dllexport) int gptoss_set_scale_layout(int mode) {
    AcquireSRWLockExclusive(&pool_lock); int r = base_gptoss_set_scale_layout(mode); ReleaseSRWLockExclusive(&pool_lock); return r;
}
__declspec(dllexport) int gptoss_get_scale_layout(void) {
    AcquireSRWLockExclusive(&pool_lock); int r = base_gptoss_get_scale_layout(); ReleaseSRWLockExclusive(&pool_lock); return r;
}
__declspec(dllexport) void gptoss_set_cold_prefetch(int on, int threshold) {
    AcquireSRWLockExclusive(&pool_lock); base_gptoss_set_cold_prefetch(on, threshold); ReleaseSRWLockExclusive(&pool_lock);
}
__declspec(dllexport) void gptoss_get_cold_prefetch(int* cfg) {
    AcquireSRWLockExclusive(&pool_lock); base_gptoss_get_cold_prefetch(cfg); ReleaseSRWLockExclusive(&pool_lock);
}
__declspec(dllexport) void gptoss_cold_prefetch_stats(long long* out5) {
    AcquireSRWLockExclusive(&pool_lock); base_gptoss_cold_prefetch_stats(out5); ReleaseSRWLockExclusive(&pool_lock);
}
__declspec(dllexport) void gptoss_affinity_probe(int threads, unsigned long long* masks) {
    AcquireSRWLockExclusive(&pool_lock); base_gptoss_affinity_probe(threads, masks); ReleaseSRWLockExclusive(&pool_lock);
}
/* Test-only deterministic failure injection. -1 disables; 0..15 fail after
 * that many worker handles. Existing pool is stopped before the next attempt. */
__declspec(dllexport) int gptoss_persistent_fail_create_after(int n) {
    if (n < -1 || n >= TEAM_MAX) return -1;
    AcquireSRWLockExclusive(&pool_lock);
    int r = pool_threads ? stop_pool() : 0;
    fail_create_after = n;
    ReleaseSRWLockExclusive(&pool_lock);
    return r;
}
/* Probe actual thread controls under OpenMP/persistent execution. out[2*N]
 * holds MXCSR then x87 CW per thread; do not infer exception-status equality. */
static unsigned* fp_probe_out;
static void fp_probe_step(int tid) {
    fp_probe_out[2*tid] = _mm_getcsr(); fp_probe_out[2*tid+1] = x87_control();
    phase_barrier(pool_threads); _mm_sfence();
}
__declspec(dllexport) int gptoss_team_fp_probe(int mode, int threads, unsigned* out) {
    AcquireSRWLockExclusive(&pool_lock);
    if (mode == 1) {
        int r = start_pool(threads); if (r) { ReleaseSRWLockExclusive(&pool_lock); return r; }
        fp_probe_out = out; dispatch(fp_probe_step);
    } else {
        #pragma omp parallel num_threads(threads)
        {
            int tid=omp_get_thread_num(); out[2*tid]=_mm_getcsr(); out[2*tid+1]=x87_control();
        }
    }
    ReleaseSRWLockExclusive(&pool_lock); return 0;
}
/* Called with pool_lock held. No jobs are in flight when teardown starts. */
static int stop_pool(void) {
    __atomic_store_n(&stop_requested, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&job_generation, 1, __ATOMIC_RELEASE);
    wake(&job_generation);
    int status = 0;
    for (int i = 0; i < pool_threads - 1; ++i) {
        if (WaitForSingleObject(workers[i], 0xffffffffUL) != 0) status = -3;
        CloseHandle(workers[i]); workers[i] = 0;
    }
    pool_threads = 0;
    return status;
}
static int start_pool(int threads) {
    if (threads < 1 || threads > TEAM_MAX) return -1;
    if (pool_threads == threads) return 0;
    if (pool_threads) { int status = stop_pool(); if (status) return status; }
    if (!wait_address || !wake_address) {
        void* module = GetModuleHandleA("kernelbase.dll");
        if (module) {
            wait_address = (wait_fn)GetProcAddress(module, "WaitOnAddress");
            wake_address = (wake_fn)GetProcAddress(module, "WakeByAddressAll");
        }
        if (!wait_address || !wake_address) return -2;
    }
    /* Windows libgomp workers use x87 PC=64 while the Python caller uses
     * PC=53 on this host. Broadcasting the caller would change expf. Capture
     * the actual reference team's per-thread environment once at pool start;
     * caller remains untouched and native workers restore their own afterward.
     * Dynamic OpenMP downsizing is unsupported rather than guessed. */
    int actual_threads = 0;
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        worker_mxcsr[tid] = _mm_getcsr(); worker_x87[tid] = x87_control();
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    if (actual_threads != threads) return -6;
    __atomic_store_n(&stop_requested, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&worker_ready, 0, __ATOMIC_RELAXED);
    pool_threads = 1;
    for (int tid = 1; tid < threads; ++tid) {
        void* handle = (fail_create_after >= 0 && tid > fail_create_after) ? 0
            : CreateThread(0, 0, worker_main, (void*)(intptr_t)tid, 0, 0);
        if (!handle) { stop_pool(); return -4; }
        workers[tid - 1] = handle; pool_threads++;
    }
    while (__atomic_load_n(&worker_ready, __ATOMIC_ACQUIRE) < threads - 1) {
        long old = __atomic_load_n(&worker_ready, __ATOMIC_ACQUIRE);
        if (old < threads - 1) wait_change(&worker_ready, old);
    }
    return 0;
}
static void dispatch(void (*step)(int)) {
    job_step = step;
    __atomic_store_n(&job_done, 0, __ATOMIC_RELAXED);
    __atomic_add_fetch(&job_generation, 1, __ATOMIC_RELEASE);
    wake(&job_generation);
    step(0);
    while (__atomic_load_n(&job_done, __ATOMIC_ACQUIRE) < pool_threads - 1) {
        long old = __atomic_load_n(&job_done, __ATOMIC_ACQUIRE);
        if (old < pool_threads - 1) wait_change(&job_done, old);
    }
}
__declspec(dllexport) int gptoss_set_persistent(int enabled) {
    AcquireSRWLockExclusive(&pool_lock);
    int status = 0;
    if (!enabled && pool_threads) status = stop_pool();
    pool_enabled = enabled != 0;
    ReleaseSRWLockExclusive(&pool_lock);
    return status;
}
__declspec(dllexport) int gptoss_persistent_shutdown(void) {
    return gptoss_set_persistent(0);
}
/* status[8]: enabled, total team threads (caller included), worker handles,
 * WaitOnAddress resolved, executed math jobs low/high32, fallback calls low/high32.
 * Counts are cumulative process lifetime and EXCLUDE no-op probes. */
__declspec(dllexport) void gptoss_persistent_status(int* status) {
    AcquireSRWLockExclusive(&pool_lock);
    status[0] = pool_enabled; status[1] = pool_threads;
    status[2] = pool_threads ? pool_threads - 1 : 0;
    status[3] = wait_address != 0;
    status[4] = (int)(uint32_t)math_jobs; status[5] = (int)(uint32_t)(math_jobs >> 32);
    status[6] = (int)(uint32_t)fallback_calls; status[7] = (int)(uint32_t)(fallback_calls >> 32);
    ReleaseSRWLockExclusive(&pool_lock);
}
/* NO-OP probe only. mode 0: fresh omp region, mode 1: persistent rendezvous.
 * A phase barrier plus sfence and visible per-thread stores occur in both.
 * gap_us is outside each measured call, busy caller work representing host
 * dispatch between native calls. Native QPC avoids Python/ctypes timing tax.
 */
__declspec(dllexport) int gptoss_team_probe_batch(int mode, int threads,
    int calls, int gap_us, double* microseconds, unsigned long long* checksum) {
    if ((mode != 0 && mode != 1) || threads < 1 || threads > TEAM_MAX || calls < 1 || gap_us < 0) return -1;
    AcquireSRWLockExclusive(&pool_lock);
    if (mode == 1) { int status = start_pool(threads); if (status) { ReleaseSRWLockExclusive(&pool_lock); return status; } }
    long long frequency; QueryPerformanceFrequency(&frequency);
    unsigned long long sum = 0;
    for (int i = 0; i < calls; ++i) {
        long long begin, end; probe_sequence++;
        QueryPerformanceCounter(&begin);
        if (mode == 1) dispatch(probe_step);
        else {
            #pragma omp parallel num_threads(threads)
            {
                const int tid = omp_get_thread_num();
                marks[tid].value = probe_sequence;
                phase_barrier(omp_get_num_threads());
                marks[tid].value = probe_sequence + 1;
                _mm_sfence();
            }
        }
        QueryPerformanceCounter(&end);
        microseconds[i] = (double)(end - begin) * 1.0e6 / frequency;
        for (int tid = 0; tid < threads; ++tid) {
            if (marks[tid].value != probe_sequence + 1) { ReleaseSRWLockExclusive(&pool_lock); return -5; }
            sum += marks[tid].value;
        }
        if (gap_us) {
            const long long target = end + (long long)((double)gap_us * frequency / 1.0e6);
            do { _mm_pause(); QueryPerformanceCounter(&begin); } while (begin < target);
        }
    }
    *checksum = sum;
    ReleaseSRWLockExclusive(&pool_lock);
    return 0;
}
