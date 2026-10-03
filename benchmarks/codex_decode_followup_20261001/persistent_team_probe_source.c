/* Bounded rendezvous probe for a candidate persistent decode team.
 * No expert weights or model inference in this initial stage. Build a SEPARATE
 * DLL; never replace gptoss_cpu_cap2.dll. WaitOnAddress is dynamically resolved.
 * Probe phase barrier matches cap2's pause/yield/block thresholds.
 */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stddef.h>

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
        wait_change(&job_generation, seen);
        seen = __atomic_load_n(&job_generation, __ATOMIC_ACQUIRE);
        if (__atomic_load_n(&stop_requested, __ATOMIC_ACQUIRE)) break;
        job_step(tid);
        __atomic_add_fetch(&job_done, 1, __ATOMIC_RELEASE);
        wake(&job_done);
    }
    return 0;
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
    __atomic_store_n(&stop_requested, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&worker_ready, 0, __ATOMIC_RELAXED);
    pool_threads = 1;
    for (int tid = 1; tid < threads; ++tid) {
        void* handle = CreateThread(0, 0, worker_main, (void*)(intptr_t)tid, 0, 0);
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
/* status[4]: enabled, total team threads (caller included), worker handles,
 * WaitOnAddress resolved. Caller must provide four int elements. */
__declspec(dllexport) void gptoss_persistent_status(int* status) {
    AcquireSRWLockExclusive(&pool_lock);
    status[0] = pool_enabled; status[1] = pool_threads;
    status[2] = pool_threads ? pool_threads - 1 : 0;
    status[3] = wait_address != 0;
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
