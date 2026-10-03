/* multi-threaded copy of one expert row (mmap'd store -> pinned staging buffer) */
#include <omp.h>
#include <stdint.h>
#include <string.h>
__declspec(dllexport) void memcpy_mt(uint8_t* dst, const uint8_t* src, size_t n, int threads){
  size_t chunk = (n + (size_t)threads - 1) / (size_t)threads;
  chunk = (chunk + 4095) & ~(size_t)4095;
  #pragma omp parallel num_threads(threads)
  {
    size_t a = (size_t)omp_get_thread_num() * chunk;
    if (a < n){ size_t b = a + chunk; if (b > n) b = n; memcpy(dst + a, src + a, b - a); }
  }
}
