#include <timer.hpp>

#include <numa.h>
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

extern "C" int numa_copy(int num_threads, size_t size, uint16_t src,
                         uint16_t dst, uint64_t *nanos) {
  omp_set_num_threads(num_threads);

  // Bind thread to src NUMA node
  if (numa_run_on_node(src) == -1) {
    ::perror("Couldn't bind thread to NUMA node");
    return -1;
  }

  // Set that we fail if can't allocate memory on specified NUMA node
  numa_set_strict(true);

  // Allocate memory on src and dst NUMA nodes
  void *__restrict__ mem_src, *__restrict__ mem_dst;
  int *__restrict__ int_src, *__restrict__ int_dst;
  if ((mem_src = numa_alloc_onnode(size, src)) == nullptr) {
    ::perror("Couldn't allocate src memory");
    return -1;
  }

  if ((mem_dst = numa_alloc_onnode(size, dst)) == nullptr) {
    numa_free(mem_src, size);
    ::perror("Couldn't allocate dst memory");
    return -1;
  }

  // Cast to integer for faster assignment
  int_src = reinterpret_cast<int *>(mem_src);
  int_dst = reinterpret_cast<int *>(mem_dst);

  // Initialize memory and make sure src memory is physically allocated
  for (size_t i = 0; i < size / sizeof(int); ++i) {
    int_src[i] = i;
    int_dst[i] = 2 * i;
  }

  //     int actual_threads = 0;
  // #pragma omp parallel
  // #pragma omp atomic
  //         actual_threads++;

  // Time the copy
  Timer::Timer memcpy_timer;
  memcpy_timer.start();

  // Perform copy
  size_t size_per_thread = size / sizeof(int) / num_threads;
#pragma omp parallel
  {
    size_t tid = (size_t)omp_get_thread_num();
    ::memcpy(int_dst + tid * size_per_thread, int_src + tid * size_per_thread,
             size_per_thread);
  }

  // Stop the timer
  *nanos = memcpy_timer.stop<std::chrono::nanoseconds>();

  // Free all memory
  numa_free(mem_src, size);
  numa_free(mem_dst, size);

  return 1;
}

int main() {
  size_t size = 1024ull * 1024ull * 1024ull;
  int runs = 10;
  uint64_t *nanos = new uint64_t[runs];

  // Warm-up run
  numa_copy(16, size, 0, 0, &nanos[0]);

  // Real run
  uint64_t sum = 0;
  for (int r = 1; r < runs; ++r) {
    numa_copy(16, size, 0, 0, &nanos[r]);
    sum += nanos[r];
  }

  uint64_t avg = sum / (runs - 1);

  // Report throughput (not equals to bandwidth!)
  std::cout << (size / 1024ull / 1024ull / 1024ull) / (((double)avg) / 1.e9)
            << std::endl;
}
