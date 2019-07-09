#include <common.h>
#include <helper.h>

#include <cstdint>

__global__ void gpu_loop_kernel(volatile CacheLine *data, uint32_t iterations, volatile Signal *signal, volatile uint64_t *result)
{
    uint64_t sum = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    // Wait for start signal
    while (*signal == WAIT);

    // Run
    for (uint32_t i = 0; i < iterations; ++i) {

        start = clock64();

        while (data->value == GPU);
        data->value = GPU;
        /* __threadfence_system(); */

        stop = clock64();
        sum += stop - start;

    }

    // Write average result
    *result = (sum / iterations);
    /* __threadfence_system(); */
}

extern "C" void gpu_loop(CacheLine *data, uint32_t iterations, Signal *signal, uint64_t *result) {
    gpu_loop_kernel<<<1,1>>>(data, iterations, signal, result);
}

extern "C" void gpu_loop_sync() {
    CHECK_CUDA(cudaDeviceSynchronize());
}
