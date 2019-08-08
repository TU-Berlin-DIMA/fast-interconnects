#include <helper.h>

#include <cstdint>
#include <cuda.h>

// X mod Y, assuming that Y is a power of 2
#define FAST_MODULO(X, Y) (X & (Y - 1))

enum MemoryOperation { Read, Write, CompareAndSwap };

/*
 * Test sequential read bandwidth
 *
 * Read #size elements from array.
 *
 * Preconditions:
 *  - None
 *
 * Postconditions:
 *  - Clock cycles are written to cycles
 */
__global__ void gpu_read_bandwidth_seq_kernel(uint32_t *data, uint32_t size, uint64_t *cycles) {
    uint32_t const global_size = gridDim.x * blockDim.x;
    uint32_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    start = clock64();

    uint32_t dummy = 0;
    for (uint32_t i = gid; i < size; i += global_size) {
        dummy += data[i];
    }

    stop = clock64();
    sum = stop - start;

    // Write result
    *cycles = sum;

    // Prevent compiler optimization
    if (sum == 0) {
        data[1] = dummy;
    }
}

/*
 * Test sequential write bandwidth
 *
 * Write #size elements to array.
 *
 * Preconditions:
 *  - None
 *
 * Postconditions:
 *  - Clock cycles are written to cycles
 *  - All array elements are filled with unspecified data
 */
__global__ void gpu_write_bandwidth_seq_kernel(uint32_t *data, uint32_t size, uint64_t *cycles) {
    uint32_t const global_size = gridDim.x * blockDim.x;
    uint32_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    start = clock64();

    for (uint32_t i = gid; i < size; i += global_size) {
        data[i] = i;
    }

    stop = clock64();
    sum = stop - start;

    // Write result
    *cycles = sum;

    // Prevent compiler optimization
    if (sum == 0) {
        data[1] = sum;
    }
}

/*
 * Test sequential CompareAndSwap bandwidth
 *
 * Write #size elements to array.
 *
 * Preconditions:
 *  - None
 *
 * Postconditions:
 *  - Clock cycles are written to cycles
 *  - All array elements are filled with unspecified data
 */
__global__ void gpu_cas_bandwidth_seq_kernel(uint32_t *data, uint32_t size, uint64_t *cycles) {
    uint32_t const global_size = gridDim.x * blockDim.x;
    uint32_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    start = clock64();

    for (uint32_t i = gid; i < size; i += global_size) {
        atomicCAS(&data[i], i, i + 1);
    }

    stop = clock64();
    sum = stop - start;

    // Write result
    *cycles = sum;
}

/*
 * Run a sequential bandwidth test
 *
 * See specific functions for pre- and postcondition details.
 */
extern "C" void gpu_bandwidth_seq(
        MemoryOperation op,
        uint32_t *data,
        uint32_t size,
        uint64_t *cycles,
        uint32_t grid,
        uint32_t block,
        CUstream stream
        )
{
    switch (op) {
        case Read:
            gpu_read_bandwidth_seq_kernel<<<grid, block, 0, stream>>>(data, size, cycles);
            break;
        case Write:
            gpu_write_bandwidth_seq_kernel<<<grid, block, 0, stream>>>(data, size, cycles);
            break;
        case CompareAndSwap:
            gpu_cas_bandwidth_seq_kernel<<<grid, block, 0, stream>>>(data, size, cycles);
            break;
        default:
            throw "Unimplemented operation!";
    }
}

/*
 * Test random read bandwidth
 *
 * Read #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - Clock cycles are written to cycles
 */
__global__ void gpu_read_bandwidth_lcg_kernel(uint32_t *data, uint32_t size, uint64_t *cycles) {
    uint32_t global_size = gridDim.x * blockDim.x;
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    // Linear congruent generator
    // Parameters according to Glibc
    // See: https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/random_r.c;hb=glibc-2.28#l364
    // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
    uint32_t a = 1103515245U;
    uint32_t c = 12345U;
    uint32_t m = 0x7fffffffU;
    uint32_t x = 67890U + gid;

    start = clock64();

    // Do measurement
    uint32_t dummy = 0;
    for (uint32_t i = gid; i < size; i += global_size) {
        // Generate next random number with LCG
        x = ((a * x + c) & m);

        // Read from a random location within data range
        uint32_t location = FAST_MODULO(x, size);
        dummy += data[location];
    }

    stop = clock64();
    sum = stop - start;

    // Write result
    *cycles = sum;

    // Prevent compiler optimization
    if (sum == 0) {
        data[1] = dummy;
    }
}

/*
 * Test random write bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - Clock cycles are written to data[0]
 *  - All other array elements are (probably) filled with random numbers
 */
__global__ void gpu_write_bandwidth_lcg_kernel(uint32_t *data, uint32_t size, uint64_t *cycles) {
    uint32_t global_size = gridDim.x * blockDim.x;
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    // Linear congruent generator
    // Parameters according to Glibc
    // See: https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/random_r.c;hb=glibc-2.28#l364
    // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
    uint32_t a = 1103515245U;
    uint32_t c = 12345U;
    uint32_t m = 0x7fffffffU;
    uint32_t x = 67890U + gid;

    start = clock64();

    // Do measurement
    for (uint32_t i = gid; i < size; i += global_size) {
        // Generate next random number with LCG
        x = ((a * x + c) & m);

        // Write to a random location within data range
        uint32_t location = FAST_MODULO(x, size);
        data[location] = x;
    }

    stop = clock64();
    sum = stop - start;

    // Write result
    *cycles = sum;

    // Prevent compiler optimization
    if (sum == 0) {
        data[1] = sum;
    }
}

/*
 * Test random CompareAndSwap bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - Clock cycles are written to data[0]
 *  - All array elements are filled with unspecified data
 */
__global__ void gpu_cas_bandwidth_lcg_kernel(uint32_t *data, uint32_t size, uint64_t *cycles) {
    uint32_t global_size = gridDim.x * blockDim.x;
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    // Linear congruent generator
    // Parameters according to Glibc
    // See: https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/random_r.c;hb=glibc-2.28#l364
    // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
    uint32_t a = 1103515245U;
    uint32_t c = 12345U;
    uint32_t m = 0x7fffffffU;
    uint32_t x = 67890U + gid;

    start = clock64();

    // Do measurement
    for (uint32_t i = gid; i < size; i += global_size) {
        // Generate next random number with LCG
        x = ((a * x + c) & m);

        // Write to a random location within data range
        uint32_t location = FAST_MODULO(x, size);
        atomicCAS(&data[location], location, x);
    }

    stop = clock64();
    sum = stop - start;

    // Write result
    *cycles = sum;
}

/*
 * Run a random bandwidth test
 *
 * See specific functions for pre- and postcondition details.
 */
extern "C" void gpu_bandwidth_lcg(
        MemoryOperation op,
        uint32_t *data,
        uint32_t size,
        uint64_t *cycles,
        uint32_t grid,
        uint32_t block,
        CUstream stream
        )
{
    switch (op) {
        case Read:
            gpu_read_bandwidth_lcg_kernel<<<grid, block, 0, stream>>>(data, size, cycles);
            break;
        case Write:
            gpu_write_bandwidth_lcg_kernel<<<grid, block, 0, stream>>>(data, size, cycles);
            break;
        case CompareAndSwap:
            gpu_cas_bandwidth_lcg_kernel<<<grid, block, 0, stream>>>(data, size, cycles);
            break;
        default:
            throw "Unimplemented operation!";
    }
}
