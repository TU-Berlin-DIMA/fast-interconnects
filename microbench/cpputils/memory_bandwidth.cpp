#include <atomic>
#include <cstdint>

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
 *  - None
 */
extern "C" void cpu_read_bandwidth_seq(uint32_t *data, std::size_t size,
                                       std::size_t tid,
                                       std::size_t num_threads) {
  std::size_t const chunk_size = (size + num_threads - 1) / num_threads;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;

  uint32_t dummy = 0;
  for (std::size_t i = begin; i < end; ++i) {
    dummy += data[i];
  }

  // Prevent compiler optimization
  if (dummy == 0) {
    data[0] = dummy;
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
 *  - All array elements are filled with unspecified data
 */
extern "C" void cpu_write_bandwidth_seq(uint32_t *data, std::size_t size,
                                        std::size_t tid,
                                        std::size_t num_threads) {
  std::size_t const chunk_size = (size + num_threads - 1) / num_threads;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;

  for (std::size_t i = begin; i < end; ++i) {
    data[i] = i;
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
 *  - All array elements are filled with unspecified data
 */
extern "C" void cpu_cas_bandwidth_seq(uint32_t *data, std::size_t size,
                                      std::size_t tid,
                                      std::size_t num_threads) {
  std::size_t const chunk_size = (size + num_threads - 1) / num_threads;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;

  for (std::size_t i = begin; i < end; ++i) {
    auto *item = reinterpret_cast<std::atomic<uint32_t> *>(&data[i]);
    uint32_t expected = (uint32_t)i;
    std::atomic_compare_exchange_strong(item, &expected, (uint32_t)i + 1);
  }
}

/*
 * Run a sequential bandwidth test
 *
 * See specific functions for pre- and postcondition details.
 */
extern "C" void cpu_bandwidth_seq(MemoryOperation op, uint32_t *data,
                                  std::size_t size, std::size_t tid,
                                  std::size_t num_threads) {
  switch (op) {
    case Read:
      cpu_read_bandwidth_seq(data, size, tid, num_threads);
      break;
    case Write:
      cpu_write_bandwidth_seq(data, size, tid, num_threads);
      break;
    case CompareAndSwap:
      cpu_cas_bandwidth_seq(data, size, tid, num_threads);
      break;
    default:
      throw "Unimplemented operation!";
  }
}

/*
 * Test random read bandwidth
 *
 * Read #size elements from array. Random memory locations are generated using
 * an efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - None
 */
extern "C" void cpu_read_bandwidth_lcg(uint32_t *data, std::size_t size,
                                       std::size_t tid,
                                       std::size_t num_threads) {
  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tid;

  // Do measurement
  uint32_t dummy = 0;
  for (std::size_t i = 0; i < size / num_threads; ++i) {
    // Generate next random number with LCG
    // Note: wrap modulo 2^64 is defined by C/C++ standard
    x = a * x + c;

    // Read from a random location within data range
    uint64_t index = FAST_MODULO(x, size);
    dummy += data[index];
  }

  // Prevent compiler optimization
  if (dummy == 0) {
    data[0] = dummy;
  }
}

/*
 * Test random write bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using
 * an efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - All array elements are (probably) filled with random numbers
 */
extern "C" void cpu_write_bandwidth_lcg(uint32_t *data, std::size_t size,
                                        std::size_t tid,
                                        std::size_t num_threads) {
  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tid;

  // Do measurement
  for (std::size_t i = 0; i < size / num_threads; ++i) {
    // Generate next random number with LCG
    // Note: wrap modulo 2^64 is defined by C/C++ standard
    x = a * x + c;

    // Write to a random location within data range
    uint64_t index = FAST_MODULO(x, size);
    data[index] = x;
  }
}

/*
 * Test random CompareAndSwap bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using
 * an efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - All array elements are filled with unspecified data
 */
extern "C" void cpu_cas_bandwidth_lcg(uint32_t *data, std::size_t size,
                                      std::size_t tid,
                                      std::size_t num_threads) {
  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tid;

  // Do measurement
  for (std::size_t i = 0; i < size / num_threads; ++i) {
    // Generate next random number with LCG
    // Note: wrap modulo 2^64 is defined by C/C++ standard
    x = a * x + c;

    // Write to a random location within data range
    uint64_t index = FAST_MODULO(x, size);
    auto *item = reinterpret_cast<std::atomic<uint32_t> *>(&data[index]);
    uint32_t expected = (uint32_t)index;
    uint32_t new_val = (uint32_t)x;
    std::atomic_compare_exchange_strong(item, &expected, new_val);
  }
}

/*
 * Run a random bandwidth test
 *
 * See specific functions for pre- and postcondition details.
 */
extern "C" void cpu_bandwidth_lcg(MemoryOperation op, uint32_t *data,
                                  std::size_t size, std::size_t tid,
                                  std::size_t num_threads) {
  switch (op) {
    case Read:
      cpu_read_bandwidth_lcg(data, size, tid, num_threads);
      break;
    case Write:
      cpu_write_bandwidth_lcg(data, size, tid, num_threads);
      break;
    case CompareAndSwap:
      cpu_cas_bandwidth_lcg(data, size, tid, num_threads);
      break;
    default:
      throw "Unimplemented operation!";
  }
}
