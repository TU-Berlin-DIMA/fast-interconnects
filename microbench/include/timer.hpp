// Copyright 2016-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <cstdint>

// Check if we are using the syscall, as it is slow
// We want to use clock_gettime()
// If it doesn't work, try setting CPP_FLAGS "--enable-libstdcxx-time=yes"
// This enables link-time checking for clock_gettime() in GCC/G++
// Checks taken from https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63400#c1
#ifdef _GLIBCXX_USE_CLOCK_REALTIME
#ifdef _GLIBCXX_USE_CLOCK_GETTIME_SYSCALL
#warning system_clock using syscall(SYS_clock_gettime, CLOCK_REALTIME, &tp);
#else
//#warning system_clock using clock_gettime(CLOCK_REALTIME, &tp);
#endif
#elif defined(_GLIBCXX_USE_GETTIMEOFDAY)
#warning system_clock using gettimeofday(&tv, 0);
#else
#warning system_clock using std::time(0);
#endif

#ifdef _GLIBCXX_USE_CLOCK_MONOTONIC
#ifdef _GLIBCXX_USE_CLOCK_GETTIME_SYSCALL
#warning steady_clock using syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &tp);
#else
//#warning steady_clock using clock_gettime(CLOCK_MONOTONIC, &tp);
#endif
#else
#warning steady_clock using time_point(system_clock::now().time_since_epoch());
#endif

namespace Timer {

// Timer for benchmarking
// Header-only implementation for compiler inlining
// Verified that G++ with -O2 optimization inlines these functions
class Timer {
 public:
  void start() { start_epoch_ = std::chrono::steady_clock::now(); }

  template <typename UnitT = std::chrono::nanoseconds>
  uint64_t stop() {
    std::chrono::steady_clock::time_point stop_epoch;
    UnitT time_span;

    stop_epoch = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<UnitT>(stop_epoch - start_epoch_);
    return time_span.count();
  }

 private:
  std::chrono::steady_clock::time_point start_epoch_;
};

}  // namespace Timer

#endif /* TIMER_HPP_ */
