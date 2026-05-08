#include "util/timer.h"

#include <chrono>

namespace mosaicvram {

Timer::Timer() { Reset(); }

void Timer::Reset() { start_ = Clock::now(); }

double Timer::ElapsedMs() const {
  const std::chrono::duration<double, std::milli> elapsed =
      Clock::now() - start_;
  return elapsed.count();
}

}  // namespace mosaicvram
