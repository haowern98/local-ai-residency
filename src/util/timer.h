#ifndef MOSAICVRAM_SRC_UTIL_TIMER_H_
#define MOSAICVRAM_SRC_UTIL_TIMER_H_

#include <chrono>

namespace mosaicvram {

class Timer {
 public:
  Timer();

  void Reset();
  double ElapsedMs() const;

 private:
  using Clock = std::chrono::steady_clock;

  Clock::time_point start_;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_UTIL_TIMER_H_
