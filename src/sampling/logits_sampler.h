#ifndef MOSAICVRAM_SRC_SAMPLING_LOGITS_SAMPLER_H_
#define MOSAICVRAM_SRC_SAMPLING_LOGITS_SAMPLER_H_

#include <cstdint>
#include <random>
#include <span>

namespace mosaicvram {

struct SamplingOptions {
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.95f;
  float min_p = 0.05f;
  float repeat_penalty = 1.1f;
  uint32_t seed = 0;
};

class LogitsSampler {
 public:
  explicit LogitsSampler(SamplingOptions options);

  void Reset(uint32_t seed);
  int64_t Sample(std::span<const float> logits,
                 std::span<const int64_t> recent_tokens);
  const SamplingOptions& options() const { return options_; }

 private:
  int64_t Greedy(std::span<const float> logits) const;

  SamplingOptions options_;
  std::mt19937 generator_;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_SAMPLING_LOGITS_SAMPLER_H_
