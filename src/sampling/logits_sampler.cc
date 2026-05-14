#include "sampling/logits_sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mosaicvram {
namespace {

struct Candidate {
  int64_t token = 0;
  float logit = 0.0f;
  double probability = 0.0;
};

bool BetterLogit(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.logit == rhs.logit) {
    return lhs.token < rhs.token;
  }
  return lhs.logit > rhs.logit;
}

float ApplyRepeatPenalty(float logit, float penalty) {
  if (penalty <= 0.0f || penalty == 1.0f) {
    return logit;
  }
  if (logit < 0.0f) {
    return logit * penalty;
  }
  return logit / penalty;
}

uint32_t DefaultSeed() {
  std::random_device device;
  return device();
}

}  // namespace

LogitsSampler::LogitsSampler(SamplingOptions options) : options_(options) {
  Reset(options_.seed);
}

void LogitsSampler::Reset(uint32_t seed) {
  generator_.seed(seed == 0 ? DefaultSeed() : seed);
}

int64_t LogitsSampler::Greedy(std::span<const float> logits) const {
  if (logits.empty()) {
    throw std::runtime_error("cannot sample from empty logits");
  }

  int64_t best_token = -1;
  float best_logit = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float logit = logits[i];
    if (!std::isfinite(logit)) {
      continue;
    }
    if (best_token < 0 || logit > best_logit) {
      best_token = static_cast<int64_t>(i);
      best_logit = logit;
    }
  }
  if (best_token < 0) {
    throw std::runtime_error("cannot sample because all logits are non-finite");
  }
  return best_token;
}

int64_t LogitsSampler::Sample(std::span<const float> logits,
                              std::span<const int64_t> recent_tokens) {
  if (logits.empty()) {
    throw std::runtime_error("cannot sample from empty logits");
  }
  if (options_.temperature <= 0.0f) {
    return Greedy(logits);
  }

  std::unordered_set<int64_t> repeated_tokens;
  repeated_tokens.reserve(recent_tokens.size());
  for (const int64_t token : recent_tokens) {
    if (token >= 0 && static_cast<std::size_t>(token) < logits.size()) {
      repeated_tokens.insert(token);
    }
  }

  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    float logit = logits[i];
    if (!std::isfinite(logit)) {
      continue;
    }
    if (repeated_tokens.contains(static_cast<int64_t>(i))) {
      logit = ApplyRepeatPenalty(logit, options_.repeat_penalty);
    }
    candidates.push_back(Candidate{static_cast<int64_t>(i), logit, 0.0});
  }
  if (candidates.empty()) {
    throw std::runtime_error("cannot sample because all logits are non-finite");
  }

  if (options_.top_k > 0 &&
      static_cast<std::size_t>(options_.top_k) < candidates.size()) {
    const auto keep_end = candidates.begin() + options_.top_k;
    std::nth_element(candidates.begin(), keep_end, candidates.end(),
                     BetterLogit);
    candidates.erase(keep_end, candidates.end());
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  for (const Candidate& candidate : candidates) {
    max_logit = std::max(max_logit, candidate.logit);
  }
  double probability_sum = 0.0;
  for (Candidate& candidate : candidates) {
    candidate.probability =
        std::exp((candidate.logit - max_logit) / options_.temperature);
    probability_sum += candidate.probability;
  }
  if (probability_sum <= 0.0 || !std::isfinite(probability_sum)) {
    return Greedy(logits);
  }

  for (Candidate& candidate : candidates) {
    candidate.probability /= probability_sum;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
              if (lhs.probability == rhs.probability) {
                return lhs.token < rhs.token;
              }
              return lhs.probability > rhs.probability;
            });

  if (options_.min_p > 0.0f && !candidates.empty()) {
    const double cutoff = candidates.front().probability * options_.min_p;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [cutoff](const Candidate& candidate) {
                                      return candidate.probability < cutoff;
                                    }),
                     candidates.end());
  }

  if (options_.top_p > 0.0f && options_.top_p < 1.0f && !candidates.empty()) {
    double cumulative = 0.0;
    std::size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
      cumulative += candidates[keep].probability;
      if (cumulative >= options_.top_p) {
        ++keep;
        break;
      }
    }
    candidates.resize(std::max<std::size_t>(keep, 1));
  }

  if (candidates.empty()) {
    return Greedy(logits);
  }

  std::vector<double> weights;
  weights.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    weights.push_back(candidate.probability);
  }

  std::discrete_distribution<std::size_t> distribution(weights.begin(),
                                                       weights.end());
  const std::size_t selected = distribution(generator_);
  return candidates[selected].token;
}

}  // namespace mosaicvram
