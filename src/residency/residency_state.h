#ifndef MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_STATE_H_
#define MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_STATE_H_

#include <cstdint>

namespace mosaicvram {

using MosaicSessionId = std::uint64_t;

enum class ResidencyState {
  kUnloaded,
  kResident,
  kContextEvicted,
  kModelEvicted,
  kRestoring,
};

inline const char* ResidencyStateName(ResidencyState state) {
  switch (state) {
    case ResidencyState::kUnloaded:
      return "Unloaded";
    case ResidencyState::kResident:
      return "Resident";
    case ResidencyState::kContextEvicted:
      return "ContextEvicted";
    case ResidencyState::kModelEvicted:
      return "ModelEvicted";
    case ResidencyState::kRestoring:
      return "Restoring";
  }
  return "Unknown";
}

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_STATE_H_
