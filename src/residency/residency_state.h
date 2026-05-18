#ifndef MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_STATE_H_
#define MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_STATE_H_

#include <cstdint>

namespace mosaicvram {

/// Stable identifier used by ResidencyController to address a backend session.
using MosaicSessionId = std::uint64_t;

/**
 * Residency state for a backend session.
 *
 * The values describe whether model/context resources are present on the GPU
 * and whether the adapter is in the middle of a restore transition.
 */
enum class ResidencyState {
  /// No model or context residency has been created.
  kUnloaded,
  /// Model and active execution state are resident.
  kResident,
  /// Context state was evicted while model residency may remain available.
  kContextEvicted,
  /// Model residency was evicted after state was saved.
  kModelEvicted,
  /// Saved state is being restored into resident backend resources.
  kRestoring,
};

/**
 * Returns the stable display name for a residency state.
 */
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
