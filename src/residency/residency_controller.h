#ifndef MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_CONTROLLER_H_
#define MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_CONTROLLER_H_

#include <map>
#include <string>

#include "residency/backend_state_adapter.h"

namespace mosaicvram {

/**
 * Result returned by ResidencyController lifecycle operations.
 */
struct ResidencyControllerResult {
  bool ok = false;
  std::string error_message;
};

/**
 * Coordinates residency lifecycle operations across backend adapters.
 *
 * The controller owns ordering and GPU ownership decisions, but it does not
 * inspect backend-specific state. Each session is manipulated through the
 * BackendStateAdapter contract.
 */
class ResidencyController {
 public:
  ResidencyController() = default;

  ResidencyController(const ResidencyController&) = delete;
  ResidencyController& operator=(const ResidencyController&) = delete;

  /**
   * Registers a session adapter that will be addressed by its session_id().
   */
  ResidencyControllerResult RegisterAdapter(BackendStateAdapter* adapter);
  /**
   * Saves a session before residency is released.
   */
  ResidencyControllerResult SaveSession(MosaicSessionId session_id);
  /**
   * Evicts context-level residency for a saved session.
   */
  ResidencyControllerResult EvictContext(MosaicSessionId session_id);
  /**
   * Evicts model-level residency for a saved session.
   */
  ResidencyControllerResult EvictModel(MosaicSessionId session_id);
  /**
   * Reloads model residency before restore.
   */
  ResidencyControllerResult ReloadModel(MosaicSessionId session_id);
  /**
   * Restores the last saved state for a session.
   */
  ResidencyControllerResult RestoreSession(MosaicSessionId session_id);
  /**
   * Checks that restored execution continues from saved state.
   */
  ResidencyControllerResult ResumeCheck(MosaicSessionId session_id);
  /**
   * Moves GPU ownership from one registered session to another.
   */
  ResidencyControllerResult SwitchGpuOwner(MosaicSessionId from_session_id,
                                           MosaicSessionId to_session_id);

  /**
   * Returns the session currently considered the GPU owner.
   */
  MosaicSessionId gpu_owner() const { return gpu_owner_; }
  /**
   * Returns a registered session's residency state.
   */
  ResidencyState ResidencyOf(MosaicSessionId session_id) const;

 private:
  BackendStateAdapter* FindAdapter(MosaicSessionId session_id) const;
  static ResidencyControllerResult Ok();
  static ResidencyControllerResult Error(const std::string& message);

  std::map<MosaicSessionId, BackendStateAdapter*> adapters_;
  MosaicSessionId gpu_owner_ = 0;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_CONTROLLER_H_
