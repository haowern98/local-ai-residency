#ifndef MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_CONTROLLER_H_
#define MOSAICVRAM_SRC_RESIDENCY_RESIDENCY_CONTROLLER_H_

#include <map>
#include <string>

#include "residency/backend_state_adapter.h"

namespace mosaicvram {

struct ResidencyControllerResult {
  bool ok = false;
  std::string error_message;
};

class ResidencyController {
 public:
  ResidencyController() = default;

  ResidencyController(const ResidencyController&) = delete;
  ResidencyController& operator=(const ResidencyController&) = delete;

  ResidencyControllerResult RegisterAdapter(BackendStateAdapter* adapter);
  ResidencyControllerResult UnregisterAdapter(MosaicSessionId session_id);
  ResidencyControllerResult SaveSession(MosaicSessionId session_id);
  ResidencyControllerResult EvictContext(MosaicSessionId session_id);
  ResidencyControllerResult EvictModel(MosaicSessionId session_id);
  ResidencyControllerResult ReloadModel(MosaicSessionId session_id);
  ResidencyControllerResult RestoreSession(MosaicSessionId session_id);
  ResidencyControllerResult ResumeCheck(MosaicSessionId session_id);
  ResidencyControllerResult SwitchGpuOwner(MosaicSessionId from_session_id,
                                           MosaicSessionId to_session_id);

  MosaicSessionId gpu_owner() const { return gpu_owner_; }
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
