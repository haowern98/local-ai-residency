#ifndef MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_ADAPTER_H_
#define MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_ADAPTER_H_

#include "reload/reload_mode.h"
#include "residency/backend_state_snapshot.h"
#include "residency/residency_state.h"

namespace mosaicvram {

class BackendStateAdapter {
 public:
  virtual ~BackendStateAdapter() = default;

  virtual BackendStateSnapshot& SaveState() = 0;
  virtual void EvictContext() = 0;
  virtual void EvictModel() = 0;
  virtual void ReloadModel() = 0;
  virtual void RestoreState() = 0;
  virtual bool ResumeCheck() = 0;
  virtual ResidencyState residency_state() const = 0;
  virtual MosaicSessionId session_id() const = 0;

  virtual ReloadMode reload_mode() const = 0;
  virtual void set_reload_policy(ReloadPolicy policy) = 0;
  virtual const ReloadPolicy& reload_policy() const = 0;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_ADAPTER_H_
