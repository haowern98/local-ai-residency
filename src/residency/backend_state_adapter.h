#ifndef MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_ADAPTER_H_
#define MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_ADAPTER_H_

#include "residency/backend_state_snapshot.h"
#include "residency/residency_state.h"

namespace mosaicvram {

/**
 * Backend-neutral lifecycle interface for GPU residency handoff.
 *
 * Implementations wrap backend-specific runtimes such as llama.cpp or ONNX
 * Runtime while exposing the same save, evict, reload, restore, and resume
 * checks to ResidencyController.
 */
class BackendStateAdapter {
 public:
  virtual ~BackendStateAdapter() = default;

  /**
   * Captures enough backend state to resume without replaying the prompt.
   *
   * The returned snapshot remains owned by the adapter because its storage and
   * interpretation are backend-specific.
   */
  virtual BackendStateSnapshot& SaveState() = 0;
  /**
   * Releases context-level residency while keeping the model reloadable.
   */
  virtual void EvictContext() = 0;
  /**
   * Releases model-level GPU residency after state has been saved.
   */
  virtual void EvictModel() = 0;
  /**
   * Reloads model residency before saved state is restored.
   */
  virtual void ReloadModel() = 0;
  /**
   * Restores the adapter's last saved backend snapshot.
   */
  virtual void RestoreState() = 0;
  /**
   * Validates that restored execution resumes from the saved state.
   */
  virtual bool ResumeCheck() = 0;
  /**
   * Returns the adapter's current residency state.
   */
  virtual ResidencyState residency_state() const = 0;
  /**
   * Returns the controller-visible session identifier for this adapter.
   */
  virtual MosaicSessionId session_id() const = 0;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_ADAPTER_H_
