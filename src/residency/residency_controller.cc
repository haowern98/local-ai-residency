#include "residency/residency_controller.h"

#include <string>

namespace mosaicvram {

ResidencyControllerResult ResidencyController::RegisterAdapter(
    BackendStateAdapter* adapter) {
  if (adapter == nullptr) {
    return Error("cannot register null residency adapter");
  }
  const MosaicSessionId session_id = adapter->session_id();
  if (session_id == 0) {
    return Error("cannot register adapter with session id 0");
  }
  if (adapters_.find(session_id) != adapters_.end()) {
    return Error("duplicate residency adapter session id");
  }
  adapters_[session_id] = adapter;
  if (gpu_owner_ == 0 &&
      adapter->residency_state() == ResidencyState::kResident) {
    gpu_owner_ = session_id;
  }
  return Ok();
}

ResidencyControllerResult ResidencyController::SaveSession(
    MosaicSessionId session_id) {
  BackendStateAdapter* adapter = FindAdapter(session_id);
  if (adapter == nullptr) {
    return Error("unknown session id");
  }
  adapter->SaveState();
  return Ok();
}

ResidencyControllerResult ResidencyController::EvictContext(
    MosaicSessionId session_id) {
  BackendStateAdapter* adapter = FindAdapter(session_id);
  if (adapter == nullptr) {
    return Error("unknown session id");
  }
  adapter->EvictContext();
  if (gpu_owner_ == session_id) {
    gpu_owner_ = 0;
  }
  return Ok();
}

ResidencyControllerResult ResidencyController::EvictModel(
    MosaicSessionId session_id) {
  BackendStateAdapter* adapter = FindAdapter(session_id);
  if (adapter == nullptr) {
    return Error("unknown session id");
  }
  adapter->EvictModel();
  if (gpu_owner_ == session_id) {
    gpu_owner_ = 0;
  }
  return Ok();
}

ResidencyControllerResult ResidencyController::ReloadModel(
    MosaicSessionId session_id) {
  BackendStateAdapter* adapter = FindAdapter(session_id);
  if (adapter == nullptr) {
    return Error("unknown session id");
  }
  adapter->ReloadModel();
  return Ok();
}

ResidencyControllerResult ResidencyController::RestoreSession(
    MosaicSessionId session_id) {
  BackendStateAdapter* adapter = FindAdapter(session_id);
  if (adapter == nullptr) {
    return Error("unknown session id");
  }
  adapter->RestoreState();
  gpu_owner_ = session_id;
  return Ok();
}

ResidencyControllerResult ResidencyController::ResumeCheck(
    MosaicSessionId session_id) {
  BackendStateAdapter* adapter = FindAdapter(session_id);
  if (adapter == nullptr) {
    return Error("unknown session id");
  }
  if (!adapter->ResumeCheck()) {
    return Error("session resume check failed");
  }
  gpu_owner_ = session_id;
  return Ok();
}

ResidencyControllerResult ResidencyController::SwitchGpuOwner(
    MosaicSessionId from_session_id, MosaicSessionId to_session_id) {
  BackendStateAdapter* from_adapter = FindAdapter(from_session_id);
  BackendStateAdapter* to_adapter = FindAdapter(to_session_id);
  if (from_adapter == nullptr || to_adapter == nullptr) {
    return Error("unknown session id");
  }
  if (from_session_id == to_session_id) {
    gpu_owner_ = to_session_id;
    return Ok();
  }

  from_adapter->SaveState();
  from_adapter->EvictModel();
  to_adapter->ReloadModel();
  to_adapter->RestoreState();
  gpu_owner_ = to_session_id;
  return Ok();
}

ResidencyState ResidencyController::ResidencyOf(
    MosaicSessionId session_id) const {
  BackendStateAdapter* adapter = FindAdapter(session_id);
  if (adapter == nullptr) {
    return ResidencyState::kUnloaded;
  }
  return adapter->residency_state();
}

BackendStateAdapter* ResidencyController::FindAdapter(
    MosaicSessionId session_id) const {
  const auto it = adapters_.find(session_id);
  if (it == adapters_.end()) {
    return nullptr;
  }
  return it->second;
}

ResidencyControllerResult ResidencyController::Ok() {
  return ResidencyControllerResult{true, ""};
}

ResidencyControllerResult ResidencyController::Error(
    const std::string& message) {
  return ResidencyControllerResult{false, message};
}

}  // namespace mosaicvram
