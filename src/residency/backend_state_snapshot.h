#ifndef MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_SNAPSHOT_H_
#define MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_SNAPSHOT_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/cuda_error.h"
#include "residency/residency_state.h"

namespace mosaicvram {

/**
 * Pinned host allocation used as a staging area for backend state.
 *
 * State snapshots can outlive GPU residency. Pinned host memory keeps copies
 * explicit and efficient for CUDA-backed adapters.
 */
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  ~PinnedHostBuffer() { Free(); }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept { MoveFrom(&other); }

  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      MoveFrom(&other);
    }
    return *this;
  }

  void Allocate(std::size_t size) {
    Free();
    if (size == 0) {
      return;
    }
    MOSAICVRAM_CUDA_CHECK(
        cudaMallocHost(reinterpret_cast<void**>(&data_), size),
        "allocate pinned host state buffer");
    size_ = size;
  }

  void Free() {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  std::uint8_t* data() { return data_; }
  const std::uint8_t* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return data_ == nullptr || size_ == 0; }

 private:
  void MoveFrom(PinnedHostBuffer* other) {
    data_ = other->data_;
    size_ = other->size_;
    other->data_ = nullptr;
    other->size_ = 0;
  }

  std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
};

/**
 * Backend-owned state captured before residency is evicted.
 *
 * The controller treats this as opaque data. Individual adapters decide whether
 * full-state and/or sequence-state buffers are meaningful for their runtime.
 */
struct BackendStateSnapshot {
  /// Session that produced this snapshot.
  MosaicSessionId session_id = 0;
  /// Human-readable backend name for reporting and diagnostics.
  std::string backend_name;
  /// Residency state at the time the snapshot was captured.
  ResidencyState source_residency = ResidencyState::kUnloaded;
  /// Full backend state, when the backend exposes one.
  PinnedHostBuffer full_state;
  /// Sequence-specific state, when the backend exposes one.
  PinnedHostBuffer sequence_state;
  /// Number of bytes captured in full_state.
  std::size_t full_state_bytes = 0;
  /// Number of bytes captured in sequence_state.
  std::size_t sequence_state_bytes = 0;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_SNAPSHOT_H_
