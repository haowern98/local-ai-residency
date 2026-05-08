#ifndef MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_SNAPSHOT_H_
#define MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_SNAPSHOT_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/cuda_error.h"
#include "residency/residency_state.h"

namespace mosaicvram {

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

struct BackendStateSnapshot {
  MosaicSessionId session_id = 0;
  std::string backend_name;
  ResidencyState source_residency = ResidencyState::kUnloaded;
  PinnedHostBuffer full_state;
  PinnedHostBuffer sequence_state;
  std::size_t full_state_bytes = 0;
  std::size_t sequence_state_bytes = 0;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RESIDENCY_BACKEND_STATE_SNAPSHOT_H_
