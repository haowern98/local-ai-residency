#include "cuda/cuda_stream.h"

#include <utility>

#include "cuda/cuda_error.h"

namespace mosaicvram {

CudaStream::CudaStream() {
  MOSAICVRAM_CUDA_CHECK(
      cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
      "cudaStreamCreateWithFlags");
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
  stream_ = std::exchange(other.stream_, nullptr);
  return *this;
}

cudaStream_t CudaStream::get() const { return stream_; }

void CudaStream::Synchronize() const {
  MOSAICVRAM_CUDA_CHECK(cudaStreamSynchronize(stream_),
                        "cudaStreamSynchronize");
}

}  // namespace mosaicvram
