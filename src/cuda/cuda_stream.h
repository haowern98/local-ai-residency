#ifndef MOSAICVRAM_SRC_CUDA_CUDA_STREAM_H_
#define MOSAICVRAM_SRC_CUDA_CUDA_STREAM_H_

#include <cuda_runtime_api.h>

namespace mosaicvram {

class CudaStream {
 public:
  CudaStream();
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  cudaStream_t get() const;
  void Synchronize() const;

 private:
  cudaStream_t stream_ = nullptr;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_CUDA_CUDA_STREAM_H_
