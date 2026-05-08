#ifndef MOSAICVRAM_SRC_CUDA_CUDA_ERROR_H_
#define MOSAICVRAM_SRC_CUDA_CUDA_ERROR_H_

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace mosaicvram {

class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t status, std::string message);

  cudaError_t status() const;

 private:
  cudaError_t status_;
};

std::string FormatCudaError(cudaError_t status, std::string_view action,
                            std::string_view file, int line);

void CheckCuda(cudaError_t status, std::string_view action,
               std::string_view file, int line);

}  // namespace mosaicvram

#define MOSAICVRAM_CUDA_CHECK(status, action) \
  ::mosaicvram::CheckCuda((status), (action), __FILE__, __LINE__)

#endif  // MOSAICVRAM_SRC_CUDA_CUDA_ERROR_H_
