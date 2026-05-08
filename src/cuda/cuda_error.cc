#include "cuda/cuda_error.h"

#include <sstream>
#include <utility>

namespace mosaicvram {

CudaError::CudaError(cudaError_t status, std::string message)
    : std::runtime_error(std::move(message)), status_(status) {}

cudaError_t CudaError::status() const { return status_; }

std::string FormatCudaError(cudaError_t status, std::string_view action,
                            std::string_view file, int line) {
  std::ostringstream message;
  message << action << " failed at " << file << ":" << line << ": "
          << cudaGetErrorName(status) << " (" << cudaGetErrorString(status)
          << ")";
  return message.str();
}

void CheckCuda(cudaError_t status, std::string_view action,
               std::string_view file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  throw CudaError(status, FormatCudaError(status, action, file, line));
}

}  // namespace mosaicvram
