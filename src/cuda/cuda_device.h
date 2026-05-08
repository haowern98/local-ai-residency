#ifndef MOSAICVRAM_SRC_CUDA_CUDA_DEVICE_H_
#define MOSAICVRAM_SRC_CUDA_CUDA_DEVICE_H_

#include <cstddef>
#include <string>

namespace mosaicvram {

struct CudaDeviceInfo {
  int index = 0;
  std::string name;
  int major = 0;
  int minor = 0;
  std::size_t total_global_mem_bytes = 0;
};

void SetCudaDevice(int device_index);
CudaDeviceInfo GetCudaDeviceInfo(int device_index);

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_CUDA_CUDA_DEVICE_H_
