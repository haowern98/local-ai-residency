#include "cuda/cuda_device.h"

#include <cuda_runtime_api.h>

#include "cuda/cuda_error.h"

namespace mosaicvram {

void SetCudaDevice(int device_index) {
  MOSAICVRAM_CUDA_CHECK(cudaSetDevice(device_index), "cudaSetDevice");
}

CudaDeviceInfo GetCudaDeviceInfo(int device_index) {
  int device_count = 0;
  MOSAICVRAM_CUDA_CHECK(cudaGetDeviceCount(&device_count),
                        "cudaGetDeviceCount");
  if (device_index < 0 || device_index >= device_count) {
    throw std::out_of_range("CUDA device index is out of range");
  }

  cudaDeviceProp prop;
  MOSAICVRAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index),
                        "cudaGetDeviceProperties");

  CudaDeviceInfo info;
  info.index = device_index;
  info.name = prop.name;
  info.major = prop.major;
  info.minor = prop.minor;
  info.total_global_mem_bytes = prop.totalGlobalMem;
  return info;
}

}  // namespace mosaicvram
