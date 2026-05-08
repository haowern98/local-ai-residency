#ifndef MOSAICVRAM_SRC_ALLOC_GPU_ALLOCATOR_H_
#define MOSAICVRAM_SRC_ALLOC_GPU_ALLOCATOR_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "alloc/allocation.h"

namespace mosaicvram {

struct AllocatorStats {
  std::size_t live_bytes = 0;
  std::size_t peak_live_bytes = 0;
  std::size_t total_allocated_bytes = 0;
  std::uint64_t allocation_count = 0;
  std::uint64_t free_count = 0;
  std::uint64_t budget_rejection_count = 0;
};

class GpuAllocator {
 public:
  explicit GpuAllocator(std::size_t budget_bytes);

  GpuAllocator(const GpuAllocator&) = delete;
  GpuAllocator& operator=(const GpuAllocator&) = delete;

  Allocation Allocate(std::size_t bytes, cudaStream_t stream);
  void Free(const Allocation& allocation, cudaStream_t stream);

  bool CanAllocate(std::size_t bytes) const;
  std::size_t budget_bytes() const;
  std::size_t live_bytes() const;
  AllocatorStats stats() const;

 private:
  std::size_t budget_bytes_ = 0;
  AllocatorStats stats_;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_ALLOC_GPU_ALLOCATOR_H_
