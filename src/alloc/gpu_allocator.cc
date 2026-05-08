#include "alloc/gpu_allocator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "cuda/cuda_error.h"

namespace mosaicvram {

namespace {

bool WouldExceedBudget(std::size_t live_bytes, std::size_t requested_bytes,
                       std::size_t budget_bytes) {
  if (budget_bytes == 0) {
    return false;
  }
  if (requested_bytes > budget_bytes) {
    return true;
  }
  return live_bytes > budget_bytes - requested_bytes;
}

}  // namespace

GpuAllocator::GpuAllocator(std::size_t budget_bytes)
    : budget_bytes_(budget_bytes) {}

Allocation GpuAllocator::Allocate(std::size_t bytes, cudaStream_t stream) {
  if (bytes == 0) {
    throw std::invalid_argument("allocation size must be greater than zero");
  }
  if (WouldExceedBudget(stats_.live_bytes, bytes, budget_bytes_)) {
    ++stats_.budget_rejection_count;
    return Allocation{};
  }

  void* ptr = nullptr;
  MOSAICVRAM_CUDA_CHECK(cudaMallocAsync(&ptr, bytes, stream),
                        "cudaMallocAsync");

  stats_.live_bytes += bytes;
  stats_.peak_live_bytes = std::max(stats_.peak_live_bytes, stats_.live_bytes);
  stats_.total_allocated_bytes += bytes;
  ++stats_.allocation_count;

  return Allocation{ptr, bytes};
}

void GpuAllocator::Free(const Allocation& allocation, cudaStream_t stream) {
  if (!IsValidAllocation(allocation)) {
    return;
  }
  if (allocation.bytes > stats_.live_bytes) {
    throw std::logic_error("free would underflow live allocation bytes");
  }

  MOSAICVRAM_CUDA_CHECK(cudaFreeAsync(allocation.ptr, stream), "cudaFreeAsync");
  stats_.live_bytes -= allocation.bytes;
  ++stats_.free_count;
}

bool GpuAllocator::CanAllocate(std::size_t bytes) const {
  return !WouldExceedBudget(stats_.live_bytes, bytes, budget_bytes_);
}

std::size_t GpuAllocator::budget_bytes() const { return budget_bytes_; }

std::size_t GpuAllocator::live_bytes() const { return stats_.live_bytes; }

AllocatorStats GpuAllocator::stats() const { return stats_; }

}  // namespace mosaicvram
