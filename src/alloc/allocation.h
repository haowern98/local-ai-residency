#ifndef MOSAICVRAM_SRC_ALLOC_ALLOCATION_H_
#define MOSAICVRAM_SRC_ALLOC_ALLOCATION_H_

#include <cstddef>

namespace mosaicvram {

struct Allocation {
  void* ptr = nullptr;
  std::size_t bytes = 0;
};

inline bool IsValidAllocation(const Allocation& allocation) {
  return allocation.ptr != nullptr && allocation.bytes > 0;
}

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_ALLOC_ALLOCATION_H_
