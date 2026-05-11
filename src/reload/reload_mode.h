#ifndef MOSAICVRAM_SRC_RELOAD_RELOAD_MODE_H_
#define MOSAICVRAM_SRC_RELOAD_RELOAD_MODE_H_

#include <cstddef>
#include <string>

namespace mosaicvram {

enum class ReloadMode { kCold, kMmap, kStreamedVram };

const char* ReloadModeName(ReloadMode mode);

struct ReloadPolicy {
  ReloadMode mode = ReloadMode::kCold;
  std::size_t ram_budget_mb = 0;
  std::size_t prefetch_mb = 0;
  std::string pack_path;
  bool create_pack = false;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RELOAD_RELOAD_MODE_H_
