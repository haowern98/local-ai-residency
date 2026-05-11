#include "reload/reload_mode.h"

namespace mosaicvram {

const char* ReloadModeName(ReloadMode mode) {
  switch (mode) {
    case ReloadMode::kCold:
      return "cold";
    case ReloadMode::kMmap:
      return "mmap";
    case ReloadMode::kStreamedVram:
      return "streamed_vram";
  }
  return "unknown";
}

}  // namespace mosaicvram
