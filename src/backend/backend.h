#ifndef MOSAICVRAM_SRC_BACKEND_BACKEND_H_
#define MOSAICVRAM_SRC_BACKEND_BACKEND_H_

#include <string>
#include <vector>

namespace mosaicvram {

enum class MemoryControlLevel {
  kLifecycleOnly,
  kBoundaryTensor,
  kAllocatorHook,
  kWarmEvictable,
};

struct BackendInfo {
  std::string name;
  MemoryControlLevel memory_control_level = MemoryControlLevel::kLifecycleOnly;
};

struct BackendRunResult {
  bool ok = false;
  std::string error_message;
};

struct BackendMetric {
  std::string key;
  std::string value;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual BackendInfo info() const = 0;
  virtual BackendRunResult Load() = 0;
  virtual BackendRunResult RunOnce() = 0;
  virtual void Unload() = 0;
  virtual std::vector<BackendMetric> Metrics() const { return {}; }
};

const char* MemoryControlLevelName(MemoryControlLevel level);

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_BACKEND_BACKEND_H_
