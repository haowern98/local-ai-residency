#ifndef MOSAICVRAM_SRC_BACKEND_BACKEND_H_
#define MOSAICVRAM_SRC_BACKEND_BACKEND_H_

#include <string>
#include <vector>

namespace mosaicvram {

/**
 * Describes how much memory ownership a backend exposes to MosaicVRAM.
 */
enum class MemoryControlLevel {
  /// Backend can be loaded/unloaded, but internal buffers are opaque.
  kLifecycleOnly,
  /// MosaicVRAM owns explicit boundary tensors passed into the backend.
  kBoundaryTensor,
  /// Reserved for backends that expose allocator interception.
  kAllocatorHook,
  /// Reserved for backends with warm eviction support.
  kWarmEvictable,
};

/**
 * Static backend metadata reported by a backend implementation.
 */
struct BackendInfo {
  std::string name;
  MemoryControlLevel memory_control_level = MemoryControlLevel::kLifecycleOnly;
};

/**
 * Result of a backend load or execution step.
 */
struct BackendRunResult {
  bool ok = false;
  std::string error_message;
};

/**
 * Key/value metric emitted by backend smoke and runtime commands.
 */
struct BackendMetric {
  std::string key;
  std::string value;
};

/**
 * Minimal runtime interface for non-LLM tensor backends.
 */
class Backend {
 public:
  virtual ~Backend() = default;

  /**
   * Returns backend identity and memory-control capabilities.
   */
  virtual BackendInfo info() const = 0;
  /**
   * Loads runtime resources required to execute the backend.
   */
  virtual BackendRunResult Load() = 0;
  /**
   * Executes one backend iteration.
   */
  virtual BackendRunResult RunOnce() = 0;
  /**
   * Releases loaded runtime resources.
   */
  virtual void Unload() = 0;
  /**
   * Returns backend-specific metrics collected during execution.
   */
  virtual std::vector<BackendMetric> Metrics() const { return {}; }
};

/**
 * Returns a stable display name for a memory-control level.
 */
const char* MemoryControlLevelName(MemoryControlLevel level);

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_BACKEND_BACKEND_H_
