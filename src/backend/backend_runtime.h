#ifndef MOSAICVRAM_SRC_BACKEND_BACKEND_RUNTIME_H_
#define MOSAICVRAM_SRC_BACKEND_BACKEND_RUNTIME_H_

#include <string>
#include <vector>

#include "backend/backend.h"

namespace mosaicvram {

struct BackendRuntimeOptions {
  int iterations = 1;
};

struct BackendRuntimeReport {
  std::string backend_name;
  MemoryControlLevel memory_control_level = MemoryControlLevel::kLifecycleOnly;
  int requested_iterations = 0;
  int successful_runs = 0;
  double total_ms = 0.0;
  double load_ms = 0.0;
  double run_ms = 0.0;
  double unload_ms = 0.0;
  bool ok = false;
  std::string error_message;
  std::vector<BackendMetric> backend_metrics;
};

class BackendRuntime {
 public:
  BackendRuntimeReport Run(const BackendRuntimeOptions& options,
                           Backend* backend) const;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_BACKEND_BACKEND_RUNTIME_H_
