#include "backend/backend_runtime.h"

#include <algorithm>

#include "util/timer.h"

namespace mosaicvram {

const char* MemoryControlLevelName(MemoryControlLevel level) {
  switch (level) {
    case MemoryControlLevel::kLifecycleOnly:
      return "LifecycleOnly";
    case MemoryControlLevel::kBoundaryTensor:
      return "BoundaryTensor";
    case MemoryControlLevel::kAllocatorHook:
      return "AllocatorHook";
    case MemoryControlLevel::kWarmEvictable:
      return "WarmEvictable";
  }
  return "Unknown";
}

BackendRuntimeReport BackendRuntime::Run(const BackendRuntimeOptions& options,
                                         Backend* backend) const {
  Timer total_timer;
  BackendRuntimeReport report;

  const BackendInfo info = backend->info();
  report.backend_name = info.name;
  report.memory_control_level = info.memory_control_level;
  report.requested_iterations = std::max(0, options.iterations);

  Timer phase_timer;
  BackendRunResult result = backend->Load();
  report.load_ms = phase_timer.ElapsedMs();
  if (!result.ok) {
    report.error_message = result.error_message;
    backend->Unload();
    report.total_ms = total_timer.ElapsedMs();
    return report;
  }

  phase_timer.Reset();
  for (int i = 0; i < report.requested_iterations; ++i) {
    result = backend->RunOnce();
    if (!result.ok) {
      report.error_message = result.error_message;
      break;
    }
    ++report.successful_runs;
  }
  report.run_ms = phase_timer.ElapsedMs();
  report.backend_metrics = backend->Metrics();

  phase_timer.Reset();
  backend->Unload();
  report.unload_ms = phase_timer.ElapsedMs();

  report.ok = report.error_message.empty() &&
              report.successful_runs == report.requested_iterations;
  report.total_ms = total_timer.ElapsedMs();
  return report;
}

}  // namespace mosaicvram
