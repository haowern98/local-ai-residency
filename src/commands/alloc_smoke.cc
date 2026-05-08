#include "commands/alloc_smoke.h"

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "alloc/allocation.h"
#include "alloc/gpu_allocator.h"
#include "cuda/cuda_device.h"
#include "cuda/cuda_error.h"
#include "cuda/cuda_stream.h"

namespace mosaicvram {
namespace {

constexpr std::size_t kBytesPerMiB = 1024 * 1024;

struct AllocSmokeOptions {
  int device_index = 0;
  std::size_t budget_mb = 1024;
  std::size_t block_mb = 64;
  int iterations = 1000;
  int live_blocks = 8;
};

std::size_t MiBToBytes(std::size_t mib) {
  if (mib > std::numeric_limits<std::size_t>::max() / kBytesPerMiB) {
    throw std::overflow_error("MiB value is too large");
  }
  return mib * kBytesPerMiB;
}

std::size_t BytesToMiB(std::size_t bytes) { return bytes / kBytesPerMiB; }

bool ParseSize(std::string_view text, std::size_t* value) {
  unsigned long long parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  if (parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  *value = static_cast<std::size_t>(parsed);
  return true;
}

bool ParseInt(std::string_view text, int* value) {
  int parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *value = parsed;
  return true;
}

void PrintAllocSmokeUsage() {
  std::cout
      << "Usage: mosaicvram.exe alloc-smoke [options]\n"
      << "\n"
      << "Options:\n"
      << "  --device <index>       CUDA device index (default: 0)\n"
      << "  --budget_mb <mib>      Allocation budget in MiB (default: 1024)\n"
      << "  --block_mb <mib>       Allocation block size in MiB (default: 64)\n"
      << "  --iters <count>        Allocation iterations (default: 1000)\n"
      << "  --live_blocks <count>  Rolling live block count (default: 8)\n";
}

bool ParseOptions(int argc, char** argv, AllocSmokeOptions* options) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintAllocSmokeUsage();
      return false;
    }

    if (i + 1 >= argc) {
      std::cerr << "Missing value for option: " << arg << "\n";
      return false;
    }

    const std::string_view value = argv[++i];
    if (arg == "--device") {
      if (!ParseInt(value, &options->device_index)) {
        std::cerr << "Invalid --device value: " << value << "\n";
        return false;
      }
    } else if (arg == "--budget_mb") {
      if (!ParseSize(value, &options->budget_mb)) {
        std::cerr << "Invalid --budget_mb value: " << value << "\n";
        return false;
      }
    } else if (arg == "--block_mb") {
      if (!ParseSize(value, &options->block_mb)) {
        std::cerr << "Invalid --block_mb value: " << value << "\n";
        return false;
      }
    } else if (arg == "--iters") {
      if (!ParseInt(value, &options->iterations)) {
        std::cerr << "Invalid --iters value: " << value << "\n";
        return false;
      }
    } else if (arg == "--live_blocks") {
      if (!ParseInt(value, &options->live_blocks)) {
        std::cerr << "Invalid --live_blocks value: " << value << "\n";
        return false;
      }
    } else {
      std::cerr << "Unknown alloc-smoke option: " << arg << "\n";
      return false;
    }
  }

  if (options->budget_mb == 0 || options->block_mb == 0 ||
      options->iterations <= 0 || options->live_blocks <= 0) {
    std::cerr << "budget_mb, block_mb, iters, and live_blocks must be > 0\n";
    return false;
  }

  return true;
}

void PrintStats(const AllocSmokeOptions& options, const CudaDeviceInfo& device,
                const AllocatorStats& stats) {
  const char* status =
      stats.budget_rejection_count == 0 ? "ok" : "budget_rejected";
  std::cout << "command=alloc-smoke\n"
            << "device_index=" << device.index << "\n"
            << "device_name=" << device.name << "\n"
            << "compute_capability=" << device.major << "." << device.minor
            << "\n"
            << "device_total_mem_mb="
            << BytesToMiB(device.total_global_mem_bytes) << "\n"
            << "budget_mb=" << options.budget_mb << "\n"
            << "block_mb=" << options.block_mb << "\n"
            << "iters=" << options.iterations << "\n"
            << "live_blocks=" << options.live_blocks << "\n"
            << "live_bytes=" << stats.live_bytes << "\n"
            << "peak_live_mb=" << BytesToMiB(stats.peak_live_bytes) << "\n"
            << "total_allocated_mb=" << BytesToMiB(stats.total_allocated_bytes)
            << "\n"
            << "allocations=" << stats.allocation_count << "\n"
            << "frees=" << stats.free_count << "\n"
            << "budget_rejections=" << stats.budget_rejection_count << "\n"
            << "status=" << status << "\n";
}

}  // namespace

int RunAllocSmokeCommand(int argc, char** argv) {
  AllocSmokeOptions options;
  if (!ParseOptions(argc, argv, &options)) {
    return EXIT_FAILURE;
  }

  try {
    SetCudaDevice(options.device_index);
    const CudaDeviceInfo device = GetCudaDeviceInfo(options.device_index);
    CudaStream stream;
    GpuAllocator allocator(MiBToBytes(options.budget_mb));

    std::deque<Allocation> live_allocations;
    const std::size_t block_bytes = MiBToBytes(options.block_mb);

    for (int i = 0; i < options.iterations; ++i) {
      while (static_cast<int>(live_allocations.size()) >= options.live_blocks) {
        allocator.Free(live_allocations.front(), stream.get());
        live_allocations.pop_front();
      }

      Allocation allocation = allocator.Allocate(block_bytes, stream.get());
      if (IsValidAllocation(allocation)) {
        live_allocations.push_back(allocation);
      }
    }

    while (!live_allocations.empty()) {
      allocator.Free(live_allocations.front(), stream.get());
      live_allocations.pop_front();
    }

    stream.Synchronize();
    PrintStats(options, device, allocator.stats());
    return EXIT_SUCCESS;
  } catch (const CudaError& error) {
    std::cerr << error.what() << "\n";
    return EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return EXIT_FAILURE;
  }
}

}  // namespace mosaicvram
