#include "commands/run_command.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "backend/backend_runtime.h"

#ifdef MOSAICVRAM_ENABLE_ONNX
#include "backend/onnx_backend.h"
#endif  // MOSAICVRAM_ENABLE_ONNX

namespace mosaicvram {
namespace {

struct RunOptions {
  std::string backend;
  std::string model_path;
  std::string io_mode = "cpu";
  std::string provider = "cuda";
  std::string input_mode = "ramp";
  std::vector<std::string> shape_overrides;
  int device_index = 0;
  int seed = 123;
  int iterations = 1;
};

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

bool ParseInt64(std::string_view text, int64_t* value) {
  int64_t parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *value = parsed;
  return true;
}

#ifdef MOSAICVRAM_ENABLE_ONNX
bool ParseShapeOverride(std::string_view text,
                        OnnxShapeOverride* shape_override) {
  const std::size_t equals_pos = text.find('=');
  if (equals_pos == std::string_view::npos || equals_pos == 0 ||
      equals_pos + 1 >= text.size()) {
    return false;
  }

  shape_override->name = std::string(text.substr(0, equals_pos));
  std::string_view shape_text = text.substr(equals_pos + 1);
  while (!shape_text.empty()) {
    const std::size_t separator_pos = shape_text.find('x');
    const std::string_view dim_text = separator_pos == std::string_view::npos
                                          ? shape_text
                                          : shape_text.substr(0, separator_pos);
    int64_t dim = 0;
    if (!ParseInt64(dim_text, &dim) || dim <= 0) {
      return false;
    }
    shape_override->shape.push_back(dim);
    if (separator_pos == std::string_view::npos) {
      break;
    }
    shape_text = shape_text.substr(separator_pos + 1);
  }
  return !shape_override->shape.empty();
}

bool ParseOnnxIoMode(const std::string& text, OnnxIoMode* io_mode) {
  if (text == "cpu") {
    *io_mode = OnnxIoMode::kCpu;
    return true;
  }
  if (text == "cuda") {
    *io_mode = OnnxIoMode::kCuda;
    return true;
  }
  return false;
}

bool ParseOnnxProvider(const std::string& text, OnnxProvider* provider) {
  if (text == "cpu") {
    *provider = OnnxProvider::kCpu;
    return true;
  }
  if (text == "cuda") {
    *provider = OnnxProvider::kCuda;
    return true;
  }
  return false;
}

bool ParseOnnxInputMode(const std::string& text, OnnxInputMode* input_mode) {
  if (text == "zero") {
    *input_mode = OnnxInputMode::kZero;
    return true;
  }
  if (text == "one") {
    *input_mode = OnnxInputMode::kOne;
    return true;
  }
  if (text == "ramp") {
    *input_mode = OnnxInputMode::kRamp;
    return true;
  }
  if (text == "random") {
    *input_mode = OnnxInputMode::kRandom;
    return true;
  }
  return false;
}
#endif  // MOSAICVRAM_ENABLE_ONNX

void PrintRunUsage() {
  std::cout << "Usage: mosaicvram.exe run --backend <name> --model <path> "
               "[--iters <count>] [--io <cpu|cuda>]\n"
            << "\n"
            << "Options:\n"
            << "  --provider <cpu|cuda>       ONNX execution provider "
               "(default: cuda)\n"
            << "  --device <index>            CUDA device index (default: 0)\n"
            << "  --input <zero|one|ramp|random>  Generated ONNX input "
               "(default: ramp)\n"
            << "  --seed <n>                  Random input seed "
               "(default: 123)\n"
            << "  --shape <name=d0xd1...>     Override dynamic tensor shape\n"
            << "\n"
            << "Backends:\n"
#ifdef MOSAICVRAM_ENABLE_ONNX
            << "  onnx    ONNX Runtime CUDA backend\n"
#endif  // MOSAICVRAM_ENABLE_ONNX
            << "\n";
}

bool ParseOptions(int argc, char** argv, RunOptions* options) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintRunUsage();
      return false;
    }

    if (i + 1 >= argc) {
      std::cerr << "Missing value for option: " << arg << "\n";
      return false;
    }

    const std::string_view value = argv[++i];
    if (arg == "--backend") {
      options->backend = std::string(value);
    } else if (arg == "--model") {
      options->model_path = std::string(value);
    } else if (arg == "--io") {
      options->io_mode = std::string(value);
    } else if (arg == "--provider") {
      options->provider = std::string(value);
    } else if (arg == "--device") {
      if (!ParseInt(value, &options->device_index)) {
        std::cerr << "Invalid --device value: " << value << "\n";
        return false;
      }
    } else if (arg == "--input") {
      options->input_mode = std::string(value);
    } else if (arg == "--seed") {
      if (!ParseInt(value, &options->seed)) {
        std::cerr << "Invalid --seed value: " << value << "\n";
        return false;
      }
    } else if (arg == "--shape") {
      options->shape_overrides.push_back(std::string(value));
    } else if (arg == "--iters") {
      if (!ParseInt(value, &options->iterations)) {
        std::cerr << "Invalid --iters value: " << value << "\n";
        return false;
      }
    } else {
      std::cerr << "Unknown run option: " << arg << "\n";
      return false;
    }
  }

  if (options->backend.empty()) {
    std::cerr << "--backend is required\n";
    return false;
  }
  if (options->model_path.empty()) {
    std::cerr << "--model is required\n";
    return false;
  }
  if (options->iterations <= 0) {
    std::cerr << "--iters must be > 0\n";
    return false;
  }
  if (options->device_index < 0) {
    std::cerr << "--device must be >= 0\n";
    return false;
  }
  if (options->seed < 0) {
    std::cerr << "--seed must be >= 0\n";
    return false;
  }
  return true;
}

std::unique_ptr<Backend> CreateBackend(const RunOptions& options,
                                       std::string* error_message) {
#ifdef MOSAICVRAM_ENABLE_ONNX
  if (options.backend == "onnx") {
    OnnxBackendOptions onnx_options;
    onnx_options.model_path = options.model_path;
    onnx_options.device_index = options.device_index;
    onnx_options.seed = static_cast<std::uint32_t>(options.seed);
    if (!ParseOnnxIoMode(options.io_mode, &onnx_options.io_mode)) {
      *error_message = "unsupported ONNX --io mode: " + options.io_mode;
      return nullptr;
    }
    if (!ParseOnnxProvider(options.provider, &onnx_options.provider)) {
      *error_message = "unsupported ONNX --provider: " + options.provider;
      return nullptr;
    }
    if (!ParseOnnxInputMode(options.input_mode, &onnx_options.input_mode)) {
      *error_message = "unsupported ONNX --input mode: " + options.input_mode;
      return nullptr;
    }
    if (onnx_options.io_mode == OnnxIoMode::kCuda &&
        onnx_options.provider != OnnxProvider::kCuda) {
      *error_message = "ONNX --io cuda requires --provider cuda";
      return nullptr;
    }
    for (const std::string& shape_text : options.shape_overrides) {
      OnnxShapeOverride shape_override;
      if (!ParseShapeOverride(shape_text, &shape_override)) {
        *error_message = "invalid ONNX --shape override: " + shape_text;
        return nullptr;
      }
      onnx_options.shape_overrides.push_back(std::move(shape_override));
    }
    return std::make_unique<OnnxBackend>(std::move(onnx_options));
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  *error_message = "unsupported backend: " + options.backend;
  return nullptr;
}

void PrintReport(const BackendRuntimeReport& report) {
  std::cout << std::fixed << std::setprecision(3) << "command=run\n"
            << "backend_name=" << report.backend_name << "\n"
            << "memory_control_level="
            << MemoryControlLevelName(report.memory_control_level) << "\n"
            << "requested_iterations=" << report.requested_iterations << "\n"
            << "successful_runs=" << report.successful_runs << "\n"
            << "total_ms=" << report.total_ms << "\n"
            << "load_ms=" << report.load_ms << "\n"
            << "run_ms=" << report.run_ms << "\n"
            << "unload_ms=" << report.unload_ms << "\n"
            << "status=" << (report.ok ? "ok" : "error") << "\n";
  for (const BackendMetric& metric : report.backend_metrics) {
    std::cout << metric.key << "=" << metric.value << "\n";
  }
  if (!report.error_message.empty()) {
    std::cout << "error_message=" << report.error_message << "\n";
  }
}

}  // namespace

int RunBackendCommand(int argc, char** argv) {
  RunOptions options;
  if (!ParseOptions(argc, argv, &options)) {
    return EXIT_FAILURE;
  }

  std::string error_message;
  std::unique_ptr<Backend> backend = CreateBackend(options, &error_message);
  if (backend == nullptr) {
    std::cerr << error_message << "\n";
    return EXIT_FAILURE;
  }

  BackendRuntimeOptions runtime_options;
  runtime_options.iterations = options.iterations;

  BackendRuntime runtime;
  const BackendRuntimeReport report =
      runtime.Run(runtime_options, backend.get());
  PrintReport(report);
  return report.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace mosaicvram
