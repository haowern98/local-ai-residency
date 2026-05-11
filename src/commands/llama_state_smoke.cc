#include "commands/llama_state_smoke.h"

#ifdef MOSAICVRAM_ENABLE_LLAMA

#include <cuda_runtime_api.h>

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <windows.h>
#include <psapi.h>

#include "cuda/cuda_error.h"
#include "reload/reload_mode.h"
#include "residency/llama_residency_adapter.h"
#include "util/timer.h"

#ifdef MOSAICVRAM_ENABLE_ONNX
#include "backend/onnx_backend.h"
#endif  // MOSAICVRAM_ENABLE_ONNX

namespace mosaicvram {
namespace {

struct VramSnapshot {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  std::size_t used_bytes = 0;
};

struct CrossBackendProofResult {
  bool requested = false;
  bool ok = true;
  std::string error_message;
  double load_ms = 0.0;
  double run_ms = 0.0;
  double unload_ms = 0.0;
  int successful_runs = 0;
  VramSnapshot before_load_vram;
  VramSnapshot after_load_vram;
  VramSnapshot after_run_vram;
  VramSnapshot after_unload_vram;
};

struct LlamaStateSmokeOptions {
  LlamaResidencyOptions llama;
  bool prompt_set = false;
  std::string onnx_model_path;
  std::string onnx_io;
  std::string onnx_provider;
  std::string onnx_input;
  int onnx_iterations = 1;
  ReloadPolicy reload_policy;
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

void PrintUsage() {
  std::cout
      << "Usage: mosaicvram.exe llama-state-smoke --model <path> --prompt "
         "<text> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --session-id <id>     Mosaic session id (default: 1)\n"
      << "  --ctx <tokens>        Context size (default: 256)\n"
      << "  --batch <tokens>      Batch size (default: 256)\n"
      << "  --gpu-layers <count>  GPU layers, -1 means all (default: -1)\n"
      << "  --device <index>      CUDA device index (default: 0)\n"
      << "  --threads <count>     CPU threads (default: 8)\n"
      << "  --seq-id <id>         llama.cpp sequence id (default: 0)\n"
      << "  --onnx-model <path>   Optional ONNX model to run between llama "
         "unload/reload\n"
      << "  --onnx-io <cpu|cuda>  Required when --onnx-model is set\n"
      << "  --onnx-provider <cpu|cuda>  Required when --onnx-model is set\n"
      << "  --onnx-input <zero|one|ramp|random>  Required when --onnx-model "
         "is set\n"
      << "  --onnx-iters <count>  ONNX runs during handoff (default: 1)\n";
}

bool ParseOptions(int argc, char** argv, LlamaStateSmokeOptions* options) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return false;
    }
    if (i + 1 >= argc) {
      std::cerr << "Missing value for option: " << arg << "\n";
      return false;
    }

    const std::string_view value = argv[++i];
    if (arg == "--model") {
      options->llama.model_path = std::string(value);
    } else if (arg == "--prompt") {
      options->llama.prompt = std::string(value);
      options->prompt_set = true;
    } else if (arg == "--session-id") {
      int session_id = 0;
      if (!ParseInt(value, &session_id) || session_id <= 0) {
        std::cerr << "Invalid --session-id value: " << value << "\n";
        return false;
      }
      options->llama.session_id = static_cast<MosaicSessionId>(session_id);
    } else if (arg == "--ctx") {
      if (!ParseInt(value, &options->llama.context_tokens)) {
        std::cerr << "Invalid --ctx value: " << value << "\n";
        return false;
      }
    } else if (arg == "--batch") {
      if (!ParseInt(value, &options->llama.batch_tokens)) {
        std::cerr << "Invalid --batch value: " << value << "\n";
        return false;
      }
    } else if (arg == "--gpu-layers") {
      if (!ParseInt(value, &options->llama.gpu_layers)) {
        std::cerr << "Invalid --gpu-layers value: " << value << "\n";
        return false;
      }
    } else if (arg == "--device") {
      if (!ParseInt(value, &options->llama.device_index)) {
        std::cerr << "Invalid --device value: " << value << "\n";
        return false;
      }
    } else if (arg == "--threads") {
      if (!ParseInt(value, &options->llama.threads)) {
        std::cerr << "Invalid --threads value: " << value << "\n";
        return false;
      }
    } else if (arg == "--seq-id") {
      int sequence_id = 0;
      if (!ParseInt(value, &sequence_id)) {
        std::cerr << "Invalid --seq-id value: " << value << "\n";
        return false;
      }
      options->llama.sequence_id = sequence_id;
    } else if (arg == "--onnx-model") {
      options->onnx_model_path = std::string(value);
    } else if (arg == "--onnx-io") {
      options->onnx_io = std::string(value);
    } else if (arg == "--onnx-provider") {
      options->onnx_provider = std::string(value);
    } else if (arg == "--onnx-input") {
      options->onnx_input = std::string(value);
    } else if (arg == "--onnx-iters") {
      if (!ParseInt(value, &options->onnx_iterations)) {
        std::cerr << "Invalid --onnx-iters value: " << value << "\n";
        return false;
      }
    } else if (arg == "--reload-mode") {
      if (value == "cold") {
        options->reload_policy.mode = ReloadMode::kCold;
      } else if (value == "mmap") {
        options->reload_policy.mode = ReloadMode::kMmap;
      } else if (value == "streamed_vram") {
        options->reload_policy.mode = ReloadMode::kStreamedVram;
      } else {
        std::cerr << "Invalid --reload-mode value: " << value
                  << " (use cold, mmap, or streamed_vram)\n";
        return false;
      }
    } else if (arg == "--ram-budget-mb") {
      int budget = 0;
      if (!ParseInt(value, &budget) || budget < 0) {
        std::cerr << "Invalid --ram-budget-mb value: " << value << "\n";
        return false;
      }
      options->reload_policy.ram_budget_mb =
          static_cast<std::size_t>(budget);
    } else if (arg == "--prefetch-mb") {
      int prefetch = 0;
      if (!ParseInt(value, &prefetch) || prefetch < 0) {
        std::cerr << "Invalid --prefetch-mb value: " << value << "\n";
        return false;
      }
      options->reload_policy.prefetch_mb =
          static_cast<std::size_t>(prefetch);
    } else if (arg == "--pack-path") {
      options->reload_policy.pack_path = std::string(value);
    } else if (arg == "--create-pack") {
      if (value == "yes") {
        options->reload_policy.create_pack = true;
      } else if (value == "no") {
        options->reload_policy.create_pack = false;
      } else {
        std::cerr << "Invalid --create-pack value: " << value
                  << " (use yes or no)\n";
        return false;
      }
    } else {
      std::cerr << "Unknown llama-state-smoke option: " << arg << "\n";
      return false;
    }
  }

  if (options->llama.model_path.empty()) {
    std::cerr << "--model is required\n";
    return false;
  }
  if (!options->prompt_set || options->llama.prompt.empty()) {
    std::cerr << "--prompt is required\n";
    return false;
  }
  if (options->llama.context_tokens <= 0 || options->llama.batch_tokens <= 0 ||
      options->llama.threads <= 0 || options->llama.device_index < 0) {
    std::cerr << "--ctx, --batch, and --threads must be > 0; --device must be "
                 ">= 0\n";
    return false;
  }
  if (options->onnx_iterations <= 0) {
    std::cerr << "--onnx-iters must be > 0\n";
    return false;
  }
  if (!options->onnx_model_path.empty() &&
      (options->onnx_io.empty() || options->onnx_provider.empty() ||
       options->onnx_input.empty())) {
    std::cerr << "--onnx-io, --onnx-provider, and --onnx-input are required "
                 "when --onnx-model is set\n";
    return false;
  }
  return true;
}

VramSnapshot CaptureVram() {
  MOSAICVRAM_CUDA_CHECK(cudaDeviceSynchronize(),
                        "synchronize before VRAM read");
  VramSnapshot snapshot;
  MOSAICVRAM_CUDA_CHECK(
      cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes),
      "read CUDA memory info");
  snapshot.used_bytes = snapshot.total_bytes - snapshot.free_bytes;
  return snapshot;
}

std::size_t BytesToMiB(std::size_t bytes) { return bytes / (1024 * 1024); }

std::size_t PositiveDelta(std::size_t before, std::size_t after) {
  if (before > after) {
    return before - after;
  }
  return 0;
}

#ifdef MOSAICVRAM_ENABLE_ONNX
bool ParseOnnxIoMode(const std::string& text, OnnxIoMode* mode) {
  if (text == "cpu") {
    *mode = OnnxIoMode::kCpu;
    return true;
  }
  if (text == "cuda") {
    *mode = OnnxIoMode::kCuda;
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

CrossBackendProofResult RunOnnxHandoff(const LlamaStateSmokeOptions& options) {
  CrossBackendProofResult result;
  result.requested = !options.onnx_model_path.empty();
  if (!result.requested) {
    return result;
  }

  OnnxBackendOptions onnx_options;
  onnx_options.model_path = options.onnx_model_path;
  onnx_options.device_index = options.llama.device_index;
  if (!ParseOnnxIoMode(options.onnx_io, &onnx_options.io_mode)) {
    result.ok = false;
    result.error_message = "unsupported ONNX --onnx-io: " + options.onnx_io;
    return result;
  }
  if (!ParseOnnxProvider(options.onnx_provider, &onnx_options.provider)) {
    result.ok = false;
    result.error_message =
        "unsupported ONNX --onnx-provider: " + options.onnx_provider;
    return result;
  }
  if (!ParseOnnxInputMode(options.onnx_input, &onnx_options.input_mode)) {
    result.ok = false;
    result.error_message =
        "unsupported ONNX --onnx-input: " + options.onnx_input;
    return result;
  }
  if (onnx_options.io_mode == OnnxIoMode::kCuda &&
      onnx_options.provider != OnnxProvider::kCuda) {
    result.ok = false;
    result.error_message = "ONNX --onnx-io cuda requires --onnx-provider cuda";
    return result;
  }

  OnnxBackend onnx_backend(std::move(onnx_options));
  result.before_load_vram = CaptureVram();

  Timer load_timer;
  const BackendRunResult load_result = onnx_backend.Load();
  result.load_ms = load_timer.ElapsedMs();
  if (!load_result.ok) {
    result.ok = false;
    result.error_message = load_result.error_message;
    onnx_backend.Unload();
    result.after_unload_vram = CaptureVram();
    return result;
  }
  result.after_load_vram = CaptureVram();

  Timer run_timer;
  for (int i = 0; i < options.onnx_iterations; ++i) {
    const BackendRunResult run_result = onnx_backend.RunOnce();
    if (!run_result.ok) {
      result.ok = false;
      result.error_message = run_result.error_message;
      break;
    }
    ++result.successful_runs;
  }
  result.run_ms = run_timer.ElapsedMs();
  result.after_run_vram = CaptureVram();

  Timer unload_timer;
  onnx_backend.Unload();
  result.unload_ms = unload_timer.ElapsedMs();
  result.after_unload_vram = CaptureVram();
  return result;
}
#else
CrossBackendProofResult RunOnnxHandoff(const LlamaStateSmokeOptions& options) {
  CrossBackendProofResult result;
  result.requested = !options.onnx_model_path.empty();
  if (result.requested) {
    result.ok = false;
    result.error_message =
        "binary was not built with MOSAICVRAM_ENABLE_ONNX=ON";
  }
  return result;
}
#endif  // MOSAICVRAM_ENABLE_ONNX

int RunProof(const LlamaStateSmokeOptions& options) {
  LlamaResidencyAdapter adapter(options.llama);
  adapter.set_reload_policy(options.reload_policy);

  adapter.Load();
  const VramSnapshot after_model_load_vram = CaptureVram();

  adapter.CreateContext();
  const VramSnapshot after_context_create_vram = CaptureVram();

  adapter.PrefillPrompt();
  const VramSnapshot after_prefill_vram = CaptureVram();

  BackendStateSnapshot& snapshot = adapter.SaveState();
  adapter.CaptureBaselineNextToken();
  adapter.CheckFullRestore();
  adapter.CheckSequenceRestore();

  const VramSnapshot before_same_context_clear_vram = CaptureVram();
  adapter.CheckSameContextClearAndRestore();
  const VramSnapshot after_same_context_clear_vram = CaptureVram();

  adapter.EvictContext();
  const VramSnapshot after_context_destroy_vram = CaptureVram();

  adapter.RecreateContext();
  const VramSnapshot after_context_recreate_vram = CaptureVram();

  adapter.CheckRecreatedFullRestore();
  adapter.CheckRecreatedSequenceRestore();
  const VramSnapshot after_context_restore_vram = CaptureVram();

  const VramSnapshot before_model_evict_vram = CaptureVram();
  adapter.EvictContext();
  const VramSnapshot after_model_evict_context_destroy_vram = CaptureVram();
  adapter.EvictModel();
  const VramSnapshot after_model_unload_vram = CaptureVram();

  const CrossBackendProofResult cross_backend_result = RunOnnxHandoff(options);
  if (!cross_backend_result.ok) {
    throw std::runtime_error("ONNX handoff failed: " +
                             cross_backend_result.error_message);
  }

  adapter.ReloadModel();
  const VramSnapshot after_model_reload_vram = CaptureVram();

  Timer context_reload_timer;
  adapter.RecreateContext();
  const double context_reload_ms = context_reload_timer.ElapsedMs();
  const VramSnapshot after_model_reload_context_create_vram = CaptureVram();

  adapter.CheckModelReloadedFullRestore();
  adapter.CheckModelReloadedSequenceRestore();
  const VramSnapshot after_model_reload_restore_vram = CaptureVram();

  const LlamaResidencyReport& report = adapter.report();
  const std::size_t context_evict_freed_bytes = PositiveDelta(
      after_prefill_vram.used_bytes, after_context_destroy_vram.used_bytes);
  const bool context_evict_freed_vram = context_evict_freed_bytes > 0;
  const std::size_t same_context_clear_freed_bytes =
      PositiveDelta(before_same_context_clear_vram.used_bytes,
                    after_same_context_clear_vram.used_bytes);
  const bool same_context_clear_freed_vram = same_context_clear_freed_bytes > 0;
  const std::size_t model_only_evict_freed_bytes =
      PositiveDelta(after_model_evict_context_destroy_vram.used_bytes,
                    after_model_unload_vram.used_bytes);
  const std::size_t deep_evict_freed_bytes = PositiveDelta(
      before_model_evict_vram.used_bytes, after_model_unload_vram.used_bytes);
  const bool model_evict_freed_vram = model_only_evict_freed_bytes > 0;
  const bool cross_backend_handoff_ok =
      !cross_backend_result.requested ||
      cross_backend_result.successful_runs == options.onnx_iterations;
  const bool ok = adapter.ResumeCheck() && context_evict_freed_vram &&
                  model_evict_freed_vram && cross_backend_handoff_ok;

  std::cout
      << "command=llama-state-smoke\n"
      << "session_id=" << adapter.session_id() << "\n"
      << "residency_state=" << ResidencyStateName(adapter.residency_state())
      << "\n"
      << "model_path=" << options.llama.model_path << "\n"
      << "prompt_tokens=" << report.prompt_tokens << "\n"
      << "context_tokens=" << adapter.context_tokens() << "\n"
      << "gpu_layers=" << options.llama.gpu_layers << "\n"
      << "reload_mode=" << ReloadModeName(options.reload_policy.mode) << "\n"
      << "first_token=" << report.first_token << "\n"
      << "baseline_next_token=" << report.baseline_next_token << "\n"
      << "full_restore_next_token=" << report.full_restore_next_token << "\n"
      << "sequence_restore_next_token=" << report.sequence_restore_next_token
      << "\n"
      << "recreated_full_next_token=" << report.recreated_full_next_token
      << "\n"
      << "recreated_sequence_next_token="
      << report.recreated_sequence_next_token << "\n"
      << "same_context_restore_next_token="
      << report.same_context_restore_next_token << "\n"
      << "model_reloaded_full_next_token="
      << report.model_reloaded_full_next_token << "\n"
      << "model_reloaded_sequence_next_token="
      << report.model_reloaded_sequence_next_token << "\n"
      << "full_state_bytes=" << snapshot.full_state_bytes << "\n"
      << "sequence_state_bytes=" << snapshot.sequence_state_bytes << "\n"
      << "restored_full_bytes=" << report.restored_full_bytes << "\n"
      << "restored_sequence_bytes=" << report.restored_sequence_bytes << "\n"
      << "recreated_full_bytes=" << report.recreated_full_bytes << "\n"
      << "recreated_sequence_bytes=" << report.recreated_sequence_bytes << "\n"
      << "same_context_restore_bytes=" << report.same_context_restore_bytes
      << "\n"
      << "model_reloaded_full_bytes=" << report.model_reloaded_full_bytes
      << "\n"
      << "model_reloaded_sequence_bytes="
      << report.model_reloaded_sequence_bytes << "\n"
      << "full_restore_match=" << (report.full_restore_match ? "yes" : "no")
      << "\n"
      << "sequence_restore_match="
      << (report.sequence_restore_match ? "yes" : "no") << "\n"
      << "recreated_full_restore_match="
      << (report.recreated_full_restore_match ? "yes" : "no") << "\n"
      << "recreated_sequence_restore_match="
      << (report.recreated_sequence_restore_match ? "yes" : "no") << "\n"
      << "same_context_restore_match="
      << (report.same_context_restore_match ? "yes" : "no") << "\n"
      << "model_reloaded_full_restore_match="
      << (report.model_reloaded_full_restore_match ? "yes" : "no") << "\n"
      << "model_reloaded_sequence_restore_match="
      << (report.model_reloaded_sequence_restore_match ? "yes" : "no") << "\n"
      << "same_context_pointer_preserved="
      << (report.same_context_pointer_preserved ? "yes" : "no") << "\n"
      << "sequence_id=" << options.llama.sequence_id << "\n"
      << "initial_model_load_ms=" << report.initial_model_load_ms << "\n"
      << "model_reload_ms=" << report.model_reload_ms << "\n"
      << "context_reload_ms=" << context_reload_ms << "\n"
      << "cross_backend_handoff_requested="
      << (cross_backend_result.requested ? "yes" : "no") << "\n"
      << "cross_backend_handoff_ok="
      << (cross_backend_handoff_ok ? "yes" : "no") << "\n"
      << "onnx_model_path=" << options.onnx_model_path << "\n"
      << "onnx_io=" << options.onnx_io << "\n"
      << "onnx_provider=" << options.onnx_provider << "\n"
      << "onnx_input=" << options.onnx_input << "\n"
      << "onnx_requested_iterations=" << options.onnx_iterations << "\n"
      << "onnx_successful_runs=" << cross_backend_result.successful_runs << "\n"
      << "onnx_load_ms=" << cross_backend_result.load_ms << "\n"
      << "onnx_run_ms=" << cross_backend_result.run_ms << "\n"
      << "onnx_unload_ms=" << cross_backend_result.unload_ms << "\n"
      << "vram_after_model_load_mb="
      << BytesToMiB(after_model_load_vram.used_bytes) << "\n"
      << "vram_after_context_create_mb="
      << BytesToMiB(after_context_create_vram.used_bytes) << "\n"
      << "vram_after_prefill_mb=" << BytesToMiB(after_prefill_vram.used_bytes)
      << "\n"
      << "vram_after_context_destroy_mb="
      << BytesToMiB(after_context_destroy_vram.used_bytes) << "\n"
      << "vram_after_context_recreate_mb="
      << BytesToMiB(after_context_recreate_vram.used_bytes) << "\n"
      << "vram_after_context_restore_mb="
      << BytesToMiB(after_context_restore_vram.used_bytes) << "\n"
      << "vram_before_model_evict_mb="
      << BytesToMiB(before_model_evict_vram.used_bytes) << "\n"
      << "vram_after_model_evict_context_destroy_mb="
      << BytesToMiB(after_model_evict_context_destroy_vram.used_bytes) << "\n"
      << "vram_after_model_unload_mb="
      << BytesToMiB(after_model_unload_vram.used_bytes) << "\n"
      << "vram_before_onnx_load_mb="
      << BytesToMiB(cross_backend_result.before_load_vram.used_bytes) << "\n"
      << "vram_after_onnx_load_mb="
      << BytesToMiB(cross_backend_result.after_load_vram.used_bytes) << "\n"
      << "vram_after_onnx_run_mb="
      << BytesToMiB(cross_backend_result.after_run_vram.used_bytes) << "\n"
      << "vram_after_onnx_unload_mb="
      << BytesToMiB(cross_backend_result.after_unload_vram.used_bytes) << "\n"
      << "vram_after_model_reload_mb="
      << BytesToMiB(after_model_reload_vram.used_bytes) << "\n"
      << "vram_after_model_reload_context_create_mb="
      << BytesToMiB(after_model_reload_context_create_vram.used_bytes) << "\n"
      << "vram_after_model_reload_restore_mb="
      << BytesToMiB(after_model_reload_restore_vram.used_bytes) << "\n"
      << "vram_before_same_context_clear_mb="
      << BytesToMiB(before_same_context_clear_vram.used_bytes) << "\n"
      << "vram_after_same_context_clear_mb="
      << BytesToMiB(after_same_context_clear_vram.used_bytes) << "\n"
      << "vram_before_load_mb="
      << BytesToMiB(report.vram_before_load_bytes) << "\n"
      << "vram_after_load_mb="
      << BytesToMiB(report.vram_after_load_bytes) << "\n"
      << "vram_after_evict_model_mb="
      << BytesToMiB(report.vram_after_evict_model_bytes) << "\n"
      << "vram_after_reload_mb="
      << BytesToMiB(report.vram_after_reload_bytes) << "\n"
      << "ram_before_reload_mb="
      << BytesToMiB(report.ram_before_reload_bytes) << "\n"
      << "ram_peak_during_reload_mb="
      << BytesToMiB(report.ram_peak_during_reload_bytes) << "\n"
      << "ram_after_reload_mb="
      << BytesToMiB(report.ram_after_reload_bytes) << "\n"
      << "context_evict_freed_mb=" << BytesToMiB(context_evict_freed_bytes)
      << "\n"
      << "context_evict_freed_vram="
      << (context_evict_freed_vram ? "yes" : "no") << "\n"
      << "same_context_clear_freed_mb="
      << BytesToMiB(same_context_clear_freed_bytes) << "\n"
      << "same_context_clear_freed_vram="
      << (same_context_clear_freed_vram ? "yes" : "no") << "\n"
      << "model_kept_loaded_during_context_evict=yes\n"
      << "model_evict_freed_mb=" << BytesToMiB(model_only_evict_freed_bytes)
      << "\n"
      << "model_evict_freed_vram=" << (model_evict_freed_vram ? "yes" : "no")
      << "\n"
      << "deep_evict_freed_mb=" << BytesToMiB(deep_evict_freed_bytes) << "\n"
      << "model_reloaded_from_path=yes\n"
      << "state_storage=pinned_host\n"
      << "status=" << (ok ? "ok" : "error") << "\n";

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int RunLlamaStateSmokeCommand(int argc, char** argv) {
  LlamaStateSmokeOptions options;
  if (!ParseOptions(argc, argv, &options)) {
    return EXIT_FAILURE;
  }

  try {
    return RunProof(options);
  } catch (const std::exception& error) {
    std::cerr << "llama-state-smoke failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_LLAMA
