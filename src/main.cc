#include <cstdlib>
#include <iostream>
#include <string_view>

#include "commands/alloc_smoke.h"
#include "commands/residency_run.h"
#include "commands/run_command.h"
#include "commands/shell_command.h"

#ifdef MOSAICVRAM_ENABLE_LLAMA
#include "commands/llama_state_smoke.h"
#endif  // MOSAICVRAM_ENABLE_LLAMA

namespace mosaicvram {
namespace {

constexpr std::string_view kVersion = "0.1.0";

void PrintUsage(std::string_view program_name) {
  std::cout << "Usage: " << program_name << " <command>\n"
            << "\n"
            << "Commands:\n"
            << "  alloc-smoke  Run CUDA allocator smoke test\n"
#ifdef MOSAICVRAM_ENABLE_LLAMA
            << "  llama-state-smoke  Prove llama.cpp warm-state restore\n"
#endif  // MOSAICVRAM_ENABLE_LLAMA
            << "  residency-run  Execute a residency plan\n"
            << "  run          Run a real local-AI backend\n"
            << "  shell        Start the interactive residency shell\n"
            << "  version    Print MosaicVRAM version\n";
}

int Run(int argc, char** argv) {
  if (argc <= 1) {
    PrintUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  const std::string_view command = argv[1];
  if (command == "version") {
    std::cout << "mosaicvram " << kVersion << "\n";
    return EXIT_SUCCESS;
  }
  if (command == "alloc-smoke") {
    return RunAllocSmokeCommand(argc - 2, argv + 2);
  }
  if (command == "residency-run") {
    return RunResidencyRunCommand(argc - 2, argv + 2);
  }
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (command == "llama-state-smoke") {
    return RunLlamaStateSmokeCommand(argc - 2, argv + 2);
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA
  if (command == "run") {
    return RunBackendCommand(argc - 2, argv + 2);
  }
  if (command == "shell") {
    return RunShellCommand(argc - 2, argv + 2);
  }

  std::cerr << "Unknown command: " << command << "\n";
  PrintUsage(argv[0]);
  return EXIT_FAILURE;
}

}  // namespace
}  // namespace mosaicvram

int main(int argc, char** argv) { return mosaicvram::Run(argc, argv); }
