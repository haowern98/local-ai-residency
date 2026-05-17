# MosaicVRAM

MosaicVRAM is a C++20/CUDA runtime for switching local AI backends in and out
of GPU residency without discarding resumable session state.

The runtime can save a backend's active state to pinned host memory, evict GPU
context or model residency, run another backend, reload the original backend,
restore state, and continue from the saved point instead of replaying the whole
prompt.

## Supported Backends

- llama.cpp GGUF models through llama.cpp state save/restore APIs.
- ONNX Runtime LLM decoder exports with explicit KV-cache inputs and outputs.
- ONNX Runtime tensor models as lifecycle or boundary-tensor workloads.

See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the tested model matrix
and the exact definition of CLI compatibility.

## Quick Start

Build from a Visual Studio x64 developer shell:

```text
cmake -S . -B build-llama-onnx -G Ninja ^
  -DMOSAICVRAM_ENABLE_LLAMA=ON ^
  -DMOSAICVRAM_LLAMA_DIR=..\llama.cpp ^
  -DMOSAICVRAM_LLAMA_BUILD_DIR=..\llama.cpp\build-cuda ^
  -DMOSAICVRAM_ENABLE_ONNX=ON ^
  -DMOSAICVRAM_ONNXRUNTIME_DIR=deps\onnxruntime-win-x64-gpu-1.25.1 ^
  -DMOSAICVRAM_ENABLE_TOKENIZER_JSON=ON ^
  -DMOSAICVRAM_TOKENIZERS_CPP_DIR=deps\tokenizers-cpp
cmake --build build-llama-onnx
```

Run the interactive CLI from a directory containing `sessions.txt`:

```text
mosaicvram.exe cli
```

Run deterministic residency validation plans:

```text
mosaicvram.exe residency-run --plan path\to\plan.txt
```

## Documentation

- [User Guide](docs/USER_GUIDE.md): CLI usage, `sessions.txt`, commands, and
  common runtime errors.
- [Developer Guide](docs/DEV_GUIDE.md): build flags, architecture,
  deterministic resume tests, and validation workflow.
- [Compatibility](docs/COMPATIBILITY.md): supported model/export matrix and
  blocked export notes.

The older focused notes are still available for narrow workflows:

- [Gemma CUDA Export Note](docs/GEMMA_CUDA_EXPORT.md): notes for producing a
  Gemma ONNX export shape that ONNX Runtime CUDA can execute.
- [GPU Smoke Tests](docs/GPU_SMOKE_TESTS.md): commands and generated plans for
  llama.cpp-only, ONNX-only, and mixed-backend GPU residency checks.

## Core Idea

Most local-AI launchers can unload one model before loading another. That frees
VRAM, but often loses active session state or requires replaying the prompt.

MosaicVRAM targets a lower-level lifecycle:

```text
save state -> evict GPU residency -> run another backend -> reload -> restore -> resume
```

Each backend adapter owns its native state format, while the residency
controller drives the same lifecycle across backends.
