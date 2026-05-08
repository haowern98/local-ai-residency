# MosaicVRAM

MosaicVRAM is a C++20/CUDA local-AI residency runtime experiment. Its core
goal is to separate a Mosaic session from the lifetime of backend GPU objects:
save backend session state to pinned host memory, evict context and model GPU
residency, reload residency later, restore the session, and resume without
replaying the whole task from scratch.

Backend support is strict. A backend is not considered a full MosaicVRAM state
adapter unless it can save state, evict residency, restore state, and resume.
See `docs/COMPATIBILITY.md` for the current backend/model compatibility matrix.

The project currently has:

- a CUDA allocator smoke path using `cudaMallocAsync`;
- an optional ONNX Runtime CUDA backend with MosaicVRAM-owned boundary tensors;
- an optional llama.cpp warm-state proof with context-level and model-level
  eviction/restoration;
- full llama.cpp and ONNX LLM residency adapters;
- a publishable `residency-run` command that executes explicit residency plans.

## Current Phase

Phase 9D: publishable residency-plan execution through `ResidencyController`.

## Style

All C++ code must follow the Google C++ Style Guide, including Google naming
conventions.

Reference: https://google.github.io/styleguide/cppguide.html

## Windows Build

Run from a Visual Studio x64 developer environment, or call `vcvars64.bat`
before configuring.

```text
cmake -S . -B build -G Ninja
cmake --build build
```

## Smoke Test

```text
mosaicvram.exe alloc-smoke --budget_mb 1024 --iters 1000
```

The command allocates and frees CUDA memory through the MosaicVRAM allocator,
tracks live/peak bytes, and rejects allocations that would exceed the configured
budget.

## Local Backend Dependencies

Large third-party dependencies and model files are local-only and ignored by
`.gitignore`.

Current local setup:

- `../llama.cpp/`: CUDA build of llama.cpp used for warm-state probing.
- `third_party/whisper.cpp/`: CUDA build of whisper.cpp.
- `deps/onnxruntime-win-x64-gpu-1.25.1/`: ONNX Runtime GPU C++ package.
- `models/`: local ONNX/model artifacts.

ONNX Runtime GPU requires the ONNX Runtime `lib` directory and CUDA/cuDNN DLLs
on `PATH`. On this machine, cuDNN 9 is available through the PyTorch CUDA
install under Python `site-packages\torch\lib`.

## ONNX Backend

ONNX support is optional at build time:

```text
cmake -S . -B build-onnx -G Ninja ^
  -DMOSAICVRAM_ENABLE_ONNX=ON ^
  -DMOSAICVRAM_ONNXRUNTIME_DIR=deps\onnxruntime-win-x64-gpu-1.25.1
cmake --build build-onnx
```

CPU-fed tensor example:

```text
mosaicvram.exe run --backend onnx --model models\vgg16-7.onnx --iters 100 --io cpu --provider cuda --input ramp
```

MosaicVRAM-owned CUDA boundary tensor example:

```text
mosaicvram.exe run --backend onnx --model models\vgg16-7.onnx --iters 100 --io cuda --provider cuda --input ramp
```

The ONNX backend uses ONNX Runtime's CUDA execution provider without modifying
ONNX Runtime source. `--io cpu` reports `LifecycleOnly`; `--io cuda` reports
`BoundaryTensor`, meaning MosaicVRAM owns the CUDA input/output buffers bound
into the ONNX session. This is not allocator-hook control inside ONNX Runtime.

The ONNX path is metadata-driven for static `float32` tensor models. Dynamic
tensor shapes can be provided with:

```text
--shape input_name=1x3x224x224
```

## ONNX LLM State Adapter

Phase 9A/9B target Microsoft Phi-3 Mini 4K Instruct ONNX,
`cuda/cuda-int4-rtn-block-32`, downloaded from Hugging Face. The local model
folder is ignored by git. The public binary does not expose the temporary probe
or validation commands used during development.

Verified Phase 9A result:

```text
input_count=66
output_count=65
past_input_count=64
present_output_count=64
logits_output_count=1
cache_surface_found=yes
status=ok
```

Representative tensors:

```text
input0_name=input_ids
input1_name=attention_mask
input2_name=past_key_values.0.key
input3_name=past_key_values.0.value
output0_name=logits
output1_name=present.0.key
output2_name=present.0.value
```

This proves Phi-3 Mini ONNX exposes the cache surface needed for a real ONNX
state adapter using `Ort::Session` and `IoBinding`; the high-level ONNX Runtime
GenAI API is not required for state ownership.

Phase 9B adds `OnnxLlmResidencyAdapter`, which owns the ONNX LLM KV cache
surface directly:

```text
SaveState()
EvictContext()
EvictModel()
ReloadModel()
RestoreState()
ResumeCheck()
```

Verified Phase 9B result:

```text
past_input_count=64
present_output_count=64
logits_output_count=1
cache_surface_found=yes
kv_state_bytes=1179648
restored_kv_state_bytes=1179648
first_token=306
baseline_next_token=29915
restored_next_token=29915
baseline_logits_checksum=103129
restored_logits_checksum=103129
vram_after_context_evict_mb=4332
vram_after_model_evict_mb=1236
context_evict_freed_mb=2
model_evict_freed_mb=3096
resume_match=yes
status=ok
```

## Residency Run

`residency-run` is the first publishable command that uses
`ResidencyController` directly. It consumes a small text plan, registers real
backend adapters, executes lifecycle operations, and reports adapter restore
facts.

Example ONNX LLM plan:

```text
session id=2 backend=onnx-llm model="models\phi3-mini-4k-instruct-onnx\cuda\cuda-int4-rtn-block-32\phi3-mini-4k-instruct-cuda-int4-rtn-block-32.onnx" tokens=1,15043,29892 device=0
step op=load session=2
step op=prefill session=2
step op=save session=2
step op=baseline session=2
step op=evict_context session=2
step op=evict_model session=2
step op=reload_model session=2
step op=restore session=2
step op=resume_check session=2
```

Run:

```text
mosaicvram.exe residency-run --plan path\to\plan.txt
```

Verified local Phase 9D ONNX LLM result:

```text
command=residency-run
sessions=1
steps=9
gpu_owner=2
session2_backend=onnx-llm
session2_residency=Resident
session2_prompt_tokens=3
session2_kv_state_bytes=1179648
session2_restored_kv_state_bytes=1179648
session2_baseline_next_token=29915
session2_restored_next_token=29915
session2_resume_match=yes
session2_initial_session_load_ms=1507.05
session2_session_reload_ms=1380.63
status=ok
```

Additional ONNX LLM compatibility sweep:

```text
Phi-3 Mini CUDA INT4 ONNX: pass
Qwen3 4B CUDA INT4 ONNX: pass
Qwen3 1.7B q4/int8/uint8/quantized ONNX: pass
Llama 3.2 1B q4f16 ONNX: pass
Qwen3 1.7B q4f16/fp16 ONNX: blocked, all-NaN logits
Gemma 3 1B q4f16/fp16 ONNX: blocked by ONNX Runtime CUDA GroupQueryAttention
AMD Qwen3 8B AWQ hybrid ONNX: blocked by custom RyzenAI op
```

The Qwen3 4B and Llama 3.2 1B passes prove the ONNX adapter is not Phi-specific.
The adapter now discovers `float16`/`float32` logits and optional
`position_ids` from the graph, and supports `float16`/`float32` KV cache
tensors.

Verified Phase 9F mixed-backend handoff through the same publishable command:

```text
llama.cpp Qwen GGUF session
save llama state to pinned host memory
evict llama context and model
load Qwen3 4B ONNX LLM
save/evict/reload/restore ONNX KV state
evict ONNX context and model
reload llama model
restore llama state
resume llama deterministically
```

Result:

```text
command=residency-run
sessions=2
steps=20
gpu_owner=1
session1_backend=llama
session1_residency=Resident
session1_full_state_bytes=52986854
session1_baseline_next_token=760
session1_restored_next_token=760
session1_resume_match=yes
session2_backend=onnx-llm
session2_residency=ModelEvicted
session2_logits_dtype=float16
session2_kv_state_bytes=442368
session2_restored_kv_state_bytes=442368
session2_baseline_next_token=279
session2_restored_next_token=279
session2_resume_match=yes
status=ok
```

## llama.cpp Warm-State Proof

llama.cpp support is optional at build time:

```text
cmake -S . -B build-llama -G Ninja ^
  -DMOSAICVRAM_ENABLE_LLAMA=ON ^
  -DMOSAICVRAM_LLAMA_DIR=..\llama.cpp ^
  -DMOSAICVRAM_LLAMA_BUILD_DIR=..\llama.cpp\build-cuda
cmake --build build-llama
```

Example:

```text
mosaicvram.exe llama-state-smoke ^
  --model ..\llama.cpp\links\qwen\Qwen3.5-9B-Q4_K_M.gguf ^
  --prompt "MosaicVRAM restores warm llama state." ^
  --ctx 256 --batch 256 --gpu-layers -1 --device 0 --threads 8
```

The command loads a real GGUF model, decodes a prompt, copies full and
per-sequence llama.cpp state into pinned host memory, restores both states, and
checks deterministic continuation.

It also proves two eviction levels:

- context eviction: destroy `llama_context`, keep `llama_model` loaded, recreate
  the context, restore state, and resume;
- model eviction: destroy `llama_context`, unload `llama_model`, reload the
  same model from path, recreate the context, restore pinned state, and resume.

Latest local Phase 8A proof on `Qwen3.5-9B-Q4_K_M.gguf`:

```text
model_reloaded_full_restore_match=yes
model_reloaded_sequence_restore_match=yes
context_evict_freed_mb=308
model_evict_freed_mb=4812
deep_evict_freed_mb=5126
state_storage=pinned_host
status=ok
```

The same-context clear probe is intentionally still reported. On this llama.cpp
build it preserves the same `llama_context*`, but does not free VRAM:

```text
same_context_pointer_preserved=yes
same_context_clear_freed_vram=no
```

## Cross-Backend Handoff Proof

When the executable is built with both `MOSAICVRAM_ENABLE_LLAMA=ON` and
`MOSAICVRAM_ENABLE_ONNX=ON`, `llama-state-smoke` can run an ONNX CUDA model
between llama model unload and llama model reload:

```text
mosaicvram.exe llama-state-smoke ^
  --model ..\llama.cpp\links\qwen\Qwen3.5-9B-Q4_K_M.gguf ^
  --prompt "MosaicVRAM restores warm llama state." ^
  --ctx 256 --batch 256 --gpu-layers -1 --device 0 --threads 8 ^
  --onnx-model models\vgg16-7.onnx ^
  --onnx-io cuda --onnx-provider cuda --onnx-input ramp --onnx-iters 5
```

Latest local Phase 8C result:

```text
session_id=1
residency_state=Resident
cross_backend_handoff_requested=yes
cross_backend_handoff_ok=yes
onnx_input=ramp
onnx_successful_runs=5
model_reloaded_full_restore_match=yes
model_reloaded_sequence_restore_match=yes
vram_after_model_unload_mb=1260
vram_after_onnx_load_mb=2326
vram_after_onnx_run_mb=2338
vram_after_onnx_unload_mb=1264
vram_after_model_reload_restore_mb=6390
context_evict_freed_mb=308
model_evict_freed_mb=4812
deep_evict_freed_mb=5126
state_storage=pinned_host
status=ok
```

This proves the current handoff sequence:

```text
llama.cpp session resident
save llama state to pinned host memory
destroy llama context
unload llama model
load and run ONNX CUDA model
unload ONNX session
reload llama model
recreate llama context
restore pinned llama state
resume deterministically
```

## Residency Architecture

Phase 8C moved the llama.cpp lifecycle/state code out of the command and into
reusable residency classes:

```text
src/residency/residency_state.h
src/residency/backend_state_snapshot.h
src/residency/backend_state_adapter.h
src/residency/llama_residency_adapter.h
src/residency/llama_residency_adapter.cc
```

The current full state adapter is `LlamaResidencyAdapter`. It implements the
contract for llama.cpp:

```text
SaveState()
EvictContext()
EvictModel()
ReloadModel()
RestoreState()
ResumeCheck()
```

ONNX Runtime remains an intervening CUDA backend in this proof. It is not yet
marked as a full MosaicVRAM state adapter.

Phase 9B adds `OnnxLlmResidencyAdapter`, so MosaicVRAM now has two full state
adapter implementations:

```text
LlamaResidencyAdapter
OnnxLlmResidencyAdapter
```

Phase 9C adds the backend-neutral `ResidencyController`:

```text
RegisterAdapter()
SaveSession()
EvictContext()
EvictModel()
ReloadModel()
RestoreSession()
ResumeCheck()
SwitchGpuOwner()
```

The controller is runtime code only. No temporary probe or validation command is
left in the published command surface.

Phase 9D adds the publishable `residency-run` command:

```text
src/commands/residency_run.h
src/commands/residency_run.cc
```

Unlike the earlier smoke/probe paths, this command is intended to stay in the
published binary. It executes explicit residency plans through
`ResidencyController` and works with the full adapters currently compiled into
the binary.
