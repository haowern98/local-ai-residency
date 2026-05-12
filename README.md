# Local AI Residency

Local AI Residency is a C++20/CUDA runtime for switching GPU ownership between
local AI backends without discarding resumable session state.

The runtime saves backend session state to pinned host memory, evicts GPU
context/model residency, reloads residency later, restores the session, and
continues execution without replaying the full task from scratch.

The current adapters support:

- llama.cpp GGUF models through llama.cpp state save/restore APIs.
- ONNX Runtime LLMs with explicit `past_key_values.*` / `present.*` KV-cache
  tensors.
- ONNX Runtime tensor models as lifecycle or boundary-tensor workloads.

See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the current model
compatibility matrix.

Long-context CUDA smoke tests are documented in
[docs/GPU_SMOKE_TESTS.md](docs/GPU_SMOKE_TESTS.md).

## Why This Exists

Most local-AI launchers can unload one model before loading another. That frees
VRAM, but usually loses active session state or requires replaying the prompt.

Local AI Residency targets a narrower lower-level problem:

```text
save state -> evict GPU residency -> run another backend -> reload -> restore -> resume
```

The cross-backend part is the important distinction. Each backend adapter owns
its native state format, but exposes the same residency lifecycle to the
controller.

## Residency Model

A full state adapter implements this contract:

```text
SaveState()
EvictContext()
EvictModel()
ReloadModel()
RestoreState()
ResumeCheck()
```

State is stored in pinned host memory while the backend is evicted:

- llama.cpp: full and sequence state are copied to pinned host memory.
- ONNX LLM: KV-cache tensors are copied from CUDA memory to pinned host memory.

After eviction, VRAM usage drops, while system RAM usage increases by the size
of the saved state. The saved state lives only inside the running process.

## Build

Run from a Visual Studio x64 developer shell.

Base build:

```text
cmake -S . -B build -G Ninja
cmake --build build
```

Build with ONNX Runtime:

```text
cmake -S . -B build-onnx -G Ninja ^
  -DMOSAICVRAM_ENABLE_ONNX=ON ^
  -DMOSAICVRAM_ONNXRUNTIME_DIR=deps\onnxruntime-win-x64-gpu-1.25.1
cmake --build build-onnx
```

Build with llama.cpp:

```text
cmake -S . -B build-llama -G Ninja ^
  -DMOSAICVRAM_ENABLE_LLAMA=ON ^
  -DMOSAICVRAM_LLAMA_DIR=..\llama.cpp ^
  -DMOSAICVRAM_LLAMA_BUILD_DIR=..\llama.cpp\build-cuda
cmake --build build-llama
```

Build with both adapters:

```text
cmake -S . -B build-llama-onnx -G Ninja ^
  -DMOSAICVRAM_ENABLE_LLAMA=ON ^
  -DMOSAICVRAM_LLAMA_DIR=..\llama.cpp ^
  -DMOSAICVRAM_LLAMA_BUILD_DIR=..\llama.cpp\build-cuda ^
  -DMOSAICVRAM_ENABLE_ONNX=ON ^
  -DMOSAICVRAM_ONNXRUNTIME_DIR=deps\onnxruntime-win-x64-gpu-1.25.1
cmake --build build-llama-onnx
```

Build with ONNX LLM text prompts from `tokenizer.json`:

This path links `tokenizers-cpp` into `mosaicvram.exe`. Rust/Cargo must be
available on `PATH` while building, but they are not runtime requirements.

```text
git clone https://github.com/mlc-ai/tokenizers-cpp.git deps\tokenizers-cpp
git -C deps\tokenizers-cpp submodule update --init --recursive
```

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

ONNX Runtime GPU requires the ONNX Runtime `lib` directory and CUDA/cuDNN DLLs
on `PATH`.

## Usage

The main publishable interface is `residency-run`:

```text
mosaicvram.exe residency-run --plan path\to\plan.txt
```

A plan declares sessions and then executes lifecycle steps:

```text
session id=1 backend=llama model="models\qwen.gguf" prompt="My name is xjghft. Remember this." ctx=512 batch=512 gpu_layers=-1 device=0
session id=2 backend=onnx-llm model="models\qwen3.onnx" tokenizer="models\qwen3" prompt="My name is xjghft. Remember this." prefill_chunk=512 device=0

step op=load session=1
step op=prefill session=1
step op=save session=1
step op=baseline session=1
step op=evict_context session=1
step op=evict_model session=1

step op=load session=2
step op=prefill session=2
step op=save session=2
step op=baseline session=2
step op=evict_context session=2
step op=evict_model session=2

step op=reload_model session=1
step op=restore session=1
step op=resume_check session=1
```

Supported step operations:

```text
load
prefill
save
baseline
evict_context
evict_model
reload_model
restore
resume_check
restore_then_generate
switch_gpu_owner
```

## Continuation Checks

`resume_check` verifies deterministic continuation against the saved baseline
next token.

For llama.cpp sessions, `restore_then_generate` accepts text:

```text
step op=restore_then_generate session=1 text=" What is my name?" max_tokens=32 expect="xjghft"
```

For ONNX LLM sessions, `restore_then_generate` accepts text when the session was
created with `tokenizer=`:

```text
step op=restore_then_generate session=2 text=" What is my name?" max_tokens=32
```

The `expect` and `expect_tokens` fields are optional. They are useful for
human-readable demos, while `resume_match=yes` remains the stricter correctness
signal.

ONNX LLM sessions accept `prefill_chunk=<tokens>` to process long prompts as
repeated `[1, chunk_len]` ONNX Runtime calls while carrying the KV cache forward.
This mirrors ONNX's tensor-shaped execution model instead of submitting one
large prompt tensor.

ONNX LLM sessions can accept `prompt=` or `prompt_file=` when the model folder
provides a Hugging Face `tokenizer.json`:

```text
session id=2 backend=onnx-llm model="models\qwen3.onnx" tokenizer="models\qwen3" prompt_file="prompts\long.txt" prefill_chunk=512 device=0
```

The `tokenizer=` value can point either to `tokenizer.json` itself or to a
directory containing `tokenizer.json`. Tokenization runs inside the C++ process;
Python and Hugging Face Transformers are not used at runtime.

`tokens_file=` remains available as an advanced pre-tokenized path. Inline
`tokens=` is not accepted for ONNX session prompts.

## Reload Metrics

`residency-run` reports derived timing fields for each session:

```text
warm_return_ms = model/session reload + saved-state restore
cold_replay_ms = initial model/session load + prompt prefill
saved_replay_ms = cold_replay_ms - warm_return_ms
```

For llama.cpp, `warm_return_ms` uses `model_reload_ms + restore_state_ms`. For
ONNX LLMs, it uses `session_reload_ms + restore_state_ms`.

## Validation Results

Mixed llama.cpp -> ONNX -> llama.cpp handoff:

```text
session1_backend=llama
session1_residency=Resident
session1_full_state_bytes=53052430
session1_baseline_next_token=248068
session1_restored_next_token=248068
session1_resume_match=yes
session1_generated_contains_expected=yes
session2_backend=onnx-llm
session2_residency=ModelEvicted
session2_decode_valid=yes
status=ok
```

Mixed ONNX -> llama.cpp -> ONNX handoff:

```text
session1_backend=onnx-llm
session1_residency=Resident
session1_kv_state_bytes=2523136
session1_restored_kv_state_bytes=2523136
session1_baseline_next_token=1184
session1_restored_next_token=1184
session1_resume_match=yes
session1_generated_contains_expected=yes
session2_backend=llama
session2_resume_match=yes
status=ok
```

ONNX LLM compatibility sweep:

```text
Phi-3 Mini CUDA INT4 ONNX: pass
Qwen3 4B CUDA INT4 ONNX: pass
Qwen3 1.7B q4/int8/uint8/quantized ONNX: pass
Llama 3.2 1B q4f16 ONNX: pass
Qwen3 1.7B q4f16/fp16 ONNX: blocked, all-NaN logits
Gemma 3 1B q4f16/fp16 ONNX: blocked by ONNX Runtime CUDA GroupQueryAttention
AMD Qwen3 8B AWQ hybrid ONNX: blocked by custom RyzenAI op
```

## ONNX Tensor Backend

The `run` command supports simple ONNX Runtime tensor workloads:

```text
mosaicvram.exe run --backend onnx --model models\vgg16-7.onnx --iters 100 --io cpu --provider cuda --input ramp
mosaicvram.exe run --backend onnx --model models\vgg16-7.onnx --iters 100 --io cuda --provider cuda --input ramp
```

`--io cpu` reports `LifecycleOnly`. `--io cuda` reports `BoundaryTensor`,
meaning Local AI Residency owns CUDA input/output buffers bound into the ONNX
session. This does not modify ONNX Runtime source and is not an internal ONNX
allocator hook.

Dynamic tensor shapes can be provided with:

```text
--shape input_name=1x3x224x224
```

## Limitations

- ONNX LLM text tokenization currently supports Hugging Face `tokenizer.json`
  files when `MOSAICVRAM_ENABLE_TOKENIZER_JSON=ON` is enabled at build time.
- SentencePiece `.model` tokenizers are not supported yet.
- `tokenizer_config.json` without a `tokenizer.json` file is not enough.
- Python tokenizers and custom Hugging Face tokenizer code are not used at
  runtime.
- ONNX LLM support requires explicit KV-cache inputs and outputs.
- Provider-specific custom ops are not portable unless the required provider is
  available at runtime.
- Saved state is process-local. It is not a persistent checkpoint format.
- Evicting model weights frees more VRAM but still requires reloading weights
  before restore.

## Code Style

All C++ code is intended to follow the Google C++ Style Guide, including naming
conventions.

Reference: https://google.github.io/styleguide/cppguide.html
