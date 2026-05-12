# GPU Smoke Tests

The `tests/` directory contains CUDA smoke tests for long-context residency
handoff. These tests are meant to verify that a saved backend session can be
evicted, restored, and resumed without replaying the full prompt.

## Layout

```text
tests/
  fixtures/
    15k_prompt.txt
    README.md
  plans/
    templates/
      llama_long_context.plan
      onnx_long_context.plan
      mixed_backend.plan
  scripts/
    make_onnx_tokens.py
  run_gpu_smoke.bat
```

Generated plans are written to `tests/generated/`. That directory is ignored by
Git.

## Requirements

Build Local AI Residency with both llama.cpp and ONNX Runtime support:

The `tokenizer.json` path links `tokenizers-cpp` into `mosaicvram.exe`.
Rust/Cargo must be available on `PATH` while building, but they are not runtime
requirements.

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

The runtime DLLs must be on `PATH`: llama.cpp CUDA DLLs, ONNX Runtime GPU DLLs,
CUDA, and cuDNN.

## PowerShell Setup

Use PowerShell environment-variable syntax. Do not use `set NAME=value`, which
is `cmd.exe` syntax.

```powershell
cd "<repo>\mosaicvram"

$env:LLAMA_MODEL = "<path>\model.gguf"
$env:ONNX_MODEL = "<path>\model.onnx"
$env:ONNX_TOKENIZER = "<path>\tokenizer_directory_with_tokenizer_json"

$env:PATH = "<path>\llama.cpp\build-cuda\bin;<path>\onnxruntime\lib;<path>\CUDA\bin;<path>\torch\lib;$env:PATH"
```

Optional settings:

```powershell
$env:MOSAICVRAM_EXE = "build-llama-onnx\mosaicvram.exe"
$env:PROMPT_FILE = "tests\fixtures\15k_prompt.txt"
$env:DEVICE = "0"
$env:THREADS = "8"
$env:LLAMA_CTX = "16384"
$env:LLAMA_BATCH = "512"
$env:LLAMA_GPU_LAYERS = "-1"
$env:ONNX_PREFILL_CHUNK = "512"
```

## Running Tests

Run llama.cpp only:

```powershell
cmd /c tests\run_gpu_smoke.bat llama
```

Run ONNX LLM only:

```powershell
cmd /c tests\run_gpu_smoke.bat onnx
```

Run mixed backend handoff:

```powershell
cmd /c tests\run_gpu_smoke.bat mixed
```

The mixed test runs both adapters in one residency plan. It saves and evicts
llama.cpp, runs ONNX, restores llama.cpp, then saves and evicts llama.cpp again
so ONNX can be restored and checked too.

## ONNX Tokenization

ONNX Runtime does not provide a tokenizer. The smoke plans pass
`tokenizer=<path>` and `prompt_file=<path>` to `residency-run`. MosaicVRAM then
loads the model-specific Hugging Face `tokenizer.json` supplied by
`ONNX_TOKENIZER` and converts the prompt text into token IDs inside the C++
process.

`ONNX_TOKENIZER` must point to `tokenizer.json` or to a directory containing
`tokenizer.json`. The smoke script does not call Python or Hugging Face
Transformers for normal runs.

The same prompt can produce different token counts for different models. For
example, a prompt that is 15,000 tokens for one tokenizer may be more or fewer
tokens for another tokenizer.

For advanced diagnostics, ONNX sessions can still accept a pre-tokenized file:

```text
tokens_file=tests\generated\onnx_tokens.txt
```

That path is not the normal user flow. The normal ONNX flow is:

```text
prompt text -> native tokenizer.json adapter -> token IDs -> ONNX Runtime tensors
```

`tests/scripts/make_onnx_tokens.py` is diagnostic-only. It can compare native
tokenizer output against Hugging Face Transformers, but `run_gpu_smoke.bat` does
not use it.

## Restore Correctness

The smoke tests do not treat a successful memory copy as proof of correctness.
They validate continuation behavior.

For each backend session, the test does this:

```text
prefill prompt
save backend state
decode one next token as the baseline
evict context/model residency
reload residency
restore saved state
decode one next token from the restored state
compare restored token against baseline token
```

A restored session is considered correct when:

```text
baseline_next_token == restored_next_token
resume_match=yes
status=ok
```

For llama.cpp, `full_state_bytes` must also be nonzero. For ONNX LLMs,
`kv_state_bytes` and `restored_kv_state_bytes` should match. These fields prove
that the adapter saved real session state, not just model metadata.

## Important Output Fields

```text
sessionN_prompt_tokens
sessionN_prefill_ms
sessionN_save_state_ms
sessionN_evict_context_ms
sessionN_evict_model_ms
sessionN_model_reload_ms
sessionN_session_reload_ms
sessionN_restore_state_ms
sessionN_resume_check_ms
sessionN_baseline_next_token
sessionN_restored_next_token
sessionN_resume_match
```

For ONNX LLMs, compatibility diagnostics are also reported:

```text
sessionN_logits_dtype
sessionN_logits_vocab_size
sessionN_logits_finite_count
sessionN_logits_nan_count
sessionN_decode_valid
sessionN_position_ids_present
sessionN_prefill_chunks
sessionN_prefill_chunk_tokens
```

If `resume_match=no`, the adapter restored something different from the saved
continuation point and the result should be treated as a failure.
