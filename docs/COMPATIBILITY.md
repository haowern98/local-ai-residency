# Compatibility

This document describes the current MosaicVRAM backend compatibility boundary.
It is intentionally narrower than "all local AI models." A backend is considered
a full state adapter only when MosaicVRAM can save session state, evict backend
GPU residency, reload residency, restore state, and resume deterministically.

## Summary

| Backend | Status | State Saved | Notes |
| --- | --- | --- | --- |
| llama.cpp GGUF | Full adapter | llama full/sequence state | Uses llama.cpp state APIs. |
| ONNX LLM explicit-KV decoder | Full adapter for supported graphs | KV cache tensors | Uses `Ort::Session` and `IoBinding`. |
| ONNX static tensor model | Boundary tensor backend | Input/output tensors only | Useful as intervening GPU workload, not a full state adapter. |
| whisper.cpp | Not integrated yet | None | Planned later. |
| TensorRT | Not integrated yet | None | Planned later. |

## llama.cpp

The llama.cpp adapter is model-agnostic at the MosaicVRAM layer as long as the
local llama.cpp build can load the GGUF and its state APIs work for the model.

Required backend behavior:

```text
load llama_model
create llama_context
prefill prompt
save state to pinned host memory
destroy context
optionally unload model
reload model
recreate context
restore state
resume deterministically
```

Validated locally:

| Model | Result |
| --- | --- |
| Qwen GGUF through llama.cpp | Pass |

## ONNX LLM

The ONNX LLM adapter supports decoder models that expose explicit KV cache
tensors through the lower-level ONNX Runtime `Ort::Session` API.

Required graph surface:

```text
input_ids input
attention_mask input
optional position_ids input
past_key_values.*.key inputs
past_key_values.*.value inputs
present.*.key outputs
present.*.value outputs
logits output
```

Currently supported:

```text
KV cache dtype: float16 or float32
logits dtype: float16 or float32
batch size: 1
CUDA execution provider
single-file ONNX or ONNX with external data files
```

Currently unsupported:

```text
hidden KV cache inside custom operators
required inputs other than input_ids, attention_mask, position_ids, and KV cache
KV cache dtypes other than float16 or float32
split decoder / decoder-with-past workflows
provider-specific custom ops not available in ONNX Runtime CUDA
models that require browser, WebGPU, RyzenAI, DirectML, or other custom runtimes
```

Validated locally:

| Model | Export | Result | Notes |
| --- | --- | --- | --- |
| Phi-3 Mini 4K Instruct ONNX | CUDA INT4 | Pass | `float16` logits, explicit KV cache. |
| Qwen3 4B ONNX | CUDA INT4 | Pass | `float16` logits, explicit KV cache. |
| Qwen3 1.7B ONNX | q4/int8/uint8/quantized | Pass | `float32` logits and `float32` KV cache. |
| Llama 3.2 1B Instruct ONNX | q4f16 | Pass | `float32` logits, explicit KV cache. |
| Gemma 3 1B ONNX | CUDA GenAI fp16 export | Pass | Requires export with empty `GroupQueryAttention` `attention_bias`; see [GEMMA_CUDA_EXPORT.md](GEMMA_CUDA_EXPORT.md). |
| Qwen3 1.7B ONNX | q4f16/fp16 | Blocked | Decode produced all-NaN logits with these exports. |
| Gemma 3 1B ONNX | q4f16/fp16 community exports | Blocked | ONNX Runtime CUDA rejected `GroupQueryAttention` with non-empty `attention_bias`. |
| AMD Qwen3 8B AWQ ONNX hybrid | RyzenAI hybrid | Blocked | Requires custom `com.ryzenai:MatMulNBits` op. |

## Mixed-Backend Handoff

The publishable `residency-run` command has validated one mixed-backend plan
using a llama.cpp session and an ONNX LLM session in the same process.

Validated sequence:

```text
save llama state
evict llama context and model
load ONNX LLM
save ONNX KV state
evict ONNX context and model
reload and restore ONNX
evict ONNX again
reload and restore llama
resume llama deterministically
```

Representative result:

```text
session1_backend=llama
session1_resume_match=yes
session2_backend=onnx-llm
session2_resume_match=yes
status=ok
```

## Compatibility Policy

MosaicVRAM should fail closed. If a backend graph exposes an unsupported state
surface, dtype, provider dependency, or required input, the adapter should reject
the model with a clear compatibility error rather than pretending it can restore
state safely.
