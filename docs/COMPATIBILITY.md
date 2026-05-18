# Compatibility

This document defines the MosaicVRAM compatibility boundary. It is intentionally
narrower than "can load a local model."

A model is compatible only when MosaicVRAM can save active session state, evict
backend GPU residency, reload residency, restore state, and continue generation
from the restored point.

## Status Definitions

| Status | Meaning |
| --- | --- |
| CLI compatible | Passes `/load`, chat, `/save`, `/evict`, `/restore`, and post-restore chat in CLI mode. |
| Deterministic resume compatible | Passes `residency-run` token-level baseline/restore comparison. |
| Partial | Loads or generates, but does not pass the full CLI restore/chat flow or produces unusable output. |
| Blocked | Fails to load, fails CUDA execution, exposes an unsupported state surface, or requires unsupported provider/custom ops. |

For user-facing compatibility claims, use **CLI compatible**. `residency-run`
is the lower-level engineering proof that restored token continuation is
deterministic.

## Backend Summary

| Backend | Status | State Saved | Notes |
| --- | --- | --- | --- |
| llama.cpp GGUF | Full adapter | llama full/sequence state | Uses llama.cpp state APIs. |
| ONNX LLM explicit-KV decoder | Full adapter for supported graphs | KV cache and supported recurrent state tensors | Uses `Ort::Session` and `IoBinding`. |
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

## ONNX LLM Requirements

The ONNX LLM adapter supports decoder models that expose explicit cache tensors
through lower-level ONNX Runtime `Ort::Session` APIs.

Supported graph surface:

```text
input_ids or inputs_embeds
attention_mask
optional position_ids
optional num_logits_to_keep
past_key_values.*.key
past_key_values.*.value
present.*.key
present.*.value
optional past_conv.* / present_conv.*
optional past_recurrent.* / present_recurrent.*
logits
```

Supported runtime properties:

```text
KV/recurrent state dtype: float16 or float32
logits dtype: float16 or float32
batch size: 1
CUDA execution provider
single-file ONNX or ONNX with external data files
tokenizer.json for CLI chat
```

If a decoder takes `inputs_embeds`, MosaicVRAM expects a sibling embedding ONNX
model that accepts token IDs and returns embeddings with the decoder's hidden
size.

Unsupported:

```text
hidden KV cache inside custom generator state
required graph inputs outside the supported surface
KV/recurrent state dtypes other than float16 or float32
provider-specific custom ops unavailable in ONNX Runtime CUDA
models requiring browser, WebGPU, RyzenAI, DirectML, or another custom runtime
```

## CLI-Compatible ONNX Models

These models passed the CLI residency flow:

```text
/load
chat generation
/save
/evict
/restore
chat generation after restore
```

| Model | Export | CLI Result | Notes |
| --- | --- | --- | --- |
| Phi-3 Mini 128K Instruct ONNX | CUDA INT4 | Compatible | Chat and post-restore chat pass. |
| Llama 3.2 3B Instruct ONNX | CUDA INT4 GenAI export | Compatible | Split `inputs_embeds` decoder with sibling embedding model. |
| DeepSeek R1 Distill Llama 8B ONNX | GPU INT4 RTN block-32 | Compatible | Emits visible `<think>` blocks by model design. |
| Keisuke Gemma 3 4B IT ONNX INT4 | GenAI-style INT4 | Compatible | Use the complete Gemma tokenizer folder for CLI chat. |

## Partial Or Not Recommended

| Model | Export | Result | Reason |
| --- | --- | --- | --- |
| willopcbeta Gemma 3 4B Text IT ONNX | q4f16 | Partial | Short save/restore tests complete, but generated text quality was poor/junk. |
| Prince-1 Gemma 3 4B IT ONNX | onnxruntime_genai | Partial | First chat works, but restore-then-chat fails with attention-mask broadcast shape mismatch. |
| Qwen3 4B ONNX | CUDA INT4 | Partial for CLI | CUDA graph runs, but CLI chat output needs tokenizer/template cleanup before marking compatible. |

## Blocked ONNX Exports

| Model | Export | Result | Reason |
| --- | --- | --- | --- |
| onnx-community Qwen3.5 4B ONNX | q4f16 split export | Blocked | ONNX Runtime CUDA rejects `GroupQueryAttention` with non-empty `attention_bias`. |
| onnx-community Gemma 3 4B IT ONNX | q4f16 split export | Blocked | Same `GroupQueryAttention attention_bias` CUDA limitation. |
| Prince-1 Qwen3.5 4B ONNX | onnxruntime-genai | Blocked | ONNX graph load fails: `past_key_values.0.key` is referenced but not declared as an input, initializer, or previous node output. |
| Qwen3 1.7B ONNX | q4f16/fp16 | Blocked | Decode produced all-NaN logits with tested exports. |
| Gemma 3 1B ONNX | q4f16/fp16 community exports | Blocked | ONNX Runtime CUDA rejected `GroupQueryAttention` with non-empty `attention_bias`. |
| AMD Qwen3 8B AWQ ONNX hybrid | RyzenAI hybrid | Blocked | Requires custom `com.ryzenai:MatMulNBits` op. |

## GroupQueryAttention Bias Limitation

Some Qwen and Gemma ONNX exports wire a non-empty `attention_bias` tensor into
Microsoft `GroupQueryAttention`. ONNX Runtime CUDA rejects that form:

```text
attention_bias is not supported in GroupQueryAttention cuda kernel
```

For MosaicVRAM, a compatible export must either avoid that node shape or export
`GroupQueryAttention` with an empty `attention_bias` input while still exposing
explicit KV-cache inputs and outputs. See [GEMMA_CUDA_EXPORT.md](GEMMA_CUDA_EXPORT.md).

## Mixed-Backend Handoff

MosaicVRAM validates mixed backend residency by saving one backend, evicting it,
running another backend, then restoring the first backend and checking
continuation.

Representative `residency-run` result:

```text
session1_backend=llama
session1_resume_match=yes
session2_backend=onnx-llm
session2_resume_match=yes
status=ok
```

For the definition of deterministic resume and the output fields used to
measure it, see [DEV_GUIDE.md](DEV_GUIDE.md#deterministic-resume).

## Compatibility Policy

MosaicVRAM should fail closed. If a backend graph exposes an unsupported state
surface, dtype, provider dependency, or required input, the adapter should reject
the model with a clear compatibility error rather than pretending it can restore
state safely.
