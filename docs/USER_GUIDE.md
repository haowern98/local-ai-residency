# User Guide

This guide covers running MosaicVRAM as a user: configuring sessions, chatting
with local models, saving residency state, evicting GPU residency, and restoring
later in the same process.

## Interactive CLI

Start the CLI from the directory containing `sessions.txt`:

```text
mosaicvram.exe cli
```

Or pass the config explicitly:

```text
mosaicvram.exe cli --config path\to\sessions.txt
```

The CLI loads the config at startup and can reload it while running:

```text
/reload-config
```

## Running From A Release Zip

Extract the release zip to a folder, then run:

```powershell
.\mosaicvram.exe cli --config .\sessions.txt
```

The executable and `sessions.txt` should live beside each other:

```text
mosaicvram-release\
  mosaicvram.exe
  sessions.txt
  README.md
  docs\
    USER_GUIDE.md
    COMPATIBILITY.md
  runtime\
    onnxruntime.dll
    onnxruntime_providers_shared.dll
    onnxruntime_providers_cuda.dll
```

Edit `sessions.txt` before launching. The `model=` and `tokenizer=` values must
point to files or directories on your machine. If the ONNX Runtime DLLs are kept
in `runtime\`, add that folder to `PATH` before running the executable, or copy
the required DLLs beside `mosaicvram.exe`.

ONNX Runtime CUDA also requires compatible NVIDIA CUDA and cuDNN runtime DLLs
available on `PATH`. TensorRT DLLs are not required unless you build and run a
TensorRT-specific configuration.

## sessions.txt

Edit `sessions.txt` manually. It contains one session per line.

llama.cpp GGUF example:

```text
session llama backend=llama model="C:\models\qwen.gguf" ctx=16384 batch=512 gpu_layers=-1 device=0 max_tokens=512 temp=0.8 min_p=0.05 seed=1234
```

ONNX LLM example:

```text
session onnx backend=onnx-llm model="C:\models\gemma\model.onnx" tokenizer="C:\models\gemma-tokenizer" prefill_chunk=512 device=0 max_tokens=512 temp=0.4 top_k=20 top_p=0.9 min_p=0.0 repeat_penalty=1.05 seed=1234 stop_strings="<end_of_turn>,<start_of_turn>"
```

For ONNX chat, `tokenizer=` must point to either `tokenizer.json` or a
directory containing `tokenizer.json`. MosaicVRAM uses the tokenizer inside the
C++ process. Python and Hugging Face Transformers are not runtime requirements.

Some ONNX exports split token embedding into a sibling ONNX model. When the main
decoder takes `inputs_embeds`, MosaicVRAM discovers and runs the sibling
embedding model before invoking the decoder.

## Commands

```text
/help
/sessions
/use <session>
/chat [session]
/load <session>
/save <session>
/evict <session>
/restore <session>
/reset <session>
/reload-config
/exit
```

`/chat` enters chat mode. In chat mode, normal text goes to the active model.
Lines starting with `/` are still commands.

Example:

```text
mosaic> /sessions
mosaic> /load onnx
onnx> My name is Gary. Remember this.
onnx> /save onnx
onnx> /evict onnx
mosaic> /restore onnx
onnx> What is my name?
onnx> /exit
```

`/save` overwrites the session's in-memory snapshot with the current live model
context. `/restore` reloads the model if needed and restores that saved
snapshot. `/reset` clears the live context and chat history while keeping the
current model loaded. It does not delete the saved snapshot, so `/restore` can
still return to the last `/save`.

Saved state is process-local. It is not written to disk and is not a persistent
checkpoint format.

## Chat Options

Chat generation options are read from `sessions.txt`.

Common llama.cpp options:

```text
max_tokens=512 temp=0.8 min_p=0.05 seed=1234
```

Common ONNX options:

```text
max_tokens=512 temp=0.4 top_k=20 top_p=0.9 min_p=0.0 repeat_penalty=1.05 seed=1234
```

If a model emits visible turn markers, set `stop_strings` to a comma-separated
list:

```text
stop_strings="<end_of_turn>,<start_of_turn>"
```

For llama.cpp sessions, MosaicVRAM first tries the GGUF chat template, then
falls back to supported built-in templates. You can override detection with
`chat_template=<name>` when needed.

For ONNX sessions, MosaicVRAM supports common tokenizer template marker
families such as:

```text
<|start_header_id|> / <|end_header_id|>
<start_of_turn> / <end_of_turn>
<|im_start|> / <|im_end|>
<|user|> / <|assistant|>
```

These are tokenizer-template markers, not model-name checks.

## Common Errors

`tokenizer.json was not found`

The ONNX session's `tokenizer=` value does not point to a tokenizer JSON file or
directory containing one.

`ONNX LLM required input is unsupported`

The graph exposes a required input that MosaicVRAM does not know how to bind.
The model may need a different export shape.

`attention_bias is not supported in GroupQueryAttention cuda kernel`

ONNX Runtime CUDA rejected the model's `GroupQueryAttention` node shape. Use a
CUDA export where that `attention_bias` input is empty, or use a different
compatible export.

`session is not loaded; use /load <session> first`

The session has no live model/context to save. Load and chat/prefill first.

## Model Compatibility

Only models that pass CLI load, chat, save, evict, restore, and post-restore
chat are considered CLI compatible. See [COMPATIBILITY.md](COMPATIBILITY.md).
