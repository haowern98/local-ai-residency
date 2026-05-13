# MosaicVRAM Shell

The shell is an interactive runtime control surface for manually loading,
chatting with, saving, evicting, reloading, and restoring configured sessions.
It is meant for quick local inspection. Reproducible validation should still use
`residency-run` plans.

Start it with:

```text
mosaicvram.exe shell --config sessions.txt
```

If the config file is missing or empty, the shell starts anyway and prints the
minimal `add` commands needed to create a llama.cpp or ONNX LLM session.

## Session Config

The config file contains session declarations:

```text
session llama backend=llama model="C:\models\qwen.gguf" ctx=16384 batch=512 gpu_layers=-1 device=0
session onnx backend=onnx-llm model="C:\models\onnx-model\model.onnx" tokenizer="C:\models\onnx-model" prefill_chunk=512 device=0
```

The shell accepts the same declarations through `add`:

```text
add llama backend=llama model="C:\models\qwen.gguf" ctx=16384 batch=512 gpu_layers=-1 device=0
add onnx backend=onnx-llm model="C:\models\onnx-model\model.onnx" tokenizer="C:\models\onnx-model" prefill_chunk=512 device=0
```

Invalid sessions stay visible in `sessions` output with an `issue` column, so a
missing `backend`, `model`, or ONNX `tokenizer` can be repaired with `set`.
Changing `backend`, `model`, or `tokenizer` on a loaded session requires
evicting the model first.

## Commands

```text
sessions
status
show <session>
add <session> backend=... model=...
set <session> <key> <value>
remove <session>
use <session>
load [session]
prefill [session] "text"
chat [session] "text" [max_tokens=N]
save [session]
evict [session] context|model
reload [session]
restore [session]
resume-check [session]
save-config [path]
load-config <path>
exit
```

`chat` auto-loads an unloaded session. For llama.cpp sessions, prompt text is
tokenized by llama.cpp. For ONNX LLM sessions, `tokenizer=` must point to a
usable Hugging Face `tokenizer.json` file or a directory containing one. If the
tokenizer path is missing or unsupported, ONNX chat fails with an error instead
of accepting raw token IDs as a fallback.

ONNX chat currently prints generated token IDs because text decoding is not part
of this phase:

```text
onnx generated token_ids=1839,128013,271
```

## Relationship To Residency Plans

The shell and `residency-run` are independent entry points. Editing a shell
session with `set` changes only the in-memory shell runtime until `save-config`
is used. It does not rewrite any residency plan. Conversely, running
`residency-run --plan ...` reads only that plan file and does not load shell
configuration.
