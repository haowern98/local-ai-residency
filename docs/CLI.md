# MosaicVRAM CLI

The CLI is the small interactive interface for trying configured sessions and
manually exercising save, evict, and restore.

Start it from the directory containing your `sessions.txt`:

```text
mosaicvram.exe cli
```

You can also pass a config path explicitly:

```text
mosaicvram.exe cli --config path\to\sessions.txt
```

## sessions.txt

Edit `sessions.txt` manually. A llama.cpp session looks like this:

```text
session llama backend=llama model="C:\models\qwen.gguf" ctx=16384 batch=512 gpu_layers=-1 device=0 max_tokens=512 temp=0.8 min_p=0.05 seed=1234
```

ONNX sessions can be listed and validated by the CLI. ONNX chat generation is
implemented in the next phase and will require decoded text output, not token ID
output:

```text
session onnx backend=onnx-llm model="C:\models\deepseek\model.onnx" tokenizer="C:\models\deepseek" prefill_chunk=512 device=0
```

After editing the file while the CLI is open, run:

```text
/reload-config
```

## Commands

```text
/help
/sessions
/use <session>
/chat [session]
/load [session]
/save [session]
/evict [session]
/restore [session]
/reset [session]
/reload-config
/exit
```

`/chat` enters chat mode. In chat mode, normal text goes to the active model.
Lines starting with `/` are still treated as commands.

No session is selected automatically at startup. Use `/use <session>` first or
pass a session directly to a command such as `/chat llama`.

```text
mosaic> /sessions
mosaic> /use llama
mosaic> /chat
llama> Hello, my name is xjghft. Remember this.
llama> /save
llama> /evict
mosaic> /restore
llama> What is my name?
llama> /exit
```

`/save` overwrites the session's in-memory snapshot with the current live model
context. `/restore` reloads the model if needed and restores that saved snapshot.
`/reset` clears the live context and chat history while keeping the current model
loaded. It does not delete the saved snapshot, so `/restore` can still return to
the last `/save`. Saved state is process-local; it is not written to disk.

Chat generation options are read from `sessions.txt`. If omitted, the defaults
are `max_tokens=512`, `temp=0.8`, `min_p=0.05`, and the llama.cpp default seed.
MosaicVRAM first tries the GGUF chat template, then falls back to supported
built-in templates. You can override detection with an explicit template name,
for example `chat_template=gemma`. If a model emits visible turn markers, set
`stop_strings` to a comma-separated list, for example
`stop_strings="<end_of_turn>,<start_of_turn>"`.
