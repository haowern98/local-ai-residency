# MosaicVRAM CLI

The CLI is the small interactive interface for trying a configured llama.cpp
session and manually exercising save, evict, and restore.

Start it from the directory containing your `sessions.txt`:

```text
mosaicvram.exe cli
```

You can also pass a config path explicitly:

```text
mosaicvram.exe cli --config path\to\sessions.txt
```

## sessions.txt

Edit `sessions.txt` manually. The first CLI phase supports llama.cpp sessions:

```text
session llama backend=llama model="C:\models\qwen.gguf" ctx=16384 batch=512 gpu_layers=-1 device=0
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

```text
mosaic> /sessions
mosaic> /use llama
mosaic> /chat
llama> Hello, my name is xjghft. Remember this.
llama> /save
llama> /evict
llama> /restore
llama> What is my name?
llama> /exit
```

`/save` overwrites the session's in-memory snapshot with the current live model
context. `/restore` reloads the model if needed and restores that saved snapshot.
`/reset` clears the live context and chat history while keeping the current model
loaded. It does not delete the saved snapshot, so `/restore` can still return to
the last `/save`. Saved state is process-local; it is not written to disk.
