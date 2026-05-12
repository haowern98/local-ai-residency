# Test Fixtures

`15k_prompt.txt` is a realistic long-context text fixture used for GPU
residency tests.

The fixture is shared across backends. Exact token counts are tokenizer-specific:

- llama.cpp GGUF models tokenize it with the model's GGUF tokenizer.
- ONNX LLM tests pass it through MosaicVRAM's tokenizer adapter with the
  matching model tokenizer before calling `Ort::Session`.

Do not treat a generated token-ID file as universal across models.
