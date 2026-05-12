# Gemma CUDA ONNX Export Note

Some Gemma 3 ONNX exports wire an `attention_bias` tensor into Microsoft
`GroupQueryAttention`. ONNX Runtime CUDA currently rejects that form:

```text
attention_bias is not supported in GroupQueryAttention cuda kernel
```

MosaicVRAM does not need a runtime code change for this. Use a CUDA-targeted
ONNX Runtime GenAI export where the `GroupQueryAttention` `attention_bias`
input is empty and the explicit KV-cache inputs/outputs remain visible.

## Working Export Shape

The export should still expose:

```text
input_ids
attention_mask
past_key_values.*.key
past_key_values.*.value
present.*.key
present.*.value
logits
```

The tested working Gemma 3 1B export had:

```text
KV cache dtype: float16
logits dtype: float16
past inputs: 52
present outputs: 52
GroupQueryAttention attention_bias input: empty
```

## Export Command

Install the CUDA GenAI builder tooling:

```text
python -m pip install --user onnxruntime-genai-cuda onnx-ir
```

Then export Gemma for CUDA:

```text
python -m onnxruntime_genai.models.builder ^
  -m google/gemma-3-1b-it ^
  -o models\gemma-3-1b-it-oga-cuda-fp16 ^
  -p fp16 ^
  -e cuda ^
  -c models\.hf_builder_cache ^
  --extra_options filename=model_fp16.onnx
```

If the builder fails with a `rope_local_base_freq` config error, use a small
export wrapper that maps the current Gemma config's sliding-attention RoPE theta
to the builder's expected field:

```python
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

if not hasattr(Gemma3TextConfig, "rope_local_base_freq"):
    Gemma3TextConfig.rope_local_base_freq = property(
        lambda self: self.rope_parameters["sliding_attention"]["rope_theta"])

from onnxruntime_genai.models.builder import create_model

create_model(
    "google/gemma-3-1b-it",
    "google/gemma-3-1b-it",
    r"models\gemma-3-1b-it-oga-cuda-fp16",
    "fp16",
    "cuda",
    r"models\.hf_builder_cache",
    filename="model_fp16.onnx",
)
```

## MosaicVRAM Check

Use the exported `model_fp16.onnx` in a normal `residency-run` plan:

```text
session id=1 backend=onnx-llm model="models\gemma-3-1b-it-oga-cuda-fp16\model_fp16.onnx" tokenizer="models\gemma-3-1b-it-oga-cuda-fp16" prompt="Remember the code xjghft." device=0
step op=load session=1
step op=prefill session=1
step op=save session=1
step op=baseline session=1
step op=evict_context session=1
step op=evict_model session=1
step op=reload_model session=1
step op=restore session=1
step op=resume_check session=1
```

Expected result:

```text
session1_backend=onnx-llm
session1_logits_dtype=float16
session1_decode_valid=yes
session1_kv_state_bytes=266240
session1_restored_kv_state_bytes=266240
session1_resume_match=yes
status=ok
```

Do not use CPU fallback for Gemma compatibility. For this project, Gemma support
only counts when CUDA decode and residency restore both pass.
