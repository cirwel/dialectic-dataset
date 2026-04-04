# Models

Local LoRA adapters. Not tracked in git (too large). See HuggingFace for hosted versions.

## Public (honest, no fabrication)

| Model | Base | Local Path | HuggingFace |
|-------|------|-----------|-------------|
| 8B v1 | Qwen/Qwen3-8B | `dialectic-lora-qwen3-8b/final/` | [hikewa/dialectic-qwen3-8b-lora](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora) |
| 1.5B v1 | Qwen/Qwen2.5-1.5B-Instruct | `dialectic-lora-1.5b/final/` | [hikewa/dialectic-qwen2.5-1.5b-lora](https://huggingface.co/hikewa/dialectic-qwen2.5-1.5b-lora) |

## Local only (first experiments)

| Model | Base | Local Path |
|-------|------|-----------|
| 0.5B v1 | Qwen/Qwen2.5-0.5B-Instruct | `dialectic-lora/final/` |
| 4B v1 | Qwen/Qwen3-4B | `dialectic-lora-qwen3-4b/final/` |

## Deleted (GIGO, v2-v8)

All v2-v8 adapters deleted locally. Private on HuggingFace (recoverable if needed).
See `docs/post_mortem.md` for why.

## Reconstruction

To reconstruct a full model from a LoRA adapter:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "models/dialectic-lora-qwen3-8b/final")
model = model.merge_and_unload()
```
