# dialectic-dataset

Training pipeline for dialectic reasoning models — fine-tuning LLMs to engage with competing perspectives, identify tensions, and reason toward grounded conclusions.

## Working Models

| Model | Base | HuggingFace | Notes |
|-------|------|-------------|-------|
| **8B v1** | Qwen/Qwen3-8B | [hikewa/dialectic-qwen3-8b-lora](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora) | Best overall |
| **1.5B v1** | Qwen/Qwen2.5-1.5B-Instruct | [hikewa/dialectic-qwen2.5-1.5b-lora](https://huggingface.co/hikewa/dialectic-qwen2.5-1.5b-lora) | Lightweight |

Try them: [HuggingFace Space](https://huggingface.co/spaces/hikewa/dialectic-reasoning)

## Project Structure

```
src/dialectic_dataset/    # Core library (scoring, formatting, training, generation)
scripts/                  # Pipeline scripts (generate, build, train, eval, upload)
tests/                    # Unit tests
data/                     # Training data, seeds, gold set
models/                   # Local LoRA adapters (not in git)
space/                    # HuggingFace Gradio Space
docs/                     # Specs, rubrics, post-mortem
archive/                  # GIGO artifacts from v2-v8 (not in git, do not reuse)
```

## Pipeline

```bash
# 1. Generate traces
python scripts/generate_diverse_traces.py

# 2. Build training data (rejects fabrication, optional: keep thinking traces)
python scripts/build_training.py --traces data/traces.jsonl --output-dir data/training --keep-thinking

# 3. Train
python scripts/train.py --base-model Qwen/Qwen3-8B --training-dir data/training --output-dir models/my-lora

# 4. Evaluate
MISTRAL_API_KEY=... python scripts/eval.py --adapter-path models/my-lora/final

# 5. Dogfood (read the outputs yourself)
python scripts/dogfood.py

# 6. Upload
python scripts/upload.py --repo hikewa/my-model --adapter-dir models/my-lora/final
```

## Lessons Learned

See [docs/post_mortem.md](docs/post_mortem.md) for the full v1-v8 post-mortem. Key takeaway: train on thinking traces, not polished outputs. Dogfood after every version.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
