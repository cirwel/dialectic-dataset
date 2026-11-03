---
title: Dialectic Reasoning Explorer
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.10.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
datasets:
  - hikewa/dialectic-reasoning-traces
models:
  - hikewa/dialectic-qwen3-8b-lora
  - hikewa/dialectic-qwen2.5-1.5b-lora
---

# Dialectic Reasoning Explorer

Interactive benchmark, `<think>` deliberation inspector, and cross-family council verification explorer for the [**dialectic-dataset**](https://github.com/CIRWEL/dialectic-dataset) project.

## What You Can Explore Here (Runs Natively on CPU)

1. **Base vs. LoRA Benchmark (14 Held-Out Prompts)**:
   - Side-by-side outputs and 7-dimension rubric scores comparing base `Qwen/Qwen3-8B` (mean score **0.1**) against [`hikewa/dialectic-qwen3-8b-lora`](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora) (mean score **6.6** on a -4 to 10 scale) across 8 tradeoff questions and 6 framing-resistance questions.
2. **Self-Verifying `<think>` + Cross-Family Council Audit (`v9`)**:
   - Inspect traces from the 6-stage self-verifying pipeline ([`docs/next_generation_design.md`](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/next_generation_design.md)), showing the preserved `<think>` deliberation trace, visible resolution, and independent Stage 2 verification + Stage 3 adversarial challenge verdicts across model families.
3. **Gold-Standard Calibration (`goldset_thinking_v1`) & Live Scorer Sandbox**:
   - Browse the 12 calibration examples (`genuine`, `surface_pattern`, `formulaic`) or paste any `<think>...</think>` trace to run the deterministic thinking-quality and falsifiability scorers in real time.

## Links

- **Code & Pipeline**: [github.com/CIRWEL/dialectic-dataset](https://github.com/CIRWEL/dialectic-dataset)
- **Dataset**: [hikewa/dialectic-reasoning-traces](https://huggingface.co/datasets/hikewa/dialectic-reasoning-traces)
- **Models**: [hikewa/dialectic-qwen3-8b-lora](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora) | [hikewa/dialectic-qwen2.5-1.5b-lora](https://huggingface.co/hikewa/dialectic-qwen2.5-1.5b-lora)
- **Post-Mortem & Next-Gen Design**: [docs/post_mortem.md](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/post_mortem.md) | [docs/next_generation_design.md](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/next_generation_design.md)
