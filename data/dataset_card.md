---
language:
- en
license: mit
task_categories:
- text-generation
tags:
- dialectic
- reasoning
- argumentation
- synthesis
- integrative-reasoning
- chain-of-thought
size_categories:
- n<1K
---

# Dialectic Reasoning Traces

Scored dialectic reasoning traces and calibration benchmarks for training models on integrative resolution under conflicting frames. Instead of list-format pros/cons or generic hedging, these traces teach models to identify real tension, surface hidden assumptions in `<think>` deliberation, make conditional commitments grounded in causal mechanisms, and reach specific resolutions.

## What's in This Repository

### 1. Core `v1` Release (`traces.jsonl` + `training/`)
255 scored dialectic reasoning traces (205 train / 25 val / 25 test) across 8 prompt families (51 prompts):
- Generated via Claude Sonnet and filtered by rule-based structural and quality checks (`score >= 0.72`).
- Checked for fabricated citations and statistics (`0` observed across the 255 traces).
- Training-set sizes for public adapters:
  - [`hikewa/dialectic-qwen2.5-1.5b-lora`](https://huggingface.co/hikewa/dialectic-qwen2.5-1.5b-lora): trained on this 205-example release.
  - [`hikewa/dialectic-qwen3-8b-lora`](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora): trained on a 408-example split of a larger 510-trace corpus (per the [training manifest](https://huggingface.co/datasets/hikewa/dialectic-reasoning-traces/blob/ab5811e95678e0b0a1f601b76111e60dfadf25e5/training_combined/manifest.json) in repo history).

### 2. Next-Generation Self-Verifying `<think>` Pipeline & Calibration (`goldset/` + `v9_pilot/`)
Following the [`v1`–`v8` post-mortem](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/post_mortem.md) — which showed that optimizing surface specificity metrics on final outputs created a Goodhart feedback loop that rewarded fabricated citations and arbitrary numbers — the [`v9` self-verifying pipeline](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/next_generation_design.md) enforces six stages:
1. **Stage 1 (`<think>` Preservation)**: Preserves unscripted first-person `<think>...</think>` deliberation traces alongside the visible resolution.
2. **Stage 2 (Empirical Claim Verification)**: Cross-family verifiers audit empirical citations, statistics, and numeric thresholds; any single fabricated claim vetoes the trace.
3. **Stage 3 (Adversarial Challenge)**: Independent model families attack the weakest point, unexamined assumptions, and strongest counterargument; the trace is kept only if the core conclusion survives (`survives_challenge=True`).
4. **Stage 4 (Deliberative `<think>` Scoring)**: Scores direction changes (`Wait—`, `Let me reconsider`), surfaced assumptions, self-correction, causal mechanisms, and think-to-output divergence while penalizing rubric echoing.
5. **Stage 5 (Verified Training Builder)**: Selects best-of-N weighting thinking quality (`0.6`) + mechanistic falsifiability (`0.4`) with hard verification gates (`require_verified=True`, `max_fabrication=0`).
6. **Stage 6 (Cross-Family Council Evaluation)**: Evaluates trained adapters via multi-family council (`unitares_council`).

Included files:
- `goldset/goldset_thinking_v1.json`: 12 hand-crafted calibration examples (`genuine`, `surface_pattern`, `formulaic`) for calibrating `<think>` deliberative scorers and LLM judges.
- `v9_pilot/council_pilot_v9.jsonl`: 5-trace cross-family council verification audit demonstrating zero empirical fabrication and strict Stage 3 adversarial filtering.

## Interactive Explorer

Try the side-by-side 8B benchmark, `<think>` council audit viewer, and live scorer sandbox in the browser:
- **[Dialectic Reasoning Explorer (Hugging Face Space)](https://huggingface.co/spaces/hikewa/dialectic-reasoning)**

## Pipeline & Source Code

Built with [cirwel/dialectic-dataset](https://github.com/CIRWEL/dialectic-dataset).
