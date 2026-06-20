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
size_categories:
- n<1K
---

# Dialectic Reasoning Traces

255 scored dialectic reasoning traces for training models on integrative resolution under conflicting frames. Instead of list-format pros/cons or generic hedging, these traces teach models to identify real tension, make conditional commitments, and reach specific resolutions.

## Version Note

This dataset contains only v1 traces — the clean, non-fabricating training data. An earlier version on this repo included augmented data from later pipeline iterations (v2-v8) that suffered from a [GIGO feedback loop](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/post_mortem.md) where the scorer rewarded fabrication and best-of-N amplified it. That data has been removed. The two models trained on this v1 data ([8B](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora), [1.5B](https://huggingface.co/hikewa/dialectic-qwen2.5-1.5b-lora)) are the only ones that don't fabricate.

## What This Is

Each trace contains a **thesis**, **antithesis**, and **synthesis** for a question that admits genuinely competing perspectives. The synthesis integrates both sides rather than picking one or hedging.

All traces were:
- Generated via Claude Sonnet
- Scored by a rule-based quality pipeline (tension quality, resolution quality, cross-phase coherence, synthesis sufficiency)
- Filtered to quality score >= 0.72 with structural checks passed
- Verified to contain no fabricated citations or statistics

## Dataset Structure

| Split | Examples |
|-------|----------|
| Train | 205 |
| Val | 25 |
| Test | 25 |

Training examples are formatted as either:
- **Explicit** (40%): thesis + antithesis + synthesis shown as separate paragraphs
- **Implicit** (60%): synthesis-led integrated response

## Categories (8 families, 51 prompts)

| Category | Count |
|----------|-------|
| ethical_tradeoff | 35 |
| design_tradeoff | 35 |
| short_practical_judgment | 35 |
| safety_vs_capability | 30 |
| conflicting_evidence | 30 |
| policy_ambiguity | 30 |
| strategy_under_constraints | 30 |
| reflective_synthesis | 30 |

## Quality Scores

- Mean: 0.769
- Range: [0.721, 0.837]
- Acceptance rate: 94.1%

## Limitations

These traces produce structured dialectic output but not genuine dialectic reasoning. The models trained on this data present "Perspective A, Perspective B, it depends" rather than synthesizing insights that neither perspective alone reveals. See the [post-mortem](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/post_mortem.md) for analysis and the [next-generation design](https://github.com/CIRWEL/dialectic-dataset/blob/main/docs/next_generation_design.md) for planned improvements.

## Intended Use

Fine-tuning small language models to produce better tradeoff reasoning — conditional answers, integrative resolution, resistance to shallow both-sides framing.

## Models Trained on This Data

- [hikewa/dialectic-qwen3-8b-lora](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora) — Best overall, honest
- [hikewa/dialectic-qwen2.5-1.5b-lora](https://huggingface.co/hikewa/dialectic-qwen2.5-1.5b-lora) — Lightweight, never fabricates

Try them: [HuggingFace Space](https://huggingface.co/spaces/hikewa/dialectic-reasoning)

## Pipeline

Built with [dialectic-dataset](https://github.com/CIRWEL/dialectic-dataset).
