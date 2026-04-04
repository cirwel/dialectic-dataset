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

255 scored dialectic reasoning traces for training models on integrative resolution under conflicting frames.

## What This Is

Each trace contains a **thesis**, **antithesis**, and **synthesis** for a question that admits genuinely competing perspectives. The synthesis integrates both sides rather than picking one or hedging.

All traces were:
- Generated via Claude Sonnet
- Scored by a rule-based quality pipeline (tension quality, resolution quality, cross-phase coherence, synthesis sufficiency)
- Filtered to quality score >= 0.72 with structural checks passed

## Dataset Structure

- `traces.jsonl` — 255 scored traces with full metadata
- `training/train.jsonl` — 205 examples in chat format (messages array)
- `training/val.jsonl` — 25 validation examples
- `training/test.jsonl` — 25 test examples

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

## Intended Use

Fine-tuning small language models to produce better tradeoff reasoning — conditional answers, integrative resolution, resistance to shallow both-sides framing.

## Pipeline

Built with [dialectic-dataset](https://github.com/hikewa/dialectic-dataset) pipeline.
