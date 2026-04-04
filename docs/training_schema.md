# Training Schema

## Purpose

Defines the final JSONL format consumed by Unsloth for fine-tuning.
Every training example is one line of JSON in a `.jsonl` file.

Two output formats exist: **explicit dialectic** and **implicit integrated**.
The training set is a mix of both.

## Dataset Composition

| Format | Proportion | Purpose |
|--------|------------|---------|
| Explicit dialectic | 40% | Teach the model to reason through competing frames visibly |
| Implicit integrated | 60% | Teach the model to produce naturally integrated answers without ceremony |

### Length Variation (within each format)

| Length tier | Proportion | Approximate token range |
|------------|------------|------------------------|
| Short | 30% | 100-250 tokens (assistant response) |
| Medium | 50% | 250-600 tokens |
| Long | 20% | 600-1200 tokens |

Length tiers prevent the model from learning a single rhetorical rhythm.

## Format 1: Explicit Dialectic

The model's response shows the dialectic structure. Not with rigid
thesis/antithesis/synthesis labels, but with visible reasoning through
competing perspectives.

### Schema

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<system_prompt>"
    },
    {
      "role": "user",
      "content": "<question>"
    },
    {
      "role": "assistant",
      "content": "<explicit_dialectic_response>"
    }
  ],
  "metadata": {
    "trace_id": "string",
    "format": "explicit",
    "category": "string",
    "length_tier": "short|medium|long",
    "quality_score": 0.0,
    "tension_quality": 0.0,
    "resolution_quality": 0.0,
    "cross_phase_coherence": 0.0,
    "synthesis_sufficiency": 0.0,
    "structural_checks_passed": true,
    "would_train_on_this": true
  }
}
```

### System Prompt

The system prompt should be minimal and NOT prescribe dialectic structure.
It should describe the reasoning capability, not the output format.

```
You reason carefully through problems by considering competing perspectives
before reaching a conclusion. You identify genuine tensions, engage with
the strongest form of each argument, and integrate insights rather than
picking sides or hedging.
```

### Example (explicit, medium length)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You reason carefully through problems by considering competing perspectives before reaching a conclusion. You identify genuine tensions, engage with the strongest form of each argument, and integrate insights rather than picking sides or hedging."
    },
    {
      "role": "user",
      "content": "Should startups optimize for speed of iteration or code quality?"
    },
    {
      "role": "assistant",
      "content": "The case for speed is straightforward: startups that ship faster learn faster. Every week of polish is a week without user feedback. Companies like early Facebook explicitly chose \"move fast and break things\" because market learning compounds — the startup that validates assumptions in two weeks beats the one that builds perfect code for two months, because the perfect code might solve the wrong problem.\n\nBut this framing has a blind spot. Technical debt isn't free. It compounds too, and in the opposite direction. Airbnb's early engineers have spoken about how shortcuts in their booking system created outages that nearly killed the company. When your codebase becomes hostile to change, iteration speed drops to zero regardless of your stated philosophy. The speed advantage is self-defeating if it destroys your ability to iterate later.\n\nThe resolution isn't \"balance\" — it's recognizing that the two optimize on different time horizons. Speed of iteration dominates when you're searching for product-market fit, because the probability of building the wrong thing is high and code longevity doesn't matter if the product doesn't survive. Code quality dominates once you've found fit and need to scale, because the probability of the codebase surviving is high and each shortcut now has a long expected lifetime. The practical answer: invest in quality at the boundaries (APIs, data models, deployment) that are expensive to change, and accept mess in the interiors (UI, business logic) that you can refactor once direction is clear."
    }
  ],
  "metadata": {
    "trace_id": "exp_001",
    "format": "explicit",
    "category": "design_tradeoff",
    "length_tier": "medium",
    "quality_score": 0.84,
    "tension_quality": 0.88,
    "resolution_quality": 0.82,
    "cross_phase_coherence": 0.85,
    "synthesis_sufficiency": 0.78,
    "structural_checks_passed": true,
    "would_train_on_this": true
  }
}
```

### What makes this explicit

The reasoning is visible: "The case for X... But this framing has a blind
spot... The resolution isn't balance — it's..." The structure is there but
it's natural prose, not labels.

## Format 2: Implicit Integrated

The model's response embodies dialectic thinking without visible structure.
The competing perspectives are woven into a single integrated answer. A
reader would perceive a thoughtful, nuanced response — not a structured
argument.

### Schema

Same JSON schema as explicit, with `"format": "implicit"`.

### Example (implicit, short length)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You reason carefully through problems by considering competing perspectives before reaching a conclusion. You identify genuine tensions, engage with the strongest form of each argument, and integrate insights rather than picking sides or hedging."
    },
    {
      "role": "user",
      "content": "Should we use microservices for our new project?"
    },
    {
      "role": "assistant",
      "content": "For a new project, start with a well-structured monolith. Microservices solve scaling and team coordination problems you don't have yet, but they front-load the distributed systems complexity — service discovery, network failures, data consistency — that will slow your initial development significantly. The teams that successfully run microservices (Netflix, Uber) started as monoliths and split services along fault lines they discovered through operating experience. Design your monolith with clear module boundaries so you can extract services later when you actually know which components need independent scaling or deployment. The architectural decision that matters now isn't monolith-vs-micro — it's whether your internal boundaries are clean enough to split later."
    }
  ],
  "metadata": {
    "trace_id": "imp_001",
    "format": "implicit",
    "category": "design_tradeoff",
    "length_tier": "short",
    "quality_score": 0.79,
    "tension_quality": 0.75,
    "resolution_quality": 0.82,
    "cross_phase_coherence": 0.80,
    "synthesis_sufficiency": 0.76,
    "structural_checks_passed": true,
    "would_train_on_this": true
  }
}
```

### What makes this implicit

The response considers microservices' appeal (scaling, team coordination),
engages with why that appeal is misleading for new projects (front-loaded
complexity), and resolves with a concrete recommendation that honors both
concerns (clean boundaries for future extraction). The dialectic reasoning
happened but the reader sees a direct, integrated answer.

## Converting Between Formats

Generation produces explicit traces (thesis/antithesis/synthesis structure).
The data formatter converts accepted explicit traces to implicit format by:

1. Removing structural markers and phase labels
2. Weaving the three phases into continuous prose
3. Leading with the resolution rather than the tension
4. Preserving the key reasoning from each phase

This conversion can be done by Claude API (format conversion prompt) or
by hand for gold set samples. The conversion should NOT add or remove
substantive content — only restructure.

## Metadata Fields

All metadata fields are for provenance and debugging. Unsloth reads only
the `messages` array. Metadata is preserved in the JSONL but not fed to
the model.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trace_id` | string | yes | Unique identifier for provenance |
| `format` | string | yes | "explicit" or "implicit" |
| `category` | string | yes | Prompt family (e.g., "ethical_tradeoff") |
| `length_tier` | string | yes | "short", "medium", or "long" |
| `quality_score` | float | yes | Composite quality score [0, 1] |
| `tension_quality` | float | yes | Tension quality bucket score |
| `resolution_quality` | float | yes | Resolution quality bucket score |
| `cross_phase_coherence` | float | yes | Coherence bucket score |
| `synthesis_sufficiency` | float | yes | Synthesis sufficiency bucket score |
| `structural_checks_passed` | bool | yes | All 7 checks passed |
| `would_train_on_this` | bool | yes | Training usefulness gate |

## File Layout

```
data/training/
  train.jsonl          # Training split (~80%)
  val.jsonl            # Validation split (~10%)
  test.jsonl           # Test split (~10%)
  manifest.json        # Dataset statistics and generation metadata
```

### manifest.json

```json
{
  "version": "1.0",
  "created": "2026-04-01T12:00:00Z",
  "total_traces_generated": 500,
  "total_traces_accepted": 320,
  "acceptance_rate": 0.64,
  "format_split": {
    "explicit": 0.40,
    "implicit": 0.60
  },
  "length_split": {
    "short": 0.30,
    "medium": 0.50,
    "long": 0.20
  },
  "category_distribution": {
    "ethical_tradeoff": 40,
    "design_tradeoff": 40,
    "safety_vs_capability": 40,
    "conflicting_evidence": 40,
    "policy_ambiguity": 40,
    "strategy_under_constraints": 40,
    "short_practical_judgment": 40,
    "reflective_synthesis": 40
  },
  "quality_stats": {
    "mean_quality_score": 0.81,
    "min_quality_score": 0.72,
    "max_quality_score": 0.95,
    "structural_reject_rate": 0.18,
    "usefulness_reject_rate": 0.08
  },
  "splits": {
    "train": 256,
    "val": 32,
    "test": 32
  }
}
```

## Unsloth Compatibility

Unsloth expects either:
1. A HuggingFace `Dataset` object with a `text` column, or
2. A JSONL file with `messages` arrays (ChatML format)

This schema uses option 2. The `messages` array follows the standard
ChatML convention that Unsloth's `FastLanguageModel` and
`SFTTrainer`/`DPOTrainer` consume directly.

Loading example:
```python
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "train.jsonl", "val": "val.jsonl"})
```

## What NOT to Include

- No EISV signals in the training data (scoring is internal, not supervision)
- No UNITARES-specific terminology
- No tool calls or function schemas
- No governance metadata beyond quality scores
- No explicit "thesis:", "antithesis:", "synthesis:" labels in the model's output
- No meta-commentary about the reasoning process ("Let me consider both sides...")
