# dialectic-dataset

Self-verifying synthetic data pipeline for training language models on **deliberative tradeoff reasoning** — teaching models to surface hidden assumptions in `<think>`, identify causal mechanisms and boundary conditions, and commit to falsifiable resolutions instead of hedging or fabricating citations.

- **Interactive Benchmark & Council Audit Explorer**: [Hugging Face Space (`hikewa/dialectic-reasoning`)](https://huggingface.co/spaces/hikewa/dialectic-reasoning)
- **Dataset & Calibration Goldset**: [Hugging Face Dataset (`hikewa/dialectic-reasoning-traces`)](https://huggingface.co/datasets/hikewa/dialectic-reasoning-traces)
- **v1–v8 Goodhart Post-Mortem**: [`docs/post_mortem.md`](docs/post_mortem.md)
- **v9 Self-Verifying Architecture**: [`docs/next_generation_design.md`](docs/next_generation_design.md)

---

## Why This Pipeline Exists: The `v1`–`v8` Goodhart Post-Mortem

In `v1` (255 traces), fine-tuning `Qwen/Qwen3-8B` on dialectic syntheses improved held-out rubric scores from **0.1** (base) to **6.6** (on a `-4` to `10` scale).

When scaling `v2`–`v8`, we introduced a rule-based falsifiability scorer that rewarded surface specificity (`number_count`, `named_entity_count`) and ran best-of-$N$ selection. This created a classic **Goodhart / GIGO feedback loop**:
1. Generator models occasionally invented authoritative-sounding studies (`"A 2022 Harvard study..."`) or arbitrary numeric cutoffs (`"500ms latency budget"`).
2. The surface scorer rewarded those traces with the highest specificity scores, and best-of-$N$ amplified them into the training split.
3. Even worse, non-vague architectural conditionals (`"If the agent mutates external state..."`) were misclassified as `vague_conditionals` unless the model stuffed a number into the sentence.
4. Fine-tuned `v2`–`v8` models learned to sound hyper-specific while fabricating citations (`v2`–`v8` data was removed from the public release and documented in [`docs/post_mortem.md`](docs/post_mortem.md)).

---

## The `v9` 6-Stage Self-Verifying `<think>` Pipeline

To eliminate surface-metric Goodharting, `v9` ([`docs/next_generation_design.md`](docs/next_generation_design.md)) shifts selection from surface rhetorical patterns to **preserved `<think>` deliberation** and **cross-family adversarial verification**:

| Stage | Module / Script | What It Enforces |
| :--- | :--- | :--- |
| **Stage 1: `<think>` Preservation & Mechanistic Falsifiability** | [`scripts/generate_falsifiable_traces.py`](scripts/generate_falsifiable_traces.py)<br>[`src/dialectic_dataset/falsifiability_extractor.py`](src/dialectic_dataset/falsifiability_extractor.py) | Preserves natural first-person `<think>...</think>` blocks; recognizes causal mechanisms and state-transition consequences as specific conditionals without requiring fabricated numbers. |
| **Stage 2: Empirical Claim Verification** | [`src/dialectic_dataset/verification_pipeline.py`](src/dialectic_dataset/verification_pipeline.py)<br>[`scripts/verify_traces.py`](scripts/verify_traces.py) | Audits empirical citations, statistics, historical events, and numeric thresholds across independent model families; **any single fabricated empirical claim vetoes the trace**. |
| **Stage 3: Cross-Family Adversarial Challenge** | [`src/dialectic_dataset/verification_pipeline.py`](src/dialectic_dataset/verification_pipeline.py) | Independent model families attack the weakest point, unexamined assumption, and strongest counterargument; trace passes only if majority agrees `survives_challenge=True`. |
| **Stage 4: `<think>` Deliberation Scorer** | [`src/dialectic_dataset/thinking_quality.py`](src/dialectic_dataset/thinking_quality.py) | Scores direction changes (`"Wait—"`, `"Let me reconsider"`), surfaced assumptions, self-correction, causal mechanisms, and think-to-output divergence while penalizing rubric echoing (`meta_echo_count`). |
| **Stage 5: Verified Training Builder** | [`scripts/build_training.py`](scripts/build_training.py) | Enforces `require_verified=True` and `max_fabrication=0`; ranks best-of-$N$ by `0.6 * thinking_composite + 0.4 * falsifiability_composite` and preserves `<think>` in training messages. |
| **Stage 6: Cross-Family Council Evaluation & Gated Upload** | [`scripts/eval.py`](scripts/eval.py)<br>[`scripts/upload.py`](scripts/upload.py) | Evaluates adapters via `--judge unitares_council`; `upload.py` refuses to upload dataset splits unless `manifest.json` has `require_verified=True` and `max_fabrication=0`. |

---

## Published Models & Datasets

| Artifact | Base / Type | Hugging Face | Notes |
| :--- | :--- | :--- | :--- |
| **Dialectic Qwen3-8B LoRA (`v1`)** | `Qwen/Qwen3-8B` | [`hikewa/dialectic-qwen3-8b-lora`](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora) | `6.6` vs `0.1` base on 14 held-out prompts |
| **Dialectic Qwen2.5-1.5B LoRA (`v1`)** | `Qwen/Qwen2.5-1.5B-Instruct` | [`hikewa/dialectic-qwen2.5-1.5b-lora`](https://huggingface.co/hikewa/dialectic-qwen2.5-1.5b-lora) | Lightweight 1.5B variant (205 examples) |
| **Dialectic Reasoning Traces & Goldset** | Dataset | [`hikewa/dialectic-reasoning-traces`](https://huggingface.co/datasets/hikewa/dialectic-reasoning-traces) | 255 `v1` traces + `goldset_thinking_v1.json` + `v9` council audit |
| **Dialectic Reasoning Explorer** | Gradio Space | [`hikewa/dialectic-reasoning`](https://huggingface.co/spaces/hikewa/dialectic-reasoning) | 3-tab CPU-native benchmark, council audit & live scorer |

---

## Quickstart: Running the Pipeline

```bash
# 1. Generate traces with preserved <think> deliberation blocks
PYTHONPATH=src python scripts/generate_falsifiable_traces.py \
  --providers ollama_gemma4 --seeds 5 --output data/traces_v9_raw.jsonl

# 2. Run Stage 2 verification + Stage 3 adversarial challenge via UNITARES cross-family council
PYTHONPATH=src python scripts/verify_traces.py \
  --input data/traces_v9_raw.jsonl \
  --output data/traces_v9_verified.jsonl \
  --mode council

# 3. Build verified training splits (enforces require_verified=True, max_fabrication=0, keeps <think>)
PYTHONPATH=src python scripts/build_training.py \
  --traces data/traces_v9_verified.jsonl \
  --output-dir data/training_v9 \
  --require-verified \
  --min-thinking-quality 0.45

# 4. Train LoRA adapter
PYTHONPATH=src python scripts/train.py \
  --base-model Qwen/Qwen3-4B \
  --training-dir data/training_v9 \
  --output-dir models/dialectic-qwen3-4b-v9

# 5. Evaluate with cross-family UNITARES council judge
PYTHONPATH=src python scripts/eval.py \
  --adapter-path models/dialectic-qwen3-4b-v9/final \
  --judge unitares_council

# 6. Upload verified dataset / Space / model to Hugging Face
PYTHONPATH=src python scripts/upload.py \
  --dataset-repo hikewa/dialectic-reasoning-traces \
  --training-dir data/training_v9 \
  --goldset data/goldset/goldset_thinking_v1.json \
  --space-repo hikewa/dialectic-reasoning
```

---

## Project Structure

```text
src/dialectic_dataset/
  falsifiability_extractor.py  # Stage 1 mechanistic & conditional falsifiability extractor
  verification_pipeline.py     # Stages 2 & 3 claim verification, adversarial challenge & UNITARES council client
  thinking_quality.py          # Stage 4 <think> deliberative quality & meta-rubric echo scorer
  signal_extractor.py          # Text tokenization and discourse signal extraction
  quality_scorer.py            # Structural & phase-coherence scoring
  trainer.py                   # LoRA SFT training wrapper
scripts/                       # CLI entrypoints (generate, verify_traces, build_training, train, eval, upload)
data/
  goldset/                     # goldset_thinking_v1.json (12 hand-crafted calibration examples)
  seeds_axiom.json             # 16 AI governance & architectural tension seeds
  seeds_expanded.json          # 80 multi-domain tension seeds
  training/                    # Clean v1 training splits (205 train / 25 val / 25 test)
space/                         # Hugging Face Gradio Space (3-tab CPU-native Explorer)
docs/                          # Post-mortem, next-generation design, rubrics, and specs
tests/                         # 128 unit tests covering all 6 pipeline stages
```

## Tests

```bash
PYTHONPATH=src pytest -q
```
