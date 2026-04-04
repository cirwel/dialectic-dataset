# Post-Mortem: Dialectic Dataset v1-v8

## Timeline

- **v1**: Basic generation prompt, no falsifiability scoring, no best-of-N. Trained 0.5B, 1.5B, 4B, 8B.
- **v2-v4**: Added structural checks, axiom constraints, quality scoring. Copy-paste versioned.
- **v5-v6**: Added falsifiability scorer. Rewarded `number_count` and `named_entity_count`. Best-of-N selection.
- **v7**: Multi-provider generation (Cerebras, Mistral, Ollama). Expanded to 131 seed questions.
- **v8**: Added grounding penalty to detect fabricated citations. Still fabricated via other patterns.

## The GIGO Feedback Loop

```
LLM generates training data
  -> scorer rewards fabrication (more numbers = higher score)
    -> best-of-N selects most fabricated outputs
      -> fine-tune learns to fabricate
        -> LLM judge can't tell real from fake
          -> scores go up while quality goes down
```

Every component was an LLM evaluating another LLM. Nobody in the loop could verify ground truth.

## Specific Mistakes

### 1. Generation prompt encouraged fabrication
The prompt said "Name real companies, cite real numbers, give your best estimate." LLMs don't know what's real — they know what sounds real. This produced confident-sounding fabrication.

### 2. Scorer rewarded fabrication
`number_count` and `named_entity_count` were positive signals in the falsifiability scorer. More fake numbers = higher score. Best-of-N then selected the most fabricated outputs as training data.

### 3. LLM judge was circular
Mistral-strict rubric measured structural quality (crux clarity, tradeoff specificity) not factual accuracy. A beautifully structured lie scores 10/10.

### 4. Training stripped all reasoning
`strip_think()` removed `<think>` blocks from every training example. Models never saw reasoning traces — they learned to produce polished conclusions without intermediate thinking. The empty `<think></think>` tags in v8 outputs were vestigial memory of a capability we trained away.

### 5. No dogfooding until the end
7 iterations of "improving" before anyone sat down and read outputs side-by-side. If we'd dogfooded at v3, we'd have caught the regression.

### 6. Copy-paste versioning
9 train scripts, 11 eval scripts, 5 build scripts — all nearly identical, diverging through accumulated edits. Made it impossible to see what actually changed between versions.

## What Survived

Out of 15 models across 8 versions, 2 don't fabricate:
- **8B v1** (hikewa/dialectic-qwen3-8b-lora) — honest, well-structured
- **1.5B v1** (hikewa/dialectic-qwen2.5-1.5b-lora) — safe, never fabricates

Both are from v1, before the optimization pipeline existed.

## Deeper Problem: No Genuine Synthesis

Even the honest v1 models don't do real dialectic reasoning. They produce:
1. Perspective A
2. Perspective B
3. "The answer depends on context"

Real synthesis means the tension itself reveals something neither side alone sees. The models never get there because the training data never contained it — LLMs generating "dialectic reasoning" produce balanced policy analysis.

## Key Insight: Train on Thinking, Not Output

The thinking process is where reasoning happens: wrestling with tensions, noticing assumptions, catching yourself hedging, reframing the question. We threw it away.

One of our fine-tuned models suggested "explicitly increasing latency" — more tokens before answering = more computational steps = better chance of genuine reasoning. This is the same principle behind o1, DeepSeek-R1, and Qwen3's thinking mode. We optimized for the opposite.

## Structural Safeguards (Now Built In)

- `build_training.py --keep-thinking`: Preserves `<think>` blocks in training data
- `build_training.py --max-fabrication 0`: Rejects traces with fabricated citations (default)
- `dogfood.py`: Side-by-side comparison of all local models on the same question
- `archive/`: GIGO artifacts isolated with clear warning, `.gitignored`
- Scripts parameterized: one `train.py`, one `eval.py` — no more copy-paste versioning

## If We Do Future Training

1. Keep thinking traces — don't strip `<think>` blocks
2. Curate for traces where thinking does something — changes direction, catches assumptions
3. Drop "dialectic" as a format constraint — train for reasoning process, not output template
4. Dogfood after every version
5. Use human-verified gold set or verifiable ground truth
6. Fabrication detection that checks claims, not just structure
