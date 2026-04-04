# Next Generation Design: Self-Verifying Pipeline

## Core Insight

LLMs can detect their own fabrication when asked directly. The v1-v8 pipeline
never asked. The eval question was "is this well-structured?" when it should
have also been "did you make any of this up?"

The verification capability exists in the same models we use for generation
and judging. The pipeline design just needs to use it.

## Design Principles

1. **Ask the obvious questions.** If a human reading the output would ask
   "is that citation real?", the pipeline should ask it too.

2. **Separate generation from verification.** The model that generates should
   not be the only model that evaluates. Different models, different prompts,
   different questions.

3. **Keep the thinking.** Train on reasoning traces, not polished conclusions.
   The thinking is where self-correction happens.

4. **Scores are not quality.** Every score must be accompanied by a dogfood
   step where outputs are actually read. A passing score is necessary but
   not sufficient.

## Pipeline Stages

### Stage 1: Generate with Thinking

Generate traces that include full `<think>` blocks. The thinking should show
genuine reasoning — backtracking, reframing, catching assumptions.

Prompt should explicitly encourage:
- Noticing when you're about to hedge
- Catching yourself making claims you can't verify
- Reframing the question when the obvious framing is too binary

### Stage 2: Self-Verification Pass

After generation, run a separate verification pass on each trace:

```
Given this response, answer these questions:
1. List every factual claim (citations, statistics, studies, named events).
   For each one: are you confident this is real, or did you generate it?
2. Does the response actually resolve the tension, or does it just
   present both sides and hedge?
3. Is there a moment where the reasoning changes direction or discovers
   something non-obvious? If not, the trace is formulaic.
4. Could this response have been written without actually engaging with
   the specific question? (i.e., is it a generic template?)
```

Use a different model or at minimum a different prompt context than generation.
Flag traces that fail any check. Don't just score — get explanations.

### Stage 3: Adversarial Challenge

For traces that pass verification, run an adversarial pass:

```
Here is a response to the question "[question]":
[response]

Your job is to find the weakest point. What's the strongest
counterargument that this response fails to address? What assumption
does it make that it doesn't examine?
```

Traces that survive adversarial challenge with their core argument intact
are stronger candidates for training data. Traces that collapse under
scrutiny should be revised or rejected.

### Stage 4: Thinking Quality Scoring

Score the `<think>` block separately from the output:

- **Direction changes**: Does the thinking change course at least once?
- **Assumption surfacing**: Does it notice and examine its own assumptions?
- **Self-correction**: Does it catch itself hedging or fabricating?
- **Reframing**: Does it question the premise of the question?
- **Specificity of reasoning**: Does it reason through mechanisms, not just
  list perspectives?

A trace with a perfect output but empty/formulaic thinking is worse than
a trace with a messy output but genuine thinking process.

### Stage 5: Build Training Data

- Keep `<think>` blocks (--keep-thinking is now the default)
- Reject any trace with fabricated claims (verified in Stage 2)
- Prefer traces with high thinking quality scores
- No best-of-N by output score alone — weight thinking quality

### Stage 6: Eval with Right Questions

Eval should ask multiple questions, not just structural quality:

| Question | What it catches |
|----------|----------------|
| "Is this well-structured?" | Low-quality formatting |
| "Did the response fabricate any claims?" | Fabrication |
| "Does it actually resolve the tension or just hedge?" | Template responses |
| "Is the reasoning non-obvious?" | Generic balanced policy analysis |
| "Would the response change if you swapped the topic?" | Lack of specificity |

### Stage 7: Mandatory Dogfood

Before any model gets uploaded or used for further training:
- Generate 5 responses to held-out questions
- Read them. Actually read them.
- Answer: "Would I trust this response if I didn't know it was AI?"

This is not automatable. It's the one step that must be manual.

## Open Questions

- **Which model verifies?** Same family but different size? Different provider?
  The verifier should have different blind spots than the generator.

- **How to score thinking quality automatically?** Direction changes and
  self-correction can be detected with heuristics (presence of "wait",
  "actually", "but that assumes", contradiction between early and late
  thinking). But formulaic thinking that mimics these patterns is the
  next failure mode.

- **What does good training data actually look like?** We still don't have
  a gold standard for "genuine reasoning trace." Before building the
  pipeline, we need 10-20 examples of what we're actually trying to produce,
  written or curated by hand.

- **Is "dialectic" still the right frame?** The process we're describing —
  genuine engagement with tensions, self-correction, non-obvious synthesis —
  might be better described as "reflective reasoning" or "deliberative
  thinking" rather than "dialectic synthesis."

## Dependencies

- Verification pass requires API access to a second model (or at minimum,
  a separate prompt context)
- Thinking quality scoring needs heuristic development + calibration against
  human judgment on ~20 examples
- Gold standard examples need to be written by hand before pipeline exists
