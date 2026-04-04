# Gold Set Labeling Rubric

## Purpose

This rubric defines how to score dialectic reasoning traces for the gold set.
The gold set calibrates the automated scorer. If the scorer disagrees with
human labels on the gold set, the scorer is wrong.

## Gold Set Composition

- **Minimum 90 samples**: 30 good, 30 mid-quality, 30 bad
- **Adversarial bad samples**: at least 10 of the 30 bad samples should be
  adversarial — traces that look superficially correct but teach bad habits
- **Expect growth**: the gold set should grow to 150-250 as scorer calibration
  reveals edge cases

## Labeling Dimensions

Each sample receives scores on 7 dimensions.

### 1. Tension Quality (1-5)

Does the trace present genuinely competing frames?

| Score | Criteria |
|-------|----------|
| 5 | Two strong, independently defensible positions that create real friction |
| 4 | Clear disagreement with substantive reasons on both sides |
| 3 | Identifiable tension but one side is noticeably weaker or less developed |
| 2 | Superficial disagreement — positions are not genuinely in conflict |
| 1 | No real tension — fake antithesis, strawman, or thesis restated negatively |

**What to look for:**
- Can a reasonable person hold each position independently? (5)
- Is the antithesis a genuine counter-position or a weak restatement? (1-2)
- Are there different underlying values, assumptions, or priorities driving each side? (4-5)
- Is the disagreement over substance or framing? Substance scores higher.

### 2. Resolution Quality (1-5)

Does the synthesis integrate rather than summarize, dodge, or pick sides?

| Score | Criteria |
|-------|----------|
| 5 | Synthesis reveals a higher-order frame that genuinely resolves the tension while preserving what matters from both sides |
| 4 | Synthesis integrates both perspectives with clear reasoning about when/how each applies |
| 3 | Synthesis acknowledges both sides and offers a partial resolution, but leaves some tension unaddressed |
| 2 | Synthesis mostly summarizes or restates both positions without genuine integration |
| 1 | Synthesis picks a side, introduces unrelated claims, or offers empty platitudes |

**What to look for:**
- Does the synthesis contain reasoning that is NOT in either thesis or antithesis? (4-5)
- Does it explain WHY the integration works, not just THAT it exists? (4-5)
- Would removing the synthesis lose information beyond what thesis + antithesis already said? (4-5)
- Is the synthesis just "both have merit" or "it depends on context" without specifics? (1-2)

### 3. Coherence (1-5)

Do the three phases form a logically connected argument?

| Score | Criteria |
|-------|----------|
| 5 | Each phase explicitly builds on the previous one; synthesis references specific points from both sides |
| 4 | Clear logical flow; synthesis addresses the core tension raised |
| 3 | Phases are related but synthesis could have been written independently |
| 2 | Phases feel disconnected; synthesis doesn't clearly address the tension |
| 1 | Phases contradict each other or the synthesis is unrelated to the tension |

**What to look for:**
- Does the antithesis directly respond to specific claims in the thesis? (4-5)
- Does the synthesis reference specific elements from both thesis and antithesis? (4-5)
- Could you shuffle the phases and it would read the same? If so, score low. (1-2)

### 4. Training Usefulness (keep / review / reject)

Would training on this trace teach good reasoning habits?

| Verdict | Criteria |
|---------|----------|
| keep | Would improve model's ability to consider competing frames and integrate them |
| review | Ambiguous — has some value but also some concerning patterns |
| reject | Would teach bad habits even if the trace is technically coherent |

**Things that push toward reject:**
- Verbose filler that inflates appearance of depth
- Correct form (thesis/antithesis/synthesis) with no actual reasoning
- Overuse of transition phrases without substance
- Resolution that a human would find unsatisfying or evasive
- Any pattern you wouldn't want a model to reproduce

### 5. Ritualization Risk (low / medium / high)

Does this trace teach empty formalism?

| Level | Criteria |
|-------|----------|
| low | The dialectic structure serves the reasoning — form follows function |
| medium | Some structural ceremony but the reasoning is still present underneath |
| high | The trace is performing dialectic rather than doing it — remove the structure and nothing of substance remains |

**Key test:** Strip all structural markers (thesis/antithesis/synthesis labels, transition phrases, hedging connectors). Is there still a substantive argument underneath? If yes: low risk. If no: high risk.

### 6. Would I Train On This (yes / no)

Final binary judgment. This is the label that matters most.

- **yes**: This trace, if the model learned to reason this way, would make it better at integrative resolution
- **no**: This trace, even if scored well on other dimensions, should not enter the training set

This is the gut check. If you hesitate, mark **no**.

### 7. Overall Verdict (keep / review / reject)

Final disposition for the gold set.

- **keep**: Score >= 4 average on dimensions 1-3, training usefulness = keep, ritualization risk = low, would_i_train_on_this = yes
- **review**: Mixed signals — some dimensions strong, others weak
- **reject**: Score < 3 average on dimensions 1-3, OR training usefulness = reject, OR ritualization risk = high, OR would_i_train_on_this = no

## Labeling Process

1. Read the full trace without looking at other labels
2. Score dimensions 1-3 (quantitative) first
3. Then score 4-6 (qualitative) based on overall impression
4. Assign overall verdict last
5. Write a 1-sentence justification for the overall verdict

## Calibration Checks

After labeling all 90 samples:
- All "good" samples should have overall verdict = keep (if not, rubric or sample is wrong)
- All adversarial bad samples should have overall verdict = reject
- Mid-quality samples should be mixed (mostly review, some keep, some reject)
- Inter-rater agreement on would_i_train_on_this should be > 80% (if two labelers)

## Edge Cases

- **Short but good**: A brief trace with genuine tension and clean resolution scores high even if terse. Length is not quality.
- **Long but empty**: A verbose trace with lots of words but no real integration scores low. Length is not quality.
- **Genuinely irreconcilable tensions**: Some tensions don't resolve cleanly. A trace that honestly acknowledges irreconcilability while explaining what each side gets right can score 3-4 on resolution quality. Not every synthesis needs to fully resolve.
- **Domain-specific knowledge**: Score the reasoning, not the factual accuracy. A trace with wrong facts but good dialectic structure is a labeling edge case — mark it review and note the issue.
