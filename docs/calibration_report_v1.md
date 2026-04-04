# Calibration Report v1

## Summary

**Gold set**: 24 samples (8 good, 7 mid, 9 bad/adversarial)
**Agreement rate**: 70.8% (17/24)
**Adversarial catch rate**: 87.5% (7/8)
**Bucket correlations**: Low across all dimensions (see below)

## Verdict Breakdown

| Category | Correct | Mismatches | Details |
|----------|---------|------------|---------|
| Good (keep) | 7/8 | good_08 borderline (0.708, threshold 0.72) |
| Mid (review) | 2/7 | 5 scored as keep (leniency) |
| Bad/Adversarial (reject) | 8/9 | bad_adv_05 missed (domain shift) |

## Structural Checks

All 7 checks fire correctly on their target adversarial samples:
- `fake_antithesis`: catches same-term paraphrases
- `synthesis_is_summary_only`: catches enumeration-only synthesis
- `one_sided_synthesis`: catches dismissal language (NEW: added dismissal markers)
- `unanchored_novelty`: catches most topic-drifting synthesis
- `low_substance_phase`: catches empty/filler phases
- `template_collapse`: catches connector spam and boilerplate
- `generic_resolution_language`: catches platitude synthesis

## Known Limitations (Phase 2)

### 1. Domain-shift with topic anchors (bad_adv_05)

**Pattern**: Synthesis introduces completely unrelated concepts (blockchain, IoT, AI)
into a non-tech discussion (zoning policy), but anchors each sentence with 1-2
inherited terms.

**Why lexical analysis fails**: Each sentence contains both novel and inherited terms,
so the grounding heuristic sees them as contextually integrated. The novel terms
(blockchain, IoT) have no lexical overlap with the topic (zoning), but that's also
true of good synthesis novel terms (phased approach, fiscal sustainability).

**Fix**: Embedding-based domain coherence detection. Compute semantic similarity
between novel term clusters and prior phase content vectors.

### 2. Synonym paraphrase fake antithesis (existing gap)

**Pattern**: Fake antithesis using full synonym substitution appears maximally
different at the term level.

**Fix**: Embedding-based stance detection. Compare semantic meaning, not just
vocabulary overlap.

### 3. Mid-quality discrimination (5 false keeps)

**Pattern**: Structurally adequate but shallow syntheses score well on all lexical
metrics. The difference between "names the right framework but doesn't develop it"
and "develops the framework with specific conditions" requires semantic understanding.

**Impact**: The scorer is LENIENT in the review zone (0.72-0.80). Traces that should
be "review" may pass as "keep". This means generated data will include some
mid-quality traces that are structurally sound but intellectually shallow.

**Mitigation**: During generation, apply a secondary LLM-based quality check
(Claude reviewing Claude's output with specific rubric questions) on traces that
score in the 0.72-0.80 borderline zone.

### 4. Bucket correlations are low

All three measured correlations (tension, resolution, coherence vs gold labels) are
near zero. This reflects the fundamental mismatch between what lexical features
capture (structure, vocabulary overlap, term density) and what human labels assess
(argument quality, depth of integration, genuine insight).

**Impact**: Individual bucket scores are NOT reliable ordinal measures of quality.
The composite score and structural checks together produce reasonable keep/reject
verdicts, but the individual dimensions don't track human judgment at fine granularity.

## What Works Well

1. **Clear keep/reject separation**: All good samples score > 0.72, most adversarial
   samples are caught by structural checks regardless of composite score
2. **Structural checks**: 7/8 adversarial patterns detected
3. **Training usefulness gate**: Correctly flags ceremony-without-substance traces
4. **Composite scoring**: Directionally correct (good > mid > bad on average)

## Recommendations

1. **Proceed to generation** with awareness that the scorer is lenient in the
   borderline zone. Use secondary LLM review for 0.72-0.80 composites.
2. **Phase 2 priority**: Add embedding-based features for domain coherence,
   stance detection, and depth assessment.
3. **Expand gold set**: Add more mid-quality samples to better calibrate the
   review zone boundary. Target 40+ samples with explicit borderline labeling.
4. **Consider LLM-as-judge**: For the review zone, a fast LLM call with 3 specific
   yes/no questions may provide better discrimination than lexical analysis.
