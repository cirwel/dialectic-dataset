# Structural Checks Specification

## Purpose

These are hard reject gates. If any check fires strongly, the trace is rejected
regardless of its weighted quality score. The scorer computes a weighted quality
score AND runs these checks independently. A trace must pass both to be kept.

Each check is a boolean: `True` = reject triggered. The checks are designed to
catch traces that would teach bad reasoning habits even when surface metrics
look decent.

## Implementation Principle

Phase 1 checks are **deterministic and rule-based**. No LLM-as-judge. Use
text similarity (Jaccard, cosine on TF-IDF or simple term vectors),
structural features (sentence count, claim density), and pattern matching.

---

## Check 1: `fake_antithesis`

### Definition
The antithesis does not genuinely oppose the thesis. It restates the thesis
in different words, presents a strawman, or concedes the thesis while
pretending to disagree.

### Triggers (reject if any)
- **High similarity**: Jaccard similarity between thesis and antithesis
  claim sets > 0.6 after stopword removal
- **Negation-only**: Antithesis is the thesis with negation words inserted
  but no new substantive claims (check: remove negation words, recompute
  similarity — if > 0.7, it's negation-only)
- **Concession pattern**: Antithesis contains > 50% agreement language
  ("while this is true", "I agree that", "certainly") before presenting
  any counter-claim
- **No counter-evidence**: Antithesis presents zero pieces of evidence or
  reasoning that are absent from the thesis

### Does NOT trigger
- Antithesis that partially agrees but then pivots to a genuine counter-point
- Antithesis that shares some evidence but draws different conclusions
- Antithesis that addresses a subset of the thesis while challenging the rest

### Implementation guidance
1. Extract claim-bearing sentences from thesis and antithesis
2. Compute pairwise term overlap (Jaccard on content words)
3. Check for negation-only pattern
4. Check for concession-dominant opening
5. Count evidence items unique to antithesis

---

## Check 2: `synthesis_is_summary_only`

### Definition
The synthesis merely restates what thesis and antithesis said without
adding integration, resolution, or new reasoning.

### Triggers (reject if any)
- **High coverage, low novelty**: > 80% of synthesis content words appear
  in thesis + antithesis, AND < 10% of synthesis content words are novel
  (not in either prior phase)
- **List structure**: Synthesis is structured as "Thesis says X, antithesis
  says Y" (check for explicit enumeration of prior positions without
  integration connectors like "therefore", "which means", "this suggests")
- **No resolution claim**: Synthesis contains zero sentences that assert
  a position not present in either thesis or antithesis
- **Balanced repetition**: Synthesis gives roughly equal word count to
  restating each side (within 30%) without a concluding integration

### Does NOT trigger
- Synthesis that references both sides but then builds a new argument
- Synthesis that quotes or paraphrases to set up an integration
- Brief synthesis that is mostly novel reasoning with minimal recap

### Implementation guidance
1. Compute novel term ratio: (synthesis_terms - thesis_terms - antithesis_terms) / synthesis_terms
2. Check for enumeration patterns ("On one hand... on the other hand...")
3. Look for resolution-indicating language that introduces new claims
4. Check if synthesis conclusion differs from both thesis and antithesis conclusions

---

## Check 3: `one_sided_synthesis`

### Definition
The synthesis clearly favors one side without meaningfully engaging with
the other. The dialectic structure is present but the conclusion was
predetermined.

### Triggers (reject if any)
- **Asymmetric reference**: Synthesis references claims from one phase > 3x
  more than the other (measured by term overlap ratio)
- **Dismissal language**: Synthesis contains dismissal of one side
  ("however, this ignores", "this argument fails", "clearly wrong") without
  engaging with the substance being dismissed
- **Conclusion alignment**: Synthesis conclusion has > 0.7 term similarity
  with thesis conclusion OR antithesis conclusion (it just picked a side)
- **Evidence exclusion**: Synthesis cites evidence from only one side

### Does NOT trigger
- Synthesis that engages with both sides but ultimately leans one direction
  with reasoning for why
- Synthesis that explains why one consideration outweighs the other in a
  specific context
- Synthesis that honestly acknowledges an asymmetry in the evidence

### Implementation guidance
1. Compute per-side reference ratios
2. Detect dismissal patterns (negation + opponent reference without engagement)
3. Compare synthesis conclusion against thesis and antithesis conclusions
4. Track evidence provenance (which side's evidence appears in synthesis)

---

## Check 4: `unanchored_novelty`

### Definition
The synthesis introduces substantial claims, evidence, or reasoning that
have no basis in either the thesis or antithesis. The new material was
not synthesized from the dialectic — it was smuggled in.

### Triggers (reject if any)
- **High novelty ratio**: > 40% of synthesis content words are novel AND
  the novel terms form coherent claims (not just filler/connectors)
- **New evidence**: Synthesis cites specific evidence (studies, examples,
  data) that appears in neither thesis nor antithesis
- **Topic drift**: Novel claims in synthesis are semantically distant from
  both thesis and antithesis topics (measured by term vector cosine
  similarity < 0.3 between novel content and prior phases)

### Does NOT trigger
- Synthesis that combines elements from both sides into a new formulation
  (novel phrasing of existing ideas is expected)
- Synthesis that draws a logical conclusion from premises in thesis and
  antithesis, even if the conclusion was never stated
- Synthesis that reframes the problem (reframing is integration, not novelty)

### Implementation guidance
1. Identify novel terms in synthesis
2. Cluster novel terms into claim-bearing segments
3. Compute semantic distance of novel segments from thesis and antithesis
4. Distinguish novel phrasing (low risk) from novel claims (high risk)

---

## Check 5: `low_substance_phase`

### Definition
One or more phases contains too little reasoning to be meaningful. A phase
that is just a label or a single vague sentence does not contribute to
dialectic reasoning.

### Triggers (reject if any)
- **Too short**: Any phase has < 2 sentences or < 30 words of content
  (excluding headers and labels)
- **No claims**: A phase contains zero assertive sentences (all questions,
  all hedging, no positions taken)
- **Pure filler**: > 60% of a phase's sentences are transition phrases,
  meta-commentary ("This is an important question"), or restating the
  prompt without adding reasoning

### Does NOT trigger
- A deliberately concise phase that still makes a clear claim with reasoning
- A phase that is shorter than others but still substantive
- A phase that uses questions rhetorically as part of building an argument

### Implementation guidance
1. Count sentences and words per phase (excluding structural labels)
2. Classify sentences: assertive, question, filler, transition
3. Check minimum thresholds per phase
4. Flag if any phase is < 25% the word count of the longest phase

---

## Check 6: `template_collapse`

### Definition
The trace follows a formulaic template without substantive content. The
structure is correct but the reasoning could be generated by filling in
blanks in a template.

### Triggers (reject if any)
- **Connector spam**: > 30% of sentences begin with discourse markers
  ("However", "Furthermore", "On the other hand", "In contrast",
  "Nevertheless") AND the sentences following these markers are generic
- **Boilerplate opening**: Thesis begins with "There are valid arguments
  on both sides" or equivalent (the conclusion is in the opening)
- **Symmetric structure**: Thesis and antithesis have nearly identical
  sentence structures (same number of sentences, same sentence lengths
  within 20%, same connector patterns) suggesting template fill-in
- **Predictable arc**: Every phase follows exactly the same structure:
  claim, evidence, qualifier, conclusion — with no variation

### Does NOT trigger
- Traces that use some discourse markers naturally as part of reasoning
- Traces that happen to have similar-length phases but different structures
- Traces that follow a logical progression that happens to be common

### Implementation guidance
1. Extract sentence-initial words/phrases
2. Compute discourse marker ratio
3. Compare structural fingerprints between phases
4. Detect boilerplate openings and closings against a reject list

---

## Check 7: `generic_resolution_language`

### Definition
The synthesis uses vague, non-committal language that could apply to any
topic. It resolves nothing specific to the actual tension.

### Triggers (reject if any)
- **Generic phrases**: Synthesis contains > 2 of these patterns (or close
  paraphrases) without following them with specific application:
  - "Both perspectives have merit"
  - "The truth lies somewhere in the middle"
  - "Context matters"
  - "It depends on the situation"
  - "We should consider all perspectives"
  - "A balanced approach is needed"
  - "There are no easy answers"
  - "Reasonable people can disagree"
- **Topic-removable**: The synthesis could apply to a completely different
  topic with minimal word changes (< 20% of content words are
  topic-specific)
- **No conditional specifics**: Synthesis says "it depends" without
  specifying what it depends ON (no if/when/where conditions that reference
  the actual domain)

### Does NOT trigger
- Synthesis that uses one generic phrase as a bridge to specific reasoning
- Synthesis that explicitly explains what makes the situation context-dependent
- Synthesis that provides concrete conditions under which each side applies

### Implementation guidance
1. Maintain a list of generic resolution phrases (expandable)
2. Count occurrences, check if followed by specific application
3. Test topic-removability: mask topic-specific nouns, check if synthesis
   still reads coherently (it shouldn't for a good synthesis)
4. Look for conditional specifics after "depends" language

---

## Anti-Gaming Test Suite

In addition to the 7 structural checks, the gold set must include explicit
adversarial samples that test the scorer's resistance to gaming:

| Adversarial pattern | What it tests | Expected: all rejected |
|---------------------|---------------|------------------------|
| High connector count, low claim density | Template collapse + low substance |
| Thesis and antithesis agree on 80% of claims | Fake antithesis |
| Synthesis is 90% quotation from prior phases | Summary-only synthesis |
| Correct dialectic form, topic is trivially simple | Ritualization risk detection |
| Very long, very verbose, zero insight | Low substance + template collapse |
| Synthesis introduces an entirely new argument | Unanchored novelty |
| "Both sides" resolution with no specifics | Generic resolution language |
| Antithesis is thesis + "but some disagree" | Fake antithesis + low substance |

The scorer must reject ALL of these. If any adversarial sample passes, the
scorer needs recalibration before generating synthetic data.

---

## Threshold Tuning

All numeric thresholds in this document (similarity ratios, word counts,
percentages) are initial values. They should be tuned during gold set
calibration:

1. Run the scorer on all 90 gold set samples
2. Compare scorer verdicts to human labels
3. Adjust thresholds to maximize agreement
4. Re-run and verify adversarial samples are still all rejected
5. Document final thresholds in the scorer's config

The goal is > 80% agreement with human labels on the gold set, with zero
false negatives on adversarial samples (no adversarial trace should pass).
