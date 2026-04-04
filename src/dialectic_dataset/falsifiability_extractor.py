"""Falsifiability feature extraction from dialectic responses.

Measures whether claims are specific enough to be wrong, whether conditionals
are invertible, and whether predictions are testable. All functions are pure,
deterministic, and stdlib-only. No LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dialectic_dataset.signal_extractor import (
    _HEDGING_WORDS,
    sentences,
    tokenize,
    word_count as _word_count,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMITMENT_PHRASES: List[str] = [
    "the answer is",
    "this fails when",
    "the evidence shows",
    "the determining factor is",
    "the key variable is",
    "the data demonstrates",
    "in practice",
    "the mechanism is",
    "this breaks down when",
    "the threshold is",
    "the cutoff is",
    "the deciding factor",
    "the critical difference",
    "what actually happens is",
    "the constraint is",
    "the bottleneck is",
    "the root cause is",
    "the failure mode is",
    "concretely",
    "specifically",
]

VAGUE_CONDITIONAL_PHRASES: List[str] = [
    "when context allows",
    "in some cases",
    "depending on circumstances",
    "when appropriate",
    "as needed",
    "where applicable",
    "when possible",
    "in certain situations",
    "under some conditions",
    "as the case may be",
    "when the time is right",
    "when conditions permit",
    "in the right context",
    "depending on the situation",
    "under the right circumstances",
    "in various contexts",
    "depending on various factors",
]

# Regex patterns for falsifiable content
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand))?\b')
_COMPARISON_RE = re.compile(
    r'\b(?:outperforms?|exceeds?|surpasses?|beats?|more\s+than|less\s+than|'
    r'greater\s+than|fewer\s+than|higher\s+than|lower\s+than|faster\s+than|'
    r'slower\s+than|doubles?|triples?|halves?|reduces?\s+by|increases?\s+by)\b',
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r'\b(?:because|causes?|leads?\s+to|results?\s+in|produces?|creates?|'
    r'generates?|triggers?|prevents?|enables?|drives?)\b',
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r'\b(?:within\s+\d+|after\s+\d+|before\s+\d+|by\s+\d{4}|'
    r'in\s+\d+\s+(?:days?|weeks?|months?|years?|hours?|quarters?)|'
    r'over\s+\d+\s+(?:days?|weeks?|months?|years?)|'
    r'per\s+(?:year|month|quarter|week|day))\b',
    re.IGNORECASE,
)
# Proper nouns: capitalized words not at sentence start, multi-word names,
# or single caps in mid-sentence
_PROPER_NOUN_RE = re.compile(r'(?<=[.!?]\s)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)')
_MID_SENTENCE_CAP_RE = re.compile(r'(?<=[a-z]\s)[A-Z][a-z]{2,}')

_CONDITIONAL_START_RE = re.compile(
    r'\b(?:if|when|provided\s+that|assuming\s+that|in\s+cases?\s+where|'
    r'given\s+that|unless|only\s+when|whenever)\b',
    re.IGNORECASE,
)

# Grounding detection — fabricated citation patterns
_FABRICATED_CITATION_RE = re.compile(
    r'(?:'
    # "a 2022 Harvard study", "the 2023 MIT report"
    r'\b(?:a|the|an)\s+(?:recent\s+)?20\d\d\s+(?:\w+\s+){0,4}'
    r'(?:study|report|survey|trial|analysis|review|paper|research|investigation)'
    r'|'
    # "Harvard's 2022 study", "MIT's internal report"
    r"\b[A-Z][a-z]+(?:'s|'s)\s+(?:\w+\s+){0,3}(?:20\d\d\s+)?"
    r'(?:study|report|survey|trial|analysis|review|paper|research|investigation)'
    r'|'
    # "a study by MIT in 2022", "research by Stanford"
    r'\b(?:study|report|survey|research)\s+(?:by|from|at)\s+[A-Z]'
    r')',
    re.IGNORECASE,
)
_ATTRIBUTION_RE = re.compile(
    r'\b(?:according\s+to|found\s+that|showed\s+that|demonstrated\s+that|'
    r'reported\s+that|concluded\s+that|as\s+(?:measured|documented|seen)\s+in)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class FalsifiabilityFeatures:
    """Features measuring falsifiability of a response."""
    word_count: int
    sentence_count: int

    # Specificity
    number_count: int
    named_entity_count: int
    comparison_count: int
    causal_claim_count: int
    temporal_claim_count: int

    # Commitment vs hedging
    commitment_phrase_count: int
    hedge_count: int
    hedge_to_commitment_ratio: float

    # Conditional quality
    total_conditionals: int
    specific_conditionals: int
    vague_conditionals: int
    conditional_specificity: float

    # Prediction density
    falsifiable_sentence_count: int
    prediction_density: float

    # Grounding
    fabricated_citation_count: int = 0
    ungrounded_stat_count: int = 0

    # Composite
    specificity_density: float = 0.0


@dataclass
class FalsifiabilityScore:
    """Tier 1 falsifiability assessment."""
    features: FalsifiabilityFeatures

    specificity_score: float
    commitment_score: float
    conditional_quality_score: float
    prediction_score: float

    composite: float
    tier1_pass: bool

    components: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _count_numbers(text: str) -> int:
    """Count numeric values (numbers, percentages, quantities)."""
    return len(_NUMBER_RE.findall(text))


def _count_named_entities(text: str) -> int:
    """Count proper nouns as a proxy for named entities.

    Looks for capitalized words in mid-sentence positions and
    multi-word capitalized sequences.
    """
    # Multi-word proper nouns
    multi = _PROPER_NOUN_RE.findall(text)
    # Single capitalized words mid-sentence
    single = _MID_SENTENCE_CAP_RE.findall(text)
    # Deduplicate by combining
    entities = set()
    for m in multi:
        entities.add(m.strip().lower())
    for s in single:
        entities.add(s.strip().lower())
    return len(entities)


def _count_comparisons(text: str) -> int:
    return len(_COMPARISON_RE.findall(text))


def _count_causal_claims(text: str) -> int:
    return len(_CAUSAL_RE.findall(text))


def _count_temporal_claims(text: str) -> int:
    return len(_TEMPORAL_RE.findall(text))


def _count_commitment_phrases(text: str) -> int:
    lower = text.lower()
    return sum(1 for p in COMMITMENT_PHRASES if p in lower)


def _count_hedges(text: str) -> int:
    tokens = tokenize(text)
    return sum(1 for w in tokens if w in _HEDGING_WORDS)


def _extract_conditional_clauses(text: str) -> List[Tuple[str, bool]]:
    """Extract conditional clauses and classify as specific vs vague.

    Returns list of (clause_text, is_specific) tuples.
    A conditional is specific if it contains a number, named entity, or
    concrete noun. It's vague if it matches a vague conditional phrase.
    """
    sents = sentences(text)
    results = []

    for sent in sents:
        if not _CONDITIONAL_START_RE.search(sent):
            continue

        # Check for vague conditional patterns
        lower = sent.lower()
        is_vague = any(vp in lower for vp in VAGUE_CONDITIONAL_PHRASES)

        if is_vague:
            results.append((sent, False))  # not specific
            continue

        # Check for specificity markers: numbers, entities, comparisons
        has_number = bool(_NUMBER_RE.search(sent))
        has_entity = bool(_MID_SENTENCE_CAP_RE.search(sent))
        has_comparison = bool(_COMPARISON_RE.search(sent))
        has_temporal = bool(_TEMPORAL_RE.search(sent))

        is_specific = has_number or has_entity or has_comparison or has_temporal
        results.append((sent, is_specific))

    return results


def _count_fabricated_citations(text: str) -> int:
    """Count patterns like 'a 2022 Harvard study' or 'the 2023 MIT report'."""
    return len(_FABRICATED_CITATION_RE.findall(text))


def _count_ungrounded_stats(text: str) -> int:
    """Count statistics attached to attribution patterns.

    Looks for sentences with both an attribution phrase and a number,
    which suggests a fabricated "according to X, Y%" pattern.
    """
    sents = sentences(text)
    count = 0
    for sent in sents:
        has_attribution = bool(_ATTRIBUTION_RE.search(sent))
        has_number = bool(_NUMBER_RE.search(sent))
        if has_attribution and has_number:
            count += 1
    return count


def _is_falsifiable_sentence(sent: str) -> bool:
    """Heuristic: does this sentence make a falsifiable claim?

    True if the sentence contains at least 2 of:
    - A number or percentage
    - A comparison ("X outperforms Y")
    - A causal claim ("because X, Y")
    - A temporal claim ("within N months")
    - A named entity with a verb
    """
    signals = 0
    if _NUMBER_RE.search(sent):
        signals += 1
    if _COMPARISON_RE.search(sent):
        signals += 1
    if _CAUSAL_RE.search(sent):
        signals += 1
    if _TEMPORAL_RE.search(sent):
        signals += 1
    if _MID_SENTENCE_CAP_RE.search(sent):
        signals += 1
    return signals >= 2


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_falsifiability_features(text: str) -> FalsifiabilityFeatures:
    """Extract all falsifiability features from a response text."""
    wc = _word_count(text)
    sents = sentences(text)
    sc = max(1, len(sents))

    # Specificity counts
    nums = _count_numbers(text)
    entities = _count_named_entities(text)
    comparisons = _count_comparisons(text)
    causals = _count_causal_claims(text)
    temporals = _count_temporal_claims(text)

    # Commitment vs hedging
    commits = _count_commitment_phrases(text)
    hedges = _count_hedges(text)
    total_ch = hedges + commits
    h2c_ratio = hedges / total_ch if total_ch > 0 else 0.5

    # Conditional quality
    conditionals = _extract_conditional_clauses(text)
    total_cond = len(conditionals)
    specific_cond = sum(1 for _, is_spec in conditionals if is_spec)
    vague_cond = sum(1 for _, is_spec in conditionals if not is_spec)
    cond_spec = specific_cond / total_cond if total_cond > 0 else 0.0

    # Prediction density
    falsifiable_sents = sum(1 for s in sents if _is_falsifiable_sentence(s))
    pred_density = falsifiable_sents / sc

    # Grounding
    fab_citations = _count_fabricated_citations(text)
    ungrounded_stats = _count_ungrounded_stats(text)

    # Specificity density
    spec_density = (nums + entities + comparisons) / max(1, wc)

    return FalsifiabilityFeatures(
        word_count=wc,
        sentence_count=sc,
        number_count=nums,
        named_entity_count=entities,
        comparison_count=comparisons,
        causal_claim_count=causals,
        temporal_claim_count=temporals,
        commitment_phrase_count=commits,
        hedge_count=hedges,
        hedge_to_commitment_ratio=h2c_ratio,
        total_conditionals=total_cond,
        specific_conditionals=specific_cond,
        vague_conditionals=vague_cond,
        conditional_specificity=cond_spec,
        falsifiable_sentence_count=falsifiable_sents,
        prediction_density=pred_density,
        fabricated_citation_count=fab_citations,
        ungrounded_stat_count=ungrounded_stats,
        specificity_density=spec_density,
    )


def score_falsifiability(features: FalsifiabilityFeatures) -> FalsifiabilityScore:
    """Compute falsifiability score from extracted features.

    Returns a FalsifiabilityScore with sub-scores and composite.
    """
    sc = max(1, features.sentence_count)

    # 1. Specificity score — shifted from raw number count toward mechanisms
    raw_spec = (
        features.number_count * 0.5
        + features.named_entity_count * 0.3
        + features.comparison_count
        + features.causal_claim_count * 0.5
    ) / sc
    specificity = min(1.0, raw_spec / 1.5)
    if (features.named_entity_count == 0 and features.number_count == 0
            and features.causal_claim_count == 0):
        specificity *= 0.5  # penalty for no concrete referents at all

    # 2. Commitment score
    commitment = 1.0 - features.hedge_to_commitment_ratio
    commitment = max(0.0, min(1.0, commitment))

    # 3. Conditional quality score
    if features.total_conditionals == 0:
        cond_quality = 0.3  # mild penalty — some questions don't need conditionals
    else:
        vague_ratio = features.vague_conditionals / features.total_conditionals
        cond_quality = features.conditional_specificity * 0.7 + (1.0 - vague_ratio) * 0.3

    # 4. Prediction score
    prediction = min(1.0, features.prediction_density / 0.4)

    # 5. Grounding penalty — penalize fabricated citations and ungrounded stats
    grounding_penalty = min(
        0.3,
        features.fabricated_citation_count * 0.10
        + features.ungrounded_stat_count * 0.03,
    )

    # Composite
    composite = (
        0.30 * specificity
        + 0.25 * commitment
        + 0.25 * cond_quality
        + 0.20 * prediction
    )
    composite = max(0.0, composite - grounding_penalty)

    # Tier 1 pass/fail
    tier1_pass = composite >= 0.45

    # Hard fails
    if (features.vague_conditionals > features.specific_conditionals
            and features.total_conditionals > 0):
        tier1_pass = False
    if features.prediction_density < 0.10 and features.sentence_count >= 3:
        tier1_pass = False
    if features.fabricated_citation_count >= 3:
        tier1_pass = False

    components = {
        "specificity": specificity,
        "commitment": commitment,
        "conditional_quality": cond_quality,
        "prediction": prediction,
        "grounding_penalty": grounding_penalty,
    }

    return FalsifiabilityScore(
        features=features,
        specificity_score=specificity,
        commitment_score=commitment,
        conditional_quality_score=cond_quality,
        prediction_score=prediction,
        composite=composite,
        tier1_pass=tier1_pass,
        components=components,
    )
