"""Stage 4: Thinking quality extraction and scoring for <think> blocks.

Evaluates whether a trace's ``<think>...</think>`` block performs genuine
deliberative work — changing direction, surfacing hidden assumptions,
catching itself hedging or fabricating, reframing binary premises, and
reasoning through mechanisms — rather than emitting empty tags or a
polished summary template.

All functions are pure, deterministic, and stdlib-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dialectic_dataset.signal_extractor import (
    jaccard,
    sentences,
    term_set,
    word_count as _word_count,
)

# ---------------------------------------------------------------------------
# Regex and phrase banks for deliberative markers inside <think>
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

DIRECTION_CHANGE_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwait\b[,\s—-]",
        r"\bactually[,\s]",
        r"\bon second thought\b",
        r"\bhold on\b",
        r"\bthat(?:'s| is) not quite right\b",
        r"\blet me reconsider\b",
        r"\bscratch that\b",
        r"\bi initially thought\b",
        r"\bat first glance\b.*\bbut\b",
        r"\bif that were true\b",
        r"\bno[,\s]+that misses\b",
        r"\bstepping back\b",
    )
)

ASSUMPTION_SURFACING_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bassumes?\s+that\b",
        r"\bhidden assumption\b",
        r"\bunderlying assumption\b",
        r"\bunexamined assumption\b",
        r"\bpremise\s+that\b",
        r"\btaking for granted\b",
        r"\bwhat if we drop\b",
        r"\bimplicit(?:ly)?\s+assum",
        r"\bpresupposes?\b",
    )
)

SELF_CORRECTION_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:can't|cannot|don't)\s+verify\b",
        r"\bavoid\s+(?:hedging|fabricating|making up|inventing)\b",
        r"\bnot\s+just\s+['\"]?it depends['\"]?\b",
        r"\binstead of\s+(?:hedging|saying both sides|splitting the difference)\b",
        r"\bi shouldn't\s+(?:cite|invent|guess|hedge)\b",
        r"\bdon't have\s+(?:a real|exact)\s+(?:citation|number|study|stat)\b",
        r"\bi was about to\b",
        r"\bwithout\s+(?:making up|fabricating)\b",
        r"\btrap here is\b",
    )
)

REFRAMING_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bfalse dichotomy\b",
        r"\bwrong question\b",
        r"\brefram(?:e|ing)\b",
        r"\breal tension is\b",
        r"\bnot\s+\w+\s+vs\.?\s+\w+[,\s]+but\b",
        r"\bunderneath both\b",
        r"\bbinary framing\b",
        r"\bwhat's actually at stake\b",
        r"\bthe deeper issue\b",
    )
)

_MECHANISM_RE = re.compile(
    r"\b(?:because|therefore|mechanism|incentive|constraint|bottleneck|"
    r"feedback loop|trade-?off|leads?\s+to|results?\s+in|causes?|prevents?|"
    r"couples?|drives?|if\s+and\s+only\s+if|when\s+.*\bthen\b)\b",
    re.IGNORECASE,
)

MIN_GENUINE_THINK_WORDS = 30


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ThinkingFeatures:
    """Extracted features from a trace's <think> block."""

    has_think_tag: bool
    think_word_count: int
    think_sentence_count: int
    direction_change_count: int
    assumption_surfacing_count: int
    self_correction_count: int
    reframing_count: int
    mechanism_count: int
    mechanism_density: float
    think_to_output_divergence: float


@dataclass
class ThinkingQualityScore:
    """Stage 4 assessment of a trace's deliberative thinking block."""

    features: ThinkingFeatures
    direction_change_score: float
    assumption_score: float
    self_correction_score: float
    reframing_score: float
    mechanism_score: float
    divergence_score: float
    formulaic_penalty: float
    composite: float
    has_genuine_thinking: bool
    components: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction & Scoring
# ---------------------------------------------------------------------------

def extract_think_block(response: str) -> Tuple[Optional[str], str]:
    """Split a response into ``(think_content, visible_output)``.

    If multiple ``<think>`` blocks exist, concatenates their contents.
    If no ``<think>`` tag is present, returns ``(None, response.strip())``.
    """
    matches = _THINK_BLOCK_RE.findall(response)
    if not matches:
        if "</think>" in response:
            think_part, rest = response.split("</think>", 1)
            think_clean = re.sub(r"^<think>", "", think_part, flags=re.IGNORECASE).strip()
            return think_clean, rest.strip()
        return None, response.strip()

    think_text = "\n\n".join(m.strip() for m in matches if m.strip())
    visible = _THINK_BLOCK_RE.sub("", response).strip()
    return think_text, visible


def _count_pattern_hits(text: str, patterns: Tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for pat in patterns if pat.search(text))


def extract_thinking_features(response: str) -> ThinkingFeatures:
    """Extract deliberative reasoning features from a response's <think> block."""
    think_text, visible_text = extract_think_block(response)
    if think_text is None:
        return ThinkingFeatures(
            has_think_tag=False,
            think_word_count=0,
            think_sentence_count=0,
            direction_change_count=0,
            assumption_surfacing_count=0,
            self_correction_count=0,
            reframing_count=0,
            mechanism_count=0,
            mechanism_density=0.0,
            think_to_output_divergence=0.0,
        )

    wc = _word_count(think_text)
    sents = sentences(think_text)
    sc = max(1, len(sents)) if wc > 0 else 0

    dir_changes = _count_pattern_hits(think_text, DIRECTION_CHANGE_PATTERNS)
    assumptions = _count_pattern_hits(think_text, ASSUMPTION_SURFACING_PATTERNS)
    corrections = _count_pattern_hits(think_text, SELF_CORRECTION_PATTERNS)
    reframes = _count_pattern_hits(think_text, REFRAMING_PATTERNS)
    mechanisms = len(_MECHANISM_RE.findall(think_text))
    mech_density = mechanisms / max(1, sc)

    think_terms = term_set(think_text)
    visible_terms = term_set(visible_text)
    if think_terms and visible_terms:
        divergence = 1.0 - jaccard(think_terms, visible_terms)
    elif think_terms:
        divergence = 1.0
    else:
        divergence = 0.0

    return ThinkingFeatures(
        has_think_tag=True,
        think_word_count=wc,
        think_sentence_count=sc,
        direction_change_count=dir_changes,
        assumption_surfacing_count=assumptions,
        self_correction_count=corrections,
        reframing_count=reframes,
        mechanism_count=mechanisms,
        mechanism_density=mech_density,
        think_to_output_divergence=divergence,
    )


def score_thinking_quality(response_or_features: str | ThinkingFeatures) -> ThinkingQualityScore:
    """Compute Stage 4 thinking quality score in [0.0, 1.0]."""
    features = (
        extract_thinking_features(response_or_features)
        if isinstance(response_or_features, str)
        else response_or_features
    )

    if not features.has_think_tag or features.think_word_count == 0:
        return ThinkingQualityScore(
            features=features,
            direction_change_score=0.0,
            assumption_score=0.0,
            self_correction_score=0.0,
            reframing_score=0.0,
            mechanism_score=0.0,
            divergence_score=0.0,
            formulaic_penalty=1.0 if features.has_think_tag else 0.0,
            composite=0.0,
            has_genuine_thinking=False,
            components={"empty_or_missing": 1.0},
        )

    direction_score = min(1.0, features.direction_change_count / 2.0)
    assumption_score = min(1.0, features.assumption_surfacing_count / 1.5)
    correction_score = min(1.0, features.self_correction_count / 1.5)
    reframing_score = min(1.0, features.reframing_count / 1.5)
    mechanism_score = min(1.0, features.mechanism_density / 0.8)

    # Sweet spot for divergence: thinking should neither be a verbatim copy
    # of the output (<0.3) nor completely unrelated (>0.95).
    div = features.think_to_output_divergence
    if div < 0.25:
        divergence_score = div / 0.25 * 0.4
    elif div <= 0.90:
        divergence_score = 1.0
    else:
        divergence_score = max(0.4, 1.0 - (div - 0.90) * 4.0)

    # Formulaic / stub penalty:
    # 1) Very short thinking (< MIN_GENUINE_THINK_WORDS)
    # 2) Deliberative cue phrases present with zero causal/mechanism words
    formulaic_penalty = 0.0
    if features.think_word_count < MIN_GENUINE_THINK_WORDS:
        formulaic_penalty += 0.40 * (
            1.0 - features.think_word_count / float(MIN_GENUINE_THINK_WORDS)
        )
    deliberative_signals = (
        features.direction_change_count
        + features.assumption_surfacing_count
        + features.self_correction_count
        + features.reframing_count
    )
    if deliberative_signals > 0 and features.mechanism_count == 0:
        formulaic_penalty += 0.25

    raw_composite = (
        0.22 * direction_score
        + 0.20 * assumption_score
        + 0.18 * correction_score
        + 0.15 * reframing_score
        + 0.15 * mechanism_score
        + 0.10 * divergence_score
    )
    composite = max(0.0, min(1.0, raw_composite - formulaic_penalty))

    has_genuine_thinking = (
        features.think_word_count >= MIN_GENUINE_THINK_WORDS
        and deliberative_signals >= 2
        and features.mechanism_count >= 1
        and composite >= 0.45
    )

    components = {
        "direction_change": round(direction_score, 4),
        "assumption_surfacing": round(assumption_score, 4),
        "self_correction": round(correction_score, 4),
        "reframing": round(reframing_score, 4),
        "mechanism": round(mechanism_score, 4),
        "divergence": round(divergence_score, 4),
        "formulaic_penalty": round(formulaic_penalty, 4),
    }

    return ThinkingQualityScore(
        features=features,
        direction_change_score=direction_score,
        assumption_score=assumption_score,
        self_correction_score=correction_score,
        reframing_score=reframing_score,
        mechanism_score=mechanism_score,
        divergence_score=divergence_score,
        formulaic_penalty=formulaic_penalty,
        composite=round(composite, 4),
        has_genuine_thinking=has_genuine_thinking,
        components=components,
    )
