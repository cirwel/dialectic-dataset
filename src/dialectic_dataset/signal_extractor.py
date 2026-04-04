"""Semantic and structural feature extraction from dialectic traces.

Extracts features that feed the scorer and structural checks.
All functions are pure, deterministic, and dependency-free (stdlib only).
No LLM calls. No ML models. Rule-based + simple term vector similarity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "about", "up", "also", "and", "but", "or", "if", "while", "because",
    "until", "although", "though", "since", "that", "this", "these",
    "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom", "whose",
})

_HEDGING_WORDS: FrozenSet[str] = frozenset({
    "maybe", "perhaps", "might", "possibly", "could", "arguably",
    "debatable", "conceivably", "potentially", "seemingly", "apparently",
    "uncertain", "unclear", "questionable", "somewhat", "partially",
    "roughly", "approximately", "tends", "likely", "unlikely",
    "probable", "probably", "improbable", "plausible",
})

_LOGICAL_CONNECTORS: FrozenSet[str] = frozenset({
    "therefore", "because", "thus", "since", "hence", "consequently",
    "accordingly", "implies", "demonstrates", "establishes",
    "proves", "shows", "indicates", "suggests", "means",
    "results", "leads", "causes", "follows",
})

_DISCOURSE_MARKERS: FrozenSet[str] = frozenset({
    "however", "furthermore", "moreover", "nevertheless", "nonetheless",
    "additionally", "conversely", "alternatively", "meanwhile",
    "subsequently", "consequently", "similarly", "likewise",
    "specifically", "particularly", "notably", "importantly",
    "significantly", "essentially", "fundamentally", "ultimately",
})

_GENERIC_RESOLUTION_PHRASES: List[str] = [
    "both perspectives have merit",
    "the truth lies somewhere in the middle",
    "context matters",
    "it depends on the situation",
    "we should consider all perspectives",
    "a balanced approach is needed",
    "there are no easy answers",
    "reasonable people can disagree",
    "both sides have valid points",
    "the answer is nuanced",
    "it's not black and white",
    "there are trade-offs on both sides",
    "moderation is key",
    "each case should be considered individually",
    "a compromise is necessary",
]

_CONCESSION_PHRASES: List[str] = [
    "while this is true",
    "i agree that",
    "certainly",
    "it is true that",
    "admittedly",
    "granted",
    "no doubt",
    "undeniably",
    "of course",
    "to be fair",
    "it's worth noting that",
    "one must acknowledge",
]

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase word tokenization. Strips punctuation."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def content_words(text: str) -> List[str]:
    """Extract content words (non-stopwords) from text."""
    return [w for w in tokenize(text) if w not in _STOPWORDS]


def sentences(text: str) -> List[str]:
    """Split text into sentences. Simple heuristic."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def term_set(text: str) -> Set[str]:
    """Unique content word set from text."""
    return set(content_words(text))


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets. Returns 0.0 for empty sets."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def cosine_terms(counter_a: Counter, counter_b: Counter) -> float:
    """Cosine similarity between two term frequency Counters."""
    common = set(counter_a) & set(counter_b)
    if not common:
        return 0.0
    dot = sum(counter_a[k] * counter_b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in counter_a.values()))
    mag_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def word_count(text: str) -> int:
    """Count words in text."""
    return len(tokenize(text))


# ---------------------------------------------------------------------------
# Feature dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PhaseFeatures:
    """Extracted features for a single phase (thesis, antithesis, or synthesis)."""

    text: str
    word_count: int
    sentence_count: int
    content_terms: Set[str] = field(repr=False)
    term_counter: Counter = field(repr=False)

    # Density features
    claim_density: float       # assertive sentences per total sentences
    hedging_density: float     # hedging words per content word
    logical_density: float     # logical connectors per sentence
    discourse_marker_ratio: float  # sentences starting with discourse markers

    # Substance
    assertive_sentence_count: int
    filler_sentence_count: int


@dataclass
class TraceFeatures:
    """Extracted features for a complete dialectic trace."""

    thesis: PhaseFeatures
    antithesis: PhaseFeatures
    synthesis: PhaseFeatures

    # Cross-phase features
    stance_divergence: float         # how different is antithesis from thesis
    thesis_inheritance: float        # how much thesis appears in synthesis
    antithesis_inheritance: float    # how much antithesis appears in synthesis
    inheritance_balance: float       # ratio balance of inheritance from both sides
    novelty_ratio: float             # fraction of synthesis terms novel to both sides
    novelty_semantic_distance: float # how far novel terms are from prior phases

    # Aggregate
    generic_resolution_count: int    # count of generic resolution phrases in synthesis
    concession_ratio: float          # concession language in antithesis opening


# ---------------------------------------------------------------------------
# Phase-level extraction
# ---------------------------------------------------------------------------

def _is_assertive(sentence: str) -> bool:
    """Heuristic: sentence makes a claim (not a question, not pure filler)."""
    s = sentence.strip()
    if s.endswith("?"):
        return False
    tokens = tokenize(s)
    if len(tokens) < 4:
        return False
    # Check for filler patterns
    filler_starts = [
        "this is an important", "this is a complex", "let us consider",
        "it is worth noting", "as we can see", "in conclusion",
        "to summarize", "in summary", "as mentioned",
    ]
    lower = s.lower()
    for filler in filler_starts:
        if lower.startswith(filler):
            return False
    return True


def _is_filler(sentence: str) -> bool:
    """Heuristic: sentence is meta-commentary or transition without substance."""
    lower = sentence.lower().strip()
    filler_patterns = [
        "this is an important question",
        "this is a complex issue",
        "there are many perspectives",
        "let us consider",
        "let me think about",
        "it is important to note",
        "as we have seen",
        "in conclusion",
        "to summarize",
        "having considered",
    ]
    for pattern in filler_patterns:
        if pattern in lower:
            return True
    # Very short sentence with only discourse markers
    tokens = tokenize(lower)
    if len(tokens) <= 3 and any(t in _DISCOURSE_MARKERS for t in tokens):
        return True
    return False


def extract_phase_features(text: str) -> PhaseFeatures:
    """Extract features from a single phase's text."""
    sents = sentences(text)
    sent_count = max(1, len(sents))
    tokens = tokenize(text)
    cwords = content_words(text)
    wcount = len(tokens)
    cword_count = max(1, len(cwords))

    # Content terms
    terms = set(cwords)
    tcounter = Counter(cwords)

    # Claim density: assertive sentences / total sentences
    assertive_count = sum(1 for s in sents if _is_assertive(s))
    filler_count = sum(1 for s in sents if _is_filler(s))
    claim_density = assertive_count / sent_count

    # Hedging density: hedging words / total words
    # Count against all tokens, not just content words, because modal verbs
    # like "could" and "might" are both stopwords and hedging indicators
    hedge_count = sum(1 for w in tokens if w in _HEDGING_WORDS)
    hedging_density = hedge_count / max(1, wcount)

    # Logical density: logical connectors / sentences
    logic_count = sum(1 for w in tokens if w in _LOGICAL_CONNECTORS)
    logical_density = logic_count / sent_count

    # Discourse marker ratio: sentences starting with a discourse marker
    dm_starts = 0
    for s in sents:
        first_word = tokenize(s)[:1]
        if first_word and first_word[0] in _DISCOURSE_MARKERS:
            dm_starts += 1
    discourse_marker_ratio = dm_starts / sent_count

    return PhaseFeatures(
        text=text,
        word_count=wcount,
        sentence_count=sent_count,
        content_terms=terms,
        term_counter=tcounter,
        claim_density=claim_density,
        hedging_density=hedging_density,
        logical_density=logical_density,
        discourse_marker_ratio=discourse_marker_ratio,
        assertive_sentence_count=assertive_count,
        filler_sentence_count=filler_count,
    )


# ---------------------------------------------------------------------------
# Cross-phase extraction
# ---------------------------------------------------------------------------

def _concession_ratio(antithesis_text: str) -> float:
    """Fraction of antithesis opening that is concession language.

    Checks the first 30% of the antithesis text for concession patterns.
    """
    lower = antithesis_text.lower()
    # Look at first 30% of text
    cutoff = max(50, len(lower) // 3)
    opening = lower[:cutoff]

    concession_hits = 0
    for phrase in _CONCESSION_PHRASES:
        if phrase in opening:
            concession_hits += 1

    # Normalize by number of sentences in opening
    opening_sents = sentences(antithesis_text[:cutoff])
    sent_count = max(1, len(opening_sents))
    return min(1.0, concession_hits / sent_count)


def _count_generic_resolutions(synthesis_text: str) -> int:
    """Count generic resolution phrases in synthesis."""
    lower = synthesis_text.lower()
    count = 0
    for phrase in _GENERIC_RESOLUTION_PHRASES:
        if phrase in lower:
            count += 1
    return count


def extract_trace_features(
    thesis_text: str,
    antithesis_text: str,
    synthesis_text: str,
) -> TraceFeatures:
    """Extract all features from a complete dialectic trace.

    Parameters
    ----------
    thesis_text:
        The thesis phase text.
    antithesis_text:
        The antithesis phase text.
    synthesis_text:
        The synthesis phase text.

    Returns
    -------
    TraceFeatures with all phase and cross-phase features populated.
    """
    thesis_f = extract_phase_features(thesis_text)
    antithesis_f = extract_phase_features(antithesis_text)
    synthesis_f = extract_phase_features(synthesis_text)

    # --- Stance divergence ---
    # How different is antithesis from thesis?
    # 1 - Jaccard gives divergence (0 = identical, 1 = completely different)
    stance_divergence = 1.0 - jaccard(
        thesis_f.content_terms, antithesis_f.content_terms
    )

    # --- Inheritance ---
    # How much of thesis/antithesis appears in synthesis?
    thesis_terms = thesis_f.content_terms
    antithesis_terms = antithesis_f.content_terms
    synthesis_terms = synthesis_f.content_terms

    # Inheritance = fraction of source terms present in synthesis
    thesis_inheritance = (
        len(thesis_terms & synthesis_terms) / max(1, len(thesis_terms))
    )
    antithesis_inheritance = (
        len(antithesis_terms & synthesis_terms) / max(1, len(antithesis_terms))
    )

    # Inheritance balance: how evenly does synthesis draw from both sides?
    # 1.0 = perfectly balanced, 0.0 = draws from only one side
    total_inheritance = thesis_inheritance + antithesis_inheritance
    if total_inheritance > 0:
        min_inh = min(thesis_inheritance, antithesis_inheritance)
        max_inh = max(thesis_inheritance, antithesis_inheritance)
        inheritance_balance = min_inh / max_inh if max_inh > 0 else 0.0
    else:
        inheritance_balance = 0.0

    # --- Novelty ---
    # Terms in synthesis that appear in neither thesis nor antithesis
    prior_terms = thesis_terms | antithesis_terms
    novel_terms = synthesis_terms - prior_terms
    novelty_ratio = len(novel_terms) / max(1, len(synthesis_terms))

    # Novelty grounding: are novel terms contextually grounded in inherited content?
    # Measure: what fraction of synthesis sentences containing novel terms also
    # contain inherited terms? High = novel terms are woven into prior context.
    # Low = novel terms appear in isolated sentences (unanchored).
    if novel_terms and prior_terms:
        synth_sents = sentences(synthesis_text)
        novel_in_context = 0
        novel_total = 0
        for sent in synth_sents:
            sent_terms = term_set(sent)
            has_novel = bool(sent_terms & novel_terms)
            has_prior = bool(sent_terms & prior_terms)
            if has_novel:
                novel_total += 1
                if has_prior:
                    novel_in_context += 1
        # Distance: 1 - grounding. High distance = novel terms are unanchored.
        grounding = novel_in_context / max(1, novel_total)
        novelty_semantic_distance = 1.0 - grounding
    else:
        novelty_semantic_distance = 0.0

    # --- Generic resolution language ---
    generic_resolution_count = _count_generic_resolutions(synthesis_text)

    # --- Concession ratio ---
    concession_rat = _concession_ratio(antithesis_text)

    return TraceFeatures(
        thesis=thesis_f,
        antithesis=antithesis_f,
        synthesis=synthesis_f,
        stance_divergence=stance_divergence,
        thesis_inheritance=thesis_inheritance,
        antithesis_inheritance=antithesis_inheritance,
        inheritance_balance=inheritance_balance,
        novelty_ratio=novelty_ratio,
        novelty_semantic_distance=novelty_semantic_distance,
        generic_resolution_count=generic_resolution_count,
        concession_ratio=concession_rat,
    )
