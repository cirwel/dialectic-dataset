"""Weighted quality scoring for dialectic traces.

Computes 4 quality buckets and a training usefulness gate.
Integrates structural checks for hard rejects.

All scoring is deterministic and rule-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dialectic_dataset.signal_extractor import (
    TraceFeatures,
    extract_trace_features,
    jaccard,
    sentences,
    term_set,
)
from dialectic_dataset.structural_checks import (
    StructuralCheckReport,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

KEEP_THRESHOLD = 0.72
REVIEW_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class TraceScore:
    """Complete quality assessment of a dialectic trace."""

    # Bucket scores [0, 1]
    tension_quality: float
    resolution_quality: float
    cross_phase_coherence: float
    synthesis_sufficiency: float

    # Composite
    composite_quality: float
    verdict: str  # "keep", "review", "reject"

    # Gates
    structural_report: StructuralCheckReport
    would_train_on_this: bool

    # Breakdown for transparency
    components: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bucket 1: Tension Quality (weight: 0.30)
# ---------------------------------------------------------------------------

def score_tension_quality(features: TraceFeatures) -> Tuple[float, Dict[str, float]]:
    """Score how genuinely the competing frames are in tension.

    Components:
    - Stance divergence: thesis and antithesis should be substantively different
    - Antithesis substance: antithesis should have its own claims, not just negate
    - Balanced depth: both sides should receive comparable treatment
    """
    components: Dict[str, float] = {}

    # Stance divergence [0, 1] — higher is better for tension
    # Sweet spot: 0.4-0.8. Too low = fake, too high = unrelated topics
    div = features.stance_divergence
    if div < 0.2:
        components["stance_divergence"] = 0.1  # Too similar
    elif div < 0.4:
        components["stance_divergence"] = 0.3 + (div - 0.2) * 2.0  # Ramping
    elif div <= 0.8:
        components["stance_divergence"] = 0.7 + (div - 0.4) * 0.75  # Good range
    else:
        components["stance_divergence"] = max(0.3, 1.0 - (div - 0.8) * 2.0)  # Too unrelated

    # Antithesis substance: claim density in antithesis
    anti_claims = features.antithesis.claim_density
    components["antithesis_substance"] = min(1.0, anti_claims / 0.7)

    # Balanced depth: word count ratio between thesis and antithesis
    t_wc = features.thesis.word_count
    a_wc = features.antithesis.word_count
    if max(t_wc, a_wc) > 0:
        balance = min(t_wc, a_wc) / max(t_wc, a_wc)
    else:
        balance = 0.0
    components["depth_balance"] = balance

    # Low concession ratio in antithesis (should challenge, not concede)
    concession_penalty = max(0.0, features.concession_ratio - 0.3)
    components["concession_penalty"] = max(0.0, 1.0 - concession_penalty * 3.0)

    # Weighted combination
    score = (
        0.35 * components["stance_divergence"]
        + 0.30 * components["antithesis_substance"]
        + 0.20 * components["depth_balance"]
        + 0.15 * components["concession_penalty"]
    )
    return min(1.0, max(0.0, score)), components


# ---------------------------------------------------------------------------
# Bucket 2: Resolution Quality (weight: 0.35)
# ---------------------------------------------------------------------------

def score_resolution_quality(features: TraceFeatures) -> Tuple[float, Dict[str, float]]:
    """Score whether synthesis integrates rather than summarizes or dodges.

    Components:
    - Novelty: synthesis should contain reasoning beyond thesis + antithesis
    - Inheritance balance: synthesis should draw from BOTH sides
    - Non-generic: should avoid vague resolution language
    - Claim density: synthesis should make specific claims
    """
    components: Dict[str, float] = {}

    # Novelty: some novel content is good, too much is suspicious
    novelty = features.novelty_ratio
    if novelty < 0.05:
        components["novelty"] = 0.1  # Just a summary
    elif novelty <= 0.35:
        components["novelty"] = 0.3 + (novelty - 0.05) * 2.33  # Good range
    else:
        components["novelty"] = max(0.2, 1.0 - (novelty - 0.35) * 3.0)  # Too much = unanchored

    # Inheritance balance: should draw from both sides
    components["inheritance_balance"] = features.inheritance_balance

    # Non-generic resolution
    generic_penalty = min(1.0, features.generic_resolution_count * 0.3)
    components["non_generic"] = 1.0 - generic_penalty

    # Synthesis claim density
    components["synthesis_claims"] = min(1.0, features.synthesis.claim_density / 0.7)

    score = (
        0.30 * components["novelty"]
        + 0.30 * components["inheritance_balance"]
        + 0.20 * components["non_generic"]
        + 0.20 * components["synthesis_claims"]
    )
    return min(1.0, max(0.0, score)), components


# ---------------------------------------------------------------------------
# Bucket 3: Cross-Phase Coherence (weight: 0.20)
# ---------------------------------------------------------------------------

def score_cross_phase_coherence(features: TraceFeatures) -> Tuple[float, Dict[str, float]]:
    """Score logical connection between phases.

    Components:
    - Thesis→antithesis connection: antithesis should reference thesis claims
    - Antithesis→synthesis connection: synthesis should address the tension
    - Progressive build: each phase should add to the previous
    """
    components: Dict[str, float] = {}

    # Thesis→antithesis connection
    # Antithesis should share some terms with thesis (it's responding to it)
    # but not too many (it's disagreeing)
    ta_sim = 1.0 - features.stance_divergence  # Jaccard similarity
    if 0.2 <= ta_sim <= 0.6:
        components["ta_connection"] = 0.7 + (0.3 * (1.0 - abs(ta_sim - 0.4) / 0.2))
    elif ta_sim < 0.2:
        components["ta_connection"] = ta_sim / 0.2 * 0.5  # Too disconnected
    else:
        components["ta_connection"] = max(0.3, 1.0 - (ta_sim - 0.6) * 2.0)  # Too similar

    # Synthesis should inherit from both (already have this)
    combined_inheritance = (
        features.thesis_inheritance + features.antithesis_inheritance
    ) / 2.0
    components["synthesis_grounding"] = min(1.0, combined_inheritance / 0.4)

    # Progressive build: synthesis should be substantively different from
    # simply concatenating thesis and antithesis
    # Use claim density progression
    t_cd = features.thesis.claim_density
    a_cd = features.antithesis.claim_density
    s_cd = features.synthesis.claim_density
    # Synthesis should maintain or increase claim density
    if s_cd >= max(t_cd, a_cd) * 0.8:
        components["progressive_density"] = min(1.0, s_cd / max(0.3, max(t_cd, a_cd)))
    else:
        components["progressive_density"] = max(0.0, s_cd / max(0.3, max(t_cd, a_cd)))

    # Logical connector usage should increase toward synthesis
    s_logic = features.synthesis.logical_density
    avg_prior_logic = (features.thesis.logical_density + features.antithesis.logical_density) / 2.0
    if s_logic >= avg_prior_logic:
        components["logical_progression"] = min(1.0, 0.5 + s_logic * 0.5)
    else:
        components["logical_progression"] = max(0.2, s_logic / max(0.1, avg_prior_logic))

    score = (
        0.30 * components["ta_connection"]
        + 0.30 * components["synthesis_grounding"]
        + 0.20 * components["progressive_density"]
        + 0.20 * components["logical_progression"]
    )
    return min(1.0, max(0.0, score)), components


# ---------------------------------------------------------------------------
# Bucket 4: Synthesis Sufficiency (weight: 0.15)
# ---------------------------------------------------------------------------

def score_synthesis_sufficiency(features: TraceFeatures) -> Tuple[float, Dict[str, float]]:
    """Score whether the synthesis is substantive enough.

    Components:
    - Length adequacy: synthesis should not be trivially short
    - Assertive content: synthesis should make specific claims
    - Low filler: synthesis should not be padded
    - Low hedging: resolution should commit to something
    """
    components: Dict[str, float] = {}

    # Length adequacy: synthesis should be >= 40% of longest phase
    max_wc = max(
        features.thesis.word_count,
        features.antithesis.word_count,
    )
    if max_wc > 0:
        length_ratio = features.synthesis.word_count / max_wc
    else:
        length_ratio = 0.0
    components["length_adequacy"] = min(1.0, length_ratio / 0.6)

    # Assertive content
    components["assertive_content"] = min(
        1.0, features.synthesis.claim_density / 0.6
    )

    # Low filler
    if features.synthesis.sentence_count > 0:
        filler_ratio = (
            features.synthesis.filler_sentence_count
            / features.synthesis.sentence_count
        )
    else:
        filler_ratio = 1.0
    components["low_filler"] = 1.0 - filler_ratio

    # Low hedging in synthesis (resolution should commit)
    hedge = features.synthesis.hedging_density
    components["commitment"] = max(0.0, 1.0 - hedge * 8.0)

    score = (
        0.25 * components["length_adequacy"]
        + 0.30 * components["assertive_content"]
        + 0.25 * components["low_filler"]
        + 0.20 * components["commitment"]
    )
    return min(1.0, max(0.0, score)), components


# ---------------------------------------------------------------------------
# Training Usefulness Gate
# ---------------------------------------------------------------------------

def assess_training_usefulness(
    features: TraceFeatures,
    tension: float,
    resolution: float,
    coherence: float,
    sufficiency: float,
    structural: StructuralCheckReport,
) -> bool:
    """Determine if a trace would be good training supervision.

    Separate from quality score. A trace can be coherent but still teach
    bad patterns.

    Returns True if the trace should be trained on.
    """
    # Hard fail: any structural check fired
    if structural.any_fired:
        return False

    # Hard fail: any bucket is very low (even if composite is ok)
    if tension < 0.35 or resolution < 0.30:
        return False

    # Ritualization risk: high discourse markers + low claim density
    avg_dm = (
        features.thesis.discourse_marker_ratio
        + features.antithesis.discourse_marker_ratio
        + features.synthesis.discourse_marker_ratio
    ) / 3.0
    avg_claims = (
        features.thesis.claim_density
        + features.antithesis.claim_density
        + features.synthesis.claim_density
    ) / 3.0
    if avg_dm > 0.25 and avg_claims < 0.5:
        return False  # Ceremony without substance

    # Verbose filler: very long but low substance
    total_words = (
        features.thesis.word_count
        + features.antithesis.word_count
        + features.synthesis.word_count
    )
    total_assertive = (
        features.thesis.assertive_sentence_count
        + features.antithesis.assertive_sentence_count
        + features.synthesis.assertive_sentence_count
    )
    total_sents = (
        features.thesis.sentence_count
        + features.antithesis.sentence_count
        + features.synthesis.sentence_count
    )
    if total_words > 500 and total_sents > 0:
        assertive_ratio = total_assertive / total_sents
        if assertive_ratio < 0.4:
            return False  # Verbose but hollow

    return True


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_trace(
    thesis_text: str,
    antithesis_text: str,
    synthesis_text: str,
    features: TraceFeatures | None = None,
) -> TraceScore:
    """Score a complete dialectic trace.

    Parameters
    ----------
    thesis_text:
        The thesis phase text.
    antithesis_text:
        The antithesis phase text.
    synthesis_text:
        The synthesis phase text.
    features:
        Pre-extracted features. If None, extracted automatically.

    Returns
    -------
    TraceScore with all bucket scores, composite, verdict, and gates.
    """
    if features is None:
        features = extract_trace_features(thesis_text, antithesis_text, synthesis_text)

    # Structural checks
    structural = run_all_checks(features)

    # Bucket scores
    tension, tension_comp = score_tension_quality(features)
    resolution, resolution_comp = score_resolution_quality(features)
    coherence, coherence_comp = score_cross_phase_coherence(features)
    sufficiency, sufficiency_comp = score_synthesis_sufficiency(features)

    # Composite quality (weighted)
    composite = (
        0.30 * tension
        + 0.35 * resolution
        + 0.20 * coherence
        + 0.15 * sufficiency
    )

    # Verdict
    if structural.any_fired:
        verdict = "reject"
    elif composite >= KEEP_THRESHOLD:
        verdict = "keep"
    elif composite >= REVIEW_THRESHOLD:
        verdict = "review"
    else:
        verdict = "reject"

    # Training usefulness gate
    would_train = assess_training_usefulness(
        features, tension, resolution, coherence, sufficiency, structural,
    )
    if not would_train and verdict == "keep":
        verdict = "review"  # Downgrade: good score but bad supervision

    # Full component breakdown
    all_components = {
        "tension_quality": tension,
        "resolution_quality": resolution,
        "cross_phase_coherence": coherence,
        "synthesis_sufficiency": sufficiency,
        **{f"tension.{k}": v for k, v in tension_comp.items()},
        **{f"resolution.{k}": v for k, v in resolution_comp.items()},
        **{f"coherence.{k}": v for k, v in coherence_comp.items()},
        **{f"sufficiency.{k}": v for k, v in sufficiency_comp.items()},
    }

    return TraceScore(
        tension_quality=round(tension, 4),
        resolution_quality=round(resolution, 4),
        cross_phase_coherence=round(coherence, 4),
        synthesis_sufficiency=round(sufficiency, 4),
        composite_quality=round(composite, 4),
        verdict=verdict,
        structural_report=structural,
        would_train_on_this=would_train,
        components={k: round(v, 4) for k, v in all_components.items()},
    )
