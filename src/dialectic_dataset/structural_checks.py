"""Hard reject checks for dialectic trace quality.

Each check is a boolean gate. If any fires strongly, the trace is rejected
regardless of its weighted quality score. All checks are deterministic and
rule-based — no LLM calls.

See docs/structural_checks_spec.md for full definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dialectic_dataset.signal_extractor import (
    TraceFeatures,
    content_words,
    jaccard,
    sentences,
    term_set,
    tokenize,
    word_count,
    _GENERIC_RESOLUTION_PHRASES,
)


@dataclass
class CheckResult:
    """Result of a single structural check."""

    name: str
    fired: bool
    reason: str = ""


@dataclass
class StructuralCheckReport:
    """Results of all structural checks on a trace."""

    checks: List[CheckResult]
    any_fired: bool
    fired_names: List[str]

    @property
    def passed(self) -> bool:
        return not self.any_fired


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_fake_antithesis(features: TraceFeatures) -> CheckResult:
    """Check 1: Does the antithesis genuinely oppose the thesis?

    Fires if:
    - Thesis-antithesis Jaccard similarity > 0.6 (too similar)
    - Antithesis has no unique assertive content (negation-only)
    - Antithesis opens with > 50% concession language
    - Antithesis has zero unique evidence/claims vs thesis
    """
    reasons: List[str] = []

    # High term overlap
    thesis_terms = features.thesis.content_terms
    anti_terms = features.antithesis.content_terms
    sim = jaccard(thesis_terms, anti_terms)
    if sim > 0.6:
        reasons.append(f"term similarity {sim:.2f} > 0.6")

    # No unique content: antithesis terms minus thesis terms
    unique_to_anti = anti_terms - thesis_terms
    if len(unique_to_anti) < 3:
        reasons.append(
            f"only {len(unique_to_anti)} unique terms in antithesis"
        )

    # Concession-heavy opening
    if features.concession_ratio > 0.5:
        reasons.append(
            f"concession ratio {features.concession_ratio:.2f} > 0.5"
        )

    # Low stance divergence
    if features.stance_divergence < 0.3:
        reasons.append(
            f"stance divergence {features.stance_divergence:.2f} < 0.3"
        )

    # Fire if 2+ signals, OR if any single signal is extreme
    extreme = (
        sim > 0.8  # Near-identical terms
        or features.concession_ratio > 0.7  # Overwhelming concession
        or features.stance_divergence < 0.15  # Almost no divergence
    )
    fired = len(reasons) >= 2 or (len(reasons) >= 1 and extreme)
    return CheckResult(
        name="fake_antithesis",
        fired=fired,
        reason="; ".join(reasons) if fired else "",
    )


def check_synthesis_is_summary_only(features: TraceFeatures) -> CheckResult:
    """Check 2: Is the synthesis just a summary of both sides?

    Fires if:
    - > 80% of synthesis terms appear in thesis + antithesis
    - < 10% of synthesis terms are novel
    - Synthesis has no resolution claim (low assertive density in final portion)
    """
    reasons: List[str] = []

    prior_terms = features.thesis.content_terms | features.antithesis.content_terms
    synth_terms = features.synthesis.content_terms

    if synth_terms:
        coverage = len(synth_terms & prior_terms) / len(synth_terms)
    else:
        coverage = 1.0

    if coverage > 0.80 and features.novelty_ratio < 0.10:
        reasons.append(
            f"coverage {coverage:.2f} > 0.80 with novelty {features.novelty_ratio:.2f} < 0.10"
        )

    # Check for enumeration pattern in synthesis
    synth_text_lower = features.synthesis.text.lower()
    enumeration_markers = [
        "on one hand", "on the other hand",
        "the first perspective", "the second perspective",
        "thesis argues", "antithesis argues",
        "proponents argue", "opponents argue",
        "some believe", "others believe",
        "supporters contend", "critics contend",
        "advocates suggest", "detractors suggest",
    ]
    enum_count = sum(1 for m in enumeration_markers if m in synth_text_lower)
    if enum_count >= 2:
        reasons.append(f"{enum_count} enumeration markers in synthesis")

    # Very low novelty overall
    if features.novelty_ratio < 0.05:
        reasons.append(f"novelty ratio {features.novelty_ratio:.2f} < 0.05")

    # Strong enumeration is sufficient on its own (>= 3 markers is overwhelming)
    extreme_enumeration = enum_count >= 3
    fired = len(reasons) >= 2 or (len(reasons) >= 1 and extreme_enumeration)
    return CheckResult(
        name="synthesis_is_summary_only",
        fired=fired,
        reason="; ".join(reasons) if fired else "",
    )


_DISMISSAL_MARKERS: List[str] = [
    "clearly shows", "clearly demonstrate", "clearly indicate",
    "evidence clearly", "data clearly",
    "overstated", "understated", "exaggerated", "overblown",
    "not genuine", "not by genuine", "disguised as",
    "myth", "fallacy", "misconception",
    "fails to account", "fails to consider", "ignores the fact",
    "simply wrong", "fundamentally flawed", "deeply misguided",
    "undeniable that", "incontrovertible", "beyond question",
    "driven by cost savings", "driven by ideology",
]


def check_one_sided_synthesis(features: TraceFeatures) -> CheckResult:
    """Check 3: Does the synthesis clearly favor one side?

    Fires if:
    - Inheritance from one side is > 3x the other
    - Synthesis conclusion has > 0.7 similarity with one side's conclusion
    - Synthesis contains dismissal language targeting one side
    """
    reasons: List[str] = []

    # Asymmetric inheritance
    t_inh = features.thesis_inheritance
    a_inh = features.antithesis_inheritance
    if t_inh > 0 or a_inh > 0:
        max_inh = max(t_inh, a_inh)
        min_inh = min(t_inh, a_inh)
        if min_inh > 0 and max_inh / min_inh > 3.0:
            dominant = "thesis" if t_inh > a_inh else "antithesis"
            reasons.append(
                f"inheritance ratio {max_inh/min_inh:.1f}x toward {dominant}"
            )
        elif min_inh == 0 and max_inh > 0.2:
            dominant = "thesis" if t_inh > a_inh else "antithesis"
            reasons.append(f"zero inheritance from {'antithesis' if dominant == 'thesis' else 'thesis'}")

    # Low inheritance balance
    if features.inheritance_balance < 0.25:
        reasons.append(
            f"inheritance balance {features.inheritance_balance:.2f} < 0.25"
        )

    # Check conclusion similarity (last sentence of each)
    synth_sents = sentences(features.synthesis.text)
    thesis_sents = sentences(features.thesis.text)
    anti_sents = sentences(features.antithesis.text)

    if synth_sents and thesis_sents and anti_sents:
        synth_conclusion = term_set(synth_sents[-1])
        thesis_conclusion = term_set(thesis_sents[-1])
        anti_conclusion = term_set(anti_sents[-1])

        sim_thesis = jaccard(synth_conclusion, thesis_conclusion)
        sim_anti = jaccard(synth_conclusion, anti_conclusion)

        if sim_thesis > 0.7 or sim_anti > 0.7:
            side = "thesis" if sim_thesis > sim_anti else "antithesis"
            score = max(sim_thesis, sim_anti)
            reasons.append(
                f"conclusion similarity {score:.2f} with {side}"
            )

    # Dismissal language: synthesis uses strong negation/dismissal targeting one side
    synth_lower = features.synthesis.text.lower()
    dismissal_count = sum(1 for m in _DISMISSAL_MARKERS if m in synth_lower)
    if dismissal_count >= 2:
        reasons.append(f"{dismissal_count} dismissal markers in synthesis")

    # Extreme: 3+ dismissals is overwhelming one-sidedness
    extreme = dismissal_count >= 3
    fired = len(reasons) >= 2 or (len(reasons) >= 1 and extreme)
    return CheckResult(
        name="one_sided_synthesis",
        fired=fired,
        reason="; ".join(reasons) if fired else "",
    )


def check_unanchored_novelty(features: TraceFeatures) -> CheckResult:
    """Check 4: Does synthesis introduce claims with no basis in prior phases?

    Fires if:
    - > 40% of synthesis terms are novel AND semantically distant
    - Novel content forms coherent claims (not just filler)
    """
    reasons: List[str] = []

    if features.novelty_ratio > 0.40:
        reasons.append(f"novelty ratio {features.novelty_ratio:.2f} > 0.40")

    if features.novelty_semantic_distance > 0.6:
        reasons.append(
            f"novelty semantic distance {features.novelty_semantic_distance:.2f} > 0.6"
        )

    # Combined: high novelty AND moderate distance = smuggled content
    if features.novelty_ratio > 0.30 and features.novelty_semantic_distance > 0.4:
        reasons.append("high novelty with ungrounded content")

    # Extreme: very high novelty AND clearly ungrounded (distance > 0.4)
    extreme = (
        features.novelty_ratio > 0.60
        and features.novelty_semantic_distance > 0.4
    )
    fired = len(reasons) >= 2 or (len(reasons) >= 1 and extreme)
    return CheckResult(
        name="unanchored_novelty",
        fired=fired,
        reason="; ".join(reasons) if fired else "",
    )


def check_low_substance_phase(features: TraceFeatures) -> CheckResult:
    """Check 5: Does any phase contain too little reasoning?

    Fires if:
    - Any phase has < 2 sentences or < 30 words
    - Any phase has zero assertive sentences
    - Any phase is > 60% filler
    - Any phase is < 25% the word count of the longest phase
    """
    reasons: List[str] = []
    phases = {
        "thesis": features.thesis,
        "antithesis": features.antithesis,
        "synthesis": features.synthesis,
    }

    max_wc = max(p.word_count for p in phases.values())

    for name, phase in phases.items():
        if phase.sentence_count < 2:
            reasons.append(f"{name}: {phase.sentence_count} sentences < 2")
        if phase.word_count < 30:
            reasons.append(f"{name}: {phase.word_count} words < 30")
        if phase.assertive_sentence_count == 0:
            reasons.append(f"{name}: zero assertive sentences")
        if phase.sentence_count > 0:
            filler_ratio = phase.filler_sentence_count / phase.sentence_count
            if filler_ratio > 0.6:
                reasons.append(f"{name}: filler ratio {filler_ratio:.2f} > 0.6")
        if max_wc > 0 and phase.word_count < 0.25 * max_wc:
            reasons.append(
                f"{name}: {phase.word_count} words < 25% of longest ({max_wc})"
            )

    fired = len(reasons) >= 1  # Any single substance failure rejects
    return CheckResult(
        name="low_substance_phase",
        fired=fired,
        reason="; ".join(reasons) if fired else "",
    )


def check_template_collapse(features: TraceFeatures) -> CheckResult:
    """Check 6: Does the trace follow a formulaic template?

    Fires if:
    - > 30% of sentences start with discourse markers AND content is generic
    - Thesis and antithesis have near-identical structural fingerprints
    - Boilerplate opening detected
    """
    reasons: List[str] = []

    # Discourse marker spam across all phases
    total_sents = (
        features.thesis.sentence_count
        + features.antithesis.sentence_count
        + features.synthesis.sentence_count
    )
    total_dm = 0
    for phase in [features.thesis, features.antithesis, features.synthesis]:
        total_dm += int(phase.discourse_marker_ratio * phase.sentence_count)

    if total_sents > 0:
        overall_dm_ratio = total_dm / total_sents
        if overall_dm_ratio > 0.30:
            reasons.append(
                f"discourse marker ratio {overall_dm_ratio:.2f} > 0.30"
            )

    # Structural similarity between thesis and antithesis
    t_sc = features.thesis.sentence_count
    a_sc = features.antithesis.sentence_count
    if t_sc > 0 and a_sc > 0:
        count_ratio = min(t_sc, a_sc) / max(t_sc, a_sc)
        wc_ratio = min(features.thesis.word_count, features.antithesis.word_count) / max(
            1, max(features.thesis.word_count, features.antithesis.word_count)
        )
        if count_ratio > 0.8 and wc_ratio > 0.8:
            # Also check discourse marker symmetry
            dm_diff = abs(
                features.thesis.discourse_marker_ratio
                - features.antithesis.discourse_marker_ratio
            )
            if dm_diff < 0.1:
                reasons.append("thesis/antithesis structural fingerprint match")

    # Boilerplate opening
    thesis_lower = features.thesis.text.lower().strip()
    boilerplate_openers = [
        "there are valid arguments on both sides",
        "this is a complex issue with",
        "this question has been debated",
        "there are compelling arguments",
        "this is a multifaceted issue",
        "people have long debated",
    ]
    for opener in boilerplate_openers:
        if thesis_lower.startswith(opener):
            reasons.append(f"boilerplate opening: '{opener[:40]}...'")
            break

    fired = len(reasons) >= 2
    return CheckResult(
        name="template_collapse",
        fired=fired,
        reason="; ".join(reasons) if fired else "",
    )


def check_generic_resolution_language(features: TraceFeatures) -> CheckResult:
    """Check 7: Does the synthesis use vague, non-committal resolution language?

    Fires if:
    - > 2 generic resolution phrases in synthesis
    - Synthesis has high hedging density AND low claim density
    - Synthesis lacks conditional specifics after "depends" language
    """
    reasons: List[str] = []

    if features.generic_resolution_count > 2:
        reasons.append(
            f"{features.generic_resolution_count} generic resolution phrases"
        )

    # High hedging + low claims in synthesis
    if (features.synthesis.hedging_density > 0.08
            and features.synthesis.claim_density < 0.5):
        reasons.append(
            f"synthesis hedging {features.synthesis.hedging_density:.2f} "
            f"with claim density {features.synthesis.claim_density:.2f}"
        )

    # "Depends" without specifics
    synth_lower = features.synthesis.text.lower()
    depends_phrases = ["it depends", "depends on", "context-dependent", "case by case"]
    has_depends = any(p in synth_lower for p in depends_phrases)
    # Check for conditional specifics: if/when/where followed by content
    conditional_specifics = len(
        [s for s in sentences(features.synthesis.text)
         if any(s.lower().strip().startswith(w) for w in ["if ", "when ", "where ", "in cases where"])]
    )
    if has_depends and conditional_specifics == 0:
        reasons.append("'depends' language without conditional specifics")

    # Generic even without exact phrase matches: very low topic-specific content
    if features.generic_resolution_count >= 1 and features.novelty_ratio < 0.08:
        reasons.append("generic phrases with very low novelty")

    fired = len(reasons) >= 2
    return CheckResult(
        name="generic_resolution_language",
        fired=fired,
        reason="; ".join(reasons) if fired else "",
    )


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

_ALL_CHECKS = [
    check_fake_antithesis,
    check_synthesis_is_summary_only,
    check_one_sided_synthesis,
    check_unanchored_novelty,
    check_low_substance_phase,
    check_template_collapse,
    check_generic_resolution_language,
]


def run_all_checks(features: TraceFeatures) -> StructuralCheckReport:
    """Run all 7 structural checks on a trace.

    Parameters
    ----------
    features:
        TraceFeatures extracted by signal_extractor.extract_trace_features().

    Returns
    -------
    StructuralCheckReport with results of all checks.
    """
    results = [check_fn(features) for check_fn in _ALL_CHECKS]
    fired = [r for r in results if r.fired]
    return StructuralCheckReport(
        checks=results,
        any_fired=len(fired) > 0,
        fired_names=[r.name for r in fired],
    )
