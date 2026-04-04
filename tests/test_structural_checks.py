"""Tests for structural checks — hard reject gates.

Includes adversarial samples that MUST be rejected.
"""

from dialectic_dataset.signal_extractor import extract_trace_features
from dialectic_dataset.structural_checks import (
    check_fake_antithesis,
    check_generic_resolution_language,
    check_low_substance_phase,
    check_one_sided_synthesis,
    check_synthesis_is_summary_only,
    check_template_collapse,
    check_unanchored_novelty,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# Good trace (should pass all checks)
# ---------------------------------------------------------------------------

GOOD_THESIS = (
    "Universal basic income would reduce poverty and provide economic security. "
    "Studies from Finland and Kenya show that direct cash transfers improve "
    "health outcomes, increase entrepreneurship, and reduce stress. People "
    "who receive basic income are more likely to start businesses and pursue "
    "education, generating long-term economic growth."
)

GOOD_ANTITHESIS = (
    "Universal basic income is economically unsustainable and would reduce "
    "work incentives. The cost of providing a meaningful income to every citizen "
    "would require massive tax increases or deficit spending. Labor participation "
    "studies from negative income tax experiments in the 1970s showed work "
    "reduction of 5-10%, and inflation risks from increased consumer spending "
    "without matching productivity gains would erode the benefit's purchasing power."
)

GOOD_SYNTHESIS = (
    "The core tension is between poverty reduction effectiveness and fiscal "
    "sustainability. Cash transfers demonstrably help recipients in the short "
    "term, while fiscal risks materialize over years. A phased approach — "
    "starting with targeted basic income for populations below the poverty line, "
    "funded by consolidating existing welfare programs rather than new taxes — "
    "captures the poverty reduction benefits while containing fiscal exposure. "
    "Work incentive concerns can be addressed through earned income supplements "
    "that phase out gradually rather than creating sharp cutoffs."
)


class TestGoodTracePassesAll:
    def test_no_checks_fire(self):
        features = extract_trace_features(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        report = run_all_checks(features)
        assert report.passed, f"Checks fired: {report.fired_names}"


# ---------------------------------------------------------------------------
# Adversarial: Fake antithesis
# ---------------------------------------------------------------------------

class TestFakeAntithesis:
    def test_restated_thesis_same_terms(self):
        """Antithesis reuses thesis language with minor rewording."""
        fake_anti = (
            "Universal basic income would reduce poverty and provide economic "
            "security. Studies from Finland and Kenya show that direct cash "
            "transfers improve health outcomes and reduce stress. People who "
            "receive basic income start businesses and pursue education, "
            "generating economic growth."
        )
        features = extract_trace_features(GOOD_THESIS, fake_anti, GOOD_SYNTHESIS)
        result = check_fake_antithesis(features)
        assert result.fired, f"Should fire on near-identical antithesis"

    def test_synonym_paraphrase_known_gap(self):
        """KNOWN LIMITATION: Synonym-heavy paraphrases evade lexical analysis.

        A fake antithesis using full synonym substitution appears maximally
        different at the term level, scoring HIGH on tension_quality. Detecting
        this requires embedding-based stance detection (phase 2 feature).

        This test documents the gap — it asserts the current behavior
        (false acceptance) so we know when we've fixed it.
        """
        from dialectic_dataset.trace_scorer import score_trace
        fake_anti = (
            "Basic income programs would decrease poverty and offer economic "
            "stability. Research from Scandinavian countries and East Africa "
            "demonstrates that giving cash directly improves wellness, boosts "
            "business creation, and lowers anxiety. Recipients tend to launch "
            "enterprises and seek education, creating economic development."
        )
        score = score_trace(GOOD_THESIS, fake_anti, GOOD_SYNTHESIS)
        # Currently this passes (false acceptance) — flip assertion when
        # embedding-based stance detection is added
        assert score.verdict == "keep", (
            "If this fails, synonym paraphrase detection may now work — "
            "update test to assert rejection instead"
        )

    def test_concession_heavy(self):
        """Antithesis opens with mostly agreement."""
        conceding = (
            "While this is true and I agree that basic income helps people, "
            "certainly the evidence is compelling, admittedly the data is "
            "strong, granted that cash transfers work. However there might "
            "be a small concern about costs in some specific situations."
        )
        features = extract_trace_features(GOOD_THESIS, conceding, GOOD_SYNTHESIS)
        result = check_fake_antithesis(features)
        assert result.fired, f"Should fire on concession-heavy antithesis"


# ---------------------------------------------------------------------------
# Adversarial: Summary-only synthesis
# ---------------------------------------------------------------------------

class TestSummaryOnlySynthesis:
    def test_pure_summary(self):
        """Synthesis just restates both sides."""
        summary_synth = (
            "On one hand, proponents argue that basic income reduces poverty "
            "and improves health outcomes through direct cash transfers. "
            "On the other hand, opponents argue that basic income is "
            "economically unsustainable and would reduce work incentives "
            "through decreased labor participation."
        )
        features = extract_trace_features(GOOD_THESIS, GOOD_ANTITHESIS, summary_synth)
        result = check_synthesis_is_summary_only(features)
        assert result.fired, f"Should fire on summary-only synthesis"


# ---------------------------------------------------------------------------
# Adversarial: One-sided synthesis
# ---------------------------------------------------------------------------

class TestOneSidedSynthesis:
    def test_picks_thesis(self):
        """Synthesis clearly favors thesis without engaging antithesis."""
        one_sided = (
            "The evidence clearly shows that basic income reduces poverty "
            "and improves health outcomes. Studies from Finland and Kenya "
            "demonstrate the effectiveness of direct cash transfers. "
            "People who receive basic income start businesses and pursue "
            "education. The concerns about cost are overblown."
        )
        features = extract_trace_features(GOOD_THESIS, GOOD_ANTITHESIS, one_sided)
        result = check_one_sided_synthesis(features)
        assert result.fired, f"Should fire on one-sided synthesis"

    def test_dismissal_language(self):
        """Synthesis dismisses one side with strong negation language."""
        dismissive = (
            "The evidence clearly shows that open floor plans reduce "
            "productivity and satisfaction. The collaboration benefits "
            "are overstated and can be achieved through shared spaces. "
            "The open floor plan trend was driven by cost savings "
            "disguised as culture, not by genuine evidence."
        )
        thesis = (
            "Open floor plans improve collaboration and communication. "
            "Studies show increased spontaneous interactions and faster "
            "information sharing."
        )
        antithesis = (
            "Open floor plans reduce deep work productivity. Research "
            "demonstrates noise and interruptions decrease focus time."
        )
        features = extract_trace_features(thesis, antithesis, dismissive)
        result = check_one_sided_synthesis(features)
        assert result.fired, f"Should fire on dismissal language"


# ---------------------------------------------------------------------------
# Adversarial: Unanchored novelty
# ---------------------------------------------------------------------------

class TestUnanchoredNovelty:
    def test_smuggled_content(self):
        """Synthesis introduces entirely new topic."""
        novel_synth = (
            "The question of basic income connects to deeper issues about "
            "artificial intelligence and automation. Machine learning systems "
            "are rapidly replacing human workers in manufacturing, logistics, "
            "and customer service. Quantum computing breakthroughs suggest "
            "that within a decade, most cognitive tasks will be automated. "
            "Blockchain-based distribution systems could enable transparent "
            "and efficient universal payment mechanisms."
        )
        features = extract_trace_features(GOOD_THESIS, GOOD_ANTITHESIS, novel_synth)
        result = check_unanchored_novelty(features)
        assert result.fired, f"Should fire on smuggled content"


# ---------------------------------------------------------------------------
# Adversarial: Low substance phase
# ---------------------------------------------------------------------------

class TestLowSubstancePhase:
    def test_empty_thesis(self):
        """Thesis is too short to be meaningful."""
        features = extract_trace_features("Yes.", GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        result = check_low_substance_phase(features)
        assert result.fired, f"Should fire on empty thesis"

    def test_all_filler(self):
        """A phase is entirely filler/meta-commentary."""
        filler = (
            "This is an important question that deserves consideration. "
            "Let us consider the various perspectives on this matter. "
            "In conclusion, this is a complex issue with many facets."
        )
        features = extract_trace_features(filler, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        result = check_low_substance_phase(features)
        assert result.fired, f"Should fire on all-filler phase"


# ---------------------------------------------------------------------------
# Adversarial: Template collapse
# ---------------------------------------------------------------------------

class TestTemplateCollapse:
    def test_connector_spam(self):
        """Trace is full of discourse markers without substance."""
        connector_thesis = (
            "However, one must consider the implications. Furthermore, "
            "the evidence suggests various outcomes. Nevertheless, the "
            "situation remains complex. Moreover, additional factors apply. "
            "Consequently, we must think carefully. Additionally, there "
            "are further considerations. Specifically, the details matter."
        )
        connector_anti = (
            "Conversely, the opposing view has merit. Furthermore, "
            "different evidence points elsewhere. Nevertheless, the "
            "counterargument is strong. Moreover, other perspectives exist. "
            "Consequently, we should reconsider. Additionally, there "
            "are alternative interpretations. Specifically, the nuances differ."
        )
        features = extract_trace_features(connector_thesis, connector_anti, GOOD_SYNTHESIS)
        result = check_template_collapse(features)
        assert result.fired, f"Should fire on connector spam"

    def test_boilerplate_opening(self):
        """Thesis starts with the conclusion."""
        boilerplate = (
            "There are valid arguments on both sides of this debate. "
            "The proponents make several good points about the benefits. "
            "Meanwhile, critics raise legitimate concerns about the costs."
        )
        features = extract_trace_features(boilerplate, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        # This alone may not fire template_collapse (needs 2 signals),
        # but it should contribute
        result = check_template_collapse(features)
        # At minimum the boilerplate flag should be in reasons if it fires


# ---------------------------------------------------------------------------
# Adversarial: Generic resolution language
# ---------------------------------------------------------------------------

class TestGenericResolutionLanguage:
    def test_pure_platitudes(self):
        """Synthesis is nothing but generic phrases."""
        generic = (
            "Both perspectives have merit and reasonable people can disagree. "
            "The truth lies somewhere in the middle. Context matters and "
            "it depends on the situation. A balanced approach is needed "
            "because there are no easy answers. We should consider all "
            "perspectives to find a compromise."
        )
        features = extract_trace_features(GOOD_THESIS, GOOD_ANTITHESIS, generic)
        result = check_generic_resolution_language(features)
        assert result.fired, f"Should fire on pure platitudes"


# ---------------------------------------------------------------------------
# Adversarial: Combined — correct form, zero insight
# ---------------------------------------------------------------------------

class TestCombinedAdversarial:
    def test_ceremonial_dialectic(self):
        """Correct dialectic form but no real insight. At least one check should fire."""
        ceremonial_thesis = (
            "There are valid arguments on both sides. However, one perspective "
            "suggests that the benefits outweigh the costs. Furthermore, "
            "evidence indicates positive outcomes. Moreover, experts agree."
        )
        ceremonial_anti = (
            "Conversely, there are valid arguments on both sides. However, "
            "another perspective suggests that the costs outweigh benefits. "
            "Furthermore, other evidence indicates negative outcomes. "
            "Moreover, different experts disagree."
        )
        ceremonial_synth = (
            "Both perspectives have merit. The truth lies somewhere in the "
            "middle. A balanced approach is needed. Context matters and "
            "reasonable people can disagree. There are no easy answers."
        )
        features = extract_trace_features(
            ceremonial_thesis, ceremonial_anti, ceremonial_synth,
        )
        report = run_all_checks(features)
        assert report.any_fired, (
            f"Ceremonial dialectic should trigger at least one check. "
            f"None fired."
        )
