"""Tests for trace_scorer — weighted quality scoring."""

from dialectic_dataset.trace_scorer import (
    KEEP_THRESHOLD,
    REVIEW_THRESHOLD,
    score_trace,
)


# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# Good trace scoring
# ---------------------------------------------------------------------------

class TestGoodTrace:
    def test_verdict_is_keep(self):
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        assert score.verdict == "keep", (
            f"Good trace should be 'keep', got '{score.verdict}' "
            f"(composite={score.composite_quality})"
        )

    def test_composite_above_threshold(self):
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        assert score.composite_quality >= KEEP_THRESHOLD

    def test_would_train_on_this(self):
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        assert score.would_train_on_this

    def test_structural_checks_pass(self):
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        assert score.structural_report.passed

    def test_all_buckets_positive(self):
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        assert score.tension_quality > 0.3
        assert score.resolution_quality > 0.3
        assert score.cross_phase_coherence > 0.2
        assert score.synthesis_sufficiency > 0.2


# ---------------------------------------------------------------------------
# Bad traces (adversarial) should be rejected
# ---------------------------------------------------------------------------

class TestBadTraceRejected:
    def test_fake_antithesis_rejected(self):
        """Near-identical antithesis (same terms, minor rewording)."""
        fake_anti = (
            "Universal basic income would reduce poverty and provide economic "
            "security. Studies from Finland and Kenya show that direct cash "
            "transfers improve health outcomes and reduce stress. People who "
            "receive basic income start businesses and pursue education, "
            "generating economic growth."
        )
        score = score_trace(GOOD_THESIS, fake_anti, GOOD_SYNTHESIS)
        assert score.verdict == "reject", (
            f"Fake antithesis should be rejected, got '{score.verdict}'"
        )

    def test_summary_synthesis_rejected(self):
        summary = (
            "On one hand, proponents argue that basic income reduces poverty "
            "and improves health outcomes through direct cash transfers. "
            "On the other hand, opponents argue that basic income is "
            "economically unsustainable and would reduce work incentives."
        )
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, summary)
        assert score.verdict != "keep", (
            f"Summary-only synthesis should not be 'keep', got '{score.verdict}'"
        )

    def test_platitude_synthesis_rejected(self):
        platitudes = (
            "Both perspectives have merit and reasonable people can disagree. "
            "The truth lies somewhere in the middle. Context matters and "
            "it depends on the situation. A balanced approach is needed "
            "because there are no easy answers."
        )
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, platitudes)
        assert score.verdict == "reject", (
            f"Platitude synthesis should be rejected, got '{score.verdict}'"
        )

    def test_empty_phase_rejected(self):
        score = score_trace("Yes.", GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        assert score.verdict == "reject"

    def test_ceremonial_trace_not_trainable(self):
        ceremonial_thesis = (
            "There are valid arguments on both sides. However, one perspective "
            "suggests that the benefits outweigh the costs. Furthermore, "
            "evidence indicates positive outcomes."
        )
        ceremonial_anti = (
            "Conversely, there are valid arguments on both sides. However, "
            "another perspective suggests that the costs outweigh benefits. "
            "Furthermore, other evidence indicates negative outcomes."
        )
        ceremonial_synth = (
            "Both perspectives have merit. The truth lies somewhere in the "
            "middle. A balanced approach is needed. Context matters."
        )
        score = score_trace(ceremonial_thesis, ceremonial_anti, ceremonial_synth)
        assert not score.would_train_on_this, (
            f"Ceremonial trace should not be trainable"
        )


# ---------------------------------------------------------------------------
# Mid-quality traces
# ---------------------------------------------------------------------------

class TestMidQualityTrace:
    def test_decent_but_not_great(self):
        """A trace that's okay but has weak synthesis."""
        weak_synth = (
            "Both the benefits of basic income and the fiscal concerns are "
            "real. Perhaps a gradual implementation could work. More research "
            "might help clarify the tradeoffs."
        )
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, weak_synth)
        # Should be review or reject, not keep
        assert score.verdict != "keep" or not score.would_train_on_this


# ---------------------------------------------------------------------------
# Component transparency
# ---------------------------------------------------------------------------

class TestComponentBreakdown:
    def test_components_present(self):
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        assert "tension_quality" in score.components
        assert "resolution_quality" in score.components
        assert "cross_phase_coherence" in score.components
        assert "synthesis_sufficiency" in score.components

    def test_all_components_bounded(self):
        score = score_trace(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)
        for key, value in score.components.items():
            assert 0.0 <= value <= 1.0, f"Component {key}={value} out of bounds"
