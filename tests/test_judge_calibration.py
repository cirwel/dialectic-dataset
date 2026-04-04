"""Tests for judge_calibration — scorer calibration against gold set."""

from dialectic_dataset.goldset import GoldLabel, GoldSample, GoldSet
from dialectic_dataset.judge_calibration import calibrate, score_goldset


def _make_sample(
    sample_id: str,
    thesis: str,
    antithesis: str,
    synthesis: str,
    verdict: str = "keep",
    tension: int = 4,
    resolution: int = 4,
    coherence: int = 4,
    would_train: bool = True,
    is_adversarial: bool = False,
) -> GoldSample:
    return GoldSample(
        sample_id=sample_id,
        category="design_tradeoff",
        question="test question",
        thesis=thesis,
        antithesis=antithesis,
        synthesis=synthesis,
        label=GoldLabel(
            tension_quality=tension,
            resolution_quality=resolution,
            coherence=coherence,
            training_usefulness=verdict,
            ritualization_risk="low",
            would_train_on_this=would_train,
            overall_verdict=verdict,
        ),
        is_adversarial=is_adversarial,
    )


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

BAD_SYNTHESIS = (
    "Both perspectives have merit. The truth lies somewhere in the middle. "
    "Context matters and it depends on the situation. A balanced approach "
    "is needed because there are no easy answers."
)


class TestScoreGoldset:
    def test_scores_all_samples(self):
        goldset = GoldSet(samples=[
            _make_sample("s1", GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS),
            _make_sample("s2", GOOD_THESIS, GOOD_ANTITHESIS, BAD_SYNTHESIS,
                         verdict="reject", tension=2, resolution=1, coherence=2,
                         would_train=False, is_adversarial=True),
        ])
        scores = score_goldset(goldset)
        assert "s1" in scores
        assert "s2" in scores
        assert len(scores) == 2


class TestCalibrate:
    def test_basic_calibration(self):
        goldset = GoldSet(samples=[
            _make_sample("good1", GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS),
            _make_sample("bad1", GOOD_THESIS, GOOD_ANTITHESIS, BAD_SYNTHESIS,
                         verdict="reject", tension=2, resolution=1, coherence=2,
                         would_train=False, is_adversarial=True),
        ])
        result = calibrate(goldset)
        assert result.calibration.total == 2
        assert result.calibration.adversarial_total == 1

    def test_recommendations_present(self):
        goldset = GoldSet(samples=[
            _make_sample("good1", GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS),
        ])
        result = calibrate(goldset)
        assert len(result.recommendations) > 0

    def test_adversarial_flag(self):
        """Adversarial samples that aren't caught should generate a CRITICAL recommendation."""
        goldset = GoldSet(samples=[
            _make_sample("adv1", GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS,
                         verdict="reject", is_adversarial=True),
        ])
        result = calibrate(goldset)
        # The scorer will likely give "keep" for a good trace labeled as adversarial reject
        if result.calibration.adversarial_catch_rate < 1.0:
            critical_recs = [r for r in result.recommendations if "CRITICAL" in r]
            assert len(critical_recs) > 0
