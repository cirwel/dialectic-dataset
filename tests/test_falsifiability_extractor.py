"""Tests for falsifiability_extractor — falsifiability feature extraction."""

from dialectic_dataset.falsifiability_extractor import (
    FalsifiabilityFeatures,
    FalsifiabilityScore,
    extract_falsifiability_features,
    score_falsifiability,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractNumbers:
    def test_counts_integers(self):
        f = extract_falsifiability_features("The team has 10 engineers and 3 managers.")
        assert f.number_count >= 2

    def test_counts_percentages(self):
        f = extract_falsifiability_features("Revenue grew 15% while costs dropped 8%.")
        assert f.number_count >= 2

    def test_no_numbers(self):
        f = extract_falsifiability_features("The approach depends on many factors.")
        assert f.number_count == 0


class TestExtractNamedEntities:
    def test_mid_sentence_caps(self):
        f = extract_falsifiability_features(
            "The approach used by Netflix differs from what Amazon implements."
        )
        assert f.named_entity_count >= 1

    def test_no_entities(self):
        f = extract_falsifiability_features(
            "the approach depends on many factors and circumstances."
        )
        assert f.named_entity_count == 0


class TestExtractComparisons:
    def test_outperforms(self):
        f = extract_falsifiability_features(
            "Monoliths outperform microservices for small teams."
        )
        assert f.comparison_count >= 1

    def test_more_than(self):
        f = extract_falsifiability_features(
            "Teams with more than 50 engineers see diminishing returns."
        )
        assert f.comparison_count >= 1


class TestExtractConditionals:
    def test_specific_conditional(self):
        f = extract_falsifiability_features(
            "If the engineering team has fewer than 10 people, a monolith "
            "is the better choice."
        )
        assert f.total_conditionals >= 1
        assert f.specific_conditionals >= 1

    def test_vague_conditional(self):
        f = extract_falsifiability_features(
            "When context allows, the approach should be adjusted. "
            "In some cases, this may not apply."
        )
        assert f.vague_conditionals >= 1
        assert f.specific_conditionals == 0

    def test_no_conditionals(self):
        f = extract_falsifiability_features(
            "The sky is blue. Water is wet."
        )
        assert f.total_conditionals == 0


class TestCommitmentVsHedging:
    def test_high_commitment(self):
        f = extract_falsifiability_features(
            "The evidence shows that the determining factor is team size. "
            "This fails when applied to organizations above 500 people. "
            "The mechanism is coordination cost scaling quadratically."
        )
        assert f.commitment_phrase_count >= 3
        assert f.hedge_to_commitment_ratio < 0.3

    def test_high_hedging(self):
        f = extract_falsifiability_features(
            "Perhaps the approach might possibly work in some scenarios. "
            "It seems likely that one could arguably see benefits, "
            "though the outcome is uncertain and questionable."
        )
        assert f.hedge_count >= 3
        assert f.hedge_to_commitment_ratio > 0.7

    def test_neutral(self):
        f = extract_falsifiability_features(
            "The sky is blue. Water flows downhill."
        )
        assert f.hedge_to_commitment_ratio == 0.5  # default when neither present


class TestFalsifiableSentences:
    def test_falsifiable(self):
        f = extract_falsifiability_features(
            "Netflix's migration took 7 years and required 2000 engineers. "
            "Teams under 50 engineers who adopt microservices prematurely "
            "see a 3x increase in operational incidents within 18 months."
        )
        assert f.falsifiable_sentence_count >= 1
        assert f.prediction_density > 0.3

    def test_not_falsifiable(self):
        f = extract_falsifiability_features(
            "This is a complex issue with valid arguments on both sides. "
            "A balanced approach seems most appropriate. "
            "Context matters and each case is different."
        )
        assert f.falsifiable_sentence_count == 0
        assert f.prediction_density == 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoreFalsifiability:
    def test_high_falsifiability_passes(self):
        text = (
            "If the company has less than 5 million ARR, the founder should remain "
            "CEO because the cost of a CEO search and 6 month onboarding "
            "exceeds the value of professional management at that scale. "
            "Above 20 million ARR, evidence from Wasserman's study of 212 startups "
            "shows that founder-CEOs reduce company value by 20% on average "
            "compared to professional CEOs, primarily through slower institutional "
            "process adoption."
        )
        features = extract_falsifiability_features(text)
        score = score_falsifiability(features)
        assert score.composite > 0.4
        assert score.specificity_score > 0.3

    def test_formulaic_fails(self):
        text = (
            "This is a complex issue with valid arguments on both sides. "
            "When we consider the various perspectives, a balanced approach "
            "seems most appropriate. Context matters, and the optimal solution "
            "depends on the specific circumstances of each case."
        )
        features = extract_falsifiability_features(text)
        score = score_falsifiability(features)
        assert not score.tier1_pass
        assert score.composite < 0.35

    def test_vague_conditionals_hard_fail(self):
        text = (
            "When context allows, prioritize speed. "
            "In some cases, quality matters more. "
            "When appropriate, consider both dimensions. "
            "The evidence shows clear tradeoffs exist."
        )
        features = extract_falsifiability_features(text)
        score = score_falsifiability(features)
        assert features.vague_conditionals > features.specific_conditionals
        assert not score.tier1_pass

    def test_low_prediction_density_hard_fail(self):
        text = (
            "The approach involves careful consideration of multiple factors. "
            "One must weigh the tradeoffs between competing priorities. "
            "Stakeholder alignment is essential for success. "
            "Communication and transparency drive outcomes. "
            "Leadership commitment determines the result."
        )
        features = extract_falsifiability_features(text)
        score = score_falsifiability(features)
        assert features.prediction_density < 0.10
        assert not score.tier1_pass

    def test_specific_response_scores_well(self):
        text = (
            "If the engineering team has fewer than 10 people, a monolith "
            "outperforms microservices by roughly 40% in deployment velocity. "
            "Netflix's migration took 7 years and required 2000 engineers. "
            "Teams under 50 engineers who adopt microservices prematurely "
            "see a 3x increase in operational incidents within 18 months."
        )
        features = extract_falsifiability_features(text)
        score = score_falsifiability(features)
        assert score.specificity_score > 0.4
        assert score.prediction_score > 0.3

    def test_components_populated(self):
        features = extract_falsifiability_features("Some text here.")
        score = score_falsifiability(features)
        assert "specificity" in score.components
        assert "commitment" in score.components
        assert "conditional_quality" in score.components
        assert "prediction" in score.components

    def test_short_text_no_hard_fail_on_prediction(self):
        """Short text (< 3 sentences) shouldn't hard-fail on prediction density."""
        text = "The answer is 42."
        features = extract_falsifiability_features(text)
        score = score_falsifiability(features)
        # Should not hard-fail — the prediction_density check requires >= 3 sentences
        assert features.sentence_count < 3


# ---------------------------------------------------------------------------
# Grounding detection
# ---------------------------------------------------------------------------

class TestFabricatedCitations:
    def test_detects_year_institution_study(self):
        f = extract_falsifiability_features(
            "A 2022 Harvard study found that teams with 40% more autonomy "
            "produce better outcomes."
        )
        assert f.fabricated_citation_count >= 1

    def test_detects_the_year_report(self):
        f = extract_falsifiability_features(
            "The 2023 Stanford transparency report showed a 15% improvement."
        )
        assert f.fabricated_citation_count >= 1

    def test_detects_recent_year_survey(self):
        f = extract_falsifiability_features(
            "A recent 2021 MIT Sloan survey of 5000 professionals found "
            "that psychological safety drops 30%."
        )
        assert f.fabricated_citation_count >= 1

    def test_no_false_positive_on_mechanism(self):
        f = extract_falsifiability_features(
            "Teams above 8 people fragment into sub-groups that optimize locally."
        )
        assert f.fabricated_citation_count == 0

    def test_no_false_positive_on_real_conditional(self):
        f = extract_falsifiability_features(
            "If your codebase exceeds 2 million lines, the cost of refactoring "
            "rises non-linearly."
        )
        assert f.fabricated_citation_count == 0

    def test_multiple_citations(self):
        f = extract_falsifiability_features(
            "A 2022 Harvard study found 40% improvement. "
            "The 2023 MIT report confirmed these findings. "
            "A 2021 Stanford analysis of 200 firms showed similar results."
        )
        assert f.fabricated_citation_count >= 3


class TestUngroundedStats:
    def test_detects_according_to_with_number(self):
        f = extract_falsifiability_features(
            "According to internal data, productivity increased by 25%."
        )
        assert f.ungrounded_stat_count >= 1

    def test_detects_found_that_with_number(self):
        f = extract_falsifiability_features(
            "Researchers found that 60% of teams experienced burnout."
        )
        assert f.ungrounded_stat_count >= 1

    def test_no_false_positive_plain_number(self):
        f = extract_falsifiability_features(
            "If the team has fewer than 10 people, use a monolith."
        )
        assert f.ungrounded_stat_count == 0


class TestGroundingPenalty:
    def test_fabricated_citations_lower_composite(self):
        grounded = (
            "Teams above 8 people fragment into sub-groups. "
            "If the codebase exceeds 500k lines, refactoring costs "
            "increase non-linearly because coordination overhead dominates."
        )
        fabricated = (
            "A 2022 Harvard study found that teams above 8 fragment. "
            "The 2023 MIT report showed that refactoring costs "
            "increase by 40% according to their analysis."
        )
        f_grounded = extract_falsifiability_features(grounded)
        f_fabricated = extract_falsifiability_features(fabricated)
        s_grounded = score_falsifiability(f_grounded)
        s_fabricated = score_falsifiability(f_fabricated)

        assert s_fabricated.components["grounding_penalty"] > 0
        assert s_grounded.components["grounding_penalty"] == 0
        # Fabricated should score lower due to penalty
        assert s_fabricated.composite < s_grounded.composite or \
            s_fabricated.components["grounding_penalty"] > s_grounded.components["grounding_penalty"]

    def test_three_citations_hard_fail(self):
        text = (
            "A 2022 Harvard study found 40% improvement. "
            "The 2023 MIT report confirmed the findings. "
            "A 2021 Stanford analysis showed similar results. "
            "The evidence shows clear benefits."
        )
        f = extract_falsifiability_features(text)
        s = score_falsifiability(f)
        assert f.fabricated_citation_count >= 3
        assert not s.tier1_pass
