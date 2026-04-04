"""Tests for signal_extractor — the feature extraction layer."""

from dialectic_dataset.signal_extractor import (
    PhaseFeatures,
    TraceFeatures,
    content_words,
    cosine_terms,
    extract_phase_features,
    extract_trace_features,
    jaccard,
    sentences,
    term_set,
    tokenize,
    word_count,
)
from collections import Counter


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert tokenize("It's a test.") == ["it's", "a", "test"]

    def test_empty(self):
        assert tokenize("") == []


class TestContentWords:
    def test_removes_stopwords(self):
        cw = content_words("the quick brown fox jumps over the lazy dog")
        assert "the" not in cw
        assert "quick" in cw
        assert "fox" in cw

    def test_empty(self):
        assert content_words("the a an") == []


class TestSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third one."
        sents = sentences(text)
        assert len(sents) == 3

    def test_single_sentence(self):
        assert len(sentences("Just one.")) == 1

    def test_question(self):
        sents = sentences("Is this right? Yes it is.")
        assert len(sents) == 2


class TestJaccard:
    def test_identical(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_empty(self):
        assert jaccard(set(), set()) == 0.0


class TestCosineTerms:
    def test_identical(self):
        c = Counter({"a": 1, "b": 2})
        assert abs(cosine_terms(c, c) - 1.0) < 0.001

    def test_disjoint(self):
        assert cosine_terms(Counter({"a": 1}), Counter({"b": 1})) == 0.0

    def test_empty(self):
        assert cosine_terms(Counter(), Counter()) == 0.0


# ---------------------------------------------------------------------------
# Phase features
# ---------------------------------------------------------------------------

class TestExtractPhaseFeatures:
    def test_basic_extraction(self):
        text = (
            "Autonomous vehicles should prioritize safety above all else. "
            "The evidence from accident data shows that human error causes "
            "94% of crashes. Therefore, removing human drivers will save lives."
        )
        features = extract_phase_features(text)
        assert features.word_count > 20
        assert features.sentence_count >= 2
        assert features.claim_density > 0
        assert features.logical_density > 0  # "therefore" present
        assert len(features.content_terms) > 5

    def test_hedging_detection(self):
        text = (
            "Perhaps autonomous vehicles might possibly reduce accidents. "
            "It seems likely that they could potentially help. "
            "Arguably, this technology is uncertain in its effects."
        )
        features = extract_phase_features(text)
        assert features.hedging_density > 0.05

    def test_filler_detection(self):
        text = (
            "This is an important question. Let us consider the evidence. "
            "In conclusion, we should think about it more."
        )
        features = extract_phase_features(text)
        assert features.filler_sentence_count >= 2

    def test_short_text(self):
        features = extract_phase_features("Yes.")
        assert features.word_count == 1
        assert features.sentence_count == 1

    def test_discourse_markers(self):
        text = (
            "However, the situation is complex. Furthermore, we must consider "
            "alternatives. Nevertheless, the evidence points one way. "
            "Moreover, other factors apply."
        )
        features = extract_phase_features(text)
        assert features.discourse_marker_ratio > 0.3


# ---------------------------------------------------------------------------
# Trace features
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
    "sustainability. Both sides have real evidence, but they address different "
    "time horizons. Cash transfers demonstrably help recipients in the short "
    "term, while fiscal risks materialize over years. A phased approach — "
    "starting with targeted basic income for populations below the poverty line, "
    "funded by consolidating existing welfare programs rather than new taxes — "
    "captures the poverty reduction benefits while containing fiscal exposure. "
    "Work incentive concerns can be addressed through earned income supplements "
    "that phase out gradually rather than creating sharp cutoffs."
)


class TestExtractTraceFeatures:
    def test_good_trace(self):
        features = extract_trace_features(GOOD_THESIS, GOOD_ANTITHESIS, GOOD_SYNTHESIS)

        # Stance divergence should be significant (opposing arguments use different terms)
        assert 0.2 < features.stance_divergence <= 1.0

        # Synthesis should inherit from both sides
        assert features.thesis_inheritance > 0.05
        assert features.antithesis_inheritance > 0.05

        # Inheritance should be somewhat balanced
        assert features.inheritance_balance > 0.2

        # Some novelty in synthesis
        assert features.novelty_ratio > 0.0

        # Low generic resolution count in a good synthesis
        assert features.generic_resolution_count <= 1

    def test_identical_thesis_antithesis(self):
        features = extract_trace_features(GOOD_THESIS, GOOD_THESIS, GOOD_SYNTHESIS)
        # Stance divergence should be ~0 (identical)
        assert features.stance_divergence < 0.1

    def test_concession_detection(self):
        conceding_anti = (
            "While this is true and I agree that basic income helps, "
            "admittedly the evidence is strong, certainly people benefit. "
            "However, there might be small cost concerns."
        )
        features = extract_trace_features(GOOD_THESIS, conceding_anti, GOOD_SYNTHESIS)
        assert features.concession_ratio > 0.3
