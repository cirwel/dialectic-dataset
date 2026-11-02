"""Tests for Stage 4 thinking quality extraction and scoring."""

from dialectic_dataset.thinking_quality import (
    extract_think_block,
    extract_thinking_features,
    score_thinking_quality,
)


class TestExtractThinkBlock:
    def test_extracts_standard_think_tag(self):
        raw = "<think>First let me consider X.\nWait, Y is the bottleneck.</think>\n\nFinal answer."
        think, visible = extract_think_block(raw)
        assert think == "First let me consider X.\nWait, Y is the bottleneck."
        assert visible == "Final answer."

    def test_no_think_tag(self):
        raw = "Just a direct response without thinking tags."
        think, visible = extract_think_block(raw)
        assert think is None
        assert visible == raw

    def test_empty_think_tag(self):
        raw = "<think></think>\n\nPolished output."
        think, visible = extract_think_block(raw)
        assert think == ""
        assert visible == "Polished output."

    def test_unclosed_think_tag_does_not_leak_into_visible(self):
        raw = "<think>Wait, let me reconsider this because the mechanism is complex..."
        think, visible = extract_think_block(raw)
        assert "Wait, let me reconsider" in think
        assert visible == ""
        score = score_thinking_quality(raw)
        assert not score.has_genuine_thinking


class TestThinkingFeaturesAndScoring:
    def test_missing_or_empty_think_scores_zero(self):
        no_tag = score_thinking_quality("Just the visible response.")
        assert no_tag.composite == 0.0
        assert not no_tag.has_genuine_thinking

        empty_tag = score_thinking_quality("<think>   </think>\n\nVisible response.")
        assert empty_tag.composite == 0.0
        assert not empty_tag.has_genuine_thinking
        assert empty_tag.formulaic_penalty == 1.0

    def test_genuine_deliberative_thinking_scores_high(self):
        trace = (
            "<think>\n"
            "At first glance, strict code review gates slow down deployment velocity, "
            "so fast-moving teams should skip them. Wait, that assumes that review latency "
            "is the primary bottleneck rather than rollback recovery time. Let me reconsider. "
            "Actually, the underlying assumption in the 'move fast' argument is that defects "
            "are cheap to detect in production. Instead of hedging or saying it depends on team "
            "maturity, I shouldn't invent a fake industry statistic here. The real tension is "
            "not speed vs quality, but synchronous human approval vs automated fault isolation: "
            "because synchronous review couples deployment cadence to reviewer availability, "
            "it causes queueing delays when PR size grows, whereas automated canary rollback "
            "prevents cascading failures without blocking small diffs.\n"
            "</think>\n\n"
            "Synchronous human review gates should apply only to irreversible schema or security "
            "boundaries; routine application changes achieve both higher velocity and lower failure "
            "rates through automated canary verification."
        )
        score = score_thinking_quality(trace)
        assert score.features.has_think_tag
        assert score.features.direction_change_count >= 2
        assert score.features.assumption_surfacing_count >= 1
        assert score.features.self_correction_count >= 1
        assert score.features.reframing_count >= 1
        assert score.features.mechanism_count >= 2
        assert score.composite >= 0.65
        assert score.has_genuine_thinking

    def test_stub_or_formulaic_thinking_penalized(self):
        stub = (
            "<think>Let me think about both sides and provide a balanced synthesis.</think>\n\n"
            "Both sides have merit and context matters."
        )
        score = score_thinking_quality(stub)
        assert score.composite < 0.25
        assert not score.has_genuine_thinking
        assert score.formulaic_penalty > 0.2
