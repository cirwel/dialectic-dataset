"""Tests for Stage 4 thinking quality extraction, scoring, and goldset calibration."""

import json
import sys
from pathlib import Path

from dialectic_dataset.falsifiability_extractor import _count_fabricated_citations
from dialectic_dataset.thinking_quality import (
    extract_think_block,
    extract_thinking_features,
    score_thinking_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_falsifiable_traces import format_with_thinking, quick_falsifiability_check  # noqa: E402


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

    def test_meta_rubric_echo_in_thinking_disqualified(self):
        meta_echo = (
            "<think>\n"
            "Here's a thinking process designed to meet all constraints:\n"
            "1. **Analyze the Core Tension:** Speed vs quality.\n"
            "2. **Initial Impulse (The Easy Answer):** Move fast because startups need feedback.\n"
            "3. **Reconsideration/Self-Correction (Wait...):** Actually, that assumes debt is cheap.\n"
            "</think>\n\n"
            "If the team has fewer than 10 engineers, prioritize speed; above 50 engineers, prioritize quality."
        )
        score = score_thinking_quality(meta_echo)
        assert score.features.meta_echo_count >= 3
        assert not score.has_genuine_thinking
        assert score.formulaic_penalty >= 0.50


class TestStage1GenerationPreservesThinking:
    def test_format_with_thinking_wraps_separate_thinking_field(self):
        combined = format_with_thinking(
            content="Visible conclusion with concrete threshold.",
            thinking="Wait, let me reconsider the hidden assumption.",
            keep_thinking=True,
        )
        assert combined.startswith("<think>\nWait, let me reconsider")
        assert "</think>\n\nVisible conclusion" in combined

    def test_format_with_thinking_strips_when_opted_out(self):
        stripped = format_with_thinking(
            content="<think>Internal thought</think>\n\nVisible conclusion.",
            keep_thinking=False,
        )
        assert "<think>" not in stripped
        assert stripped == "Visible conclusion."


class TestGoldsetThinkingCalibration:
    def test_goldset_thinking_v1_achieves_clean_separation(self):
        gold_path = PROJECT_ROOT / "data" / "goldset" / "goldset_thinking_v1.json"
        with open(gold_path) as f:
            data = json.load(f)

        samples = data["samples"]
        assert len(samples) == 12

        for sample in samples:
            sid = sample["sample_id"]
            resp = sample["response"]
            expected = sample["expected"]

            think_score = score_thinking_quality(resp)
            _, visible = extract_think_block(resp)
            fab_count = _count_fabricated_citations(resp)

            assert think_score.has_genuine_thinking is expected["has_genuine_thinking"], (
                f"{sid}: expected has_genuine_thinking={expected['has_genuine_thinking']}, "
                f"got {think_score.has_genuine_thinking} (composite={think_score.composite})"
            )
            if "min_thinking_composite" in expected:
                assert think_score.composite >= expected["min_thinking_composite"], (
                    f"{sid}: composite {think_score.composite} < {expected['min_thinking_composite']}"
                )
            if "max_thinking_composite" in expected:
                assert think_score.composite <= expected["max_thinking_composite"], (
                    f"{sid}: composite {think_score.composite} > {expected['max_thinking_composite']}"
                )
            if expected["fabricated_citations"] > 0:
                assert fab_count >= expected["fabricated_citations"], (
                    f"{sid}: expected fabricated citations, got {fab_count}"
                )
            else:
                assert fab_count == 0, f"{sid}: unexpected fabricated citation count {fab_count}"

            passes_gate = (
                think_score.has_genuine_thinking
                and fab_count == 0
                and bool(visible)
            )
            assert passes_gate is expected["should_pass_training_gate"], (
                f"{sid}: training gate mismatch"
            )

