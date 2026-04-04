"""Tests for data_formatter — training format conversion."""

import json
import tempfile
from pathlib import Path

from dialectic_dataset.data_formatter import (
    SYSTEM_PROMPT,
    format_explicit,
    format_implicit,
    prepare_dataset,
    trace_to_examples,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trace(
    trace_id: str = "test_001",
    category: str = "design_tradeoff",
    length_tier: str = "medium",
    quality: float = 0.80,
) -> dict:
    return {
        "trace_id": trace_id,
        "question": "Should startups optimize for speed or quality?",
        "category": category,
        "difficulty": "standard",
        "length_tier": length_tier,
        "thesis": "Speed matters because startups that ship faster learn faster.",
        "antithesis": "Quality matters because technical debt compounds and kills iteration speed.",
        "synthesis": "Speed dominates during search for product-market fit. Quality dominates after fit is found.",
        "composite_quality": quality,
        "tension_quality": 0.85,
        "resolution_quality": 0.80,
        "cross_phase_coherence": 0.75,
        "synthesis_sufficiency": 0.78,
        "verdict": "keep",
        "would_train_on_this": True,
        "structural_checks_passed": True,
        "fired_checks": [],
    }


# ---------------------------------------------------------------------------
# Format tests
# ---------------------------------------------------------------------------

class TestFormatExplicit:
    def test_contains_all_phases(self):
        result = format_explicit(
            "Test question?",
            "Thesis text.",
            "Antithesis text.",
            "Synthesis text.",
        )
        assert "Thesis text." in result
        assert "Antithesis text." in result
        assert "Synthesis text." in result

    def test_phases_separated(self):
        result = format_explicit(
            "Q?", "Phase A.", "Phase B.", "Phase C.",
        )
        parts = result.split("\n\n")
        assert len(parts) == 3


class TestFormatImplicit:
    def test_contains_synthesis(self):
        result = format_implicit(
            "Q?",
            "Thesis content.",
            "Antithesis content.",
            "The key insight is conditional resolution.",
        )
        assert "key insight" in result

    def test_returns_string(self):
        result = format_implicit("Q?", "T.", "A.", "S.")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Training example conversion
# ---------------------------------------------------------------------------

class TestTraceToExamples:
    def test_explicit_format(self):
        trace = _make_trace()
        ex = trace_to_examples(trace, format_type="explicit")
        assert ex.format_type == "explicit"
        assert len(ex.messages) == 3
        assert ex.messages[0]["role"] == "system"
        assert ex.messages[1]["role"] == "user"
        assert ex.messages[2]["role"] == "assistant"

    def test_implicit_format(self):
        trace = _make_trace()
        ex = trace_to_examples(trace, format_type="implicit")
        assert ex.format_type == "implicit"
        assert len(ex.messages) == 3

    def test_system_prompt_matches_spec(self):
        trace = _make_trace()
        ex = trace_to_examples(trace)
        assert ex.messages[0]["content"] == SYSTEM_PROMPT

    def test_metadata_present(self):
        trace = _make_trace()
        ex = trace_to_examples(trace)
        d = ex.to_jsonl_dict()
        assert "messages" in d
        assert "metadata" in d
        assert d["metadata"]["trace_id"] == "test_001"
        assert d["metadata"]["format"] == "explicit"
        assert d["metadata"]["quality_score"] == 0.80

    def test_jsonl_serializable(self):
        trace = _make_trace()
        ex = trace_to_examples(trace)
        # Should not raise
        serialized = json.dumps(ex.to_jsonl_dict())
        parsed = json.loads(serialized)
        assert parsed["messages"][1]["content"] == trace["question"]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

class TestPrepareDataset:
    def test_creates_files(self):
        traces = [_make_trace(f"t_{i}") for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = prepare_dataset(traces, output_dir=tmpdir, seed=42)
            assert (Path(tmpdir) / "train.jsonl").exists()
            assert (Path(tmpdir) / "val.jsonl").exists()
            assert (Path(tmpdir) / "test.jsonl").exists()
            assert (Path(tmpdir) / "manifest.json").exists()

    def test_split_sizes(self):
        traces = [_make_trace(f"t_{i}") for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = prepare_dataset(traces, output_dir=tmpdir, seed=42)
            assert stats.split_counts["train"] >= 15  # ~80%
            assert stats.split_counts["val"] >= 1
            assert stats.split_counts["test"] >= 1
            total = sum(stats.split_counts.values())
            assert total == 20

    def test_format_ratio(self):
        traces = [_make_trace(f"t_{i}") for i in range(100)]
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = prepare_dataset(
                traces, output_dir=tmpdir, seed=42, explicit_ratio=0.40,
            )
            explicit = stats.format_counts.get("explicit", 0)
            implicit = stats.format_counts.get("implicit", 0)
            assert 35 <= explicit <= 45  # ~40%
            assert 55 <= implicit <= 65  # ~60%

    def test_manifest_valid_json(self):
        traces = [_make_trace(f"t_{i}") for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = prepare_dataset(traces, output_dir=tmpdir)
            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["version"] == "1.0"
            assert "format_split" in manifest
            assert "splits" in manifest

    def test_jsonl_lines_are_valid(self):
        traces = [_make_trace(f"t_{i}") for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_dataset(traces, output_dir=tmpdir)
            train_path = Path(tmpdir) / "train.jsonl"
            with open(train_path) as f:
                for line in f:
                    data = json.loads(line)
                    assert "messages" in data
                    assert len(data["messages"]) == 3
                    assert data["messages"][0]["role"] == "system"

    def test_deterministic_with_seed(self):
        traces = [_make_trace(f"t_{i}") for i in range(20)]
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            prepare_dataset(traces, output_dir=d1, seed=42)
            prepare_dataset(traces, output_dir=d2, seed=42)
            t1 = (Path(d1) / "train.jsonl").read_text()
            t2 = (Path(d2) / "train.jsonl").read_text()
            assert t1 == t2
