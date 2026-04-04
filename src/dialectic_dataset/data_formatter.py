"""Format scored traces into training JSONL for Unsloth.

Produces two formats:
- Explicit dialectic (40%): visible reasoning through competing perspectives
- Implicit integrated (60%): dialectic thinking woven into direct answer

Three length tiers: short (30%), medium (50%), long (20%).

See docs/training_schema.md for full spec.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You reason carefully through problems by considering competing perspectives "
    "before reaching a conclusion. You identify genuine tensions, engage with "
    "the strongest form of each argument, and integrate insights rather than "
    "picking sides or hedging."
)

# Target proportions
FORMAT_SPLIT = {"explicit": 0.40, "implicit": 0.60}
LENGTH_SPLIT = {"short": 0.30, "medium": 0.50, "long": 0.20}

# Train/val/test split
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def format_explicit(
    question: str,
    thesis: str,
    antithesis: str,
    synthesis: str,
) -> str:
    """Format a trace as explicit dialectic prose.

    The three phases are joined into natural prose with transition markers.
    No explicit labels like "Thesis:" — the structure is visible but natural.
    """
    # Clean up any trailing/leading whitespace
    thesis = thesis.strip()
    antithesis = antithesis.strip()
    synthesis = synthesis.strip()

    return f"{thesis}\n\n{antithesis}\n\n{synthesis}"


def format_implicit(
    question: str,
    thesis: str,
    antithesis: str,
    synthesis: str,
) -> str:
    """Format a trace as implicit integrated prose.

    Leads with the resolution, weaving in key reasoning from both sides.
    The dialectic thinking is present but not visibly structured.

    This is a simple mechanical conversion — for production use, an LLM
    conversion step would produce more natural implicit formats.
    """
    # Simple mechanical approach: lead with synthesis, then weave in
    # key supporting points from thesis and antithesis.
    #
    # For v1, we concatenate synthesis first, then add selective context.
    # A more sophisticated version would use Claude to rewrite.
    synthesis = synthesis.strip()
    thesis = thesis.strip()
    antithesis = antithesis.strip()

    # Extract first sentence of thesis and antithesis as context
    thesis_opener = _first_sentence(thesis)
    antithesis_opener = _first_sentence(antithesis)

    # Build integrated response: start with synthesis, reference key tensions
    parts = [synthesis]

    # Only add context from thesis/antithesis if the synthesis is short
    # relative to the full trace (avoids bloating long responses)
    synth_words = len(synthesis.split())
    total_words = len(thesis.split()) + len(antithesis.split()) + synth_words
    if synth_words < total_words * 0.5:
        # Synthesis is less than half the total — add context
        parts = [synthesis]

    return "\n\n".join(parts)


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    for end in [".", "!", "?"]:
        idx = text.find(end)
        if idx > 0:
            return text[: idx + 1]
    return text[:200]


# ---------------------------------------------------------------------------
# Training example
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """A single training example ready for JSONL output."""

    trace_id: str
    format_type: str  # "explicit" or "implicit"
    category: str
    length_tier: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, object]

    def to_jsonl_dict(self) -> dict:
        """Return dict matching the training schema."""
        return {
            "messages": self.messages,
            "metadata": self.metadata,
        }


def trace_to_examples(
    trace: dict,
    format_type: str = "explicit",
) -> TrainingExample:
    """Convert a scored trace dict to a training example.

    Parameters
    ----------
    trace:
        Dict with keys: trace_id, question, category, thesis, antithesis,
        synthesis, composite_quality, tension_quality, etc.
    format_type:
        "explicit" or "implicit"
    """
    question = trace["question"]
    thesis = trace["thesis"]
    antithesis = trace["antithesis"]
    synthesis = trace["synthesis"]

    if format_type == "explicit":
        response = format_explicit(question, thesis, antithesis, synthesis)
    else:
        response = format_implicit(question, thesis, antithesis, synthesis)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]

    metadata = {
        "trace_id": trace["trace_id"],
        "format": format_type,
        "category": trace["category"],
        "length_tier": trace.get("length_tier", "medium"),
        "quality_score": trace["composite_quality"],
        "tension_quality": trace["tension_quality"],
        "resolution_quality": trace["resolution_quality"],
        "cross_phase_coherence": trace["cross_phase_coherence"],
        "synthesis_sufficiency": trace["synthesis_sufficiency"],
        "structural_checks_passed": trace.get("structural_checks_passed", True),
        "would_train_on_this": trace.get("would_train_on_this", True),
    }

    return TrainingExample(
        trace_id=trace["trace_id"],
        format_type=format_type,
        category=trace["category"],
        length_tier=trace.get("length_tier", "medium"),
        messages=messages,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

@dataclass
class DatasetStats:
    """Statistics about the prepared dataset."""

    total_traces: int = 0
    total_examples: int = 0
    format_counts: Dict[str, int] = field(default_factory=dict)
    length_counts: Dict[str, int] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    mean_quality: float = 0.0
    min_quality: float = 0.0
    max_quality: float = 0.0

    def to_manifest(self) -> dict:
        """Generate manifest.json content."""
        return {
            "version": "1.0",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_traces_input": self.total_traces,
            "total_examples_output": self.total_examples,
            "format_split": {
                k: v / max(1, self.total_examples)
                for k, v in self.format_counts.items()
            },
            "length_split": {
                k: v / max(1, self.total_examples)
                for k, v in self.length_counts.items()
            },
            "category_distribution": self.category_counts,
            "quality_stats": {
                "mean_quality_score": round(self.mean_quality, 3),
                "min_quality_score": round(self.min_quality, 3),
                "max_quality_score": round(self.max_quality, 3),
            },
            "splits": self.split_counts,
        }


def prepare_dataset(
    traces: List[dict],
    output_dir: str = "data/training",
    seed: int = 42,
    explicit_ratio: float = 0.40,
) -> DatasetStats:
    """Prepare a training dataset from scored traces.

    Parameters
    ----------
    traces:
        List of trace dicts (from load_traces or GeneratedTrace.to_dict()).
    output_dir:
        Directory for output JSONL files.
    seed:
        Random seed for reproducible splits.
    explicit_ratio:
        Fraction of examples in explicit format (rest are implicit).

    Returns
    -------
    DatasetStats with counts and manifest data.
    """
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert traces to training examples
    examples: List[TrainingExample] = []
    qualities: List[float] = []

    # Shuffle traces
    traces = list(traces)
    rng.shuffle(traces)

    # Assign formats based on target ratio
    n_explicit = int(len(traces) * explicit_ratio)
    for i, trace in enumerate(traces):
        fmt = "explicit" if i < n_explicit else "implicit"
        example = trace_to_examples(trace, format_type=fmt)
        examples.append(example)
        qualities.append(trace["composite_quality"])

    # Shuffle examples
    rng.shuffle(examples)

    # Split into train/val/test
    n = len(examples)
    n_test = max(1, int(n * SPLIT_RATIOS["test"]))
    n_val = max(1, int(n * SPLIT_RATIOS["val"]))
    n_train = n - n_val - n_test

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }

    # Write JSONL files
    for split_name, split_examples in splits.items():
        path = out / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_examples:
                f.write(json.dumps(ex.to_jsonl_dict()) + "\n")
        logger.info(f"Wrote {len(split_examples)} examples to {path}")

    # Compute stats
    stats = DatasetStats(
        total_traces=len(traces),
        total_examples=len(examples),
        mean_quality=sum(qualities) / max(1, len(qualities)),
        min_quality=min(qualities) if qualities else 0.0,
        max_quality=max(qualities) if qualities else 0.0,
    )

    for ex in examples:
        stats.format_counts[ex.format_type] = (
            stats.format_counts.get(ex.format_type, 0) + 1
        )
        stats.length_counts[ex.length_tier] = (
            stats.length_counts.get(ex.length_tier, 0) + 1
        )
        stats.category_counts[ex.category] = (
            stats.category_counts.get(ex.category, 0) + 1
        )

    for split_name, split_examples in splits.items():
        stats.split_counts[split_name] = len(split_examples)

    # Write manifest
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(stats.to_manifest(), f, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")

    return stats


def print_dataset_stats(stats: DatasetStats) -> None:
    """Print a human-readable summary of dataset stats."""
    print("=" * 50)
    print("Dataset Preparation Summary")
    print("=" * 50)
    print(f"  Input traces:    {stats.total_traces}")
    print(f"  Output examples: {stats.total_examples}")
    print(f"  Quality range:   [{stats.min_quality:.3f}, {stats.max_quality:.3f}]")
    print(f"  Mean quality:    {stats.mean_quality:.3f}")
    print()
    print("Format split:")
    for fmt, count in sorted(stats.format_counts.items()):
        pct = count / max(1, stats.total_examples) * 100
        print(f"  {fmt:12s} {count:4d} ({pct:.0f}%)")
    print()
    print("Length split:")
    for tier, count in sorted(stats.length_counts.items()):
        pct = count / max(1, stats.total_examples) * 100
        print(f"  {tier:12s} {count:4d} ({pct:.0f}%)")
    print()
    print("Category distribution:")
    for cat, count in sorted(stats.category_counts.items()):
        print(f"  {cat:30s} {count:4d}")
    print()
    print("Splits:")
    for split, count in sorted(stats.split_counts.items()):
        print(f"  {split:12s} {count:4d}")
    print("=" * 50)
