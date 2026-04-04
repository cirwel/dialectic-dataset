"""Gold set management for scorer calibration.

Defines gold sample data structures, load/save, and calibration statistics.
The gold set is the source of truth — if the scorer disagrees with gold
labels, the scorer is wrong.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoldLabel:
    """Human labels for a gold set sample."""

    tension_quality: int              # 1-5
    resolution_quality: int           # 1-5
    coherence: int                    # 1-5
    training_usefulness: str          # "keep", "review", "reject"
    ritualization_risk: str           # "low", "medium", "high"
    would_train_on_this: bool         # yes/no
    overall_verdict: str              # "keep", "review", "reject"
    justification: str = ""           # 1-sentence reason for verdict


@dataclass
class GoldSample:
    """A single gold set sample with text and labels."""

    sample_id: str
    category: str                     # prompt family
    question: str                     # the prompt that generated the trace
    thesis: str
    antithesis: str
    synthesis: str
    label: GoldLabel
    is_adversarial: bool = False      # True if this is an adversarial bad sample
    adversarial_type: str = ""        # e.g., "fake_antithesis", "summary_synthesis"
    notes: str = ""                   # labeler notes


@dataclass
class GoldSet:
    """The full gold set for scorer calibration."""

    samples: List[GoldSample]
    version: str = "1.0"
    description: str = ""

    @property
    def good_samples(self) -> List[GoldSample]:
        return [s for s in self.samples if s.label.overall_verdict == "keep"]

    @property
    def mid_samples(self) -> List[GoldSample]:
        return [s for s in self.samples if s.label.overall_verdict == "review"]

    @property
    def bad_samples(self) -> List[GoldSample]:
        return [s for s in self.samples if s.label.overall_verdict == "reject"]

    @property
    def adversarial_samples(self) -> List[GoldSample]:
        return [s for s in self.samples if s.is_adversarial]

    def by_category(self, category: str) -> List[GoldSample]:
        return [s for s in self.samples if s.category == category]

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of the gold set."""
        return {
            "total": len(self.samples),
            "good": len(self.good_samples),
            "mid": len(self.mid_samples),
            "bad": len(self.bad_samples),
            "adversarial": len(self.adversarial_samples),
            "categories": sorted(set(s.category for s in self.samples)),
            "would_train_yes": sum(
                1 for s in self.samples if s.label.would_train_on_this
            ),
            "would_train_no": sum(
                1 for s in self.samples if not s.label.would_train_on_this
            ),
        }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _label_to_dict(label: GoldLabel) -> Dict[str, Any]:
    return asdict(label)


def _label_from_dict(d: Dict[str, Any]) -> GoldLabel:
    return GoldLabel(**d)


def _sample_to_dict(sample: GoldSample) -> Dict[str, Any]:
    d = asdict(sample)
    return d


def _sample_from_dict(d: Dict[str, Any]) -> GoldSample:
    label_data = d.pop("label")
    label = _label_from_dict(label_data)
    return GoldSample(label=label, **d)


def save_goldset(goldset: GoldSet, path: str) -> None:
    """Save gold set to a JSON file."""
    data = {
        "version": goldset.version,
        "description": goldset.description,
        "samples": [_sample_to_dict(s) for s in goldset.samples],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_goldset(path: str) -> GoldSet:
    """Load gold set from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    samples = [_sample_from_dict(s) for s in data["samples"]]
    return GoldSet(
        samples=samples,
        version=data.get("version", "1.0"),
        description=data.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of comparing scorer verdicts to gold set labels."""

    total: int
    agreement_count: int
    agreement_rate: float
    adversarial_caught: int
    adversarial_total: int
    adversarial_catch_rate: float
    mismatches: List[Dict[str, Any]]  # sample_id, gold_verdict, scorer_verdict


def compute_calibration(
    goldset: GoldSet,
    scorer_verdicts: Dict[str, str],
) -> CalibrationResult:
    """Compare scorer verdicts against gold set labels.

    Parameters
    ----------
    goldset:
        The gold set with human labels.
    scorer_verdicts:
        Dict mapping sample_id to scorer verdict ("keep", "review", "reject").

    Returns
    -------
    CalibrationResult with agreement rate and mismatch details.
    """
    total = 0
    agreements = 0
    adversarial_caught = 0
    adversarial_total = 0
    mismatches: List[Dict[str, Any]] = []

    for sample in goldset.samples:
        if sample.sample_id not in scorer_verdicts:
            continue

        total += 1
        gold_v = sample.label.overall_verdict
        scorer_v = scorer_verdicts[sample.sample_id]

        if gold_v == scorer_v:
            agreements += 1
        else:
            mismatches.append({
                "sample_id": sample.sample_id,
                "category": sample.category,
                "gold_verdict": gold_v,
                "scorer_verdict": scorer_v,
                "is_adversarial": sample.is_adversarial,
            })

        if sample.is_adversarial:
            adversarial_total += 1
            if scorer_v == "reject":
                adversarial_caught += 1

    return CalibrationResult(
        total=total,
        agreement_count=agreements,
        agreement_rate=agreements / max(1, total),
        adversarial_caught=adversarial_caught,
        adversarial_total=adversarial_total,
        adversarial_catch_rate=(
            adversarial_caught / max(1, adversarial_total)
        ),
        mismatches=mismatches,
    )
