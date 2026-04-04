"""Calibrate the scorer against the gold set.

Runs the scorer on all gold samples and compares verdicts to human labels.
Produces a calibration report identifying mismatches and suggesting
threshold adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from dialectic_dataset.goldset import CalibrationResult, GoldSample, GoldSet, compute_calibration
from dialectic_dataset.trace_scorer import TraceScore, score_trace


@dataclass
class DetailedCalibration:
    """Full calibration report with per-sample scores and diagnostics."""

    calibration: CalibrationResult
    per_sample: Dict[str, TraceScore]
    bucket_correlations: Dict[str, float]  # bucket name -> rank correlation with gold
    recommendations: List[str]


def score_goldset(goldset: GoldSet) -> Dict[str, TraceScore]:
    """Score all gold set samples and return a dict of sample_id -> TraceScore."""
    results: Dict[str, TraceScore] = {}
    for sample in goldset.samples:
        score = score_trace(sample.thesis, sample.antithesis, sample.synthesis)
        results[sample.sample_id] = score
    return results


def _rank_correlation(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation between two lists.

    Simple implementation without scipy dependency.
    Returns value in [-1, 1].
    """
    n = len(xs)
    if n < 3:
        return 0.0

    def _rank(vals: List[float]) -> List[float]:
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(indexed):
            ranks[idx] = float(rank + 1)
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)

    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))


def _gold_score_for_dimension(label_value: int) -> float:
    """Map a 1-5 label to [0, 1] range."""
    return (label_value - 1) / 4.0


def calibrate(goldset: GoldSet) -> DetailedCalibration:
    """Run full calibration of the scorer against the gold set.

    Returns
    -------
    DetailedCalibration with agreement rates, correlations, and
    actionable recommendations.
    """
    per_sample = score_goldset(goldset)

    # Build verdict dict
    scorer_verdicts = {
        sid: score.verdict for sid, score in per_sample.items()
    }

    # Core calibration
    cal = compute_calibration(goldset, scorer_verdicts)

    # Bucket correlations: compare scorer bucket scores to gold labels
    bucket_map = {
        "tension_quality": "tension_quality",
        "resolution_quality": "resolution_quality",
        "cross_phase_coherence": "coherence",
    }

    bucket_correlations: Dict[str, float] = {}
    for scorer_bucket, gold_field in bucket_map.items():
        scorer_vals: List[float] = []
        gold_vals: List[float] = []
        for sample in goldset.samples:
            if sample.sample_id in per_sample:
                score = per_sample[sample.sample_id]
                scorer_vals.append(getattr(score, scorer_bucket))
                gold_vals.append(
                    _gold_score_for_dimension(getattr(sample.label, gold_field))
                )
        bucket_correlations[scorer_bucket] = _rank_correlation(scorer_vals, gold_vals)

    # Generate recommendations
    recommendations = _generate_recommendations(cal, bucket_correlations, goldset, per_sample)

    return DetailedCalibration(
        calibration=cal,
        per_sample=per_sample,
        bucket_correlations=bucket_correlations,
        recommendations=recommendations,
    )


def _generate_recommendations(
    cal: CalibrationResult,
    correlations: Dict[str, float],
    goldset: GoldSet,
    per_sample: Dict[str, TraceScore],
) -> List[str]:
    """Generate actionable recommendations from calibration results."""
    recs: List[str] = []

    # Agreement rate
    if cal.agreement_rate < 0.8:
        recs.append(
            f"Agreement rate {cal.agreement_rate:.1%} is below 80% target. "
            f"Review {len(cal.mismatches)} mismatches and adjust thresholds."
        )

    # Adversarial catch rate
    if cal.adversarial_catch_rate < 1.0:
        missed = cal.adversarial_total - cal.adversarial_caught
        recs.append(
            f"CRITICAL: {missed} adversarial sample(s) not rejected. "
            f"Do not generate synthetic data until all adversarial samples are caught."
        )

    # Bucket correlations
    for bucket, corr in correlations.items():
        if corr < 0.5:
            recs.append(
                f"Low correlation ({corr:.2f}) between scorer's {bucket} "
                f"and gold labels. Consider revising the bucket's feature weights."
            )

    # Check for systematic over/under scoring
    false_keeps = [
        m for m in cal.mismatches
        if m["gold_verdict"] in ("review", "reject") and m["scorer_verdict"] == "keep"
    ]
    false_rejects = [
        m for m in cal.mismatches
        if m["gold_verdict"] == "keep" and m["scorer_verdict"] == "reject"
    ]

    if len(false_keeps) > len(false_rejects) * 2:
        recs.append(
            f"Scorer is too lenient: {len(false_keeps)} false keeps vs "
            f"{len(false_rejects)} false rejects. Consider raising KEEP_THRESHOLD."
        )
    elif len(false_rejects) > len(false_keeps) * 2:
        recs.append(
            f"Scorer is too strict: {len(false_rejects)} false rejects vs "
            f"{len(false_keeps)} false keeps. Consider lowering KEEP_THRESHOLD."
        )

    # Training usefulness gate
    usefulness_mismatches = 0
    for sample in goldset.samples:
        if sample.sample_id in per_sample:
            scorer_would_train = per_sample[sample.sample_id].would_train_on_this
            gold_would_train = sample.label.would_train_on_this
            if scorer_would_train != gold_would_train:
                usefulness_mismatches += 1

    if usefulness_mismatches > 0:
        total = len([s for s in goldset.samples if s.sample_id in per_sample])
        rate = usefulness_mismatches / max(1, total)
        if rate > 0.2:
            recs.append(
                f"Training usefulness gate disagrees on {usefulness_mismatches} "
                f"samples ({rate:.0%}). Review assess_training_usefulness() heuristics."
            )

    if not recs:
        recs.append("All calibration targets met. Scorer is ready for synthetic generation.")

    return recs


def print_calibration_report(result: DetailedCalibration) -> None:
    """Print a human-readable calibration report."""
    cal = result.calibration
    print("=" * 60)
    print("Scorer Calibration Report")
    print("=" * 60)
    print(f"  Total samples:         {cal.total}")
    print(f"  Agreement rate:        {cal.agreement_rate:.1%}")
    print(f"  Adversarial caught:    {cal.adversarial_caught}/{cal.adversarial_total}")
    print(f"  Adversarial rate:      {cal.adversarial_catch_rate:.0%}")
    print()
    print("Bucket correlations (scorer vs gold):")
    for bucket, corr in result.bucket_correlations.items():
        status = "ok" if corr >= 0.5 else "LOW"
        print(f"  {bucket:30s} r={corr:.3f}  [{status}]")
    print()
    if result.calibration.mismatches:
        print(f"Mismatches ({len(result.calibration.mismatches)}):")
        for m in result.calibration.mismatches:
            adv = " [ADVERSARIAL]" if m["is_adversarial"] else ""
            print(
                f"  {m['sample_id']:20s} "
                f"gold={m['gold_verdict']:8s} "
                f"scorer={m['scorer_verdict']:8s}"
                f"{adv}"
            )
    print()
    print("Recommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    print("=" * 60)
