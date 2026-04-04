"""Convert gold set good samples to the generated traces format.

Used for smoke-testing the prepare pipeline without API generation.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dialectic_dataset.goldset import load_goldset
from dialectic_dataset.trace_scorer import score_trace


def main():
    goldset = load_goldset("data/goldset/goldset_v1.json")

    traces = []
    for sample in goldset.samples:
        if sample.label.overall_verdict != "keep":
            continue

        score = score_trace(sample.thesis, sample.antithesis, sample.synthesis)

        trace = {
            "trace_id": f"gold_{sample.sample_id}",
            "question": sample.question,
            "category": sample.category,
            "difficulty": "standard",
            "length_tier": "medium",
            "thesis": sample.thesis,
            "antithesis": sample.antithesis,
            "synthesis": sample.synthesis,
            "composite_quality": score.composite_quality,
            "tension_quality": score.tension_quality,
            "resolution_quality": score.resolution_quality,
            "cross_phase_coherence": score.cross_phase_coherence,
            "synthesis_sufficiency": score.synthesis_sufficiency,
            "verdict": score.verdict,
            "would_train_on_this": score.would_train_on_this,
            "structural_checks_passed": score.structural_report.passed,
            "fired_checks": score.structural_report.fired_names,
            "generation_model": "human",
            "generated_at": time.time(),
        }
        traces.append(trace)

    output = Path("data/generated")
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "traces.jsonl"

    with open(out_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(traces)} traces to {out_path}")
    for t in traces:
        print(f"  {t['trace_id']:25s} q={t['composite_quality']:.3f} v={t['verdict']} train={t['would_train_on_this']}")


if __name__ == "__main__":
    main()
