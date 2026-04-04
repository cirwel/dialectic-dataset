"""Build training data from a pool of scored traces.

Loads traces from JSONL files, filters by quality, applies best-of-N per
question, rejects fabricated citations, and splits into train/val/test.

Usage:
    python scripts/build_training.py --traces data/traces.jsonl --output-dir data/training
    python scripts/build_training.py --traces pool1.jsonl pool2.jsonl --min-composite 0.5 --top-k 2
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dialectic_dataset.falsifiability_extractor import (
    extract_falsifiability_features,
    score_falsifiability,
    _count_fabricated_citations,
)

SYSTEM_PROMPT = (
    "You reason carefully through problems by considering competing perspectives "
    "before reaching a conclusion. You identify genuine tensions, engage with "
    "the strongest form of each argument, and integrate insights rather than "
    "picking sides or hedging."
)


def trace_to_training(trace: dict, keep_thinking: bool = False) -> dict:
    response = trace["response"]
    if not keep_thinking and "</think>" in response:
        response = response.split("</think>", 1)[1].strip()

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": trace["question"]},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "trace_id": trace.get("trace_id", "unknown"),
            "domain": trace.get("domain", "unknown"),
            "generation_model": trace.get("generation_model", "unknown"),
            "falsifiability_composite": trace.get("falsifiability_composite", 0),
            "word_count": len(response.split()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build training data from trace pool")
    parser.add_argument("--traces", nargs="+", required=True, help="Input trace JSONL files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-composite", type=float, default=0.0, help="Min falsifiability composite")
    parser.add_argument("--top-k", type=int, default=1, help="Top-K per question")
    parser.add_argument("--max-fabrication", type=int, default=0, help="Max fabricated citations allowed (default: 0)")
    parser.add_argument("--keep-thinking", action="store_true", help="Keep <think> blocks in training data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load traces from all input files
    traces = []
    for trace_path in args.traces:
        count = 0
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                count += 1

                if t.get("quality_gate") not in (None, "pass"):
                    continue

                # Re-score and check fabrication
                response = t["response"]
                if "</think>" in response:
                    response = response.split("</think>", 1)[1].strip()

                fab = _count_fabricated_citations(response)
                if fab > args.max_fabrication:
                    continue

                features = extract_falsifiability_features(response)
                score = score_falsifiability(features)
                t["falsifiability_composite"] = round(score.composite, 4)
                traces.append(t)

        logger.info(f"Loaded {trace_path}: {count} total")

    logger.info(f"Combined pool: {len(traces)} traces (after fabrication filter)")

    # Filter by composite
    passing = [t for t in traces if t.get("falsifiability_composite", 0) >= args.min_composite]
    logger.info(f"Passing (composite >= {args.min_composite}): {len(passing)}")

    if not passing:
        logger.error("No passing traces!")
        return

    # Best-of-N per question
    by_question = defaultdict(list)
    for t in passing:
        by_question[t["question"]].append(t)

    selected = []
    for question, group in by_question.items():
        group.sort(key=lambda t: t["falsifiability_composite"], reverse=True)
        selected.extend(group[:args.top_k])

    logger.info(f"Selected {len(selected)} traces ({len(by_question)} questions, top-{args.top_k})")

    # Convert to training format
    examples = [trace_to_training(t, keep_thinking=args.keep_thinking) for t in selected]

    # Shuffle and split: 85/7.5/7.5
    random.seed(args.seed)
    random.shuffle(examples)

    n = len(examples)
    n_val = max(1, int(n * 0.075))
    n_test = max(1, int(n * 0.075))
    n_train = n - n_val - n_test

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }

    for name, data in splits.items():
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"Wrote {len(data)} {name} examples to {path}")

    # Manifest
    composites = sorted(t["falsifiability_composite"] for t in selected)
    manifest = {
        "total_in_pool": len(traces),
        "passing": len(passing),
        "selected": len(selected),
        "unique_questions": len(by_question),
        "top_k": args.top_k,
        "min_composite": args.min_composite,
        "max_fabrication": args.max_fabrication,
        "keep_thinking": args.keep_thinking,
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "composite_median": composites[len(composites) // 2],
        "composite_min": composites[0],
        "composite_max": composites[-1],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
