"""Build training data from a pool of scored traces.

Loads traces from JSONL files, filters by quality and Stage 2/3 verification,
scores Stage 4 <think> deliberative quality, applies best-of-N per question
using combined thinking + falsifiability selection, rejects fabricated
citations, and splits into train/val/test.

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
from dialectic_dataset.thinking_quality import (
    extract_think_block,
    score_thinking_quality,
)

SYSTEM_PROMPT = (
    "You reason carefully through problems by considering competing perspectives "
    "before reaching a conclusion. You identify genuine tensions, engage with "
    "the strongest form of each argument, and integrate insights rather than "
    "picking sides or hedging."
)


def compute_selection_score(
    falsifiability_composite: float,
    thinking_composite: float,
    has_think_tag: bool,
    thinking_weight: float = 0.6,
) -> float:
    """Compute best-of-N selection score weighting thinking quality over raw output metrics."""
    if not has_think_tag:
        return round(falsifiability_composite, 4)
    w = max(0.0, min(1.0, thinking_weight))
    return round(w * thinking_composite + (1.0 - w) * falsifiability_composite, 4)


def passes_verification_gates(trace: dict, require_verified: bool = False) -> bool:
    """Check Stage 2 verification and Stage 3 adversarial survival gates on a trace."""
    ver_pass = trace.get("verification_pass")
    adv_pass = trace.get("adversarial_survives")

    if ver_pass is False or adv_pass is False:
        return False
    if trace.get("any_fabrication_detected") is True:
        return False
    if require_verified and (ver_pass is not True or adv_pass is not True):
        return False
    return True


def trace_to_training(trace: dict, keep_thinking: bool = True) -> dict:
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
            "thinking_composite": trace.get("thinking_composite", 0),
            "selection_score": trace.get(
                "selection_score", trace.get("falsifiability_composite", 0)
            ),
            "verification_pass": trace.get("verification_pass"),
            "adversarial_survives": trace.get("adversarial_survives"),
            "word_count": len(response.split()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build training data from trace pool")
    parser.add_argument("--traces", nargs="+", required=True, help="Input trace JSONL files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-composite", type=float, default=0.0, help="Min falsifiability composite")
    parser.add_argument(
        "--min-thinking-quality",
        type=float,
        default=0.0,
        help="Min Stage 4 thinking quality composite (default: 0.0)",
    )
    parser.add_argument(
        "--thinking-weight",
        type=float,
        default=0.6,
        help="Weight on thinking quality vs falsifiability in best-of-N selection (default: 0.6)",
    )
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Require explicit Stage 2 verification_pass=True and Stage 3 adversarial_survives=True",
    )
    parser.add_argument("--top-k", type=int, default=1, help="Top-K per question")
    parser.add_argument("--max-fabrication", type=int, default=0, help="Max fabricated citations allowed (default: 0)")
    parser.add_argument(
        "--keep-thinking",
        dest="keep_thinking",
        action="store_true",
        default=True,
        help="Keep <think> blocks in training data (default: True)",
    )
    parser.add_argument(
        "--strip-thinking",
        dest="keep_thinking",
        action="store_false",
        help="Strip <think> blocks from training data",
    )
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

                if not passes_verification_gates(t, require_verified=args.require_verified):
                    continue

                raw_response = t["response"]
                think_score = score_thinking_quality(raw_response)
                if think_score.composite < args.min_thinking_quality:
                    continue

                _, visible_response = extract_think_block(raw_response)

                fab = _count_fabricated_citations(visible_response)
                if fab > args.max_fabrication:
                    continue

                features = extract_falsifiability_features(visible_response)
                score = score_falsifiability(features)
                fals_comp = round(score.composite, 4)
                think_comp = round(think_score.composite, 4)

                t["falsifiability_composite"] = fals_comp
                t["thinking_composite"] = think_comp
                t["has_genuine_thinking"] = think_score.has_genuine_thinking
                t["selection_score"] = compute_selection_score(
                    falsifiability_composite=fals_comp,
                    thinking_composite=think_comp,
                    has_think_tag=think_score.features.has_think_tag,
                    thinking_weight=args.thinking_weight,
                )
                traces.append(t)

        logger.info(f"Loaded {trace_path}: {count} total")

    logger.info(f"Combined pool: {len(traces)} traces (after fabrication & verification filters)")

    # Filter by composite
    passing = [t for t in traces if t.get("falsifiability_composite", 0) >= args.min_composite]
    logger.info(f"Passing (composite >= {args.min_composite}): {len(passing)}")

    if not passing:
        logger.error("No passing traces!")
        return

    # Best-of-N per question ranked by combined selection_score
    by_question = defaultdict(list)
    for t in passing:
        by_question[t["question"]].append(t)

    selected = []
    for question, group in by_question.items():
        group.sort(
            key=lambda t: (
                t.get("selection_score", t.get("falsifiability_composite", 0)),
                t.get("thinking_composite", 0),
                t.get("falsifiability_composite", 0),
            ),
            reverse=True,
        )
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
    thinking_scores = sorted(t.get("thinking_composite", 0.0) for t in selected)
    selection_scores = sorted(t.get("selection_score", 0.0) for t in selected)
    manifest = {
        "total_in_pool": len(traces),
        "passing": len(passing),
        "selected": len(selected),
        "unique_questions": len(by_question),
        "top_k": args.top_k,
        "min_composite": args.min_composite,
        "min_thinking_quality": args.min_thinking_quality,
        "thinking_weight": args.thinking_weight,
        "require_verified": args.require_verified,
        "max_fabrication": args.max_fabrication,
        "keep_thinking": args.keep_thinking,
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "composite_median": composites[len(composites) // 2],
        "composite_min": composites[0],
        "composite_max": composites[-1],
        "thinking_median": thinking_scores[len(thinking_scores) // 2],
        "selection_median": selection_scores[len(selection_scores) // 2],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
