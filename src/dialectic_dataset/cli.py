"""CLI commands for the dialectic dataset pipeline.

Three commands:
- dialectic-generate: Generate traces via Claude API
- dialectic-prepare: Score, filter, format into training JSONL
- dialectic-eval: Evaluate scorer calibration against gold set
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# dialectic-generate
# ---------------------------------------------------------------------------

def cmd_generate() -> None:
    """Generate dialectic traces via Claude API."""
    parser = argparse.ArgumentParser(
        prog="dialectic-generate",
        description="Generate dialectic reasoning traces via Claude API",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--target", "-n", type=int, default=None,
        help="Target number of accepted traces",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=None,
        help="Categories to generate from",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for generated traces",
    )
    parser.add_argument(
        "--concurrent", type=int, default=None,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=None,
        help="Generation temperature",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    from dialectic_dataset.config import Config
    from dialectic_dataset.trace_generator import (
        GenerationConfig,
        generate_batch,
        save_traces,
    )

    config = Config.load(args.config)

    # Apply CLI overrides
    gen_config = GenerationConfig(
        model=args.model or config.model,
        temperature=args.temperature if args.temperature is not None else config.temperature,
        max_tokens=config.max_tokens,
        min_quality=config.min_quality,
        require_structural_pass=config.require_structural_pass,
        require_would_train=config.require_would_train,
        max_per_prompt=config.max_per_prompt,
        max_per_category=config.max_per_category,
        near_duplicate_threshold=config.near_duplicate_threshold,
        max_concurrent=args.concurrent or config.max_concurrent,
        output_dir=args.output or config.output_dir,
    )

    target = args.target or config.target_count
    categories = args.categories or config.categories

    print(f"Generating traces with {gen_config.model}")
    print(f"  Temperature: {gen_config.temperature}")
    print(f"  Min quality: {gen_config.min_quality}")
    print(f"  Max concurrent: {gen_config.max_concurrent}")
    if target:
        print(f"  Target count: {target}")
    if categories:
        print(f"  Categories: {', '.join(categories)}")
    print()

    result = asyncio.run(
        generate_batch(gen_config, categories=categories, target_count=target)
    )

    print(f"\nGeneration complete:")
    print(f"  Accepted: {len(result.accepted)}")
    print(f"  Rejected: {len(result.rejected)}")
    print(f"  Errors: {result.errors}")
    print(f"  Duplicates skipped: {result.duplicates_skipped}")
    print(f"  Acceptance rate: {result.acceptance_rate:.1%}")

    if result.accepted:
        output_dir = Path(args.output or config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "traces.jsonl"
        save_traces(result.accepted, str(output_path))
        print(f"\nSaved to {output_path}")

        # Also save rejected for analysis
        if result.rejected:
            rejected_path = output_dir / "rejected.jsonl"
            save_traces(result.rejected, str(rejected_path))
            print(f"Rejected traces saved to {rejected_path}")


# ---------------------------------------------------------------------------
# dialectic-prepare
# ---------------------------------------------------------------------------

def cmd_prepare() -> None:
    """Score, filter, and format traces into training JSONL."""
    parser = argparse.ArgumentParser(
        prog="dialectic-prepare",
        description="Prepare training dataset from generated traces",
    )
    parser.add_argument(
        "input", type=str,
        help="Path to generated traces JSONL file",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for training files",
    )
    parser.add_argument(
        "--explicit-ratio", type=float, default=None,
        help="Fraction of explicit format examples (default: 0.40)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    from dialectic_dataset.config import Config
    from dialectic_dataset.data_formatter import (
        prepare_dataset,
        print_dataset_stats,
    )
    from dialectic_dataset.trace_generator import load_traces

    config = Config.load(args.config)

    # Load traces
    traces = load_traces(args.input)
    print(f"Loaded {len(traces)} traces from {args.input}")

    # Filter accepted traces
    accepted = [
        t for t in traces
        if t.get("verdict") == "keep"
        and t.get("would_train_on_this", True)
        and t.get("structural_checks_passed", True)
    ]
    print(f"  Accepted: {len(accepted)} / {len(traces)}")

    if not accepted:
        print("No accepted traces to prepare. Exiting.")
        sys.exit(1)

    # Prepare dataset
    output_dir = args.output or config.training_dir
    explicit_ratio = args.explicit_ratio if args.explicit_ratio is not None else config.explicit_ratio
    seed = args.seed if args.seed is not None else config.seed

    stats = prepare_dataset(
        accepted,
        output_dir=output_dir,
        seed=seed,
        explicit_ratio=explicit_ratio,
    )
    print()
    print_dataset_stats(stats)


# ---------------------------------------------------------------------------
# dialectic-eval
# ---------------------------------------------------------------------------

def cmd_eval() -> None:
    """Evaluate scorer calibration against gold set."""
    parser = argparse.ArgumentParser(
        prog="dialectic-eval",
        description="Evaluate scorer calibration against gold set",
    )
    parser.add_argument(
        "--goldset", "-g", type=str, default=None,
        help="Path to gold set JSON file",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output with per-sample breakdown",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    from dialectic_dataset.config import Config
    from dialectic_dataset.goldset import load_goldset
    from dialectic_dataset.judge_calibration import (
        calibrate,
        print_calibration_report,
    )

    config = Config.load(args.config)
    goldset_path = args.goldset or config.goldset_path

    print(f"Loading gold set from {goldset_path}")
    goldset = load_goldset(goldset_path)
    print(f"Loaded {len(goldset.samples)} samples")
    print()

    result = calibrate(goldset)
    print_calibration_report(result)

    if args.verbose:
        print("\nPer-sample details:")
        header = f"{'ID':20s} {'Gold':8s} {'Scorer':8s} {'Composite':>9s} {'Train?':6s} {'Fired checks'}"
        print(header)
        print("-" * 80)
        for sample in goldset.samples:
            sid = sample.sample_id
            score = result.per_sample[sid]
            gold_v = sample.label.overall_verdict
            fired = ", ".join(score.structural_report.fired_names) or "-"
            match = "OK" if gold_v == score.verdict else "MISS"
            print(
                f"{sid:20s} {gold_v:8s} {score.verdict:8s} "
                f"{score.composite_quality:9.3f} "
                f"{'Y' if score.would_train_on_this else 'N':6s} "
                f"{fired}  [{match}]"
            )
