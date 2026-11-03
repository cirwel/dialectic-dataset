"""Upload verified datasets, LoRA adapters, or the Gradio Space to HuggingFace Hub.

Enforces verification gates on dataset uploads so unverified or fabrication-contaminated
splits cannot be accidentally pushed to the public dataset repository.

Usage:
    # Upload a LoRA adapter (with optional eval JSONs)
    python scripts/upload.py --repo hikewa/my-model --adapter-dir models/my-lora/final --eval-files results/rubric.json

    # Upload verified dataset artifacts & dataset card to hikewa/dialectic-reasoning-traces
    python scripts/upload.py --dataset-repo hikewa/dialectic-reasoning-traces --training-dir data/training_v9 --goldset data/goldset/goldset_thinking_v1.json

    # Sync space/ directory to the HuggingFace Gradio Space
    python scripts/upload.py --space-repo hikewa/dialectic-reasoning --space-dir space
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import create_repo, upload_file, upload_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def verify_training_manifest_gate(
    manifest_path: Path,
    allow_unverified: bool = False,
) -> dict:
    """Validate that a training manifest passed verification and zero-fabrication gates."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    if not allow_unverified:
        if not manifest.get("require_verified", False):
            raise ValueError(
                f"Refusing to upload dataset from {manifest_path}: "
                "manifest has require_verified=False (pass --allow-unverified to override)."
            )
        if manifest.get("max_fabrication", 1) != 0:
            raise ValueError(
                f"Refusing to upload dataset from {manifest_path}: "
                f"manifest has max_fabrication={manifest.get('max_fabrication')} (must be 0)."
            )
    return manifest


def upload_model_adapter(
    repo_id: str,
    adapter_dir: Path,
    eval_files: list[str],
    private: bool = False,
) -> None:
    adapter_dir = adapter_dir.resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    logger.info(f"Creating/checking model repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    logger.info(f"Uploading adapter from {adapter_dir}")
    upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")
    logger.info(f"Adapter uploaded: https://huggingface.co/{repo_id}")

    for fpath_str in eval_files:
        fpath = Path(fpath_str)
        if fpath.exists():
            logger.info(f"Uploading eval: {fpath.name}")
            upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"eval/{fpath.name}",
                repo_id=repo_id,
                repo_type="model",
            )


def upload_dataset_artifacts(
    dataset_repo: str,
    training_dir: Optional[Path] = None,
    training_prefix: str = "training_v9",
    goldset_path: Optional[Path] = None,
    pilot_traces_path: Optional[Path] = None,
    dataset_card_path: Optional[Path] = None,
    allow_unverified: bool = False,
) -> None:
    if training_dir is not None:
        training_dir = training_dir.resolve()
        manifest_path = training_dir / "manifest.json"
        verify_training_manifest_gate(manifest_path, allow_unverified=allow_unverified)
        logger.info(f"Uploading verified training split from {training_dir} -> {training_prefix}/")
        upload_folder(
            folder_path=str(training_dir),
            path_in_repo=training_prefix,
            repo_id=dataset_repo,
            repo_type="dataset",
        )

    if goldset_path is not None and goldset_path.exists():
        logger.info(f"Uploading goldset {goldset_path.name} -> goldset/{goldset_path.name}")
        upload_file(
            path_or_fileobj=str(goldset_path.resolve()),
            path_in_repo=f"goldset/{goldset_path.name}",
            repo_id=dataset_repo,
            repo_type="dataset",
        )

    if pilot_traces_path is not None and pilot_traces_path.exists():
        logger.info(f"Uploading council pilot traces {pilot_traces_path.name} -> v9_pilot/{pilot_traces_path.name}")
        upload_file(
            path_or_fileobj=str(pilot_traces_path.resolve()),
            path_in_repo=f"v9_pilot/{pilot_traces_path.name}",
            repo_id=dataset_repo,
            repo_type="dataset",
        )

    if dataset_card_path is not None and dataset_card_path.exists():
        logger.info(f"Uploading dataset card {dataset_card_path} -> README.md")
        upload_file(
            path_or_fileobj=str(dataset_card_path.resolve()),
            path_in_repo="README.md",
            repo_id=dataset_repo,
            repo_type="dataset",
        )


def upload_space(space_repo: str, space_dir: Path) -> None:
    space_dir = space_dir.resolve()
    if not space_dir.exists():
        raise FileNotFoundError(f"Space directory not found: {space_dir}")
    logger.info(f"Uploading Space from {space_dir} -> https://huggingface.co/spaces/{space_repo}")
    upload_folder(
        folder_path=str(space_dir),
        repo_id=space_repo,
        repo_type="space",
        ignore_patterns=["__pycache__/**", "*.pyc", ".DS_Store"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload adapter, dataset, or Space to HuggingFace")
    parser.add_argument("--repo", help="HF model repo ID (e.g. hikewa/my-model)")
    parser.add_argument("--adapter-dir", help="Local adapter directory")
    parser.add_argument("--eval-files", nargs="*", default=[], help="Eval JSON files to upload")
    parser.add_argument("--private", action="store_true", help="Create model repo as private")

    parser.add_argument("--dataset-repo", help="HF dataset repo ID (e.g. hikewa/dialectic-reasoning-traces)")
    parser.add_argument("--training-dir", help="Verified training split directory with manifest.json")
    parser.add_argument("--training-prefix", default="training_v9", help="Subfolder in dataset repo")
    parser.add_argument("--goldset", help="Path to goldset JSON to upload under goldset/")
    parser.add_argument("--pilot-traces", help="Path to verified council traces JSONL to upload under v9_pilot/")
    parser.add_argument("--dataset-card", help="Path to dataset_card.md to upload as README.md")
    parser.add_argument("--allow-unverified", action="store_true", help="Override require_verified manifest gate")

    parser.add_argument("--space-repo", help="HF Space repo ID (e.g. hikewa/dialectic-reasoning)")
    parser.add_argument("--space-dir", default="space", help="Local Space directory (default: space)")
    args = parser.parse_args()

    if args.repo and args.adapter_dir:
        upload_model_adapter(
            repo_id=args.repo,
            adapter_dir=Path(args.adapter_dir),
            eval_files=args.eval_files,
            private=args.private,
        )

    if args.dataset_repo:
        upload_dataset_artifacts(
            dataset_repo=args.dataset_repo,
            training_dir=Path(args.training_dir) if args.training_dir else None,
            training_prefix=args.training_prefix,
            goldset_path=Path(args.goldset) if args.goldset else None,
            pilot_traces_path=Path(args.pilot_traces) if args.pilot_traces else None,
            dataset_card_path=Path(args.dataset_card) if args.dataset_card else None,
            allow_unverified=args.allow_unverified,
        )

    if args.space_repo:
        upload_space(space_repo=args.space_repo, space_dir=Path(args.space_dir))


if __name__ == "__main__":
    main()
