"""Upload a LoRA adapter and optional eval results to HuggingFace Hub.

Usage:
    python scripts/upload.py --repo hikewa/my-model --adapter-dir models/my-lora/final
    python scripts/upload.py --repo hikewa/my-model --adapter-dir models/my-lora/final --eval-files results/rubric.json
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import create_repo, upload_folder, upload_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload adapter to HuggingFace")
    parser.add_argument("--repo", required=True, help="HF repo ID (e.g. hikewa/my-model)")
    parser.add_argument("--adapter-dir", required=True, help="Local adapter directory")
    parser.add_argument("--eval-files", nargs="*", default=[], help="Eval JSON files to upload")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    logger.info(f"Creating repo: {args.repo}")
    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)

    logger.info(f"Uploading adapter from {adapter_dir}")
    upload_folder(folder_path=str(adapter_dir), repo_id=args.repo, repo_type="model")
    logger.info(f"Adapter uploaded: https://huggingface.co/{args.repo}")

    for fpath in args.eval_files:
        fpath = Path(fpath)
        if fpath.exists():
            logger.info(f"Uploading eval: {fpath.name}")
            upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"eval/{fpath.name}",
                repo_id=args.repo,
                repo_type="model",
            )

    print(f"\nDone: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
