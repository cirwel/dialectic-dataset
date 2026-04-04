"""Train a LoRA adapter on dialectic reasoning data.

Usage:
    python scripts/train.py --base-model Qwen/Qwen3-8B --training-dir data/training --output-dir models/my-lora
    python scripts/train.py --base-model Qwen/Qwen3-4B --training-dir data/training --epochs 5 --lr 1e-4
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dialectic_dataset.trainer import TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter")
    parser.add_argument("--base-model", required=True, help="HF base model (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--training-dir", required=True, help="Directory with train.jsonl/val.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.base_model,
        training_dir=str(Path(args.training_dir).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )

    adapter_path = train(config)
    print(f"\nDone. Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
