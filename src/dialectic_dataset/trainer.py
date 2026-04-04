"""Fine-tune a language model on dialectic reasoning traces.

Uses HuggingFace transformers + PEFT (LoRA) + TRL SFTTrainer.
Supports Apple Silicon (MPS) and CUDA backends.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Configuration for fine-tuning."""

    # Model
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 2048

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Paths
    training_dir: str = "data/training"
    output_dir: str = "models/dialectic-lora"

    # Device
    device: Optional[str] = None  # auto-detect if None

    def get_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(training_dir: str) -> Dict[str, Dataset]:
    """Load training JSONL files into HuggingFace datasets."""
    data_files = {}
    train_path = Path(training_dir)

    for split in ["train", "val", "test"]:
        path = train_path / f"{split}.jsonl"
        if path.exists():
            data_files[split] = str(path)

    if not data_files:
        raise FileNotFoundError(f"No JSONL files found in {training_dir}")

    datasets = load_dataset("json", data_files=data_files)

    # Rename "val" to "validation" if present (HF convention)
    if "val" in datasets:
        datasets["validation"] = datasets.pop("val")

    return datasets


def format_chat_for_training(example: dict) -> dict:
    """Format a training example for SFTTrainer.

    SFTTrainer expects either a 'text' field or 'messages' field.
    Our JSONL already has 'messages' in the right format.
    """
    return {"messages": example["messages"]}


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer(
    config: TrainConfig,
) -> tuple:
    """Load base model with LoRA adapter.

    Returns (model, tokenizer).
    """
    device = config.get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Loading base model: {config.base_model}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
    }

    if device == "cuda":
        # Use 4-bit quantization on CUDA for memory efficiency
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["dtype"] = torch.bfloat16
    elif device == "mps":
        model_kwargs["dtype"] = torch.float32  # MPS needs float32 for stability
    else:
        model_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        **model_kwargs,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({trainable/total:.2%})"
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: TrainConfig) -> str:
    """Run fine-tuning and return the output directory.

    Returns path to the saved LoRA adapter.
    """
    device = config.get_device()

    # Load data
    logger.info(f"Loading training data from {config.training_dir}")
    datasets = load_training_data(config.training_dir)

    train_ds = datasets.get("train")
    eval_ds = datasets.get("validation")

    if train_ds is None:
        raise ValueError("No training data found")

    logger.info(f"Training examples: {len(train_ds)}")
    if eval_ds:
        logger.info(f"Validation examples: {len(eval_ds)}")

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)

    # Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        max_length=config.max_seq_length,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        report_to="none",
        fp16=False,
        bf16=(device == "cuda"),
        dataloader_pin_memory=False,  # Required for MPS
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    result = trainer.train()

    # Save LoRA adapter
    adapter_path = output_dir / "final"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"Saved LoRA adapter to {adapter_path}")

    # Log metrics
    metrics = result.metrics
    logger.info(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"Training runtime: {metrics.get('train_runtime', 'N/A'):.1f}s")

    return str(adapter_path)


# ---------------------------------------------------------------------------
# Inference with fine-tuned model
# ---------------------------------------------------------------------------

def load_finetuned(adapter_path: str, device: Optional[str] = None) -> tuple:
    """Load a fine-tuned model (base + LoRA adapter).

    Returns (model, tokenizer).
    """
    from peft import PeftModel

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # Load adapter config to find base model
    import json as _json
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = _json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path", "")

    # Load base model
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Generate a response from the fine-tuned model."""
    if system_prompt is None:
        system_prompt = (
            "You reason carefully through problems by considering competing "
            "perspectives before reaching a conclusion. You identify genuine "
            "tensions, engage with the strongest form of each argument, and "
            "integrate insights rather than picking sides or hedging."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()
