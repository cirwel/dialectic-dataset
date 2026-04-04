"""Run the same question through ALL local models for side-by-side comparison."""

import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTION = "When does pragmatic compromise become unprincipled capitulation?"

SYSTEM_PROMPT = (
    "You reason carefully through problems by considering competing perspectives "
    "before reaching a conclusion. You identify genuine tensions, engage with "
    "the strongest form of each argument, and integrate insights rather than "
    "picking sides or hedging. Be specific through mechanisms and conditions, "
    "not through citations or statistics."
)

GEN_KWARGS = dict(
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.1,
)

MODELS = [
    ("0.5B v1 (Qwen2.5)", "dialectic-lora"),
    ("1.5B v1 (Qwen2.5)", "dialectic-lora-1.5b"),
    ("4B v1", "dialectic-lora-qwen3-4b"),
    ("8B v1", "dialectic-lora-qwen3-8b"),
]


def strip_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def run_model(name, adapter_dir):
    adapter_path = str(PROJECT_ROOT / "models" / adapter_dir / "final")

    with open(Path(adapter_path) / "adapter_config.json") as f:
        base_name = json.load(f).get("base_model_name_or_path", "unknown")

    print(f"\n{'='*70}")
    print(f"  {name}  (base: {base_name})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    elapsed = time.time() - t0

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    answer = strip_think(raw)

    print(f"  [{len(answer.split())} words, {elapsed:.1f}s]")
    print()
    print(answer)

    del model, base, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {"name": name, "base": base_name, "answer": answer, "words": len(answer.split()), "time": elapsed}


if __name__ == "__main__":
    print(f"Question: {QUESTION}\n")

    results = []
    for name, adapter_dir in MODELS:
        try:
            r = run_model(name, adapter_dir)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR: {name}: {e}\n")
            results.append({"name": name, "error": str(e)})

    # Save results
    out_path = PROJECT_ROOT / "data" / "eval" / "dogfood_all.json"
    with open(out_path, "w") as f:
        json.dump({"question": QUESTION, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")
