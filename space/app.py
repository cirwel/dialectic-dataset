"""Dialectic Reasoning Chatbot — Gradio Space with ZeroGPU."""

import gc

import spaces
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODELS = {
    "Qwen3-8B (recommended)": {
        "base": "Qwen/Qwen3-8B",
        "adapter": "hikewa/dialectic-qwen3-8b-lora",
    },
    "Qwen2.5-1.5B": {
        "base": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter": "hikewa/dialectic-qwen2.5-1.5b-lora",
    },
}

DEFAULT_MODEL = "Qwen3-8B (recommended)"

SYSTEM_PROMPT = (
    "You reason carefully through problems by considering competing "
    "perspectives before reaching a conclusion. You identify genuine "
    "tensions, engage with the strongest form of each argument, and "
    "integrate insights rather than picking sides or hedging."
)

loaded = {"name": None, "model": None, "tokenizer": None}


def load_model(model_name):
    global loaded
    if loaded["name"] == model_name:
        return loaded["model"], loaded["tokenizer"]

    # Free previous model
    if loaded["model"] is not None:
        del loaded["model"]
        loaded["model"] = None
        gc.collect()
        torch.cuda.empty_cache()

    cfg = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["adapter"], trust_remote_code=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.float16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, cfg["adapter"])
    model = model.to("cuda")
    model.eval()

    loaded["name"] = model_name
    loaded["model"] = model
    loaded["tokenizer"] = tokenizer
    return model, tokenizer


@spaces.GPU
def respond(message, history, model_name):
    model, tokenizer = load_model(model_name)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            messages.append({"role": "user", "content": msg[0]})
            messages.append({"role": "assistant", "content": msg[1]})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Dropdown(
            choices=list(MODELS.keys()),
            value=DEFAULT_MODEL,
            label="Model",
        ),
    ],
    title="Dialectic Reasoning",
    description=(
        "Fine-tuned on 510 dialectic reasoning traces. "
        "Ask a question involving competing perspectives."
    ),
    examples=[
        ["Should AI systems be transparent about their reasoning, even when transparency reduces performance?"],
        ["Is it better to optimize for individual freedom or collective wellbeing?"],
        ["When does pragmatic compromise become unprincipled capitulation?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
