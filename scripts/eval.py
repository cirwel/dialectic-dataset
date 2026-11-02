"""Evaluate a fine-tuned adapter with Mistral-strict rubric + fabrication checks.

Usage:
    MISTRAL_API_KEY=... python scripts/eval.py --adapter-path models/dialectic-lora-qwen3-8b/final
    MISTRAL_API_KEY=... python scripts/eval.py --adapter-path models/dialectic-lora-qwen3-8b/final --output-dir results/
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dialectic_dataset.evaluator import TRADEOFF_EVAL_PROMPTS, RESISTANCE_EVAL_PROMPTS
from dialectic_dataset.falsifiability_extractor import (
    extract_falsifiability_features,
    score_falsifiability,
    _count_fabricated_citations,
    _count_ungrounded_stats,
)
from dialectic_dataset.thinking_quality import (
    extract_think_block,
    score_thinking_quality,
)
from dialectic_dataset.verification_pipeline import (
    DEFAULT_COUNCIL_HOST_IDS,
    UnitaresConsultBackend,
)

EVAL_PROMPTS = TRADEOFF_EVAL_PROMPTS + RESISTANCE_EVAL_PROMPTS

SYSTEM_PROMPT = (
    "You reason carefully through problems by considering competing perspectives "
    "before reaching a conclusion. You identify genuine tensions, engage with "
    "the strongest form of each argument, and integrate insights rather than "
    "picking sides or hedging."
)

GEN_KWARGS = dict(
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.1,
)

JUDGE_STRICT = (
    "You are a strict evaluator of dialectic reasoning quality. "
    "Be demanding — most responses should score 0 or 1, not 2. "
    "A score of 2 means genuinely exceptional, not merely adequate.\n\n"
    "Question: {question}\n\n"
    "Response: {response}\n\n"
    "Score each dimension 0/1/2 using these strict criteria:\n\n"
    "1. crux_clarity: 0=no crux identified, 1=names a tension but vaguely, "
    "2=pinpoints the SPECIFIC variable that determines the answer\n"
    "2. conditional_commitment: 0=no if/then, 1=one weak conditional, "
    "2=multiple concrete if/then commitments with specific conditions\n"
    "3. tradeoff_specificity: 0=generic (\"there are costs and benefits\"), "
    "1=one side has concrete details, 2=BOTH sides have specific numbers/examples/mechanisms\n"
    "4. resolution_depth: 0=restates the tension without resolving, "
    "1=picks a side with justification, 2=finds a higher-order principle that genuinely integrates both\n"
    "5. generic_hedge: 0=takes clear positions throughout, "
    "1=some hedging phrases (\"it depends\", \"both have merit\", \"context matters\"), "
    "2=pervasive hedging that avoids commitment (PENALTY)\n"
    "6. actionability: 0=abstract discussion only, 1=general recommendations, "
    "2=concrete next steps someone could actually follow\n"
    "7. one_sided_collapse: 0=genuinely engages both sides, "
    "1=acknowledges but dismisses one side, 2=strawmans or ignores one side (PENALTY)\n\n"
    "IMPORTANT: Be strict. If the response uses phrases like \"optimal approach\", "
    "\"nuanced\", \"case-by-case\", \"both perspectives have value\", or "
    "\"it ultimately depends\" — that is hedging (generic_hedge >= 1). "
    "If the response follows a formulaic thesis/antithesis/synthesis structure "
    "without genuine integration, resolution_depth should be 0 or 1.\n\n"
    "Return ONLY valid JSON with these keys: crux_clarity, conditional_commitment, "
    "tradeoff_specificity, resolution_depth, generic_hedge, actionability, one_sided_collapse\n"
    "No markdown fences."
)


def load_model(adapter_path):
    logger.info(f"Loading adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(Path(adapter_path) / "adapter_config.json") as f:
        base_name = json.load(f).get("base_model_name_or_path", "unknown")

    dtype = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device} (base: {base_name})")
    return model, tokenizer


def generate(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def strip_think(text):
    _, visible = extract_think_block(text)
    return visible


def parse_json(text):
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                return json.loads(p)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


def compute_aggregate(rubric):
    pos = sum(rubric.get(k, 0) for k in [
        "crux_clarity", "conditional_commitment",
        "tradeoff_specificity", "resolution_depth", "actionability",
    ])
    pen = rubric.get("generic_hedge", 0) + rubric.get("one_sided_collapse", 0)
    total = pos - pen
    verdict = "strong" if total >= 7 else "mixed" if total >= 4 else "weak" if total >= 0 else "bad"
    return {"positive_sum": pos, "penalty_sum": pen, "total": total, "verdict": verdict}


def score_mistral(api_key, question, response):
    prompt = JUDGE_STRICT.format(question=question, response=response)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers, json=body, timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return parse_json(text)
        except Exception as e:
            logger.warning(f"    Mistral retry {attempt+1}: {e}")
            time.sleep(3)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned adapter")
    parser.add_argument("--adapter-path", required=True, help="Path to adapter directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: data/eval/)")
    parser.add_argument("--label", default=None, help="Label for this eval run")
    parser.add_argument(
        "--judge",
        choices=("mistral", "unitares_consult", "unitares_council", "none"),
        default="mistral",
        help="Judge backend for Stage 6 evaluation (default: mistral)",
    )
    parser.add_argument(
        "--unitares-url",
        default="http://127.0.0.1:8767",
        help="UNITARES HTTP base URL when --judge is unitares_consult or unitares_council",
    )
    parser.add_argument(
        "--host-id",
        default="claude:host-adapter",
        help="Host adapter when --judge=unitares_consult",
    )
    parser.add_argument(
        "--host-ids",
        nargs="+",
        default=list(DEFAULT_COUNCIL_HOST_IDS),
        help="Council host adapters when --judge=unitares_council",
    )
    parser.add_argument(
        "--client-session-id",
        default=None,
        help="UNITARES bound client_session_id (or set UNITARES_CLIENT_SESSION_ID env var)",
    )
    args = parser.parse_args()

    adapter_path = str(Path(args.adapter_path).resolve())
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or Path(adapter_path).parent.name
    api_key = os.environ.get("MISTRAL_API_KEY", "")

    # Step 1: Generate responses
    logger.info(f"=== Generating responses ({label}) ===")
    model, tokenizer = load_model(adapter_path)
    responses = []
    for prompt in EVAL_PROMPTS:
        logger.info(f"  {prompt['id']}")
        resp = generate(model, tokenizer, prompt["question"])
        answer = strip_think(resp)
        think_score = score_thinking_quality(resp)

        features = extract_falsifiability_features(answer)
        fscore = score_falsifiability(features)
        fab_count = _count_fabricated_citations(resp)
        ung_count = _count_ungrounded_stats(answer)

        responses.append({
            "prompt_id": prompt["id"],
            "question": prompt["question"],
            "category": prompt["category"],
            "response": resp,
            "answer_only": answer,
            "word_count": len(answer.split()),
            "thinking_quality": {
                "composite": think_score.composite,
                "has_genuine_thinking": think_score.has_genuine_thinking,
                "components": think_score.components,
            },
            "falsifiability": {
                "composite": round(fscore.composite, 4),
                "grounding_penalty": round(fscore.components.get("grounding_penalty", 0), 4),
            },
            "fabrication": {
                "fabricated_citations": fab_count,
                "ungrounded_stats": ung_count,
            },
        })
        logger.info(
            f"    {len(answer.split())} words, think={think_score.composite:.2f}, fab_cite={fab_count}"
        )

    resp_path = output_dir / f"responses_{label}.json"
    with open(resp_path, "w") as f:
        json.dump({"model": label, "responses": responses}, f, indent=2)

    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Step 2: Judge scoring (Mistral or UNITARES consult/council)
    if args.judge in ("unitares_consult", "unitares_council"):
        logger.info(f"=== UNITARES {args.judge} Stage 6 verification & adversarial challenge ===")
        backend = (
            UnitaresConsultBackend(
                base_url=args.unitares_url,
                privacy="cloud_allowed",
                host_id=args.host_id,
                client_session_id=args.client_session_id,
            )
            if args.judge == "unitares_consult"
            else UnitaresConsultBackend(
                base_url=args.unitares_url,
                privacy="cloud_allowed",
                host_ids=args.host_ids,
                client_session_id=args.client_session_id,
            )
        )
        council_scored = []
        for r in responses:
            logger.info(f"  Evaluating via {args.judge}: {r['prompt_id']}")
            report = backend.evaluate_trace(r["question"], r["response"])
            council_scored.append({
                "prompt_id": r["prompt_id"],
                "category": r["category"],
                "verification_pass": report.verification_pass,
                "adversarial_survives": report.adversarial_survives,
                "overall_pass": report.overall_pass,
                "any_fabrication_detected": report.any_fabrication_detected,
                "report": report.to_dict(),
            })
        rubric_path = output_dir / f"council_eval_{label}.json"
        with open(rubric_path, "w") as f:
            json.dump({"model": label, "judge": args.judge, "scores": council_scored}, f, indent=2)
        pass_count = sum(1 for s in council_scored if s["overall_pass"])
        print(f"\n{'='*60}\nEVAL ({args.judge}): {label}\n{'='*60}")
        print(f"  Overall pass: {pass_count}/{len(council_scored)}")
        print(f"  Results: {rubric_path}")
        return

    if args.judge == "none" or not api_key:
        logger.warning("No MISTRAL_API_KEY (or --judge=none) — skipping Mistral judge scoring")
        return

    logger.info("=== Mistral-strict rubric scoring ===")
    scored = []
    for r in responses:
        logger.info(f"  Scoring: {r['prompt_id']}")
        rubric = score_mistral(api_key, r["question"], r["answer_only"])
        if rubric is None:
            continue
        scored.append({
            "prompt_id": r["prompt_id"],
            "category": r["category"],
            "rubric": rubric,
            "aggregate": compute_aggregate(rubric),
        })
        time.sleep(1)

    rubric_path = output_dir / f"rubric_{label}.json"
    with open(rubric_path, "w") as f:
        json.dump({"model": label, "judge": "mistral-large-latest", "scores": scored}, f, indent=2)

    # Summary
    avg = sum(s["aggregate"]["total"] for s in scored) / len(scored) if scored else 0
    verdicts = {}
    for s in scored:
        v = s["aggregate"]["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
    total_fab = sum(r["fabrication"]["fabricated_citations"] for r in responses)
    fab_any = sum(1 for r in responses if r["fabrication"]["fabricated_citations"] > 0)

    print(f"\n{'='*60}")
    print(f"EVAL: {label}")
    print(f"{'='*60}")
    print(f"  Avg score: {avg:.1f}")
    print(f"  Verdicts: {verdicts}")
    print(f"  Fabrication: {total_fab} citations in {fab_any}/{len(responses)} responses")
    print(f"  Results: {rubric_path}")


if __name__ == "__main__":
    main()
