"""Generate falsifiable dialectic traces using multiple model backends.

Instead of negative constraints (don't hedge, don't use formula), this script
uses positive constraints demanding specific, checkable, falsifiable claims.

Usage:
    python scripts/generate_falsifiable_traces.py
    python scripts/generate_falsifiable_traces.py --providers ollama_gemma4
    python scripts/generate_falsifiable_traces.py --dry-run
    python scripts/generate_falsifiable_traces.py --seeds 5
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = PROJECT_ROOT / "data" / "seeds_axiom.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "traces_v8.jsonl"

# Add src to path
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dialectic_dataset.falsifiability_extractor import (
    extract_falsifiability_features,
    score_falsifiability,
)

# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

PROVIDERS = {
    "ollama_gemma4": {
        "name": "Ollama / Gemma 4 12B",
        "base_url": "http://localhost:11434/v1/chat/completions",
        "model": "gemma4:latest",
        "env_key": "_OLLAMA",
    },
    "ollama_gemma4_26b": {
        "name": "Ollama / Gemma 4 26B MoE",
        "base_url": "http://localhost:11434/v1/chat/completions",
        "model": "gemma4:26b",
        "env_key": "_OLLAMA",
    },
    "cerebras_qwen": {
        "name": "Cerebras / Qwen3-235B",
        "base_url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "qwen-3-235b-a22b-instruct-2507",
        "env_key": "CEREBRAS_API_KEY",
    },
    "openrouter_nemotron": {
        "name": "OpenRouter / Nemotron 3 Super 120B",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "env_key": "OPENROUTER_API_KEY",
    },
    "openrouter_qwen36": {
        "name": "OpenRouter / Qwen3.6 Plus",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "qwen/qwen3.6-plus:free",
        "env_key": "OPENROUTER_API_KEY",
    },
    "openrouter_glm": {
        "name": "OpenRouter / GLM-4.5 Air",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "z-ai/glm-4.5-air:free",
        "env_key": "OPENROUTER_API_KEY",
    },
    "openrouter_llama": {
        "name": "OpenRouter / Llama 3.3 70B",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "env_key": "OPENROUTER_API_KEY",
    },
    "mistral_large": {
        "name": "Mistral / Large 3",
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
    },
}


# ---------------------------------------------------------------------------
# Falsifiability prompt
# ---------------------------------------------------------------------------

FALSIFIABLE_PROMPT = """You are reasoning through a genuine tension.

Your response MUST contain:
1. At least one claim with a specific mechanism, threshold, or condition (real numbers welcome but not required)
2. At least one conditional that names the SPECIFIC variable — not "context" or "circumstances" but a measurable thing (team size, revenue, timeline, survival rate, etc.)
3. At least one prediction that could be checked — something that would be wrong if your reasoning is wrong
4. A clear position — not "both sides have merit" but "do X when Y, do Z when W"

Your response MUST NOT contain:
- "It depends" without specifying on WHAT with a concrete variable
- "Both sides have merit", "balanced approach", "context matters", "it ultimately depends"
- Labeled thesis/antithesis/synthesis, numbered lists, or pros/cons tables
- Resolution phrases that could apply to any question
- Fabricated studies, surveys, or reports (e.g. "a 2022 Harvard study found...")
- Made-up statistics attributed to institutions or researchers

Be specific through MECHANISMS and CONDITIONS, not through citations or statistics. "Teams above 8 people fragment into sub-groups that optimize locally" is good. "A 2022 Harvard study found 40% improvement" is bad unless you are certain that study exists. Do NOT invent studies, surveys, or statistics. If you cite a number, it must be one you are confident is real.

Specificity comes from naming the concrete variables, thresholds, and failure modes — not from fabricating evidence.

Question: {question}

Respond directly. No JSON. No labels. Just reason through it with specific, checkable claims."""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_ollama_chat(model: str, prompt: str) -> str | None:
    """Fallback to Ollama's native /api/chat for thinking models (e.g. Gemma 4)
    that return empty content via the OpenAI-compatible endpoint.
    Uses num_predict=4096 so thinking models have budget for both
    reasoning and the actual response."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 4096},
    }
    try:
        resp = requests.post("http://localhost:11434/api/chat", json=body, timeout=600)
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        text = msg.get("content", "")
        return text.strip() if text else None
    except Exception as e:
        logger.warning(f"  Ollama chat API error: {e}")
        return None


def call_provider(provider_key: str, question: str) -> str | None:
    """Call a provider and return raw response text."""
    cfg = PROVIDERS[provider_key]
    is_ollama = cfg["env_key"] == "_OLLAMA"
    api_key = None if is_ollama else os.environ.get(cfg["env_key"])
    if not api_key and not is_ollama:
        logger.warning(f"  No API key for {cfg['env_key']}, skipping {cfg['name']}")
        return None

    prompt = FALSIFIABLE_PROMPT.format(question=question)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if "openrouter" in cfg["base_url"]:
        headers["HTTP-Referer"] = "https://github.com/hikewa/dialectic-dataset"
        headers["X-Title"] = "dialectic-dataset"

    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    for attempt in range(3):
        try:
            resp = requests.post(cfg["base_url"], headers=headers, json=body, timeout=180)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "")
                try:
                    wait = max(int(retry_after), 10)
                except (ValueError, TypeError):
                    wait = 60
                logger.warning(f"  Rate limited on {cfg['name']}, waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content", "")
            if not text and is_ollama:
                logger.info(f"  Empty via OpenAI endpoint, trying native chat API")
                text = _call_ollama_chat(cfg["model"], prompt)
                if text:
                    return strip_artifacts(text)
            if not text:
                logger.warning(f"  Empty content from {cfg['name']}")
                continue
            return strip_artifacts(text)

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout on {cfg['name']}, attempt {attempt + 1}")
            time.sleep(5)
        except requests.exceptions.HTTPError as e:
            logger.error(f"  HTTP error on {cfg['name']}: {e}")
            if attempt < 2:
                time.sleep(10)
        except (KeyError, IndexError) as e:
            logger.warning(f"  Parse error on {cfg['name']}: {e}")
            if attempt < 2:
                time.sleep(2)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            logger.warning(f"  Connection error on {cfg['name']}: {e}")
            if attempt < 2:
                time.sleep(10)

    return None


def strip_artifacts(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


# ---------------------------------------------------------------------------
# Falsifiability quality gate
# ---------------------------------------------------------------------------

def quick_falsifiability_check(response: str) -> tuple[bool, list[str], float]:
    """Fast falsifiability gate. Returns (passed, issues, composite)."""
    features = extract_falsifiability_features(response)
    score = score_falsifiability(features)

    issues = []

    if score.specificity_score < 0.15:
        issues.append(f"low_specificity: {score.specificity_score:.2f}")
    if score.commitment_score < 0.3:
        issues.append(f"low_commitment: {score.commitment_score:.2f}")
    if (features.vague_conditionals > features.specific_conditionals
            and features.total_conditionals > 0):
        issues.append(f"vague_conditionals: {features.vague_conditionals}/{features.total_conditionals}")
    if features.prediction_density < 0.10 and features.sentence_count >= 3:
        issues.append(f"low_pred_density: {features.prediction_density:.2f}")

    # Also check for banned phrases
    lower = response.lower()
    banned = [
        "it depends on context", "both sides have merit", "balanced approach",
        "context matters", "it ultimately depends", "optimal approach",
        "both perspectives have value", "nuanced",
    ]
    for phrase in banned:
        if phrase in lower:
            issues.append(f"banned: '{phrase}'")

    # Word count
    wc = len(response.split())
    if wc < 80:
        issues.append(f"too_short: {wc}")
    if wc > 700:
        issues.append(f"too_long: {wc}")

    # Template structure
    for label in ["thesis:", "antithesis:", "synthesis:", "pros:", "cons:"]:
        if label in lower:
            issues.append(f"template: '{label}'")

    passed = len(issues) == 0
    return passed, issues, score.composite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_available_providers(provider_keys: list[str]) -> list[str]:
    available = []
    for key in provider_keys:
        cfg = PROVIDERS[key]
        if cfg["env_key"] == "_OLLAMA":
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=3)
                if r.ok:
                    available.append(key)
                    logger.info(f"Provider available: {cfg['name']} (local)")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Ollama not running — {cfg['name']} unavailable")
        elif os.environ.get(cfg["env_key"]):
            available.append(key)
            logger.info(f"Provider available: {cfg['name']}")
        else:
            logger.warning(f"No key for {cfg['env_key']} — {cfg['name']} unavailable")
    return available


def main():
    parser = argparse.ArgumentParser(description="Generate falsifiable dialectic traces")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--providers", type=str, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--retries", type=int, default=2,
                        help="Max retries per seed if quality gate fails")
    parser.add_argument("--seeds-file", type=str, default=None,
                        help="Path to seeds JSON (default: seeds_axiom.json)")
    args = parser.parse_args()

    # Load seeds
    seeds_path = Path(args.seeds_file) if args.seeds_file else SEEDS_PATH
    with open(seeds_path) as f:
        seeds = json.load(f)
    if args.seeds:
        seeds = seeds[:args.seeds]
    logger.info(f"Loaded {len(seeds)} seed questions")

    # Determine providers
    provider_keys = list(PROVIDERS.keys())
    if args.providers:
        requested = [p.strip() for p in args.providers.split(",")]
        provider_keys = [k for k in requested if k in PROVIDERS]

    available = get_available_providers(provider_keys)
    if not available and not args.dry_run:
        logger.error("No providers available")
        return

    # Assign seeds to providers
    assignments = []
    for i, seed in enumerate(seeds):
        provider = available[i % len(available)] if available else "dry_run"
        assignments.append((seed, provider))

    if args.dry_run:
        print(f"\n{'ID':12s} {'Domain':16s} {'Provider':30s}")
        print("-" * 58)
        for seed, prov in assignments:
            prov_name = PROVIDERS.get(prov, {}).get("name", prov)
            print(f"{seed['id']:12s} {seed['domain']:16s} {prov_name:30s}")
        return

    # Resume support
    done_ids = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                t = json.loads(line)
                if t.get("quality_gate") == "pass":
                    done_ids.add(t["trace_id"])
        logger.info(f"Resuming: {len(done_ids)} passing traces already done")

    traces = []
    failures = []

    for i, (seed, provider_key) in enumerate(assignments):
        trace_id = f"fals_{seed['id']}_{provider_key}"

        if trace_id in done_ids:
            logger.info(f"[{i + 1}/{len(assignments)}] Skipping {trace_id} (done)")
            continue

        best_response = None
        best_composite = -1.0
        best_issues = []

        for retry in range(args.retries + 1):
            if retry > 0:
                logger.info(f"  Retry {retry}/{args.retries}")

            logger.info(f"[{i + 1}/{len(assignments)}] {seed['id']} via {PROVIDERS[provider_key]['name']}")
            response = call_provider(provider_key, seed["question"])

            if response is None:
                logger.error(f"  API FAILED: {seed['id']}")
                failures.append(seed["id"])
                break

            passed, issues, composite = quick_falsifiability_check(response)

            if composite > best_composite:
                best_response = response
                best_composite = composite
                best_issues = issues

            if passed:
                logger.info(f"  PASS — composite={composite:.3f}, {len(response.split())} words")
                break
            else:
                logger.warning(f"  FAIL (composite={composite:.3f}): {issues}")
                if retry < args.retries:
                    time.sleep(args.delay)

        if best_response is None:
            continue

        passed = len(best_issues) == 0
        trace = {
            "trace_id": trace_id,
            "question": seed["question"],
            "domain": seed["domain"],
            "source": seed.get("source", ""),
            "response": best_response,
            "generation_model": PROVIDERS[provider_key]["name"],
            "generated_at": int(time.time()),
            "prompt_version": "falsifiable_v1",
            "word_count": len(best_response.split()),
            "falsifiability_composite": round(best_composite, 4),
            "quality_gate": "pass" if passed else "fail",
            "quality_issues": best_issues,
        }

        traces.append(trace)
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(trace) + "\n")

        logger.info(f"  Saved: gate={'pass' if passed else 'fail'}, composite={best_composite:.3f}")

        if i < len(assignments) - 1:
            time.sleep(args.delay)

    # Summary
    passed_count = sum(1 for t in traces if t["quality_gate"] == "pass")
    failed_count = sum(1 for t in traces if t["quality_gate"] == "fail")

    print(f"\n{'=' * 60}")
    print("FALSIFIABLE TRACE GENERATION")
    print(f"{'=' * 60}")
    print(f"  Generated:     {len(traces)}")
    print(f"  Quality pass:  {passed_count}")
    print(f"  Quality fail:  {failed_count}")
    print(f"  API failures:  {len(failures)}")
    print(f"  Output:        {OUTPUT_PATH}")

    if traces:
        composites = [t["falsifiability_composite"] for t in traces]
        composites.sort()
        print(f"\n  Composite distribution:")
        print(f"    Min:    {composites[0]:.4f}")
        print(f"    Median: {composites[len(composites)//2]:.4f}")
        print(f"    Max:    {composites[-1]:.4f}")


if __name__ == "__main__":
    main()
