"""Generate axiom-constrained dialectic traces using multiple model backends.

Instead of prescribing thesis/antithesis/synthesis structure, this script
constrains generation with the Synthetic Life Axioms — telling models what
NOT to do (hedge, perform formula, resolve prematurely) and letting them
find their own structure.

Output is response-only: no decomposition into phases.

Usage:
    python scripts/generate_axiom_traces.py
    python scripts/generate_axiom_traces.py --providers ollama_gemma4,ollama_gemma4_26b
    python scripts/generate_axiom_traces.py --dry-run
    python scripts/generate_axiom_traces.py --resume
    python scripts/generate_axiom_traces.py --seeds 5  # first N seeds only
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
OUTPUT_PATH = PROJECT_ROOT / "data" / "traces_axiom.jsonl"

# ---------------------------------------------------------------------------
# Provider configs (same as generate_diverse_traces.py)
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
# Few-shot examples from UNITARES papers
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """Here are examples of the reasoning quality expected — not the format, but the depth and honesty of engagement:

Example 1 (on AI identity):
"The identity problem in AI agents cannot be solved by memory alone — it requires embodiment. Current approaches treat identity as an external add-on: memory database, ID string. This is fundamentally unstable. A purely chat-based agent's life begins and ends with the chat session. By giving agents something analogous to a body with which to experience consequences and maintain internal state, we enable continuity that is lived rather than simulated. Identity emerges from ongoing sensorimotor interaction and self-maintenance, not from having perfect episodic memory."

Example 2 (on ephemeral agents and governance):
"Self-governance for ephemeral agents isn't impossible, it's different. The knowledge graph isn't a substitute self — it's a coordination substrate. Coherence metrics measure trajectory consistency, not personal continuity. The question isn't whether an ephemeral agent can govern itself (it can't, in the traditional sense). The question is whether a sequence of ephemeral agents, sharing a knowledge graph and coherence metrics, can collectively maintain a governance trajectory. The answer appears to be yes, provided the substrate captures the right invariants."

Example 3 (on credential vs behavioral identity):
"Identity today is conferred, not earned. An agent has an identity because we gave it one, not because it developed characteristics that make it recognizably itself. This creates practical problems: if the UUID is lost, identity is lost. If we copy the agent, which copy is real? A compromised agent with the right credentials is indistinguishable from the original. The alternative: define identity by the patterns that persist in behavior over time — attractor basins, recovery dynamics, preference profiles. Two copies with the same UUID diverge in trajectory and become different agents. Identity becomes something you do, not something you're called."
"""

# ---------------------------------------------------------------------------
# Axiom-constrained prompt
# ---------------------------------------------------------------------------

AXIOM_PROMPT = """You are reasoning through a genuine tension. Your constraints:

NEGATIVE (violations disqualify your response):
- Do not resolve the tension before honestly facing it
- Do not hedge with "it depends", "both sides have merit", "context matters", "optimal approach", "nuanced", "balanced approach", "case-by-case", "various perspectives"
- Do not produce formulaic structure (labeled thesis/antithesis/synthesis, numbered lists, pros/cons tables)
- Do not let the response seem wiser than the reasoning supports
- Do not counterfeit depth through fluency alone

POSITIVE (your response should):
- Identify what single variable or condition determines the answer
- Make concrete conditional commitments (when X, do Y)
- Reach a position — not describe the landscape
- Let the tension develop before resolving it
- Ground claims in specific mechanisms, examples, or numbers

{few_shot}

Question: {question}

Respond directly. No JSON. No labels. No preamble. Just reason through it."""


# ---------------------------------------------------------------------------
# API call (adapted from generate_diverse_traces.py)
# ---------------------------------------------------------------------------

def call_provider(provider_key: str, question: str, use_few_shot: bool = True) -> str | None:
    """Call a provider and return raw response text, or None on failure."""
    cfg = PROVIDERS[provider_key]
    is_ollama = cfg["env_key"] == "_OLLAMA"
    api_key = None if is_ollama else os.environ.get(cfg["env_key"])
    if not api_key and not is_ollama:
        logger.warning(f"  No API key for {cfg['env_key']}, skipping {cfg['name']}")
        return None

    few_shot = FEW_SHOT_EXAMPLES if use_few_shot else ""
    prompt = AXIOM_PROMPT.format(question=question, few_shot=few_shot)

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

    return None


def strip_artifacts(text: str) -> str:
    """Strip think tags and other artifacts from response."""
    text = text.strip()
    # Strip <think>...</think> blocks
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


# ---------------------------------------------------------------------------
# Quick quality gate (fast heuristic, before expensive LLM scoring)
# ---------------------------------------------------------------------------

BANNED_PHRASES = [
    "it depends on context", "case-by-case basis", "both sides have merit",
    "optimal approach", "various perspectives", "nuanced", "balanced approach",
    "both perspectives have value", "context matters", "it ultimately depends",
    "reasonable people disagree", "there are valid points on both sides",
    "the truth lies somewhere in the middle",
]


def quick_quality_check(response: str) -> tuple[bool, list[str]]:
    """Fast heuristic check. Returns (passed, list of issues)."""
    issues = []
    lower = response.lower()

    # Banned phrases
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            issues.append(f"banned_phrase: '{phrase}'")

    # Too short
    word_count = len(response.split())
    if word_count < 80:
        issues.append(f"too_short: {word_count} words")

    # Too long
    if word_count > 700:
        issues.append(f"too_long: {word_count} words")

    # Formulaic structure: numbered lists
    lines = response.strip().split("\n")
    numbered_lines = sum(1 for l in lines if l.strip()[:2] in ["1.", "2.", "3.", "4.", "5."])
    if numbered_lines >= 3:
        issues.append(f"numbered_list: {numbered_lines} items")

    # Formulaic: explicit thesis/antithesis/synthesis labels
    for label in ["thesis:", "antithesis:", "synthesis:", "**thesis**", "**antithesis**", "**synthesis**",
                   "## thesis", "## antithesis", "## synthesis", "pros:", "cons:"]:
        if label in lower:
            issues.append(f"template_label: '{label}'")

    # Bullet-heavy
    bullet_lines = sum(1 for l in lines if l.strip().startswith("- ") or l.strip().startswith("* "))
    if bullet_lines >= 4:
        issues.append(f"bullet_heavy: {bullet_lines} bullets")

    passed = len(issues) == 0
    return passed, issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_available_providers(provider_keys: list[str]) -> list[str]:
    """Check which providers are available."""
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
    parser = argparse.ArgumentParser(description="Generate axiom-constrained dialectic traces")
    parser.add_argument("--dry-run", action="store_true", help="Print assignments only")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated trace IDs")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between API calls")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated provider keys")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Limit to first N seeds (for test batches)")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Omit few-shot examples from prompt")
    args = parser.parse_args()

    # Load seeds
    with open(SEEDS_PATH) as f:
        seeds = json.load(f)
    if args.seeds:
        seeds = seeds[:args.seeds]
    logger.info(f"Loaded {len(seeds)} seed questions")

    # Determine providers
    provider_keys = list(PROVIDERS.keys())
    if args.providers:
        requested = [p.strip() for p in args.providers.split(",")]
        provider_keys = [k for k in requested if k in PROVIDERS]
        if not provider_keys:
            logger.error(f"No valid providers in: {args.providers}")
            return

    available = get_available_providers(provider_keys)
    if not available and not args.dry_run:
        logger.error("No providers available")
        return

    # Assign seeds to providers (rotate)
    assignments = []
    for i, seed in enumerate(seeds):
        provider = available[i % len(available)] if available else "dry_run"
        assignments.append((seed, provider))

    # Domain stats
    domain_counts = {}
    for seed, _ in assignments:
        d = seed["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    logger.info(f"Domain distribution: {domain_counts}")
    logger.info(f"Total: {len(assignments)} traces, {len(available)} providers")

    if args.dry_run:
        print(f"\n{'ID':12s} {'Domain':16s} {'Provider':30s}")
        print("-" * 58)
        for seed, prov in assignments:
            prov_name = PROVIDERS.get(prov, {}).get("name", prov)
            print(f"{seed['id']:12s} {seed['domain']:16s} {prov_name:30s}")
        return

    # Load existing trace IDs if resuming
    done_ids = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                t = json.loads(line)
                done_ids.add(t["trace_id"])
        logger.info(f"Resuming: {len(done_ids)} traces already done")

    # Generate
    traces = []
    failures = []
    quality_fails = []

    for i, (seed, provider_key) in enumerate(assignments):
        trace_id = f"axiom_{seed['id']}_{provider_key}"

        if trace_id in done_ids:
            logger.info(f"[{i + 1}/{len(assignments)}] Skipping {trace_id} (done)")
            continue

        logger.info(f"[{i + 1}/{len(assignments)}] {seed['id']} via {PROVIDERS[provider_key]['name']}")

        response = call_provider(provider_key, seed["question"], use_few_shot=not args.no_few_shot)

        if response is None:
            logger.error(f"  FAILED: {seed['id']}")
            failures.append(seed["id"])
            continue

        # Quick quality gate
        passed, issues = quick_quality_check(response)
        if not passed:
            logger.warning(f"  Quality issues: {issues}")
            quality_fails.append({"id": seed["id"], "issues": issues})
            # Still save it but mark as failed
            trace = {
                "trace_id": trace_id,
                "question": seed["question"],
                "domain": seed["domain"],
                "source": seed.get("source", ""),
                "response": response,
                "generation_model": PROVIDERS[provider_key]["name"],
                "generated_at": int(time.time()),
                "prompt_version": "axiom_v1",
                "word_count": len(response.split()),
                "quality_gate": "fail",
                "quality_issues": issues,
            }
        else:
            trace = {
                "trace_id": trace_id,
                "question": seed["question"],
                "domain": seed["domain"],
                "source": seed.get("source", ""),
                "response": response,
                "generation_model": PROVIDERS[provider_key]["name"],
                "generated_at": int(time.time()),
                "prompt_version": "axiom_v1",
                "word_count": len(response.split()),
                "quality_gate": "pass",
                "quality_issues": [],
            }

        traces.append(trace)

        # Append immediately (crash-safe)
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(trace) + "\n")

        logger.info(f"  OK — {trace['word_count']} words, gate={trace['quality_gate']}")

        if i < len(assignments) - 1:
            time.sleep(args.delay)

    # Summary
    passed = sum(1 for t in traces if t["quality_gate"] == "pass")
    failed_quality = sum(1 for t in traces if t["quality_gate"] == "fail")

    print(f"\n{'=' * 60}")
    print("Generation complete")
    print(f"{'=' * 60}")
    print(f"  Generated:     {len(traces)}")
    print(f"  Quality pass:  {passed}")
    print(f"  Quality fail:  {failed_quality}")
    print(f"  API failures:  {len(failures)}")
    print(f"  Output:        {OUTPUT_PATH}")

    if quality_fails:
        print(f"\n  Quality failures:")
        for qf in quality_fails:
            print(f"    {qf['id']}: {qf['issues']}")

    # Domain breakdown
    domain_ok = {}
    for t in traces:
        if t["quality_gate"] == "pass":
            d = t["domain"]
            domain_ok[d] = domain_ok.get(d, 0) + 1
    print(f"\n  Passing traces by domain:")
    for d, c in sorted(domain_ok.items()):
        print(f"    {d:20s} {c}")


if __name__ == "__main__":
    main()
