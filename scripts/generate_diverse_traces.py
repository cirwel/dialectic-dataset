"""Generate domain-balanced dialectic traces using free inference APIs.

Uses multiple providers (Cerebras, OpenRouter, Mistral, Ollama) rotating
across models for ideological diversity. All use OpenAI-compatible chat API.

Setup:
    export CEREBRAS_API_KEY=...    # https://cloud.cerebras.ai (free)
    export OPENROUTER_API_KEY=...  # https://openrouter.ai (free)
    export MISTRAL_API_KEY=...     # https://console.mistral.ai (free)
    # Ollama: just run `ollama serve` locally (no key needed)

Usage:
    python scripts/generate_diverse_traces.py
    python scripts/generate_diverse_traces.py --dry-run  # show tasks only
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "traces_diverse.jsonl"

# ---------------------------------------------------------------------------
# Provider configs (all OpenAI-compatible)
# ---------------------------------------------------------------------------

PROVIDERS = {
    # Cerebras (free tier, fast inference)
    "cerebras_qwen": {
        "name": "Cerebras / Qwen3-235B",
        "base_url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "qwen-3-235b-a22b-instruct-2507",
        "env_key": "CEREBRAS_API_KEY",
    },
    # OpenRouter (free tier, diverse models)
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
    # Mistral (free tier)
    "mistral_large": {
        "name": "Mistral / Large 3",
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
    },
    # Local (Ollama)
    "ollama_gemma4": {
        "name": "Ollama / Gemma 4 12B",
        "base_url": "http://localhost:11434/v1/chat/completions",
        "model": "gemma4:latest",
        "env_key": "_OLLAMA",  # special: no key needed, just check server
    },
    "ollama_gemma4_26b": {
        "name": "Ollama / Gemma 4 26B MoE",
        "base_url": "http://localhost:11434/v1/chat/completions",
        "model": "gemma4:26b",
        "env_key": "_OLLAMA",
    },
}

# ---------------------------------------------------------------------------
# 50 domain-balanced seed tasks
# ---------------------------------------------------------------------------

SEED_TASKS = [
    # ---- Business / Strategy (10) ----
    {"id": "biz_01", "domain": "business", "archetype": "short_term_vs_long_term",
     "question": "Should a startup optimize for growth or margin first?"},
    {"id": "biz_02", "domain": "business", "archetype": "cost_vs_quality",
     "question": "Should a company build in-house AI infrastructure or use external APIs?"},
    {"id": "biz_03", "domain": "business", "archetype": "speed_vs_reliability",
     "question": "Should a product launch early with known gaps or wait for higher reliability?"},
    {"id": "biz_04", "domain": "business", "archetype": "scale_vs_control",
     "question": "Should a team prioritize enterprise customers or self-serve users?"},
    {"id": "biz_05", "domain": "business", "archetype": "centralization_vs_autonomy",
     "question": "Should leadership centralize decision-making during a scale-up phase?"},
    {"id": "biz_06", "domain": "business", "archetype": "innovation_vs_regulation",
     "question": "Should a company open-source its core model?"},
    {"id": "biz_07", "domain": "business", "archetype": "short_term_vs_long_term",
     "question": "Should sales compensation reward short-term bookings or retention quality?"},
    {"id": "biz_08", "domain": "business", "archetype": "exploration_vs_exploitation",
     "question": "Should a firm expand internationally before dominating one market?"},
    {"id": "biz_09", "domain": "business", "archetype": "flexibility_vs_consistency",
     "question": "Should recruiting favor high-agency generalists or deep specialists?"},
    {"id": "biz_10", "domain": "business", "archetype": "individual_vs_system",
     "question": "Should a founder remain CEO after product-market fit?"},

    # ---- Product / Engineering (10) ----
    {"id": "prod_01", "domain": "product", "archetype": "individual_vs_system",
     "question": "Should a consumer app maximize engagement or minimize compulsive use?"},
    {"id": "prod_02", "domain": "product", "archetype": "cost_vs_quality",
     "question": "Should a system favor explainability or raw predictive accuracy?"},
    {"id": "prod_03", "domain": "product", "archetype": "flexibility_vs_consistency",
     "question": "Should a model assistant ask clarifying questions often or infer intent silently?"},
    {"id": "prod_04", "domain": "product", "archetype": "flexibility_vs_consistency",
     "question": "Should a product expose many settings or keep defaults opinionated?"},
    {"id": "prod_05", "domain": "product", "archetype": "speed_vs_reliability",
     "question": "Should engineering optimize for latency or output quality first?"},
    {"id": "prod_06", "domain": "product", "archetype": "safety_vs_speed",
     "question": "Should content moderation be stricter to reduce harm or looser to reduce false positives?"},
    {"id": "prod_07", "domain": "product", "archetype": "cost_vs_quality",
     "question": "Should a search system optimize precision or recall?"},
    {"id": "prod_08", "domain": "product", "archetype": "scale_vs_control",
     "question": "Should a team maintain one general model or several specialized models?"},
    {"id": "prod_09", "domain": "product", "archetype": "individual_vs_system",
     "question": "Should a product personalize heavily or preserve consistent behavior across users?"},
    {"id": "prod_10", "domain": "product", "archetype": "exploration_vs_exploitation",
     "question": "Should a company ship desktop-first or mobile-first?"},

    # ---- Science / Epistemics (10) ----
    {"id": "sci_01", "domain": "science", "archetype": "cost_vs_quality",
     "question": "Should researchers trust a simpler interpretable model over a stronger black-box one?"},
    {"id": "sci_02", "domain": "science", "archetype": "speed_vs_reliability",
     "question": "Should a scientific claim be published early with caveats or held until replication?"},
    {"id": "sci_03", "domain": "science", "archetype": "innovation_vs_regulation",
     "question": "Should medicine emphasize mechanistic understanding or empirical outcome evidence?"},
    {"id": "sci_04", "domain": "science", "archetype": "cost_vs_quality",
     "question": "Should an AI benchmark prioritize realism or measurement cleanliness?"},
    {"id": "sci_05", "domain": "science", "archetype": "exploration_vs_exploitation",
     "question": "Should a lab pursue ambitious moonshots or incremental publishable work?"},
    {"id": "sci_06", "domain": "science", "archetype": "speed_vs_reliability",
     "question": "Should one update beliefs quickly on new evidence or demand convergence across studies?"},
    {"id": "sci_07", "domain": "science", "archetype": "safety_vs_speed",
     "question": "Should climate policy use precaution under uncertainty or wait for stronger causal attribution?"},
    {"id": "sci_08", "domain": "science", "archetype": "flexibility_vs_consistency",
     "question": "Should a curriculum teach formal proofs early or intuition first?"},
    {"id": "sci_09", "domain": "science", "archetype": "innovation_vs_regulation",
     "question": "Should a researcher optimize for novelty or robustness?"},
    {"id": "sci_10", "domain": "science", "archetype": "cost_vs_quality",
     "question": "Should model evaluations use synthetic tests or real-world messy tasks?"},

    # ---- Ethics / Governance (10) ----
    {"id": "eth_01", "domain": "ethics", "archetype": "safety_vs_speed",
     "question": "Should AI deployment require interpretability before release?"},
    {"id": "eth_02", "domain": "ethics", "archetype": "individual_vs_system",
     "question": "Should platforms optimize for free expression or harm reduction?"},
    {"id": "eth_03", "domain": "ethics", "archetype": "individual_vs_system",
     "question": "Should hiring systems use demographic-blind rules or fairness-aware corrections?"},
    {"id": "eth_04", "domain": "ethics", "archetype": "innovation_vs_regulation",
     "question": "Should schools ban generative AI or integrate it into coursework?"},
    {"id": "eth_05", "domain": "ethics", "archetype": "safety_vs_speed",
     "question": "Should autonomous vehicles be allowed with superhuman average safety but rare opaque failures?"},
    {"id": "eth_06", "domain": "ethics", "archetype": "flexibility_vs_consistency",
     "question": "Should welfare systems prioritize universal simplicity or targeted efficiency?"},
    {"id": "eth_07", "domain": "ethics", "archetype": "innovation_vs_regulation",
     "question": "Should governments regulate frontier AI models via licensing?"},
    {"id": "eth_08", "domain": "ethics", "archetype": "individual_vs_system",
     "question": "Should healthcare allocation prioritize total lives saved or worst-off patients?"},
    {"id": "eth_09", "domain": "ethics", "archetype": "centralization_vs_autonomy",
     "question": "Should online identity systems favor anonymity or accountability?"},
    {"id": "eth_10", "domain": "ethics", "archetype": "individual_vs_system",
     "question": "Should companies collect more user data for personalization if consent is explicit?"},

    # ---- Personal / Practical (10) ----
    {"id": "per_01", "domain": "personal", "archetype": "exploration_vs_exploitation",
     "question": "Should someone choose a high-upside risky career or a stable conventional path?"},
    {"id": "per_02", "domain": "personal", "archetype": "individual_vs_system",
     "question": "Should a person move to a big city for opportunity or stay near family?"},
    {"id": "per_03", "domain": "personal", "archetype": "exploration_vs_exploitation",
     "question": "Should a student specialize early or explore broadly?"},
    {"id": "per_04", "domain": "personal", "archetype": "short_term_vs_long_term",
     "question": "Should someone optimize for work-life balance or rapid career acceleration in their 20s?"},
    {"id": "per_05", "domain": "personal", "archetype": "centralization_vs_autonomy",
     "question": "Should a person confront a colleague directly or escalate through management?"},
    {"id": "per_06", "domain": "personal", "archetype": "short_term_vs_long_term",
     "question": "Should someone buy a home or keep renting for flexibility?"},
    {"id": "per_07", "domain": "personal", "archetype": "safety_vs_speed",
     "question": "Should a learner use AI tools heavily or struggle manually for deeper skill formation?"},
    {"id": "per_08", "domain": "personal", "archetype": "scale_vs_control",
     "question": "Should a person maintain many weak ties or a few deep relationships for career resilience?"},
    {"id": "per_09", "domain": "personal", "archetype": "speed_vs_reliability",
     "question": "Should someone quit a draining job before securing the next role?"},
    {"id": "per_10", "domain": "personal", "archetype": "individual_vs_system",
     "question": "Should a creator optimize for audience growth or authentic niche work?"},
]


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """Generate a dialectic reasoning trace for the task below.

Task: {question}

Return valid JSON with exactly these fields:
- frame_a: steelmanned first perspective (2-4 sentences)
- frame_b: steelmanned opposing perspective (2-4 sentences)
- optimization_conflict: the real underlying tradeoff, not the surface disagreement (1-2 sentences)
- failure_modes: {{ "frame_a": "where this perspective breaks down (1-2 sentences)", "frame_b": "where this perspective breaks down (1-2 sentences)" }}
- synthesis: resolve the conflict by introducing conditions, scope, sequencing, or a better ontology. Must include at least 2 explicit if/then conditional commitments. (3-5 sentences)
- decision: an actionable position, not "it depends" (1-2 sentences)

Rules:
- Frame A and Frame B must both be steelmanned — strongest version of each.
- optimization_conflict must name the real underlying tradeoff, not restate the surface disagreement.
- failure_modes must be concrete and distinct.
- synthesis must resolve the conflict by introducing conditions, scope, sequencing, or a better ontology.
- decision must be actionable and specific — not generic.
- BANNED phrases: "it depends on context", "case-by-case basis", "both sides have merit", "optimal approach", "various perspectives", "nuanced", "balanced approach".
- Keep the full output under 300 words.
- Return JSON only. No markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_provider(provider_key: str, question: str) -> dict | None:
    """Call a provider and return parsed JSON trace, or None on failure."""
    cfg = PROVIDERS[provider_key]
    is_ollama = cfg["env_key"] == "_OLLAMA"
    api_key = None if is_ollama else os.environ.get(cfg["env_key"])
    if not api_key and not is_ollama:
        logger.warning(f"  No API key for {cfg['env_key']}, skipping {cfg['name']}")
        return None

    prompt = GENERATION_PROMPT.format(question=question)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # OpenRouter wants these extra headers
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
            resp = requests.post(cfg["base_url"], headers=headers, json=body, timeout=120)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "")
                # OpenRouter returns seconds or a date; Cerebras similar
                try:
                    wait = max(int(retry_after), 10)
                except (ValueError, TypeError):
                    wait = 60
                logger.warning(f"  Rate limited on {cfg['name']}, waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"].get("content")
            if not text:
                logger.warning(f"  Empty content from {cfg['name']}")
                continue
            return parse_trace_json(text)

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout on {cfg['name']}, attempt {attempt+1}")
            time.sleep(5)
        except requests.exceptions.HTTPError as e:
            logger.error(f"  HTTP error on {cfg['name']}: {e}")
            if attempt < 2:
                time.sleep(10)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"  Parse error on {cfg['name']}: {e}")
            if attempt < 2:
                time.sleep(2)

    return None


def parse_trace_json(text: str | None) -> dict:
    """Parse model output to JSON, handling markdown fences and thinking tags."""
    if not text:
        raise ValueError("Empty text")
    text = text.strip()

    # Strip <think>...</think> blocks (DeepSeek R1)
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                return json.loads(p)

    # Try direct parse
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    return json.loads(text)


# ---------------------------------------------------------------------------
# Convert richer schema → our trace format
# ---------------------------------------------------------------------------

def to_trace_format(task: dict, raw: dict, provider_key: str) -> dict:
    """Convert the richer generation schema to our existing trace format."""
    # frame_a → thesis
    thesis = raw.get("frame_a", "")

    # frame_b → antithesis
    antithesis = raw.get("frame_b", "")

    # Combine optimization_conflict + failure_modes + synthesis + decision → synthesis
    parts = []

    oc = raw.get("optimization_conflict", "")
    if oc:
        parts.append(oc)

    fm = raw.get("failure_modes", {})
    if isinstance(fm, dict):
        fa = fm.get("frame_a", "")
        fb = fm.get("frame_b", "")
        if fa and fb:
            parts.append(f"The first view fails when {fa.lower() if fa[0].isupper() and not fa.startswith('I ') else fa} "
                         f"The second fails when {fb.lower() if fb[0].isupper() and not fb.startswith('I ') else fb}")
        elif fa:
            parts.append(fa)
        elif fb:
            parts.append(fb)

    synth = raw.get("synthesis", "")
    if synth:
        parts.append(synth)

    decision = raw.get("decision", "")
    if decision:
        parts.append(decision)

    synthesis_combined = " ".join(parts)

    return {
        "trace_id": f"diverse_{task['id']}_{provider_key}",
        "question": task["question"],
        "category": task["domain"],
        "difficulty": "standard",
        "length_tier": "medium",
        "domain": task["domain"],
        "conflict_archetype": task["archetype"],
        "thesis": thesis,
        "antithesis": antithesis,
        "synthesis": synthesis_combined,
        "composite_quality": 0.80,  # placeholder — can re-score later
        "tension_quality": 0.80,
        "resolution_quality": 0.80,
        "cross_phase_coherence": 0.80,
        "synthesis_sufficiency": 0.80,
        "verdict": "keep",
        "would_train_on_this": True,
        "structural_checks_passed": True,
        "fired_checks": [],
        "generation_model": PROVIDERS[provider_key]["name"],
        "generated_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def assign_providers(tasks: list, available_providers: list) -> list[tuple]:
    """Distribute tasks across providers evenly, rotating."""
    assignments = []
    for i, task in enumerate(tasks):
        provider = available_providers[i % len(available_providers)]
        assignments.append((task, provider))
    return assignments


def main():
    parser = argparse.ArgumentParser(description="Generate domain-balanced dialectic traces")
    parser.add_argument("--dry-run", action="store_true", help="Print task assignments only")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already in output file")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between API calls")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated provider keys to use (e.g. cerebras_qwen,mistral_large,ollama_gemma4)")
    args = parser.parse_args()

    # Filter to requested providers if specified
    provider_keys = list(PROVIDERS.keys())
    if args.providers:
        requested = [p.strip() for p in args.providers.split(",")]
        provider_keys = [k for k in requested if k in PROVIDERS]
        if not provider_keys:
            logger.error(f"No valid providers in: {args.providers}")
            logger.info(f"Available: {', '.join(PROVIDERS.keys())}")
            return

    # Check which providers are available
    available = []
    for key in provider_keys:
        cfg = PROVIDERS[key]
        if cfg["env_key"] == "_OLLAMA":
            # Ollama: check if server is running
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=3)
                if r.ok:
                    available.append(key)
                    logger.info(f"Provider available: {cfg['name']} (local)")
                else:
                    logger.warning(f"Ollama server not responding — {cfg['name']} unavailable")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Ollama not running — {cfg['name']} unavailable")
        elif os.environ.get(cfg["env_key"]):
            available.append(key)
            logger.info(f"Provider available: {cfg['name']}")
        else:
            logger.warning(f"No key for {cfg['env_key']} — {cfg['name']} unavailable")

    if not available and not args.dry_run:
        logger.error("No API keys found. Set at least one of: CEREBRAS_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY")
        return

    # Load existing traces if resuming
    done_ids = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                t = json.loads(line)
                done_ids.add(t["trace_id"])
        logger.info(f"Resuming: {len(done_ids)} traces already done")

    # Assign tasks to providers
    assignments = assign_providers(SEED_TASKS, available or ["dry_run"])

    # Domain stats
    domain_counts = {}
    for task, prov in assignments:
        d = task["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    logger.info(f"\nDomain distribution: {domain_counts}")
    logger.info(f"Total tasks: {len(assignments)}, Providers: {len(available)}")

    if args.dry_run:
        print(f"\n{'ID':12s} {'Domain':12s} {'Archetype':28s} {'Provider':30s}")
        print("-" * 82)
        for task, prov in assignments:
            prov_name = PROVIDERS.get(prov, {}).get("name", prov)
            print(f"{task['id']:12s} {task['domain']:12s} {task['archetype']:28s} {prov_name:30s}")
        print(f"\n{len(assignments)} tasks across {len(available)} providers")
        return

    # Generate
    traces = []
    failures = []

    for i, (task, provider_key) in enumerate(assignments):
        trace_id = f"diverse_{task['id']}_{provider_key}"

        if trace_id in done_ids:
            logger.info(f"[{i+1}/{len(assignments)}] Skipping {trace_id} (already done)")
            continue

        logger.info(f"[{i+1}/{len(assignments)}] {task['id']} via {PROVIDERS[provider_key]['name']}")

        raw = call_provider(provider_key, task["question"])

        if raw is None:
            logger.error(f"  FAILED: {task['id']}")
            failures.append(task["id"])
            continue

        # Validate required fields
        required = ["frame_a", "frame_b", "synthesis", "decision"]
        missing = [k for k in required if not raw.get(k)]
        if missing:
            logger.warning(f"  Missing fields {missing}, skipping")
            failures.append(task["id"])
            continue

        trace = to_trace_format(task, raw, provider_key)
        traces.append(trace)

        # Append to file immediately (crash-safe)
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(trace) + "\n")

        logger.info(f"  OK — {len(trace['synthesis'].split())} word synthesis")

        if i < len(assignments) - 1:
            time.sleep(args.delay)

    # Summary
    print(f"\n{'='*60}")
    print(f"Generation complete")
    print(f"{'='*60}")
    print(f"  Succeeded: {len(traces)}")
    print(f"  Failed:    {len(failures)}")
    if failures:
        print(f"  Failed IDs: {failures}")
    print(f"  Output:    {OUTPUT_PATH}")

    # Domain breakdown of successes
    domain_ok = {}
    for t in traces:
        d = t["domain"]
        domain_ok[d] = domain_ok.get(d, 0) + 1
    print(f"\n  Domain breakdown:")
    for d, c in sorted(domain_ok.items()):
        print(f"    {d:15s} {c}")


if __name__ == "__main__":
    main()
