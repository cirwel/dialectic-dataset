"""Generate dialectic traces via Claude API.

Produces thesis/antithesis/synthesis traces for each prompt,
scores them, and filters based on quality thresholds.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dialectic_dataset.prompts import Prompt, get_prompts, ALL_PROMPTS
from dialectic_dataset.trace_scorer import TraceScore, score_trace, KEEP_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Configuration for trace generation."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.8

    # Filtering
    min_quality: float = KEEP_THRESHOLD
    require_structural_pass: bool = True
    require_would_train: bool = True

    # Diversity
    max_per_prompt: int = 3           # Max accepted traces per prompt
    max_per_category: int = 50        # Max accepted traces per category
    near_duplicate_threshold: float = 0.6  # Jaccard similarity for dedup

    # Retries
    max_retries_per_prompt: int = 5   # How many attempts before giving up
    retry_delay: float = 1.0

    # Concurrency
    max_concurrent: int = 5

    # Output
    output_dir: str = "data/generated"


# ---------------------------------------------------------------------------
# Generation prompts (sent to Claude)
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are generating training data for dialectic reasoning. For the question \
provided, produce a structured analysis with three clearly labeled sections.

Requirements:
- **Thesis**: Present the strongest case for one side. Use specific evidence, \
examples, or reasoning. No hedging or "both sides have merit" openings. \
Minimum 3 sentences.
- **Antithesis**: Present the strongest OPPOSING case. This must genuinely \
challenge the thesis with different evidence or reasoning, not just restate it \
with different words. Minimum 3 sentences.
- **Synthesis**: Integrate the tension into a resolution that acknowledges what's \
valid in BOTH sides. Do NOT just summarize both sides. Do NOT pick one side. \
Instead, identify the underlying variable that determines when each side is right, \
or find a conditional/contextual resolution. Minimum 4 sentences.

Output as JSON with keys "thesis", "antithesis", "synthesis". Each value is a \
string of natural prose (no bullet points, no sub-headers)."""


LENGTH_INSTRUCTIONS = {
    "short": "Keep each section to 2-4 sentences. Be concise and direct.",
    "medium": "Each section should be 3-6 sentences. Develop arguments with specifics.",
    "long": (
        "Each section should be 5-10 sentences. Develop arguments thoroughly "
        "with evidence, examples, and nuanced reasoning."
    ),
}


def _build_generation_prompt(prompt: Prompt, length_tier: str) -> str:
    """Build the user prompt for generation."""
    length_note = LENGTH_INSTRUCTIONS.get(length_tier, LENGTH_INSTRUCTIONS["medium"])
    return f"{prompt.question}\n\n{length_note}"


# ---------------------------------------------------------------------------
# Trace dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratedTrace:
    """A generated and scored dialectic trace."""

    trace_id: str
    prompt: Prompt
    length_tier: str
    thesis: str
    antithesis: str
    synthesis: str
    score: TraceScore
    generation_model: str
    generated_at: float  # unix timestamp

    @property
    def accepted(self) -> bool:
        return self.score.verdict == "keep" and self.score.would_train_on_this

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "question": self.prompt.question,
            "category": self.prompt.category,
            "difficulty": self.prompt.difficulty,
            "length_tier": self.length_tier,
            "thesis": self.thesis,
            "antithesis": self.antithesis,
            "synthesis": self.synthesis,
            "composite_quality": self.score.composite_quality,
            "tension_quality": self.score.tension_quality,
            "resolution_quality": self.score.resolution_quality,
            "cross_phase_coherence": self.score.cross_phase_coherence,
            "synthesis_sufficiency": self.score.synthesis_sufficiency,
            "verdict": self.score.verdict,
            "would_train_on_this": self.score.would_train_on_this,
            "structural_checks_passed": self.score.structural_report.passed,
            "fired_checks": self.score.structural_report.fired_names,
            "generation_model": self.generation_model,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

def _synthesis_fingerprint(synthesis: str) -> Set[str]:
    """Extract content word set for dedup."""
    from dialectic_dataset.signal_extractor import term_set
    return term_set(synthesis)


def _is_near_duplicate(
    new_synthesis: str,
    existing_fingerprints: List[Set[str]],
    threshold: float,
) -> bool:
    """Check if a synthesis is too similar to existing accepted traces."""
    from dialectic_dataset.signal_extractor import jaccard
    new_fp = _synthesis_fingerprint(new_synthesis)
    for existing_fp in existing_fingerprints:
        if jaccard(new_fp, existing_fp) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Single trace generation
# ---------------------------------------------------------------------------

async def _generate_one(
    client,  # anthropic.AsyncAnthropic
    prompt: Prompt,
    length_tier: str,
    config: GenerationConfig,
) -> Optional[GeneratedTrace]:
    """Generate a single trace via Claude API.

    Returns None if generation fails or response can't be parsed.
    """
    user_prompt = _build_generation_prompt(prompt, length_tier)

    try:
        response = await client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=GENERATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        logger.warning(f"API error for '{prompt.question[:50]}...': {e}")
        return None

    # Extract text from response
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    # Parse JSON from response
    try:
        # Handle potential markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        data = json.loads(cleaned)
        thesis = data.get("thesis", "").strip()
        antithesis = data.get("antithesis", "").strip()
        synthesis = data.get("synthesis", "").strip()

        if not thesis or not antithesis or not synthesis:
            logger.warning(f"Empty phase in response for '{prompt.question[:50]}...'")
            return None

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Parse error for '{prompt.question[:50]}...': {e}")
        return None

    # Score the trace
    trace_score = score_trace(thesis, antithesis, synthesis)

    trace_id = f"gen_{uuid.uuid4().hex[:12]}"

    return GeneratedTrace(
        trace_id=trace_id,
        prompt=prompt,
        length_tier=length_tier,
        thesis=thesis,
        antithesis=antithesis,
        synthesis=synthesis,
        score=trace_score,
        generation_model=config.model,
        generated_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Results from a batch generation run."""

    accepted: List[GeneratedTrace] = field(default_factory=list)
    rejected: List[GeneratedTrace] = field(default_factory=list)
    errors: int = 0
    duplicates_skipped: int = 0

    @property
    def total_generated(self) -> int:
        return len(self.accepted) + len(self.rejected) + self.errors

    @property
    def acceptance_rate(self) -> float:
        total = len(self.accepted) + len(self.rejected)
        return len(self.accepted) / max(1, total)

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for trace in self.accepted:
            cat = trace.prompt.category
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def length_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for trace in self.accepted:
            tier = trace.length_tier
            counts[tier] = counts.get(tier, 0) + 1
        return counts


async def generate_batch(
    config: GenerationConfig,
    prompts: Optional[List[Prompt]] = None,
    categories: Optional[List[str]] = None,
    target_count: Optional[int] = None,
) -> GenerationResult:
    """Generate a batch of dialectic traces.

    Parameters
    ----------
    config:
        Generation configuration.
    prompts:
        Specific prompts to use. If None, uses all prompts.
    categories:
        Filter to specific categories. Ignored if prompts is provided.
    target_count:
        Target number of accepted traces. If None, generates
        max_per_prompt * len(prompts) traces.

    Returns
    -------
    GenerationResult with accepted and rejected traces.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for generation. "
            "Install with: pip install dialectic-dataset[gen]"
        )

    client = AsyncAnthropic()

    if prompts is None:
        prompts = get_prompts(categories=categories)

    result = GenerationResult()
    fingerprints: List[Set[str]] = []  # For dedup
    category_counts: Dict[str, int] = {}
    prompt_counts: Dict[str, int] = {}

    # Build work queue: (prompt, length_tier) pairs
    length_tiers = ["short", "medium", "medium", "medium", "long"]  # 20/60/20 ish
    work_queue: List[Tuple[Prompt, str]] = []
    for prompt in prompts:
        for tier in length_tiers:
            work_queue.append((prompt, tier))

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _process_one(prompt: Prompt, length_tier: str) -> None:
        # Check caps
        prompt_key = prompt.question[:80]
        if prompt_counts.get(prompt_key, 0) >= config.max_per_prompt:
            return
        if category_counts.get(prompt.category, 0) >= config.max_per_category:
            return
        if target_count and len(result.accepted) >= target_count:
            return

        async with semaphore:
            for attempt in range(config.max_retries_per_prompt):
                trace = await _generate_one(client, prompt, length_tier, config)

                if trace is None:
                    result.errors += 1
                    if attempt < config.max_retries_per_prompt - 1:
                        await asyncio.sleep(config.retry_delay)
                    continue

                # Check quality
                if not trace.accepted:
                    result.rejected.append(trace)
                    logger.info(
                        f"Rejected: {trace.trace_id} "
                        f"(verdict={trace.score.verdict}, "
                        f"composite={trace.score.composite_quality:.3f}, "
                        f"fired={trace.score.structural_report.fired_names})"
                    )
                    continue

                # Check near-duplicate
                if _is_near_duplicate(
                    trace.synthesis, fingerprints, config.near_duplicate_threshold
                ):
                    result.duplicates_skipped += 1
                    logger.info(f"Duplicate skipped: {trace.trace_id}")
                    continue

                # Accept
                result.accepted.append(trace)
                fingerprints.append(_synthesis_fingerprint(trace.synthesis))
                prompt_counts[prompt_key] = prompt_counts.get(prompt_key, 0) + 1
                category_counts[prompt.category] = (
                    category_counts.get(prompt.category, 0) + 1
                )
                logger.info(
                    f"Accepted: {trace.trace_id} "
                    f"(quality={trace.score.composite_quality:.3f}, "
                    f"cat={prompt.category}, tier={length_tier})"
                )
                break  # Got an accepted trace, move on

    # Run concurrently
    tasks = [_process_one(p, t) for p, t in work_queue]
    await asyncio.gather(*tasks)

    return result


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------

def save_traces(traces: List[GeneratedTrace], path: str) -> None:
    """Save generated traces to a JSONL file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict()) + "\n")
    logger.info(f"Saved {len(traces)} traces to {output}")


def load_traces(path: str) -> List[dict]:
    """Load generated traces from a JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces
