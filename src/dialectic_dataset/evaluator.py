"""Evaluate fine-tuned models on held-out prompts.

Two evaluation dimensions:
1. Tradeoff reasoning quality — can the model consider competing frames?
2. Fake-dialectic resistance — does the model resist producing ceremonial dialectic?
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dialectic_dataset.trace_scorer import score_trace, KEEP_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Held-out evaluation prompts
# ---------------------------------------------------------------------------

# Tradeoff prompts: questions requiring genuine integrative reasoning
# These are NOT in the training set prompt families
TRADEOFF_EVAL_PROMPTS = [
    {
        "id": "eval_tradeoff_01",
        "question": (
            "A small company has a toxic but highly productive employee. "
            "Should they fire this person?"
        ),
        "category": "tradeoff",
    },
    {
        "id": "eval_tradeoff_02",
        "question": (
            "Should a country invest heavily in space exploration when "
            "it has significant poverty and infrastructure problems?"
        ),
        "category": "tradeoff",
    },
    {
        "id": "eval_tradeoff_03",
        "question": (
            "Is it better to have strict immigration controls that protect "
            "existing workers, or open borders that maximize overall "
            "economic growth?"
        ),
        "category": "tradeoff",
    },
    {
        "id": "eval_tradeoff_04",
        "question": (
            "Should a software company open-source its core product to "
            "build community, even though competitors will benefit?"
        ),
        "category": "tradeoff",
    },
    {
        "id": "eval_tradeoff_05",
        "question": (
            "A manager must choose between promoting a loyal long-term "
            "employee or hiring a more qualified external candidate. "
            "What should they do?"
        ),
        "category": "tradeoff",
    },
    {
        "id": "eval_tradeoff_06",
        "question": (
            "Should universities prioritize research output or teaching "
            "quality when evaluating faculty for tenure?"
        ),
        "category": "tradeoff",
    },
    {
        "id": "eval_tradeoff_07",
        "question": (
            "Is it ethical to use animals in medical research that could "
            "save human lives?"
        ),
        "category": "tradeoff",
    },
    {
        "id": "eval_tradeoff_08",
        "question": (
            "Should parents monitor their teenager's phone and social "
            "media to protect them, even if it violates their privacy?"
        ),
        "category": "tradeoff",
    },
]

# Fake-dialectic resistance prompts: questions designed to elicit
# ceremonial both-sides reasoning. A well-trained model should give
# direct, integrated answers rather than formulaic dialectic.
RESISTANCE_EVAL_PROMPTS = [
    {
        "id": "eval_resist_01",
        "question": "What are the pros and cons of renewable energy?",
        "category": "resistance",
        "trap": "list_format",
    },
    {
        "id": "eval_resist_02",
        "question": (
            "Present both sides of the debate about whether college "
            "is worth the cost."
        ),
        "category": "resistance",
        "trap": "both_sides_request",
    },
    {
        "id": "eval_resist_03",
        "question": "Is capitalism better than socialism?",
        "category": "resistance",
        "trap": "false_binary",
    },
    {
        "id": "eval_resist_04",
        "question": (
            "Give me a balanced perspective on gun control in America."
        ),
        "category": "resistance",
        "trap": "balance_request",
    },
    {
        "id": "eval_resist_05",
        "question": "What do you think about work-life balance?",
        "category": "resistance",
        "trap": "vague_topic",
    },
    {
        "id": "eval_resist_06",
        "question": (
            "Consider the arguments for and against remote work, then "
            "provide your analysis."
        ),
        "category": "resistance",
        "trap": "structured_request",
    },
]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalResponse:
    """A single evaluation response."""

    prompt_id: str
    question: str
    category: str
    response: str
    word_count: int
    # Quality signals (not full scoring — response isn't in thesis/anti/synth format)
    has_conditional_reasoning: bool   # Uses "if/when/depends on" with specifics
    has_specific_claims: bool         # Makes concrete claims, not just generalities
    avoids_ceremony: bool             # Doesn't use "both sides", "balanced", etc.
    avoids_list_format: bool          # Doesn't use bullet-point pros/cons
    reaches_conclusion: bool          # Takes a position or gives conditional answer


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    model_name: str
    responses: List[EvalResponse] = field(default_factory=list)
    eval_time: float = 0.0

    @property
    def tradeoff_responses(self) -> List[EvalResponse]:
        return [r for r in self.responses if r.category == "tradeoff"]

    @property
    def resistance_responses(self) -> List[EvalResponse]:
        return [r for r in self.responses if r.category == "resistance"]

    def tradeoff_scores(self) -> Dict[str, float]:
        """Score tradeoff reasoning quality."""
        resps = self.tradeoff_responses
        if not resps:
            return {}
        n = len(resps)
        return {
            "conditional_reasoning_rate": sum(r.has_conditional_reasoning for r in resps) / n,
            "specific_claims_rate": sum(r.has_specific_claims for r in resps) / n,
            "reaches_conclusion_rate": sum(r.reaches_conclusion for r in resps) / n,
            "avg_word_count": sum(r.word_count for r in resps) / n,
        }

    def resistance_scores(self) -> Dict[str, float]:
        """Score fake-dialectic resistance."""
        resps = self.resistance_responses
        if not resps:
            return {}
        n = len(resps)
        return {
            "avoids_ceremony_rate": sum(r.avoids_ceremony for r in resps) / n,
            "avoids_list_format_rate": sum(r.avoids_list_format for r in resps) / n,
            "reaches_conclusion_rate": sum(r.reaches_conclusion for r in resps) / n,
            "avg_word_count": sum(r.word_count for r in resps) / n,
        }


# ---------------------------------------------------------------------------
# Response analysis (heuristic)
# ---------------------------------------------------------------------------

def analyze_response(response: str) -> Dict[str, bool]:
    """Analyze a response for quality signals.

    Uses heuristic checks — not a full scorer, since the response
    isn't structured as thesis/antithesis/synthesis.
    """
    lower = response.lower()
    words = lower.split()
    word_count = len(words)

    # Conditional reasoning: uses specific conditionals, not just "it depends"
    conditional_phrases = [
        "if the", "when the", "in cases where", "depends on whether",
        "the key variable is", "the deciding factor",
        "under conditions", "assuming that",
    ]
    has_conditional = any(p in lower for p in conditional_phrases)

    # Specific claims: contains numbers, names, or concrete assertions
    has_specific = (
        any(c.isdigit() for c in response)
        or any(word[0].isupper() and len(word) > 2 for word in response.split() if word)
        or "for example" in lower
        or "specifically" in lower
        or "in practice" in lower
        or "evidence shows" in lower
    )

    # Ceremony avoidance: doesn't use generic dialectic phrases
    ceremony_markers = [
        "both sides have", "both perspectives", "there are pros and cons",
        "on one hand", "on the other hand",
        "the truth lies somewhere", "balanced approach",
        "reasonable people can disagree",
    ]
    ceremony_count = sum(1 for m in ceremony_markers if m in lower)
    avoids_ceremony = ceremony_count <= 1

    # List format avoidance
    has_list = (
        response.count("- ") > 2
        or response.count("• ") > 2
        or (response.count("\n1.") >= 1 and response.count("\n2.") >= 1)
        or ("pros:" in lower and "cons:" in lower)
    )
    avoids_list = not has_list

    # Reaches conclusion: doesn't end with uncertainty
    last_100 = lower[-100:] if len(lower) > 100 else lower
    non_conclusion_endings = [
        "it depends", "there are no easy answers",
        "more research is needed", "time will tell",
        "only time will tell", "remains to be seen",
    ]
    reaches_conclusion = not any(e in last_100 for e in non_conclusion_endings)

    return {
        "has_conditional_reasoning": has_conditional,
        "has_specific_claims": has_specific,
        "avoids_ceremony": avoids_ceremony,
        "avoids_list_format": avoids_list,
        "reaches_conclusion": reaches_conclusion,
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    prompts: Optional[List[dict]] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> EvalResult:
    """Run evaluation on a model.

    Parameters
    ----------
    model:
        The model to evaluate (can be base or fine-tuned).
    tokenizer:
        The tokenizer.
    prompts:
        List of prompt dicts. If None, uses all eval prompts.
    max_new_tokens:
        Max tokens to generate per response.
    temperature:
        Generation temperature.

    Returns
    -------
    EvalResult with all responses and scores.
    """
    from dialectic_dataset.trainer import generate_response

    if prompts is None:
        prompts = TRADEOFF_EVAL_PROMPTS + RESISTANCE_EVAL_PROMPTS

    result = EvalResult(model_name=str(getattr(model, "name_or_path", "unknown")))
    start = time.time()

    for prompt_info in prompts:
        pid = prompt_info["id"]
        question = prompt_info["question"]
        category = prompt_info["category"]

        logger.info(f"Evaluating: {pid}")
        response = generate_response(
            model, tokenizer, question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        analysis = analyze_response(response)

        eval_resp = EvalResponse(
            prompt_id=pid,
            question=question,
            category=category,
            response=response,
            word_count=len(response.split()),
            **analysis,
        )
        result.responses.append(eval_resp)

    result.eval_time = time.time() - start
    return result


def print_eval_report(result: EvalResult) -> None:
    """Print evaluation results."""
    print("=" * 60)
    print(f"Evaluation Report: {result.model_name}")
    print(f"Eval time: {result.eval_time:.1f}s")
    print("=" * 60)

    tradeoff = result.tradeoff_scores()
    if tradeoff:
        print("\nTradeoff Reasoning Quality:")
        for key, val in tradeoff.items():
            if isinstance(val, float) and val <= 1.0:
                print(f"  {key:35s} {val:.0%}")
            else:
                print(f"  {key:35s} {val:.0f}")

    resistance = result.resistance_scores()
    if resistance:
        print("\nFake-Dialectic Resistance:")
        for key, val in resistance.items():
            if isinstance(val, float) and val <= 1.0:
                print(f"  {key:35s} {val:.0%}")
            else:
                print(f"  {key:35s} {val:.0f}")

    print("\nPer-response details:")
    for r in result.responses:
        flags = []
        if r.has_conditional_reasoning:
            flags.append("conditional")
        if r.has_specific_claims:
            flags.append("specific")
        if r.avoids_ceremony:
            flags.append("no-ceremony")
        if r.avoids_list_format:
            flags.append("no-list")
        if r.reaches_conclusion:
            flags.append("concludes")
        flag_str = ", ".join(flags) if flags else "none"
        print(f"  {r.prompt_id:20s} {r.word_count:4d}w [{flag_str}]")

    print("=" * 60)


def save_eval_result(result: EvalResult, path: str) -> None:
    """Save evaluation results to JSON."""
    output = {
        "model_name": result.model_name,
        "eval_time": result.eval_time,
        "tradeoff_scores": result.tradeoff_scores(),
        "resistance_scores": result.resistance_scores(),
        "responses": [
            {
                "prompt_id": r.prompt_id,
                "question": r.question,
                "category": r.category,
                "response": r.response,
                "word_count": r.word_count,
                "has_conditional_reasoning": r.has_conditional_reasoning,
                "has_specific_claims": r.has_specific_claims,
                "avoids_ceremony": r.avoids_ceremony,
                "avoids_list_format": r.avoids_list_format,
                "reaches_conclusion": r.reaches_conclusion,
            }
            for r in result.responses
        ],
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved eval results to {path}")
