"""Dialectic Reasoning Explorer — Interactive Benchmark, Goldset & Council Audit Space.

Designed to work cleanly on both CPU-only Hugging Face Spaces and GPU environments:
1. Side-by-Side Model Comparison (Base Qwen3-8B vs Dialectic Qwen3-8B LoRA across 14 held-out prompts)
2. Self-Verifying <think> Trace & Cross-Family Council Audit Explorer (live Stage 1 + Stage 4 scorer + council verdicts)
3. Live Scorer Sandbox (paste any <think> + resolution trace to score falsifiability & deliberative thinking quality on CPU)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Lightweight stdlib scorers (works standalone on HF Spaces without src/)
# ---------------------------------------------------------------------------

_DIRECTION_CHANGE_RE = re.compile(
    r"\b(?:wait|actually|hold\s+on|no\s*[—,-]|on\s+second\s+thought|"
    r"Wait\s*[—,-]|Actually\s*,|Let\s+me\s+reconsider|Looking\s+closer|"
    r"Scratch\s+that|Wait\s+a\s+second|Wait\s+though)\b",
    re.IGNORECASE,
)
_ASSUMPTION_RE = re.compile(
    r"\b(?:assumes?|assuming|assumption|hidden\s+premise|implicit\s+premise|"
    r"presupposes?|takes?\s+for\s+granted|underneath\s+this|real\s+question\s+is|"
    r"framing\s+assumes)\b",
    re.IGNORECASE,
)
_SELF_CORRECTION_RE = re.compile(
    r"\b(?:my\s+initial\s+instinct|at\s+first\s+glance|i\s+was\s+about\s+to|"
    r"tempting\s+to\s+say|naive\s+answer|surface-level|wrong\s+way\s+to\s+frame|"
    r"instead\s+of\s+hedging|rather\s+than\s+splitting\s+the\s+difference|"
    r"avoid\s+inventing)\b",
    re.IGNORECASE,
)
_MECHANISM_RE = re.compile(
    r"\b(?:because|causes?|leads?\s+to|results?\s+in|mechanism|incentive|"
    r"feedback\s+loop|bottleneck|constraint|coupling|blast\s+radius|"
    r"coordination\s+cost|asymmetry|irreversible|reversible|state\s+transition|"
    r"invariant|rollback|boundary\s+condition|failure\s+mode)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|when|unless|provided\s+that|only\s+when|whenever)\b",
    re.IGNORECASE,
)
_FABRICATED_CITATION_RE = re.compile(
    r"\b(?:(?:a|the|an)\s+(?:recent\s+)?20\d\d\s+(?:\w+\s+){0,4}"
    r"(?:study|report|survey|trial|analysis|review|paper|research)|"
    r"(?:study|report|survey|research)\s+(?:by|from|at)\s+[A-Z])",
    re.IGNORECASE,
)


def extract_think_and_visible(text: str) -> Tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return "", text.strip()
    think = m.group(1).strip()
    visible = (text[: m.start()] + text[m.end() :]).strip()
    return think, visible


def score_trace_interactive(text: str) -> Dict[str, Any]:
    think, visible = extract_think_and_visible(text)
    dir_changes = len(_DIRECTION_CHANGE_RE.findall(think))
    assumptions = len(_ASSUMPTION_RE.findall(think))
    corrections = len(_SELF_CORRECTION_RE.findall(think))
    think_mechs = len(_MECHANISM_RE.findall(think))

    vis_conds = len(_CONDITIONAL_RE.findall(visible))
    vis_mechs = len(_MECHANISM_RE.findall(visible))
    fab_cites = len(_FABRICATED_CITATION_RE.findall(text))

    think_score = min(
        1.0,
        0.25 * min(1.0, dir_changes / 2)
        + 0.25 * min(1.0, assumptions / 2)
        + 0.25 * min(1.0, corrections / 2)
        + 0.25 * min(1.0, think_mechs / 4),
    )
    fals_score = min(
        1.0,
        0.45 * min(1.0, vis_conds / 3)
        + 0.55 * min(1.0, vis_mechs / 5)
        - 0.25 * fab_cites,
    )
    fals_score = max(0.0, fals_score)

    return {
        "has_think_block": bool(think),
        "think_word_count": len(think.split()),
        "visible_word_count": len(visible.split()),
        "thinking_composite": round(think_score, 3),
        "falsifiability_composite": round(fals_score, 3),
        "signals": {
            "direction_changes_in_think": dir_changes,
            "assumptions_surfaced_in_think": assumptions,
            "self_corrections_in_think": corrections,
            "mechanisms_in_think": think_mechs,
            "conditionals_in_conclusion": vis_conds,
            "mechanisms_in_conclusion": vis_mechs,
            "fabricated_citation_flags": fab_cites,
        },
    }


# ---------------------------------------------------------------------------
# Load bundled datasets
# ---------------------------------------------------------------------------

def _load_benchmarks() -> List[Dict[str, Any]]:
    path = HERE / "benchmark_8b.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("benchmarks", [])


def _load_council_traces() -> List[Dict[str, Any]]:
    path = HERE / "council_pilot_v9.jsonl"
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _load_goldset() -> List[Dict[str, Any]]:
    path = HERE / "goldset_thinking_v1.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("samples", [])


BENCHMARKS = _load_benchmarks()
COUNCIL_TRACES = _load_council_traces()
GOLDSET = _load_goldset()


# ---------------------------------------------------------------------------
# Tab 1: Base vs Fine-Tuned 8B Benchmark Explorer
# ---------------------------------------------------------------------------

BENCHMARK_CHOICES = [
    f"[{b['prompt_id']}] ({b['category']}) {b['question']}"
    for b in BENCHMARKS
]


def render_benchmark(choice: str) -> Tuple[str, str, str]:
    if not BENCHMARKS:
        return "No benchmark data found.", "", ""
    idx = BENCHMARK_CHOICES.index(choice) if choice in BENCHMARK_CHOICES else 0
    item = BENCHMARKS[idx]

    b_agg = item["base_aggregate"]
    f_agg = item["adapter_aggregate"]
    b_rub = item["base_rubric"]
    f_rub = item["adapter_rubric"]

    header_md = (
        f"### Prompt: *{item['question']}*\n"
        f"**Category:** `{item['category']}` &nbsp;|&nbsp; "
        f"**Base Qwen3-8B Score:** `{b_agg['total']} / 10` ({b_agg['verdict']}) &nbsp;|&nbsp; "
        f"**Dialectic Qwen3-8B LoRA Score:** `{f_agg['total']} / 10` ({f_agg['verdict']})\n\n"
        "| Rubric Dimension | Base `Qwen/Qwen3-8B` | Fine-Tuned `hikewa/dialectic-qwen3-8b-lora` |\n"
        "| :--- | :---: | :---: |\n"
        f"| **Crux Clarity** (0–2) | {b_rub['crux_clarity']} | **{f_rub['crux_clarity']}** |\n"
        f"| **Conditional Commitment** (0–2) | {b_rub['conditional_commitment']} | **{f_rub['conditional_commitment']}** |\n"
        f"| **Tradeoff Specificity** (0–2) | {b_rub['tradeoff_specificity']} | **{f_rub['tradeoff_specificity']}** |\n"
        f"| **Resolution Depth** (0–2) | {b_rub['resolution_depth']} | **{f_rub['resolution_depth']}** |\n"
        f"| **Actionability** (0–2) | {b_rub['actionability']} | **{f_rub['actionability']}** |\n"
        f"| *Generic Hedge Penalty* (0–2) | -{b_rub['generic_hedge']} | -{f_rub['generic_hedge']} |\n"
        f"| *One-Sided Collapse Penalty* (0–2) | -{b_rub['one_sided_collapse']} | -{f_rub['one_sided_collapse']} |\n"
        f"| **Net Score (-4 to 10)** | **{b_agg['total']}** | **{f_agg['total']}** |"
    )

    base_md = f"#### Base `Qwen/Qwen3-8B` (`{b_agg['total']}/10`)\n\n{item['base_response']}"
    ft_md = f"#### Fine-Tuned `hikewa/dialectic-qwen3-8b-lora` (`{f_agg['total']}/10`)\n\n{item['adapter_response']}"
    return header_md, base_md, ft_md


# ---------------------------------------------------------------------------
# Tab 2: v9 Self-Verifying <think> + Cross-Family Council Audit
# ---------------------------------------------------------------------------

COUNCIL_CHOICES = [
    f"[{t['trace_id']}] ({t['quality_gate'].upper()}) {t['question'][:85]}..."
    for t in COUNCIL_TRACES
]


def render_council_trace(choice: str) -> Tuple[str, str, str, str]:
    if not COUNCIL_TRACES:
        return "No council traces found.", "", "", ""
    idx = COUNCIL_CHOICES.index(choice) if choice in COUNCIL_CHOICES else 0
    rec = COUNCIL_TRACES[idx]
    think, visible = extract_think_and_visible(rec["response"])

    gate_badge = "PASS (Accepted into Training Pool)" if rec["quality_gate"] == "pass" else "REJECT (Vetoed by Adversarial Council)"
    summary_md = (
        f"### Question: *{rec['question']}*\n"
        f"**Final Pipeline Verdict:** `{gate_badge}`\n\n"
        f"- **Stage 1 Falsifiability Composite:** `{rec['falsifiability_composite']:.3f}`\n"
        f"- **Stage 2 Empirical Fabrication Check:** `{'0 fabricated claims' if not rec['any_fabrication_detected'] else 'Fabrication flagged'}` (`verification_pass={rec['verification_pass']}`)\n"
        f"- **Stage 3 Adversarial Challenge Survival:** `{rec['adversarial_survives']}`\n"
        f"- **Stage 4 `<think>` Deliberation Quality:** `{rec['thinking_composite']:.3f}` (`has_genuine_thinking={rec['has_genuine_thinking']}`)"
    )

    think_md = f"#### `<think>` Deliberation Trace\n\n> {think}"
    vis_md = f"#### Visible Resolution\n\n{visible}"

    rep = rec.get("council_report", {})
    council_lines = ["#### Cross-Family Council Verdicts (`openai` + `google`)\n"]
    for v in rep.get("verifications", []):
        council_lines.append(
            f"- **Stage 2 Verification (`{v['reviewer_id']}`):** "
            f"`resolves_tension={v['resolves_tension']}`, "
            f"`has_non_obvious_pivot={v['has_non_obvious_pivot']}`, "
            f"`is_topic_specific={v['is_topic_specific']}`\n"
            f"  - *Summary:* {v['summary']}"
        )
    for c in rep.get("challenges", []):
        council_lines.append(
            f"- **Stage 3 Adversarial Challenge (`{c['reviewer_id']}`):** "
            f"`survives_challenge={c['survives_challenge']}`\n"
            f"  - *Weakest Point:* {c['weakest_point']}\n"
            f"  - *Strongest Counterargument:* {c['strongest_counterargument']}\n"
            f"  - *Rationale:* {c['survival_rationale']}"
        )

    return summary_md, think_md, vis_md, "\n".join(council_lines)


# ---------------------------------------------------------------------------
# Tab 3: Gold-Standard Calibration & Live Scorer Sandbox
# ---------------------------------------------------------------------------

GOLDSET_CHOICES = [
    f"[{s['sample_id']}] ({'PASS' if s.get('expected', {}).get('should_pass_training_gate') else 'FAIL'}) {s['question']}"
    for s in GOLDSET
]


def load_goldset_into_sandbox(choice: str) -> Tuple[str, Dict[str, Any]]:
    if not GOLDSET:
        return "", {}
    idx = GOLDSET_CHOICES.index(choice) if choice in GOLDSET_CHOICES else 0
    s = GOLDSET[idx]
    scored = score_trace_interactive(s["response"])
    scored["goldset_expected"] = s.get("expected", {})
    scored["goldset_domain"] = s.get("domain", "")
    return s["response"], scored


# ---------------------------------------------------------------------------
# Gradio UI Assembly
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Dialectic Reasoning Explorer") as demo:
        gr.Markdown(
            "# Dialectic Reasoning Explorer\n"
            "Interactive benchmark and verification inspector for the "
            "[**dialectic-dataset**](https://github.com/CIRWEL/dialectic-dataset) pipeline, "
            "[**hikewa/dialectic-qwen3-8b-lora**](https://huggingface.co/hikewa/dialectic-qwen3-8b-lora), and "
            "[**hikewa/dialectic-reasoning-traces**](https://huggingface.co/datasets/hikewa/dialectic-reasoning-traces)."
        )

        with gr.Tabs():
            with gr.Tab("1. Base vs LoRA Benchmark (14 Held-Out Prompts)"):
                gr.Markdown(
                    "Compare the base `Qwen/Qwen3-8B` against the fine-tuned `hikewa/dialectic-qwen3-8b-lora` "
                    "across all 14 held-out evaluation prompts (8 tradeoff prompts + 6 framing-resistance prompts). "
                    "Mean rubric score improved from **0.1** (base) to **6.6** (LoRA) on a -4 to 10 scale."
                )
                bench_dropdown = gr.Dropdown(
                    choices=BENCHMARK_CHOICES,
                    value=BENCHMARK_CHOICES[0] if BENCHMARK_CHOICES else None,
                    label="Select Held-Out Evaluation Prompt",
                )
                bench_header = gr.Markdown()
                with gr.Row():
                    bench_base = gr.Markdown()
                    bench_ft = gr.Markdown()

                bench_dropdown.change(
                    render_benchmark,
                    inputs=[bench_dropdown],
                    outputs=[bench_header, bench_base, bench_ft],
                )
                demo.load(
                    render_benchmark,
                    inputs=[bench_dropdown],
                    outputs=[bench_header, bench_base, bench_ft],
                )

            with gr.Tab("2. Self-Verifying <think> + Council Audit (v9)"):
                gr.Markdown(
                    "Inspect traces from the **6-stage self-verifying pipeline** (`docs/next_generation_design.md`). "
                    "Instead of rewarding surface rhetorical patterns or unverified statistics, every trace preserves "
                    "its `<think>` deliberation block and undergoes **Stage 2 empirical claim verification** and "
                    "**Stage 3 adversarial challenge** across independent model families."
                )
                council_dropdown = gr.Dropdown(
                    choices=COUNCIL_CHOICES,
                    value=COUNCIL_CHOICES[0] if COUNCIL_CHOICES else None,
                    label="Select Trace from Council Audit Batch",
                )
                council_summary = gr.Markdown()
                with gr.Row():
                    council_think = gr.Markdown()
                    council_vis = gr.Markdown()
                council_verdicts = gr.Markdown()

                council_dropdown.change(
                    render_council_trace,
                    inputs=[council_dropdown],
                    outputs=[council_summary, council_think, council_vis, council_verdicts],
                )
                demo.load(
                    render_council_trace,
                    inputs=[council_dropdown],
                    outputs=[council_summary, council_think, council_vis, council_verdicts],
                )

            with gr.Tab("3. Goldset & Live Scorer Sandbox"):
                gr.Markdown(
                    "Load any of the 12 hand-crafted calibration examples (`goldset_thinking_v1`: 6 genuine "
                    "deliberative traces, 6 negative failure-mode traces) or paste your own `<think>...</think>` "
                    "+ conclusion trace to run the deterministic thinking-quality and falsifiability extractors."
                )
                gold_dropdown = gr.Dropdown(
                    choices=GOLDSET_CHOICES,
                    value=GOLDSET_CHOICES[0] if GOLDSET_CHOICES else None,
                    label="Load Calibration Sample (goldset_thinking_v1)",
                )
                sandbox_input = gr.Textbox(
                    lines=12,
                    label="Reasoning Trace (<think>...</think> + visible resolution)",
                    placeholder="<think>\nAt first glance... Wait—that assumes...\n</think>\n\nIf X, then Y because...",
                )
                score_btn = gr.Button("Score Trace", variant="primary")
                sandbox_output = gr.JSON(label="Stage 1 + Stage 4 Deterministic Score Breakdown")

                gold_dropdown.change(
                    load_goldset_into_sandbox,
                    inputs=[gold_dropdown],
                    outputs=[sandbox_input, sandbox_output],
                )
                score_btn.click(
                    score_trace_interactive,
                    inputs=[sandbox_input],
                    outputs=[sandbox_output],
                )
                demo.load(
                    load_goldset_into_sandbox,
                    inputs=[gold_dropdown],
                    outputs=[sandbox_input, sandbox_output],
                )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(ssr_mode=False)
