"""Prompt families for dialectic reasoning dataset generation.

Eight categories of reasoning problems that require integrative resolution.
Each category contains questions that admit genuinely competing perspectives —
not trick questions, not questions with obvious right answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Prompt:
    """A single prompt for dialectic trace generation."""

    question: str
    category: str
    difficulty: str  # "standard" or "hard"


# ---------------------------------------------------------------------------
# Category 1: Ethical Tradeoffs
# ---------------------------------------------------------------------------

ETHICAL_TRADEOFFS: List[Prompt] = [
    Prompt(
        "A hospital has one ventilator and two patients who need it equally. "
        "One is 25 with no dependents, the other is 45 with three children. "
        "How should the hospital decide?",
        "ethical_tradeoff", "standard",
    ),
    Prompt(
        "Should a company lay off 10% of staff now to avoid probable "
        "bankruptcy that would eliminate all jobs in 18 months?",
        "ethical_tradeoff", "standard",
    ),
    Prompt(
        "A journalist discovers that publishing a true story will likely "
        "cause a wrongly accused person to lose their job before the "
        "correction can be issued. Should they publish?",
        "ethical_tradeoff", "standard",
    ),
    Prompt(
        "Should autonomous vehicles be programmed to minimize total "
        "casualties even if that means occasionally sacrificing the passenger?",
        "ethical_tradeoff", "hard",
    ),
    Prompt(
        "A doctor believes a patient's family is making a harmful medical "
        "decision for a competent adult patient who agrees with the family. "
        "Should the doctor intervene?",
        "ethical_tradeoff", "standard",
    ),
    Prompt(
        "Should a nonprofit accept a large donation from a corporation "
        "whose practices conflict with the nonprofit's mission?",
        "ethical_tradeoff", "standard",
    ),
    Prompt(
        "A teacher discovers a student plagiarized their final essay. The "
        "student is undocumented and expulsion would trigger deportation "
        "proceedings. What should the teacher do?",
        "ethical_tradeoff", "hard",
    ),
]


# ---------------------------------------------------------------------------
# Category 2: Design Tradeoffs
# ---------------------------------------------------------------------------

DESIGN_TRADEOFFS: List[Prompt] = [
    Prompt(
        "Should startups optimize for speed of iteration or code quality?",
        "design_tradeoff", "standard",
    ),
    Prompt(
        "When designing a database schema, should you normalize fully or "
        "denormalize for read performance?",
        "design_tradeoff", "standard",
    ),
    Prompt(
        "Should a team adopt a new framework that's technically superior "
        "but that nobody on the team has production experience with?",
        "design_tradeoff", "standard",
    ),
    Prompt(
        "Is it better to build a custom solution that fits your needs "
        "exactly or adopt an established open-source tool with tradeoffs?",
        "design_tradeoff", "standard",
    ),
    Prompt(
        "Should an API be designed for ease of use by beginners or power "
        "and flexibility for experts?",
        "design_tradeoff", "standard",
    ),
    Prompt(
        "When two modules need to communicate, should you couple them "
        "directly for simplicity or add a message queue for resilience?",
        "design_tradeoff", "hard",
    ),
    Prompt(
        "Should you write comprehensive tests before shipping a prototype "
        "that needs user feedback to validate the core assumption?",
        "design_tradeoff", "standard",
    ),
]


# ---------------------------------------------------------------------------
# Category 3: Safety vs Capability
# ---------------------------------------------------------------------------

SAFETY_VS_CAPABILITY: List[Prompt] = [
    Prompt(
        "Should AI systems be released publicly for broad benefit even if "
        "misuse is likely, or restricted to reduce risk at the cost of "
        "slower beneficial progress?",
        "safety_vs_capability", "standard",
    ),
    Prompt(
        "A new drug shows strong efficacy in trials but has a rare severe "
        "side effect. The condition it treats is debilitating but not fatal. "
        "Should it be approved?",
        "safety_vs_capability", "standard",
    ),
    Prompt(
        "Should children be allowed to use social media if it benefits "
        "their social development but carries mental health risks?",
        "safety_vs_capability", "standard",
    ),
    Prompt(
        "Should a city allow experimental autonomous bus routes that would "
        "improve transit access but haven't completed full safety "
        "certification?",
        "safety_vs_capability", "standard",
    ),
    Prompt(
        "Should genetic editing be permitted for disease prevention if "
        "the same technology enables cosmetic enhancement?",
        "safety_vs_capability", "hard",
    ),
    Prompt(
        "Should security researchers publicly disclose zero-day "
        "vulnerabilities to pressure vendors into patching, or report "
        "privately and wait?",
        "safety_vs_capability", "standard",
    ),
]


# ---------------------------------------------------------------------------
# Category 4: Conflicting Evidence
# ---------------------------------------------------------------------------

CONFLICTING_EVIDENCE: List[Prompt] = [
    Prompt(
        "Two well-designed studies on remote work reach opposite "
        "conclusions: one finds productivity increases, the other finds "
        "it decreases. How should a company setting policy interpret this?",
        "conflicting_evidence", "standard",
    ),
    Prompt(
        "Nutritional research shows both that moderate coffee consumption "
        "reduces cardiovascular risk and that it increases anxiety "
        "disorders. What should a health-conscious person do?",
        "conflicting_evidence", "standard",
    ),
    Prompt(
        "Historical evidence shows that economic sanctions sometimes "
        "accelerate regime change and sometimes entrench authoritarian "
        "governments. Should sanctions be applied to a specific new case?",
        "conflicting_evidence", "hard",
    ),
    Prompt(
        "Some studies show that homework improves student outcomes while "
        "others show it increases stress without measurable learning gains. "
        "Should a school district require homework?",
        "conflicting_evidence", "standard",
    ),
    Prompt(
        "Research on open-plan offices shows improved collaboration but "
        "decreased deep work productivity. Should a growing company adopt "
        "an open floor plan?",
        "conflicting_evidence", "standard",
    ),
    Prompt(
        "Clinical evidence shows a therapy is effective for 70% of patients "
        "but harmful for 10%, with no reliable way to predict who falls "
        "into which group. Should it be the default treatment?",
        "conflicting_evidence", "hard",
    ),
]


# ---------------------------------------------------------------------------
# Category 5: Policy Ambiguity
# ---------------------------------------------------------------------------

POLICY_AMBIGUITY: List[Prompt] = [
    Prompt(
        "Should cities implement congestion pricing that reduces traffic "
        "but disproportionately affects lower-income commuters?",
        "policy_ambiguity", "standard",
    ),
    Prompt(
        "Should a university adopt need-blind admissions if doing so would "
        "reduce the financial aid available per admitted student?",
        "policy_ambiguity", "standard",
    ),
    Prompt(
        "Should a government subsidize fossil fuel workers' transition to "
        "renewable energy jobs if the transition timeline is uncertain and "
        "the subsidies reduce funding for other social programs?",
        "policy_ambiguity", "hard",
    ),
    Prompt(
        "Should a city rezone single-family neighborhoods for higher "
        "density housing to address a shortage, against existing "
        "residents' objections?",
        "policy_ambiguity", "standard",
    ),
    Prompt(
        "Should a school district implement a lottery system for magnet "
        "school admissions instead of merit-based selection?",
        "policy_ambiguity", "standard",
    ),
    Prompt(
        "Should a country accept climate refugees before establishing "
        "international frameworks for burden-sharing?",
        "policy_ambiguity", "hard",
    ),
]


# ---------------------------------------------------------------------------
# Category 6: Strategy Under Constraints
# ---------------------------------------------------------------------------

STRATEGY_UNDER_CONSTRAINTS: List[Prompt] = [
    Prompt(
        "You have budget to either hire two junior engineers or one senior "
        "engineer. Your team is mid-sized with a complex legacy codebase "
        "and an aggressive roadmap. Which do you choose?",
        "strategy_under_constraints", "standard",
    ),
    Prompt(
        "A product has strong retention but weak acquisition. Should the "
        "team focus on growth marketing or continue improving the product "
        "for existing users?",
        "strategy_under_constraints", "standard",
    ),
    Prompt(
        "You can invest in making your current product 30% better or "
        "building a new product that addresses a gap in the market. "
        "Resources allow only one. Which do you pick?",
        "strategy_under_constraints", "standard",
    ),
    Prompt(
        "A small country must choose between investing in education "
        "(30-year payoff) or infrastructure (5-year payoff). The current "
        "government faces re-election in 4 years. What should they "
        "prioritize?",
        "strategy_under_constraints", "hard",
    ),
    Prompt(
        "Your competitor just launched a feature your users are requesting. "
        "Do you build a quick version now or take three months to build a "
        "differentiated version?",
        "strategy_under_constraints", "standard",
    ),
    Prompt(
        "A nonprofit can serve 1000 people with moderate impact or 100 "
        "people with transformative impact. Same budget. Which approach "
        "is better?",
        "strategy_under_constraints", "standard",
    ),
]


# ---------------------------------------------------------------------------
# Category 7: Short Practical Judgment
# ---------------------------------------------------------------------------

SHORT_PRACTICAL_JUDGMENT: List[Prompt] = [
    Prompt(
        "Should you tell a friend their business idea is flawed if they've "
        "already quit their job to pursue it?",
        "short_practical_judgment", "standard",
    ),
    Prompt(
        "Is it better to ship a feature with a known minor bug or delay "
        "the release to fix it?",
        "short_practical_judgment", "standard",
    ),
    Prompt(
        "Should you accept a promotion that increases pay but moves you "
        "away from the work you're best at?",
        "short_practical_judgment", "standard",
    ),
    Prompt(
        "A colleague takes credit for your idea in a meeting. Do you "
        "correct them publicly or address it privately afterward?",
        "short_practical_judgment", "standard",
    ),
    Prompt(
        "Should you negotiate a job offer aggressively when you have no "
        "alternative offers?",
        "short_practical_judgment", "standard",
    ),
    Prompt(
        "Your team disagrees on an approach and you're the tiebreaker. "
        "Both options are viable. Do you pick the one you believe in or "
        "the one with more team support?",
        "short_practical_judgment", "standard",
    ),
    Prompt(
        "Should you give honest negative feedback on a peer's code review "
        "when they're having a bad week and you know it?",
        "short_practical_judgment", "standard",
    ),
]


# ---------------------------------------------------------------------------
# Category 8: Reflective Synthesis
# ---------------------------------------------------------------------------

REFLECTIVE_SYNTHESIS: List[Prompt] = [
    Prompt(
        "Is consistency a virtue or a trap? When should you change your "
        "mind versus hold firm?",
        "reflective_synthesis", "standard",
    ),
    Prompt(
        "Does measuring something always change it? Consider cases where "
        "measurement helps versus where it distorts.",
        "reflective_synthesis", "standard",
    ),
    Prompt(
        "Is specialization or generalization a better strategy for an "
        "individual's career in a rapidly changing economy?",
        "reflective_synthesis", "standard",
    ),
    Prompt(
        "Does transparency always build trust, or are there cases where "
        "it erodes it?",
        "reflective_synthesis", "standard",
    ),
    Prompt(
        "Is it better to optimize for avoiding mistakes or for maximizing "
        "upside? When does each strategy dominate?",
        "reflective_synthesis", "hard",
    ),
    Prompt(
        "Should institutions prioritize stability or adaptability? What "
        "determines which matters more?",
        "reflective_synthesis", "standard",
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROMPT_FAMILIES: Dict[str, List[Prompt]] = {
    "ethical_tradeoff": ETHICAL_TRADEOFFS,
    "design_tradeoff": DESIGN_TRADEOFFS,
    "safety_vs_capability": SAFETY_VS_CAPABILITY,
    "conflicting_evidence": CONFLICTING_EVIDENCE,
    "policy_ambiguity": POLICY_AMBIGUITY,
    "strategy_under_constraints": STRATEGY_UNDER_CONSTRAINTS,
    "short_practical_judgment": SHORT_PRACTICAL_JUDGMENT,
    "reflective_synthesis": REFLECTIVE_SYNTHESIS,
}

ALL_PROMPTS: List[Prompt] = []
for family in PROMPT_FAMILIES.values():
    ALL_PROMPTS.extend(family)


def get_prompts(
    categories: List[str] | None = None,
    difficulty: str | None = None,
) -> List[Prompt]:
    """Return prompts filtered by category and/or difficulty.

    Parameters
    ----------
    categories:
        List of category names to include. None returns all.
    difficulty:
        "standard" or "hard". None returns both.

    Returns
    -------
    Filtered list of Prompt instances.
    """
    prompts = ALL_PROMPTS

    if categories is not None:
        valid = set(categories)
        prompts = [p for p in prompts if p.category in valid]

    if difficulty is not None:
        prompts = [p for p in prompts if p.difficulty == difficulty]

    return prompts
