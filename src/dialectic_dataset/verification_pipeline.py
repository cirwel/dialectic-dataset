"""Stages 2 & 3: Self-verification, adversarial challenge, and UNITARES council integration.

Implements the next-generation verification and adversarial challenge pipeline
from ``docs/next_generation_design.md``, with support for cross-family review
via UNITARES ``consult`` (single host or multi-host council across
``claude:host-adapter``, ``codex:host-adapter``, and ``antigravity:host-adapter``).
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Stage 2: Verification Pass
# ---------------------------------------------------------------------------

@dataclass
class VerificationClaim:
    """A single factual claim extracted during Stage 2 verification."""

    claim: str
    status: str  # "confident_real" | "generated_or_uncertain"
    reason: str = ""

    @property
    def is_fabricated_or_uncertain(self) -> bool:
        return self.status.strip().lower() != "confident_real"


@dataclass
class VerificationResult:
    """Stage 2 structured verification assessment of a generated trace."""

    claims: List[VerificationClaim] = field(default_factory=list)
    resolves_tension: bool = False
    has_non_obvious_pivot: bool = False
    is_topic_specific: bool = False
    summary: str = ""
    reviewer_id: str = "self"
    raw_response: str = ""

    @property
    def fabricated_claims_count(self) -> int:
        return sum(1 for c in self.claims if c.is_fabricated_or_uncertain)

    @property
    def passed(self) -> bool:
        """Pass requires zero fabricated/uncertain claims and all three reasoning checks."""
        return (
            self.fabricated_claims_count == 0
            and self.resolves_tension
            and self.has_non_obvious_pivot
            and self.is_topic_specific
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["fabricated_claims_count"] = self.fabricated_claims_count
        data["passed"] = self.passed
        return data


def build_verification_prompt(question: str, response: str) -> str:
    """Build the Stage 2 4-question verification prompt from next_generation_design.md."""
    return (
        "Review the following dialectic reasoning trace critically and answer "
        "four verification questions. Do NOT grade formatting or section structure.\n\n"
        f"QUESTION:\n{question.strip()}\n\n"
        f"RESPONSE (including <think> if present):\n{response.strip()}\n\n"
        "Answer the following four checks and return ONLY valid JSON matching the schema below:\n"
        "1. List every specific factual claim (studies, statistics, historical events, "
        "named systems, concrete numbers). For each, mark status as either "
        "\"confident_real\" (verifiably real) or \"generated_or_uncertain\" "
        "(invented, composite, or unverified).\n"
        "2. Does the conclusion actually resolve the tension, or does it just "
        "present both sides and hedge (\"it depends\", \"balance both\")?\n"
        "3. Is there a specific moment in the reasoning where the argument "
        "changes direction or discovers something non-obvious?\n"
        "4. Would a domain expert find the analysis specific to this topic, or "
        "could the same template apply to any topic with the nouns swapped?\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "claims": [\n'
        '    {"claim": "...", "status": "confident_real | generated_or_uncertain", "reason": "..."}\n'
        "  ],\n"
        '  "resolves_tension": true,\n'
        '  "has_non_obvious_pivot": true,\n'
        '  "is_topic_specific": true,\n'
        '  "summary": "Brief explanation of verdict"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Stage 3: Adversarial Challenge
# ---------------------------------------------------------------------------

@dataclass
class AdversarialChallengeResult:
    """Stage 3 adversarial critique of a reasoning trace."""

    weakest_point: str = ""
    strongest_counterargument: str = ""
    unexamined_assumption: str = ""
    survives_challenge: bool = False
    survival_rationale: str = ""
    reviewer_id: str = "adversary"
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_adversarial_prompt(question: str, response: str) -> str:
    """Build the Stage 3 adversarial challenge prompt from next_generation_design.md."""
    return (
        "Read the following dialectic analysis and challenge it adversarially. "
        "Do NOT evaluate politeness or formatting; attack the substance of the reasoning.\n\n"
        f"QUESTION:\n{question.strip()}\n\n"
        f"ANALYSIS:\n{response.strip()}\n\n"
        "Identify:\n"
        "1. The weakest point in the reasoning (where a claim is asserted without mechanism).\n"
        "2. The strongest counterargument that the analysis failed to address.\n"
        "3. Any assumption the analysis makes without examining.\n"
        "4. Whether the core conclusion survives this challenge (`survives_challenge`: true/false).\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "weakest_point": "...",\n'
        '  "strongest_counterargument": "...",\n'
        '  "unexamined_assumption": "...",\n'
        '  "survives_challenge": true,\n'
        '  "survival_rationale": "..."\n'
        "}"
    )


# ---------------------------------------------------------------------------
# JSON / Text Response Parsers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from model output."""
    text = raw_text.strip()
    if not text:
        return None

    fence_match = _JSON_FENCE_RE.search(text)
    candidates = [fence_match.group(1)] if fence_match else []
    candidates.append(text)

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_verification_response(
    raw_text: str,
    reviewer_id: str = "self",
) -> VerificationResult:
    """Parse Stage 2 verification output (JSON or fallback heuristic)."""
    parsed = _extract_json_object(raw_text)
    if parsed is not None:
        raw_claims = parsed.get("claims") or []
        claims: List[VerificationClaim] = []
        if isinstance(raw_claims, list):
            for item in raw_claims:
                if isinstance(item, dict):
                    claims.append(
                        VerificationClaim(
                            claim=str(item.get("claim", "")),
                            status=str(item.get("status", "generated_or_uncertain")),
                            reason=str(item.get("reason", "")),
                        )
                    )
        return VerificationResult(
            claims=claims,
            resolves_tension=bool(parsed.get("resolves_tension", False)),
            has_non_obvious_pivot=bool(parsed.get("has_non_obvious_pivot", False)),
            is_topic_specific=bool(parsed.get("is_topic_specific", False)),
            summary=str(parsed.get("summary", "")),
            reviewer_id=reviewer_id,
            raw_response=raw_text,
        )

    # Conservative fallback if reviewer returned plain text instead of JSON
    lower = raw_text.lower()
    has_fabricated = any(
        kw in lower for kw in ("fabricated", "generated_or_uncertain", "invented", "made up")
    )
    claims = (
        [VerificationClaim(claim="unstructured_flag", status="generated_or_uncertain", reason="flagged in text")]
        if has_fabricated
        else []
    )
    resolves = "resolves_tension: true" in lower or (
        "resolves the tension" in lower and "does not resolve" not in lower
    )
    pivot = "has_non_obvious_pivot: true" in lower or "non-obvious" in lower
    specific = "is_topic_specific: true" in lower or (
        "topic-specific" in lower and "generic template" not in lower
    )
    return VerificationResult(
        claims=claims,
        resolves_tension=resolves,
        has_non_obvious_pivot=pivot,
        is_topic_specific=specific,
        summary=raw_text.strip()[:300],
        reviewer_id=reviewer_id,
        raw_response=raw_text,
    )


def parse_adversarial_response(
    raw_text: str,
    reviewer_id: str = "adversary",
) -> AdversarialChallengeResult:
    """Parse Stage 3 adversarial challenge output (JSON or fallback heuristic)."""
    parsed = _extract_json_object(raw_text)
    if parsed is not None:
        return AdversarialChallengeResult(
            weakest_point=str(parsed.get("weakest_point", "")),
            strongest_counterargument=str(parsed.get("strongest_counterargument", "")),
            unexamined_assumption=str(parsed.get("unexamined_assumption", "")),
            survives_challenge=bool(parsed.get("survives_challenge", False)),
            survival_rationale=str(parsed.get("survival_rationale", "")),
            reviewer_id=reviewer_id,
            raw_response=raw_text,
        )

    lower = raw_text.lower()
    survives = (
        "survives_challenge: true" in lower
        or ("core argument survives" in lower and "does not survive" not in lower)
    )
    return AdversarialChallengeResult(
        weakest_point="",
        strongest_counterargument="",
        unexamined_assumption="",
        survives_challenge=survives,
        survival_rationale=raw_text.strip()[:300],
        reviewer_id=reviewer_id,
        raw_response=raw_text,
    )


# ---------------------------------------------------------------------------
# Cross-Family Council Aggregation
# ---------------------------------------------------------------------------

@dataclass
class CouncilVerificationReport:
    """Aggregated Stage 2 + Stage 3 verdict across one or more reviewer families."""

    verifications: List[VerificationResult] = field(default_factory=list)
    challenges: List[AdversarialChallengeResult] = field(default_factory=list)
    any_fabrication_detected: bool = False
    total_fabricated_claims: int = 0
    verification_pass: bool = False
    adversarial_survives: bool = False
    overall_pass: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verifications": [v.to_dict() for v in self.verifications],
            "challenges": [c.to_dict() for c in self.challenges],
            "any_fabrication_detected": self.any_fabrication_detected,
            "total_fabricated_claims": self.total_fabricated_claims,
            "verification_pass": self.verification_pass,
            "adversarial_survives": self.adversarial_survives,
            "overall_pass": self.overall_pass,
        }


def aggregate_council_verdicts(
    verifications: Sequence[VerificationResult],
    challenges: Sequence[AdversarialChallengeResult],
) -> CouncilVerificationReport:
    """Combine verdicts across reviewers with strict fabrication veto and majority survival.

    Rules:
    - Any single reviewer flagging a fabricated/uncertain claim vetoes ``verification_pass``.
    - Majority of verification reviewers must agree ``resolves_tension``,
      ``has_non_obvious_pivot``, and ``is_topic_specific`` are true.
    - Majority of adversarial reviewers must agree ``survives_challenge`` is true.
    """
    ver_list = list(verifications)
    chal_list = list(challenges)

    total_fab = sum(v.fabricated_claims_count for v in ver_list)
    any_fab = total_fab > 0

    if ver_list:
        n_ver = len(ver_list)
        majority_ver = (n_ver // 2) + 1
        resolves_ok = sum(1 for v in ver_list if v.resolves_tension) >= majority_ver
        pivot_ok = sum(1 for v in ver_list if v.has_non_obvious_pivot) >= majority_ver
        specific_ok = sum(1 for v in ver_list if v.is_topic_specific) >= majority_ver
        verification_pass = (not any_fab) and resolves_ok and pivot_ok and specific_ok
    else:
        verification_pass = False

    if chal_list:
        n_chal = len(chal_list)
        majority_chal = (n_chal // 2) + 1
        adversarial_survives = (
            sum(1 for c in chal_list if c.survives_challenge) >= majority_chal
        )
    else:
        adversarial_survives = True

    overall_pass = verification_pass and adversarial_survives

    return CouncilVerificationReport(
        verifications=ver_list,
        challenges=chal_list,
        any_fabrication_detected=any_fab,
        total_fabricated_claims=total_fab,
        verification_pass=verification_pass,
        adversarial_survives=adversarial_survives,
        overall_pass=overall_pass,
    )


# ---------------------------------------------------------------------------
# UNITARES `consult` / Council Backend
# ---------------------------------------------------------------------------

DEFAULT_COUNCIL_HOST_IDS: Tuple[str, ...] = (
    "claude:host-adapter",
    "codex:host-adapter",
    "antigravity:host-adapter",
)

TransportFn = Callable[[str, Dict[str, Any], float], Dict[str, Any]]


def _default_http_transport(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


class UnitaresConsultBackend:
    """Client for UNITARES ``consult`` tool supporting single-host and multi-family council modes."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8767",
        privacy: str = "cloud_allowed",
        host_id: Optional[str] = None,
        host_ids: Optional[Sequence[str]] = None,
        model: Optional[str] = None,
        client_session_id: Optional[str] = None,
        timeout: float = 120.0,
        transport: Optional[TransportFn] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.privacy = privacy
        self.host_id = host_id
        self.host_ids = list(host_ids) if host_ids else None
        self.model = model
        self.client_session_id = client_session_id
        self.timeout = timeout
        self._transport = transport or _default_http_transport

    def _call_consult(self, question: str, context: str) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {
            "purpose": "critique",
            "question": question,
            "context": context,
            "privacy": self.privacy,
        }
        if self.host_ids:
            arguments["host_ids"] = self.host_ids
        elif self.host_id:
            arguments["host_id"] = self.host_id
        if self.model:
            arguments["model"] = self.model
        if self.client_session_id:
            arguments["client_session_id"] = self.client_session_id

        payload = {
            "name": "consult",
            "arguments": arguments,
        }
        return self._transport(f"{self.base_url}/v1/tools/call", payload, self.timeout)

    @staticmethod
    def _extract_consult_responses(api_result: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract ``(reviewer_id, response_text)`` pairs from UNITARES consult output.

        Handles both single-perspective responses (``{"response": "..."}``) and
        council responses (``{"perspectives": [{"host_id": ..., "response": ...}]}``),
        as well as MCP-wrapped ``result`` / ``content`` envelopes.
        """
        inner: Any = api_result
        if isinstance(inner, dict) and "result" in inner:
            inner = inner["result"]
        if isinstance(inner, dict) and "content" in inner and isinstance(inner["content"], list):
            for block in inner["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        inner = json.loads(block["text"])
                        break
                    except (json.JSONDecodeError, KeyError):
                        inner = {"response": block.get("text", "")}
                        break

        if not isinstance(inner, dict):
            return [("consult", str(inner))]

        # Multi-host council response
        perspectives = inner.get("perspectives") or inner.get("results")
        if isinstance(perspectives, list) and perspectives:
            pairs: List[Tuple[str, str]] = []
            for idx, item in enumerate(perspectives):
                if isinstance(item, dict):
                    rev = str(
                        item.get("host_id")
                        or item.get("model")
                        or item.get("backend")
                        or f"council_{idx}"
                    )
                    text = str(item.get("response") or item.get("text") or "")
                    if text.strip():
                        pairs.append((rev, text))
            if pairs:
                return pairs

        # Single-host consult response
        reviewer = str(
            inner.get("host_id")
            or inner.get("model")
            or inner.get("backend")
            or "consult"
        )
        text = str(inner.get("response") or inner.get("answer") or "")
        return [(reviewer, text)]

    def run_verification(self, question: str, response: str) -> List[VerificationResult]:
        """Run Stage 2 verification via UNITARES ``consult``."""
        prompt = build_verification_prompt(question, response)
        raw = self._call_consult(
            question="Verify factual claims, tension resolution, reasoning pivot, and specificity.",
            context=prompt,
        )
        pairs = self._extract_consult_responses(raw)
        return [
            parse_verification_response(text, reviewer_id=rev)
            for rev, text in pairs
        ]

    def run_adversarial_challenge(
        self,
        question: str,
        response: str,
    ) -> List[AdversarialChallengeResult]:
        """Run Stage 3 adversarial challenge via UNITARES ``consult``."""
        prompt = build_adversarial_prompt(question, response)
        raw = self._call_consult(
            question="Challenge this analysis adversarially and determine if the core conclusion survives.",
            context=prompt,
        )
        pairs = self._extract_consult_responses(raw)
        return [
            parse_adversarial_response(text, reviewer_id=rev)
            for rev, text in pairs
        ]

    def evaluate_trace(self, question: str, response: str) -> CouncilVerificationReport:
        """Run both Stage 2 verification and Stage 3 adversarial challenge and aggregate."""
        verifications = self.run_verification(question, response)
        challenges = self.run_adversarial_challenge(question, response)
        return aggregate_council_verdicts(verifications, challenges)
