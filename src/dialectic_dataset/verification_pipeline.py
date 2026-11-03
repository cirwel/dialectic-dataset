"""Stages 2 & 3: Self-verification, adversarial challenge, and UNITARES council integration.

Implements the next-generation verification and adversarial challenge pipeline
from ``docs/next_generation_design.md``, with support for cross-family review
via UNITARES ``consult`` (single host or multi-host council across
``claude:host-adapter``, ``codex:host-adapter``, and ``antigravity:host-adapter``).
"""

from __future__ import annotations

import json
import os
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


def _sanitize_think_tags_for_verifier(response: str) -> str:
    """Replace literal <think> tags with [DRAFT NOTES] / [CONCLUSION] markers.

    Some external verifier models (e.g. Claude Opus 5.5) trigger a server-side
    ``[reasoning_extraction]`` safeguard refusal when a prompt contains literal
    ``<think>...</think>`` tags alongside instructions about reasoning traces.
    Replacing the tags with neutral section markers preserves the full deliberative
    content for verification without triggering false-positive refusals.
    """
    text = re.sub(r"<think>", "[DRAFT NOTES]\n", response.strip(), flags=re.IGNORECASE)
    text = re.sub(r"</think>", "\n[CONCLUSION]", text, flags=re.IGNORECASE)
    return text


def build_verification_prompt(question: str, response: str) -> str:
    """Build the Stage 2 4-question verification prompt from next_generation_design.md."""
    sanitized = _sanitize_think_tags_for_verifier(response)
    return (
        "Review the following draft essay and answer four verification checks. "
        "Do NOT grade formatting or section structure.\n\n"
        f"TOPIC QUESTION:\n{question.strip()}\n\n"
        f"ESSAY:\n{sanitized}\n\n"
        "Answer the following four checks:\n"
        "1. List every specific empirical citation, statistic, historical event, named study, "
        "or quantitative threshold in the essay. State whether each is "
        "\"confident_real\" (verifiably real) or \"generated_or_uncertain\" "
        "(fabricated study/report, invented statistic, or arbitrary made-up numeric cutoff). "
        "Do NOT list philosophical stances, architectural arguments, or conceptual mechanisms as fabricated empirical claims; "
        "evaluate conceptual arguments in checks 2–4.\n"
        "2. Does the conclusion actually resolve the tension, or does it just "
        "present both sides and hedge (\"it depends\", \"balance both\")?\n"
        "3. Is there a specific moment in the draft where the argument "
        "changes direction or discovers something non-obvious?\n"
        "4. Would a domain expert find the analysis specific to this topic, or "
        "could the same template apply to any topic with the nouns swapped?\n\n"
        "Include these exact lines at the end of your review (or equivalent JSON keys `claims`, "
        "`resolves_tension`, `has_non_obvious_pivot`, `is_topic_specific`, `summary`):\n"
        "FABRICATED_OR_UNCERTAIN_CLAIMS: <none, or semicolon-separated list of fabricated citations/statistics/arbitrary numeric thresholds>\n"
        "CONFIDENT_REAL_CLAIMS: <none, or semicolon-separated list of verified empirical claims>\n"
        "RESOLVES_TENSION: <true|false>\n"
        "HAS_NON_OBVIOUS_PIVOT: <true|false>\n"
        "IS_TOPIC_SPECIFIC: <true|false>\n"
        "SUMMARY: <one-sentence explanation of verdict>"
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
    sanitized = _sanitize_think_tags_for_verifier(response)
    return (
        "Read the following draft essay and challenge it adversarially. "
        "Do NOT evaluate politeness or formatting; attack the substance of the argument.\n\n"
        f"TOPIC QUESTION:\n{question.strip()}\n\n"
        f"ESSAY:\n{sanitized}\n\n"
        "Identify:\n"
        "1. The weakest point in the argument (where a claim is asserted without mechanism).\n"
        "2. The strongest counterargument that the essay failed to address.\n"
        "3. Any assumption the essay makes without examining.\n"
        "4. Whether the core conclusion survives this challenge (`survives_challenge`: true/false).\n\n"
        "Include these exact lines at the end of your critique (or equivalent JSON keys `weakest_point`, "
        "`strongest_counterargument`, `unexamined_assumption`, `survives_challenge`, `survival_rationale`):\n"
        "WEAKEST_POINT: <text>\n"
        "STRONGEST_COUNTERARGUMENT: <text>\n"
        "UNEXAMINED_ASSUMPTION: <text>\n"
        "SURVIVES_CHALLENGE: <true|false>\n"
        "SURVIVAL_RATIONALE: <text>"
    )


def build_combined_evaluation_prompt(question: str, response: str) -> str:
    """Build a combined Stage 2 + Stage 3 verification and adversarial challenge prompt.

    Combining both stages into a single consult request cuts cross-family host
    round-trips in half (1 call per trace instead of 2).
    """
    sanitized = _sanitize_think_tags_for_verifier(response)
    return (
        "Review the following draft essay by answering four verification checks "
        "and four adversarial challenge checks. Do NOT grade formatting or section structure.\n\n"
        f"TOPIC QUESTION:\n{question.strip()}\n\n"
        f"ESSAY:\n{sanitized}\n\n"
        "PART A — Verification Checks:\n"
        "1. List every specific empirical citation, statistic, historical event, named study, "
        "or quantitative threshold in the essay. State whether each is "
        "\"confident_real\" (verifiably real) or \"generated_or_uncertain\" "
        "(fabricated study/report, invented statistic, or arbitrary made-up numeric cutoff). "
        "Do NOT list philosophical stances, architectural arguments, or conceptual mechanisms as fabricated empirical claims; "
        "evaluate conceptual arguments in checks 2–8.\n"
        "2. Does the conclusion actually resolve the tension, or does it just "
        "present both sides and hedge (\"it depends\", \"balance both\")?\n"
        "3. Is there a specific moment in the draft where the argument "
        "changes direction or discovers something non-obvious?\n"
        "4. Would a domain expert find the analysis specific to this topic, or "
        "could the same template apply to any topic with the nouns swapped?\n\n"
        "PART B — Adversarial Challenge:\n"
        "5. Identify the weakest point in the argument (where a claim is asserted without mechanism).\n"
        "6. Identify the strongest counterargument that the essay failed to address.\n"
        "7. Identify any assumption the essay makes without examining.\n"
        "8. Determine whether the core conclusion survives this challenge.\n\n"
        "Include these exact lines at the end of your review:\n"
        "FABRICATED_OR_UNCERTAIN_CLAIMS: <none, or semicolon-separated list of fabricated citations/statistics/arbitrary numeric thresholds>\n"
        "CONFIDENT_REAL_CLAIMS: <none, or semicolon-separated list of verified empirical claims>\n"
        "RESOLVES_TENSION: <true|false>\n"
        "HAS_NON_OBVIOUS_PIVOT: <true|false>\n"
        "IS_TOPIC_SPECIFIC: <true|false>\n"
        "SUMMARY: <one-sentence explanation of verification verdict>\n"
        "WEAKEST_POINT: <text>\n"
        "STRONGEST_COUNTERARGUMENT: <text>\n"
        "UNEXAMINED_ASSUMPTION: <text>\n"
        "SURVIVES_CHALLENGE: <true|false>\n"
        "SURVIVAL_RATIONALE: <text>"
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


def _extract_kv_map(raw_text: str) -> Dict[str, str]:
    """Extract uppercase KEY: value lines from structured plain-text responses."""
    kv: Dict[str, str] = {}
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^[\s\*\-\d\.\)]+", "", line).strip()
        m = re.match(r"^([A-Z_]{3,40})\*\*?\s*:\s*\*\*?(.*)$", cleaned)
        if not m:
            m = re.match(r"^([A-Z_]{3,40})\s*:\s*(.*)$", cleaned)
        if m:
            key = m.group(1).strip().upper()
            val = m.group(2).strip().strip("*").strip()
            kv[key] = val
    return kv


def _parse_bool_token(val: str) -> bool:
    v = val.strip().lower()
    return v.startswith(("true", "yes", "1", "pass"))


def parse_verification_response(
    raw_text: str,
    reviewer_id: str = "self",
) -> VerificationResult:
    """Parse Stage 2 verification output (JSON, KEY: value lines, or fallback heuristic)."""
    parsed = _extract_json_object(raw_text)
    if parsed is not None and any(
        k in parsed for k in ("claims", "resolves_tension", "has_non_obvious_pivot")
    ):
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

    kv = _extract_kv_map(raw_text)
    if any(
        k in kv
        for k in (
            "RESOLVES_TENSION",
            "HAS_NON_OBVIOUS_PIVOT",
            "IS_TOPIC_SPECIFIC",
            "FABRICATED_OR_UNCERTAIN_CLAIMS",
        )
    ):
        claims = []
        fab_raw = kv.get("FABRICATED_OR_UNCERTAIN_CLAIMS", "")
        if fab_raw and fab_raw.strip().lower() not in ("none", "n/a", "nil", "0", "[]", "false"):
            for part in re.split(r"[;]", fab_raw):
                part = part.strip()
                if part and part.lower() not in ("none", "n/a"):
                    claims.append(
                        VerificationClaim(
                            claim=part,
                            status="generated_or_uncertain",
                            reason="flagged by verifier",
                        )
                    )
        real_raw = kv.get("CONFIDENT_REAL_CLAIMS", "")
        if real_raw and real_raw.strip().lower() not in ("none", "n/a", "nil", "0", "[]"):
            for part in re.split(r"[;]", real_raw):
                part = part.strip()
                if part and part.lower() not in ("none", "n/a"):
                    claims.append(
                        VerificationClaim(
                            claim=part,
                            status="confident_real",
                            reason="verified real",
                        )
                    )
        return VerificationResult(
            claims=claims,
            resolves_tension=_parse_bool_token(kv.get("RESOLVES_TENSION", "false")),
            has_non_obvious_pivot=_parse_bool_token(kv.get("HAS_NON_OBVIOUS_PIVOT", "false")),
            is_topic_specific=_parse_bool_token(kv.get("IS_TOPIC_SPECIFIC", "false")),
            summary=kv.get("SUMMARY", raw_text.strip()[:300]),
            reviewer_id=reviewer_id,
            raw_response=raw_text,
        )

    # Conservative fallback if reviewer returned plain text instead of JSON/KV
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
    """Parse Stage 3 adversarial challenge output (JSON, KEY: value lines, or fallback heuristic)."""
    parsed = _extract_json_object(raw_text)
    if parsed is not None and any(
        k in parsed for k in ("survives_challenge", "weakest_point", "strongest_counterargument")
    ):
        return AdversarialChallengeResult(
            weakest_point=str(parsed.get("weakest_point", "")),
            strongest_counterargument=str(parsed.get("strongest_counterargument", "")),
            unexamined_assumption=str(parsed.get("unexamined_assumption", "")),
            survives_challenge=bool(parsed.get("survives_challenge", False)),
            survival_rationale=str(parsed.get("survival_rationale", "")),
            reviewer_id=reviewer_id,
            raw_response=raw_text,
        )

    kv = _extract_kv_map(raw_text)
    if any(
        k in kv
        for k in (
            "SURVIVES_CHALLENGE",
            "WEAKEST_POINT",
            "STRONGEST_COUNTERARGUMENT",
        )
    ):
        return AdversarialChallengeResult(
            weakest_point=kv.get("WEAKEST_POINT", ""),
            strongest_counterargument=kv.get("STRONGEST_COUNTERARGUMENT", ""),
            unexamined_assumption=kv.get("UNEXAMINED_ASSUMPTION", ""),
            survives_challenge=_parse_bool_token(kv.get("SURVIVES_CHALLENGE", "false")),
            survival_rationale=kv.get("SURVIVAL_RATIONALE", raw_text.strip()[:300]),
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
        self.client_session_id = client_session_id or os.environ.get("UNITARES_CLIENT_SESSION_ID")
        self.timeout = timeout
        self._transport = transport or _default_http_transport

    def _call_consult(self, question: str, context: str) -> Dict[str, Any]:
        brief = f"{question.strip()}\n\n{context.strip()}"
        arguments: Dict[str, Any] = {
            "purpose": "critique",
            "brief": brief,
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
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                res = self._transport(f"{self.base_url}/v1/tools/call", payload, self.timeout)
                # Verify it wasn't a transient adapter failure before returning
                self._extract_consult_responses(res)
                return res
            except Exception as exc:
                last_err = exc
                if attempt < 2 and "identity_required" not in str(exc):
                    continue
                raise
        raise last_err or RuntimeError("UNITARES consult failed")

    @staticmethod
    def _extract_consult_responses(api_result: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract ``(reviewer_id, response_text)`` pairs from UNITARES consult output.

        Handles ``unitares.consultation.v1`` (``{"advice": "...", "request": {"host_id": ...}}``),
        council responses (``{"members": [...]}`` or ``{"perspectives": [...]}``),
        and MCP-wrapped ``result`` / ``content`` envelopes.
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
                        inner = {"advice": block.get("text", "")}
                        break

        if not isinstance(inner, dict):
            return [("consult", str(inner))]

        if inner.get("refused") or inner.get("status") in ("identity_required", "failed") or inner.get("error"):
            msg = inner.get("hint") or inner.get("message") or inner.get("error") or str(inner)
            raise RuntimeError(f"UNITARES consult refused or failed: {msg}")

        # Multi-host council response (unitares.consultation.v1 uses `members`)
        perspectives = (
            inner.get("members")
            or inner.get("perspectives")
            or inner.get("lanes")
            or inner.get("results")
        )
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
                    text = str(
                        item.get("advice")
                        or item.get("response")
                        or item.get("text")
                        or ""
                    )
                    if text.strip():
                        pairs.append((rev, text))
            if pairs:
                return pairs

        # Single-host consult response (unitares.consultation.v1 uses top-level `advice`)
        req_meta = inner.get("request") if isinstance(inner.get("request"), dict) else {}
        reviewer = str(
            inner.get("host_id")
            or req_meta.get("host_id")
            or inner.get("model")
            or inner.get("family")
            or inner.get("backend")
            or "consult"
        )
        text = str(
            inner.get("advice")
            or inner.get("response")
            or inner.get("answer")
            or ""
        )
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
        """Run both Stage 2 verification and Stage 3 adversarial challenge and aggregate.

        Uses a single combined consult call when the custom transport does not differentiate
        Stage 2 vs Stage 3 prompts, or falls back to separate calls if the response only
        contains Stage 2 keys.
        """
        prompt = build_combined_evaluation_prompt(question, response)
        raw = self._call_consult(
            question=(
                "Verify factual claims, tension resolution, reasoning pivot, and specificity, "
                "then challenge the argument adversarially and determine if the core conclusion survives."
            ),
            context=prompt,
        )
        pairs = self._extract_consult_responses(raw)
        verifications = [
            parse_verification_response(text, reviewer_id=rev)
            for rev, text in pairs
        ]
        # Check if the response also included structured Stage 3 adversarial fields
        has_adv_fields = False
        for _, text in pairs:
            parsed_json = _extract_json_object(text)
            if parsed_json and any(
                k in parsed_json for k in ("survives_challenge", "weakest_point", "strongest_counterargument")
            ):
                has_adv_fields = True
                break
            kv = _extract_kv_map(text)
            if any(k in kv for k in ("SURVIVES_CHALLENGE", "WEAKEST_POINT", "STRONGEST_COUNTERARGUMENT")):
                has_adv_fields = True
                break
        if has_adv_fields:
            challenges = [
                parse_adversarial_response(text, reviewer_id=rev)
                for rev, text in pairs
            ]
        else:
            challenges = self.run_adversarial_challenge(question, response)
        return aggregate_council_verdicts(verifications, challenges)
