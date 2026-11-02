"""Tests for Stages 2 & 3 verification pipeline, UNITARES council aggregation, and Stage 5 training selection."""

import json
import sys
from pathlib import Path

from dialectic_dataset.verification_pipeline import (
    AdversarialChallengeResult,
    UnitaresConsultBackend,
    VerificationClaim,
    VerificationResult,
    aggregate_council_verdicts,
    build_adversarial_prompt,
    build_verification_prompt,
    parse_adversarial_response,
    parse_verification_response,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_training import (  # noqa: E402
    compute_selection_score,
    passes_verification_gates,
    trace_to_training,
)
from verify_traces import enrich_trace_record  # noqa: E402


class TestStage2VerificationParsing:
    def test_prompts_contain_four_questions_and_schema(self):
        prompt = build_verification_prompt(
            "Should we shard early?",
            "<think>Wait, let me reconsider.</think>\nWait until write contention.",
        )
        assert "confident_real" in prompt
        assert "generated_or_uncertain" in prompt
        assert "RESOLVES_TENSION" in prompt
        assert "HAS_NON_OBVIOUS_PIVOT" in prompt
        assert "IS_TOPIC_SPECIFIC" in prompt
        assert "[DRAFT NOTES]" in prompt
        assert "<think>" not in prompt

    def test_parses_structured_kv_verification_from_claude_adapter(self):
        raw = (
            "Advisory critique:\n"
            "1. The 500 ms threshold is unsupported.\n\n"
            "FABRICATED_OR_UNCERTAIN_CLAIMS: a 500 ms latency budget is the threshold for ACID; summarization reduces variance\n"
            "CONFIDENT_REAL_CLAIMS: ACID guarantees transactional state updates\n"
            "RESOLVES_TENSION: false\n"
            "HAS_NON_OBVIOUS_PIVOT: true\n"
            "IS_TOPIC_SPECIFIC: true\n"
            "SUMMARY: Strong pivot in notes, but conclusion invents a 500 ms threshold."
        )
        res = parse_verification_response(raw, reviewer_id="claude:host-adapter")
        assert res.fabricated_claims_count == 2
        assert len(res.claims) == 3
        assert not res.resolves_tension
        assert res.has_non_obvious_pivot
        assert res.is_topic_specific
        assert not res.passed

    def test_parses_clean_json_verification(self):
        payload = {
            "claims": [
                {
                    "claim": "CAP theorem limits availability under partition",
                    "status": "confident_real",
                    "reason": "Standard distributed systems theorem",
                }
            ],
            "resolves_tension": True,
            "has_non_obvious_pivot": True,
            "is_topic_specific": True,
            "summary": "Grounded and specific",
        }
        res = parse_verification_response(f"```json\n{json.dumps(payload)}\n```", reviewer_id="claude")
        assert res.reviewer_id == "claude"
        assert res.fabricated_claims_count == 0
        assert res.passed

    def test_fabricated_claim_fails_verification(self):
        payload = {
            "claims": [
                {
                    "claim": "2023 Stanford study of 412 microservices",
                    "status": "generated_or_uncertain",
                    "reason": "No such study exists",
                }
            ],
            "resolves_tension": True,
            "has_non_obvious_pivot": True,
            "is_topic_specific": True,
            "summary": "Contains fabricated citation",
        }
        res = parse_verification_response(json.dumps(payload), reviewer_id="codex")
        assert res.fabricated_claims_count == 1
        assert not res.passed


class TestStage3AdversarialParsing:
    def test_parses_adversarial_json(self):
        prompt = build_adversarial_prompt("Monolith vs microservices?", "Use modular monolith.")
        assert "survives_challenge" in prompt

        payload = {
            "weakest_point": "Assumes module boundaries are enforced in CI",
            "strongest_counterargument": "Independent scaling of hot paths",
            "unexamined_assumption": "Single deployment artifact fits memory",
            "survives_challenge": True,
            "survival_rationale": "Core mechanism holds for teams under 30 engineers.",
        }
        res = parse_adversarial_response(json.dumps(payload), reviewer_id="claude")
        assert res.survives_challenge
        assert "CI" in res.weakest_point


class TestCouncilAggregation:
    def test_single_reviewer_fabrication_vetoes_council(self):
        v_clean = VerificationResult(
            claims=[VerificationClaim(claim="Real fact", status="confident_real")],
            resolves_tension=True,
            has_non_obvious_pivot=True,
            is_topic_specific=True,
            reviewer_id="claude:host-adapter",
        )
        v_flagged = VerificationResult(
            claims=[VerificationClaim(claim="Fake 2022 MIT study", status="generated_or_uncertain")],
            resolves_tension=True,
            has_non_obvious_pivot=True,
            is_topic_specific=True,
            reviewer_id="codex:host-adapter",
        )
        c_pass = AdversarialChallengeResult(survives_challenge=True, reviewer_id="claude:host-adapter")

        report = aggregate_council_verdicts([v_clean, v_flagged], [c_pass])
        assert report.any_fabrication_detected
        assert report.total_fabricated_claims == 1
        assert not report.verification_pass
        assert not report.overall_pass

    def test_majority_council_passes_when_no_fabrication(self):
        v1 = VerificationResult(
            claims=[],
            resolves_tension=True,
            has_non_obvious_pivot=True,
            is_topic_specific=True,
            reviewer_id="claude:host-adapter",
        )
        v2 = VerificationResult(
            claims=[],
            resolves_tension=True,
            has_non_obvious_pivot=True,
            is_topic_specific=True,
            reviewer_id="codex:host-adapter",
        )
        c1 = AdversarialChallengeResult(survives_challenge=True, reviewer_id="claude:host-adapter")
        c2 = AdversarialChallengeResult(survives_challenge=False, reviewer_id="codex:host-adapter")
        c3 = AdversarialChallengeResult(survives_challenge=True, reviewer_id="antigravity:host-adapter")

        report = aggregate_council_verdicts([v1, v2], [c1, c2, c3])
        assert not report.any_fabrication_detected
        assert report.verification_pass
        assert report.adversarial_survives
        assert report.overall_pass


class TestUnitaresConsultBackend:
    def test_council_mode_calls_unitares_with_host_ids(self):
        calls = []

        def fake_transport(url, payload, timeout):
            calls.append((url, payload))
            if "Verify factual claims" in payload["arguments"]["brief"]:
                return {
                    "result": {
                        "perspectives": [
                            {
                                "host_id": "claude:host-adapter",
                                "advice": json.dumps(
                                    {
                                        "claims": [],
                                        "resolves_tension": True,
                                        "has_non_obvious_pivot": True,
                                        "is_topic_specific": True,
                                        "summary": "Clean",
                                    }
                                ),
                            },
                            {
                                "host_id": "codex:host-adapter",
                                "advice": json.dumps(
                                    {
                                        "claims": [],
                                        "resolves_tension": True,
                                        "has_non_obvious_pivot": True,
                                        "is_topic_specific": True,
                                        "summary": "Clean",
                                    }
                                ),
                            },
                        ]
                    }
                }
            return {
                "result": {
                    "perspectives": [
                        {
                            "host_id": "claude:host-adapter",
                            "advice": json.dumps(
                                {
                                    "weakest_point": "Minor",
                                    "strongest_counterargument": "Edge case",
                                    "unexamined_assumption": "None critical",
                                    "survives_challenge": True,
                                    "survival_rationale": "Holds up",
                                }
                            ),
                        },
                        {
                            "host_id": "codex:host-adapter",
                            "advice": json.dumps(
                                {
                                    "weakest_point": "Minor",
                                    "strongest_counterargument": "Edge case",
                                    "unexamined_assumption": "None critical",
                                    "survives_challenge": True,
                                    "survival_rationale": "Holds up",
                                }
                            ),
                        },
                    ]
                }
            }

        backend = UnitaresConsultBackend(
            base_url="http://127.0.0.1:8767",
            privacy="cloud_allowed",
            host_ids=["claude:host-adapter", "codex:host-adapter"],
            transport=fake_transport,
        )
        report = backend.evaluate_trace("Question?", "<think>Wait, let me reconsider</think>Answer.")
        assert len(calls) == 2
        assert calls[0][1]["name"] == "consult"
        assert calls[0][1]["arguments"]["host_ids"] == ["claude:host-adapter", "codex:host-adapter"]
        assert report.overall_pass


class TestStage5BuildTrainingIntegration:
    def test_keep_thinking_is_default(self):
        trace = {
            "trace_id": "t1",
            "question": "Q?",
            "response": "<think>Genuine deliberation here.</think>\n\nVisible resolution.",
            "falsifiability_composite": 0.5,
            "thinking_composite": 0.8,
            "selection_score": 0.68,
        }
        example = trace_to_training(trace)
        assert "<think>Genuine deliberation here.</think>" in example["messages"][2]["content"]

        stripped = trace_to_training(trace, keep_thinking=False)
        assert "<think>" not in stripped["messages"][2]["content"]
        assert stripped["messages"][2]["content"] == "Visible resolution."

    def test_selection_score_prefers_genuine_thinking_over_number_stuffing(self):
        # Trace A: high falsifiability (0.75) but empty/stub thinking (0.10)
        score_a = compute_selection_score(
            falsifiability_composite=0.75,
            thinking_composite=0.10,
            has_think_tag=True,
            thinking_weight=0.6,
        )
        # Trace B: moderate falsifiability (0.55) with strong deliberative thinking (0.85)
        score_b = compute_selection_score(
            falsifiability_composite=0.55,
            thinking_composite=0.85,
            has_think_tag=True,
            thinking_weight=0.6,
        )
        assert score_b > score_a

    def test_verification_gates_reject_failed_traces(self):
        assert not passes_verification_gates({"verification_pass": False})
        assert not passes_verification_gates({"adversarial_survives": False})
        assert not passes_verification_gates({"any_fabrication_detected": True})
        assert passes_verification_gates({}, require_verified=False)
        assert not passes_verification_gates({}, require_verified=True)
        assert passes_verification_gates(
            {"verification_pass": True, "adversarial_survives": True},
            require_verified=True,
        )

    def test_enrich_trace_record_offline_flags_fabricated_citation(self):
        trace = {
            "trace_id": "t_fab",
            "question": "Does autonomy improve velocity?",
            "response": (
                "<think>Wait, let me reconsider.</think>\n\n"
                "A 2022 Harvard study found that autonomy increases velocity by 40%."
            ),
        }
        enriched = enrich_trace_record(trace, backend=None)
        assert enriched["fabricated_citations_regex"] >= 1
        assert enriched["any_fabrication_detected"] is True
        assert enriched["quality_gate"] == "reject"

