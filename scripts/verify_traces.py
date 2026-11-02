"""Run Stages 2-4 verification, adversarial challenge, and thinking quality scoring on trace JSONL.

Supports three execution modes:
- ``offline``: deterministic Stage 4 ``<think>`` quality scoring + citation fabrication check only.
- ``consult``: Stage 4 + Stage 2/3 via a single UNITARES ``consult`` backend (local Ollama or single host_id).
- ``council``: Stage 4 + Stage 2/3 across a cross-family UNITARES council (e.g. claude + codex + antigravity).

Usage:
    python scripts/verify_traces.py --input data/traces.jsonl --output data/verified_traces.jsonl --mode offline
    python scripts/verify_traces.py --input data/traces.jsonl --output data/verified_traces.jsonl --mode council
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dialectic_dataset.falsifiability_extractor import (
    _count_fabricated_citations,
    extract_falsifiability_features,
    score_falsifiability,
)
from dialectic_dataset.thinking_quality import (
    extract_think_block,
    score_thinking_quality,
)
from dialectic_dataset.verification_pipeline import (
    DEFAULT_COUNCIL_HOST_IDS,
    UnitaresConsultBackend,
)


def enrich_trace_record(
    trace: Dict[str, Any],
    backend: Optional[UnitaresConsultBackend] = None,
) -> Dict[str, Any]:
    """Score Stage 4 thinking quality and optionally run Stage 2/3 verification via UNITARES."""
    enriched = dict(trace)
    raw_response = str(enriched.get("response", ""))
    question = str(enriched.get("question", ""))

    think_score = score_thinking_quality(raw_response)
    _, visible_response = extract_think_block(raw_response)

    fab_citations = _count_fabricated_citations(raw_response)
    fals_features = extract_falsifiability_features(visible_response)
    fals_score = score_falsifiability(fals_features)

    enriched["thinking_composite"] = think_score.composite
    enriched["has_genuine_thinking"] = think_score.has_genuine_thinking
    enriched["thinking_components"] = think_score.components
    enriched["fabricated_citations_regex"] = fab_citations
    enriched["falsifiability_composite"] = round(fals_score.composite, 4)

    if backend is not None:
        report = backend.evaluate_trace(question=question, response=raw_response)
        enriched["verification_pass"] = report.verification_pass and (fab_citations == 0)
        enriched["adversarial_survives"] = report.adversarial_survives
        enriched["any_fabrication_detected"] = report.any_fabrication_detected or (fab_citations > 0)
        enriched["council_report"] = report.to_dict()
        enriched["quality_gate"] = (
            "pass"
            if (report.overall_pass and fab_citations == 0)
            else "reject"
        )
    else:
        enriched["any_fabrication_detected"] = fab_citations > 0
        if fab_citations > 0:
            enriched["quality_gate"] = "reject"

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 2-4 verification, adversarial challenge, and thinking quality scoring"
    )
    parser.add_argument("--input", required=True, help="Input trace JSONL path")
    parser.add_argument("--output", required=True, help="Output enriched trace JSONL path")
    parser.add_argument(
        "--mode",
        choices=("offline", "consult", "council"),
        default="offline",
        help="Verification mode: offline (Stage 4 only), consult (single host/local), council (multi-family)",
    )
    parser.add_argument(
        "--unitares-url",
        default="http://127.0.0.1:8767",
        help="UNITARES governance HTTP base URL",
    )
    parser.add_argument(
        "--privacy",
        choices=("local", "cloud_allowed"),
        default="cloud_allowed",
        help="Privacy setting passed to UNITARES consult",
    )
    parser.add_argument(
        "--host-id",
        default="claude:host-adapter",
        help="Single host adapter ID when --mode=consult and --privacy=cloud_allowed",
    )
    parser.add_argument(
        "--host-ids",
        nargs="+",
        default=list(DEFAULT_COUNCIL_HOST_IDS),
        help="Council host adapter IDs when --mode=council",
    )
    parser.add_argument("--model", default=None, help="Optional model override for consult")
    parser.add_argument(
        "--client-session-id",
        default=None,
        help="UNITARES bound client_session_id (or set UNITARES_CLIENT_SESSION_ID env var)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max traces to process (0 = all)")
    args = parser.parse_args()

    backend: Optional[UnitaresConsultBackend] = None
    if args.mode == "consult":
        backend = UnitaresConsultBackend(
            base_url=args.unitares_url,
            privacy=args.privacy,
            host_id=args.host_id if args.privacy == "cloud_allowed" else None,
            model=args.model,
            client_session_id=args.client_session_id,
        )
    elif args.mode == "council":
        backend = UnitaresConsultBackend(
            base_url=args.unitares_url,
            privacy="cloud_allowed",
            host_ids=args.host_ids,
            model=args.model,
            client_session_id=args.client_session_id,
        )

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    passed = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)
            enriched = enrich_trace_record(trace, backend=backend)
            if enriched.get("quality_gate") in (None, "pass"):
                passed += 1
            fout.write(json.dumps(enriched) + "\n")
            processed += 1
            if args.limit and processed >= args.limit:
                break

    logger.info(
        f"Processed {processed} traces ({passed} passing) -> {out_path} [mode={args.mode}]"
    )


if __name__ == "__main__":
    main()
