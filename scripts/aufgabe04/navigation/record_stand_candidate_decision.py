"""Record an evidence-backed camera/operator decision for a survey candidate.

This command changes only persistent survey state.  It does not invoke the
camera, route planner, or any motion publisher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_CONFIRMED,
    STATUS_REJECTED,
    decide_candidate,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    survey_status,
    write_stand_survey_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", required=True, type=Path)
    parser.add_argument("--decision-receipt-json", required=True, type=Path)
    return parser


def _load_receipt(path: Path, *, survey_id: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid candidate decision receipt: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("candidate decision receipt must use schema_version 1")
    if payload.get("survey_id") != survey_id:
        raise ValueError("candidate decision receipt belongs to another survey")
    candidate_uid = payload.get("candidate_uid")
    if not isinstance(candidate_uid, str) or not candidate_uid.strip():
        raise ValueError("candidate decision receipt has no candidate_uid")
    decision = payload.get("decision")
    if decision not in {STATUS_CONFIRMED, STATUS_REJECTED}:
        raise ValueError("candidate decision must be confirmed or rejected")
    source = payload.get("decision_source")
    if source == "operator":
        if payload.get("operator_confirmed") is not True:
            raise ValueError(
                "operator decision receipt requires operator_confirmed=true"
            )
    elif source == "camera_evidence":
        evidence_path = payload.get("camera_evidence_path")
        if not isinstance(evidence_path, str) or not evidence_path.strip():
            raise ValueError(
                "camera decision receipt requires camera_evidence_path"
            )
    else:
        raise ValueError(
            "decision_source must be operator or camera_evidence"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = load_coverage_survey_plan(args.survey_root / "coverage_plan.json")
        progress = load_survey_progress(
            args.survey_root / "coverage_progress.json",
            plan,
        )
        registry_path = args.survey_root / "stand_registry.json"
        registry = load_stand_survey_registry(registry_path, plan)
        receipt = _load_receipt(
            args.decision_receipt_json,
            survey_id=plan.survey_id,
        )
        candidate_uid = str(receipt["candidate_uid"])
        current = registry.candidate_for(candidate_uid)
        if current is None:
            raise ValueError(f"unknown survey candidate {candidate_uid!r}")
        if current.status not in {"pending_camera", str(receipt["decision"])}:
            raise ValueError(
                "candidate is not ready for a camera/operator decision: "
                f"status={current.status}"
            )
        registry = decide_candidate(
            registry,
            candidate_uid,
            status=str(receipt["decision"]),
        )
        decisions_dir = args.survey_root / "decisions"
        decision_path = decisions_dir / f"{candidate_uid}.json"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        canonical = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if decision_path.exists() and decision_path.read_text() != canonical:
            raise ValueError(
                f"refusing conflicting candidate decision: {decision_path}"
            )
        decision_path.write_text(canonical)
        write_stand_survey_registry(registry_path, registry, plan)
        status = {
            "schema_version": 1,
            "status": "candidate_decision_recorded",
            "motion_published": False,
            **survey_status(plan, progress, registry),
            "candidate_uid": candidate_uid,
            "decision": receipt["decision"],
            "decision_receipt": str(decision_path),
        }
        (args.survey_root / "survey_summary.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

