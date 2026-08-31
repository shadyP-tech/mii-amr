"""Record an evidence-backed camera/operator decision for a survey candidate.

This command changes only persistent survey state.  It does not invoke the
camera, route planner, or any motion publisher.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_CONFIRMED,
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    coverage_survey_plan_sha256,
    decide_candidate,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    survey_status,
    write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_admission import (
    ExactTwoCameraHandoffArtifact,
    exact_two_camera_handoff_sha256,
    load_exact_two_camera_handoff,
    require_handoff_candidate_support,
    validate_live_candidate_snapshot_binding,
)
from scripts.aufgabe04.navigation.approach.camera_decision_geometry_binding import (
    CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION,
    CAMERA_FRAME_BINDING_RECEIPT_FIELDS,
    require_camera_recommendation_binding,
    require_projected_camera_candidate_binding,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    FrozenCandidate,
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", required=True, type=Path)
    parser.add_argument("--decision-receipt-json", required=True, type=Path)
    parser.add_argument("--exact-two-camera-handoff-json", type=Path)
    parser.add_argument("--candidate-snapshot-json", type=Path)
    parser.add_argument("--camera-candidate-snapshot-json", type=Path)
    parser.add_argument("--candidate-frame-projection-json", type=Path)
    return parser


def _load_receipt(path: Path, *, survey_id: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid candidate decision receipt: {exc}") from exc
    schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if type(schema_version) is not int or schema_version not in {1, 2, 3}:
        raise ValueError(
            "candidate decision receipt must use schema_version 1, 2, or 3"
        )
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
    if payload["schema_version"] in {
        2,
        CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION,
    }:
        expected_fields = {
            "schema_version",
            "survey_id",
            "candidate_uid",
            "decision",
            "decision_source",
            "camera_evidence_path",
            "exact_two_camera_handoff_path",
            "exact_two_camera_handoff_sha256",
            "candidate_snapshot_path",
            "candidate_snapshot_sha256",
            "candidate_support_class",
            "camera_recommendation_sha256",
        }
        if (
            payload["schema_version"]
            == CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION
        ):
            expected_fields.update(CAMERA_FRAME_BINDING_RECEIPT_FIELDS)
        if set(payload) != expected_fields:
            raise ValueError(
                "exact-two candidate decision receipt fields mismatch"
            )
        if source != "camera_evidence":
            raise ValueError(
                "exact-two receipt requires camera_evidence decision_source"
            )
        path_fields = [
            "exact_two_camera_handoff_path",
            "candidate_snapshot_path",
            "candidate_support_class",
        ]
        hash_fields = [
            "exact_two_camera_handoff_sha256",
            "candidate_snapshot_sha256",
            "camera_recommendation_sha256",
        ]
        if (
            payload["schema_version"]
            == CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION
        ):
            path_fields.extend(
                (
                    "camera_candidate_snapshot_path",
                    "candidate_frame_projection_path",
                )
            )
            hash_fields.extend(
                (
                    "camera_candidate_snapshot_sha256",
                    "candidate_frame_projection_sha256",
                )
            )
        for name in path_fields:
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"exact-two receipt requires {name}")
        for name in hash_fields:
            value = payload.get(name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(
                    f"exact-two receipt requires lowercase {name}"
                )
    return payload


def _paths_match(receipt_value: object, live_path: Path) -> bool:
    if not isinstance(receipt_value, str) or not receipt_value.strip():
        return False
    return Path(receipt_value).resolve() == Path(live_path).resolve()


def _require_live_snapshot_registry_geometry(
    snapshot: CandidateSnapshot,
    registry: StandSurveyRegistry,
    handoff: ExactTwoCameraHandoffArtifact,
) -> None:
    registry_by_uid = {
        candidate.candidate_uid: candidate
        for candidate in registry.candidates
    }
    if set(snapshot.candidate_uids) != set(registry_by_uid):
        raise ValueError(
            "candidate snapshot UID population differs from live registry"
        )
    for frozen in snapshot.candidates:
        live = registry_by_uid[frozen.candidate_uid]
        sealed_evidence = require_handoff_candidate_support(
            handoff,
            frozen.candidate_uid,
        )
        expected_numbers = (
            frozen.geometry.x_m,
            frozen.geometry.y_m,
            frozen.geometry.radius_m,
            frozen.geometry.uncertainty_m,
            frozen.geometry.keepout_radius_m,
            frozen.confidence,
            float(frozen.hit_count),
            frozen.first_seen_sec,
            frozen.last_seen_sec,
        )
        live_numbers = (
            live.x_m,
            live.y_m,
            live.radius_m,
            live.uncertainty_m,
            live.keepout_radius_m,
            live.confidence,
            float(live.hit_count),
            live.first_seen_sec,
            live.last_seen_sec,
        )
        if any(
            not math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-12)
            for first, second in zip(expected_numbers, live_numbers)
        ):
            raise ValueError(
                "candidate snapshot geometry/evidence differs from live "
                f"registry for {frozen.candidate_uid!r}"
            )
        if tuple(sorted(frozen.source.observation_ids)) != tuple(
            sorted(live.source_observation_ids)
        ):
            raise ValueError(
                "candidate snapshot observation ancestry differs from live "
                f"registry for {frozen.candidate_uid!r}"
            )
        if tuple(sorted(live.viewpoint_ids)) != tuple(
            sealed_evidence.viewpoint_ids
        ):
            raise ValueError(
                "candidate viewpoint support differs from the sealed handoff "
                f"for {frozen.candidate_uid!r}"
            )
        if live.status not in {
            sealed_evidence.registry_status,
            STATUS_CONFIRMED,
            STATUS_REJECTED,
        }:
            raise ValueError(
                "candidate lifecycle status differs from the sealed handoff "
                f"for {frozen.candidate_uid!r}"
            )


def _validate_exact_two_decision_contract(
    receipt: dict[str, object],
    *,
    handoff_path: Path,
    snapshot_path: Path,
    camera_snapshot_path: Path | None,
    projection_path: Path | None,
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
) -> FrozenCandidate:
    if not _paths_match(
        receipt["exact_two_camera_handoff_path"], handoff_path
    ):
        raise ValueError("receipt exact-two camera handoff path mismatch")
    if not _paths_match(receipt["candidate_snapshot_path"], snapshot_path):
        raise ValueError("receipt candidate snapshot path mismatch")

    handoff = load_exact_two_camera_handoff(handoff_path)
    handoff_sha256 = exact_two_camera_handoff_sha256(handoff)
    if receipt["exact_two_camera_handoff_sha256"] != handoff_sha256:
        raise ValueError("receipt exact-two camera handoff SHA-256 mismatch")
    if (
        handoff.survey_id != plan.survey_id
        or handoff.planning_frame != plan.planning_frame
        or handoff.map_bundle_sha256 != plan.map_bundle_sha256
        or handoff.plan_sha256 != coverage_survey_plan_sha256(plan)
    ):
        raise ValueError(
            "exact-two camera handoff differs from the live coverage plan"
        )
    if handoff.camera_population_ready is not True:
        raise ValueError("exact-two camera handoff population is not ready")
    if handoff.motion_authorized is not False:
        raise ValueError("exact-two camera handoff must not authorize motion")

    snapshot = load_candidate_snapshot(
        snapshot_path,
        required_map_bundle_sha256=plan.map_bundle_sha256,
    )
    if receipt["candidate_snapshot_sha256"] != candidate_snapshot_sha256(
        snapshot
    ):
        raise ValueError("receipt candidate snapshot SHA-256 mismatch")
    validate_live_candidate_snapshot_binding(
        handoff,
        snapshot,
        candidate_snapshot_path=snapshot_path,
    )
    _require_live_snapshot_registry_geometry(snapshot, registry, handoff)

    candidate_uid = str(receipt["candidate_uid"])
    candidate = snapshot.candidate_for(candidate_uid)
    if candidate is None:
        raise ValueError(
            "candidate is not listed in the exact-two camera handoff snapshot"
        )
    evidence = require_handoff_candidate_support(
        handoff,
        candidate_uid,
        str(receipt["candidate_support_class"]),
    )
    if evidence.support_class is None:
        raise ValueError(
            "candidate snapshot source is outside the exact-two camera contract"
        )
    recommendation_candidate = candidate
    if (
        receipt["schema_version"]
        == CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION
    ):
        if camera_snapshot_path is None or projection_path is None:
            raise ValueError(
                "schema_version 3 requires camera snapshot and projection"
            )
        recommendation_candidate = (
            require_projected_camera_candidate_binding(
                receipt,
                canonical_snapshot_path=snapshot_path,
                canonical_snapshot=snapshot,
                registry=registry,
                source_registry_sha256=handoff.source_registry_sha256,
                camera_snapshot_path=camera_snapshot_path,
                projection_path=projection_path,
                candidate_uid=candidate_uid,
            )
        )
    elif camera_snapshot_path is not None or projection_path is not None:
        raise ValueError(
            "schema_version 2 cannot use camera frame projection arguments"
        )
    require_camera_recommendation_binding(
        receipt,
        candidate=recommendation_candidate,
        planning_frame=plan.planning_frame,
    )
    return candidate


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
        exact_two_arguments = (
            args.exact_two_camera_handoff_json,
            args.candidate_snapshot_json,
        )
        camera_frame_arguments = (
            args.camera_candidate_snapshot_json,
            args.candidate_frame_projection_json,
        )
        if receipt["schema_version"] == 1:
            if any(
                value is not None
                for value in (*exact_two_arguments, *camera_frame_arguments)
            ):
                raise ValueError(
                    "schema_version 1 decisions cannot use exact-two handoff "
                    "arguments"
                )
            if current.status not in {
                STATUS_PENDING_CAMERA,
                str(receipt["decision"]),
            }:
                raise ValueError(
                    "candidate is not ready for a camera/operator decision: "
                    f"status={current.status}"
                )
        else:
            if any(value is None for value in exact_two_arguments):
                raise ValueError(
                    "exact-two decisions require exact-two camera "
                    "handoff and candidate snapshot arguments"
                )
            projected_schema = (
                receipt["schema_version"]
                == CAMERA_DECISION_PROJECTED_RECEIPT_SCHEMA_VERSION
            )
            if projected_schema and any(
                value is None for value in camera_frame_arguments
            ):
                raise ValueError(
                    "schema_version 3 decisions require camera candidate "
                    "snapshot and frame projection arguments"
                )
            if not projected_schema and any(
                value is not None for value in camera_frame_arguments
            ):
                raise ValueError(
                    "schema_version 2 decisions cannot use camera frame "
                    "projection arguments"
                )
            handoff_json = args.exact_two_camera_handoff_json
            snapshot_json = args.candidate_snapshot_json
            if handoff_json is None or snapshot_json is None:
                raise ValueError(
                    "schema_version 2 exact-two arguments are incomplete"
                )
            _validate_exact_two_decision_contract(
                receipt,
                handoff_path=handoff_json,
                snapshot_path=snapshot_json,
                camera_snapshot_path=args.camera_candidate_snapshot_json,
                projection_path=args.candidate_frame_projection_json,
                plan=plan,
                registry=registry,
            )
            if current.status not in {
                STATUS_PENDING_CAMERA,
                STATUS_PROVISIONAL,
                str(receipt["decision"]),
            }:
                raise ValueError(
                    "candidate is not eligible for this exact-two camera "
                    f"decision: status={current.status}"
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
