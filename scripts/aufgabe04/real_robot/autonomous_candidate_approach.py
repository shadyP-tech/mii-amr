"""Motion-edge-free candidate approach orchestration for autonomous exploration.

The parent runner owns ROS sampling, passive camera process lifecycle, operator
authorization, and the only child motion edge.  This module owns deterministic
candidate ordering, offline route planning/validation, the bounded opposite-face
fallback, and final identity/facing artifacts behind injected effects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.foundation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    CandidatePreapproachPlan,
    CandidatePreapproachUnreachableError,
    plan_candidate_preapproach,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import (
    plan_and_select_camera_candidate,
)
from scripts.aufgabe04.navigation.approach.camera_axis_binding import (
    load_opposite_face_normal,
)
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateSelectionConfig,
    NoFeasibleCameraCandidateError,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_admission import (
    exact_two_camera_handoff_sha256,
    load_exact_two_camera_handoff,
    require_handoff_candidate_support,
    validate_live_candidate_snapshot_binding,
    validate_live_registry_binding,
)
from scripts.aufgabe04.navigation.planning.global_planner import plan_route
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.record_stand_candidate_decision import (
    main as record_stand_candidate_decision,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
    coverage_survey_plan_sha256,
    load_stand_survey_registry,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    REAL_VIEWPOINT_SOURCE,
    load_recommendation,
    normalize_angle,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.autonomous_candidate_startup_recovery import (
    CandidateRoutineIdentity,
    CandidateStartupRecoveryAttempt,
    CandidateStartupRecoveryConfig,
    CandidateStartupRecoveryEffects,
    execute_candidate_motion_with_startup_recovery,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    FrozenCandidate,
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)
from scripts.aufgabe04.stations.create_station_identity_registry import (
    create_registry,
)
from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity,
    write_station_identity_registry,
)


SealedRoute = Mapping[str, str]


class CandidateApproachPoseError(RuntimeError):
    """Fail closed when a live pose effect violates the pure pose contract."""

    def __init__(
        self,
        *,
        context: str,
        observed_type: str,
        reason: str,
        candidate_uid: str | None = None,
    ) -> None:
        self.context = context
        self.observed_type = observed_type
        self.reason = reason
        self.candidate_uid = candidate_uid
        candidate_text = (
            "" if candidate_uid is None else f" for {candidate_uid}"
        )
        super().__init__(
            "candidate pose contract failed during "
            f"{context}{candidate_text}: {reason}; "
            f"observed {observed_type}, expected finite Pose2D"
        )

    def to_failure_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "failure_phase": "candidate_approach_pose_contract",
            "candidate_phase": self.context,
            "expected_pose_type": "Pose2D",
            "observed_pose_type": self.observed_type,
            "pose_contract_reason": self.reason,
        }
        if self.candidate_uid is not None:
            fields["candidate_uid"] = self.candidate_uid
        return fields


@dataclass(frozen=True)
class CandidateApproachConfig:
    session_root: Path
    survey_root: Path
    session_id: str
    semantic_map_id: str
    planning_frame: str
    map_yaml: Path
    plan: CoverageSurveyPlan
    snapshot: CandidateSnapshot
    snapshot_path: Path
    approach_offset_m: float
    inflation_radius_m: float
    candidate_transit_radius_m: float
    physical_clearance: Mapping[str, float]
    uncertainty_sigma_multiplier: float
    localization_branch_proof_id: str
    mission_leg_motion_authorization_json: Path
    startup_reseal_motion_authorization_json: Path | None = None
    max_startup_reseals_per_leg: int = 0
    exact_two_camera_handoff_path: Path | None = None
    exact_two_camera_handoff_sha256: str | None = None
    camera_selection_linear_speed_mps: float = 0.055
    camera_selection_angular_speed_radps: float = 0.18


@dataclass(frozen=True)
class CandidatePreapproachRequest:
    map_yaml: Path
    semantic_map_id: str
    plan: CoverageSurveyPlan
    snapshot: CandidateSnapshot
    snapshot_path: Path
    candidate_uid: str
    start: Pose2D
    output_dir: Path
    approach_offset_m: float
    inflation_radius_m: float
    candidate_transit_radius_m: float
    physical_clearance: Mapping[str, float]
    approach_normal_rad: float | None = None
    axis_observation_path: Path | None = None
    prepared_plan: CandidatePreapproachPlan | None = None
    selection_evidence: Mapping[str, object] | None = None


@dataclass(frozen=True)
class CameraCandidateSelectionRequest:
    config: CandidateApproachConfig
    current_pose: Pose2D
    unresolved: frozenset[str]
    support_class_by_uid: Mapping[str, str] | None


@dataclass(frozen=True)
class CameraCandidateInitialSelection:
    candidate_uid: str
    prepared_plan: CandidatePreapproachPlan | None
    evidence: Mapping[str, object]
    motion_authorized: bool = False


@dataclass(frozen=True)
class CandidateMotionLegRequest:
    sealed: SealedRoute
    run_id: str
    session_root: Path
    candidate_snapshot_path: Path
    uncertainty_map_yaml: Path
    uncertainty_sigma_multiplier: float
    localization_branch_proof_id: str
    mission_authorization_json: Path
    session_id: str
    semantic_map_id: str
    mission_leg_kind: MissionLegKind
    mission_leg_index: int
    target_id: str
    permit_json_path: Path


@dataclass(frozen=True)
class CandidateObservationRequest:
    candidate: FrozenCandidate
    output_dir: Path
    attempt_index: int


@dataclass(frozen=True)
class CandidateObservation:
    recommendation_path: Path | None
    qr_id: str | None
    axis_observation_path: Path | None


@dataclass(frozen=True)
class FacingValidationRequest:
    config: CandidateApproachConfig
    candidate: FrozenCandidate
    recommendation_path: Path
    current_pose: Pose2D
    output_dir: Path


@dataclass(frozen=True)
class CandidateDecisionRequest:
    survey_root: Path
    receipt_path: Path
    exact_two_camera_handoff_path: Path | None = None
    candidate_snapshot_path: Path | None = None


@dataclass(frozen=True)
class CandidateApproachComplete:
    stand_count: int
    visit_order: tuple[str, ...]
    identity_registry_path: Path
    identity_registry_sha256: str
    stand_facing_catalog_path: Path
    stand_facing_catalog_sha256: str
    facing_records: tuple[Mapping[str, object], ...]
    motion_authorized: bool = False

    def to_mission_summary_fields(self) -> dict[str, object]:
        return {
            "stand_count": self.stand_count,
            "stand_facing_catalog": str(self.stand_facing_catalog_path),
            "stand_facing_catalog_sha256": self.stand_facing_catalog_sha256,
            "station_identity_registry": str(self.identity_registry_path),
            "station_identity_registry_sha256": (
                self.identity_registry_sha256
            ),
            "motion_authorized": self.motion_authorized,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )


def validate_candidate_approach_handoff(
    config: CandidateApproachConfig,
) -> Mapping[str, str] | None:
    """Validate an optional exact-two handoff before any live robot effect.

    The normal full-coverage path deliberately has no exact-two artifact and
    returns ``None``.  Exact-two mode must provide a paired path/hash, a live
    immutable snapshot, and the unmodified source registry.  This keeps stale
    or tampered evidence from reaching route planning or a motion permit.
    """

    handoff_path = config.exact_two_camera_handoff_path
    handoff_sha256 = config.exact_two_camera_handoff_sha256
    if handoff_path is None and handoff_sha256 is None:
        return None
    if handoff_path is None or handoff_sha256 is None:
        raise ValueError(
            "exact-two camera handoff path and SHA-256 must be provided together"
        )

    live_snapshot = load_candidate_snapshot(
        config.snapshot_path,
        required_map_bundle_sha256=config.plan.map_bundle_sha256,
    )
    if live_snapshot != config.snapshot:
        raise ValueError(
            "candidate snapshot object differs from its live artifact"
        )
    handoff = load_exact_two_camera_handoff(handoff_path)
    actual_handoff_sha256 = exact_two_camera_handoff_sha256(handoff)
    if actual_handoff_sha256 != handoff_sha256:
        raise ValueError("exact-two camera handoff SHA-256 mismatch")
    if (
        handoff.survey_id != config.plan.survey_id
        or handoff.planning_frame != config.planning_frame
        or handoff.map_bundle_sha256 != config.plan.map_bundle_sha256
        or handoff.plan_sha256 != coverage_survey_plan_sha256(config.plan)
    ):
        raise ValueError(
            "exact-two camera handoff differs from the live coverage plan"
        )
    validate_live_candidate_snapshot_binding(
        handoff,
        live_snapshot,
        candidate_snapshot_path=config.snapshot_path,
    )
    if handoff.camera_population_ready is not True:
        raise ValueError("exact-two camera handoff population is not ready")
    if handoff.motion_authorized is not False:
        raise ValueError("exact-two camera handoff must not authorize motion")

    live_registry = load_stand_survey_registry(
        config.survey_root / "stand_registry.json",
        config.plan,
    )
    validate_live_registry_binding(handoff, live_registry)

    support_by_uid: dict[str, str] = {}
    for candidate in live_snapshot.candidates:
        evidence = require_handoff_candidate_support(
            handoff,
            candidate.candidate_uid,
        )
        if evidence.support_class is None:
            raise ValueError(
                "exact-two camera candidate has no sealed support class: "
                f"{candidate.candidate_uid!r}"
            )
        support_by_uid[candidate.candidate_uid] = evidence.support_class
    if not support_by_uid:
        raise ValueError("exact-two camera handoff contains no candidates")
    return support_by_uid


def build_camera_candidate_decision_receipt(
    *,
    config: CandidateApproachConfig,
    candidate: FrozenCandidate,
    recommendation_path: Path,
    exact_two_support_by_uid: Mapping[str, str] | None,
) -> dict[str, object]:
    """Build the mode-scoped receipt consumed by the stopped state writer."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "survey_id": config.plan.survey_id,
        "candidate_uid": candidate.candidate_uid,
        "decision": "confirmed",
        "decision_source": "camera_evidence",
        "camera_evidence_path": str(recommendation_path),
    }
    if exact_two_support_by_uid is None:
        return payload
    support_class = exact_two_support_by_uid.get(candidate.candidate_uid)
    if support_class is None:
        raise RuntimeError(
            "candidate is absent from the exact-two camera handoff: "
            f"{candidate.candidate_uid}"
        )
    handoff_path = config.exact_two_camera_handoff_path
    handoff_sha256 = config.exact_two_camera_handoff_sha256
    if handoff_path is None or handoff_sha256 is None:
        raise RuntimeError("validated exact-two camera handoff is unavailable")
    payload.update(
        {
            "schema_version": 2,
            "exact_two_camera_handoff_path": str(handoff_path),
            "exact_two_camera_handoff_sha256": handoff_sha256,
            "candidate_snapshot_path": str(config.snapshot_path),
            "candidate_snapshot_sha256": candidate_snapshot_sha256(
                config.snapshot
            ),
            "candidate_support_class": support_class,
            "camera_recommendation_sha256": _file_sha256(
                recommendation_path
            ),
        }
    )
    return payload


def nearest_candidate(
    snapshot: CandidateSnapshot,
    current_pose: Pose2D,
    unresolved: set[str],
) -> FrozenCandidate | None:
    """Legacy deterministic helper retained only as a test/compatibility seam.

    Production camera exploration uses the route-aware
    :func:`_select_initial_preapproach` effect instead.
    """

    options = tuple(
        candidate
        for candidate in snapshot.candidates
        if candidate.candidate_uid in unresolved
    )
    if not options:
        return None
    return min(
        options,
        key=lambda candidate: (
            math.hypot(
                current_pose.x_m - candidate.geometry.x_m,
                current_pose.y_m - candidate.geometry.y_m,
            ),
            candidate.candidate_uid,
        ),
    )


def opposite_face_normal(axis_observation_path: Path) -> float:
    """Compatibility wrapper for the pure axis-observation binding."""

    return load_opposite_face_normal(axis_observation_path)


def bounded_approach_offsets(
    requested_m: float,
    minimum_m: float,
    *,
    step_m: float = 0.05,
) -> tuple[float, ...]:
    """Return descending standoffs without crossing the physical minimum."""

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (requested_m, minimum_m, step_m)
    ):
        raise ValueError("approach offsets and step must be finite and positive")
    if requested_m + 1.0e-9 < minimum_m:
        raise ValueError("requested approach offset is below physical minimum")
    values = []
    current = requested_m
    while current > minimum_m + 1.0e-9:
        values.append(round(current, 6))
        current -= step_m
    if not values or abs(values[-1] - minimum_m) > 1.0e-9:
        values.append(round(minimum_m, 6))
    return tuple(values)


def is_approach_feasibility_failure(exc: ValueError) -> bool:
    if isinstance(exc, CandidatePreapproachUnreachableError):
        return True
    message = str(exc)
    return (
        "candidate pre-approach A* failed" in message
        or "target is blocked" in message
    )


def validate_facing_pose(request: FacingValidationRequest) -> dict[str, object]:
    config = request.config
    candidate = request.candidate
    recommendation = load_recommendation(
        request.recommendation_path,
        expected_frame=config.planning_frame,
        expected_source=REAL_VIEWPOINT_SOURCE,
        expected_simulation_only=False,
    )
    if recommendation.stand_id != candidate.candidate_uid:
        raise ValueError(
            "viewpoint recommendation stand_id mismatch: "
            f"expected {candidate.candidate_uid!r}, "
            f"got {recommendation.stand_id!r}"
        )
    target = recommendation.material_target.pose
    grid, map_bundle = load_occupancy_grid_with_bundle(
        config.map_yaml,
        semantic_map_id=config.semantic_map_id,
        planning_frame=config.planning_frame,
    )
    if map_bundle.bundle_sha256 != config.plan.map_bundle_sha256:
        raise ValueError("facing-pose validation map differs from survey")
    costmap = (
        Costmap.from_occupancy_grid(grid)
        .with_arena_bounds(config.plan.arena_bounds)
        .with_inflation(config.inflation_radius_m)
    )
    other_keepouts = tuple(
        Station(
            item.candidate_uid,
            StationPose(item.geometry.x_m, item.geometry.y_m, 0.0),
            0.0,
            item.geometry.keepout_radius_m,
        )
        for item in config.snapshot.candidates
        if item.candidate_uid != candidate.candidate_uid
    )
    if other_keepouts:
        costmap = costmap.with_station_keepouts(other_keepouts)
    route = plan_route(
        costmap,
        request.current_pose,
        target,
        snap_radius_m=config.plan.config.snap_radius_m,
    )
    if route.route is None or route.failure is not None:
        reason = route.failure.reason if route.failure is not None else "no route"
        raise ValueError(f"computed QR-facing pose is not A*-reachable: {reason}")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    route_path = request.output_dir / "facing_pose_validation_route.csv"
    diagnostics_path = (
        request.output_dir / "facing_pose_validation_diagnostics.json"
    )
    write_route_csv(route_path, (route,), final_yaw_by_leg={0: target.yaw_rad})
    write_diagnostics_json(
        diagnostics_path,
        (route,),
        metadata={
            "route_kind": "facing_pose_validation_only",
            "motion_authorized": False,
            "map_bundle_sha256": map_bundle.bundle_sha256,
            "candidate_uid": candidate.candidate_uid,
            "arena_boundary_overlay": True,
            "arena_bounds": config.plan.arena_bounds.to_metadata(),
            "inflation_radius_m": config.inflation_radius_m,
        },
    )
    qr_face = next(
        (
            face
            for face in recommendation.face_candidates
            if face.face_id == recommendation.material_target.face_id
        ),
        None,
    )
    if qr_face is None:
        raise ValueError("recommendation target face is absent from candidates")
    stand_axis = (qr_face.outward_normal_rad - math.pi / 2.0) % math.pi
    return {
        "candidate_uid": candidate.candidate_uid,
        "stand_center": {
            "x_m": candidate.geometry.x_m,
            "y_m": candidate.geometry.y_m,
        },
        "stand_axis_rad_axial": stand_axis,
        "qr_outward_normal_rad": normalize_angle(qr_face.outward_normal_rad),
        "facing_pose": {
            "x_m": target.x_m,
            "y_m": target.y_m,
            "yaw_rad": target.yaw_rad,
        },
        "axis_confidence": recommendation.axis_confidence,
        "axis_sample_count": recommendation.axis_sample_count,
        "recommendation_json": str(request.recommendation_path),
        "validation_route_csv": str(route_path),
        "motion_to_facing_pose_authorized": False,
    }


def commit_candidate_decision(request: CandidateDecisionRequest) -> None:
    argv = [
        "--survey-root",
        str(request.survey_root),
        "--decision-receipt-json",
        str(request.receipt_path),
    ]
    if request.exact_two_camera_handoff_path is not None:
        if request.candidate_snapshot_path is None:
            raise RuntimeError(
                "exact-two candidate decision requires its candidate snapshot"
            )
        argv.extend(
            [
                "--exact-two-camera-handoff-json",
                str(request.exact_two_camera_handoff_path),
                "--candidate-snapshot-json",
                str(request.candidate_snapshot_path),
            ]
        )
    elif request.candidate_snapshot_path is not None:
        raise RuntimeError(
            "candidate snapshot decision argument requires exact-two handoff"
        )
    try:
        returncode = record_stand_candidate_decision(argv)
    except SystemExit as exc:
        raise RuntimeError("failed to commit camera candidate decision") from exc
    if returncode != 0:
        raise RuntimeError("failed to commit camera candidate decision")


def _plan_preapproach_from_request(
    request: CandidatePreapproachRequest,
) -> dict[str, str]:
    return plan_candidate_preapproach(
        map_yaml=request.map_yaml,
        semantic_map_id=request.semantic_map_id,
        plan=request.plan,
        snapshot=request.snapshot,
        snapshot_path=request.snapshot_path,
        candidate_uid=request.candidate_uid,
        start=request.start,
        output_dir=request.output_dir,
        approach_offset_m=request.approach_offset_m,
        inflation_radius_m=request.inflation_radius_m,
        candidate_transit_radius_m=request.candidate_transit_radius_m,
        physical_clearance=request.physical_clearance,
        approach_normal_rad=request.approach_normal_rad,
        axis_observation_path=request.axis_observation_path,
        prepared_plan=request.prepared_plan,
        selection_evidence=request.selection_evidence,
    )


def _select_initial_preapproach(
    request: CameraCandidateSelectionRequest,
) -> CameraCandidateInitialSelection:
    config = request.config
    planned = plan_and_select_camera_candidate(
        map_yaml=config.map_yaml,
        semantic_map_id=config.semantic_map_id,
        plan=config.plan,
        snapshot=config.snapshot,
        current_pose=request.current_pose,
        unresolved=request.unresolved,
        approach_offset_m=config.approach_offset_m,
        inflation_radius_m=config.inflation_radius_m,
        candidate_transit_radius_m=config.candidate_transit_radius_m,
        physical_clearance=config.physical_clearance,
        selection_config=CameraCandidateSelectionConfig(
            linear_speed_mps=config.camera_selection_linear_speed_mps,
            angular_speed_radps=config.camera_selection_angular_speed_radps,
        ),
        support_class_by_uid=request.support_class_by_uid,
    )
    return CameraCandidateInitialSelection(
        candidate_uid=planned.selected_candidate_uid,
        prepared_plan=planned.selected_plan,
        evidence=planned.to_evidence(),
    )


def _validate_initial_selection(
    selection: CameraCandidateInitialSelection,
    *,
    unresolved: set[str],
) -> None:
    if not isinstance(selection, CameraCandidateInitialSelection):
        raise TypeError(
            "camera selection effect must return CameraCandidateInitialSelection"
        )
    if selection.motion_authorized is not False:
        raise RuntimeError("camera candidate selection must not authorize motion")
    if selection.candidate_uid not in unresolved:
        raise RuntimeError("camera selector returned a resolved or unknown candidate")
    if not isinstance(selection.evidence, Mapping):
        raise TypeError("camera candidate selection evidence must be a mapping")
    evidence_uid = selection.evidence.get("selected_candidate_uid")
    if evidence_uid is not None and evidence_uid != selection.candidate_uid:
        raise RuntimeError("camera candidate selection evidence UID mismatch")
    evidence_motion = selection.evidence.get("motion_authorized")
    if evidence_motion not in (None, False):
        raise RuntimeError("camera candidate evidence must not authorize motion")
    if (
        selection.prepared_plan is not None
        and selection.prepared_plan.candidate_uid != selection.candidate_uid
    ):
        raise RuntimeError("camera selector returned a route for another candidate")


@dataclass(frozen=True)
class CandidateApproachEffects:
    read_current_pose: Callable[[], Pose2D]
    run_motion_leg: Callable[[CandidateMotionLegRequest], MotionLegOutcome]
    capture_observation: Callable[
        [CandidateObservationRequest], CandidateObservation
    ]
    plan_preapproach: Callable[
        [CandidatePreapproachRequest], SealedRoute
    ] = _plan_preapproach_from_request
    select_initial_preapproach: Callable[
        [CameraCandidateSelectionRequest], CameraCandidateInitialSelection
    ] = _select_initial_preapproach
    validate_facing: Callable[
        [FacingValidationRequest], Mapping[str, object]
    ] = validate_facing_pose
    commit_decision: Callable[
        [CandidateDecisionRequest], None
    ] = commit_candidate_decision
    admit_startup_localization: Callable[[Path], Pose2D] | None = None
    run_startup_reseal_motion_leg: Callable[
        [CandidateMotionLegRequest, CandidateStartupRecoveryAttempt],
        MotionLegOutcome,
    ] | None = None
    event_sink: Callable[[Path, Mapping[str, object]], None] = _append_jsonl
    clock: Callable[[], float] = time.time


def _read_finite_pose2d(
    effects: CandidateApproachEffects,
    *,
    context: str,
    candidate_uid: str | None = None,
) -> Pose2D:
    pose = effects.read_current_pose()
    if not isinstance(pose, Pose2D):
        raise CandidateApproachPoseError(
            context=context,
            candidate_uid=candidate_uid,
            observed_type=type(pose).__name__,
            reason="read_current_pose returned the wrong type",
        )
    try:
        values = (float(pose.x_m), float(pose.y_m), float(pose.yaw_rad))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateApproachPoseError(
            context=context,
            candidate_uid=candidate_uid,
            observed_type=type(pose).__name__,
            reason="pose coordinates are not numeric",
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise CandidateApproachPoseError(
            context=context,
            candidate_uid=candidate_uid,
            observed_type=type(pose).__name__,
            reason="pose coordinates are not finite",
        )
    return Pose2D(*values)


def _motion_request(
    *,
    config: CandidateApproachConfig,
    sealed: SealedRoute,
    run_id: str,
    candidate_snapshot_path: Path,
    leg_kind: MissionLegKind,
    candidate_index: int,
    target_id: str,
) -> CandidateMotionLegRequest:
    return CandidateMotionLegRequest(
        sealed=sealed,
        run_id=run_id,
        session_root=config.session_root,
        candidate_snapshot_path=candidate_snapshot_path,
        uncertainty_map_yaml=config.map_yaml,
        uncertainty_sigma_multiplier=config.uncertainty_sigma_multiplier,
        localization_branch_proof_id=config.localization_branch_proof_id,
        mission_authorization_json=(
            config.mission_leg_motion_authorization_json
        ),
        session_id=config.session_id,
        semantic_map_id=config.semantic_map_id,
        mission_leg_kind=leg_kind,
        mission_leg_index=candidate_index,
        target_id=target_id,
        permit_json_path=(
            config.session_root
            / "motion_authorization"
            / "mission_legs"
            / f"{run_id}_permit.json"
        ).absolute(),
    )


def _candidate_routine_identity(
    request: CandidateMotionLegRequest,
) -> CandidateRoutineIdentity:
    return CandidateRoutineIdentity(
        session_id=request.session_id,
        semantic_map_id=request.semantic_map_id,
        routine_kind=request.mission_leg_kind.value,
        routine_index=request.mission_leg_index,
        target_id=request.target_id,
        run_id=request.run_id,
    )


def _execute_candidate_motion(
    *,
    config: CandidateApproachConfig,
    effects: CandidateApproachEffects,
    candidate_root: Path,
    plan_request: CandidatePreapproachRequest,
    initial_sealed: SealedRoute,
    run_id: str,
    leg_kind: MissionLegKind,
    candidate_index: int,
    target_id: str,
) -> MotionLegOutcome:
    """Run one candidate routine with bounded same-identity startup recovery."""

    initial_request = _motion_request(
        config=config,
        sealed=initial_sealed,
        run_id=run_id,
        candidate_snapshot_path=(
            plan_request.output_dir / "candidate_snapshot.json"
        ),
        leg_kind=leg_kind,
        candidate_index=candidate_index,
        target_id=target_id,
    )

    def admit_localization(evidence_path: Path) -> Pose2D:
        if effects.admit_startup_localization is None:
            raise RuntimeError(
                "candidate startup recovery has no stationary localization "
                "admission effect"
            )
        return effects.admit_startup_localization(evidence_path)

    def replan_same_routine(
        attempt: CandidateStartupRecoveryAttempt,
    ) -> CandidateMotionLegRequest:
        replacement_plan = replace(
            plan_request,
            start=attempt.fresh_start_pose,
            output_dir=attempt.source_root,
            prepared_plan=None,
            selection_evidence=None,
        )
        replacement_sealed = effects.plan_preapproach(replacement_plan)
        return _motion_request(
            config=config,
            sealed=replacement_sealed,
            run_id=attempt.identity.run_id,
            candidate_snapshot_path=(
                attempt.source_root / "candidate_snapshot.json"
            ),
            leg_kind=leg_kind,
            candidate_index=candidate_index,
            target_id=target_id,
        )

    def run_replacement(
        request: CandidateMotionLegRequest,
        attempt: CandidateStartupRecoveryAttempt,
    ) -> MotionLegOutcome:
        if effects.run_startup_reseal_motion_leg is None:
            raise RuntimeError(
                "candidate startup recovery has no startup-reseal motion effect"
            )
        return effects.run_startup_reseal_motion_leg(request, attempt)

    return execute_candidate_motion_with_startup_recovery(
        initial_request,
        config=CandidateStartupRecoveryConfig(
            initial_identity=_candidate_routine_identity(initial_request),
            recovery_root=(
                candidate_root / f"{leg_kind.value}_startup_reseals"
            ),
            event_log_path=config.session_root / "adaptive_replans.jsonl",
            max_startup_reseals=config.max_startup_reseals_per_leg,
        ),
        effects=CandidateStartupRecoveryEffects(
            run_initial=effects.run_motion_leg,
            run_replacement=run_replacement,
            admit_fresh_stationary_localization=admit_localization,
            replan_same_routine=replan_same_routine,
            describe_request=_candidate_routine_identity,
            event_sink=lambda path, payload: effects.event_sink(path, payload),
            clock=effects.clock,
        ),
    )


def execute_candidate_approach_phase(
    config: CandidateApproachConfig,
    effects: CandidateApproachEffects,
) -> CandidateApproachComplete:
    """Execute the post-coverage candidate state machine behind live effects."""

    exact_two_support_by_uid = validate_candidate_approach_handoff(config)
    unresolved = set(config.snapshot.candidate_uids)
    facing_records: list[Mapping[str, object]] = []
    identities: list[StationIdentity] = []
    visit_order: list[str] = []
    candidate_index = 0

    while unresolved:
        current = _read_finite_pose2d(
            effects,
            context="initial_candidate_selection",
        )
        try:
            selection = effects.select_initial_preapproach(
                CameraCandidateSelectionRequest(
                    config=config,
                    current_pose=current,
                    unresolved=frozenset(unresolved),
                    support_class_by_uid=exact_two_support_by_uid,
                )
            )
        except NoFeasibleCameraCandidateError as exc:
            effects.event_sink(
                config.session_root / "candidate_selection.jsonl",
                {
                    **exc.to_evidence(),
                    "event": "camera_candidate_selection_failed",
                    "timestamp_unix_sec": effects.clock(),
                    "motion_authorized": False,
                },
            )
            raise
        _validate_initial_selection(selection, unresolved=unresolved)
        candidate = config.snapshot.candidate_for(selection.candidate_uid)
        if candidate is None:
            raise RuntimeError("selected candidate disappeared from snapshot")
        candidate_root = (
            config.session_root
            / "candidates"
            / f"{candidate_index:03d}_{candidate.candidate_uid}"
        )
        source_root = candidate_root / "preapproach_source"
        preapproach_plan_request = CandidatePreapproachRequest(
            map_yaml=config.map_yaml,
            semantic_map_id=config.semantic_map_id,
            plan=config.plan,
            snapshot=config.snapshot,
            snapshot_path=config.snapshot_path,
            candidate_uid=candidate.candidate_uid,
            start=current,
            output_dir=source_root,
            approach_offset_m=config.approach_offset_m,
            inflation_radius_m=config.inflation_radius_m,
            candidate_transit_radius_m=config.candidate_transit_radius_m,
            physical_clearance=config.physical_clearance,
            prepared_plan=selection.prepared_plan,
            selection_evidence=selection.evidence,
        )
        selection_log_path = config.session_root / "candidate_selection.jsonl"
        effects.event_sink(
            selection_log_path,
            {
                **dict(selection.evidence),
                "event": "camera_candidate_ranked",
                "timestamp_unix_sec": effects.clock(),
                "selected_candidate_uid": candidate.candidate_uid,
                "route_materialized": False,
                "motion_authorized": False,
            },
        )
        try:
            sealed = effects.plan_preapproach(preapproach_plan_request)
        except Exception as exc:
            effects.event_sink(
                selection_log_path,
                {
                    **dict(selection.evidence),
                    "event": "camera_candidate_route_materialization_failed",
                    "timestamp_unix_sec": effects.clock(),
                    "selected_candidate_uid": candidate.candidate_uid,
                    "route_materialized": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "motion_authorized": False,
                },
            )
            raise
        materialized_event = {
            **dict(selection.evidence),
            "event": "camera_candidate_route_materialized",
            "timestamp_unix_sec": effects.clock(),
            "selected_candidate_uid": candidate.candidate_uid,
            "materialized_candidate_uid": candidate.candidate_uid,
            "selected_route_reused_for_materialization": (
                selection.prepared_plan is not None
            ),
            "route_materialized": True,
            "motion_authorized": False,
        }
        effects.event_sink(
            selection_log_path,
            materialized_event,
        )
        candidate_run_id = (
            f"{config.session_id}_candidate_{candidate_index:03d}"
        )
        outcome = _execute_candidate_motion(
            config=config,
            effects=effects,
            candidate_root=candidate_root,
            plan_request=preapproach_plan_request,
            initial_sealed=sealed,
            run_id=candidate_run_id,
            leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            candidate_index=candidate_index,
            target_id=candidate.candidate_uid,
        )
        observation = effects.capture_observation(
            CandidateObservationRequest(
                candidate=candidate,
                output_dir=candidate_root / "camera_lidar_attempt_00",
                attempt_index=0,
            )
        )

        if observation.recommendation_path is None:
            if observation.axis_observation_path is None:
                raise RuntimeError(
                    "observer returned neither QR recommendation nor axis"
                )
            opposite_normal = opposite_face_normal(
                observation.axis_observation_path
            )
            opposite_start = _read_finite_pose2d(
                effects,
                context="opposite_face_preapproach",
                candidate_uid=candidate.candidate_uid,
            )
            opposite_source = candidate_root / "opposite_face_source"
            opposite_sealed = None
            opposite_plan_request = None
            feasibility_failures = []
            for inspection_offset_m in bounded_approach_offsets(
                config.approach_offset_m,
                float(config.physical_clearance["minimum_active_standoff_m"]),
            ):
                try:
                    opposite_plan_request = CandidatePreapproachRequest(
                        map_yaml=config.map_yaml,
                        semantic_map_id=config.semantic_map_id,
                        plan=config.plan,
                        snapshot=config.snapshot,
                        snapshot_path=config.snapshot_path,
                        candidate_uid=candidate.candidate_uid,
                        start=opposite_start,
                        output_dir=opposite_source,
                        approach_offset_m=inspection_offset_m,
                        inflation_radius_m=config.inflation_radius_m,
                        candidate_transit_radius_m=(
                            config.candidate_transit_radius_m
                        ),
                        physical_clearance=config.physical_clearance,
                        approach_normal_rad=opposite_normal,
                        axis_observation_path=(
                            observation.axis_observation_path
                        ),
                    )
                    opposite_sealed = effects.plan_preapproach(
                        opposite_plan_request
                    )
                    break
                except ValueError as exc:
                    if not is_approach_feasibility_failure(exc):
                        raise
                    feasibility_failures.append(
                        f"{inspection_offset_m:.3f} m: {exc}"
                    )
            if opposite_sealed is None or opposite_plan_request is None:
                raise RuntimeError(
                    "no physically allowed opposite-face approach was "
                    "A*-reachable: " + "; ".join(feasibility_failures)
                )
            opposite_run_id = f"{candidate_run_id}_opposite"
            opposite_outcome = _execute_candidate_motion(
                config=config,
                effects=effects,
                candidate_root=candidate_root,
                plan_request=opposite_plan_request,
                initial_sealed=opposite_sealed,
                run_id=opposite_run_id,
                leg_kind=MissionLegKind.OPPOSITE_FACE,
                candidate_index=candidate_index,
                target_id=candidate.candidate_uid,
            )
            observation = effects.capture_observation(
                CandidateObservationRequest(
                    candidate=candidate,
                    output_dir=candidate_root / "camera_lidar_attempt_01",
                    attempt_index=1,
                )
            )
            if observation.recommendation_path is None:
                raise RuntimeError(
                    "QR side remained unresolved after opposite-face "
                    f"inspection for {candidate.candidate_uid}"
                )

        if observation.qr_id is None:
            raise RuntimeError("camera recommendation has no QR identity")
        stopped_pose = _read_finite_pose2d(
            effects,
            context="stopped_facing_validation",
            candidate_uid=candidate.candidate_uid,
        )
        facing = dict(
            effects.validate_facing(
                FacingValidationRequest(
                    config=config,
                    candidate=candidate,
                    recommendation_path=observation.recommendation_path,
                    current_pose=stopped_pose,
                    output_dir=candidate_root,
                )
            )
        )
        facing["qr_id"] = observation.qr_id
        receipt = candidate_root / "candidate_decision.json"
        receipt_payload = build_camera_candidate_decision_receipt(
            config=config,
            candidate=candidate,
            recommendation_path=observation.recommendation_path,
            exact_two_support_by_uid=exact_two_support_by_uid,
        )
        _write_json(receipt, receipt_payload)
        effects.commit_decision(
            CandidateDecisionRequest(
                survey_root=config.survey_root,
                receipt_path=receipt,
                exact_two_camera_handoff_path=(
                    config.exact_two_camera_handoff_path
                ),
                candidate_snapshot_path=(
                    config.snapshot_path
                    if exact_two_support_by_uid is not None
                    else None
                ),
            )
        )
        facing_records.append(facing)
        identities.append(
            StationIdentity(
                candidate.candidate_uid,
                observation.qr_id,
                f"station_{observation.qr_id}",
            )
        )
        visit_order.append(candidate.candidate_uid)
        unresolved.discard(candidate.candidate_uid)
        candidate_index += 1

    identity_registry, _source_sha = create_registry(
        candidate_snapshot=config.snapshot,
        mappings=identities,
        registry_id=f"{config.session_id}_identities",
        created_unix_sec=effects.clock(),
    )
    identity_path = config.session_root / "station_identity_registry.json"
    identity_sha256 = write_station_identity_registry(
        identity_path,
        identity_registry,
    )
    catalog = {
        "schema_version": 1,
        "catalog_kind": "real_autonomous_stand_facing_poses",
        "session_id": config.session_id,
        "planning_frame": config.planning_frame,
        "map_bundle_sha256": config.plan.map_bundle_sha256,
        "coverage_plan_sha256": coverage_survey_plan_sha256(config.plan),
        "candidate_snapshot_sha256": candidate_snapshot_sha256(
            config.snapshot
        ),
        "station_identity_registry_sha256": identity_sha256,
        "stand_count": len(facing_records),
        "records": sorted(
            facing_records,
            key=lambda item: str(item["candidate_uid"]),
        ),
    }
    catalog_path = config.session_root / "stand_facing_catalog.json"
    catalog_sha256 = write_content_hashed_json(
        catalog_path,
        catalog,
        hash_field="stand_facing_catalog_sha256",
    )
    return CandidateApproachComplete(
        stand_count=len(facing_records),
        visit_order=tuple(visit_order),
        identity_registry_path=identity_path,
        identity_registry_sha256=identity_sha256,
        stand_facing_catalog_path=catalog_path,
        stand_facing_catalog_sha256=catalog_sha256,
        facing_records=tuple(facing_records),
    )


__all__ = [
    "CandidateApproachComplete",
    "CandidateApproachConfig",
    "CandidateApproachEffects",
    "CandidateApproachPoseError",
    "CameraCandidateInitialSelection",
    "CameraCandidateSelectionRequest",
    "CandidateDecisionRequest",
    "CandidateMotionLegRequest",
    "CandidateObservation",
    "CandidateObservationRequest",
    "CandidatePreapproachRequest",
    "FacingValidationRequest",
    "bounded_approach_offsets",
    "build_camera_candidate_decision_receipt",
    "execute_candidate_approach_phase",
    "nearest_candidate",
    "opposite_face_normal",
    "plan_candidate_preapproach",
    "validate_candidate_approach_handoff",
    "validate_facing_pose",
]
