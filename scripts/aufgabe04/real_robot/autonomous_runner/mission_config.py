"""Pure autonomous-mission policy, hashing, and snapshot construction."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_report import (
    evidence_only_reconciliation_policy_contract,
)
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    DynamicApproachConfig,
    minimum_static_obstacle_inflation_m,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    coverage_survey_plan_sha256,
)
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    stand_width_profile_from_radius,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_measured_physical_stand_model,
)
from scripts.aufgabe04.real_robot.execution.child_runner import (
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_LIDAR_STOP_DISTANCE_M,
    DEFAULT_TRACKING_TUBE_RADIUS_M,
)
from scripts.aufgabe04.real_robot.mission.coverage import (
    CoverageCompletionPolicy,
)
from scripts.aufgabe04.real_robot.mission.modes import AutonomousRunMode
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
)

DEFAULT_LIDAR_CLEARANCE_MARGIN_M = 0.02
DEFAULT_CANDIDATE_LIDAR_ENVELOPE_RADIUS_M = 0.06

def _coverage_completion_policy(
    mode: AutonomousRunMode,
    *,
    exact_inspection_point_count: int | None,
) -> CoverageCompletionPolicy:
    """Select the evidence gate without expanding camera-motion scope."""

    if not isinstance(mode, AutonomousRunMode):
        raise ValueError("autonomous run mode is required")
    if mode is AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA:
        if exact_inspection_point_count != 2:
            raise ValueError(
                "execute-exact-two-camera requires exactly two inspection "
                "points"
            )
        return CoverageCompletionPolicy.EXACT_TWO_CAMERA_VALIDATION
    if mode in {
        AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT,
        AutonomousRunMode.RESUME_NEXT_COVERAGE_LEG,
    } and exact_inspection_point_count == 2:
        return CoverageCompletionPolicy.EXACT_TWO_LIDAR_CHECKPOINT
    return CoverageCompletionPolicy.CAMERA_READY

def _plan_exact_inspection_point_count(
    plan: CoverageSurveyPlan,
    *,
    requested: int | None,
) -> int | None:
    """Use frozen plan scope, with a compatibility seam for test doubles."""

    config = getattr(plan, "config", None)
    value = getattr(config, "exact_inspection_point_count", requested)
    if value not in {None, 2}:
        raise ValueError("frozen plan has an invalid exact inspection-point count")
    return value

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _checkpoint_config_sha256(args) -> str:
    """Bind checkpoint continuation to all behavior-relevant run settings."""

    stand_model_sha256 = (
        None
        if args.stand_model_profile is None
        else load_measured_physical_stand_model(
            args.stand_model_profile
        ).sha256
    )
    morphology_profile = stand_width_profile_from_radius(
        CoverageSurveyConfig().candidate_radius_m
    )
    return payload_sha256(
        {
            "schema_version": 2,
            "semantic_map_id": args.semantic_map_id,
            "expected_stand_count": args.expected_stand_count,
            "inspection_stop_spacing_m": args.inspection_stop_spacing_m,
            "exact_inspection_point_count": (
                args.exact_inspection_point_count
            ),
            "lidar_epoch_sec": args.lidar_epoch_sec,
            "candidate_approach_offset_m": args.candidate_approach_offset_m,
            "final_facing_offset_m": args.final_facing_offset_m,
            "axis_sample_count": args.axis_sample_count,
            "camera_timeout_sec": args.camera_timeout_sec,
            "stand_model_sha256": stand_model_sha256,
            "lidar_track_morphology_profile": (
                morphology_profile.to_evidence_dict()
            ),
            "lidar_proposal_width_bounds_m": {
                "minimum": 0.03,
                "maximum": 0.45,
            },
            "lidar_visibility_reconciliation": (
                evidence_only_reconciliation_policy_contract()
            ),
            "max_blockage_replans_per_leg": (
                args.max_blockage_replans_per_leg
            ),
            "max_startup_reseals_per_leg": args.max_startup_reseals_per_leg,
            "max_runtime_localization_reseals_per_leg": (
                args.max_runtime_localization_reseals_per_leg
            ),
            "max_localization_readiness_retries_per_leg": (
                args.max_localization_readiness_retries_per_leg
            ),
            "uncertainty_sigma_multiplier": (
                args.uncertainty_sigma_multiplier
            ),
            "certified_route_tube_radius_m": (
                DEFAULT_TRACKING_TUBE_RADIUS_M
            ),
            "collision_margin_m": DEFAULT_COLLISION_MARGIN_M,
            "lidar_stop_distance_m": DEFAULT_LIDAR_STOP_DISTANCE_M,
            "lidar_clearance_margin_m": DEFAULT_LIDAR_CLEARANCE_MARGIN_M,
        }
    )

def _default_session_id(run_mode: AutonomousRunMode) -> str:
    mode_token = run_mode.value.replace("-", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return f"stand_explore_{mode_token}_{timestamp}"

def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def _physical_clearance(
    profile,
    *,
    approach_offset_m: float,
    stand_model_profile=None,
) -> dict[str, float | None]:
    stand_collision_radius_m = DEFAULT_CANDIDATE_LIDAR_ENVELOPE_RADIUS_M
    stand_base_circumscribed_radius_m = None
    stand_model_tolerance_m = None
    if stand_model_profile is not None:
        navigation_radius = stand_model_profile.navigation_footprint_radius_m
        if navigation_radius is None:
            raise ValueError("stand model has no navigation footprint radius")
        stand_collision_radius_m = navigation_radius
        stand_base_circumscribed_radius_m = (
            stand_model_profile.base_circumscribed_radius_m
        )
        stand_model_tolerance_m = stand_model_profile.tolerance_m
    shared = {
        "stand_position_uncertainty_m": 0.02,
        "robot_radius_m": profile.robot_radius_m,
        "collision_margin_m": DEFAULT_COLLISION_MARGIN_M,
        "tracking_margin_m": DEFAULT_TRACKING_TUBE_RADIUS_M,
        "standoff_distance_m": approach_offset_m,
        "lidar_stop_distance_m": DEFAULT_LIDAR_STOP_DISTANCE_M,
        "scan_origin_to_base_offset_m": profile.scan_origin_to_base_offset_m,
        "lidar_clearance_margin_m": DEFAULT_LIDAR_CLEARANCE_MARGIN_M,
        "minimum_non_target_keepout_radius_m": 0.31,
    }
    collision_config = DynamicApproachConfig(
        stand_radius_m=stand_collision_radius_m,
        **shared,
    )
    lidar_config = DynamicApproachConfig(
        # This radius is the detector's cross-section at LiDAR height.  The
        # larger floor-level base belongs only in collision clearance.
        stand_radius_m=DEFAULT_CANDIDATE_LIDAR_ENVELOPE_RADIUS_M,
        **shared,
    )
    minimum_collision_standoff_m = collision_config.stand_keepout_radius_m
    # Candidate uncertainty shifts both the base collision envelope and the
    # LiDAR-visible cross-section.  DynamicApproachConfig includes it in the
    # former; add it exactly once to the latter here.
    minimum_lidar_standoff_m = (
        lidar_config.minimum_lidar_standoff_m
        + float(shared["stand_position_uncertainty_m"])
    )
    minimum_active_standoff_m = max(
        minimum_collision_standoff_m,
        minimum_lidar_standoff_m,
    )
    minimum_candidate_transit_radius_m = max(
        minimum_active_standoff_m,
        float(shared["minimum_non_target_keepout_radius_m"])
        + float(shared["tracking_margin_m"]),
    )
    return {
        "minimum_static_inflation_m": minimum_static_obstacle_inflation_m(
            robot_radius_m=profile.robot_radius_m,
            tracking_margin_m=DEFAULT_TRACKING_TUBE_RADIUS_M,
            lidar_stop_distance_m=DEFAULT_LIDAR_STOP_DISTANCE_M,
            scan_origin_to_base_offset_m=profile.scan_origin_to_base_offset_m,
            lidar_clearance_margin_m=DEFAULT_LIDAR_CLEARANCE_MARGIN_M,
        ),
        "minimum_active_standoff_m": minimum_active_standoff_m,
        "minimum_collision_standoff_m": minimum_collision_standoff_m,
        "minimum_lidar_standoff_m": minimum_lidar_standoff_m,
        "minimum_candidate_transit_radius_m": (
            minimum_candidate_transit_radius_m
        ),
        "stand_collision_radius_m": stand_collision_radius_m,
        "stand_lidar_radius_m": DEFAULT_CANDIDATE_LIDAR_ENVELOPE_RADIUS_M,
        "stand_base_circumscribed_radius_m": stand_base_circumscribed_radius_m,
        "stand_model_tolerance_m": stand_model_tolerance_m,
    }

def candidate_snapshot_from_registry(
    registry: StandSurveyRegistry,
    plan: CoverageSurveyPlan,
    *,
    registry_path: Path,
    snapshot_id: str,
):
    """Freeze the persistent multi-viewpoint registry for route certification."""

    pending = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status == STATUS_PENDING_CAMERA
    )
    if not pending:
        raise ValueError("stand registry has no pending-camera candidates")
    detector_config_sha256 = payload_sha256(
        {
            "source": "stand_coverage_survey",
            "plan_sha256": coverage_survey_plan_sha256(plan),
            "config": {
                "candidate_merge_distance_m": (
                    plan.config.candidate_merge_distance_m
                ),
                "minimum_candidate_confidence": (
                    plan.config.minimum_candidate_confidence
                ),
                "minimum_distinct_viewpoints": (
                    plan.config.minimum_distinct_viewpoints
                ),
                "minimum_candidate_hits": plan.config.minimum_candidate_hits,
                "lidar_track_morphology_profile": (
                    stand_width_profile_from_radius(
                        plan.config.candidate_radius_m
                    ).to_evidence_dict()
                ),
            },
        }
    )
    return new_candidate_snapshot(
        snapshot_id=snapshot_id,
        created_unix_sec=max(candidate.last_seen_sec for candidate in pending),
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        candidates=(
            FrozenCandidate(
                candidate_uid=candidate.candidate_uid,
                geometry=CandidateGeometry(
                    x_m=candidate.x_m,
                    y_m=candidate.y_m,
                    radius_m=candidate.radius_m,
                    uncertainty_m=candidate.uncertainty_m,
                    keepout_radius_m=candidate.keepout_radius_m,
                ),
                source=CandidateSource(
                    source_kind="lidar/stand_coverage_survey",
                    source_artifact_sha256=_file_sha256(registry_path),
                    detector_config_sha256=detector_config_sha256,
                    observation_ids=candidate.source_observation_ids,
                ),
                confidence=candidate.confidence,
                hit_count=candidate.hit_count,
                first_seen_sec=candidate.first_seen_sec,
                last_seen_sec=candidate.last_seen_sec,
            )
            for candidate in pending
        ),
    )
