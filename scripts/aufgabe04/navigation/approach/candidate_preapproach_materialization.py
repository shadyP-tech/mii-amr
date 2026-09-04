"""Artifact materialization for a selected camera candidate pre-approach.

This is the only pre-approach module that creates files.  It validates the
prepared route, snapshot, physical-clearance contract, and motion-neutral
selection evidence before creating the output directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Mapping

from scripts.aufgabe04.navigation.approach.candidate_goal_cell_selection import (
    validate_goal_cell_selection_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import (
    compute_candidate_preapproach_plan,
    validate_approach_outside_transit_keepout,
    validate_physical_clearance,
)
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    BacksideAxisFrameProjection,
    BacksideAxisPlanningObservation,
    load_backside_axis_planning_observation,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import (
    CandidatePreapproachPlan,
)
from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import (
    CAMERA_AXIS_FACE_BEARING_MODE,
    ROBOT_TO_STAND_BEARING_MODE,
    seal_detected_stand_preapproach,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    normalize_angle,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
)
from scripts.aufgabe04.navigation.execution.route_context import file_sha256
from scripts.aufgabe04.navigation.foundation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)


def materialize_candidate_preapproach_plan(
    prepared: CandidatePreapproachPlan,
    *,
    snapshot: CandidateSnapshot,
    snapshot_path: Path,
    output_dir: Path,
    physical_clearance: Mapping[str, float],
    axis_observation_path: Path | None = None,
    approach_normal_rad: float | None = None,
    selection_evidence: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Validate, write, and seal one previously computed candidate plan."""

    _validate_selection_evidence(prepared, selection_evidence)
    candidate = snapshot.candidate_for(prepared.candidate_uid)
    if candidate is None:
        raise ValueError("prepared route candidate is absent from snapshot")
    if prepared.map_bundle_sha256 != snapshot.map_bundle_sha256:
        raise ValueError("prepared route map differs from candidate snapshot")
    snapshot_sha256 = candidate_snapshot_sha256(snapshot)
    if prepared.candidate_snapshot_sha256 != snapshot_sha256:
        raise ValueError("prepared route has a stale candidate snapshot binding")
    expected_clearance = validate_physical_clearance(
        physical_clearance,
        inflation_radius_m=prepared.inflation_radius_m,
        candidate_transit_radius_m=prepared.candidate_transit_radius_m,
    )
    prepared_clearance = (
        prepared.minimum_active_standoff_m,
        prepared.minimum_candidate_transit_radius_m,
        prepared.minimum_static_inflation_m,
    )
    if expected_clearance != prepared_clearance:
        raise ValueError("prepared route has the wrong physical-clearance binding")
    validate_approach_outside_transit_keepout(
        approach_offset_m=prepared.approach_offset_m,
        candidate_transit_radius_m=prepared.candidate_transit_radius_m,
        map_resolution_m=prepared.dry_run.grid.metadata.resolution,
    )
    if (approach_normal_rad is None) != (axis_observation_path is None):
        raise ValueError(
            "axis-selected approach requires both normal and observation"
        )
    expected_mode = (
        ROBOT_TO_STAND_BEARING_MODE
        if approach_normal_rad is None
        else CAMERA_AXIS_FACE_BEARING_MODE
    )
    if prepared.approach_bearing_mode != expected_mode:
        raise ValueError("prepared route has the wrong approach bearing mode")
    _validate_source_artifacts(
        snapshot_path=snapshot_path,
        expected_snapshot_sha256=snapshot_sha256,
        axis_observation_path=axis_observation_path,
    )
    axis_observation = None
    if axis_observation_path is not None:
        axis_observation = load_backside_axis_planning_observation(
            axis_observation_path
        )
        validate_backside_axis_candidate_binding(
            axis_observation,
            candidate_uid=prepared.candidate_uid,
            planning_frame=snapshot.planning_frame,
            candidate_x_m=candidate.geometry.x_m,
            candidate_y_m=candidate.geometry.y_m,
        )
    _validate_approach_bearing_binding(
        prepared=prepared,
        candidate_x_m=candidate.geometry.x_m,
        candidate_y_m=candidate.geometry.y_m,
        approach_normal_rad=approach_normal_rad,
        axis_observation=axis_observation,
    )
    _validate_goal_cell_policy_binding(
        prepared=prepared,
        candidate_x_m=candidate.geometry.x_m,
        candidate_y_m=candidate.geometry.y_m,
        approach_normal_rad=approach_normal_rad,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    route_csv = output_dir / "route.csv"
    diagnostics_json = output_dir / "route_diagnostics.json"
    pipeline_summary = output_dir / "pipeline_summary.json"
    local_snapshot = output_dir / "candidate_snapshot.json"
    shutil.copyfile(snapshot_path, local_snapshot)
    local_axis_observation = None
    if axis_observation_path is not None:
        local_axis_observation = output_dir / "axis_observation.json"
        shutil.copyfile(axis_observation_path, local_axis_observation)

    route_results = (prepared.result,)
    write_route_csv(
        route_csv,
        route_results,
        final_yaw_by_leg={0: prepared.terminal_yaw_rad},
    )
    metadata = dict(prepared.dry_run.metadata)
    planning_order = (
        "route-aware-camera-selection"
        if selection_evidence is not None
        else (
            "bounded-opposite-face"
            if axis_observation_path is not None
            else "specified-candidate"
        )
    )
    metadata.update(
        {
            "source": "lidar_detected_stand_exploration",
            "order": planning_order,
            "plan_mode": "next-candidate",
            "stand_count": 1,
            "candidate_transit_radius_m": prepared.candidate_transit_radius_m,
            "inflation_radius_m": prepared.inflation_radius_m,
            "approach_offset_m": prepared.approach_offset_m,
            "approach_bearing_mode": prepared.approach_bearing_mode,
            "physical_clearance_enforced": True,
            "physical_clearance": dict(physical_clearance),
            "candidate_snapshot_json": str(local_snapshot),
            "candidate_snapshot_sha256": snapshot_sha256,
            "map_bundle_sha256": prepared.map_bundle_sha256,
            "planning_frame": snapshot.planning_frame,
            "selected_candidate_stand_id": prepared.candidate_uid,
            "exact_start_connector": prepared.connector.to_metadata(),
            "route_start_pose_provenance": {
                "source": "autonomous_candidate_current_pose",
                "planning_frame": snapshot.planning_frame,
                "pose": {
                    "x_m": prepared.start.x_m,
                    "y_m": prepared.start.y_m,
                    "yaw_rad": prepared.start.yaw_rad,
                },
            },
            "line_of_sight_route_optimization": {
                "enabled": True,
                "legs": [prepared.smoothing.to_metadata()],
                "input_point_count": prepared.smoothing.input_point_count,
                "output_point_count": prepared.smoothing.output_point_count,
                "optimized_leg_count": int(prepared.smoothing.optimized),
            },
            "selected_approach_pose": {
                "x_m": prepared.selected_approach_pose.x_m,
                "y_m": prepared.selected_approach_pose.y_m,
                "yaw_rad": prepared.selected_approach_pose.yaw_rad,
            },
            "candidate_route_metrics": {
                "route_length_m": prepared.route_length_m,
                "initial_turn_rad": prepared.initial_turn_rad,
                "turn_burden_rad": prepared.turn_burden_rad,
                "distance_to_stand_m": prepared.distance_to_stand_m,
                "endpoint_standoff_m": prepared.endpoint_standoff_m,
                "inside_requested_standoff": prepared.inside_requested_standoff,
            },
        }
    )
    if selection_evidence is not None:
        metadata["camera_candidate_selection"] = dict(selection_evidence)
    if prepared.goal_cell_selection is not None:
        metadata["goal_cell_selection"] = (
            prepared.goal_cell_selection.to_metadata()
        )
    if local_axis_observation is not None:
        axis_metadata: dict[str, object] = {
            "axis_observation_json": str(local_axis_observation.resolve()),
            "axis_observation_sha256": file_sha256(local_axis_observation),
            "selected_face_normal_rad": normalize_angle(
                float(approach_normal_rad)
            ),
        }
        if isinstance(axis_observation, BacksideAxisFrameProjection):
            axis_metadata.update(
                {
                    "axis_evidence_kind": "backside_axis_frame_projection",
                    "source_axis_observation_json": str(
                        axis_observation.source_axis_observation_path
                    ),
                    "source_axis_observation_sha256": (
                        axis_observation.source_axis_observation_sha256
                    ),
                    "axis_frame_projection_sha256": (
                        axis_observation.projection_sha256
                    ),
                }
            )
        else:
            axis_metadata["axis_evidence_kind"] = "native_backside_axis_observation"
        metadata.update(axis_metadata)
    write_diagnostics_json(diagnostics_json, route_results, metadata=metadata)
    _write_json(
        pipeline_summary,
        {
            "schema_version": 2,
            "status": "observe_and_plan_complete",
            "motion_published": False,
            "selected_candidate_uid": prepared.candidate_uid,
            "selected_approach_pose": metadata["selected_approach_pose"],
            "physical_clearance": dict(physical_clearance),
            "selection_strategy": planning_order,
            "planning_order": planning_order,
        },
    )
    return dict(seal_detected_stand_preapproach(pipeline_root=output_dir))


def plan_candidate_preapproach(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    plan: CoverageSurveyPlan,
    snapshot: CandidateSnapshot,
    snapshot_path: Path,
    candidate_uid: str,
    start: Pose2D,
    output_dir: Path,
    approach_offset_m: float,
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
    physical_clearance: Mapping[str, float],
    approach_normal_rad: float | None = None,
    axis_observation_path: Path | None = None,
    prepared_plan: CandidatePreapproachPlan | None = None,
    selection_evidence: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Compute when needed, then materialize the exact selected route."""

    if (approach_normal_rad is None) != (axis_observation_path is None):
        raise ValueError(
            "axis-selected approach requires both normal and observation"
        )
    selected = prepared_plan or compute_candidate_preapproach_plan(
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        plan=plan,
        snapshot=snapshot,
        candidate_uid=candidate_uid,
        start=start,
        approach_offset_m=approach_offset_m,
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
        physical_clearance=physical_clearance,
        approach_normal_rad=approach_normal_rad,
    )
    _validate_prepared_plan_binding(
        selected,
        candidate_uid=candidate_uid,
        start=start,
        approach_offset_m=approach_offset_m,
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
    )
    return materialize_candidate_preapproach_plan(
        selected,
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        physical_clearance=physical_clearance,
        axis_observation_path=axis_observation_path,
        approach_normal_rad=approach_normal_rad,
        selection_evidence=selection_evidence,
    )


def _validate_selection_evidence(
    prepared: CandidatePreapproachPlan,
    selection_evidence: Mapping[str, object] | None,
) -> None:
    if selection_evidence is None:
        return
    if selection_evidence.get("selected_candidate_uid") != prepared.candidate_uid:
        raise ValueError(
            "selection_evidence selected_candidate_uid differs from prepared route"
        )
    motion_authorized = selection_evidence.get("motion_authorized")
    if motion_authorized is not None and motion_authorized is not False:
        raise ValueError(
            "selection_evidence motion_authorized must be absent, null, or false"
        )


def _validate_source_artifacts(
    *,
    snapshot_path: Path,
    expected_snapshot_sha256: str,
    axis_observation_path: Path | None,
) -> None:
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.is_file():
        raise ValueError("candidate snapshot source artifact is missing")
    source_snapshot = load_candidate_snapshot(snapshot_path)
    if candidate_snapshot_sha256(source_snapshot) != expected_snapshot_sha256:
        raise ValueError("candidate snapshot source artifact has the wrong binding")
    if axis_observation_path is not None and not Path(
        axis_observation_path
    ).is_file():
        raise ValueError("axis observation source artifact is missing")


def _validate_approach_bearing_binding(
    *,
    prepared: CandidatePreapproachPlan,
    candidate_x_m: float,
    candidate_y_m: float,
    approach_normal_rad: float | None,
    axis_observation: BacksideAxisPlanningObservation | None,
) -> None:
    if approach_normal_rad is None:
        expected_bearing_rad = math.atan2(
            candidate_y_m - prepared.start.y_m,
            candidate_x_m - prepared.start.x_m,
        )
    else:
        if not math.isfinite(approach_normal_rad):
            raise ValueError("approach face normal must be finite")
        if axis_observation is None:
            raise ValueError(
                "axis-selected approach lacks a validated observation"
            )
        observed_normal_rad = axis_observation.opposite_face_normal_rad
        if abs(
            normalize_angle(observed_normal_rad - approach_normal_rad)
        ) > 1.0e-9:
            raise ValueError(
                "axis observation no longer resolves to the prepared face normal"
            )
        expected_bearing_rad = normalize_angle(approach_normal_rad + math.pi)
    if abs(
        normalize_angle(
            prepared.approach_bearing_rad - expected_bearing_rad
        )
    ) > 1.0e-9:
        raise ValueError("prepared route has the wrong approach-bearing binding")


def validate_backside_axis_candidate_binding(
    observation: BacksideAxisPlanningObservation,
    *,
    candidate_uid: str,
    planning_frame: str,
    candidate_x_m: float,
    candidate_y_m: float,
    center_tolerance_m: float = 1.0e-6,
) -> None:
    """Bind a validated backside receipt to the exact frozen candidate."""

    if observation.stand_id != candidate_uid:
        raise ValueError("axis observation stand ID does not match candidate")
    if observation.planning_frame != planning_frame:
        raise ValueError(
            "axis observation planning frame does not match snapshot"
        )
    if (
        math.hypot(
            observation.stand_x_m - candidate_x_m,
            observation.stand_y_m - candidate_y_m,
        )
        > center_tolerance_m
    ):
        raise ValueError(
            "axis observation stand center does not match candidate geometry"
        )


def _validate_goal_cell_policy_binding(
    *,
    prepared: CandidatePreapproachPlan,
    candidate_x_m: float,
    candidate_y_m: float,
    approach_normal_rad: float | None,
) -> None:
    evidence = prepared.goal_cell_selection
    if approach_normal_rad is None:
        if evidence is not None:
            raise ValueError(
                "robot-bearing approach must not carry axis goal-cell selection"
            )
        return
    if evidence is None:
        raise ValueError("camera-axis approach lacks goal-cell selection evidence")
    expected_requested_goal = Pose2D(
        candidate_x_m
        + prepared.approach_offset_m * math.cos(approach_normal_rad),
        candidate_y_m
        + prepared.approach_offset_m * math.sin(approach_normal_rad),
        prepared.approach_bearing_rad,
    )
    validate_goal_cell_selection_binding(
        evidence,
        base_costmap=prepared.dry_run.base_costmap,
        planning_costmap=prepared.dry_run.planning_costmap,
        result=prepared.result,
        expected_requested_goal=expected_requested_goal,
        stand=Pose2D(candidate_x_m, candidate_y_m, 0.0),
        minimum_standoff_m=prepared.minimum_active_standoff_m,
    )


def _validate_prepared_plan_binding(
    prepared: CandidatePreapproachPlan,
    *,
    candidate_uid: str,
    start: Pose2D,
    approach_offset_m: float,
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
) -> None:
    expected = (
        prepared.candidate_uid == candidate_uid
        and prepared.start == start
        and math.isclose(
            prepared.approach_offset_m,
            approach_offset_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            prepared.inflation_radius_m,
            inflation_radius_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            prepared.candidate_transit_radius_m,
            candidate_transit_radius_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    if not expected:
        raise ValueError(
            "prepared candidate route differs from materialization request"
        )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "materialize_candidate_preapproach_plan",
    "plan_candidate_preapproach",
    "validate_backside_axis_candidate_binding",
]
