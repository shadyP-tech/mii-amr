"""Motion-free coverage replanning and recovery artifact helpers.

This module contains only deterministic planning, artifact I/O, and semantic-log
validation.  It does not import the autonomous parent, ROS clients, subprocess,
or prompting code, and therefore cannot authorize or publish robot motion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.aufgabe04.navigation.foundation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.runtime_localization_reseal import (
    evaluate_runtime_localization_reseal,
)
from scripts.aufgabe04.navigation.coverage.stand_blockage_replan import (
    replan_transient_blockage_from_overlay,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
    coverage_survey_plan_sha256,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    plan_survey_leg_to_viewpoint,
)
from scripts.aufgabe04.navigation.coverage.transient_overlay_resume_state import (
    TransientOverlayResumeState,
    bind_transient_overlay_resume_state_to_diagnostics,
    load_jsonl_event_objects,
    refresh_transient_overlay_resume_state,
    update_transient_overlay_resume_state_from_events,
    write_transient_overlay_resume_state,
)
from scripts.aufgabe04.real_robot.execution.child_runner import (
    DEFAULT_TRACKING_TUBE_RADIUS_M,
    MotionLegOutcome,
)


ROOT = Path(__file__).resolve().parents[4]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def is_resealable_startup_mismatch(outcome: MotionLegOutcome) -> bool:
    """Return whether a no-motion startup rejection may be safely resealed."""

    details = outcome.stop_details
    return (
        outcome.status == "stopped"
        and not outcome.motion_published
        and outcome.stop_reason == "pose outside certified startup segment"
        and details.get("source") == "execution_route_certificate"
        and details.get("phase") == "before_motion_confirmation"
        and isinstance(details.get("route_pose"), dict)
    )


def is_runtime_localization_reseal_required(outcome: MotionLegOutcome) -> bool:
    """Return whether a stopped child satisfies the recovery classifier."""

    return evaluate_runtime_localization_reseal(
        status=outcome.status,
        motion_published=outcome.motion_published,
        stop_details=outcome.stop_details,
    ).eligible


def adopted_blockage_replans_for_run(
    semantic_log: Path,
    *,
    run_id: str,
    start_offset: int = 0,
) -> list[dict[str, object]]:
    """Load post-admission blockage adoptions from one child semantic log."""

    path = Path(semantic_log)
    if not path.exists():
        raise RuntimeError(f"child semantic log is unavailable: {path}")
    try:
        payloads = load_jsonl_event_objects(path, start_offset=start_offset)
    except ValueError as exc:
        raise RuntimeError(
            f"cannot validate adopted blockage-replan state in {path}: {exc}"
        ) from exc
    adopted = [
        dict(payload)
        for payload in payloads
        if payload.get("event") == "transient_navigation_blockage_replanned"
    ]
    for payload in adopted:
        if payload.get("run_id") != run_id:
            raise RuntimeError(
                "adopted blockage-replan event belongs to another child run"
            )
        if payload.get("post_plan_runtime_revalidated") is not True:
            raise RuntimeError(
                "blockage replan lacks post-plan runtime adoption evidence"
            )
        if payload.get("semantic_survey_evidence") is not False:
            raise RuntimeError(
                "blockage replan was incorrectly marked as survey evidence"
            )
    return adopted


def advance_transient_overlay_resume_state(
    *,
    outcome: MotionLegOutcome,
    previous_state: TransientOverlayResumeState | None,
    plan_path: Path,
    leg_index: int,
    target_viewpoint_id: str,
    max_replans: int,
    require_uncertainty_admission: bool,
    artifact_root: Path,
    survey_root: Path,
) -> TransientOverlayResumeState | None:
    """Fold only post-adoption child events into cumulative overlay state."""

    adopted = adopted_blockage_replans_for_run(
        outcome.semantic_log_path,
        run_id=outcome.run_id,
        start_offset=outcome.semantic_log_start_offset,
    )
    if not adopted:
        return previous_state
    plan = load_coverage_survey_plan(plan_path)
    if require_uncertainty_admission:
        for event in adopted:
            if event.get("replacement_route_uncertainty_accepted") is not True:
                raise RuntimeError(
                    "adopted blockage replan lacks accepted uncertainty evidence"
                )
    absolute_events: list[dict[str, object]] = []
    for event in adopted:
        absolute_event = dict(event)
        for field in (
            "replacement_route_csv",
            "transient_obstacle_overlay_json",
        ):
            value = absolute_event.get(field)
            if isinstance(value, str) and not Path(value).is_absolute():
                absolute_event[field] = str(ROOT / value)
        absolute_events.append(absolute_event)
    try:
        return update_transient_overlay_resume_state_from_events(
            absolute_events,
            plan=plan,
            coverage_leg_index=leg_index,
            target_viewpoint_id=target_viewpoint_id,
            max_replans=max_replans,
            artifact_root=artifact_root,
            expected_survey_root=survey_root,
            expected_session_root=artifact_root,
            previous_state=previous_state,
            source_run_id=outcome.run_id,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"adopted transient blockage resume state is invalid: {exc}"
        ) from exc


def startup_reseal_pose(outcome: MotionLegOutcome) -> Pose2D:
    """Parse and validate the rejected route pose used for startup resealing."""

    if not is_resealable_startup_mismatch(outcome):
        raise ValueError("outcome is not a resealable startup mismatch")
    raw = outcome.stop_details["route_pose"]
    assert isinstance(raw, dict)
    pose = Pose2D(float(raw["x_m"]), float(raw["y_m"]), float(raw["yaw_rad"]))
    if not all(
        math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError("startup mismatch pose must be finite")
    return pose


def coverage_reseal_suffix(
    *,
    startup_reseal_index: int,
    runtime_localization_reseal_index: int,
) -> str:
    """Build the stable run/artifact suffix for bounded reseal attempts."""

    if startup_reseal_index < 0 or runtime_localization_reseal_index < 0:
        raise ValueError("reseal indices must be non-negative")
    parts = []
    if startup_reseal_index:
        parts.append(f"startup_reseal_{startup_reseal_index:03d}")
    if runtime_localization_reseal_index:
        parts.append(
            "runtime_localization_reseal_"
            f"{runtime_localization_reseal_index:03d}"
        )
    return "" if not parts else "_" + "_".join(parts)


def load_coverage_plan(plan_path: Path) -> CoverageSurveyPlan:
    """Load a coverage plan through the public replanning boundary."""

    return load_coverage_survey_plan(plan_path)


def _replan_coverage_source_from_pose(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    survey_root: Path,
    plan_path: Path,
    expected_target_viewpoint_id: str,
    current_pose: Pose2D,
    rejected_outcome: MotionLegOutcome,
    reseal_index: int,
    output_dir: Path,
    reseal_kind: str,
    status: str,
) -> dict[str, str]:
    """Rebuild a complete motion-free A* coverage leg from a fresh map pose."""

    if reseal_index <= 0:
        raise ValueError(f"{reseal_kind} reseal index must be positive")
    plan = load_coverage_survey_plan(plan_path)
    progress = load_survey_progress(survey_root / "coverage_progress.json", plan)
    registry = load_stand_survey_registry(survey_root / "stand_registry.json", plan)
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError(f"{reseal_kind} reseal map differs from coverage plan")
    next_leg = plan_survey_leg_to_viewpoint(
        grid,
        plan=plan,
        progress=progress,
        registry=registry,
        current_pose=current_pose,
        target_viewpoint_id=expected_target_viewpoint_id,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    route_path = output_dir / "route.csv"
    diagnostics_path = output_dir / "route_diagnostics.json"
    summary_path = output_dir / f"{reseal_kind}_reseal_summary.json"
    write_route_csv(
        route_path,
        (next_leg.route_result,),
        final_yaw_by_leg={0: next_leg.viewpoint.pose.yaw_rad},
    )
    write_diagnostics_json(
        diagnostics_path,
        (next_leg.route_result,),
        metadata={
            "schema_version": 1,
            "route_kind": "stand_coverage_survey",
            "motion_authorized": False,
            "survey_id": plan.survey_id,
            "plan_sha256": coverage_survey_plan_sha256(plan),
            "map_bundle_sha256": plan.map_bundle_sha256,
            "target_viewpoint_id": next_leg.viewpoint.viewpoint_id,
            "target_pose": {
                "x_m": next_leg.viewpoint.pose.x_m,
                "y_m": next_leg.viewpoint.pose.y_m,
                "yaw_rad": next_leg.viewpoint.pose.yaw_rad,
            },
            "candidate_keepout_count": sum(
                1
                for candidate in registry.candidates
                if candidate.status != "rejected"
            ),
            "unreachable_viewpoint_ids_before_target": list(
                next_leg.unreachable_viewpoint_ids
            ),
            "inflation_radius_m": plan.config.inflation_radius_m,
            "exact_start_connector": next_leg.exact_start_connector.to_metadata(),
            "line_of_sight_route_optimization": (
                next_leg.route_smoothing.to_metadata()
            ),
            "arena_boundary_overlay": True,
            "arena_bounds": plan.arena_bounds.to_metadata(),
            "reseal_kind": reseal_kind,
            f"{reseal_kind}_reseal": True,
            f"{reseal_kind}_reseal_index": reseal_index,
            "rejected_run_id": rejected_outcome.run_id,
            "rejected_stop_details": rejected_outcome.stop_details,
        },
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "motion_published": False,
        "reseal_kind": reseal_kind,
        f"{reseal_kind}_reseal_index": reseal_index,
        "rejected_run_id": rejected_outcome.run_id,
        "target_viewpoint_id": next_leg.viewpoint.viewpoint_id,
        "fresh_start_pose": {
            "x_m": current_pose.x_m,
            "y_m": current_pose.y_m,
            "yaw_rad": current_pose.yaw_rad,
        },
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
    }
    _write_json(summary_path, summary)
    return {
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
        "summary_json": str(summary_path),
    }


def replan_startup_source(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    survey_root: Path,
    plan_path: Path,
    expected_target_viewpoint_id: str,
    current_pose: Pose2D,
    rejected_outcome: MotionLegOutcome,
    reseal_index: int,
    output_dir: Path,
) -> dict[str, str]:
    """Rebuild a complete motion-free A* leg from a rejected live pose."""

    return _replan_coverage_source_from_pose(
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        survey_root=survey_root,
        plan_path=plan_path,
        expected_target_viewpoint_id=expected_target_viewpoint_id,
        current_pose=current_pose,
        rejected_outcome=rejected_outcome,
        reseal_index=reseal_index,
        output_dir=output_dir,
        reseal_kind="startup",
        status="startup_route_replanned",
    )


def replan_runtime_localization_source(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    survey_root: Path,
    plan_path: Path,
    expected_target_viewpoint_id: str,
    current_pose: Pose2D,
    rejected_outcome: MotionLegOutcome,
    reseal_index: int,
    output_dir: Path,
) -> dict[str, str]:
    """Rebuild a complete A* leg after post-motion localization reseal."""

    return _replan_coverage_source_from_pose(
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        survey_root=survey_root,
        plan_path=plan_path,
        expected_target_viewpoint_id=expected_target_viewpoint_id,
        current_pose=current_pose,
        rejected_outcome=rejected_outcome,
        reseal_index=reseal_index,
        output_dir=output_dir,
        reseal_kind="runtime_localization",
        status="runtime_localization_route_replanned",
    )


def replan_source_preserving_transient_overlay(
    *,
    state: TransientOverlayResumeState,
    plan: CoverageSurveyPlan,
    map_yaml: Path,
    semantic_map_id: str,
    survey_root: Path,
    target_viewpoint_id: str,
    current_pose: Pose2D,
    rejected_outcome: MotionLegOutcome,
    output_dir: Path,
    robot_radius_m: float,
    recovery_kind: str,
    artifact_root: Path,
) -> tuple[dict[str, str], TransientOverlayResumeState, Path, str]:
    """Replan from a fresh pose and bind the inherited overlay to diagnostics."""

    replanned = replan_transient_blockage_from_overlay(
        survey_root=survey_root,
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        target_viewpoint_id=target_viewpoint_id,
        current_pose=current_pose,
        overlay_path=Path(state.transient_obstacle_overlay_path),
        output_dir=output_dir,
        robot_radius_m=robot_radius_m,
        rejected_run_id=rejected_outcome.run_id,
        rejected_stop_details=rejected_outcome.stop_details,
        recovery_kind=recovery_kind,
        tracking_tube_radius_m=DEFAULT_TRACKING_TUBE_RADIUS_M,
    )
    refreshed_state = refresh_transient_overlay_resume_state(
        state,
        overlay_path=Path(replanned["transient_obstacle_overlay_json"]),
        plan=plan,
        artifact_root=artifact_root,
    )
    state_path = output_dir / "transient_overlay_resume_state.json"
    state_sha256 = write_transient_overlay_resume_state(
        state_path,
        refreshed_state,
        plan=plan,
    )
    bound_diagnostics = output_dir / "route_diagnostics_resume_bound.json"
    bind_transient_overlay_resume_state_to_diagnostics(
        Path(replanned["diagnostics_json"]),
        bound_diagnostics,
        resume_state_path=state_path,
        plan=plan,
    )
    return (
        {**replanned, "diagnostics_json": str(bound_diagnostics)},
        refreshed_state,
        state_path,
        state_sha256,
    )


__all__ = [
    "adopted_blockage_replans_for_run",
    "advance_transient_overlay_resume_state",
    "coverage_reseal_suffix",
    "is_resealable_startup_mismatch",
    "is_runtime_localization_reseal_required",
    "load_coverage_plan",
    "replan_runtime_localization_source",
    "replan_source_preserving_transient_overlay",
    "replan_startup_source",
    "startup_reseal_pose",
]
