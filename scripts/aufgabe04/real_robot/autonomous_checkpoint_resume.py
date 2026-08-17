"""Fail-closed restoration and fresh replanning from a coverage checkpoint.

Checkpoint evidence is never a motion permit.  This module restores only the
survey state needed to continue and always computes a new exact-start A* leg
from a freshly admitted map pose in a new session directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.map_io import (
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    CoverageSurveyPlan,
    coverage_survey_plan_sha256,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    plan_next_survey_leg,
)
from scripts.aufgabe04.real_robot.autonomous_session_manifest import (
    AutonomousSessionManifest,
    AutonomousSessionManifestError,
    admit_autonomous_session_manifest,
    autonomous_session_manifest_sha256,
)


@dataclass(frozen=True)
class AdmittedCoverageResume:
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest: AutonomousSessionManifest


@dataclass(frozen=True)
class RestoredCoverageResume:
    survey_root: Path
    plan: CoverageSurveyPlan
    plan_path: Path
    leg_index: int
    target_viewpoint_id: str
    route_csv: Path
    diagnostics_json: Path
    parent_checkpoint_path: Path
    parent_checkpoint_sha256: str


def admit_coverage_resume(
    checkpoint_path: Path,
    *,
    new_session_id: str,
    robot_id: str,
    robot_profile_sha256: str,
    calibration_profile_sha256: str,
    physical_site_sha256: str,
    map_bundle_sha256: str,
    config_sha256: str,
) -> AdmittedCoverageResume:
    """Admit exact checkpoint bytes and current hardware/config identities."""

    checkpoint_candidate = Path(checkpoint_path)
    if checkpoint_candidate.is_symlink():
        raise AutonomousSessionManifestError(
            "artifact_unavailable",
            "resume checkpoint path must not be a symlink",
        )
    try:
        source = checkpoint_candidate.resolve(strict=True)
    except OSError as exc:
        raise AutonomousSessionManifestError(
            "artifact_unavailable",
            f"resume checkpoint is unavailable: {checkpoint_candidate}",
        ) from exc
    if checkpoint_candidate.is_absolute() and checkpoint_candidate != source:
        raise AutonomousSessionManifestError(
            "artifact_unavailable",
            "resume checkpoint path must be canonical and must not traverse symlinks",
        )
    manifest = admit_autonomous_session_manifest(source)
    if new_session_id == manifest.session_id:
        raise AutonomousSessionManifestError(
            "provenance_mismatch",
            "resume must use a new session_id and fresh child identities",
        )
    expected = {
        "robot_id": robot_id,
        "robot_profile_sha256": robot_profile_sha256,
        "calibration_profile_sha256": calibration_profile_sha256,
        "physical_site_sha256": physical_site_sha256,
        "map_bundle_sha256": map_bundle_sha256,
        "config_sha256": config_sha256,
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise AutonomousSessionManifestError(
                "provenance_mismatch",
                f"resume {field} differs from checkpoint",
            )
    return AdmittedCoverageResume(
        checkpoint_path=source,
        checkpoint_sha256=autonomous_session_manifest_sha256(manifest),
        manifest=manifest,
    )


def restore_and_replan_coverage_resume(
    admitted: AdmittedCoverageResume,
    *,
    survey_root: Path,
    map_yaml: Path,
    semantic_map_id: str,
    current_pose: Pose2D,
) -> RestoredCoverageResume:
    """Restore checkpoint snapshots and create a fresh non-authorizing leg."""

    destination = Path(survey_root)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise AutonomousSessionManifestError(
            "immutable_conflict",
            f"refusing to reuse resumed survey root: {destination}",
        ) from exc

    manifest = admitted.manifest
    restored = {
        "coverage_plan": destination / "coverage_plan.json",
        "coverage_progress": destination / "coverage_progress.json",
        "survey_summary": destination / "survey_summary.json",
        "stand_registry": destination / "stand_registry.json",
        "lidar_observer_summary": (
            destination
            / "resume_provenance"
            / "prior_lidar_observer_summary.json"
        ),
    }
    restored["lidar_observer_summary"].parent.mkdir(parents=True)
    for name, output in restored.items():
        reference = getattr(manifest, name)
        _copy_verified(reference.path, reference.sha256, output, name)

    plan_path = restored["coverage_plan"]
    plan = load_coverage_survey_plan(plan_path)
    progress = load_survey_progress(restored["coverage_progress"], plan)
    registry = load_stand_survey_registry(restored["stand_registry"], plan)
    if len(progress.visited_viewpoint_ids) != manifest.completed_coverage_legs:
        raise AutonomousSessionManifestError(
            "invalid_cursor",
            "checkpoint progress count differs from completed coverage legs",
        )
    summary = _load_json_object(restored["survey_summary"], "survey summary")
    if summary.get("next_viewpoint_id") != manifest.next_viewpoint_id:
        raise AutonomousSessionManifestError(
            "invalid_cursor",
            "checkpoint summary next viewpoint differs from manifest cursor",
        )

    grid, map_bundle = load_occupancy_grid_with_bundle(
        Path(map_yaml),
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if (
        plan.map_bundle_sha256 != manifest.map_bundle_sha256
        or map_bundle.bundle_sha256 != manifest.map_bundle_sha256
    ):
        raise AutonomousSessionManifestError(
            "provenance_mismatch",
            "resume map bundle differs from checkpoint plan",
        )
    next_leg = plan_next_survey_leg(
        grid,
        plan=plan,
        progress=progress,
        registry=registry,
        current_pose=current_pose,
    )
    if next_leg is None or (
        next_leg.viewpoint.viewpoint_id != manifest.next_viewpoint_id
    ):
        raise AutonomousSessionManifestError(
            "invalid_cursor",
            "fresh planner did not preserve the checkpoint next viewpoint",
        )

    leg_index = manifest.completed_coverage_legs
    legs_root = destination / "legs"
    route_path = legs_root / f"leg_{leg_index:03d}_route.csv"
    diagnostics_path = legs_root / f"leg_{leg_index:03d}_diagnostics.json"
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
            "exact_start_connector": (
                next_leg.exact_start_connector.to_metadata()
            ),
            "arena_boundary_overlay": True,
            "arena_bounds": plan.arena_bounds.to_metadata(),
            "resume_checkpoint_manifest": str(admitted.checkpoint_path),
            "resume_checkpoint_manifest_sha256": (
                admitted.checkpoint_sha256
            ),
            "resume_parent_session_id": manifest.session_id,
            "resume_motion_authorized": False,
            "fresh_start_pose": {
                "x_m": current_pose.x_m,
                "y_m": current_pose.y_m,
                "yaw_rad": current_pose.yaw_rad,
            },
        },
    )
    receipt = {
        "schema_version": 1,
        "status": "coverage_checkpoint_restored_and_replanned",
        "parent_session_id": manifest.session_id,
        "parent_checkpoint_manifest": str(admitted.checkpoint_path),
        "parent_checkpoint_manifest_sha256": admitted.checkpoint_sha256,
        "completed_coverage_legs": leg_index,
        "next_viewpoint_id": manifest.next_viewpoint_id,
        "fresh_route_csv": str(route_path),
        "fresh_diagnostics_json": str(diagnostics_path),
        "motion_authorized": False,
        "old_motion_permits_reused": False,
    }
    receipt_path = destination / "resume_admission.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RestoredCoverageResume(
        survey_root=destination,
        plan=plan,
        plan_path=plan_path,
        leg_index=leg_index,
        target_viewpoint_id=manifest.next_viewpoint_id,
        route_csv=route_path,
        diagnostics_json=diagnostics_path,
        parent_checkpoint_path=admitted.checkpoint_path,
        parent_checkpoint_sha256=admitted.checkpoint_sha256,
    )


def _copy_verified(
    source_text: str,
    expected_sha256: str,
    destination: Path,
    name: str,
) -> None:
    source = Path(source_text)
    if source.is_symlink() or not source.is_file():
        raise AutonomousSessionManifestError(
            "artifact_unavailable", f"resume {name} is unavailable"
        )
    data = source.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise AutonomousSessionManifestError(
            "hash_mismatch", f"resume {name} hash mismatch"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AutonomousSessionManifestError(
            "immutable_conflict", f"cannot restore {name}: {exc}"
        ) from exc


def _load_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutonomousSessionManifestError(
            "artifact_corrupt", f"invalid {name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AutonomousSessionManifestError(
            "artifact_corrupt", f"{name} must contain an object"
        )
    return payload


__all__ = [
    "AdmittedCoverageResume",
    "RestoredCoverageResume",
    "admit_coverage_resume",
    "restore_and_replan_coverage_resume",
]
