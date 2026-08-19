"""Fuse one stopped LiDAR observation epoch into a coverage survey.

This command consumes the receipt produced by ``stand_explorer_node.py
--summary-json``.  It marks one planned viewpoint visited, updates stable
candidate IDs and keepouts, and, on the legacy CLI path, writes a newly planned
next leg.  Importable callers can split epoch commit from fresh-localization
next-leg planning.  This module does not publish velocity or execute a leg.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.content_hashed_evidence import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_first_detected_station import (
    validate_observation_provenance,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    coverage_survey_plan_sha256,
    fuse_confirmed_stands,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    mark_viewpoint_visited,
    plan_next_survey_leg,
    survey_status,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.navigation.stand_candidate_static_map_admission import (
    evaluate_stand_candidate_static_map_admission,
)
from scripts.aufgabe04.perception.stand_confirmation import (
    StandConfirmationAccumulator,
    StandConfirmationConfig,
)
from scripts.aufgabe04.perception.stand_observation import (
    load_observation_jsonl,
    validated_observation_stream_clock,
)
from scripts.aufgabe04.real_robot.autonomous_session_manifest import (
    admit_autonomous_session_manifest,
    autonomous_session_manifest_sha256,
)


@dataclass(frozen=True)
class PreparedCoverageStop:
    survey_root: Path
    plan_path: Path
    progress_path: Path
    registry_path: Path
    summary_path: Path
    epoch_path: Path
    map_yaml: Path
    semantic_map_id: str
    grid: object
    map_bundle: object
    plan: object
    progress: object
    registry: object
    viewpoint: object
    scan_pose: Pose2D
    viewpoint_error_m: float
    observer_summary_json: Path
    observations_path: Path
    observer_summary: dict[str, object]
    observations: object
    stands: object
    next_leg: object | None
    raw_stands: object = ()
    static_map_candidate_admission: object | None = None
    static_map_candidate_admission_payload: dict[str, object] | None = None
    static_map_candidate_admission_path: Path | None = None
    static_map_candidate_admission_sha256: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument("--viewpoint-id", required=True)
    parser.add_argument("--observer-summary-json", required=True, type=Path)
    parser.add_argument(
        "--observations-jsonl",
        type=Path,
        default=None,
        help=(
            "Defaults to output_jsonl in the observer summary. The file may be "
            "absent only when accepted_observation_count is zero."
        ),
    )
    parser.add_argument("--arrival-tolerance-m", type=float, default=0.18)
    parser.add_argument(
        "--scan-to-base-position-offset-m",
        type=float,
        default=0.05,
        help="Allowed planar scan-frame mounting offset from base_footprint.",
    )
    return parser


def _load_summary(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid observer summary: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("observer summary must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported observer summary schema")
    if payload.get("motion_published") is not False:
        raise ValueError("observer summary must declare motion_published=false")
    processed = payload.get("processed_scan_count")
    if type(processed) is not int or processed <= 0:
        raise ValueError("observer summary contains no processed scans")
    return payload


def _summary_scan_pose(payload: dict[str, object]) -> Pose2D:
    raw = payload.get("scan_frame_pose_in_planning_frame")
    if not isinstance(raw, dict):
        raise ValueError("observer summary has no final scan-frame pose")
    try:
        pose = Pose2D(
            float(raw["x_m"]),
            float(raw["y_m"]),
            float(raw["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("observer summary scan-frame pose is invalid") from exc
    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)):
        raise ValueError("observer summary scan-frame pose must be finite")
    return pose


def _observations_from_epoch(
    *,
    summary: dict[str, object],
    observations_path: Path,
    map_yaml: Path,
    map_bundle,
    plan,
):
    accepted_count = summary.get("accepted_observation_count")
    if type(accepted_count) is not int or accepted_count < 0:
        raise ValueError("observer summary accepted_observation_count is invalid")
    if accepted_count == 0:
        if observations_path.exists() and observations_path.stat().st_size > 0:
            raise ValueError(
                "observer summary reports no observations but JSONL is non-empty"
            )
        return ()
    if not observations_path.exists():
        raise ValueError("observer summary reports observations but JSONL is missing")
    observations = load_observation_jsonl(observations_path)
    if len(observations) != accepted_count:
        raise ValueError(
            "observer summary/JSONL observation count mismatch: "
            f"{accepted_count} != {len(observations)}"
        )
    runtime = summary.get("runtime_config")
    timing = summary.get("timing_limits")
    if not isinstance(runtime, dict) or not isinstance(timing, dict):
        raise ValueError("observer summary runtime/timing metadata is invalid")
    required_base_frame = str(runtime.get("base_frame", ""))
    required_localization_source = str(runtime.get("localization_source", ""))
    for observation in observations:
        validate_observation_provenance(
            observation,
            map_yaml=map_yaml,
            required_map_frame=plan.planning_frame,
            required_base_frame=required_base_frame,
            required_localization_source=required_localization_source,
            max_tf_age_sec=float(timing["max_tf_age_sec"]),
            max_scan_age_sec=float(timing["max_scan_age_sec"]),
            max_future_timestamp_sec=float(timing["max_future_timestamp_sec"]),
            max_tf_scan_skew_sec=float(timing["max_tf_scan_skew_sec"]),
            expected_map_yaml_sha256=map_bundle.yaml_sha256,
            expected_map_bundle_sha256=map_bundle.bundle_sha256,
        )
    validated_observation_stream_clock(observations)
    return observations


def _epoch_stands(observations, plan):
    if not observations:
        return ()
    accumulator = StandConfirmationAccumulator(
        config=StandConfirmationConfig(
            merge_distance_m=plan.config.candidate_merge_distance_m,
            min_hits=plan.config.minimum_candidate_hits,
            max_age_sec=plan.config.observation_epoch_max_age_sec,
            min_confidence=plan.config.minimum_candidate_confidence,
            min_boundary_clearance_m=(
                plan.config.minimum_boundary_clearance_m
            ),
        ),
        arena_bounds=plan.arena_bounds,
    )
    return accumulator.add_observations(observations)


def _validate_pose(pose: Pose2D, name: str) -> None:
    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)):
        raise ValueError(f"{name} must be finite")


def _validate_localization_evidence_path(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a normal file")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, *, label: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value


def _validate_file_sha256(path: Path, expected_sha256: str, *, label: str) -> str:
    expected_sha256 = _validate_sha256(expected_sha256, label=f"{label} SHA-256")
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    return actual_sha256


def _next_leg_artifact_paths(
    survey_root: Path,
    progress,
) -> tuple[Path, Path]:
    leg_index = len(progress.visited_viewpoint_ids)
    legs_dir = survey_root / "legs"
    return (
        legs_dir / f"leg_{leg_index:03d}_route.csv",
        legs_dir / f"leg_{leg_index:03d}_diagnostics.json",
    )


def _validate_next_leg_artifacts_absent(route_path: Path, diagnostics_path: Path) -> None:
    if route_path.exists() or diagnostics_path.exists():
        raise ValueError("refusing to overwrite next-leg artifacts")


def _prepare_stand_coverage_stop(
    *,
    survey_root: Path,
    map_yaml: Path,
    viewpoint_id: str,
    observer_summary_json: Path,
    semantic_map_id: str = "",
    observations_jsonl: Path | None = None,
    arrival_tolerance_m: float = 0.18,
    scan_to_base_position_offset_m: float = 0.05,
    next_leg_start_pose: Pose2D | None = None,
    next_leg_localization_evidence_json: Path | None = None,
) -> PreparedCoverageStop:
    survey_root = Path(survey_root)
    map_yaml = Path(map_yaml)
    observer_summary_json = Path(observer_summary_json)
    if observations_jsonl is not None:
        observations_jsonl = Path(observations_jsonl)
    if next_leg_localization_evidence_json is not None:
        next_leg_localization_evidence_json = Path(
            next_leg_localization_evidence_json
        )

    if not math.isfinite(arrival_tolerance_m) or arrival_tolerance_m <= 0.0:
        raise ValueError("arrival tolerance must be finite and positive")
    if (
        not math.isfinite(scan_to_base_position_offset_m)
        or scan_to_base_position_offset_m < 0.0
    ):
        raise ValueError(
            "scan-to-base position offset must be finite and non-negative"
        )
    if next_leg_start_pose is not None:
        _validate_pose(next_leg_start_pose, "next-leg start pose")
    if (
        next_leg_localization_evidence_json is not None
        and next_leg_start_pose is None
    ):
        raise ValueError(
            "next-leg localization evidence requires a fresh start pose"
        )
    if next_leg_localization_evidence_json is not None:
        next_leg_localization_evidence_json = _validate_localization_evidence_path(
            next_leg_localization_evidence_json,
            label="next-leg localization evidence",
        )
    plan_path = survey_root / "coverage_plan.json"
    progress_path = survey_root / "coverage_progress.json"
    registry_path = survey_root / "stand_registry.json"
    summary_path = survey_root / "survey_summary.json"
    plan = load_coverage_survey_plan(plan_path)
    progress = load_survey_progress(progress_path, plan)
    registry = load_stand_survey_registry(registry_path, plan)
    if viewpoint_id in progress.visited_viewpoint_ids:
        raise ValueError(f"viewpoint {viewpoint_id!r} is already visited")
    viewpoint = plan.viewpoint_for(viewpoint_id)
    if viewpoint is None:
        raise ValueError(f"unknown viewpoint {viewpoint_id!r}")

    resolved_semantic_map_id = semantic_map_id or map_yaml.stem
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=resolved_semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("runtime map bundle differs from coverage plan")
    observer_summary = _load_summary(observer_summary_json)
    if observer_summary.get("map_bundle_sha256") != plan.map_bundle_sha256:
        raise ValueError("observer summary map bundle differs from coverage plan")
    if observer_summary.get("planning_frame") != plan.planning_frame:
        raise ValueError("observer summary planning frame differs from plan")
    scan_pose = _summary_scan_pose(observer_summary)
    viewpoint_error_m = math.hypot(
        scan_pose.x_m - viewpoint.pose.x_m,
        scan_pose.y_m - viewpoint.pose.y_m,
    )
    allowed_error_m = arrival_tolerance_m + scan_to_base_position_offset_m
    if viewpoint_error_m > allowed_error_m + 1.0e-12:
        raise ValueError(
            "observation was not captured at the planned viewpoint: "
            f"error={viewpoint_error_m:.3f} m > {allowed_error_m:.3f} m"
        )

    summary_output = Path(str(observer_summary.get("output_jsonl", "")))
    observations_path = observations_jsonl or summary_output
    if not str(observations_path):
        raise ValueError("observer summary has no observations JSONL path")
    if observations_path.resolve() != summary_output.resolve():
        raise ValueError("observations path differs from the observer summary output")
    observations = _observations_from_epoch(
        summary=observer_summary,
        observations_path=observations_path,
        map_yaml=map_yaml,
        map_bundle=map_bundle,
        plan=plan,
    )
    raw_stands = _epoch_stands(observations, plan)
    static_map_candidate_admission = (
        evaluate_stand_candidate_static_map_admission(
            Costmap.from_occupancy_grid(grid).with_arena_bounds(
                plan.arena_bounds
            ),
            raw_stands,
            candidate_radius_m=plan.config.candidate_radius_m,
            candidate_uncertainty_m=plan.config.candidate_uncertainty_m,
        )
    )
    static_map_candidate_admission_payload = {
        **static_map_candidate_admission.to_evidence_dict(),
        "survey_id": plan.survey_id,
        "viewpoint_id": viewpoint.viewpoint_id,
        "planning_frame": plan.planning_frame,
        "map_bundle_sha256": plan.map_bundle_sha256,
        "coverage_plan_sha256": coverage_survey_plan_sha256(plan),
    }
    static_map_candidate_admission_sha256 = payload_sha256(
        static_map_candidate_admission_payload
    )
    static_map_candidate_admission_path = (
        survey_root
        / "epochs"
        / (
            f"{viewpoint.viewpoint_id}_static_map_candidate_admission_"
            f"{static_map_candidate_admission_sha256}.json"
        )
    )
    stands = static_map_candidate_admission.admitted_stands
    registry = fuse_confirmed_stands(
        registry,
        stands,
        viewpoint_id=viewpoint.viewpoint_id,
        config=plan.config,
    )
    progress = mark_viewpoint_visited(plan, progress, viewpoint.viewpoint_id)

    # Prove that the enlarged keepout registry still admits a next leg before
    # committing the mutable epoch/progress artifacts.
    next_leg = plan_next_survey_leg(
        grid,
        plan=plan,
        progress=progress,
        registry=registry,
        current_pose=next_leg_start_pose or scan_pose,
    )

    epoch_path = survey_root / "epochs" / f"{viewpoint.viewpoint_id}.json"
    if epoch_path.exists():
        raise ValueError(f"refusing to overwrite survey epoch: {epoch_path}")
    return PreparedCoverageStop(
        survey_root=survey_root,
        plan_path=plan_path,
        progress_path=progress_path,
        registry_path=registry_path,
        summary_path=summary_path,
        epoch_path=epoch_path,
        map_yaml=map_yaml,
        semantic_map_id=resolved_semantic_map_id,
        grid=grid,
        map_bundle=map_bundle,
        plan=plan,
        progress=progress,
        registry=registry,
        viewpoint=viewpoint,
        scan_pose=scan_pose,
        viewpoint_error_m=viewpoint_error_m,
        observer_summary_json=observer_summary_json,
        observations_path=observations_path,
        observer_summary=observer_summary,
        observations=observations,
        stands=stands,
        next_leg=next_leg,
        raw_stands=raw_stands,
        static_map_candidate_admission=static_map_candidate_admission,
        static_map_candidate_admission_payload=(
            static_map_candidate_admission_payload
        ),
        static_map_candidate_admission_path=(
            static_map_candidate_admission_path
        ),
        static_map_candidate_admission_sha256=(
            static_map_candidate_admission_sha256
        ),
    )


def _write_committed_stop_state(
    prepared: PreparedCoverageStop,
    *,
    next_leg_start_pose: Pose2D,
    next_leg_start_pose_source: str,
    next_leg_localization_evidence_json: Path | None,
    next_route_path: Path | None,
    next_diagnostics_path: Path | None,
    write_summary: bool = True,
) -> dict[str, object]:
    prepared.epoch_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        prepared.static_map_candidate_admission is None
        or prepared.static_map_candidate_admission_payload is None
        or prepared.static_map_candidate_admission_path is None
        or prepared.static_map_candidate_admission_sha256 is None
    ):
        raise ValueError("prepared stop has no static-map candidate evidence")
    written_admission_sha256 = write_content_hashed_json(
        prepared.static_map_candidate_admission_path,
        prepared.static_map_candidate_admission_payload,
        hash_field="static_map_candidate_admission_sha256",
    )
    if written_admission_sha256 != prepared.static_map_candidate_admission_sha256:
        raise RuntimeError("static-map candidate evidence hash changed while writing")
    admission = prepared.static_map_candidate_admission
    epoch = {
        "schema_version": 1,
        "survey_id": prepared.plan.survey_id,
        "viewpoint_id": prepared.viewpoint.viewpoint_id,
        "planned_pose": {
            "x_m": prepared.viewpoint.pose.x_m,
            "y_m": prepared.viewpoint.pose.y_m,
            "yaw_rad": prepared.viewpoint.pose.yaw_rad,
        },
        "observed_scan_pose": {
            "x_m": prepared.scan_pose.x_m,
            "y_m": prepared.scan_pose.y_m,
            "yaw_rad": prepared.scan_pose.yaw_rad,
        },
        "next_leg_start_pose": {
            "x_m": next_leg_start_pose.x_m,
            "y_m": next_leg_start_pose.y_m,
            "yaw_rad": next_leg_start_pose.yaw_rad,
        },
        "next_leg_start_pose_source": next_leg_start_pose_source,
        "next_leg_localization_evidence_json": (
            None
            if next_leg_localization_evidence_json is None
            else str(next_leg_localization_evidence_json)
        ),
        "viewpoint_error_m": prepared.viewpoint_error_m,
        "observer_summary_json": str(prepared.observer_summary_json),
        "observations_jsonl": str(prepared.observations_path),
        "processed_scan_count": prepared.observer_summary["processed_scan_count"],
        "accepted_observation_count": len(prepared.observations),
        "confirmed_epoch_candidate_count": len(prepared.raw_stands),
        "static_map_candidate_admitted_count": len(admission.admitted_stands),
        "static_map_candidate_rejected_count": len(admission.rejected_stands),
        "static_map_candidate_admission_json": str(
            prepared.static_map_candidate_admission_path
        ),
        "static_map_candidate_admission_sha256": (
            prepared.static_map_candidate_admission_sha256
        ),
    }
    prepared.epoch_path.write_text(json.dumps(epoch, indent=2, sort_keys=True) + "\n")
    write_survey_progress(prepared.progress_path, prepared.progress, prepared.plan)
    write_stand_survey_registry(prepared.registry_path, prepared.registry, prepared.plan)

    next_viewpoint_id = (
        None
        if prepared.next_leg is None
        else prepared.next_leg.viewpoint.viewpoint_id
    )
    status = {
        "schema_version": 1,
        "status": "coverage_stop_recorded",
        "motion_published": False,
        "motion_scope": "stopped_observation_epoch",
        **survey_status(prepared.plan, prepared.progress, prepared.registry),
        "recorded_viewpoint_id": prepared.viewpoint.viewpoint_id,
        "epoch_json": str(prepared.epoch_path),
        "confirmed_epoch_candidate_count": len(prepared.raw_stands),
        "static_map_candidate_admitted_count": len(admission.admitted_stands),
        "static_map_candidate_rejected_count": len(admission.rejected_stands),
        "static_map_candidate_admission_json": str(
            prepared.static_map_candidate_admission_path
        ),
        "static_map_candidate_admission_sha256": (
            prepared.static_map_candidate_admission_sha256
        ),
        "next_viewpoint_id": next_viewpoint_id,
        "next_route_csv": None if next_route_path is None else str(next_route_path),
        "next_diagnostics_json": (
            None if next_diagnostics_path is None else str(next_diagnostics_path)
        ),
    }
    if write_summary:
        prepared.summary_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )
    return status


def _write_next_leg_artifacts(
    *,
    prepared: PreparedCoverageStop,
    next_leg,
    route_path: Path,
    diagnostics_path: Path,
    metadata_overrides: dict[str, object] | None = None,
) -> None:
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
            "survey_id": prepared.plan.survey_id,
            "plan_sha256": coverage_survey_plan_sha256(prepared.plan),
            "map_bundle_sha256": prepared.plan.map_bundle_sha256,
            "target_viewpoint_id": next_leg.viewpoint.viewpoint_id,
            "target_pose": {
                "x_m": next_leg.viewpoint.pose.x_m,
                "y_m": next_leg.viewpoint.pose.y_m,
                "yaw_rad": next_leg.viewpoint.pose.yaw_rad,
            },
            "candidate_keepout_count": sum(
                1
                for candidate in prepared.registry.candidates
                if candidate.status != "rejected"
            ),
            "unreachable_viewpoint_ids_before_target": list(
                next_leg.unreachable_viewpoint_ids
            ),
            "inflation_radius_m": prepared.plan.config.inflation_radius_m,
            "exact_start_connector": next_leg.exact_start_connector.to_metadata(),
            "line_of_sight_route_optimization": (
                next_leg.route_smoothing.to_metadata()
            ),
            "arena_boundary_overlay": True,
            "arena_bounds": prepared.plan.arena_bounds.to_metadata(),
            **(metadata_overrides or {}),
        },
    )


def commit_stand_coverage_stop(
    *,
    survey_root: Path,
    map_yaml: Path,
    viewpoint_id: str,
    observer_summary_json: Path,
    semantic_map_id: str = "",
    observations_jsonl: Path | None = None,
    arrival_tolerance_m: float = 0.18,
    scan_to_base_position_offset_m: float = 0.05,
) -> dict[str, object]:
    """Persist one stopped epoch without writing non-authorizing next-route artifacts."""

    prepared = _prepare_stand_coverage_stop(
        survey_root=survey_root,
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        viewpoint_id=viewpoint_id,
        observer_summary_json=observer_summary_json,
        observations_jsonl=observations_jsonl,
        arrival_tolerance_m=arrival_tolerance_m,
        scan_to_base_position_offset_m=scan_to_base_position_offset_m,
    )
    return _write_committed_stop_state(
        prepared,
        next_leg_start_pose=prepared.scan_pose,
        next_leg_start_pose_source="observer_frozen_scan_pose",
        next_leg_localization_evidence_json=None,
        next_route_path=None,
        next_diagnostics_path=None,
    )


def plan_next_stand_coverage_leg(
    *,
    survey_root: Path,
    map_yaml: Path,
    expected_next_viewpoint_id: str,
    current_pose: Pose2D,
    localization_evidence_json: Path,
    localization_evidence_sha256: str,
    checkpoint_manifest_json: Path,
    checkpoint_manifest_sha256: str,
    semantic_map_id: str = "",
) -> dict[str, object]:
    """Plan and persist the next route only from committed state and fresh localization."""

    survey_root = Path(survey_root)
    map_yaml = Path(map_yaml)
    expected_next_viewpoint_id = str(expected_next_viewpoint_id).strip()
    if not expected_next_viewpoint_id:
        raise ValueError("expected next viewpoint ID is required")
    _validate_pose(current_pose, "current pose")
    localization_evidence_json = _validate_localization_evidence_path(
        localization_evidence_json,
        label="localization evidence",
    )
    localization_evidence_sha256 = _validate_file_sha256(
        localization_evidence_json,
        localization_evidence_sha256,
        label="localization evidence",
    )
    checkpoint_manifest_sha256 = _validate_sha256(
        checkpoint_manifest_sha256,
        label="checkpoint manifest SHA-256",
    )
    checkpoint_manifest = admit_autonomous_session_manifest(
        Path(checkpoint_manifest_json)
    )
    admitted_checkpoint_sha256 = autonomous_session_manifest_sha256(
        checkpoint_manifest
    )
    if admitted_checkpoint_sha256 != checkpoint_manifest_sha256:
        raise ValueError(
            "checkpoint manifest SHA-256 mismatch: "
            f"{admitted_checkpoint_sha256} != {checkpoint_manifest_sha256}"
        )
    checkpoint_manifest_json = Path(checkpoint_manifest_json).resolve(
        strict=True
    )
    summary_path = survey_root / "survey_summary.json"
    _validate_localization_evidence_path(
        summary_path,
        label="committed survey summary",
    )
    try:
        committed_summary_bytes = summary_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"invalid committed survey summary: {exc}") from exc
    committed_summary_sha256 = hashlib.sha256(committed_summary_bytes).hexdigest()
    try:
        summary = json.loads(committed_summary_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid committed survey summary: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError("committed survey summary must contain a JSON object")
    if summary.get("status") != "coverage_stop_recorded":
        raise ValueError("committed survey summary is not a coverage stop")
    if summary.get("next_route_csv") is not None or summary.get("next_diagnostics_json") is not None:
        raise ValueError("committed survey summary already has next-leg artifacts")
    committed_next_viewpoint_id = summary.get("next_viewpoint_id")
    if committed_next_viewpoint_id != expected_next_viewpoint_id:
        raise ValueError(
            "expected next viewpoint differs from committed survey summary: "
            f"{expected_next_viewpoint_id!r} != {committed_next_viewpoint_id!r}"
        )

    plan_path = survey_root / "coverage_plan.json"
    progress_path = survey_root / "coverage_progress.json"
    registry_path = survey_root / "stand_registry.json"
    for state_path, label in (
        (plan_path, "committed coverage plan"),
        (progress_path, "committed coverage progress"),
        (registry_path, "committed stand registry"),
    ):
        _validate_localization_evidence_path(state_path, label=label)
    plan = load_coverage_survey_plan(plan_path)
    progress = load_survey_progress(progress_path, plan)
    registry = load_stand_survey_registry(registry_path, plan)
    current_state_hashes = {
        "coverage_plan": _file_sha256(plan_path),
        "coverage_progress": _file_sha256(progress_path),
        "survey_summary": committed_summary_sha256,
        "stand_registry": _file_sha256(registry_path),
    }
    checkpoint_state_hashes = {
        "coverage_plan": checkpoint_manifest.coverage_plan.sha256,
        "coverage_progress": checkpoint_manifest.coverage_progress.sha256,
        "survey_summary": checkpoint_manifest.survey_summary.sha256,
        "stand_registry": checkpoint_manifest.stand_registry.sha256,
    }
    if current_state_hashes != checkpoint_state_hashes:
        mismatched = sorted(
            name
            for name in current_state_hashes
            if current_state_hashes[name] != checkpoint_state_hashes[name]
        )
        raise ValueError(
            "checkpoint does not bind the committed survey state: "
            + ", ".join(mismatched)
        )
    if checkpoint_manifest.next_viewpoint_id != expected_next_viewpoint_id:
        raise ValueError(
            "checkpoint next viewpoint differs from the committed target"
        )
    if checkpoint_manifest.completed_coverage_legs != len(
        progress.visited_viewpoint_ids
    ):
        raise ValueError(
            "checkpoint completed-leg count differs from committed progress"
        )
    if checkpoint_manifest.map_bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("checkpoint map bundle differs from coverage plan")
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id or map_yaml.stem,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("runtime map bundle differs from coverage plan")
    next_leg = plan_next_survey_leg(
        grid,
        plan=plan,
        progress=progress,
        registry=registry,
        current_pose=current_pose,
    )
    if next_leg is None:
        raise ValueError("committed survey has no next viewpoint to plan")
    if next_leg.viewpoint.viewpoint_id != expected_next_viewpoint_id:
        raise ValueError(
            "fresh localization changed the next coverage target: "
            f"{next_leg.viewpoint.viewpoint_id!r} != {expected_next_viewpoint_id!r}"
        )
    route_path, diagnostics_path = _next_leg_artifact_paths(survey_root, progress)
    _validate_next_leg_artifacts_absent(route_path, diagnostics_path)
    prepared = PreparedCoverageStop(
        survey_root=survey_root,
        plan_path=plan_path,
        progress_path=progress_path,
        registry_path=registry_path,
        summary_path=summary_path,
        epoch_path=Path(str(summary.get("epoch_json", ""))),
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id or map_yaml.stem,
        grid=grid,
        map_bundle=map_bundle,
        plan=plan,
        progress=progress,
        registry=registry,
        viewpoint=plan.viewpoint_for(str(summary.get("recorded_viewpoint_id", ""))),
        scan_pose=current_pose,
        viewpoint_error_m=0.0,
        observer_summary_json=Path(""),
        observations_path=Path(""),
        observer_summary={},
        observations=(),
        stands=(),
        next_leg=next_leg,
    )
    _write_next_leg_artifacts(
        prepared=prepared,
        next_leg=next_leg,
        route_path=route_path,
        diagnostics_path=diagnostics_path,
        metadata_overrides={
            "checkpoint_manifest_json": str(checkpoint_manifest_json),
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "next_leg_localization_evidence_json": str(localization_evidence_json),
            "next_leg_localization_evidence_sha256": (
                localization_evidence_sha256
            ),
            "committed_survey_summary_json": str(summary_path),
            "committed_survey_summary_sha256": committed_summary_sha256,
        },
    )
    return {
        "schema_version": 1,
        "status": "next_coverage_leg_prepared",
        "motion_published": False,
        "motion_authorized": False,
        "motion_scope": "non_authorizing_next_leg_preparation",
        "committed_survey_summary_json": str(summary_path),
        "committed_survey_summary_sha256": committed_summary_sha256,
        "checkpoint_manifest_json": str(checkpoint_manifest_json),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "recorded_viewpoint_id": summary.get("recorded_viewpoint_id"),
        "next_viewpoint_id": next_leg.viewpoint.viewpoint_id,
        "next_route_csv": str(route_path),
        "next_diagnostics_json": str(diagnostics_path),
        "next_leg_start_pose": {
            "x_m": current_pose.x_m,
            "y_m": current_pose.y_m,
            "yaw_rad": current_pose.yaw_rad,
        },
        "next_leg_start_pose_source": "fresh_stationary_localization_admission",
        "next_leg_localization_evidence_json": str(localization_evidence_json),
        "next_leg_localization_evidence_sha256": localization_evidence_sha256,
    }


def record_stand_coverage_stop(
    *,
    survey_root: Path,
    map_yaml: Path,
    viewpoint_id: str,
    observer_summary_json: Path,
    semantic_map_id: str = "",
    observations_jsonl: Path | None = None,
    arrival_tolerance_m: float = 0.18,
    scan_to_base_position_offset_m: float = 0.05,
    next_leg_start_pose: Pose2D | None = None,
    next_leg_localization_evidence_json: Path | None = None,
) -> dict[str, object]:
    """Fuse one stopped observation epoch and return its persisted status.

    Unlike :func:`main`, this importable API never converts failures to
    ``SystemExit``.  Callers that own a wider mission boundary can therefore
    record the original ``ValueError``/``OSError`` as mission-failure evidence.
    """

    prepared = _prepare_stand_coverage_stop(
        survey_root=survey_root,
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        viewpoint_id=viewpoint_id,
        observer_summary_json=observer_summary_json,
        observations_jsonl=observations_jsonl,
        arrival_tolerance_m=arrival_tolerance_m,
        scan_to_base_position_offset_m=scan_to_base_position_offset_m,
        next_leg_start_pose=next_leg_start_pose,
        next_leg_localization_evidence_json=next_leg_localization_evidence_json,
    )
    route_path = None
    diagnostics_path = None
    if prepared.next_leg is not None:
        route_path, diagnostics_path = _next_leg_artifact_paths(
            prepared.survey_root,
            prepared.progress,
        )
        _validate_next_leg_artifacts_absent(route_path, diagnostics_path)
    next_pose = next_leg_start_pose or prepared.scan_pose
    source = (
        "fresh_stationary_localization_admission"
        if next_leg_start_pose is not None
        else "observer_frozen_scan_pose"
    )
    status = _write_committed_stop_state(
        prepared,
        next_leg_start_pose=next_pose,
        next_leg_start_pose_source=source,
        next_leg_localization_evidence_json=next_leg_localization_evidence_json,
        next_route_path=route_path,
        next_diagnostics_path=diagnostics_path,
        write_summary=False,
    )
    if prepared.next_leg is not None:
        _write_next_leg_artifacts(
            prepared=prepared,
            next_leg=prepared.next_leg,
            route_path=route_path,
            diagnostics_path=diagnostics_path,
        )
    prepared.summary_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    return status


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        status = record_stand_coverage_stop(
            survey_root=args.survey_root,
            map_yaml=args.map,
            semantic_map_id=args.semantic_map_id,
            viewpoint_id=args.viewpoint_id,
            observer_summary_json=args.observer_summary_json,
            observations_jsonl=args.observations_jsonl,
            arrival_tolerance_m=args.arrival_tolerance_m,
            scan_to_base_position_offset_m=(
                args.scan_to_base_position_offset_m
            ),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
