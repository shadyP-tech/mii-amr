"""Fuse one stopped LiDAR observation epoch into a coverage survey.

This command consumes the receipt produced by ``stand_explorer_node.py
--summary-json``.  It marks one planned viewpoint visited, updates stable
candidate IDs and keepouts, and writes a newly planned next leg.  It does not
publish velocity or execute that leg.
"""

from __future__ import annotations

import argparse
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
from scripts.aufgabe04.perception.stand_confirmation import (
    StandConfirmationAccumulator,
    StandConfirmationConfig,
)
from scripts.aufgabe04.perception.stand_observation import (
    load_observation_jsonl,
    validated_observation_stream_clock,
)


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
) -> dict[str, object]:
    """Fuse one stopped observation epoch and return its persisted status.

    Unlike :func:`main`, this importable API never converts failures to
    ``SystemExit``.  Callers that own a wider mission boundary can therefore
    record the original ``ValueError``/``OSError`` as mission-failure evidence.
    """

    survey_root = Path(survey_root)
    map_yaml = Path(map_yaml)
    observer_summary_json = Path(observer_summary_json)
    if observations_jsonl is not None:
        observations_jsonl = Path(observations_jsonl)

    if not math.isfinite(arrival_tolerance_m) or arrival_tolerance_m <= 0.0:
        raise ValueError("arrival tolerance must be finite and positive")
    if (
        not math.isfinite(scan_to_base_position_offset_m)
        or scan_to_base_position_offset_m < 0.0
    ):
        raise ValueError(
            "scan-to-base position offset must be finite and non-negative"
        )
    plan_path = survey_root / "coverage_plan.json"
    progress_path = survey_root / "coverage_progress.json"
    registry_path = survey_root / "stand_registry.json"
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
    stands = _epoch_stands(observations, plan)
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
        current_pose=scan_pose,
    )
    next_viewpoint_id = None
    next_route_path = None
    next_diagnostics_path = None
    if next_leg is not None:
        leg_index = len(progress.visited_viewpoint_ids)
        legs_dir = survey_root / "legs"
        next_route_path = legs_dir / f"leg_{leg_index:03d}_route.csv"
        next_diagnostics_path = legs_dir / f"leg_{leg_index:03d}_diagnostics.json"
        if next_route_path.exists() or next_diagnostics_path.exists():
            raise ValueError("refusing to overwrite next-leg artifacts")

    epoch_path = survey_root / "epochs" / f"{viewpoint.viewpoint_id}.json"
    if epoch_path.exists():
        raise ValueError(f"refusing to overwrite survey epoch: {epoch_path}")
    epoch_path.parent.mkdir(parents=True, exist_ok=True)
    epoch = {
        "schema_version": 1,
        "survey_id": plan.survey_id,
        "viewpoint_id": viewpoint.viewpoint_id,
        "planned_pose": {
            "x_m": viewpoint.pose.x_m,
            "y_m": viewpoint.pose.y_m,
            "yaw_rad": viewpoint.pose.yaw_rad,
        },
        "observed_scan_pose": {
            "x_m": scan_pose.x_m,
            "y_m": scan_pose.y_m,
            "yaw_rad": scan_pose.yaw_rad,
        },
        "viewpoint_error_m": viewpoint_error_m,
        "observer_summary_json": str(observer_summary_json),
        "observations_jsonl": str(observations_path),
        "processed_scan_count": observer_summary["processed_scan_count"],
        "accepted_observation_count": len(observations),
        "confirmed_epoch_candidate_count": len(stands),
    }
    epoch_path.write_text(json.dumps(epoch, indent=2, sort_keys=True) + "\n")
    write_survey_progress(progress_path, progress, plan)
    write_stand_survey_registry(registry_path, registry, plan)

    if next_leg is not None:
        write_route_csv(
            next_route_path,
            (next_leg.route_result,),
            final_yaw_by_leg={0: next_leg.viewpoint.pose.yaw_rad},
        )
        write_diagnostics_json(
            next_diagnostics_path,
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
            },
        )
        next_viewpoint_id = next_leg.viewpoint.viewpoint_id

    status = {
        "schema_version": 1,
        "status": "coverage_stop_recorded",
        "motion_published": False,
        **survey_status(plan, progress, registry),
        "recorded_viewpoint_id": viewpoint.viewpoint_id,
        "epoch_json": str(epoch_path),
        "next_viewpoint_id": next_viewpoint_id,
        "next_route_csv": None if next_route_path is None else str(next_route_path),
        "next_diagnostics_json": (
            None if next_diagnostics_path is None else str(next_diagnostics_path)
        ),
    }
    (survey_root / "survey_summary.json").write_text(
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
