#!/usr/bin/env python3
"""Prepare immutable per-candidate commands for a passive real survey.

The generated observer commands contain no motion publisher.  The generated
planner commands only validate already committed observations and update the
arrival catalog; real dynamic route generation is rejected by the planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.artifacts.content_store import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.planning.map_io import freeze_map_bundle
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_measured_physical_stand_model,
    resolve_head_center_height_m,
)
from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256,
    load_camera_calibration,
    load_real_robot_profile,
    real_robot_profile_sha256,
)
from scripts.aufgabe04.real_robot.observer.contract import (
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    load_station_identity_registry,
    station_identity_registry_sha256,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observer_command(args, candidate, identity, output):
    geometry = candidate.geometry
    return [
        sys.executable,
        "scripts/aufgabe04/real_robot/entrypoints/passive_viewpoint_node.py",
        "--robot-profile",
        str(args.robot_profile),
        "--camera-calibration",
        str(args.camera_calibration),
        "--stand-model-profile",
        str(args.stand_model_profile),
        "--stream-id",
        f"{args.session_id}_{candidate.candidate_uid}",
        "--stand-id",
        identity.qr_id,
        "--expected-qr-id",
        identity.qr_id,
        "--stand-x",
        str(geometry.x_m),
        "--stand-y",
        str(geometry.y_m),
        "--stand-radius-m",
        str(geometry.radius_m),
        "--stand-uncertainty-m",
        str(geometry.uncertainty_m),
        "--target-distance-m",
        str(args.target_distance_m),
        "--stand-head-center-height-m",
        str(args.stand_head_center_height_m),
        "--consensus-frames",
        str(args.axis_sample_count),
        "--status-json",
        str(output / "observer_status.json"),
        "--status-events-jsonl",
        str(output / "observer_events.jsonl"),
        "--recommended-pose-json",
        str(output / "recommendation.json"),
        "--debug-dir",
        str(output / "perception_debug"),
        "--once",
    ]


def _planner_command(
    args,
    profile,
    snapshot,
    registry,
    candidate,
    output,
    *,
    map_bundle_sha256,
    survey_config_sha256,
    survey_input_binding_sha256,
):
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/entrypoints/plan_synchronized_viewpoint.py",
        "--environment",
        "real",
        "--map",
        str(args.map),
        "--start-x",
        "0",
        "--start-y",
        "0",
        "--start-yaw",
        "0",
        "--start-from-recommendation",
        "--recommended-pose-json",
        str(output / "recommendation.json"),
        "--route-csv",
        str(output / "validation_route.csv"),
        "--diagnostics-json",
        str(output / "validation_diagnostics.json"),
        "--stream-id",
        f"{args.session_id}_{candidate.candidate_uid}",
        "--workflow-mode",
        "survey-only",
        "--arrival-pose-catalog",
        str(args.catalog),
        "--catalog-id",
        args.catalog_id,
        "--candidate-uid",
        candidate.candidate_uid,
        "--world-id",
        profile.physical_site_id,
        "--world-sha256",
        profile.physical_site_sha256,
        "--session-id",
        args.session_id,
        "--axis-sample-count",
        str(args.axis_sample_count),
        "--map-frame",
        profile.map_frame,
        "--semantic-map-id",
        args.semantic_map_id,
        "--expected-map-bundle-sha256",
        map_bundle_sha256,
        "--candidate-snapshot-sha256",
        candidate_snapshot_sha256(snapshot),
        "--station-identity-registry-sha256",
        station_identity_registry_sha256(registry),
        "--survey-config-sha256",
        survey_config_sha256,
        "--calibration-profile-sha256",
        profile.calibration_profile_sha256,
        "--survey-input-binding-sha256",
        survey_input_binding_sha256,
        "--robot-radius-m",
        str(profile.robot_radius_m),
        "--scan-origin-to-base-offset-m",
        str(profile.scan_origin_to_base_offset_m),
        "--lidar-stop-distance-m",
        str(args.lidar_stop_distance_m),
        "--lidar-clearance-margin-m",
        str(args.lidar_clearance_margin_m),
        "--standoff-distance-m",
        str(args.target_distance_m),
    ]
    for candidate_uid in snapshot.candidate_uids:
        command.extend(["--expected-candidate-uid", candidate_uid])
    for other in snapshot.candidates:
        if other.candidate_uid == candidate.candidate_uid:
            continue
        command.extend(
            [
                "--known-stand-keepout",
                str(other.geometry.x_m),
                str(other.geometry.y_m),
                str(other.geometry.keepout_radius_m),
            ]
        )
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-profile", required=True, type=Path)
    parser.add_argument("--camera-calibration", required=True, type=Path)
    parser.add_argument(
        "--stand-model-profile",
        required=True,
        type=Path,
        help="Content-hashed measured physical stand geometry.",
    )
    parser.add_argument("--physical-site", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--semantic-map-id", required=True)
    parser.add_argument("--candidate-snapshot", required=True, type=Path)
    parser.add_argument("--station-identity-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--survey-manifest", required=True, type=Path)
    parser.add_argument("--axis-sample-count", type=int, default=7)
    parser.add_argument("--target-distance-m", type=float, default=0.33)
    parser.add_argument(
        "--stand-head-center-height-m",
        type=float,
        default=None,
        help=(
            "Optional consistency assertion; the measured stand profile is "
            "the authoritative source."
        ),
    )
    parser.add_argument("--lidar-stop-distance-m", type=float, default=0.18)
    parser.add_argument("--lidar-clearance-margin-m", type=float, default=0.02)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.axis_sample_count < 7:
        parser.error("--axis-sample-count must be at least seven")
    for name in (
        "target_distance_m",
        "lidar_stop_distance_m",
        "lidar_clearance_margin_m",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        profile = load_real_robot_profile(args.robot_profile)
        calibration = load_camera_calibration(args.camera_calibration)
        stand_model = load_measured_physical_stand_model(
            args.stand_model_profile
        )
        args.stand_head_center_height_m = resolve_head_center_height_m(
            stand_model,
            args.stand_head_center_height_m,
        )
        calibration_sha256 = camera_calibration_sha256(calibration)
        if calibration_sha256 != profile.calibration_profile_sha256:
            raise ValueError("robot profile and camera calibration differ")
        if _file_sha256(args.physical_site) != profile.physical_site_sha256:
            raise ValueError("physical site descriptor differs from robot profile")
        if args.physical_site.stem != profile.physical_site_id:
            raise ValueError(
                "physical site filename stem must equal profile physical_site_id"
            )
        map_bundle = freeze_map_bundle(
            args.map,
            semantic_map_id=args.semantic_map_id,
            planning_frame=profile.map_frame,
        )
        snapshot = load_candidate_snapshot(
            args.candidate_snapshot,
            required_map_bundle_sha256=map_bundle.bundle_sha256,
        )
        if snapshot.planning_frame != profile.map_frame:
            raise ValueError("candidate snapshot frame differs from real profile")
        registry = load_station_identity_registry(
            args.station_identity_registry,
            candidate_snapshot=snapshot,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        survey_config = {
            "schema_version": 1,
            "config_kind": "real_passive_arrival_survey",
            "session_id": args.session_id,
            "robot_profile_sha256": real_robot_profile_sha256(profile),
            "calibration_profile_sha256": calibration_sha256,
            "planning_frame": profile.map_frame,
            "axis_sample_count": args.axis_sample_count,
            "target_distance_m": args.target_distance_m,
            "stand_model_profile_id": stand_model.profile_id,
            "stand_model_profile_sha256": stand_model.sha256,
            "stand_model_environment": stand_model.environment,
            "stand_model_measurement_status": (
                stand_model.measurement_status
            ),
            "stand_head_width_m": stand_model.head_width_m,
            "stand_head_height_m": stand_model.head_height_m,
            "stand_head_center_height_m": args.stand_head_center_height_m,
            "lidar_stop_distance_m": args.lidar_stop_distance_m,
            "lidar_clearance_margin_m": args.lidar_clearance_margin_m,
            "observer_version": PASSIVE_VIEWPOINT_OBSERVER_VERSION,
            "motion_capability": "none",
        }
        survey_config_sha256 = payload_sha256(survey_config)
        survey_config_path = args.output_dir / (
            f"survey_config_{survey_config_sha256}.json"
        )
        write_content_hashed_json(
            survey_config_path,
            survey_config,
            hash_field="survey_config_sha256",
        )
        survey_binding = {
            "schema_version": 1,
            "binding_kind": "real_passive_survey_inputs",
            "session_id": args.session_id,
            "map_bundle_sha256": map_bundle.bundle_sha256,
            "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
            "station_identity_registry_sha256": (
                station_identity_registry_sha256(registry)
            ),
            "real_robot_profile_sha256": real_robot_profile_sha256(profile),
            "physical_site_sha256": profile.physical_site_sha256,
            "survey_config_sha256": survey_config_sha256,
            "calibration_profile_sha256": calibration_sha256,
            "stand_model_profile_sha256": stand_model.sha256,
        }
        binding_sha256 = payload_sha256(survey_binding)
        binding_path = args.output_dir / f"survey_inputs_{binding_sha256}.json"
        write_content_hashed_json(
            binding_path,
            survey_binding,
            hash_field="survey_input_binding_sha256",
        )
        candidates = []
        for candidate in snapshot.candidates:
            identity = registry.for_candidate(candidate.candidate_uid)
            if identity is None:
                raise ValueError(
                    f"candidate {candidate.candidate_uid!r} has no identity mapping"
                )
            output = args.output_dir / candidate.candidate_uid
            candidates.append(
                {
                    "candidate_uid": candidate.candidate_uid,
                    "qr_id": identity.qr_id,
                    "server_station_id": identity.server_station_id,
                    "operator_positioning": (
                        "Manually position the stopped robot at a clear view of "
                        "this stand; no generated command moves the robot."
                    ),
                    "observer_command": _observer_command(
                        args,
                        candidate,
                        identity,
                        output,
                    ),
                    "catalog_validation_command": _planner_command(
                        args,
                        profile,
                        snapshot,
                        registry,
                        candidate,
                        output,
                        map_bundle_sha256=map_bundle.bundle_sha256,
                        survey_config_sha256=survey_config_sha256,
                        survey_input_binding_sha256=binding_sha256,
                    ),
                }
            )
        finalize_command = [
            sys.executable,
            "scripts/aufgabe04/real_robot/entrypoints/finalize_passive_survey.py",
            "--robot-profile",
            str(args.robot_profile),
            "--camera-calibration",
            str(args.camera_calibration),
            "--physical-site",
            str(args.physical_site),
            "--map",
            str(args.map),
            "--semantic-map-id",
            args.semantic_map_id,
            "--candidate-snapshot",
            str(args.candidate_snapshot),
            "--station-identity-registry",
            str(args.station_identity_registry),
            "--catalog",
            str(args.catalog),
            "--session-id",
            args.session_id,
            "--survey-config",
            str(survey_config_path),
            "--survey-input-binding",
            str(binding_path),
            "--survey-manifest",
            str(args.survey_manifest),
        ]
        plan = {
            "schema_version": 1,
            "plan_kind": "real_passive_survey",
            "session_id": args.session_id,
            "motion_capability": "none",
            "robot_profile": str(args.robot_profile),
            "robot_profile_sha256": real_robot_profile_sha256(profile),
            "stand_model_profile": str(args.stand_model_profile),
            "stand_model_profile_sha256": stand_model.sha256,
            "survey_config": str(survey_config_path),
            "survey_config_sha256": survey_config_sha256,
            "survey_input_binding": str(binding_path),
            "survey_input_binding_sha256": binding_sha256,
            "catalog": str(args.catalog),
            "candidate_runs": candidates,
            "finalize_command": finalize_command,
        }
        plan_sha256 = payload_sha256(plan)
        plan_path = args.output_dir / f"passive_survey_plan_{plan_sha256}.json"
        write_content_hashed_json(
            plan_path,
            plan,
            hash_field="real_experiment_plan_sha256",
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "motion_capability": "none",
                    "plan": str(plan_path),
                    "candidate_count": len(candidates),
                    "survey_config": str(survey_config_path),
                    "survey_input_binding": str(binding_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
