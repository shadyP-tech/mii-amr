#!/usr/bin/env python3
"""Run one sealed real-robot route leg through the existing certified runner.

Dry-run is the default.  ``--execute`` still delegates to the inner runner's
ROS preflight and typed ``RUN`` prompt and automatically wraps the attempt in a
real-run evidence bundle.  This adapter is explicitly unloaded; it does not
admit puck custody or loaded-footprint assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.artifacts.manifest_store import load_survey_manifest
from scripts.aufgabe04.navigation.execution.mission_execution_gate import (
    ARTIFACT_DESCRIPTOR_HASH_FIELD,
    validate_planner_config_descriptor,
)
from scripts.aufgabe04.real_robot.configuration.profile import load_real_robot_profile


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_profile_artifact_bindings(profile, survey, planner_config_path: Path) -> None:
    if survey.planning_frame != profile.map_frame:
        raise ValueError("real survey frame differs from robot profile")
    if survey.calibration_profile.sha256 != profile.calibration_profile_sha256:
        raise ValueError("real survey calibration differs from robot profile")
    planner_config = load_content_hashed_json(
        planner_config_path,
        hash_field=ARTIFACT_DESCRIPTOR_HASH_FIELD,
    )
    validate_planner_config_descriptor(planner_config)
    for field, expected in (
        ("robot_radius_m", profile.robot_radius_m),
        (
            "scan_origin_to_base_offset_m",
            profile.scan_origin_to_base_offset_m,
        ),
    ):
        observed = float(planner_config[field])
        if abs(observed - expected) > 1.0e-9:
            raise ValueError(
                f"planner {field} differs from sealed real robot profile"
            )


def build_runner_command(args, profile) -> list[str]:
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/entrypoints/run_single_station_segment.py",
        "--route-csv",
        str(args.route_csv),
        "--diagnostics-json",
        str(args.diagnostics_json),
        "--route-certificate-json",
        str(args.route_certificate_json),
        "--route-bundle-json",
        str(args.route_bundle_json),
        "--planner-config-json",
        str(args.planner_config_json),
        "--mission-plan-manifest",
        str(args.mission_plan_manifest),
        "--survey-manifest",
        str(args.survey_manifest),
        "--runtime-map-bundle-json",
        str(args.runtime_map_bundle_json),
        "--runtime-environment",
        str(args.physical_site),
        "--candidate-snapshot",
        str(args.candidate_snapshot),
        "--station-identity-registry",
        str(args.station_identity_registry),
        "--arrival-pose-catalog",
        str(args.arrival_pose_catalog),
        "--task-snapshot",
        str(args.task_snapshot),
        "--robot-id",
        profile.robot_id,
        "--leg-index",
        str(args.leg_index),
        "--run-id",
        args.run_id,
        "--namespace",
        profile.namespace,
        "--scan-topic",
        profile.scan_topic,
        "--odom-topic",
        profile.odom_topic,
        "--cmd-vel-topic",
        profile.cmd_vel_topic,
        "--amcl-topic",
        profile.amcl_topic,
        "--map-frame",
        profile.map_frame,
        "--odom-frame",
        profile.odom_frame,
        "--base-frame",
        profile.base_frame,
        "--localization-source",
        profile.localization_source,
        "--max-linear-mps",
        str(profile.max_linear_speed_mps),
        "--max-angular-radps",
        str(profile.max_angular_speed_radps),
        "--operator-note",
        f"UNLOADED; {args.operator_note}".strip(),
        "--prompt-for-initialpose",
    ]
    for publisher in args.allowed_cmd_vel_publisher:
        command.extend(["--allowed-cmd-vel-publisher", publisher])
    if not args.execute:
        command.append("--dry-run")
    return command


def build_execution_command(args, profile) -> list[str]:
    runner = build_runner_command(args, profile)
    if not args.execute:
        return runner
    return [
        "scripts/common/run_with_bundle.sh",
        "--namespace",
        profile.namespace,
        "--cmd-vel-topic",
        profile.cmd_vel_topic,
        "--scan-topic",
        profile.scan_topic,
        "--odom-topic",
        profile.odom_topic,
        "--amcl-topic",
        profile.amcl_topic,
        "--map-frame",
        profile.map_frame,
        "--odom-frame",
        profile.odom_frame,
        "--base-frame",
        profile.base_frame,
        "--output-root",
        str(args.output_root),
        args.run_id,
        "--",
        *runner,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-profile", required=True, type=Path)
    parser.add_argument("--physical-site", required=True, type=Path)
    parser.add_argument("--route-csv", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--route-certificate-json", required=True, type=Path)
    parser.add_argument("--route-bundle-json", required=True, type=Path)
    parser.add_argument("--planner-config-json", required=True, type=Path)
    parser.add_argument("--mission-plan-manifest", required=True, type=Path)
    parser.add_argument("--survey-manifest", required=True, type=Path)
    parser.add_argument("--runtime-map-bundle-json", required=True, type=Path)
    parser.add_argument("--candidate-snapshot", required=True, type=Path)
    parser.add_argument("--station-identity-registry", required=True, type=Path)
    parser.add_argument("--arrival-pose-catalog", required=True, type=Path)
    parser.add_argument("--task-snapshot", required=True, type=Path)
    parser.add_argument("--leg-index", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=[],
    )
    parser.add_argument("--operator-note", default="")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/real_runs"),
    )
    parser.add_argument(
        "--confirm-unloaded",
        action="store_true",
        help="Required operator assertion that no puck/cargo is fitted.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Permit the inner runner to reach its physical RUN prompt. Without "
            "this flag the command is always --dry-run."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.leg_index < 0:
        parser.error("--leg-index must be non-negative")
    if not args.confirm_unloaded:
        parser.error("--confirm-unloaded is required")
    try:
        profile = load_real_robot_profile(args.robot_profile)
        survey = load_survey_manifest(args.survey_manifest)
        if survey.environment != "real":
            raise ValueError("unloaded real segment requires a real survey manifest")
        validate_profile_artifact_bindings(
            profile,
            survey,
            args.planner_config_json,
        )
        site_sha256 = _file_sha256(args.physical_site)
        if (
            args.physical_site.stem != profile.physical_site_id
            or site_sha256 != profile.physical_site_sha256
            or survey.environment_descriptor.artifact_id != profile.physical_site_id
            or survey.environment_descriptor.sha256 != site_sha256
        ):
            raise ValueError(
                "runtime physical site differs from robot profile or real survey"
            )
        command = build_execution_command(args, profile)
        if args.execute:
            print(
                "Physical safety requirements: clear arena; operator beside the "
                "robot; Ctrl+C and physical stop ready; separate exact-topic "
                "zero Twist terminal ready; no cargo; no competing controller."
            )
        return subprocess.run(command, check=False).returncode
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
