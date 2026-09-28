"""Standalone, unloaded server tour over authenticated exploration artifacts.

The camera process is never started. ROS imports and physical effects are lazy,
so the default artifact preview works without a running robot or server.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploration-session", type=Path, required=True,
                        help="Completed exploration directory containing mission_summary.json.")
    parser.add_argument("--robot-profile", type=Path, required=True)
    parser.add_argument("--physical-site", type=Path,
                        help="Override docs/setups/{profile.physical_site_id}.json.")
    parser.add_argument("--server-robot-id", required=True,
                        help="Stable team/robot identity on the FastAPI server.")
    parser.add_argument("--server-base-url", default="http://10.42.0.1:8000")
    parser.add_argument("--http-timeout-sec", type=float, default=5.0)
    parser.add_argument("--stations", type=int, default=3,
                        help="Production visits requested from the random-plan API (3–100).")
    parser.add_argument("--server-plan-only", action="store_true",
                        help="Omit supplemental visits to stands absent from the server plan.")
    parser.add_argument("--tour-id", default=None)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/aufgabe04/real/station_tours"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="Allow a new RUN-confirmed tour, server writes and physical motion.")
    mode.add_argument("--dry-run", action="store_true",
                      help="Validate saved artifacts only; this is the default.")
    parser.add_argument("--confirm-unloaded", action="store_true",
                        help="Assert that the robot has no puck/cargo attached.")
    parser.add_argument("--confirm-odom-continuity", action="store_true",
                        help="Assert no base restart, odometry reset or odom-frame change since exploration.")
    return parser


def build_navigation_effects(profile, output_root: Path):
    """Reuse the sole certified motion edge without running camera exploration."""
    from scripts.aufgabe04.real_robot.autonomous_runner.runtime import (
        _admit_candidate_planning_frame, _run_motion_leg,
    )
    from scripts.aufgabe04.real_robot.coverage_leg.models import MissionLegPermitContext
    from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import StoredPoseNavigationEffects

    runtime = profile.resolved_runtime()

    def capture(path):
        return _admit_candidate_planning_frame(runtime, output_root, evidence_path=path)

    def move(request):
        return _run_motion_leg(
            profile=profile, sealed=request.sealed, run_id=request.run_id,
            session_root=request.session_root, execute=True,
            candidate_snapshot=request.candidate_snapshot_path,
            uncertainty_map_yaml=request.uncertainty_map_yaml,
            uncertainty_sigma_multiplier=request.uncertainty_sigma_multiplier,
            localization_branch_proof_id=request.localization_branch_proof_id,
            sensor_timing_readiness_phase=None,
            mission_leg_permit_context=MissionLegPermitContext(
                mission_authorization_json=request.mission_authorization_json,
                session_id=request.session_id, semantic_map_id=request.semantic_map_id,
                mission_leg_kind=request.mission_leg_kind,
                mission_leg_index=request.mission_leg_index, target_id=request.target_id,
                permit_json_path=request.permit_json_path,
            ),
        )

    return StoredPoseNavigationEffects(capture, move)


def execute_tour(session, args, output_root: Path, tour_id: str):
    """Authorize only this new tour; never reuse the exploration's RUN."""
    from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
        MissionLegKind, MissionLegMotionAuthorization,
        TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
        write_mission_leg_motion_authorization,
    )
    from scripts.aufgabe04.logistics.station_tour import run_station_tour
    from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import execute_stored_pose_navigation
    from scripts.aufgabe04.task_client.station_tour_client import StationTourClient

    effects = build_navigation_effects(session.profile, output_root)
    effects.admit_planning_frame(output_root / "preflight/before_authorization.json")
    print("Unloaded stand tour: saved Start pose first, then the server's randomized targets.")
    print("Travel limits: 0.15 m/s and 0.60 rad/s; slower near corners and final poses.")
    print("Server actions use timed waits only; this runner does not manipulate physical cargo.")
    print("Keep the arena clear, the operator beside the robot and the physical stop ready.")
    if input("Type RUN to authorize this server station tour: ").strip() != "RUN":
        raise RuntimeError("operator did not authorize the station tour")
    runtime = session.profile.resolved_runtime()
    authorization_path = output_root / "motion_authorization/tour.json"
    write_mission_leg_motion_authorization(authorization_path, MissionLegMotionAuthorization(
        session_id=tour_id, robot_id=session.profile.robot_id, namespace=runtime.namespace,
        cmd_vel_topic=runtime.cmd_vel_topic, semantic_map_id=session.config.semantic_map_id,
        localization_branch_proof_id=session.config.localization_branch_proof_id,
        allowed_leg_kinds=(MissionLegKind.STORED_POSE_TOUR,),
        scope_text=TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
        operator_confirmation="RUN",
    ))
    config = replace(session.config, mission_leg_motion_authorization_json=authorization_path)

    def navigate(qr_id, visit_index):
        print(f"Visit {visit_index}: driving to saved {qr_id} pose", flush=True)
        return execute_stored_pose_navigation(
            session.poses_by_qr[qr_id], config, effects,
            tour_session_id=tour_id, visit_index=visit_index,
            output_root=output_root / "visits" / f"{visit_index:03d}",
        )

    client = StationTourClient(base_url=args.server_base_url,
                              robot_id=args.server_robot_id,
                              timeout_sec=args.http_timeout_sec)
    return run_station_tour(client, session.poses_by_qr, navigate, output_root / "server",
                            station_visits=args.stations, cover_all_stands=not args.server_plan_only)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 3 <= args.stations <= 100:
        parser.error("--stations must be between 3 and 100")
    if not 0 < args.http_timeout_sec <= 60:
        parser.error("--http-timeout-sec must be finite and in (0, 60]")
    if not args.server_robot_id.strip():
        parser.error("--server-robot-id must be nonempty")
    if args.execute and not (args.confirm_unloaded and args.confirm_odom_continuity):
        parser.error("--execute requires --confirm-unloaded and --confirm-odom-continuity")
    tour_id = args.tour_id or datetime.now(timezone.utc).strftime("station_tour_%Y%m%dT%H%M%S_%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", tour_id) or ".." in tour_id:
        parser.error("--tour-id must be a simple identifier of 1–80 characters without '..'")
    output_root = args.output_root.resolve() / tour_id
    args.exploration_session = args.exploration_session.resolve()
    args.robot_profile = args.robot_profile.resolve()
    if args.physical_site is not None:
        args.physical_site = args.physical_site.resolve()
    created_output = False
    invocation_cwd = Path.cwd()
    try:
        # Stored exploration evidence may contain repository-relative paths.
        # Its original path strings stay intact for the content-hash checks.
        os.chdir(ROOT)
        from scripts.aufgabe04.real_robot.mission.stored_pose_session import load_stored_pose_session
        from scripts.aufgabe04.logistics.station_tour import validate_available_qr_ids
        from scripts.aufgabe04.task_client.station_tour_client import StationTourClient

        # Validate URL/timeout before authorization or ROS interaction.
        StationTourClient(base_url=args.server_base_url, robot_id=args.server_robot_id,
                          timeout_sec=args.http_timeout_sec)
        site_override = {} if args.physical_site is None else {"physical_site_path": args.physical_site.resolve()}
        session = load_stored_pose_session(args.exploration_session.resolve(), args.robot_profile.resolve(), **site_override)
        qr_count = validate_available_qr_ids(session.poses_by_qr)
        preview = {
            "tour_id": tour_id, "exploration_session": str(args.exploration_session.resolve()),
            "physical_site_override": None if args.physical_site is None else str(args.physical_site.resolve()),
            "server_base_url": args.server_base_url, "server_robot_id": args.server_robot_id,
            "randomize_request": {"qr_count": qr_count, "stations": args.stations},
            "first_qr_id": "Start", "poses": {
                qr: {"candidate_uid": saved.candidate_uid, "pose_kind": saved.evidence["pose_kind"],
                     "stored_pose": asdict(saved.pose), "source_frame": saved.source_frame.to_evidence()}
                for qr, saved in sorted(session.poses_by_qr.items())
            },
            "mode": "execute" if args.execute else "artifact_preview",
            "odom_continuity_asserted": args.confirm_odom_continuity,
            "unloaded_asserted": args.confirm_unloaded,
            "physical_cargo_actions": False,
            "cover_all_stands": not args.server_plan_only,
        }
        output_root.mkdir(parents=True, exist_ok=False)
        created_output = True
        _write_json(output_root / "inputs.json", preview)
        if not args.execute:
            print(f"Validated {len(session.poses_by_qr)} saved poses, including Start.")
            print(f"Preview: {output_root / 'inputs.json'}")
            print("No motion or server requests. Add --execute and both confirmation flags to run.")
            return 0
        result = execute_tour(session, args, output_root, tour_id)
        _write_json(output_root / "result.json", result)
        print(f"Station tour completed. Evidence: {output_root}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, KeyboardInterrupt) as exc:
        if created_output:
            failure = output_root / "failure.json"
            if not failure.exists():
                _write_json(failure, {"tour_id": tour_id, "status": "stopped", "error": str(exc) or type(exc).__name__})
        print(f"Station tour stopped: {exc or type(exc).__name__}")
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        os.chdir(invocation_cwd)
