"""Run one validated Aufgabe 04 station-route segment on a TurtleBot."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.ros_preflight import run_ros_preflight
from scripts.aufgabe04.navigation.ros_runtime_config import (
    RuntimeConfig,
    resolve_runtime_config,
)
from scripts.aufgabe04.navigation.run_events import configure_event_logger, emit_event
from scripts.aufgabe04.navigation.safety_checks import (
    validate_route_diagnostics_json,
    validate_speed_limits,
)
from scripts.aufgabe04.navigation.segment_run_logger import append_segment_run
from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    FollowerConfig,
    run_simple_waypoint_follower,
)
from scripts.aufgabe04.navigation.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg, poses_from_waypoints


DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/station_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/station_route_diagnostics.json")
DEFAULT_RUN_LOG = Path("results/aufgabe04/station_segment_runs.csv")
DEFAULT_EVENT_LOG_DIR = Path("results/aufgabe04/run_events")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--leg-index", type=int, required=True)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--robot-id", default="tb3")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--localization-source", default="amcl", choices=["amcl", "tf"])
    parser.add_argument("--allow-sim-time", action="store_true")
    parser.add_argument("--max-linear-mps", type=float, default=0.05)
    parser.add_argument("--max-angular-radps", type=float, default=0.15)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.08)
    parser.add_argument("--heading-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--min-obstacle-distance-m", type=float, default=0.20)
    parser.add_argument("--thinning-min-spacing-m", type=float, default=0.15)
    parser.add_argument("--max-scan-age-sec", type=float, default=1.0)
    parser.add_argument("--max-odom-age-sec", type=float, default=1.0)
    parser.add_argument("--max-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--max-amcl-age-sec", type=float, default=2.0)
    parser.add_argument("--waypoint-timeout-sec", type=float, default=45.0)
    parser.add_argument("--initial-distance-limit-m", type=float, default=0.35)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-noop", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--semantic-log", type=Path, default=None)
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=[],
        help="Namespace-qualified node identity allowed in preflight, e.g. /robot1/controller_server",
    )
    return parser


def _physical_checklist(args, resolved) -> None:
    print("\nThis command will publish to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - clear the arena and station approach zones")
    print("  - keep an operator beside the robot")
    print("  - keep Ctrl+C ready in this terminal and physical stop available")
    print(f"  - keep a separate terminal ready to publish zero Twist to {resolved.cmd_vel_topic}")
    print("  - verify the resolved namespace, topics, and frames match this robot")
    print("  - verify no active Nav2 goal/controller or other follower is publishing velocity commands")
    print("  - verify scan, odom, TF, and configured localization data are fresh")
    print("  - verify exactly one AMCL or SLAM source owns the route localization transform")
    print("  - verify real-robot runtime nodes are not using simulated time")
    print(f"Run ID: {args.run_id}")
    print(f"Resolved cmd_vel: {resolved.cmd_vel_topic}")


def _confirm_motion(args, resolved) -> bool:
    if args.yes:
        return True
    _physical_checklist(args, resolved)
    response = input("Type RUN to start station-segment following: ").strip()
    return response == "RUN"


def _base_log_row(args, resolved, leg, preflight_ok: bool) -> Dict[str, object]:
    configured = resolved.configured
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "robot_id": args.robot_id,
        "namespace": resolved.namespace,
        "configured_cmd_vel_topic": configured.cmd_vel_topic,
        "resolved_cmd_vel_topic": resolved.cmd_vel_topic,
        "configured_scan_topic": configured.scan_topic,
        "resolved_scan_topic": resolved.scan_topic,
        "configured_odom_topic": configured.odom_topic,
        "resolved_odom_topic": resolved.odom_topic,
        "map_frame": resolved.map_frame,
        "odom_frame": resolved.odom_frame,
        "base_frame": resolved.base_frame,
        "leg_index": leg.leg_index,
        "raw_point_count": len(leg.raw_waypoints),
        "executable_point_count": len(leg.executable_waypoints),
        "route_length_m": f"{leg.route_length_m:.6f}",
        "preflight_ok": preflight_ok,
        "operator_note": args.operator_note,
    }


def _append_result(args, resolved, leg, preflight_ok: bool, result: FollowerResult) -> None:
    row = _base_log_row(args, resolved, leg, preflight_ok)
    row.update(
        {
            "status": result.status,
            "stop_reason": result.stop_reason,
            "duration_sec": f"{result.duration_sec:.3f}",
            "distance_estimate_m": f"{result.distance_estimate_m:.6f}",
            "motion_published": result.motion_published,
            "semantic_log_path": args.semantic_log,
            "preflight_json_path": args.preflight_json or "",
        }
    )
    append_segment_run(args.results_csv, row)


def _append_status_result(
    args,
    resolved,
    leg,
    *,
    preflight_ok: bool,
    status: str,
    stop_reason: str,
) -> None:
    _append_result(
        args,
        resolved,
        leg,
        preflight_ok,
        FollowerResult(status, stop_reason, 0.0, 0.0, False),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_id = args.run_id or f"aufgabe04-segment-{uuid.uuid4().hex[:8]}"
    args.semantic_log = args.semantic_log or DEFAULT_EVENT_LOG_DIR / f"{args.run_id}.jsonl"
    event_logger = configure_event_logger(args.semantic_log)
    require_motion = not args.allow_noop
    runtime_config = RuntimeConfig(
        namespace=args.namespace,
        scan_topic=args.scan_topic,
        odom_topic=args.odom_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        amcl_topic=args.amcl_topic,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        localization_source=args.localization_source,
        use_sim_time=args.allow_sim_time,
    )
    resolved = resolve_runtime_config(runtime_config)
    emit_event(
        event_logger,
        "run_started",
        run_id=args.run_id,
        robot_id=args.robot_id,
        route_csv=str(args.route_csv),
        diagnostics_json=str(args.diagnostics_json),
        leg_index=args.leg_index,
        results_csv=str(args.results_csv),
        semantic_log_path=str(args.semantic_log),
        preflight_json_path=str(args.preflight_json or ""),
    )
    emit_event(
        event_logger,
        "runtime_resolved",
        run_id=args.run_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        resolved_cmd_vel_topic=resolved.cmd_vel_topic,
        resolved_scan_topic=resolved.scan_topic,
        resolved_odom_topic=resolved.odom_topic,
        resolved_amcl_topic=resolved.amcl_topic,
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        localization_source=resolved.localization_source,
        ros_domain_id=resolved.ros_domain_id,
        allow_sim_time=args.allow_sim_time,
    )
    try:
        leg = load_route_leg(
            args.route_csv,
            args.leg_index,
            require_motion=require_motion,
            thinning_min_spacing_m=args.thinning_min_spacing_m,
        )
    except (OSError, ValueError) as exc:
        emit_event(
            event_logger,
            "route_validation_failed",
            run_id=args.run_id,
            leg_index=args.leg_index,
            status="failed",
            stop_reason=str(exc),
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="route_validation_failed",
            stop_reason=str(exc),
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        parser.exit(2, f"error: route validation failed: {exc}\n")

    diagnostics_status = validate_route_diagnostics_json(
        args.diagnostics_json,
        args.leg_index,
        csv_point_count=len(leg.raw_waypoints),
        require_motion=require_motion,
    )
    speed_status = validate_speed_limits(args.max_linear_mps, args.max_angular_radps)
    pure_failures = diagnostics_status.failures + speed_status.failures
    if pure_failures:
        stop_reason = "; ".join(pure_failures)
        emit_event(
            event_logger,
            "route_validation_failed",
            run_id=args.run_id,
            leg_index=args.leg_index,
            status="failed",
            failures=pure_failures,
        )
        _append_status_result(
            args,
            resolved,
            leg,
            preflight_ok=False,
            status="route_validation_failed",
            stop_reason=stop_reason,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="route_validation_failed",
            stop_reason=stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        parser.exit(2, "error: validation failed:\n" + "\n".join(f"- {failure}" for failure in pure_failures) + "\n")

    emit_event(
        event_logger,
        "route_validated",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        raw_point_count=len(leg.raw_waypoints),
        executable_point_count=len(leg.executable_waypoints),
        route_length_m=leg.route_length_m,
        require_motion=require_motion,
        allow_noop=args.allow_noop,
    )
    print("Resolved runtime config:")
    print(json.dumps(resolved.as_log_dict(), indent=2, sort_keys=True))
    print(f"Semantic log: {args.semantic_log}")
    print(f"Results CSV: {args.results_csv}")
    print(
        "Route leg: "
        f"raw={len(leg.raw_waypoints)} executable={len(leg.executable_waypoints)} "
        f"length={leg.route_length_m:.3f}m"
    )
    if args.allow_noop and leg.route_length_m <= 0.0:
        result = FollowerResult("noop", "zero-length leg", 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=False, result=result)
        emit_event(
            event_logger,
            "dry_run_completed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            status=result.status,
            stop_reason=result.stop_reason,
            motion_published=result.motion_published,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        print("No-op leg logged; no motion was published.")
        return 0

    try:
        preflight = run_ros_preflight(
            resolved,
            max_scan_age_sec=args.max_scan_age_sec,
            max_odom_age_sec=args.max_odom_age_sec,
            max_tf_age_sec=args.max_tf_age_sec,
            max_amcl_age_sec=args.max_amcl_age_sec,
            allowed_cmd_vel_publishers=args.allowed_cmd_vel_publisher,
            require_real_time=not args.allow_sim_time,
        )
    except RuntimeError as exc:
        stop_reason = str(exc)
        emit_event(
            event_logger,
            "preflight_failed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            failures=[stop_reason],
            observations=[],
            runtime_config=resolved.as_log_dict(),
        )
        _append_status_result(
            args,
            resolved,
            leg,
            preflight_ok=False,
            status="preflight_unavailable",
            stop_reason=stop_reason,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="preflight_unavailable",
            stop_reason=stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        parser.exit(2, f"error: ROS preflight failed to run: {exc}\n")
    preflight_text = json.dumps(preflight.to_json_dict(), indent=2, sort_keys=True)
    if args.preflight_json is not None:
        args.preflight_json.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_json.write_text(preflight_text + "\n")
    print(preflight_text)
    if not preflight.ok:
        emit_event(
            event_logger,
            "preflight_failed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            failures=preflight.failures,
            observations=[observation.data | {"name": observation.name, "ok": observation.ok, "detail": observation.detail} for observation in preflight.observations],
            runtime_config=preflight.runtime_config,
        )
        result = FollowerResult("preflight_failed", "; ".join(preflight.failures), 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=False, result=result)
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            stop_reason=result.stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 1
    emit_event(
        event_logger,
        "preflight_passed",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        failures=[],
        observations=[observation.data | {"name": observation.name, "ok": observation.ok, "detail": observation.detail} for observation in preflight.observations],
        runtime_config=preflight.runtime_config,
    )
    if args.dry_run:
        result = FollowerResult("dry_run_ok", "", 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=True, result=result)
        emit_event(
            event_logger,
            "dry_run_completed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            status=result.status,
            motion_published=result.motion_published,
            results_csv=str(args.results_csv),
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 0
    if not _confirm_motion(args, resolved):
        result = FollowerResult("aborted", "operator did not type RUN", 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=True, result=result)
        emit_event(
            event_logger,
            "operator_aborted",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            status=result.status,
            stop_reason=result.stop_reason,
            motion_published=result.motion_published,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            stop_reason=result.stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 1

    follower_config = FollowerConfig(
        controller=ControllerConfig(
            max_linear_mps=args.max_linear_mps,
            max_angular_radps=args.max_angular_radps,
            goal_tolerance_m=args.goal_tolerance_m,
            heading_tolerance_rad=args.heading_tolerance_rad,
        ),
        min_obstacle_distance_m=args.min_obstacle_distance_m,
        max_scan_age_sec=args.max_scan_age_sec,
        max_odom_age_sec=args.max_odom_age_sec,
        max_tf_age_sec=args.max_tf_age_sec,
        waypoint_timeout_sec=args.waypoint_timeout_sec,
        initial_distance_limit_m=args.initial_distance_limit_m,
    )
    emit_event(
        event_logger,
        "motion_started",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        resolved_cmd_vel_topic=resolved.cmd_vel_topic,
    )
    result = run_simple_waypoint_follower(
        resolved,
        poses_from_waypoints(leg.executable_waypoints),
        follower_config,
    )
    _append_result(args, resolved, leg, preflight_ok=True, result=result)
    emit_event(
        event_logger,
        "motion_completed" if result.status == "completed" else "safety_stop",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        status=result.status,
        stop_reason=result.stop_reason,
        duration_sec=result.duration_sec,
        distance_estimate_m=result.distance_estimate_m,
        motion_published=result.motion_published,
    )
    emit_event(
        event_logger,
        "run_finished",
        run_id=args.run_id,
        final_status=result.status,
        stop_reason=result.stop_reason,
        results_csv=str(args.results_csv),
        semantic_log_path=str(args.semantic_log),
        preflight_json_path=str(args.preflight_json or ""),
    )
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
