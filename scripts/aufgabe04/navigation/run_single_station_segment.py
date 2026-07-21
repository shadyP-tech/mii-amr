"""Run one validated Aufgabe 04 station-route segment on a TurtleBot."""

from __future__ import annotations

import argparse
import json
import math
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
from scripts.aufgabe04.navigation.dynamic_route_handoff import DynamicRouteSource
from scripts.aufgabe04.navigation.route_revision_store import (
    LoadedRouteRevision,
    RouteRevisionError,
    read_committed_revision,
    read_route_revision,
)
from scripts.aufgabe04.navigation.safety_checks import (
    catalog_start_egress_certificate,
    validate_catalog_route_binding_json,
    validate_route_diagnostics_json,
    validate_speed_limits,
)
from scripts.aufgabe04.navigation.segment_run_logger import append_segment_run
from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    DYNAMIC_VIEWPOINT_ROUTE_KINDS,
    FollowerConfig,
    PHYSICAL_ROUTE_KINDS,
    STATIC_PHYSICAL_ROUTE_KINDS,
    run_simple_waypoint_follower,
)
from scripts.aufgabe04.navigation.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg, poses_from_waypoints


DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/station_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/station_route_diagnostics.json")
DEFAULT_RUN_LOG = Path("results/aufgabe04/station_segment_runs.csv")
DEFAULT_EVENT_LOG_DIR = Path("results/aufgabe04/run_events")
_CATALOG_ROUTE_INITIAL_DISTANCE_LIMIT_M = 0.15


def _execution_initial_distance_limit(requested_m: float, route_kind: str) -> float:
    """Prevent an unchecked long join onto a frozen catalog route."""

    if route_kind in STATIC_PHYSICAL_ROUTE_KINDS:
        return min(requested_m, _CATALOG_ROUTE_INITIAL_DISTANCE_LIMIT_M)
    return requested_m


def _load_execution_route_leg(
    route_csv_path: Path,
    leg_index: int,
    *,
    require_motion: bool,
    requested_thinning_min_spacing_m: float,
    authoritative_dynamic_route: bool,
):
    """Load a leg without weakening a collision-certified physical route.

    Generic CSV thinning is useful for legacy dense grid routes, but it joins
    retained points with unchecked straight chords.  Dynamic manifest routes
    already disable it.  Frozen catalog routes are likewise prevalidated and
    must retain their exact A* polyline plus protected terminal corridor.

    Route kind is stored in the CSV itself, so a non-authoritative route is
    first parsed normally and then reloaded without thinning when it identifies
    itself as a static physical catalog route.  No motion can occur between
    those pure reads.
    """

    initial_spacing = (
        0.0 if authoritative_dynamic_route else requested_thinning_min_spacing_m
    )
    leg = load_route_leg(
        route_csv_path,
        leg_index,
        require_motion=require_motion,
        thinning_min_spacing_m=initial_spacing,
    )
    if (
        not authoritative_dynamic_route
        and initial_spacing > 0.0
        and leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS
    ):
        leg = load_route_leg(
            route_csv_path,
            leg_index,
            require_motion=require_motion,
            thinning_min_spacing_m=0.0,
        )
    return leg


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
    parser.add_argument("--max-linear-mps", type=float, default=0.055)
    parser.add_argument("--max-angular-radps", type=float, default=0.18)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.08)
    parser.add_argument(
        "--physical-goal-tolerance-m",
        type=float,
        default=0.03,
        help=(
            "Simulation dynamic physical-face routes use at most this terminal "
            "position tolerance; acquisition and sampling retain --goal-tolerance-m."
        ),
    )
    parser.add_argument("--heading-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--lookahead-distance-m", type=float, default=0.18)
    parser.add_argument("--slow-heading-error-rad", type=float, default=0.75)
    parser.add_argument("--stop-heading-error-rad", type=float, default=1.25)
    parser.add_argument("--min-linear-speed-scale", type=float, default=0.35)
    parser.add_argument("--max-progress-advance-m", type=float, default=0.45)
    parser.add_argument("--min-obstacle-distance-m", type=float, default=0.20)
    parser.add_argument("--front-obstacle-slow-distance-m", type=float, default=0.38)
    parser.add_argument("--front-obstacle-sector-rad", type=float, default=0.6108652381980153)
    parser.add_argument("--thinning-min-spacing-m", type=float, default=0.15)
    parser.add_argument("--max-scan-age-sec", type=float, default=1.0)
    parser.add_argument("--max-odom-age-sec", type=float, default=1.0)
    parser.add_argument("--max-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--max-amcl-age-sec", type=float, default=2.0)
    parser.add_argument("--preflight-observation-window-sec", type=float, default=2.0)
    parser.add_argument("--initial-sensor-wait-sec", type=float, default=2.0)
    parser.add_argument("--waypoint-timeout-sec", type=float, default=45.0)
    parser.add_argument(
        "--axis-acquisition-wait-timeout-sec",
        type=float,
        default=12.0,
        help=(
            "Simulation-only stationary hold at an axis_acquisition goal while "
            "waiting for a committed physical-face route revision."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-timeout-sec",
        type=float,
        default=30.0,
        help=(
            "Simulation-only total budget for the viewpoint-sampling phase, "
            "including travel and stationary observation."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-goal-tolerance-m",
        type=float,
        default=0.01,
        help=(
            "Simulation-only position tolerance for tangential camera samples; "
            "kept tighter than generic transit so angular viewpoint corrections "
            "are not consumed by position tolerance."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-heading-tolerance-rad",
        type=float,
        default=math.radians(5.0),
        help=(
            "Simulation-only terminal heading tolerance for camera sampling; "
            "kept tight enough for the stand to satisfy the image-centering gate."
        ),
    )
    parser.add_argument("--stuck-timeout-sec", type=float, default=8.0)
    parser.add_argument("--stuck-progress-epsilon-m", type=float, default=0.03)
    parser.add_argument(
        "--stuck-heading-progress-epsilon-rad",
        type=float,
        default=0.10,
        help=(
            "Minimum controlled-heading improvement that resets the stuck "
            "watchdog while turning toward the active pursuit waypoint."
        ),
    )
    parser.add_argument("--initial-distance-limit-m", type=float, default=0.35)
    parser.add_argument(
        "--dynamic-route-refresh-sec",
        type=float,
        default=0.0,
        help="Simulation-only: hot-reload an atomically replaced A* route at this interval.",
    )
    parser.add_argument(
        "--route-manifest",
        type=Path,
        default=None,
        help="Authoritative simulation route-revision manifest (required for dynamic viewpoint routes).",
    )
    parser.add_argument("--max-route-manifest-age-sec", type=float, default=7.0)
    parser.add_argument("--max-route-observation-age-sec", type=float, default=6.0)
    parser.add_argument("--max-route-join-distance-m", type=float, default=0.35)
    parser.add_argument(
        "--dynamic-route-join-tolerance-m",
        type=float,
        default=0.02,
        help=(
            "Simulation-only tolerance for completing the certified join anchor "
            "after a live route revision."
        ),
    )
    parser.add_argument(
        "--start-egress-waypoint-tolerance-m",
        type=float,
        default=0.02,
        help=(
            "Simulation-only release tolerance for waypoint 1 when a route "
            "uses a certified start-cell raster exemption."
        ),
    )
    parser.add_argument(
        "--start-egress-alignment-tolerance-rad",
        type=float,
        default=0.10,
        help=(
            "Simulation-only heading error below which translation may begin "
            "toward a certified start-egress vertex."
        ),
    )
    parser.add_argument(
        "--start-egress-max-linear-mps",
        type=float,
        default=0.03,
        help=(
            "Simulation-only linear-speed cap while pursuing a certified "
            "start-egress vertex."
        ),
    )
    parser.add_argument(
        "--dynamic-route-terminal-lock-distance-m",
        type=float,
        default=0.42,
        help=(
            "Simulation-only distance at which the installed terminal route remains "
            "valid while newer target revisions continue to be polled."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-noop", action="store_true")
    parser.add_argument(
        "--prompt-for-initialpose",
        action="store_true",
        help=(
            "Pause immediately before ROS preflight so the operator can click "
            "RViz 2D Pose Estimate and refresh AMCL."
        ),
    )
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
    print(f"  - after RUN, wait up to {args.initial_sensor_wait_sec:.1f}s for follower scan/odom/TF before motion")
    print(f"Run ID: {args.run_id}")
    print(f"Resolved cmd_vel: {resolved.cmd_vel_topic}")


def _confirm_motion(args, resolved) -> bool:
    if args.allow_sim_time:
        print("Simulation run detected (--allow-sim-time); starting without a blocking RUN prompt.")
        return True
    _physical_checklist(args, resolved)
    response = input("Type RUN to start station-segment following: ").strip()
    return response == "RUN"


def _prompt_for_initialpose(args, resolved) -> None:
    if not args.prompt_for_initialpose:
        return
    print("\nInitial-pose refresh required before ROS preflight.")
    print("AMCL often publishes only once after RViz 2D Pose Estimate.")
    print("The preflight subscriber must already be active, so do not click yet.")
    print(f"AMCL topic: {resolved.amcl_topic}")
    print(
        "Press Enter here, then immediately click 2D Pose Estimate in RViz "
        f"during the next {args.preflight_observation_window_sec:.1f}s."
    )
    input("Press Enter, then click 2D Pose Estimate immediately: ")


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


def _observation_log_rows(observations) -> list[dict[str, object]]:
    return [
        {
            **observation.data,
            "name": observation.name,
            "ok": observation.ok,
            "detail": observation.detail,
        }
        for observation in observations
    ]


def _authoritative_route_paths(
    args,
) -> tuple[Path, Path, LoadedRouteRevision | None]:
    manifest_path = args.route_manifest
    if manifest_path is None:
        candidate = args.route_csv.with_suffix(".manifest.json")
        if candidate.exists():
            manifest_path = candidate
    if manifest_path is None:
        return args.route_csv, args.diagnostics_json, None
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"route manifest does not exist: {manifest_path}")
    committed = read_committed_revision(
        manifest_path,
        now_unix_sec=datetime.now(timezone.utc).timestamp(),
        # A one-shot synchronized camera/LiDAR route is no safer to execute
        # stale than a hot-reloaded one.  Every authoritative simulation
        # revision is freshness-gated before preflight or motion.
        max_manifest_age_sec=args.max_route_manifest_age_sec,
        max_observation_age_sec=args.max_route_observation_age_sec,
    )
    if committed.status != "active" or committed.route_path is None:
        raise ValueError(f"authoritative route is withdrawn: {committed.reason}")
    if committed.diagnostics_path is None:
        raise ValueError("authoritative route manifest has no diagnostics artifact")
    args.route_manifest = manifest_path
    return committed.route_path, committed.diagnostics_path, committed


def _revalidate_authoritative_route_before_motion(
    args, committed: LoadedRouteRevision
) -> None:
    """Require the exact initially validated revision to remain live."""

    latest = read_route_revision(
        committed.manifest_path,
        expected_stream_id=str(committed.manifest["stream_id"]),
        expected_writer_id=committed.writer_id,
        last_route_revision=committed.route_revision,
        last_manifest_sha256=committed.manifest_sha256,
        max_manifest_age_sec=args.max_route_manifest_age_sec,
        max_observation_age_sec=args.max_route_observation_age_sec,
        now_unix_sec=datetime.now(timezone.utc).timestamp(),
    )
    same_authorized_route = (
        latest.status == "active"
        and latest.route_hash == committed.route_hash
        and latest.target_revision == committed.target_revision
        and latest.writer_id == committed.writer_id
        and latest.writer_generation == committed.writer_generation
    )
    if not latest.duplicate and not same_authorized_route:
        raise RouteRevisionError(
            "route_changed_before_motion",
            "authoritative route changed or was withdrawn before motion authorization",
        )
    if latest.status != "active" or latest.route_hash != committed.route_hash:
        raise RouteRevisionError(
            "route_changed_before_motion",
            "authoritative route artifact changed before motion authorization",
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dynamic_route_refresh_sec < 0.0:
        parser.error("--dynamic-route-refresh-sec must be non-negative")
    if args.dynamic_route_refresh_sec > 0.0 and not args.allow_sim_time:
        parser.error("dynamic route hot-reload is simulation-only and requires --allow-sim-time")
    if args.max_route_manifest_age_sec <= 0.0 or args.max_route_observation_age_sec <= 0.0:
        parser.error("dynamic route freshness limits must be positive")
    if args.max_route_join_distance_m <= 0.0:
        parser.error("--max-route-join-distance-m must be positive")
    if (
        not math.isfinite(args.axis_acquisition_wait_timeout_sec)
        or args.axis_acquisition_wait_timeout_sec <= 0.0
    ):
        parser.error("--axis-acquisition-wait-timeout-sec must be positive")
    if (
        not math.isfinite(args.viewpoint_sampling_timeout_sec)
        or args.viewpoint_sampling_timeout_sec <= 0.0
    ):
        parser.error("--viewpoint-sampling-timeout-sec must be positive")
    if (
        not math.isfinite(args.physical_goal_tolerance_m)
        or args.physical_goal_tolerance_m <= 0.0
    ):
        parser.error("--physical-goal-tolerance-m must be positive")
    if (
        not math.isfinite(args.viewpoint_sampling_goal_tolerance_m)
        or args.viewpoint_sampling_goal_tolerance_m <= 0.0
    ):
        parser.error("--viewpoint-sampling-goal-tolerance-m must be positive")
    if (
        not math.isfinite(args.viewpoint_sampling_heading_tolerance_rad)
        or args.viewpoint_sampling_heading_tolerance_rad <= 0.0
    ):
        parser.error("--viewpoint-sampling-heading-tolerance-rad must be positive")
    if (
        not math.isfinite(args.dynamic_route_join_tolerance_m)
        or args.dynamic_route_join_tolerance_m <= 0.0
    ):
        parser.error("--dynamic-route-join-tolerance-m must be positive")
    if (
        not math.isfinite(args.start_egress_waypoint_tolerance_m)
        or args.start_egress_waypoint_tolerance_m <= 0.0
    ):
        parser.error("--start-egress-waypoint-tolerance-m must be positive")
    if (
        not math.isfinite(args.start_egress_alignment_tolerance_rad)
        or args.start_egress_alignment_tolerance_rad <= 0.0
        or args.start_egress_alignment_tolerance_rad > math.pi / 2.0
    ):
        parser.error(
            "--start-egress-alignment-tolerance-rad must be in (0, pi/2]"
        )
    if (
        not math.isfinite(args.start_egress_max_linear_mps)
        or args.start_egress_max_linear_mps <= 0.0
    ):
        parser.error("--start-egress-max-linear-mps must be positive")
    if (
        not math.isfinite(args.stuck_heading_progress_epsilon_rad)
        or args.stuck_heading_progress_epsilon_rad <= 0.0
    ):
        parser.error("--stuck-heading-progress-epsilon-rad must be positive")
    if (
        not math.isfinite(args.dynamic_route_terminal_lock_distance_m)
        or args.dynamic_route_terminal_lock_distance_m <= 0.0
    ):
        parser.error("--dynamic-route-terminal-lock-distance-m must be positive")
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
    try:
        route_csv_path, diagnostics_json_path, committed_route = _authoritative_route_paths(args)
    except (OSError, ValueError, RouteRevisionError) as exc:
        emit_event(
            event_logger,
            "route_manifest_rejected",
            run_id=args.run_id,
            status="failed",
            stop_reason=str(exc),
            route_manifest=str(args.route_manifest or ""),
        )
        parser.exit(2, f"error: authoritative route validation failed: {exc}\n")
    if committed_route is not None and not args.allow_sim_time:
        parser.exit(2, "error: authoritative dynamic route is simulation-only\n")
    if args.dynamic_route_refresh_sec > 0.0 and committed_route is None:
        parser.exit(2, "error: dynamic route refresh requires an authoritative route manifest\n")
    emit_event(
        event_logger,
        "run_started",
        run_id=args.run_id,
        robot_id=args.robot_id,
        route_csv=str(args.route_csv),
        diagnostics_json=str(args.diagnostics_json),
        authoritative_route_csv=str(route_csv_path),
        authoritative_diagnostics_json=str(diagnostics_json_path),
        route_manifest=str(args.route_manifest or ""),
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
    if committed_route is not None:
        manifest = committed_route.manifest
        emit_event(
            event_logger,
            "authoritative_route_resolved",
            run_id=args.run_id,
            leg_index=args.leg_index,
            route_manifest=str(committed_route.manifest_path),
            manifest_sha256=committed_route.manifest_sha256,
            stream_id=manifest["stream_id"],
            writer_id=committed_route.writer_id,
            writer_generation=committed_route.writer_generation,
            route_revision=committed_route.route_revision,
            target_revision=committed_route.target_revision,
            route_sha256=committed_route.route_hash,
            published_unix_sec=manifest["published_unix_sec"],
            observation_unix_sec=manifest["observation_unix_sec"],
            source_robot_pose=manifest.get("source_robot_pose", {}),
            target=manifest.get("target", {}),
            previous_route_length_m=manifest.get("previous_route_length_m"),
            new_route_length_m=manifest.get("new_route_length_m"),
        )
    try:
        leg = _load_execution_route_leg(
            route_csv_path,
            args.leg_index,
            require_motion=require_motion,
            requested_thinning_min_spacing_m=args.thinning_min_spacing_m,
            authoritative_dynamic_route=committed_route is not None,
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

    if leg.route_kind in DYNAMIC_VIEWPOINT_ROUTE_KINDS and not leg.simulation_only:
        parser.exit(2, "error: dynamic viewpoint route is missing simulation_only provenance\n")
    if leg.route_kind in DYNAMIC_VIEWPOINT_ROUTE_KINDS and committed_route is None:
        parser.exit(2, "error: dynamic viewpoint route requires its authoritative manifest\n")
    if leg.simulation_only and not args.allow_sim_time:
        parser.exit(
            2,
            "error: simulation-only synchronized-viewpoint routes require --allow-sim-time\n",
        )
    if committed_route is not None and leg.route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        parser.exit(
            2,
            f"error: authoritative route has unknown dynamic route kind: {leg.route_kind!r}\n",
        )

    diagnostics_status = validate_route_diagnostics_json(
        diagnostics_json_path,
        args.leg_index,
        csv_point_count=len(leg.raw_waypoints),
        require_motion=require_motion,
    )
    catalog_binding_status = (
        validate_catalog_route_binding_json(diagnostics_json_path, leg)
        if leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS
        else None
    )
    catalog_egress_certificate = None
    catalog_egress_failures = []
    if leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS:
        try:
            catalog_egress_certificate = catalog_start_egress_certificate(
                diagnostics_json_path,
                leg,
            )
        except ValueError as exc:
            catalog_egress_failures.append(
                f"catalog start-egress certificate is invalid: {exc}"
            )
    speed_status = validate_speed_limits(args.max_linear_mps, args.max_angular_radps)
    pure_failures = (
        diagnostics_status.failures
        + ([] if catalog_binding_status is None else catalog_binding_status.failures)
        + catalog_egress_failures
        + speed_status.failures
    )
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

    _prompt_for_initialpose(args, resolved)

    try:
        preflight = run_ros_preflight(
            resolved,
            max_scan_age_sec=args.max_scan_age_sec,
            max_odom_age_sec=args.max_odom_age_sec,
            max_tf_age_sec=args.max_tf_age_sec,
            max_amcl_age_sec=args.max_amcl_age_sec,
            observation_window_sec=args.preflight_observation_window_sec,
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
            observations=_observation_log_rows(preflight.observations),
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
        observations=_observation_log_rows(preflight.observations),
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

    if committed_route is not None:
        try:
            _revalidate_authoritative_route_before_motion(args, committed_route)
        except (OSError, RouteRevisionError) as exc:
            stop_reason = f"authoritative route revalidation failed: {exc}"
            emit_event(
                event_logger,
                "route_manifest_rejected",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status="stopped",
                phase="immediately_before_motion",
                stop_reason=stop_reason,
                route_manifest=str(committed_route.manifest_path),
            )
            result = FollowerResult(
                "stopped",
                stop_reason,
                0.0,
                0.0,
                False,
                {
                    "fault_code": getattr(exc, "code", "route_revalidation_io"),
                    "fail_closed": True,
                },
            )
            _append_result(args, resolved, leg, preflight_ok=True, result=result)
            emit_event(
                event_logger,
                "safety_stop",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status=result.status,
                stop_reason=result.stop_reason,
                motion_published=False,
                stop_details=result.stop_details,
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

    execution_initial_distance_limit_m = _execution_initial_distance_limit(
        args.initial_distance_limit_m,
        leg.route_kind,
    )
    static_start_join_clearance_m = (
        None
        if catalog_egress_certificate is None
        or not catalog_egress_certificate.required
        or catalog_egress_certificate.start_join_clearance_m is None
        else min(
            execution_initial_distance_limit_m,
            catalog_egress_certificate.start_join_clearance_m,
        )
    )
    follower_config = FollowerConfig(
        controller=ControllerConfig(
            max_linear_mps=args.max_linear_mps,
            max_angular_radps=args.max_angular_radps,
            goal_tolerance_m=args.goal_tolerance_m,
            heading_tolerance_rad=args.heading_tolerance_rad,
            lookahead_distance_m=args.lookahead_distance_m,
            slow_heading_error_rad=args.slow_heading_error_rad,
            stop_heading_error_rad=args.stop_heading_error_rad,
            min_linear_speed_scale=args.min_linear_speed_scale,
            max_progress_advance_m=args.max_progress_advance_m,
            enforce_heading_corridor=leg.route_kind in PHYSICAL_ROUTE_KINDS,
        ),
        min_obstacle_distance_m=args.min_obstacle_distance_m,
        front_obstacle_slow_distance_m=args.front_obstacle_slow_distance_m,
        front_obstacle_sector_rad=args.front_obstacle_sector_rad,
        max_scan_age_sec=args.max_scan_age_sec,
        max_odom_age_sec=args.max_odom_age_sec,
        max_tf_age_sec=args.max_tf_age_sec,
        initial_sensor_wait_sec=args.initial_sensor_wait_sec,
        waypoint_timeout_sec=args.waypoint_timeout_sec,
        stuck_timeout_sec=args.stuck_timeout_sec,
        stuck_progress_epsilon_m=args.stuck_progress_epsilon_m,
        stuck_heading_progress_epsilon_rad=(
            args.stuck_heading_progress_epsilon_rad
        ),
        initial_distance_limit_m=execution_initial_distance_limit_m,
        allowed_cmd_vel_publishers=tuple(args.allowed_cmd_vel_publisher),
        dynamic_route_refresh_sec=args.dynamic_route_refresh_sec,
        dynamic_join_tolerance_m=args.dynamic_route_join_tolerance_m,
        start_egress_waypoint_tolerance_m=(
            args.start_egress_waypoint_tolerance_m
        ),
        start_egress_alignment_tolerance_rad=(
            args.start_egress_alignment_tolerance_rad
        ),
        start_egress_max_linear_mps=args.start_egress_max_linear_mps,
        initial_start_egress_waypoint_index=(
            None
            if catalog_egress_certificate is None
            else catalog_egress_certificate.waypoint_index
        ),
        initial_start_join_clearance_m=static_start_join_clearance_m,
        initial_route_kind=leg.route_kind,
        axis_acquisition_wait_timeout_sec=args.axis_acquisition_wait_timeout_sec,
        viewpoint_sampling_timeout_sec=args.viewpoint_sampling_timeout_sec,
        viewpoint_sampling_goal_tolerance_m=(
            args.viewpoint_sampling_goal_tolerance_m
        ),
        viewpoint_sampling_heading_tolerance_rad=(
            args.viewpoint_sampling_heading_tolerance_rad
        ),
        physical_goal_tolerance_m=args.physical_goal_tolerance_m,
    )
    waypoint_provider = None
    route_update_callback = None
    if committed_route is not None:
        assert committed_route is not None and args.route_manifest is not None
        route_source = DynamicRouteSource(
            args.route_manifest,
            stream_id=str(committed_route.manifest["stream_id"]),
            leg_index=args.leg_index,
            expected_writer_id=committed_route.writer_id,
            max_manifest_age_sec=args.max_route_manifest_age_sec,
            max_observation_age_sec=args.max_route_observation_age_sec,
            max_join_distance_m=args.max_route_join_distance_m,
            terminal_route_lock_distance_m=(
                args.dynamic_route_terminal_lock_distance_m
            ),
            # The dynamic planner already emitted a collision-checked,
            # shortcut route. Generic thinning could create an unchecked
            # chord, so authoritative dynamic revisions are never re-thinned.
            thinning_min_spacing_m=0.0,
        )

        def waypoint_provider(pose):
            return route_source.poll(pose)

        def route_update_callback(update):
            event_name = {
                "dynamic_route_adopted": "route_reloaded",
                "dynamic_route_withdrawn": "route_withdrawn",
                "dynamic_route_rejected": "route_reload_rejected",
                "dynamic_route_stopped": "route_reload_rejected",
                "dynamic_survey_completed": "survey_completed",
            }.get(update.event_name, update.event_name)
            if event_name is None:
                return
            emit_event(
                event_logger,
                event_name,
                run_id=args.run_id,
                leg_index=args.leg_index,
                **dict(update.event_fields),
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
        waypoint_provider,
        route_update_callback,
    )
    _append_result(args, resolved, leg, preflight_ok=True, result=result)
    motion_event_fields = {
        "run_id": args.run_id,
        "leg_index": leg.leg_index,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "duration_sec": result.duration_sec,
        "distance_estimate_m": result.distance_estimate_m,
        "motion_published": result.motion_published,
    }
    if result.status != "completed":
        motion_event_fields["stop_details"] = result.stop_details or {}
    emit_event(
        event_logger,
        "motion_completed" if result.status == "completed" else "safety_stop",
        **motion_event_fields,
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
