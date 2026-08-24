"""Pure route loading, identity, and startup-certificate checks."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.driving_behavior import (
    DYNAMIC_VIEWPOINT_ROUTE_KINDS,
    STATIC_PHYSICAL_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.execution_route_certificate import (
    execution_route_certificate_sha256,
    load_execution_route_certificate,
    validate_execution_route_identity,
)
from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.mission_execution_gate import DiagnosticsSnapshot
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.route_revision_store import (
    LoadedRouteRevision,
    RouteRevisionError,
    read_committed_revision,
    read_route_revision,
)
from scripts.aufgabe04.navigation.waypoint_csv import (
    SelectedRouteLeg,
    load_route_leg,
    poses_from_waypoints,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    certified_static_startup_decision,
)

_CATALOG_ROUTE_INITIAL_DISTANCE_LIMIT_M = 0.15

def _execution_initial_distance_limit(requested_m: float, route_kind: str) -> float:
    """Prevent an unchecked long join onto a frozen catalog route."""

    if route_kind in STATIC_PHYSICAL_ROUTE_KINDS:
        return min(requested_m, _CATALOG_ROUTE_INITIAL_DISTANCE_LIMIT_M)
    return requested_m

def _static_start_preflight_rejection(
    preflight: RosPreflightResult,
    leg: SelectedRouteLeg,
    *,
    map_frame: str,
    base_frame: str,
    tracking_tube_radius_m: float,
) -> FollowerResult | None:
    """Reject a stale sealed start before motion confirmation is requested."""

    if leg.route_kind not in STATIC_PHYSICAL_ROUTE_KINDS:
        return None
    raw_pose = preflight.route_pose
    if not isinstance(raw_pose, Mapping):
        return FollowerResult(
            "stopped",
            "preflight did not provide a route-frame startup pose",
            0.0,
            0.0,
            False,
            {
                "reason": "preflight did not provide a route-frame startup pose",
                "source": "ros_preflight",
                "phase": "before_motion_confirmation",
                "fail_closed": True,
            },
        )
    try:
        pose = Pose2D(
            float(raw_pose["x_m"]),
            float(raw_pose["y_m"]),
            float(raw_pose["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return FollowerResult(
            "stopped",
            "preflight route-frame startup pose is invalid",
            0.0,
            0.0,
            False,
            {
                "reason": "preflight route-frame startup pose is invalid",
                "source": "ros_preflight",
                "phase": "before_motion_confirmation",
                "error": str(exc),
                "fail_closed": True,
            },
        )
    if (
        raw_pose.get("frame_id") != map_frame
        or raw_pose.get("child_frame_id") != base_frame
        or not all(
            math.isfinite(value)
            for value in (pose.x_m, pose.y_m, pose.yaw_rad)
        )
    ):
        return FollowerResult(
            "stopped",
            "preflight route-frame startup pose is invalid",
            0.0,
            0.0,
            False,
            {
                "reason": "preflight route-frame startup pose is invalid",
                "source": "ros_preflight",
                "phase": "before_motion_confirmation",
                "route_pose": dict(raw_pose),
                "fail_closed": True,
            },
        )

    decision = certified_static_startup_decision(
        pose,
        poses_from_waypoints(leg.executable_waypoints),
        tracking_tube_radius_m=tracking_tube_radius_m,
    )
    if decision.ok:
        return None
    return FollowerResult(
        "stopped",
        "pose outside certified startup segment",
        0.0,
        0.0,
        False,
        {
            **decision.route_check.to_log_dict(),
            "reason": "pose outside certified startup segment",
            "certificate_reason": decision.route_check.reason,
            "startup_target_candidates": [0, 1],
            "source": "execution_route_certificate",
            "phase": "before_motion_confirmation",
            "route_pose": {
                "frame_id": map_frame,
                "child_frame_id": base_frame,
                "x_m": pose.x_m,
                "y_m": pose.y_m,
                "yaw_rad": pose.yaw_rad,
            },
            "fail_closed": True,
        },
    )

def _simulation_odom_fallback_admission_failure(
    args,
    resolved,
    leg: SelectedRouteLeg,
    *,
    route_purpose: str,
    authoritative_dynamic_route: bool,
) -> str:
    """Return why the explicit direct-odometry recovery is not admissible."""

    if not args.allow_simulation_odom_after_stale_tf:
        return ""
    if not args.allow_sim_time:
        return (
            "--allow-simulation-odom-after-stale-tf requires "
            "--allow-sim-time"
        )
    if resolved.use_sim_time is not True:
        return (
            "simulation odometry recovery requires resolved "
            "use_sim_time=true"
        )
    if resolved.localization_source != "tf":
        return (
            "simulation odometry recovery requires "
            "--localization-source tf"
        )
    if resolved.map_frame != resolved.odom_frame:
        return (
            "simulation odometry recovery requires map_frame == odom_frame"
        )
    if not leg.simulation_only:
        return (
            "simulation odometry recovery requires "
            "route simulation_only=true"
        )

    static_survey = (
        leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS
        and route_purpose == "survey"
        and args.allow_unbound_survey_simulation_route
    )
    dynamic_viewpoint_survey = (
        leg.route_kind in DYNAMIC_VIEWPOINT_ROUTE_KINDS
        and authoritative_dynamic_route
        and route_purpose in ("", "survey")
    )
    if not (static_survey or dynamic_viewpoint_survey):
        return (
            "simulation odometry recovery is admitted only for an explicit "
            "static route_purpose=survey demonstration or an authoritative "
            "dynamic viewpoint survey; logistics, legacy, and unknown static "
            "route purposes are rejected"
        )
    return ""

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

def _runtime_command_owner(namespace: str) -> str:
    normalized = str(namespace).strip()
    if not normalized or normalized == "/":
        return "/aufgabe04_simple_waypoint_follower"
    return f"/{normalized.strip('/')}/aufgabe04_simple_waypoint_follower"

def _execution_certificate_failures(
    *,
    route_leg: SelectedRouteLeg,
    diagnostics_snapshot: DiagnosticsSnapshot,
    explicit_certificate_path: Path | None,
    route_kind: str,
    runtime_namespace: str,
    runtime_planning_frame: str,
    tracking_tube_radius_m: float,
) -> list[str]:
    try:
        metadata = diagnostics_snapshot.metadata
        recorded_path = metadata.get("route_certificate_path")
        certificate_path = explicit_certificate_path
        if certificate_path is None:
            if not isinstance(recorded_path, str) or not recorded_path:
                raise ValueError("physical route has no execution certificate")
            certificate_path = Path(recorded_path)
        certificate = load_execution_route_certificate(certificate_path)
        recorded_hash = metadata.get("route_certificate_sha256")
        if (
            not isinstance(recorded_hash, str)
            or recorded_hash != execution_route_certificate_sha256(certificate)
        ):
            raise ValueError(
                "execution route certificate SHA-256 does not match diagnostics"
            )
        map_bundle_sha256 = metadata.get("map_bundle_sha256", "")
        candidate_snapshot_sha256 = metadata.get(
            "candidate_snapshot_sha256", ""
        )
        diagnostics_planning_frame = str(metadata.get("planning_frame", ""))
        if diagnostics_planning_frame != runtime_planning_frame:
            raise ValueError(
                "route diagnostics planning frame differs from runtime map frame: "
                f"diagnostics={diagnostics_planning_frame!r}, "
                f"runtime={runtime_planning_frame!r}"
            )
        validate_execution_route_identity(
            certificate,
            route_path=route_leg.source_path,
            route_snapshot_sha256=route_leg.source_sha256,
            planning_frame=runtime_planning_frame,
            route_kind=route_kind,
            waypoint_count=route_leg.source_waypoint_count,
            command_owner=_runtime_command_owner(runtime_namespace),
            map_bundle_sha256=str(map_bundle_sha256),
            candidate_snapshot_sha256=str(candidate_snapshot_sha256),
        )
        if abs(certificate.tracking_tube_radius_m - tracking_tube_radius_m) > 1.0e-9:
            raise ValueError(
                "runtime tracking tube differs from execution certificate"
            )
        return []
    except (OSError, ValueError) as exc:
        return [f"execution route certificate is invalid: {exc}"]

def _resolved_map_execution_certificate(
    args,
    diagnostics_snapshot: DiagnosticsSnapshot,
):
    certificate_path = args.route_certificate_json
    if certificate_path is None:
        recorded_path = diagnostics_snapshot.metadata.get(
            "route_certificate_path"
        )
        if not isinstance(recorded_path, str) or not recorded_path:
            raise ValueError("physical route has no execution certificate")
        certificate_path = Path(recorded_path)
    certificate = load_execution_route_certificate(certificate_path)
    return certificate, execution_route_certificate_sha256(certificate)

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

