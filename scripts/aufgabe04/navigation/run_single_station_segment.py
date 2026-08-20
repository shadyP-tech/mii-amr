"""Run one validated Aufgabe 04 station-route segment on a TurtleBot."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.ros_preflight import (
    RosPreflightResult,
    run_ros_preflight,
)
from scripts.aufgabe04.navigation.ros_runtime_config import (
    RuntimeConfig,
    resolve_topic,
    resolve_runtime_config,
)
from scripts.aufgabe04.navigation.run_events import configure_event_logger, emit_event
from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    DynamicRouteSource,
    RouteUpdate,
    RouteUpdateKind,
    validate_arena_boundary_evidence,
)
from scripts.aufgabe04.navigation.driving_behavior import (
    CATALOG_PHYSICAL_ROUTE_KINDS,
    CommandSmoothingConfig,
    DYNAMIC_VIEWPOINT_ROUTE_KINDS,
    HEADING_CORRIDOR_ROUTE_KINDS,
    PHYSICAL_ROUTE_KINDS,
    STATIC_PHYSICAL_ROUTE_KINDS,
    controller_config_for_route_kind,
)
from scripts.aufgabe04.navigation.content_hashed_evidence import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.coverage_replan_coordinator import (
    CoverageReplanCoordinator,
)
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
from scripts.aufgabe04.navigation.detected_stand_preapproach import (
    DETECTED_STAND_PREAPPROACH_ROUTE_KIND,
    validate_detected_stand_preapproach_binding,
)
from scripts.aufgabe04.navigation.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
    validate_stand_discovery_route_binding,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    load_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.segment_run_logger import append_segment_run
from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.execution_route_certificate import (
    execution_route_certificate_sha256,
    load_execution_route_certificate,
    validate_execution_route_identity,
)
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.mission_leg_motion_consumption import (
    consume_mission_leg_motion_permit,
    default_mission_leg_motion_consumption_receipt_path,
    mission_leg_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    MissionLegKind,
    MissionLegMotionPermit,
    mission_leg_motion_permit_sha256,
    validate_mission_leg_motion_permit_for_execution,
)
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    OdomExecutionCertificate,
    PlanarTransform2D,
    odom_execution_certificate_sha256,
    odom_pose_to_map,
    pose_route_sha256,
    transform_map_route_to_odom,
    validate_odom_execution_identity,
    write_odom_execution_certificate,
)
from scripts.aufgabe04.navigation.odom_route_adapter import (
    OdomExecutionContext,
    adapt_map_route_update_to_odom,
    evaluate_map_odom_stationary_stability,
)
from scripts.aufgabe04.navigation.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    evaluate_route_uncertainty_admission,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.runtime_motion_authorization import (
    RuntimeLocalizationMotionPermit,
    runtime_localization_motion_permit_sha256,
    validate_runtime_localization_motion_permit_for_execution,
)
from scripts.aufgabe04.navigation.runtime_motion_consumption import (
    consume_runtime_motion_permit,
    default_runtime_motion_consumption_receipt_path,
    runtime_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    StartupResealMotionPermit,
    startup_reseal_motion_permit_sha256,
    validate_startup_reseal_motion_permit_for_execution,
)
from scripts.aufgabe04.navigation.startup_reseal_motion_consumption import (
    consume_startup_reseal_motion_permit,
    default_startup_reseal_motion_consumption_receipt_path,
    startup_reseal_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.mission_execution_gate import (
    DiagnosticsSnapshot,
    MissionExecutionBinding,
    load_diagnostics_snapshot,
    validate_logistics_execution_bundle,
)
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
)
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    run_simple_waypoint_follower,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading import (
    intermediate_terminal_heading_entry_tolerance_m,
)
from scripts.aufgabe04.navigation.transient_blockage_policy import (
    DEFAULT_LINEAR_MOTION_FLOOR_MPS,
    PersistentObstacleConfig,
)
from scripts.aufgabe04.navigation.transient_overlay_resume_state import (
    TransientOverlayResumeState,
    transient_overlay_resume_state_sha256,
    validate_transient_overlay_resume_state_diagnostics_binding,
)
from scripts.aufgabe04.navigation.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.waypoint_csv import (
    SelectedRouteLeg,
    load_route_leg,
    poses_from_waypoints,
)


DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/station_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/station_route_diagnostics.json")


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """Append one post-adoption mission event or fail the zero-held handoff."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
            + "\n"
        )


DEFAULT_RUN_LOG = Path("results/aufgabe04/station_segment_runs.csv")
DEFAULT_EVENT_LOG_DIR = Path("results/aufgabe04/run_events")
_CATALOG_ROUTE_INITIAL_DISTANCE_LIMIT_M = 0.15
LEGACY_SIMULATION_ROUTE_KIND = "legacy_simulation_waypoint"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--route-certificate-json", type=Path, default=None)
    parser.add_argument("--mission-plan-manifest", type=Path, default=None)
    parser.add_argument("--survey-manifest", type=Path, default=None)
    parser.add_argument("--route-bundle-json", type=Path, default=None)
    parser.add_argument("--planner-config-json", type=Path, default=None)
    parser.add_argument("--runtime-map-bundle-json", type=Path, default=None)
    parser.add_argument("--runtime-environment", type=Path, default=None)
    parser.add_argument("--candidate-snapshot", type=Path, default=None)
    parser.add_argument("--coverage-plan", type=Path, default=None)
    parser.add_argument("--coverage-transient-replan-survey-root", type=Path)
    parser.add_argument("--coverage-transient-replan-session-root", type=Path)
    parser.add_argument("--coverage-transient-replan-map", type=Path)
    parser.add_argument("--coverage-transient-replan-semantic-map-id", default="")
    parser.add_argument("--coverage-transient-replan-target-viewpoint-id", default="")
    parser.add_argument("--coverage-transient-replan-robot-radius-m", type=float)
    parser.add_argument("--coverage-transient-replan-max-count", type=int, default=0)
    parser.add_argument("--coverage-transient-replan-leg-index", type=int)
    parser.add_argument(
        "--coverage-transient-replan-resume-state-json",
        type=Path,
        help=(
            "Internal immutable state for resuming a cumulative transient "
            "overlay after a stopped localization reseal."
        ),
    )
    parser.add_argument("--station-identity-registry", type=Path, default=None)
    parser.add_argument("--arrival-pose-catalog", type=Path, default=None)
    parser.add_argument("--task-snapshot", type=Path, default=None)
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
    parser.add_argument(
        "--execution-pose-frame",
        choices=["map", "odom"],
        default="map",
        help=(
            "Controller pose/route frame. The odom option freezes one "
            "map->odom transform after stopped preflight; AMCL then becomes "
            "a consistency monitor and never supplies pursuit poses."
        ),
    )
    parser.add_argument("--odom-execution-certificate-json", type=Path)
    parser.add_argument("--uncertainty-budget-json", type=Path)
    parser.add_argument("--uncertainty-map-yaml", type=Path)
    parser.add_argument("--localization-branch-proof-id", default="")
    parser.add_argument(
        "--mission-motion-authorization-json",
        type=Path,
        help=(
            "Internal autonomous-wrapper evidence for the mission-level RUN. "
            "It never bypasses gates by itself."
        ),
    )
    parser.add_argument(
        "--runtime-localization-motion-permit-json",
        type=Path,
        help=(
            "Internal one-run, same-target runtime-localization recovery permit."
        ),
    )
    parser.add_argument(
        "--startup-reseal-motion-authorization-json",
        type=Path,
        help=(
            "Internal autonomous-wrapper evidence that the mission RUN covers "
            "bounded same-target pre-motion startup recovery."
        ),
    )
    parser.add_argument(
        "--startup-reseal-motion-permit-json",
        type=Path,
        help="Internal one-use permit for this exact startup-reseal child.",
    )
    parser.add_argument("--startup-reseal-target-viewpoint-id", default="")
    parser.add_argument("--startup-reseal-semantic-map-id", default="")
    parser.add_argument(
        "--mission-leg-motion-authorization-json",
        type=Path,
        help=(
            "Internal autonomous-wrapper evidence for the one-time mission RUN "
            "covering separately sealed routine child legs."
        ),
    )
    parser.add_argument(
        "--mission-leg-motion-permit-json",
        type=Path,
        help="Internal one-use permit for this exact routine child leg.",
    )
    parser.add_argument(
        "--mission-leg-kind",
        choices=[
            MissionLegKind.COVERAGE.value,
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
            MissionLegKind.OPPOSITE_FACE.value,
        ],
    )
    parser.add_argument("--mission-leg-index", type=int)
    parser.add_argument("--mission-leg-target-id", default="")
    parser.add_argument("--mission-leg-semantic-map-id", default="")
    parser.add_argument("--mission-leg-dry-preflight-json", type=Path)
    parser.add_argument(
        "--mission-leg-dry-odom-certificate-json", type=Path
    )
    parser.add_argument(
        "--mission-leg-dry-uncertainty-budget-json", type=Path
    )
    parser.add_argument(
        "--mission-session-id",
        default="",
        help="Session identity bound by an internal motion permit.",
    )
    parser.add_argument("--uncertainty-robot-radius-m", type=float)
    parser.add_argument(
        "--uncertainty-collision-margin-m", type=float, default=0.02
    )
    parser.add_argument(
        "--uncertainty-odom-drift-bound-m", type=float, default=0.02
    )
    parser.add_argument(
        "--uncertainty-braking-latency-distance-m",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--uncertainty-sigma-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--uncertainty-heading-lever-arm-m", type=float, default=None
    )
    parser.add_argument(
        "--uncertainty-clearance-sample-spacing-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--max-map-odom-yaw-drift-rad", type=float, default=0.10
    )
    parser.add_argument(
        "--max-map-odom-translation-drift-m",
        type=float,
        default=0.15,
        help=(
            "Hard cap for live map<-odom translation drift. The effective "
            "threshold is the smaller of this cap and the localization "
            "position allowance already charged to the route budget."
        ),
    )
    parser.add_argument("--allow-sim-time", action="store_true")
    parser.add_argument(
        "--allow-simulation-odom-after-stale-tf",
        action="store_true",
        help=(
            "Simulation-survey-only recovery after the existing zero plus "
            "bounded stale-TF retry. Disabled by default and never admitted "
            "for logistics or real-time runs."
        ),
    )
    parser.add_argument("--max-linear-mps", type=float, default=0.055)
    parser.add_argument("--max-angular-radps", type=float, default=0.18)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.08)
    parser.add_argument(
        "--physical-waypoint-tolerance-m",
        type=float,
        default=0.02,
        help=(
            "Capture tolerance for intermediate vertices on certified physical "
            "routes; it must remain inside the certified route tube."
        ),
    )
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
    parser.add_argument(
        "--disable-command-smoothing",
        action="store_true",
        help=(
            "Publish raw controller commands after safety checks. By default "
            "normal commands are rate-limited; hard stops still publish zero "
            "immediately."
        ),
    )
    parser.add_argument(
        "--max-linear-accel-mps2",
        type=float,
        default=0.10,
        help="Maximum acceleration for ordinary shaped linear commands.",
    )
    parser.add_argument(
        "--max-angular-accel-radps2",
        type=float,
        default=0.60,
        help="Maximum acceleration for ordinary shaped angular commands.",
    )
    parser.add_argument("--min-obstacle-distance-m", type=float, default=0.20)
    parser.add_argument(
        "--omnidirectional-hard-stop-distance-m",
        type=float,
        default=0.12,
        help=(
            "Unconditional all-angle LiDAR stop used during certified "
            "directional blockage recovery. Normal translation retains "
            "--min-obstacle-distance-m in its commanded-motion sector."
        ),
    )
    parser.add_argument("--front-obstacle-slow-distance-m", type=float, default=0.38)
    parser.add_argument("--front-obstacle-sector-rad", type=float, default=0.6108652381980153)
    parser.add_argument("--thinning-min-spacing-m", type=float, default=0.15)
    parser.add_argument("--max-scan-age-sec", type=float, default=1.0)
    parser.add_argument("--max-odom-age-sec", type=float, default=1.0)
    parser.add_argument("--max-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--max-amcl-age-sec", type=float, default=2.0)
    parser.add_argument(
        "--max-future-timestamp-sec",
        type=float,
        default=0.25,
        help="Reject sensor or TF stamps farther than this into the future.",
    )
    parser.add_argument(
        "--certified-route-tube-radius-m",
        type=float,
        default=0.03,
        help=(
            "Maximum live base/pursuit deviation from the active certified "
            "polyline segment on physical stand approaches."
        ),
    )
    parser.add_argument(
        "--max-localization-tf-future-sec",
        type=float,
        default=1.1,
        help=(
            "AMCL map->odom forward-stamp allowance. Keep this at least as "
            "large as AMCL transform_tolerance; it does not relax sensor stamps."
        ),
    )
    parser.add_argument(
        "--certified-route-chord-sample-spacing-m",
        type=float,
        default=0.01,
        help="Sampling spacing for runtime pursuit-chord certificate checks.",
    )
    parser.add_argument(
        "--certified-corner-max-reacquire-attempts",
        type=int,
        default=2,
        help=(
            "Maximum bounded exact-vertex reacquisitions while a discovery "
            "corner remains inside the certified route tube."
        ),
    )
    parser.add_argument("--preflight-observation-window-sec", type=float, default=2.0)
    parser.add_argument(
        "--nomotion-update-service",
        default="/request_nomotion_update",
        help="Stationary AMCL refresh service used after preflight subscribers exist.",
    )
    parser.add_argument(
        "--nomotion-update-timeout-sec",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--runtime-nomotion-update-service",
        default="request_nomotion_update",
        help=(
            "Namespace-relative AMCL no-motion service used only for bounded "
            "runtime TF reacquisition. This is separate from the preflight "
            "refresh service."
        ),
    )
    parser.add_argument(
        "--runtime-nomotion-update-timeout-sec",
        type=float,
        default=2.0,
        help=(
            "Fail-closed runtime AMCL refresh budget in seconds; must be "
            "positive and no greater than 2.0."
        ),
    )
    parser.add_argument(
        "--stationary-amcl-sample-count",
        type=int,
        default=5,
        help="Forced no-motion AMCL samples required before physical motion.",
    )
    parser.add_argument(
        "--stationary-amcl-sample-interval-sec",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-stationary-amcl-position-spread-m",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--max-stationary-amcl-yaw-spread-rad",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--max-stationary-amcl-position-std-m",
        type=float,
        default=0.015,
        help=(
            "Maximum reported AMCL planar one-sigma uncertainty before "
            "physical motion; defaults to half the 0.03 m route tube."
        ),
    )
    parser.add_argument(
        "--max-stationary-amcl-yaw-std-rad",
        type=float,
        default=0.03,
        help="Maximum reported AMCL yaw one-sigma uncertainty before motion.",
    )
    parser.add_argument(
        "--skip-nomotion-update-before-preflight",
        action="store_true",
        help="Disable the automatic stationary AMCL refresh.",
    )
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
        "--viewpoint-sampling-target-timeout-sec",
        type=float,
        default=30.0,
        help=(
            "Simulation-only convergence budget for one material sampling "
            "target; newer targets reset this clock but not the total phase clock."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-goal-tolerance-m",
        type=float,
        default=0.01,
        help=(
            "Simulation-only position tolerance for axis acquisition and "
            "tangential camera samples; kept tighter than generic transit so "
            "angular viewpoint corrections are not consumed by position error."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
        type=float,
        default=INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
        help=(
            "Simulation-only bounded terminal-yaw latch radius. Once the "
            "strict position target is captured, leaving this radius stops "
            "the follower instead of resuming point-bearing pursuit."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-target-distance-m",
        type=float,
        default=0.33,
        help=(
            "Simulation-only nominal stand-center distance encoded by each "
            "intermediate target pose."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
        type=float,
        default=INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
        help=(
            "Simulation-only maximum pose-to-target drift retained during "
            "zero-linear terminal yaw; the stand-distance annulus is checked "
            "independently."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-heading-tolerance-rad",
        type=float,
        default=math.radians(5.0),
        help=(
            "Simulation-only terminal heading tolerance for axis acquisition "
            "and camera sampling; kept tight enough for the stand to satisfy "
            "the image-centering gate."
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
    parser.add_argument(
        "--linear-motion-floor-mps",
        type=float,
        default=DEFAULT_LINEAR_MOTION_FLOOR_MPS,
        help=(
            "Minimum clearance-scaled nonzero linear command considered capable "
            "of physical motion; smaller commands hold zero and enter stationary "
            "blockage confirmation on certified routes."
        ),
    )
    parser.add_argument(
        "--blockage-confirmation-timeout-sec",
        type=float,
        default=1.2,
        help="Bounded zero-hold window for persistent front-obstacle evidence.",
    )
    parser.add_argument(
        "--blockage-confirmation-min-samples",
        type=int,
        default=3,
        help="Minimum fresh, distinct front scans required while stopped.",
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
    parser.add_argument(
        "--allow-legacy-simulation-route",
        action="store_true",
        help=(
            "Explicitly allow route_kind=legacy_simulation_waypoint. The CSV "
            "must also declare simulation_only=true and --allow-sim-time is required."
        ),
    )
    parser.add_argument(
        "--allow-unbound-survey-simulation-route",
        action="store_true",
        help=(
            "Allow a static route_purpose=survey demonstration only when both "
            "the CSV and runtime are explicitly simulation-only. Logistics "
            "routes can never use this escape hatch."
        ),
    )
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
        "--controller-trace-jsonl",
        type=Path,
        default=None,
        help=(
            "Append-only controller evidence path. Bundled physical runs derive "
            "controller_trace.jsonl automatically."
        ),
    )
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=[],
        help="Namespace-qualified node identity allowed in preflight, e.g. /robot1/controller_server",
    )
    return parser


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


def _preflight_pose(
    raw: object,
    *,
    frame_id: str,
    child_frame_id: str,
    name: str,
) -> Pose2D:
    if not isinstance(raw, Mapping):
        raise ValueError(f"preflight did not provide {name}")
    if (
        raw.get("frame_id") != frame_id
        or raw.get("child_frame_id") != child_frame_id
    ):
        raise ValueError(f"preflight {name} frame identity mismatch")
    try:
        pose = Pose2D(
            float(raw["x_m"]),
            float(raw["y_m"]),
            float(raw["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"preflight {name} is malformed: {exc}") from exc
    if not all(
        math.isfinite(value)
        for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError(f"preflight {name} is non-finite")
    return pose


def _preflight_map_from_odom(
    preflight: RosPreflightResult,
    *,
    map_frame: str,
    odom_frame: str,
) -> tuple[PlanarTransform2D, float, float]:
    raw = preflight.map_from_odom
    if not isinstance(raw, Mapping):
        raise ValueError("preflight did not provide a direct map->odom transform")
    if (
        raw.get("target_frame") != map_frame
        or raw.get("source_frame") != odom_frame
    ):
        raise ValueError("preflight map->odom transform frame identity mismatch")
    try:
        transform = PlanarTransform2D(
            float(raw["x_m"]),
            float(raw["y_m"]),
            float(raw["yaw_rad"]),
        )
        stamp_sec = float(raw["stamp_sec"])
        capture_time_sec = float(raw["capture_time_sec"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"preflight map->odom transform is malformed: {exc}") from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in (
        stamp_sec,
        capture_time_sec,
    )):
        raise ValueError("preflight map->odom timestamps must be finite/non-negative")
    return transform, stamp_sec, capture_time_sec


def _preflight_stationary_map_from_odom_window(
    preflight: RosPreflightResult,
    *,
    map_frame: str,
    odom_frame: str,
) -> tuple[tuple[PlanarTransform2D, ...], tuple[dict[str, object], ...]]:
    """Validate ordered direct-TF samples paired with stationary AMCL."""

    raw_samples = preflight.stationary_map_from_odom_samples
    if not isinstance(raw_samples, list) or len(raw_samples) < 2:
        raise ValueError(
            "preflight did not provide at least two stationary direct "
            "map->odom transform samples"
        )
    transforms: list[PlanarTransform2D] = []
    provenance: list[dict[str, object]] = []
    previous_receipt_nanoseconds: int | None = None
    previous_stamp_nanoseconds: int | None = None
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"preflight stationary map->odom sample {index} is malformed"
            )
        if (
            raw.get("source") != "direct_dynamic_tf"
            or raw.get("target_frame") != map_frame
            or raw.get("source_frame") != odom_frame
            or raw.get("observed_target_frame") != map_frame
            or raw.get("observed_source_frame") != odom_frame
            or raw.get("amcl_sample_index") != index
        ):
            raise ValueError(
                "preflight stationary map->odom sample provenance or frame "
                f"identity mismatch at index {index}"
            )
        try:
            transform = PlanarTransform2D(
                float(raw["x_m"]),
                float(raw["y_m"]),
                float(raw["yaw_rad"]),
            )
            stamp_sec = float(raw["stamp_sec"])
            receipt_time_sec = float(raw["receipt_time_sec"])
            capture_time_sec = float(raw["capture_time_sec"])
            stamp_nanoseconds = raw["stamp_nanoseconds"]
            receipt_time_nanoseconds = raw["receipt_time_nanoseconds"]
            capture_time_nanoseconds = raw["capture_time_nanoseconds"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"preflight stationary map->odom sample {index} is malformed: "
                f"{exc}"
            ) from exc
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (stamp_sec, receipt_time_sec, capture_time_sec)
        ):
            raise ValueError(
                f"preflight stationary map->odom sample {index} has invalid "
                "timestamps"
            )
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in (
                stamp_nanoseconds,
                receipt_time_nanoseconds,
                capture_time_nanoseconds,
            )
        ):
            raise ValueError(
                f"preflight stationary map->odom sample {index} has invalid "
                "nanosecond timestamps"
            )
        for seconds, nanoseconds, name in (
            (stamp_sec, stamp_nanoseconds, "stamp"),
            (receipt_time_sec, receipt_time_nanoseconds, "receipt"),
            (capture_time_sec, capture_time_nanoseconds, "capture"),
        ):
            if not math.isclose(
                seconds,
                nanoseconds / 1_000_000_000.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    f"preflight stationary map->odom sample {index} {name} "
                    "second/nanosecond timestamps disagree"
                )
        if capture_time_nanoseconds < receipt_time_nanoseconds:
            raise ValueError(
                f"preflight stationary map->odom sample {index} was captured "
                "before receipt"
            )
        if (
            previous_receipt_nanoseconds is not None
            and receipt_time_nanoseconds <= previous_receipt_nanoseconds
        ):
            raise ValueError(
                "preflight stationary map->odom samples do not have strictly "
                "newer direct-TF receipts"
            )
        if (
            previous_stamp_nanoseconds is not None
            and stamp_nanoseconds <= previous_stamp_nanoseconds
        ):
            raise ValueError(
                "preflight stationary map->odom samples do not have strictly "
                "newer direct-TF stamps"
            )
        transforms.append(transform)
        provenance.append(
            {
                "sample_index": index,
                "amcl_sample_index": index,
                "source": "direct_dynamic_tf",
                "stamp_sec": stamp_sec,
                "stamp_nanoseconds": stamp_nanoseconds,
                "receipt_time_sec": receipt_time_sec,
                "receipt_time_nanoseconds": receipt_time_nanoseconds,
                "capture_time_sec": capture_time_sec,
                "capture_time_nanoseconds": capture_time_nanoseconds,
            }
        )
        previous_receipt_nanoseconds = receipt_time_nanoseconds
        previous_stamp_nanoseconds = stamp_nanoseconds
    return tuple(transforms), tuple(provenance)


def _conservative_preflight_covariance(
    preflight: RosPreflightResult,
) -> tuple[PlanarCovariance, float, dict[str, object]]:
    if not preflight.stationary_amcl_samples:
        raise ValueError("preflight has no accepted stationary AMCL samples")
    maximum_position_variance_m2 = 0.0
    maximum_yaw_variance_rad2 = 0.0
    sample_evidence = []
    for index, sample in enumerate(preflight.stationary_amcl_samples):
        raw_covariance = sample.get("covariance")
        if not isinstance(raw_covariance, list) or len(raw_covariance) != 36:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is incomplete"
            )
        try:
            values = [float(value) for value in raw_covariance]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is malformed"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"preflight AMCL sample {index} covariance is non-finite"
            )
        xx_m2 = values[0]
        xy_m2 = values[1]
        yx_m2 = values[6]
        yy_m2 = values[7]
        yaw_variance_rad2 = values[35]
        symmetry_tolerance = max(
            1.0e-12,
            1.0e-6 * max(abs(xy_m2), abs(yx_m2)),
        )
        if abs(xy_m2 - yx_m2) > symmetry_tolerance:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is asymmetric"
            )
        covariance = PlanarCovariance(
            xx_m2,
            0.5 * (xy_m2 + yx_m2),
            yy_m2,
        )
        largest_position_variance_m2 = 0.5 * (
            covariance.xx_m2
            + covariance.yy_m2
            + math.hypot(
                covariance.xx_m2 - covariance.yy_m2,
                2.0 * covariance.xy_m2,
            )
        )
        if yaw_variance_rad2 < 0.0:
            raise ValueError(
                f"preflight AMCL sample {index} yaw covariance is negative"
            )
        maximum_position_variance_m2 = max(
            maximum_position_variance_m2,
            largest_position_variance_m2,
        )
        maximum_yaw_variance_rad2 = max(
            maximum_yaw_variance_rad2,
            yaw_variance_rad2,
        )
        sample_evidence.append(
            {
                "sample_index": index,
                "xx_m2": covariance.xx_m2,
                "xy_m2": covariance.xy_m2,
                "yy_m2": covariance.yy_m2,
                "yaw_variance_rad2": yaw_variance_rad2,
                "largest_position_variance_m2": (
                    largest_position_variance_m2
                ),
            }
        )
    # An isotropic envelope at the largest observed eigenvalue dominates each
    # accepted sample in every route-normal direction. This is conservative;
    # it does not turn a five-sample spread into an accuracy claim.
    covariance_envelope = PlanarCovariance(
        maximum_position_variance_m2,
        0.0,
        maximum_position_variance_m2,
    )
    return (
        covariance_envelope,
        math.sqrt(maximum_yaw_variance_rad2),
        {
            "envelope_kind": "isotropic_maximum_eigenvalue",
            "sample_count": len(sample_evidence),
            "samples": sample_evidence,
        },
    )


def _angle_distance_rad(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _covariance_bounded_continuity_limits(
    covariance: PlanarCovariance,
    *,
    heading_sigma_rad: float,
    sigma_multiplier: float,
    translation_hard_cap_m: float,
    yaw_hard_cap_rad: float,
) -> tuple[float, float]:
    """Reuse, rather than duplicate, the route's localization allowance."""

    allocated_translation_m = sigma_multiplier * math.sqrt(
        covariance.xx_m2
    )
    allocated_yaw_rad = sigma_multiplier * heading_sigma_rad
    return (
        min(translation_hard_cap_m, allocated_translation_m),
        min(yaw_hard_cap_rad, allocated_yaw_rad),
    )


def _admit_stationary_map_from_odom_window(
    preflight: RosPreflightResult,
    *,
    map_frame: str,
    odom_frame: str,
    final_map_from_odom: PlanarTransform2D,
    final_stamp_sec: float,
    final_capture_time_sec: float,
    max_translation_drift_m: float,
    max_yaw_drift_rad: float,
) -> tuple[PlanarTransform2D, dict[str, object]]:
    """Bind the final certificate transform to a stable direct-TF window."""

    samples, provenance = _preflight_stationary_map_from_odom_window(
        preflight,
        map_frame=map_frame,
        odom_frame=odom_frame,
    )
    final_provenance = provenance[-1]
    if (
        final_capture_time_sec
        < float(final_provenance["capture_time_sec"])
        or final_stamp_sec < float(final_provenance["stamp_sec"])
    ):
        raise ValueError(
            "preflight final map->odom transform predates its stationary "
            "sample window"
        )
    stability = evaluate_map_odom_stationary_stability(
        (*samples, final_map_from_odom),
        max_translation_drift_m=max_translation_drift_m,
        max_yaw_drift_rad=max_yaw_drift_rad,
    )
    if not stability.accepted:
        raise ValueError(
            "preflight stationary map->odom transform window rejected: "
            f"{stability.reason}"
        )
    admitted = stability.frozen_map_from_odom
    if admitted is None:
        raise ValueError(
            "preflight stationary map->odom admission did not freeze a transform"
        )
    if admitted != final_map_from_odom:
        raise ValueError(
            "preflight stationary map->odom admission changed the final transform"
        )
    evidence = stability.to_evidence()
    evidence["sample_provenance"] = [
        *provenance,
        {
            "sample_index": len(samples),
            "source": "final_preflight_tf_lookup",
            "stamp_sec": final_stamp_sec,
            "capture_time_sec": final_capture_time_sec,
        },
    ]
    return admitted, evidence


def _build_odom_execution_admission(
    *,
    args,
    resolved,
    leg: SelectedRouteLeg,
    preflight: RosPreflightResult,
    diagnostics_snapshot: DiagnosticsSnapshot,
) -> tuple[
    tuple[Pose2D, ...],
    OdomExecutionContext,
    dict[str, object],
    "_OdomRouteUncertaintyGate",
]:
    """Freeze and seal one map-route projection for odom-only control."""

    map_route = poses_from_waypoints(leg.executable_waypoints)
    map_pose = _preflight_pose(
        preflight.route_pose,
        frame_id=resolved.map_frame,
        child_frame_id=resolved.base_frame,
        name="map-frame base pose",
    )
    odom_pose = _preflight_pose(
        preflight.odom_pose,
        frame_id=resolved.odom_frame,
        child_frame_id=resolved.base_frame,
        name="odom-frame base pose",
    )
    map_from_odom, transform_stamp_sec, transform_capture_time_sec = (
        _preflight_map_from_odom(
            preflight,
            map_frame=resolved.map_frame,
            odom_frame=resolved.odom_frame,
        )
    )
    composed_map_pose = odom_pose_to_map(odom_pose, map_from_odom)
    composition_position_error_m = math.hypot(
        composed_map_pose.x_m - map_pose.x_m,
        composed_map_pose.y_m - map_pose.y_m,
    )
    composition_yaw_error_rad = _angle_distance_rad(
        composed_map_pose.yaw_rad,
        map_pose.yaw_rad,
    )
    if composition_position_error_m > args.certified_route_tube_radius_m:
        raise ValueError(
            "preflight map/odom/transform composition exceeds the certified "
            f"route tube: {composition_position_error_m:.6f} m"
        )
    if composition_yaw_error_rad > args.max_stationary_amcl_yaw_spread_rad:
        raise ValueError(
            "preflight map/odom/transform yaw composition is inconsistent: "
            f"{composition_yaw_error_rad:.6f} rad"
        )
    odom_route = transform_map_route_to_odom(map_route, map_from_odom)
    route_yaw_lever_arm_m = max(
        math.hypot(
            pose.x_m - map_route[0].x_m,
            pose.y_m - map_route[0].y_m,
        )
        for pose in map_route
    ) + args.uncertainty_robot_radius_m
    startup_decision = certified_static_startup_decision(
        odom_pose,
        odom_route,
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
    )
    if not startup_decision.ok:
        raise ValueError(
            "odom pose is outside the transformed certified startup segment: "
            + startup_decision.route_check.reason
        )

    covariance, heading_sigma_rad, covariance_evidence = (
        _conservative_preflight_covariance(preflight)
    )
    allocated_translation_drift_m = (
        args.uncertainty_sigma_multiplier
        * math.sqrt(covariance.xx_m2)
    )
    allocated_yaw_drift_rad = (
        args.uncertainty_sigma_multiplier * heading_sigma_rad
    )
    (
        continuity_translation_limit_m,
        continuity_yaw_limit_rad,
    ) = _covariance_bounded_continuity_limits(
        covariance,
        heading_sigma_rad=heading_sigma_rad,
        sigma_multiplier=args.uncertainty_sigma_multiplier,
        translation_hard_cap_m=args.max_map_odom_translation_drift_m,
        yaw_hard_cap_rad=args.max_map_odom_yaw_drift_rad,
    )
    map_from_odom, stationary_stability_evidence = (
        _admit_stationary_map_from_odom_window(
            preflight,
            map_frame=resolved.map_frame,
            odom_frame=resolved.odom_frame,
            final_map_from_odom=map_from_odom,
            final_stamp_sec=transform_stamp_sec,
            final_capture_time_sec=transform_capture_time_sec,
            max_translation_drift_m=continuity_translation_limit_m,
            max_yaw_drift_rad=continuity_yaw_limit_rad,
        )
    )
    arena_bounds = validate_arena_boundary_evidence(
        diagnostics_snapshot.metadata
    )
    base_costmap = Costmap.from_occupancy_grid(
        load_occupancy_grid(args.uncertainty_map_yaml)
    ).with_arena_bounds(arena_bounds)
    admission_config = RouteUncertaintyAdmissionConfig(
        robot_radius_m=args.uncertainty_robot_radius_m,
        collision_margin_m=args.uncertainty_collision_margin_m,
        fixed_odom_tracking_bound_m=args.certified_route_tube_radius_m,
        empirical_odom_drift_bound_m=(
            args.uncertainty_odom_drift_bound_m
        ),
        braking_latency_distance_m=(
            args.uncertainty_braking_latency_distance_m
        ),
        localization_sigma_multiplier=args.uncertainty_sigma_multiplier,
        # The same covariance envelope is used twice: once as reserved route
        # clearance here and once as the maximum live map<-odom correction the
        # monitor may accept. It is not charged a second time.
        heading_sigma_rad=heading_sigma_rad,
        heading_lever_arm_m=(
            args.uncertainty_robot_radius_m
            if args.uncertainty_heading_lever_arm_m is None
            else args.uncertainty_heading_lever_arm_m
        ),
        sampling_spacing_m=args.uncertainty_clearance_sample_spacing_m,
        heading_reference_x_m=map_route[0].x_m,
        heading_reference_y_m=map_route[0].y_m,
    )
    admission = evaluate_route_uncertainty_admission(
        base_costmap,
        map_route,
        covariance,
        admission_config,
    )
    if not admission.decision.accepted:
        limiting = admission.decision.limiting_segment_id or "unknown"
        margin = admission.decision.remaining_margin_m
        margin_text = "unknown" if margin is None else f"{margin:.6f} m"
        raise ValueError(
            "route uncertainty budget exhausted: "
            f"limiting_segment={limiting} remaining_margin={margin_text}"
        )

    map_certificate, map_certificate_sha256 = (
        _resolved_map_execution_certificate(args, diagnostics_snapshot)
    )
    branch_evidence = {
        "schema_version": 1,
        "proof_id": args.localization_branch_proof_id,
        "method": "operator_known_start_or_asymmetric_landmark_attestation",
        "map_frame": resolved.map_frame,
        "map_bundle_sha256": str(
            diagnostics_snapshot.metadata.get("map_bundle_sha256", "")
        ),
        "source_map_execution_certificate_sha256": map_certificate_sha256,
        "claim_boundary": (
            "operator branch selection; covariance alone does not resolve "
            "symmetric-map aliases"
        ),
    }
    ambiguity_evidence_sha256 = payload_sha256(branch_evidence)
    budget_payload = {
        "schema_version": 1,
        "source": "route_uncertainty_admission",
        "admission": admission.to_evidence_dict(),
        "covariance_envelope": covariance_evidence,
        "runtime_map_odom_continuity_allocation": {
            "position_covariance_allocation_m": (
                allocated_translation_drift_m
            ),
            "yaw_covariance_allocation_rad": allocated_yaw_drift_rad,
            "translation_hard_cap_m": (
                args.max_map_odom_translation_drift_m
            ),
            "yaw_hard_cap_rad": args.max_map_odom_yaw_drift_rad,
            "effective_translation_limit_m": (
                continuity_translation_limit_m
            ),
            "effective_yaw_limit_rad": continuity_yaw_limit_rad,
            "route_yaw_lever_arm_m": route_yaw_lever_arm_m,
            "threshold_contract": (
                "live correction must remain within the same covariance "
                "allowance already reserved in route clearance"
            ),
        },
        "stationary_map_from_odom_stability": (
            stationary_stability_evidence
        ),
        "localization_branch_evidence": branch_evidence,
        "preflight_composition": {
            "position_error_m": composition_position_error_m,
            "yaw_error_rad": composition_yaw_error_rad,
        },
    }
    uncertainty_budget_sha256 = write_content_hashed_json(
        args.uncertainty_budget_json,
        budget_payload,
        hash_field="route_uncertainty_artifact_sha256",
    )

    odom_certificate = OdomExecutionCertificate(
        source_map_route_sha256=pose_route_sha256(map_route),
        source_map_execution_certificate_sha256=map_certificate_sha256,
        transformed_odom_route_sha256=pose_route_sha256(odom_route),
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        map_from_odom=map_from_odom,
        transform_stamp_sec=transform_stamp_sec,
        transform_capture_time_sec=transform_capture_time_sec,
        waypoint_count=len(map_route),
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
        command_owner=_runtime_command_owner(resolved.namespace),
        uncertainty_budget_sha256=uncertainty_budget_sha256,
        ambiguity_evidence_sha256=ambiguity_evidence_sha256,
    )
    odom_certificate_sha256 = write_odom_execution_certificate(
        args.odom_execution_certificate_json,
        odom_certificate,
    )
    validate_odom_execution_identity(
        odom_certificate,
        source_map_route=map_route,
        source_map_execution_certificate_sha256=map_certificate_sha256,
        transformed_odom_route=odom_route,
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
        command_owner=_runtime_command_owner(resolved.namespace),
        map_from_odom=map_from_odom,
        transform_stamp_sec=transform_stamp_sec,
        transform_capture_time_sec=transform_capture_time_sec,
        uncertainty_budget_sha256=uncertainty_budget_sha256,
        ambiguity_evidence_sha256=ambiguity_evidence_sha256,
    )
    context = OdomExecutionContext(
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        frozen_map_from_odom=map_from_odom,
        certificate_sha256=odom_certificate_sha256,
        max_map_from_odom_translation_drift_m=(
            continuity_translation_limit_m
        ),
        max_map_from_odom_yaw_drift_rad=continuity_yaw_limit_rad,
    )
    replacement_route_gate = _OdomRouteUncertaintyGate(
        costmap=base_costmap,
        covariance=covariance,
        config=admission_config,
        evidence_root=(
            None
            if args.coverage_transient_replan_session_root is None
            else Path(args.coverage_transient_replan_session_root)
            / "odom_execution_replans"
        ),
    )
    return (
        odom_route,
        context,
        {
            "odom_execution_certificate_sha256": odom_certificate_sha256,
            "odom_execution_certificate_json": str(
                args.odom_execution_certificate_json
            ),
            "uncertainty_budget_sha256": uncertainty_budget_sha256,
            "uncertainty_budget_json": str(args.uncertainty_budget_json),
            "ambiguity_evidence_sha256": ambiguity_evidence_sha256,
            "source_map_execution_certificate_sha256": (
                map_certificate_sha256
            ),
            "source_map_route_sha256": pose_route_sha256(map_route),
            "transformed_odom_route_sha256": pose_route_sha256(odom_route),
            "minimum_remaining_margin_m": (
                admission.decision.remaining_margin_m
            ),
            "map_from_odom": {
                "x_m": map_from_odom.x_m,
                "y_m": map_from_odom.y_m,
                "yaw_rad": map_from_odom.yaw_rad,
                "stamp_sec": transform_stamp_sec,
                "capture_time_sec": transform_capture_time_sec,
            },
            "map_execution_certificate_route_kind": map_certificate.route_kind,
        },
        replacement_route_gate,
    )


class _OdomRouteUncertaintyGate:
    """Re-admit each replacement map route before odom transformation."""

    def __init__(
        self,
        *,
        costmap: Costmap,
        covariance: PlanarCovariance,
        config: RouteUncertaintyAdmissionConfig,
        evidence_root: Path | None,
    ) -> None:
        self._costmap = costmap
        self._covariance = covariance
        self._config = config
        self._evidence_root = evidence_root

    def adapt(
        self,
        update: RouteUpdate,
        context: OdomExecutionContext,
    ) -> RouteUpdate:
        if update.kind is not RouteUpdateKind.ADOPT:
            return update
        admission = evaluate_route_uncertainty_admission(
            self._costmap,
            update.waypoints,
            self._covariance,
            self._config,
        )
        evidence_sha256 = route_uncertainty_admission_evidence_sha256(
            admission
        )
        evidence_path = None
        if self._evidence_root is not None:
            route_revision = (
                "unknown"
                if update.route_revision is None
                else f"{update.route_revision:06d}"
            )
            route_hash_prefix = str(update.route_hash or "unhashed")[:16]
            evidence_path = (
                self._evidence_root
                / (
                    f"route_revision_{route_revision}_"
                    f"{route_hash_prefix}_uncertainty.json"
                )
            )
            stored_hash = write_content_hashed_json(
                evidence_path,
                admission.to_evidence_dict(),
                hash_field="route_uncertainty_admission_sha256",
            )
            if stored_hash != evidence_sha256:
                raise ValueError(
                    "replacement route uncertainty evidence hash mismatch"
                )
        evidence_fields = {
            "replacement_route_uncertainty_admission_sha256": (
                evidence_sha256
            ),
            "replacement_route_uncertainty_admission_json": (
                "" if evidence_path is None else str(evidence_path)
            ),
            "replacement_route_uncertainty_accepted": (
                admission.decision.accepted
            ),
            "replacement_route_uncertainty_remaining_margin_m": (
                admission.decision.remaining_margin_m
            ),
            "replacement_route_uncertainty_limiting_segment_id": (
                admission.decision.limiting_segment_id
            ),
        }
        if not admission.decision.accepted:
            return RouteUpdate(
                kind=RouteUpdateKind.REJECT,
                reason="replacement route uncertainty budget exhausted",
                route_revision=update.route_revision,
                target_revision=update.target_revision,
                route_hash=update.route_hash,
                requires_zero_cycle=True,
                event_name="dynamic_route_rejected",
                event_fields={
                    **dict(update.event_fields),
                    **evidence_fields,
                    "fail_closed": True,
                },
            )
        adapted = adapt_map_route_update_to_odom(update, context)
        return RouteUpdate(
            kind=adapted.kind,
            waypoints=adapted.waypoints,
            target_index=adapted.target_index,
            reason=adapted.reason,
            route_revision=adapted.route_revision,
            target_revision=adapted.target_revision,
            route_hash=adapted.route_hash,
            requires_zero_cycle=True,
            event_name=adapted.event_name,
            event_fields={
                **dict(adapted.event_fields),
                **evidence_fields,
            },
        )


class _OdomBlockageRecoveryAdapter:
    """Keep the planner map-native and adapt only its sealed handoff."""

    def __init__(
        self,
        provider,
        context: OdomExecutionContext,
        uncertainty_gate: _OdomRouteUncertaintyGate,
    ) -> None:
        self._provider = provider
        self._context = context
        self._uncertainty_gate = uncertainty_gate

    def __call__(
        self,
        map_pose: Pose2D,
        stop_reason: str,
        stop_details: Mapping[str, object],
    ) -> RouteUpdate | None:
        update = self._provider(map_pose, stop_reason, stop_details)
        if update is None:
            return None
        return self._uncertainty_gate.adapt(update, self._context)


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


def _validated_runtime_localization_motion_permit(
    args,
    resolved,
    *,
    route_csv_path: Path,
    diagnostics_path: Path,
) -> RuntimeLocalizationMotionPermit | None:
    """Return the exact recovery permit or preserve normal interactive motion."""

    paths = (
        args.mission_motion_authorization_json,
        args.runtime_localization_motion_permit_json,
    )
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError(
            "mission motion authorization and runtime localization permit "
            "must be supplied together"
        )
    if args.dry_run:
        raise ValueError("runtime localization motion permit is live-run only")
    if args.allow_sim_time:
        raise ValueError(
            "runtime localization motion permit is physical-runtime only"
        )
    if args.execution_pose_frame != "odom":
        raise ValueError(
            "runtime localization motion permit requires odom execution"
        )
    if args.route_certificate_json is None:
        raise ValueError(
            "runtime localization motion permit requires a map route certificate"
        )
    if args.coverage_transient_replan_leg_index is None:
        raise ValueError(
            "runtime localization motion permit requires a coverage leg index"
        )
    target_viewpoint_id = str(
        args.coverage_transient_replan_target_viewpoint_id
    ).strip()
    semantic_map_id = str(
        args.coverage_transient_replan_semantic_map_id
    ).strip()
    session_id = str(args.mission_session_id).strip()
    if not target_viewpoint_id or not semantic_map_id or not session_id:
        raise ValueError(
            "runtime localization motion permit requires session, semantic map, "
            "and target identities"
        )
    return validate_runtime_localization_motion_permit_for_execution(
        args.runtime_localization_motion_permit_json,
        master_authorization_path=args.mission_motion_authorization_json,
        session_id=session_id,
        run_id=args.run_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        cmd_vel_topic=resolved.cmd_vel_topic,
        semantic_map_id=semantic_map_id,
        target_viewpoint_id=target_viewpoint_id,
        leg_index=args.coverage_transient_replan_leg_index,
        localization_branch_proof_id=args.localization_branch_proof_id,
        route_csv_path=route_csv_path,
        diagnostics_path=diagnostics_path,
        map_route_certificate_path=args.route_certificate_json,
    )


def _validated_startup_reseal_motion_permit(
    args,
    resolved,
    *,
    route_csv_path: Path,
    diagnostics_path: Path,
) -> StartupResealMotionPermit | None:
    """Return one exact startup-reseal permit or preserve normal prompting."""

    fields = (
        args.startup_reseal_motion_authorization_json,
        args.startup_reseal_motion_permit_json,
        str(args.startup_reseal_target_viewpoint_id).strip() or None,
        str(args.startup_reseal_semantic_map_id).strip() or None,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise ValueError(
            "startup-reseal motion authorization arguments must be supplied together"
        )
    if any(
        value is not None
        for value in (
            args.mission_motion_authorization_json,
            args.runtime_localization_motion_permit_json,
            args.mission_leg_motion_authorization_json,
            args.mission_leg_motion_permit_json,
        )
    ):
        raise ValueError(
            "startup-reseal, routine-leg, and runtime-localization permits "
            "are mutually exclusive"
        )
    if args.dry_run:
        raise ValueError("startup-reseal motion permit is live-run only")
    if args.allow_sim_time:
        raise ValueError("startup-reseal motion permit is physical-runtime only")
    if args.execution_pose_frame != "odom":
        raise ValueError("startup-reseal motion permit requires odom execution")
    if args.route_certificate_json is None:
        raise ValueError(
            "startup-reseal motion permit requires a map route certificate"
        )
    if args.coverage_transient_replan_leg_index is None:
        raise ValueError(
            "startup-reseal motion permit requires a coverage leg index"
        )
    session_id = str(args.mission_session_id).strip()
    if not session_id:
        raise ValueError("startup-reseal motion permit requires mission_session_id")
    return validate_startup_reseal_motion_permit_for_execution(
        args.startup_reseal_motion_permit_json,
        master_authorization_path=(
            args.startup_reseal_motion_authorization_json
        ),
        run_id=args.run_id,
        session_id=session_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        cmd_vel_topic=resolved.cmd_vel_topic,
        semantic_map_id=str(args.startup_reseal_semantic_map_id).strip(),
        target_viewpoint_id=str(
            args.startup_reseal_target_viewpoint_id
        ).strip(),
        leg_index=args.coverage_transient_replan_leg_index,
        localization_branch_proof_id=args.localization_branch_proof_id,
        route_csv_path=route_csv_path,
        diagnostics_path=diagnostics_path,
        map_route_certificate_path=args.route_certificate_json,
    )


def _validated_coverage_replan_resume_state(
    args,
    *,
    diagnostics_path: Path,
) -> TransientOverlayResumeState | None:
    """Integrity-load one inherited overlay without authorizing motion."""

    state_path = args.coverage_transient_replan_resume_state_json
    if state_path is None:
        return None
    if not args.coverage_transient_replan_enabled:
        raise ValueError(
            "transient overlay resume state requires coverage replanning"
        )
    if args.coverage_plan is None:
        raise ValueError(
            "transient overlay resume state requires the coverage plan"
        )
    survey_plan_path = (
        Path(args.coverage_transient_replan_survey_root)
        / "coverage_plan.json"
    )
    try:
        supplied_plan_path = Path(args.coverage_plan).resolve(strict=True)
        expected_plan_path = survey_plan_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("coverage resume plan is unavailable") from exc
    if supplied_plan_path != expected_plan_path:
        raise ValueError(
            "coverage resume state and replanner use different plans"
        )
    plan = load_coverage_survey_plan(supplied_plan_path)
    state = validate_transient_overlay_resume_state_diagnostics_binding(
        diagnostics_path,
        resume_state_path=state_path,
        plan=plan,
        expected_coverage_leg_index=(
            args.coverage_transient_replan_leg_index
        ),
        expected_target_viewpoint_id=(
            args.coverage_transient_replan_target_viewpoint_id
        ),
        expected_max_replans=args.coverage_transient_replan_max_count,
    )
    if args.run_id in state.source_run_ids:
        raise ValueError(
            "coverage resume state cannot be replayed by a source child run"
        )
    return state


def _validated_mission_leg_motion_permit(
    args,
    resolved,
    *,
    route_csv_path: Path,
    diagnostics_path: Path,
) -> MissionLegMotionPermit | None:
    """Return one exact routine-leg permit or preserve interactive motion."""

    fields = (
        args.mission_leg_motion_authorization_json,
        args.mission_leg_motion_permit_json,
        args.mission_leg_kind,
        args.mission_leg_index,
        str(args.mission_leg_target_id).strip() or None,
        str(args.mission_leg_semantic_map_id).strip() or None,
        args.mission_leg_dry_preflight_json,
        args.mission_leg_dry_odom_certificate_json,
        args.mission_leg_dry_uncertainty_budget_json,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise ValueError(
            "mission-leg motion authorization arguments must be supplied together"
        )
    if (
        args.mission_motion_authorization_json is not None
        or args.runtime_localization_motion_permit_json is not None
    ):
        raise ValueError(
            "routine mission-leg and runtime-localization permits are "
            "mutually exclusive"
        )
    if args.dry_run:
        raise ValueError("mission-leg motion permit is live-run only")
    if args.allow_sim_time:
        raise ValueError("mission-leg motion permit is physical-runtime only")
    if args.execution_pose_frame != "odom":
        raise ValueError("mission-leg motion permit requires odom execution")
    if args.route_certificate_json is None:
        raise ValueError(
            "mission-leg motion permit requires a map route certificate"
        )
    session_id = str(args.mission_session_id).strip()
    if not session_id:
        raise ValueError("mission-leg motion permit requires mission_session_id")
    assert args.mission_leg_kind is not None
    assert args.mission_leg_index is not None
    return validate_mission_leg_motion_permit_for_execution(
        args.mission_leg_motion_permit_json,
        master_authorization_path=(
            args.mission_leg_motion_authorization_json
        ),
        session_id=session_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        cmd_vel_topic=resolved.cmd_vel_topic,
        semantic_map_id=str(args.mission_leg_semantic_map_id).strip(),
        localization_branch_proof_id=args.localization_branch_proof_id,
        run_id=args.run_id,
        mission_leg_kind=args.mission_leg_kind,
        mission_leg_index=args.mission_leg_index,
        target_id=str(args.mission_leg_target_id).strip(),
        route_csv_path=route_csv_path,
        diagnostics_path=diagnostics_path,
        map_route_certificate_path=args.route_certificate_json,
        dry_preflight_path=args.mission_leg_dry_preflight_json,
        dry_odom_certificate_path=(
            args.mission_leg_dry_odom_certificate_json
        ),
        dry_uncertainty_budget_path=(
            args.mission_leg_dry_uncertainty_budget_json
        ),
    )


def _record_motion_authorization_rejection(
    *,
    args,
    resolved,
    leg,
    event_logger,
    failure: object,
) -> int:
    """Persist one no-motion authorization failure with terminal evidence."""

    stop_reason = f"motion authorization rejected: {failure}"
    stop_details = {
        "reason": stop_reason,
        "fault_code": "motion_authorization_rejected",
        "source": "motion_authorization",
        "motion_published": False,
        "fail_closed": True,
    }
    result = FollowerResult(
        "preflight_failed",
        stop_reason,
        0.0,
        0.0,
        False,
        stop_details,
    )
    _append_result(args, resolved, leg, preflight_ok=False, result=result)
    emit_event(
        event_logger,
        "motion_authorization_rejected",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        status=result.status,
        stop_reason=stop_reason,
        motion_published=False,
        stop_details=stop_details,
    )
    emit_event(
        event_logger,
        "preflight_failed",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        status=result.status,
        failures=[stop_reason],
        observations=[],
        runtime_config=resolved.as_log_dict(),
        motion_published=False,
    )
    emit_event(
        event_logger,
        "run_finished",
        run_id=args.run_id,
        final_status=result.status,
        stop_reason=stop_reason,
        results_csv=str(args.results_csv),
        semantic_log_path=str(args.semantic_log),
        preflight_json_path=str(args.preflight_json or ""),
    )
    return 1


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
    recovery_fields = (
        args.coverage_transient_replan_survey_root,
        args.coverage_transient_replan_session_root,
        args.coverage_transient_replan_map,
        args.coverage_transient_replan_robot_radius_m,
        args.coverage_transient_replan_leg_index,
    )
    args.coverage_transient_replan_enabled = any(
        value is not None for value in recovery_fields
    ) or bool(
        args.coverage_transient_replan_semantic_map_id
        or args.coverage_transient_replan_target_viewpoint_id
        or args.coverage_transient_replan_max_count
        or args.coverage_transient_replan_resume_state_json is not None
    )
    if args.coverage_transient_replan_enabled and (
        any(value is None for value in recovery_fields)
        or not args.coverage_transient_replan_semantic_map_id
        or not args.coverage_transient_replan_target_viewpoint_id
        or args.coverage_transient_replan_max_count <= 0
    ):
        parser.error(
            "physical coverage transient replanning requires survey root, "
            "session root, map, semantic map ID, target viewpoint ID, positive "
            "robot radius, and positive max count"
        )
    args.localization_branch_proof_id = str(
        args.localization_branch_proof_id
    ).strip()
    odom_execution_enabled = args.execution_pose_frame == "odom"
    if odom_execution_enabled:
        missing = [
            flag
            for flag, value in (
                (
                    "--odom-execution-certificate-json",
                    args.odom_execution_certificate_json,
                ),
                ("--uncertainty-budget-json", args.uncertainty_budget_json),
                ("--uncertainty-map-yaml", args.uncertainty_map_yaml),
                (
                    "--localization-branch-proof-id",
                    args.localization_branch_proof_id,
                ),
                (
                    "--uncertainty-robot-radius-m",
                    args.uncertainty_robot_radius_m,
                ),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(
                "odom execution requires " + ", ".join(missing)
            )
        if args.localization_source != "amcl":
            parser.error(
                "odom execution requires AMCL as the global consistency monitor"
            )
        if args.map_frame == args.odom_frame:
            parser.error("odom execution requires distinct map and odom frames")
        if args.dynamic_route_refresh_sec > 0.0:
            parser.error(
                "odom execution does not admit simulation route hot-reload"
            )
        if args.allow_simulation_odom_after_stale_tf:
            parser.error(
                "odom execution may not enable the simulation stale-TF fallback"
            )
        uncertainty_values = (
            (
                "--uncertainty-robot-radius-m",
                args.uncertainty_robot_radius_m,
                True,
            ),
            (
                "--uncertainty-collision-margin-m",
                args.uncertainty_collision_margin_m,
                False,
            ),
            (
                "--uncertainty-odom-drift-bound-m",
                args.uncertainty_odom_drift_bound_m,
                False,
            ),
            (
                "--uncertainty-braking-latency-distance-m",
                args.uncertainty_braking_latency_distance_m,
                False,
            ),
            (
                "--uncertainty-sigma-multiplier",
                args.uncertainty_sigma_multiplier,
                True,
            ),
            (
                "--uncertainty-clearance-sample-spacing-m",
                args.uncertainty_clearance_sample_spacing_m,
                True,
            ),
            (
                "--max-map-odom-yaw-drift-rad",
                args.max_map_odom_yaw_drift_rad,
                True,
            ),
            (
                "--max-map-odom-translation-drift-m",
                args.max_map_odom_translation_drift_m,
                True,
            ),
        )
        for flag, value, strictly_positive in uncertainty_values:
            if (
                value is None
                or not math.isfinite(value)
                or (value <= 0.0 if strictly_positive else value < 0.0)
            ):
                qualifier = "positive" if strictly_positive else "non-negative"
                parser.error(f"{flag} must be finite and {qualifier}")
        if args.uncertainty_heading_lever_arm_m is not None and (
            not math.isfinite(args.uncertainty_heading_lever_arm_m)
            or args.uncertainty_heading_lever_arm_m <= 0.0
        ):
            parser.error(
                "--uncertainty-heading-lever-arm-m must be finite and positive"
            )
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
        not math.isfinite(args.viewpoint_sampling_target_timeout_sec)
        or args.viewpoint_sampling_target_timeout_sec <= 0.0
    ):
        parser.error("--viewpoint-sampling-target-timeout-sec must be positive")
    if (
        not math.isfinite(args.physical_waypoint_tolerance_m)
        or args.physical_waypoint_tolerance_m <= 0.0
    ):
        parser.error("--physical-waypoint-tolerance-m must be positive")
    if (
        not math.isfinite(args.physical_goal_tolerance_m)
        or args.physical_goal_tolerance_m <= 0.0
    ):
        parser.error("--physical-goal-tolerance-m must be positive")
    if (
        not math.isfinite(args.max_future_timestamp_sec)
        or args.max_future_timestamp_sec < 0.0
    ):
        parser.error("--max-future-timestamp-sec must be non-negative")
    if (
        not math.isfinite(args.max_localization_tf_future_sec)
        or args.max_localization_tf_future_sec < 0.0
    ):
        parser.error("--max-localization-tf-future-sec must be non-negative")
    if (
        not math.isfinite(args.nomotion_update_timeout_sec)
        or args.nomotion_update_timeout_sec <= 0.0
    ):
        parser.error("--nomotion-update-timeout-sec must be positive")
    args.runtime_nomotion_update_service = str(
        args.runtime_nomotion_update_service
    ).strip()
    if not args.runtime_nomotion_update_service:
        parser.error("--runtime-nomotion-update-service must not be empty")
    if (
        not math.isfinite(args.runtime_nomotion_update_timeout_sec)
        or args.runtime_nomotion_update_timeout_sec <= 0.0
        or args.runtime_nomotion_update_timeout_sec > 2.0
    ):
        parser.error(
            "--runtime-nomotion-update-timeout-sec must be finite and in (0, 2.0]"
        )
    if args.stationary_amcl_sample_count < 2:
        parser.error("--stationary-amcl-sample-count must be at least 2")
    if (
        args.skip_nomotion_update_before_preflight
        and not args.allow_sim_time
        and args.localization_source == "amcl"
    ):
        parser.error(
            "real AMCL runs may not skip the stationary localization gate"
        )
    for flag, value in (
        (
            "--stationary-amcl-sample-interval-sec",
            args.stationary_amcl_sample_interval_sec,
        ),
        (
            "--max-stationary-amcl-position-spread-m",
            args.max_stationary_amcl_position_spread_m,
        ),
        (
            "--max-stationary-amcl-yaw-spread-rad",
            args.max_stationary_amcl_yaw_spread_rad,
        ),
        (
            "--max-stationary-amcl-position-std-m",
            args.max_stationary_amcl_position_std_m,
        ),
        (
            "--max-stationary-amcl-yaw-std-rad",
            args.max_stationary_amcl_yaw_std_rad,
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{flag} must be positive")
    if (
        not math.isfinite(args.certified_route_tube_radius_m)
        or args.certified_route_tube_radius_m <= 0.0
    ):
        parser.error("--certified-route-tube-radius-m must be positive")
    localization_position_limit_m = 0.5 * args.certified_route_tube_radius_m
    localization_tube_limits = [
        (
            "--max-stationary-amcl-position-spread-m",
            args.max_stationary_amcl_position_spread_m,
        )
    ]
    if not odom_execution_enabled:
        localization_tube_limits.append(
            (
                "--max-stationary-amcl-position-std-m",
                args.max_stationary_amcl_position_std_m,
            )
        )
    for flag, value in localization_tube_limits:
        if value > localization_position_limit_m:
            parser.error(
                f"{flag} must not exceed half the certified route tube "
                f"({localization_position_limit_m:.6f} m)"
            )
    if args.physical_goal_tolerance_m > args.certified_route_tube_radius_m:
        parser.error(
            "--physical-goal-tolerance-m must not exceed "
            "--certified-route-tube-radius-m"
        )
    if (
        args.physical_waypoint_tolerance_m
        > args.certified_route_tube_radius_m
    ):
        parser.error(
            "--physical-waypoint-tolerance-m must not exceed "
            "--certified-route-tube-radius-m"
        )
    if (
        not math.isfinite(args.certified_route_chord_sample_spacing_m)
        or args.certified_route_chord_sample_spacing_m <= 0.0
    ):
        parser.error("--certified-route-chord-sample-spacing-m must be positive")
    if args.certified_corner_max_reacquire_attempts < 0:
        parser.error(
            "--certified-corner-max-reacquire-attempts must be non-negative"
        )
    if (
        not math.isfinite(args.viewpoint_sampling_goal_tolerance_m)
        or args.viewpoint_sampling_goal_tolerance_m <= 0.0
    ):
        parser.error("--viewpoint-sampling-goal-tolerance-m must be positive")
    if (
        not math.isfinite(
            args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        )
        or args.viewpoint_sampling_terminal_heading_hold_tolerance_m <= 0.0
        or args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        > INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
        or args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        < min(
            args.viewpoint_sampling_goal_tolerance_m,
            INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
        )
    ):
        parser.error(
            "--viewpoint-sampling-terminal-heading-hold-tolerance-m must "
            "be no smaller than the effective entry tolerance and no "
            "greater than 0.020"
        )
    if (
        not math.isfinite(args.viewpoint_sampling_heading_tolerance_rad)
        or args.viewpoint_sampling_heading_tolerance_rad <= 0.0
    ):
        parser.error("--viewpoint-sampling-heading-tolerance-rad must be positive")
    if (
        not math.isfinite(args.viewpoint_sampling_target_distance_m)
        or args.viewpoint_sampling_target_distance_m
        <= args.viewpoint_sampling_terminal_heading_hold_tolerance_m
    ):
        parser.error(
            "--viewpoint-sampling-target-distance-m must be finite and "
            "greater than the radial hold tolerance"
        )
    if (
        not math.isfinite(
            args.viewpoint_sampling_terminal_heading_target_envelope_radius_m
        )
        or args.viewpoint_sampling_terminal_heading_target_envelope_radius_m
        < args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        or args.viewpoint_sampling_terminal_heading_target_envelope_radius_m
        > INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    ):
        parser.error(
            "--viewpoint-sampling-terminal-heading-target-envelope-radius-m "
            "must be no smaller than the radial hold tolerance and no "
            "greater than 0.030"
        )
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
        not math.isfinite(args.linear_motion_floor_mps)
        or args.linear_motion_floor_mps <= 0.0
        or args.linear_motion_floor_mps > args.max_linear_mps
    ):
        parser.error(
            "--linear-motion-floor-mps must be positive and no greater than "
            "--max-linear-mps"
        )
    smoothing_values = {
        "--max-linear-accel-mps2": args.max_linear_accel_mps2,
        "--max-angular-accel-radps2": args.max_angular_accel_radps2,
    }
    for name, value in smoothing_values.items():
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if (
        not args.disable_command_smoothing
        and args.max_linear_accel_mps2 / 10.0 + 1.0e-12
        < args.linear_motion_floor_mps
    ):
        parser.error(
            "--max-linear-accel-mps2 must reach "
            "--linear-motion-floor-mps within one 10 Hz control period"
        )
    if (
        not math.isfinite(args.blockage_confirmation_timeout_sec)
        or args.blockage_confirmation_timeout_sec < 0.5
    ):
        parser.error(
            "--blockage-confirmation-timeout-sec must be finite and at least 0.5"
        )
    if not 3 <= args.blockage_confirmation_min_samples <= 7:
        parser.error(
            "--blockage-confirmation-min-samples must be between 3 and 7"
        )
    if (
        not math.isfinite(args.dynamic_route_terminal_lock_distance_m)
        or args.dynamic_route_terminal_lock_distance_m <= 0.0
    ):
        parser.error("--dynamic-route-terminal-lock-distance-m must be positive")
    args.run_id = args.run_id or f"aufgabe04-segment-{uuid.uuid4().hex[:8]}"
    args.semantic_log = args.semantic_log or DEFAULT_EVENT_LOG_DIR / f"{args.run_id}.jsonl"
    bundle_dir = os.environ.get("MII_AMR_RUN_BUNDLE_DIR", "").strip()
    if (
        args.controller_trace_jsonl is None
        and not args.dry_run
        and bundle_dir
    ):
        args.controller_trace_jsonl = Path(bundle_dir) / "controller_trace.jsonl"
    if (
        args.controller_trace_jsonl is not None
        and not args.dry_run
        and args.controller_trace_jsonl.exists()
    ):
        parser.error(
            "refusing to append controller evidence to an existing trace: "
            f"{args.controller_trace_jsonl}"
        )
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
    resolved_runtime_nomotion_update_service = resolve_topic(
        args.runtime_nomotion_update_service,
        resolved.namespace,
    )
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
        controller_trace_jsonl=str(args.controller_trace_jsonl or ""),
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
        allow_simulation_odom_after_stale_tf_requested=(
            args.allow_simulation_odom_after_stale_tf
        ),
        preflight_nomotion_update_service=args.nomotion_update_service,
        preflight_nomotion_update_timeout_sec=args.nomotion_update_timeout_sec,
        runtime_nomotion_update_service=(
            resolved_runtime_nomotion_update_service
        ),
        runtime_nomotion_update_service_configured=(
            args.runtime_nomotion_update_service
        ),
        runtime_nomotion_update_timeout_sec=(
            args.runtime_nomotion_update_timeout_sec
        ),
        amcl_edge_future_tolerance_sec=(
            args.max_localization_tf_future_sec
        ),
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

    try:
        diagnostics_snapshot = load_diagnostics_snapshot(
            diagnostics_json_path,
            require_metadata=leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS,
        )
    except ValueError as exc:
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
        parser.exit(2, f"error: route diagnostics validation failed: {exc}\n")

    diagnostics_metadata = diagnostics_snapshot.payload.get("metadata")
    route_purpose_value = (
        diagnostics_metadata.get("route_purpose")
        if isinstance(diagnostics_metadata, dict)
        else None
    )
    route_purpose = (
        route_purpose_value
        if isinstance(route_purpose_value, str)
        else ""
    )
    known_route_kinds = DYNAMIC_VIEWPOINT_ROUTE_KINDS | STATIC_PHYSICAL_ROUTE_KINDS
    if leg.route_kind == LEGACY_SIMULATION_ROUTE_KIND:
        if not args.allow_legacy_simulation_route:
            parser.exit(
                2,
                "error: legacy simulation route requires "
                "--allow-legacy-simulation-route\n",
            )
        if not leg.simulation_only or not args.allow_sim_time:
            parser.exit(
                2,
                "error: legacy route escape hatch is simulation-only and requires "
                "simulation_only=true plus --allow-sim-time\n",
            )
        if committed_route is not None:
            parser.exit(2, "error: legacy simulation route cannot use a route manifest\n")
    elif leg.route_kind not in known_route_kinds:
        parser.exit(
            2,
            f"error: missing or unknown Aufgabe04 route kind: {leg.route_kind!r}\n",
        )
    if odom_execution_enabled and leg.route_kind not in STATIC_PHYSICAL_ROUTE_KINDS:
        parser.exit(
            2,
            "error: odom execution currently requires a sealed static physical route\n",
        )
    if leg.route_kind in DYNAMIC_VIEWPOINT_ROUTE_KINDS and not leg.simulation_only:
        parser.exit(2, "error: dynamic viewpoint route is missing simulation_only provenance\n")
    if (
        args.coverage_transient_replan_enabled
        and leg.route_kind != STAND_DISCOVERY_ROUTE_KIND
    ):
        parser.exit(
            2,
            "error: physical transient replanning is restricted to "
            "stand_discovery_corridor\n",
        )
    if args.coverage_transient_replan_enabled and args.allow_sim_time:
        parser.exit(
            2,
            "error: physical transient replanning is not a simulation route mode\n",
        )
    try:
        coverage_replan_resume_state = (
            _validated_coverage_replan_resume_state(
                args,
                diagnostics_path=diagnostics_json_path,
            )
        )
    except (OSError, ValueError) as exc:
        emit_event(
            event_logger,
            "transient_overlay_resume_state_rejected",
            run_id=args.run_id,
            leg_index=args.leg_index,
            status="failed_closed",
            stop_reason=str(exc),
            motion_published=False,
        )
        parser.exit(
            2,
            f"error: transient overlay resume state rejected: {exc}\n",
        )
    if coverage_replan_resume_state is not None:
        emit_event(
            event_logger,
            "transient_overlay_resume_state_validated",
            run_id=args.run_id,
            leg_index=args.leg_index,
            resume_state_json=str(
                args.coverage_transient_replan_resume_state_json
            ),
            resume_state_sha256=(
                transient_overlay_resume_state_sha256(
                    coverage_replan_resume_state
                )
            ),
            completed_replan_count=(
                coverage_replan_resume_state.completed_replan_count
            ),
            max_replans=coverage_replan_resume_state.max_replans,
            remaining_replans=(
                coverage_replan_resume_state.remaining_replans
            ),
            motion_continues_authorized=False,
            automatic_motion_authorized=False,
        )
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
    simulation_odom_fallback_admission_failure = (
        _simulation_odom_fallback_admission_failure(
            args,
            resolved,
            leg,
            route_purpose=route_purpose,
            authoritative_dynamic_route=committed_route is not None,
        )
    )
    if simulation_odom_fallback_admission_failure:
        parser.exit(
            2,
            "error: "
            + simulation_odom_fallback_admission_failure
            + "\n",
        )
    allow_simulation_odom_after_stale_tf = bool(
        args.allow_simulation_odom_after_stale_tf
    )

    diagnostics_status = validate_route_diagnostics_json(
        diagnostics_json_path,
        args.leg_index,
        csv_point_count=len(leg.raw_waypoints),
        require_motion=require_motion,
        diagnostics_payload=diagnostics_snapshot.payload,
    )
    catalog_binding_status = (
        validate_catalog_route_binding_json(
            diagnostics_json_path,
            leg,
            catalog_path_override=args.arrival_pose_catalog,
            diagnostics_payload=diagnostics_snapshot.payload,
        )
        if leg.route_kind in CATALOG_PHYSICAL_ROUTE_KINDS
        else None
    )
    detected_stand_binding_status = (
        validate_detected_stand_preapproach_binding(
            diagnostics_json_path,
            leg,
            candidate_snapshot_path=args.candidate_snapshot,
            diagnostics_payload=diagnostics_snapshot.payload,
        )
        if leg.route_kind == DETECTED_STAND_PREAPPROACH_ROUTE_KIND
        else None
    )
    stand_discovery_binding_status = (
        validate_stand_discovery_route_binding(
            diagnostics_json_path,
            leg,
            coverage_plan_path=args.coverage_plan,
            diagnostics_payload=diagnostics_snapshot.payload,
        )
        if leg.route_kind == STAND_DISCOVERY_ROUTE_KIND
        else None
    )
    catalog_egress_certificate = None
    catalog_egress_failures = []
    execution_certificate_failures = []
    mission_execution_failures = []
    mission_execution_binding: MissionExecutionBinding | None = None
    if leg.route_kind in CATALOG_PHYSICAL_ROUTE_KINDS:
        try:
            catalog_egress_certificate = catalog_start_egress_certificate(
                diagnostics_json_path,
                leg,
                diagnostics_payload=diagnostics_snapshot.payload,
            )
        except ValueError as exc:
            catalog_egress_failures.append(
                f"catalog start-egress certificate is invalid: {exc}"
            )
    if leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS:
        execution_certificate_failures = _execution_certificate_failures(
            route_leg=leg,
            diagnostics_snapshot=diagnostics_snapshot,
            explicit_certificate_path=args.route_certificate_json,
            route_kind=leg.route_kind,
            runtime_namespace=resolved.namespace,
            runtime_planning_frame=resolved.map_frame,
            tracking_tube_radius_m=args.certified_route_tube_radius_m,
        )
        try:
            validate_arena_boundary_evidence(diagnostics_snapshot.metadata)
            if leg.route_kind == DETECTED_STAND_PREAPPROACH_ROUTE_KIND:
                if route_purpose != "pre_approach":
                    raise ValueError(
                        "detected stand route requires route_purpose=pre_approach"
                    )
                if args.candidate_snapshot is None:
                    raise ValueError(
                        "detected stand pre-approach requires --candidate-snapshot"
                    )
            elif leg.route_kind == STAND_DISCOVERY_ROUTE_KIND:
                if route_purpose != "stand_discovery":
                    raise ValueError(
                        "stand discovery route requires "
                        "route_purpose=stand_discovery"
                    )
                if args.coverage_plan is None:
                    raise ValueError(
                        "stand discovery route requires --coverage-plan"
                    )
            elif route_purpose == "logistics":
                missing = [
                    option
                    for option, value in (
                        ("--mission-plan-manifest", args.mission_plan_manifest),
                        ("--survey-manifest", args.survey_manifest),
                        ("--route-bundle-json", args.route_bundle_json),
                        ("--planner-config-json", args.planner_config_json),
                        ("--runtime-map-bundle-json", args.runtime_map_bundle_json),
                        ("--runtime-environment", args.runtime_environment),
                        ("--candidate-snapshot", args.candidate_snapshot),
                        (
                            "--station-identity-registry",
                            args.station_identity_registry,
                        ),
                        ("--arrival-pose-catalog", args.arrival_pose_catalog),
                        ("--task-snapshot", args.task_snapshot),
                    )
                    if value is None
                ]
                if missing:
                    raise ValueError(
                        "logistics execution requires " + ", ".join(missing)
                    )
                mission_execution_binding = validate_logistics_execution_bundle(
                    route_leg=leg,
                    diagnostics_path=diagnostics_json_path,
                    route_certificate_path=args.route_certificate_json,
                    mission_plan_path=args.mission_plan_manifest,
                    survey_manifest_path=args.survey_manifest,
                    route_bundle_path=args.route_bundle_json,
                    planner_config_path=args.planner_config_json,
                    runtime_map_bundle_path=args.runtime_map_bundle_json,
                    runtime_environment_path=args.runtime_environment,
                    candidate_snapshot_path=args.candidate_snapshot,
                    station_identity_registry_path=(
                        args.station_identity_registry
                    ),
                    arrival_pose_catalog_path=args.arrival_pose_catalog,
                    task_snapshot_path=args.task_snapshot,
                    robot_id=args.robot_id,
                    runtime_planning_frame=resolved.map_frame,
                    diagnostics_snapshot=diagnostics_snapshot,
                )
            elif route_purpose == "survey":
                if not (
                    args.allow_unbound_survey_simulation_route
                    and args.allow_sim_time
                    and leg.simulation_only
                ):
                    raise ValueError(
                        "static survey route is unbound to a task mission; a "
                        "simulation demonstration requires "
                        "--allow-unbound-survey-simulation-route, "
                        "--allow-sim-time, and simulation_only=true"
                    )
            else:
                raise ValueError(
                    f"static route has missing or unknown route_purpose: {route_purpose!r}"
                )
        except (OSError, ValueError) as exc:
            mission_execution_failures.append(
                f"mission execution binding is invalid: {exc}"
            )
    speed_status = validate_speed_limits(args.max_linear_mps, args.max_angular_radps)
    pure_failures = (
        diagnostics_status.failures
        + ([] if catalog_binding_status is None else catalog_binding_status.failures)
        + (
            []
            if detected_stand_binding_status is None
            else detected_stand_binding_status.failures
        )
        + (
            []
            if stand_discovery_binding_status is None
            else stand_discovery_binding_status.failures
        )
        + catalog_egress_failures
        + execution_certificate_failures
        + mission_execution_failures
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
            max_future_timestamp_sec=args.max_future_timestamp_sec,
            max_localization_tf_future_sec=(
                args.max_localization_tf_future_sec
            ),
            observation_window_sec=args.preflight_observation_window_sec,
            allowed_cmd_vel_publishers=args.allowed_cmd_vel_publisher,
            require_real_time=not args.allow_sim_time,
            request_nomotion_update=(
                resolved.localization_source == "amcl"
                and not args.skip_nomotion_update_before_preflight
            ),
            nomotion_update_service=args.nomotion_update_service,
            nomotion_update_timeout_sec=args.nomotion_update_timeout_sec,
            stationary_amcl_sample_count=(
                args.stationary_amcl_sample_count
            ),
            stationary_amcl_sample_interval_sec=(
                args.stationary_amcl_sample_interval_sec
            ),
            max_stationary_amcl_position_spread_m=(
                args.max_stationary_amcl_position_spread_m
            ),
            max_stationary_amcl_yaw_spread_rad=(
                args.max_stationary_amcl_yaw_spread_rad
            ),
            max_stationary_amcl_position_std_m=(
                args.max_stationary_amcl_position_std_m
            ),
            max_stationary_amcl_yaw_std_rad=(
                args.max_stationary_amcl_yaw_std_rad
            ),
            execution_pose_owner=(
                "odom" if args.execution_pose_frame == "odom" else ""
            ),
            global_consistency_monitor=(
                "amcl" if args.execution_pose_frame == "odom" else ""
            ),
            # The runner constructs and validates the certificate immediately
            # from this stopped capture before dry-run success or RUN can be
            # reached. This flag selects the intended ownership contract; the
            # resulting artifact is still mandatory below.
            frozen_map_transform_certified=(
                args.execution_pose_frame == "odom"
            ),
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
    startup_rejection = _static_start_preflight_rejection(
        preflight,
        leg,
        map_frame=resolved.map_frame,
        base_frame=resolved.base_frame,
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
    )
    if startup_rejection is not None:
        _append_result(
            args,
            resolved,
            leg,
            preflight_ok=True,
            result=startup_rejection,
        )
        emit_event(
            event_logger,
            "startup_route_rejected",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            coverage_leg_index=(
                args.coverage_transient_replan_leg_index
            ),
            target_viewpoint_id=(
                args.coverage_transient_replan_target_viewpoint_id
            ),
            status=startup_rejection.status,
            stop_reason=startup_rejection.stop_reason,
            motion_published=False,
            stop_details=startup_rejection.stop_details,
        )
        emit_event(
            event_logger,
            "safety_stop",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            coverage_leg_index=(
                args.coverage_transient_replan_leg_index
            ),
            target_viewpoint_id=(
                args.coverage_transient_replan_target_viewpoint_id
            ),
            status=startup_rejection.status,
            stop_reason=startup_rejection.stop_reason,
            motion_published=False,
            stop_details=startup_rejection.stop_details,
            duration_sec=0.0,
            distance_estimate_m=0.0,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=startup_rejection.status,
            stop_reason=startup_rejection.stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 1
    execution_waypoints = poses_from_waypoints(leg.executable_waypoints)
    odom_execution_context: OdomExecutionContext | None = None
    odom_execution_evidence: dict[str, object] = {}
    odom_replacement_route_gate: _OdomRouteUncertaintyGate | None = None
    if args.execution_pose_frame == "odom":
        try:
            (
                execution_waypoints,
                odom_execution_context,
                odom_execution_evidence,
                odom_replacement_route_gate,
            ) = _build_odom_execution_admission(
                args=args,
                resolved=resolved,
                leg=leg,
                preflight=preflight,
                diagnostics_snapshot=diagnostics_snapshot,
            )
        except (OSError, ValueError) as exc:
            stop_reason = f"odom execution admission failed: {exc}"
            stop_details = {
                "reason": stop_reason,
                "fault_code": "odom_execution_admission_failed",
                "execution_pose_owner": "odom",
                "global_consistency_monitor": "amcl",
                "motion_published": False,
                "fail_closed": True,
            }
            result = FollowerResult(
                "preflight_failed",
                stop_reason,
                0.0,
                0.0,
                False,
                stop_details,
            )
            _append_result(
                args,
                resolved,
                leg,
                preflight_ok=False,
                result=result,
            )
            emit_event(
                event_logger,
                "odom_execution_admission_failed",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status=result.status,
                stop_reason=stop_reason,
                motion_published=False,
                stop_details=stop_details,
            )
            emit_event(
                event_logger,
                "safety_stop",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status=result.status,
                stop_reason=stop_reason,
                motion_published=False,
                stop_details=stop_details,
                duration_sec=0.0,
                distance_estimate_m=0.0,
            )
            emit_event(
                event_logger,
                "run_finished",
                run_id=args.run_id,
                final_status=result.status,
                stop_reason=stop_reason,
                results_csv=str(args.results_csv),
                semantic_log_path=str(args.semantic_log),
                preflight_json_path=str(args.preflight_json or ""),
            )
            return 1
        emit_event(
            event_logger,
            "odom_execution_sealed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            **odom_execution_evidence,
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
            execution_pose_frame=args.execution_pose_frame,
            odom_execution_evidence=odom_execution_evidence,
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
    runtime_motion_permit = None
    mission_leg_motion_permit = None
    startup_reseal_motion_permit = None
    try:
        runtime_motion_permit = _validated_runtime_localization_motion_permit(
            args,
            resolved,
            route_csv_path=route_csv_path,
            diagnostics_path=diagnostics_json_path,
        )
        mission_leg_motion_permit = _validated_mission_leg_motion_permit(
            args,
            resolved,
            route_csv_path=route_csv_path,
            diagnostics_path=diagnostics_json_path,
        )
        startup_reseal_motion_permit = (
            _validated_startup_reseal_motion_permit(
                args,
                resolved,
                route_csv_path=route_csv_path,
                diagnostics_path=diagnostics_json_path,
            )
        )
    except ValueError as exc:
        return _record_motion_authorization_rejection(
            args=args,
            resolved=resolved,
            leg=leg,
            event_logger=event_logger,
            failure=exc,
        )

    if (
        runtime_motion_permit is None
        and mission_leg_motion_permit is None
        and startup_reseal_motion_permit is None
        and not _confirm_motion(args, resolved)
    ):
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

    if mission_execution_binding is not None:
        try:
            current_diagnostics_snapshot = load_diagnostics_snapshot(
                diagnostics_json_path
            )
            current_binding = validate_logistics_execution_bundle(
                route_leg=leg,
                diagnostics_path=diagnostics_json_path,
                route_certificate_path=args.route_certificate_json,
                mission_plan_path=args.mission_plan_manifest,
                survey_manifest_path=args.survey_manifest,
                route_bundle_path=args.route_bundle_json,
                planner_config_path=args.planner_config_json,
                runtime_map_bundle_path=args.runtime_map_bundle_json,
                runtime_environment_path=args.runtime_environment,
                candidate_snapshot_path=args.candidate_snapshot,
                station_identity_registry_path=args.station_identity_registry,
                arrival_pose_catalog_path=args.arrival_pose_catalog,
                task_snapshot_path=args.task_snapshot,
                robot_id=args.robot_id,
                runtime_planning_frame=resolved.map_frame,
                diagnostics_snapshot=current_diagnostics_snapshot,
            )
            if current_binding != mission_execution_binding:
                raise ValueError("mission execution artifacts changed before motion")
        except (OSError, ValueError) as exc:
            stop_reason = f"mission execution revalidation failed: {exc}"
            result = FollowerResult(
                "stopped",
                stop_reason,
                0.0,
                0.0,
                False,
                {"fault_code": "mission_revalidation_failed", "fail_closed": True},
            )
            _append_result(args, resolved, leg, preflight_ok=True, result=result)
            emit_event(
                event_logger,
                "mission_execution_rejected",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status="stopped",
                phase="immediately_before_motion",
                stop_reason=stop_reason,
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
            enforce_heading_corridor=(
                leg.route_kind in HEADING_CORRIDOR_ROUTE_KINDS
            ),
            exact_vertex_pursuit=leg.route_kind in PHYSICAL_ROUTE_KINDS,
        ),
        command_smoothing=CommandSmoothingConfig(
            enabled=not args.disable_command_smoothing,
            max_linear_accel_mps2=args.max_linear_accel_mps2,
            max_angular_accel_radps2=args.max_angular_accel_radps2,
        ),
        min_obstacle_distance_m=args.min_obstacle_distance_m,
        omnidirectional_hard_stop_distance_m=(
            args.omnidirectional_hard_stop_distance_m
        ),
        front_obstacle_slow_distance_m=args.front_obstacle_slow_distance_m,
        front_obstacle_sector_rad=args.front_obstacle_sector_rad,
        max_scan_age_sec=args.max_scan_age_sec,
        max_odom_age_sec=args.max_odom_age_sec,
        max_tf_age_sec=args.max_tf_age_sec,
        max_future_timestamp_sec=args.max_future_timestamp_sec,
        amcl_edge_future_tolerance_sec=(
            args.max_localization_tf_future_sec
        ),
        runtime_nomotion_update_service=(
            args.runtime_nomotion_update_service
        ),
        runtime_nomotion_update_timeout_sec=(
            args.runtime_nomotion_update_timeout_sec
        ),
        allow_simulation_odom_after_stale_tf=(
            allow_simulation_odom_after_stale_tf
        ),
        initial_sensor_wait_sec=args.initial_sensor_wait_sec,
        waypoint_timeout_sec=args.waypoint_timeout_sec,
        stuck_timeout_sec=args.stuck_timeout_sec,
        stuck_progress_epsilon_m=args.stuck_progress_epsilon_m,
        stuck_heading_progress_epsilon_rad=(
            args.stuck_heading_progress_epsilon_rad
        ),
        linear_motion_floor_mps=args.linear_motion_floor_mps,
        blockage_confirmation_timeout_sec=(
            args.blockage_confirmation_timeout_sec
        ),
        persistent_obstacle_config=PersistentObstacleConfig(
            min_distinct_samples=args.blockage_confirmation_min_samples,
            min_front_range_m=args.omnidirectional_hard_stop_distance_m,
            max_front_range_m=args.front_obstacle_slow_distance_m,
            front_sector_half_width_rad=args.front_obstacle_sector_rad,
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
        viewpoint_sampling_target_timeout_sec=(
            args.viewpoint_sampling_target_timeout_sec
        ),
        viewpoint_sampling_goal_tolerance_m=(
            args.viewpoint_sampling_goal_tolerance_m
        ),
        viewpoint_sampling_terminal_heading_hold_tolerance_m=(
            args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        viewpoint_sampling_target_distance_m=(
            args.viewpoint_sampling_target_distance_m
        ),
        viewpoint_sampling_terminal_heading_target_envelope_radius_m=(
            args
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m
        ),
        viewpoint_sampling_heading_tolerance_rad=(
            args.viewpoint_sampling_heading_tolerance_rad
        ),
        physical_waypoint_tolerance_m=args.physical_waypoint_tolerance_m,
        physical_goal_tolerance_m=args.physical_goal_tolerance_m,
        certified_route_tube_radius_m=args.certified_route_tube_radius_m,
        certified_route_chord_sample_spacing_m=(
            args.certified_route_chord_sample_spacing_m
        ),
        certified_corner_max_reacquire_attempts=(
            args.certified_corner_max_reacquire_attempts
        ),
    )
    resolved_controller_config = controller_config_for_route_kind(
        follower_config.controller,
        leg.route_kind,
        viewpoint_sampling_goal_tolerance_m=(
            follower_config.viewpoint_sampling_goal_tolerance_m
        ),
        viewpoint_sampling_heading_tolerance_rad=(
            follower_config.viewpoint_sampling_heading_tolerance_rad
        ),
        physical_waypoint_tolerance_m=(
            follower_config.physical_waypoint_tolerance_m
        ),
        physical_goal_tolerance_m=follower_config.physical_goal_tolerance_m,
    )
    resolved_terminal_goal_tolerance_m = (
        resolved_controller_config.goal_tolerance_m
        if resolved_controller_config.terminal_goal_tolerance_m is None
        else resolved_controller_config.terminal_goal_tolerance_m
    )
    emit_event(
        event_logger,
        "controller_config_resolved",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        route_kind=leg.route_kind,
        max_linear_mps=follower_config.controller.max_linear_mps,
        max_angular_radps=follower_config.controller.max_angular_radps,
        min_obstacle_distance_m=follower_config.min_obstacle_distance_m,
        omnidirectional_hard_stop_distance_m=(
            follower_config.omnidirectional_hard_stop_distance_m
        ),
        coverage_transient_replan_enabled=(
            args.coverage_transient_replan_enabled
        ),
        coverage_transient_replan_max_count=(
            args.coverage_transient_replan_max_count
        ),
        coverage_transient_replan_resume_state_json=str(
            args.coverage_transient_replan_resume_state_json or ""
        ),
        coverage_transient_replan_initial_count=(
            0
            if coverage_replan_resume_state is None
            else coverage_replan_resume_state.completed_replan_count
        ),
        coverage_transient_replan_remaining_count=(
            args.coverage_transient_replan_max_count
            if coverage_replan_resume_state is None
            else coverage_replan_resume_state.remaining_replans
        ),
        linear_motion_floor_mps=follower_config.linear_motion_floor_mps,
        blockage_confirmation_timeout_sec=(
            follower_config.blockage_confirmation_timeout_sec
        ),
        blockage_confirmation_thresholds=(
            follower_config.persistent_obstacle_config.to_log_dict()
            if follower_config.persistent_obstacle_config is not None
            else {}
        ),
        controller_trace_jsonl=str(args.controller_trace_jsonl or ""),
        effective_goal_tolerance_m=resolved_terminal_goal_tolerance_m,
        effective_intermediate_goal_tolerance_m=(
            resolved_controller_config.goal_tolerance_m
        ),
        effective_terminal_goal_tolerance_m=(
            resolved_terminal_goal_tolerance_m
        ),
        intermediate_terminal_heading_entry_tolerance_m=(
            intermediate_terminal_heading_entry_tolerance_m(
                resolved_controller_config
            )
        ),
        intermediate_terminal_heading_hold_tolerance_m=(
            follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        intermediate_terminal_heading_distance_comparison_epsilon_m=(
            INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
        ),
        intermediate_terminal_heading_effective_hold_limit_m=(
            follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
            + INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
        ),
        intermediate_terminal_heading_target_distance_m=(
            follower_config.viewpoint_sampling_target_distance_m
        ),
        intermediate_terminal_heading_target_envelope_radius_m=(
            follower_config
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m
        ),
        intermediate_terminal_heading_minimum_stand_distance_m=(
            follower_config.viewpoint_sampling_target_distance_m
            - follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        intermediate_terminal_heading_maximum_stand_distance_m=(
            follower_config.viewpoint_sampling_target_distance_m
            + follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        heading_tolerance_rad=resolved_controller_config.heading_tolerance_rad,
        enforce_heading_corridor=(
            resolved_controller_config.enforce_heading_corridor
        ),
        slow_heading_error_rad=follower_config.controller.slow_heading_error_rad,
        stop_heading_error_rad=follower_config.controller.stop_heading_error_rad,
        exact_vertex_pursuit=resolved_controller_config.exact_vertex_pursuit,
        exact_vertex_alignment_enabled=(
            resolved_controller_config.exact_vertex_pursuit
        ),
        command_smoothing_enabled=follower_config.command_smoothing.enabled,
        max_linear_accel_mps2=(
            follower_config.command_smoothing.max_linear_accel_mps2
        ),
        max_angular_accel_radps2=(
            follower_config.command_smoothing.max_angular_accel_radps2
        ),
        start_egress_waypoint_index=(
            follower_config.initial_start_egress_waypoint_index
        ),
        start_egress_waypoint_tolerance_m=(
            follower_config.start_egress_waypoint_tolerance_m
        ),
        start_egress_alignment_tolerance_rad=(
            follower_config.start_egress_alignment_tolerance_rad
        ),
        start_egress_max_linear_mps=(
            follower_config.start_egress_max_linear_mps
        ),
        initial_start_join_clearance_m=(
            follower_config.initial_start_join_clearance_m
        ),
        certified_route_tube_radius_m=(
            follower_config.certified_route_tube_radius_m
        ),
        certified_route_chord_sample_spacing_m=(
            follower_config.certified_route_chord_sample_spacing_m
        ),
        certified_corner_transition_enabled=(
            leg.route_kind == "stand_discovery_corridor"
        ),
        certified_corner_turn_threshold_rad=(
            follower_config.certified_corner_turn_threshold_rad
        ),
        certified_corner_release_tolerance_m=(
            follower_config.certified_corner_release_tolerance_m
        ),
        certified_corner_hold_tolerance_m=(
            follower_config.certified_corner_hold_tolerance_m
        ),
        certified_corner_alignment_tolerance_rad=(
            follower_config.certified_corner_alignment_tolerance_rad
        ),
        certified_corner_max_reacquire_attempts=(
            follower_config.certified_corner_max_reacquire_attempts
        ),
        allow_simulation_odom_after_stale_tf=(
            follower_config.allow_simulation_odom_after_stale_tf
        ),
        amcl_edge_future_tolerance_sec=(
            follower_config.amcl_edge_future_tolerance_sec
        ),
        runtime_nomotion_update_service=(
            resolved_runtime_nomotion_update_service
        ),
        runtime_nomotion_update_service_configured=(
            follower_config.runtime_nomotion_update_service
        ),
        runtime_nomotion_update_timeout_sec=(
            follower_config.runtime_nomotion_update_timeout_sec
        ),
        route_purpose=route_purpose,
        route_simulation_only=leg.simulation_only,
    )
    waypoint_provider = None
    blockage_recovery_provider = None

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
        if (
            event_name == "transient_navigation_blockage_replanned"
            and args.coverage_transient_replan_enabled
        ):
            # The coordinator prepares artifacts while zero is held, but this
            # callback runs only after the follower has atomically installed
            # the replacement.  Persist "replanned" here so the parent never
            # mistakes a merely prepared route for an adopted one.
            _append_jsonl(
                Path(args.coverage_transient_replan_session_root)
                / "adaptive_replans.jsonl",
                {
                    "schema_version": 1,
                    "event": event_name,
                    "timestamp": time.time(),
                    "run_id": args.run_id,
                    "leg_index": args.coverage_transient_replan_leg_index,
                    **dict(update.event_fields),
                },
            )

    if args.coverage_transient_replan_enabled:
        if coverage_replan_resume_state is not None:
            try:
                live_resume_state = _validated_coverage_replan_resume_state(
                    args,
                    diagnostics_path=diagnostics_json_path,
                )
                if live_resume_state != coverage_replan_resume_state:
                    raise ValueError(
                        "transient overlay resume state changed before motion"
                    )
            except (OSError, ValueError) as exc:
                return _record_motion_authorization_rejection(
                    args=args,
                    resolved=resolved,
                    leg=leg,
                    event_logger=event_logger,
                    failure=(
                        "transient overlay resume-state revalidation failed: "
                        f"{exc}"
                    ),
                )
        blockage_recovery_provider = CoverageReplanCoordinator(
            survey_root=args.coverage_transient_replan_survey_root,
            session_root=args.coverage_transient_replan_session_root,
            map_yaml=args.coverage_transient_replan_map,
            semantic_map_id=args.coverage_transient_replan_semantic_map_id,
            target_viewpoint_id=(
                args.coverage_transient_replan_target_viewpoint_id
            ),
            run_id=args.run_id,
            coverage_leg_index=args.coverage_transient_replan_leg_index,
            route_leg_index=leg.leg_index,
            command_owner=_runtime_command_owner(resolved.namespace),
            robot_radius_m=args.coverage_transient_replan_robot_radius_m,
            max_replans=args.coverage_transient_replan_max_count,
            replan_count=(
                0
                if coverage_replan_resume_state is None
                else coverage_replan_resume_state.completed_replan_count
            ),
            overlay_path=(
                None
                if coverage_replan_resume_state is None
                else Path(
                    coverage_replan_resume_state.transient_obstacle_overlay_path
                )
            ),
            adopted_route_hashes=(
                set()
                if coverage_replan_resume_state is None
                else set(
                    coverage_replan_resume_state.adopted_route_sha256s
                )
                | {leg.source_sha256}
            ),
            tracking_tube_radius_m=args.certified_route_tube_radius_m,
            forward_translation_heading_limit_rad=(
                follower_config.controller.stop_heading_error_rad
            ),
            reverse_connector_alignment_tolerance_rad=(
                follower_config.start_egress_alignment_tolerance_rad
            ),
        )
        if odom_execution_context is not None:
            assert odom_replacement_route_gate is not None
            blockage_recovery_provider = _OdomBlockageRecoveryAdapter(
                blockage_recovery_provider,
                odom_execution_context,
                odom_replacement_route_gate,
            )

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
    if mission_leg_motion_permit is not None:
        try:
            mission_leg_receipt_path = (
                default_mission_leg_motion_consumption_receipt_path(
                    args.mission_leg_motion_permit_json
                )
            )
            mission_leg_receipt = consume_mission_leg_motion_permit(
                permit_path=args.mission_leg_motion_permit_json,
                permit=mission_leg_motion_permit,
                session_id=args.mission_session_id,
                run_id=args.run_id,
                mission_leg_kind=(
                    mission_leg_motion_permit.mission_leg_kind
                ),
                mission_leg_index=(
                    mission_leg_motion_permit.mission_leg_index
                ),
                target_id=mission_leg_motion_permit.target_id,
            )
        except ValueError as exc:
            return _record_motion_authorization_rejection(
                args=args,
                resolved=resolved,
                leg=leg,
                event_logger=event_logger,
                failure=exc,
            )
        print(
            "Using the mission-level RUN for this exact routine child leg. "
            "All live gates passed and its one-use receipt was claimed; no "
            "additional operator input is requested."
        )
        emit_event(
            event_logger,
            "mission_leg_motion_permit_consumed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            mission_leg_kind=(
                mission_leg_motion_permit.mission_leg_kind.value
            ),
            mission_leg_index=(
                mission_leg_motion_permit.mission_leg_index
            ),
            target_id=mission_leg_motion_permit.target_id,
            coverage_leg_index=(
                mission_leg_motion_permit.mission_leg_index
                if mission_leg_motion_permit.mission_leg_kind
                is MissionLegKind.COVERAGE
                else None
            ),
            target_viewpoint_id=(
                mission_leg_motion_permit.target_id
                if mission_leg_motion_permit.mission_leg_kind
                is MissionLegKind.COVERAGE
                else ""
            ),
            mission_leg_motion_authorization_json=str(
                args.mission_leg_motion_authorization_json
            ),
            mission_leg_motion_permit_json=str(
                args.mission_leg_motion_permit_json
            ),
            mission_leg_motion_permit_sha256=(
                mission_leg_motion_permit_sha256(
                    mission_leg_motion_permit
                )
            ),
            mission_leg_motion_consumption_receipt_json=str(
                mission_leg_receipt_path
            ),
            mission_leg_motion_consumption_receipt_sha256=(
                mission_leg_motion_consumption_receipt_sha256(
                    mission_leg_receipt
                )
            ),
            covered_by_initial_mission_run=True,
            additional_typed_run_required=False,
        )
    if startup_reseal_motion_permit is not None:
        try:
            startup_receipt_path = (
                default_startup_reseal_motion_consumption_receipt_path(
                    args.startup_reseal_motion_permit_json
                )
            )
            startup_receipt = consume_startup_reseal_motion_permit(
                permit_path=args.startup_reseal_motion_permit_json,
                permit=startup_reseal_motion_permit,
                session_id=args.mission_session_id,
                run_id=args.run_id,
                leg_index=startup_reseal_motion_permit.leg_index,
                target_viewpoint_id=(
                    startup_reseal_motion_permit.target_viewpoint_id
                ),
                reseal_index=startup_reseal_motion_permit.reseal_index,
            )
        except ValueError as exc:
            return _record_motion_authorization_rejection(
                args=args,
                resolved=resolved,
                leg=leg,
                event_logger=event_logger,
                failure=exc,
            )
        print(
            "Using the mission-level RUN for this exact bounded, same-target "
            "startup recovery. All live gates passed and the one-use receipt "
            "was claimed; no additional operator input is requested."
        )
        emit_event(
            event_logger,
            "startup_reseal_motion_permit_consumed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            target_viewpoint_id=(
                startup_reseal_motion_permit.target_viewpoint_id
            ),
            coverage_leg_index=startup_reseal_motion_permit.leg_index,
            recovery_source_kind=(
                startup_reseal_motion_permit.recovery_source_kind
            ),
            reseal_index=startup_reseal_motion_permit.reseal_index,
            rejected_run_id=startup_reseal_motion_permit.rejected_run_id,
            startup_reseal_motion_authorization_json=str(
                args.startup_reseal_motion_authorization_json
            ),
            startup_reseal_motion_permit_json=str(
                args.startup_reseal_motion_permit_json
            ),
            startup_reseal_motion_permit_sha256=(
                startup_reseal_motion_permit_sha256(
                    startup_reseal_motion_permit
                )
            ),
            startup_reseal_motion_consumption_receipt_json=str(
                startup_receipt_path
            ),
            startup_reseal_motion_consumption_receipt_sha256=(
                startup_reseal_motion_consumption_receipt_sha256(
                    startup_receipt
                )
            ),
            covered_by_initial_mission_run=True,
            additional_typed_run_required=False,
        )
    if runtime_motion_permit is not None:
        try:
            receipt_path = default_runtime_motion_consumption_receipt_path(
                args.runtime_localization_motion_permit_json
            )
            runtime_motion_receipt = consume_runtime_motion_permit(
                permit_path=args.runtime_localization_motion_permit_json,
                permit=runtime_motion_permit,
                session_id=args.mission_session_id,
                run_id=args.run_id,
                leg_index=runtime_motion_permit.leg_index,
                target_viewpoint_id=(
                    runtime_motion_permit.target_viewpoint_id
                ),
                reseal_index=runtime_motion_permit.reseal_index,
            )
        except ValueError as exc:
            return _record_motion_authorization_rejection(
                args=args,
                resolved=resolved,
                leg=leg,
                event_logger=event_logger,
                failure=exc,
            )
        print(
            "Using the mission-level RUN for this exact bounded, same-target "
            "runtime-localization recovery. All live gates passed and the "
            "one-use receipt was claimed; no additional operator input is "
            "requested."
        )
        emit_event(
            event_logger,
            "runtime_localization_motion_permit_consumed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            target_viewpoint_id=(
                runtime_motion_permit.target_viewpoint_id
            ),
            coverage_leg_index=runtime_motion_permit.leg_index,
            reseal_index=runtime_motion_permit.reseal_index,
            rejected_run_id=runtime_motion_permit.rejected_run_id,
            mission_motion_authorization_json=str(
                args.mission_motion_authorization_json
            ),
            runtime_localization_motion_permit_json=str(
                args.runtime_localization_motion_permit_json
            ),
            runtime_localization_motion_permit_sha256=(
                runtime_localization_motion_permit_sha256(
                    runtime_motion_permit
                )
            ),
            runtime_motion_consumption_receipt_json=str(receipt_path),
            runtime_motion_consumption_receipt_sha256=(
                runtime_motion_consumption_receipt_sha256(
                    runtime_motion_receipt
                )
            ),
            covered_by_initial_mission_run=True,
            additional_typed_run_required=False,
        )
    emit_event(
        event_logger,
        "motion_started",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        coverage_leg_index=args.coverage_transient_replan_leg_index,
        target_viewpoint_id=(
            args.coverage_transient_replan_target_viewpoint_id
        ),
        # This is an execution-attempt boundary, emitted immediately before
        # entering the follower.  It is not evidence of a nonzero Twist.
        motion_published=False,
        event_semantics="child_execution_attempt_started_before_follower",
        resolved_cmd_vel_topic=resolved.cmd_vel_topic,
    )
    follower_kwargs = {}
    if blockage_recovery_provider is not None:
        follower_kwargs["blockage_recovery_provider"] = (
            blockage_recovery_provider
        )
    if args.controller_trace_jsonl is not None:
        follower_kwargs["controller_trace_path"] = (
            args.controller_trace_jsonl
        )
    if odom_execution_context is not None:
        follower_kwargs["odom_execution_context"] = odom_execution_context
    result = run_simple_waypoint_follower(
        resolved,
        execution_waypoints,
        follower_config,
        waypoint_provider,
        route_update_callback,
        **follower_kwargs,
    )
    _append_result(args, resolved, leg, preflight_ok=True, result=result)
    motion_event_fields = {
        "run_id": args.run_id,
        "leg_index": leg.leg_index,
        # ``leg_index`` above selects a row in this child route artifact and
        # is normally zero.  Keep the autonomous coverage identity explicit
        # so recovery permits never confuse route-local and mission indices.
        "coverage_leg_index": (
            args.coverage_transient_replan_leg_index
        ),
        "target_viewpoint_id": (
            args.coverage_transient_replan_target_viewpoint_id
        ),
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
