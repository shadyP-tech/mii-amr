"""Scoped unloaded travel limits for the sealed post-exploration Start leg.

These are commanded limits, not a measured maximum safe vehicle speed.
ROBOTIS lists 0.22 m/s and 2.84 rad/s for Burger; this policy stays below
those limits while retaining live clearance, exact-route and stop checks:
https://emanual.robotis.com/docs/en/platform/turtlebot3/features/
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from scripts.aufgabe04.navigation.control.waypoint_controller import (
    distance,
    route_vertex_turn_angle_rad,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind,
)


RETURN_TO_START_SPEED_POLICY = "unloaded_return_to_start"
RETURN_TO_START_LINEAR_MPS = 0.15
RETURN_TO_START_ANGULAR_RADPS = 0.60
RETURN_TO_START_SENSOR_AGE_SEC = 0.25
# Reserve 7.5 cm in addition to robot radius, collision margin, tracking tube,
# odom drift and localization covariance. At the cruise cap this allocates
# 0.5 seconds of travel; stale sensing is stopped after .25 s plus a 10 Hz
# control period. Hardware stopping performance remains a commissioning check.
RETURN_TO_START_BRAKING_LATENCY_DISTANCE_M = 0.075
RETURN_TO_START_PRECISE_DISTANCE_M = 0.18
RETURN_TO_START_PRECISE_LINEAR_MPS = 0.055
RETURN_TO_START_PRECISE_ANGULAR_RADPS = 0.18


def return_to_start_speed_policy_arguments() -> list[str]:
    return [
        "--motion-speed-policy", RETURN_TO_START_SPEED_POLICY,
        "--max-scan-age-sec", str(RETURN_TO_START_SENSOR_AGE_SEC),
        "--max-odom-age-sec", str(RETURN_TO_START_SENSOR_AGE_SEC),
        "--max-tf-age-sec", str(RETURN_TO_START_SENSOR_AGE_SEC),
        "--uncertainty-braking-latency-distance-m",
        str(RETURN_TO_START_BRAKING_LATENCY_DISTANCE_M),
    ]


def return_to_start_speed_policy_failures(args, *, route_kind, simulation_only):
    """Admit the higher ceiling only in the exact physical return scope.

The live permit itself is validated and consumed by the existing runner;
merely declaring evidence or supplying a route can never replace that permit.
"""
    if getattr(args, "motion_speed_policy", "exploration") != RETURN_TO_START_SPEED_POLICY:
        return []
    failures = []
    if route_kind != "admitted_candidate_pose":
        failures.append("fast Start travel requires admitted_candidate_pose")
    if simulation_only or args.allow_sim_time or args.execution_pose_frame != "odom":
        failures.append("fast Start travel requires physical odom execution")
    kind = (args.mission_leg_evidence_kind if args.dry_run else args.mission_leg_kind)
    if kind != MissionLegKind.RETURN_TO_START.value:
        failures.append("fast Start travel requires return_to_start mission identity")
    if not str(args.operator_note).startswith("UNLOADED "):
        failures.append("fast Start travel requires an unloaded run declaration")
    if not args.dry_run and args.mission_leg_motion_permit_json is None:
        failures.append("fast Start travel requires an exact routine mission-leg permit")
    exact = {
        "max_linear_mps": RETURN_TO_START_LINEAR_MPS,
        "max_angular_radps": RETURN_TO_START_ANGULAR_RADPS,
        "max_scan_age_sec": RETURN_TO_START_SENSOR_AGE_SEC,
        "max_odom_age_sec": RETURN_TO_START_SENSOR_AGE_SEC,
        "max_tf_age_sec": RETURN_TO_START_SENSOR_AGE_SEC,
        "uncertainty_braking_latency_distance_m": RETURN_TO_START_BRAKING_LATENCY_DISTANCE_M,
        "min_obstacle_distance_m": 0.20,
        "front_obstacle_slow_distance_m": 0.38,
        "max_linear_accel_mps2": 0.10,
        "max_angular_accel_radps2": 0.60,
    }
    for name, expected in exact.items():
        if getattr(args, name, None) != expected:
            failures.append(f"fast Start travel requires {name}={expected}")
    if args.disable_command_smoothing:
        failures.append("fast Start travel requires command smoothing")
    return failures


def return_to_start_speed_policy_evidence(args) -> dict[str, object]:
    """Persist the dry/live command envelope under the permit's artifact hash."""
    if getattr(args, "motion_speed_policy", "exploration") != RETURN_TO_START_SPEED_POLICY:
        return {}
    names = (
        "max_linear_mps", "max_angular_radps", "max_scan_age_sec",
        "max_odom_age_sec", "max_tf_age_sec", "min_obstacle_distance_m",
        "front_obstacle_slow_distance_m", "uncertainty_braking_latency_distance_m",
        "max_linear_accel_mps2", "max_angular_accel_radps2",
        "disable_command_smoothing",
    )
    return {
        "policy": RETURN_TO_START_SPEED_POLICY,
        **{name: getattr(args, name) for name in names},
        "precise_distance_m": RETURN_TO_START_PRECISE_DISTANCE_M,
        "precise_linear_mps": RETURN_TO_START_PRECISE_LINEAR_MPS,
        "precise_angular_radps": RETURN_TO_START_PRECISE_ANGULAR_RADPS,
    }


def validate_return_to_start_speed_evidence(args, permit) -> None:
    fast_requested = (
        getattr(args, "motion_speed_policy", "exploration") == RETURN_TO_START_SPEED_POLICY
    )
    if not fast_requested and getattr(permit, "mission_leg_kind", None) != MissionLegKind.RETURN_TO_START:
        return
    try:
        raw = Path(permit.dry_preflight_path).read_bytes()
    except OSError as exc:
        raise ValueError("fast Start travel dry preflight is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != permit.dry_preflight_sha256:
        raise ValueError("fast Start travel dry preflight hash mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("fast Start travel dry preflight must be an object")
    sealed_policy = payload.get("travel_speed_policy", {})
    if sealed_policy != return_to_start_speed_policy_evidence(args):
        raise ValueError("fast Start travel dry/live speed policy mismatch")


def controller_for_return_to_start_phase(config, *, route_kind, pose, waypoints):
    """Retain precise approach/turn speeds around the goal and real corners.

Collinear sampling vertices do not slow cruise. Runtime route geometry and
the normal certified-corner stop/turn state machine remain authoritative.
"""
    if route_kind != "admitted_candidate_pose" or not waypoints:
        return config
    if (
        len(waypoints) == 2
        and (waypoints[0].x_m, waypoints[0].y_m)
        == (waypoints[1].x_m, waypoints[1].y_m)
    ):
        # This geometry is admitted only by the explicit stationary-turn
        # CSV/diagnostics contract. Never translate to chase odometry drift.
        return replace(
            config,
            max_linear_mps=0.0,
            max_angular_radps=min(config.max_angular_radps, RETURN_TO_START_PRECISE_ANGULAR_RADPS),
        )
    precise_vertices = [waypoints[-1]]
    precise_vertices.extend(
        waypoints[index]
        for index in range(1, len(waypoints) - 1)
        if route_vertex_turn_angle_rad(waypoints, index) >= 0.20
    )
    if any(distance(pose, vertex) <= RETURN_TO_START_PRECISE_DISTANCE_M for vertex in precise_vertices):
        return replace(
            config,
            max_linear_mps=min(config.max_linear_mps, RETURN_TO_START_PRECISE_LINEAR_MPS),
            max_angular_radps=min(config.max_angular_radps, RETURN_TO_START_PRECISE_ANGULAR_RADPS),
        )
    return config
