"""Load immutable route-uncertainty inputs from localization preflight evidence.

This module is deliberately ROS-free.  It binds one successful, persisted
preplanning localization result to the exact planning start, derives the
conservative AMCL covariance envelope, and constructs the admission config
shared by route selectors.  Consumers remain responsible for evaluating a
route and for writing their own decision evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.amcl_covariance_envelope import (
    conservative_amcl_covariance_envelope,
)
from scripts.aufgabe04.navigation.localization.candidate_planning_pose import (
    admitted_candidate_planning_pose,
)


PREFLIGHT_ROUTE_POSE_BASIS = "preflight_route_pose"
COMPOSED_CANDIDATE_POSE_BASIS = "direct_map_from_odom_times_observed_odom_pose"


@dataclass(frozen=True)
class PreflightRouteUncertaintyContext:
    """Content-bound localization inputs for ROS-free route admission."""

    preflight_json: Path
    preflight_sha256: str
    expected_start: Pose2D
    planning_frame: str
    covariance_evidence: Mapping[str, object]
    admission_config: RouteUncertaintyAdmissionConfig
    covariance: PlanarCovariance
    pose_basis: str
    pose_provenance: Mapping[str, object]


def load_preflight_route_uncertainty_context(
    *,
    preflight_json: Path,
    expected_start: Pose2D,
    planning_frame: str,
    robot_radius_m: float,
    collision_margin_m: float,
    tracking_tube_radius_m: float,
    odom_drift_bound_m: float,
    braking_latency_distance_m: float,
    sigma_multiplier: float,
    clearance_sample_spacing_m: float,
    pose_basis: str = PREFLIGHT_ROUTE_POSE_BASIS,
    odom_frame: str | None = None,
) -> PreflightRouteUncertaintyContext:
    """Strictly bind persisted preflight evidence to route-admission inputs.

    Survey routes retain the recorded route pose by default.  Candidate routes
    explicitly select the composed basis used by their planning frame; both
    paths bind the exact start to this one immutable source document.
    """

    preflight_path = Path(preflight_json)
    if preflight_path.is_symlink():
        raise ValueError("route uncertainty preflight path must not be a symlink")
    try:
        raw = preflight_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            "route uncertainty preflight evidence is unavailable: "
            f"{preflight_path}"
        ) from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "route uncertainty preflight evidence is malformed"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise ValueError(
            "route uncertainty selection requires a successful preplanning "
            "localization preflight"
        )

    admitted_pose, pose_provenance = _admitted_pose_evidence(
        payload,
        pose_basis=pose_basis,
        planning_frame=planning_frame,
        odom_frame=odom_frame,
    )
    _validate_preflight_start(
        admitted_pose,
        expected_start=expected_start,
        planning_frame=planning_frame,
    )
    samples = payload.get("stationary_amcl_samples")
    if not isinstance(samples, list) or any(
        not isinstance(sample, Mapping) for sample in samples
    ):
        raise ValueError(
            "route uncertainty preflight AMCL samples are malformed"
        )
    covariance, heading_sigma_rad, covariance_evidence = (
        conservative_amcl_covariance_envelope(samples)
    )
    admission_config = RouteUncertaintyAdmissionConfig(
        robot_radius_m=robot_radius_m,
        collision_margin_m=collision_margin_m,
        fixed_odom_tracking_bound_m=tracking_tube_radius_m,
        empirical_odom_drift_bound_m=odom_drift_bound_m,
        braking_latency_distance_m=braking_latency_distance_m,
        localization_sigma_multiplier=sigma_multiplier,
        heading_sigma_rad=heading_sigma_rad,
        heading_lever_arm_m=robot_radius_m,
        sampling_spacing_m=clearance_sample_spacing_m,
        heading_reference_x_m=expected_start.x_m,
        heading_reference_y_m=expected_start.y_m,
    )
    return PreflightRouteUncertaintyContext(
        preflight_json=preflight_path,
        preflight_sha256=hashlib.sha256(raw).hexdigest(),
        expected_start=expected_start,
        planning_frame=_nonempty_token(planning_frame, "planning_frame"),
        covariance_evidence=covariance_evidence,
        admission_config=admission_config,
        covariance=covariance,
        pose_basis=pose_basis,
        pose_provenance=pose_provenance,
    )


def _admitted_pose_evidence(
    payload: Mapping[str, object],
    *,
    pose_basis: str,
    planning_frame: str,
    odom_frame: str | None,
) -> tuple[object, Mapping[str, object]]:
    if pose_basis == PREFLIGHT_ROUTE_POSE_BASIS:
        route_pose = payload.get("route_pose")
        return route_pose, {
            "pose_basis": pose_basis,
            "route_pose": deepcopy(route_pose),
        }
    if pose_basis != COMPOSED_CANDIDATE_POSE_BASIS:
        raise ValueError("route uncertainty preflight pose basis is unsupported")

    config = payload.get("runtime_config")
    if not isinstance(config, Mapping):
        raise ValueError("route uncertainty preflight has no runtime frame configuration")
    requested_map_frame = _frame_id(planning_frame, "map_frame")
    requested_odom_frame = _frame_id(odom_frame, "odom_frame")
    for requested_frame, name in (
        (requested_map_frame, "map_frame"),
        (requested_odom_frame, "odom_frame"),
    ):
        if requested_frame != _frame_id(config.get(name), f"runtime_config.{name}"):
            raise ValueError(f"route uncertainty preflight {name} configuration mismatch")

    pose, provenance = admitted_candidate_planning_pose(
        payload,
        map_frame=requested_map_frame,
        odom_frame=requested_odom_frame,
    )
    return {
        "frame_id": planning_frame,
        "x_m": pose.x_m,
        "y_m": pose.y_m,
        "yaw_rad": pose.yaw_rad,
    }, provenance


def _frame_id(value: object, name: str) -> str:
    frame = _nonempty_token(value, name).strip().strip("/")
    return _nonempty_token(frame, name)


def _validate_preflight_start(
    value: object,
    *,
    expected_start: Pose2D,
    planning_frame: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(
            "route uncertainty preflight has no admitted route pose"
        )
    if value.get("frame_id") != _nonempty_token(
        planning_frame, "planning_frame"
    ):
        raise ValueError(
            "route uncertainty preflight route-pose frame mismatch"
        )
    try:
        observed = Pose2D(
            float(value["x_m"]),
            float(value["y_m"]),
            float(value["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "route uncertainty preflight route pose is malformed"
        ) from exc
    values = (
        observed.x_m,
        observed.y_m,
        observed.yaw_rad,
        expected_start.x_m,
        expected_start.y_m,
        expected_start.yaw_rad,
    )
    if not all(math.isfinite(item) for item in values):
        raise ValueError(
            "route uncertainty route-pose binding is non-finite"
        )
    if observed != expected_start:
        raise ValueError(
            "route uncertainty preflight route pose does not match "
            "the admitted planning start"
        )


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonempty_token(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "COMPOSED_CANDIDATE_POSE_BASIS",
    "PREFLIGHT_ROUTE_POSE_BASIS",
    "PreflightRouteUncertaintyContext",
    "load_preflight_route_uncertainty_context",
]
