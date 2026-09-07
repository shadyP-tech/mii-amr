"""Load immutable route-uncertainty inputs from localization preflight evidence.

This module is deliberately ROS-free.  It binds one successful, persisted
preplanning localization result to the exact planning start, derives the
conservative AMCL covariance envelope, and constructs the admission config
shared by route selectors.  Consumers remain responsible for evaluating a
route and for writing their own decision evidence.
"""

from __future__ import annotations

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
) -> PreflightRouteUncertaintyContext:
    """Strictly bind persisted preflight evidence to route-admission inputs."""

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

    _validate_preflight_start(
        payload.get("route_pose"),
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
    )


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
    "PreflightRouteUncertaintyContext",
    "load_preflight_route_uncertainty_context",
]
