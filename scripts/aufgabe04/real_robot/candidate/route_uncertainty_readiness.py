"""Bind stopped candidate-selection preflight to pure route admission.

This adapter contains no ROS and publishes no commands.  The autonomous
runner first persists the stationary localization preflight; this module then
loads that immutable artifact with the same physical constants used by the
station-segment child and returns a motion-neutral selection context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_defaults import (
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_TRACKING_TUBE_RADIUS_M,
    DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
    DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
    DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.preflight_route_uncertainty_context import (
    COMPOSED_CANDIDATE_POSE_BASIS,
    load_preflight_route_uncertainty_context,
)


@dataclass(frozen=True)
class CandidateRouteUncertaintyReadinessRequest:
    """Exact stopped epoch and policy inputs for candidate route selection."""

    preflight_json: Path
    expected_start: Pose2D
    planning_frame: str
    odom_frame: str
    robot_radius_m: float
    sigma_multiplier: float


def load_candidate_route_uncertainty_readiness(
    request: CandidateRouteUncertaintyReadinessRequest,
) -> CandidateRouteUncertaintyContext:
    """Return the exact child-parity uncertainty envelope for selection."""

    if not isinstance(request, CandidateRouteUncertaintyReadinessRequest):
        raise TypeError(
            "candidate route uncertainty readiness request has the wrong type"
        )
    context = load_preflight_route_uncertainty_context(
        preflight_json=request.preflight_json,
        expected_start=request.expected_start,
        planning_frame=request.planning_frame,
        odom_frame=request.odom_frame,
        pose_basis=COMPOSED_CANDIDATE_POSE_BASIS,
        robot_radius_m=request.robot_radius_m,
        collision_margin_m=DEFAULT_COLLISION_MARGIN_M,
        tracking_tube_radius_m=DEFAULT_TRACKING_TUBE_RADIUS_M,
        odom_drift_bound_m=DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
        braking_latency_distance_m=(
            DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M
        ),
        sigma_multiplier=request.sigma_multiplier,
        clearance_sample_spacing_m=(
            DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M
        ),
    )
    return CandidateRouteUncertaintyContext(
        covariance=context.covariance,
        admission_config=context.admission_config,
        source_evidence={
            "source_preplanning_localization_json": str(
                context.preflight_json
            ),
            "source_preplanning_localization_sha256": (
                context.preflight_sha256
            ),
            "planning_frame": context.planning_frame,
            "pose_basis": context.pose_basis,
            "pose_provenance": dict(context.pose_provenance),
            "admitted_start_pose": {
                "x_m": context.expected_start.x_m,
                "y_m": context.expected_start.y_m,
                "yaw_rad": context.expected_start.yaw_rad,
            },
            "covariance_envelope": dict(context.covariance_evidence),
            "child_budget_defaults_shared": True,
            "selection_only": True,
            "motion_authorized": False,
        },
    )


__all__ = [
    "CandidateRouteUncertaintyReadinessRequest",
    "load_candidate_route_uncertainty_readiness",
]
