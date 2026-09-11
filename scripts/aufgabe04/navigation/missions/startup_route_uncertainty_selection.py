"""Pre-checkpoint adapter for uncertainty-aware startup route selection.

The adapter binds pure route-option ranking to one already persisted,
successful stationary localization preflight.  It writes a content-hashed
selection receipt before returning a target or reporting that every option
was rejected.  It has no ROS or command-publication dependency and must only
be used before a survey target is committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_selection import (
    RouteUncertaintySelectionOption,
    evaluate_route_uncertainty_selection,
    route_uncertainty_selection_evidence_sha256,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.preflight_route_uncertainty_context import (
    PREFLIGHT_ROUTE_POSE_BASIS,
    load_preflight_route_uncertainty_context,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap


STARTUP_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION = 1
STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD = (
    "startup_route_uncertainty_selection_sha256"
)


class StartupRouteUncertaintySelectionRejected(ValueError):
    """Every reachable startup option failed exact uncertainty admission."""

    def __init__(
        self,
        *,
        evidence_path: Path,
        evidence_sha256: str,
        reason: str,
    ) -> None:
        self.evidence_path = Path(evidence_path)
        self.evidence_sha256 = evidence_sha256
        self.reason = reason
        super().__init__(
            "startup route uncertainty selection rejected every reachable "
            f"option: {reason}; evidence={self.evidence_path}; "
            f"sha256={evidence_sha256}"
        )


@dataclass(frozen=True)
class StartupRouteUncertaintySelector:
    """Callable selection policy bound to one stationary AMCL preflight."""

    preflight_json: Path
    preflight_sha256: str
    evidence_json: Path
    expected_start: Pose2D
    planning_frame: str
    covariance_evidence: Mapping[str, object]
    admission_config: RouteUncertaintyAdmissionConfig
    covariance: PlanarCovariance

    def __call__(
        self,
        base_costmap: Costmap,
        legs: Sequence[object],
    ) -> tuple[str, Mapping[str, object]]:
        if not isinstance(base_costmap, Costmap):
            raise ValueError("startup route selector requires a base Costmap")
        frozen_legs = tuple(legs)
        if not frozen_legs:
            raise ValueError("startup route selector requires reachable legs")

        options: list[RouteUncertaintySelectionOption] = []
        for plan_order, leg in enumerate(frozen_legs):
            try:
                viewpoint_id = leg.viewpoint.viewpoint_id
                route = leg.route_result.route
            except AttributeError as exc:
                raise ValueError(
                    "startup route selector received a malformed survey leg"
                ) from exc
            if route is None:
                raise ValueError(
                    "startup route selector received a leg without a route"
                )
            try:
                points = route.points
            except AttributeError as exc:
                raise ValueError(
                    "startup route selector received malformed route points"
                ) from exc
            if not isinstance(viewpoint_id, str) or not viewpoint_id.strip():
                raise ValueError(
                    "startup route selector received an invalid viewpoint ID"
                )
            map_route = tuple(point.pose for point in points)
            options.append(
                RouteUncertaintySelectionOption(
                    option_id=viewpoint_id,
                    plan_order=plan_order,
                    map_route=map_route,
                )
            )

        decision = evaluate_route_uncertainty_selection(
            base_costmap,
            tuple(options),
            self.covariance,
            self.admission_config,
        )
        selection_evidence_sha256 = (
            route_uncertainty_selection_evidence_sha256(decision)
        )
        payload = {
            "schema_version": (
                STARTUP_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION
            ),
            "phase": "precheckpoint_initial_coverage_route_selection",
            "source_preplanning_localization_json": str(self.preflight_json),
            "source_preplanning_localization_sha256": self.preflight_sha256,
            "planning_frame": self.planning_frame,
            "admitted_start_pose": {
                "x_m": self.expected_start.x_m,
                "y_m": self.expected_start.y_m,
                "yaw_rad": self.expected_start.yaw_rad,
            },
            "covariance_envelope": dict(self.covariance_evidence),
            "admission_config": self.admission_config.to_evidence_dict(),
            "selection_evidence_sha256": selection_evidence_sha256,
            "selection": decision.to_evidence_dict(),
            "target_committed_before_selection": False,
            "retargeting_allowed_after_selection": False,
            "motion_authorized": False,
            "motion_published": False,
        }
        artifact_sha256 = write_content_hashed_json(
            self.evidence_json,
            payload,
            hash_field=STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
        )
        if not decision.ready or decision.selected_option_id is None:
            raise StartupRouteUncertaintySelectionRejected(
                evidence_path=self.evidence_json,
                evidence_sha256=artifact_sha256,
                reason=decision.reason,
            )

        selected = decision.ranked_options[0]
        return (
            decision.selected_option_id,
            {
                "policy": "exact_route_uncertainty_margin",
                "startup_route_uncertainty_selection_json": str(
                    self.evidence_json
                ),
                "startup_route_uncertainty_selection_sha256": (
                    artifact_sha256
                ),
                "selection_evidence_sha256": selection_evidence_sha256,
                "selected_viewpoint_id": decision.selected_option_id,
                "selected_minimum_remaining_margin_m": (
                    selected.minimum_remaining_margin_m
                ),
                "evaluated_option_count": len(decision.ranked_options),
                "motion_authorized": False,
            },
        )


def load_startup_route_uncertainty_selector(
    *,
    preflight_json: Path,
    evidence_json: Path,
    expected_start: Pose2D,
    planning_frame: str,
    robot_radius_m: float,
    collision_margin_m: float,
    tracking_tube_radius_m: float,
    odom_drift_bound_m: float,
    braking_latency_distance_m: float,
    sigma_multiplier: float,
    clearance_sample_spacing_m: float,
) -> StartupRouteUncertaintySelector:
    """Load and strictly bind the selector to preplanning evidence."""

    evidence_path = Path(evidence_json)
    context = load_preflight_route_uncertainty_context(
        preflight_json=preflight_json,
        expected_start=expected_start,
        planning_frame=planning_frame,
        pose_basis=PREFLIGHT_ROUTE_POSE_BASIS,
        robot_radius_m=robot_radius_m,
        collision_margin_m=collision_margin_m,
        tracking_tube_radius_m=tracking_tube_radius_m,
        odom_drift_bound_m=odom_drift_bound_m,
        braking_latency_distance_m=braking_latency_distance_m,
        sigma_multiplier=sigma_multiplier,
        clearance_sample_spacing_m=clearance_sample_spacing_m,
    )
    return StartupRouteUncertaintySelector(
        preflight_json=context.preflight_json,
        preflight_sha256=context.preflight_sha256,
        evidence_json=evidence_path,
        expected_start=context.expected_start,
        planning_frame=context.planning_frame,
        covariance_evidence=context.covariance_evidence,
        admission_config=context.admission_config,
        covariance=context.covariance,
    )


__all__ = [
    "STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD",
    "STARTUP_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION",
    "StartupRouteUncertaintySelectionRejected",
    "StartupRouteUncertaintySelector",
    "load_startup_route_uncertainty_selector",
]
