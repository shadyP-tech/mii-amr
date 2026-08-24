"""ROS-free static-map admission for an uncertainty-aware route.

The global route stays in ``map`` coordinates.  This module samples that
immutable centreline against an *uninflated* static ``Costmap`` and converts
the resulting clearance lower bounds into the explicit terms consumed by
``route_uncertainty_budget``.  It is admission evidence only: it neither
generates velocity commands nor alters a route.

Clearance between adjacent samples is bounded with the 1-Lipschitz property
of distance-to-obstacle: if the largest sample gap is ``d``, the minimum
clearance anywhere between the samples is at least the smallest sampled
clearance minus ``d / 2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    point_clearance_to_blocked_m,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
    RouteClearanceSegment,
    RouteUncertaintyBudgetDecision,
    evaluate_route_uncertainty_budget,
    uncertainty_budget_evidence_sha256,
)


ROUTE_UNCERTAINTY_ADMISSION_SCHEMA_VERSION = 1
_MAX_SAMPLE_INTERVALS_PER_SEGMENT = 1_000_000


@dataclass(frozen=True)
class RouteUncertaintyAdmissionConfig:
    """Immutable physical bounds used for static-route admission.

    ``heading_sigma_rad`` is converted to a conservative linear contribution.
    Without a reference point, the legacy constant lever arm is used.  With a
    reference point, every segment receives its own arm: the farthest endpoint
    distance from that frozen reference plus the robot radius, never less than
    ``heading_lever_arm_m``.  This avoids charging a distant route endpoint's
    yaw displacement to an unrelated short start connector.  The multiplier
    is evidence policy, not a probability guarantee.
    """

    robot_radius_m: float
    collision_margin_m: float
    fixed_odom_tracking_bound_m: float
    empirical_odom_drift_bound_m: float
    braking_latency_distance_m: float
    localization_sigma_multiplier: float
    heading_sigma_rad: float
    heading_lever_arm_m: float
    sampling_spacing_m: float
    heading_reference_x_m: float | None = None
    heading_reference_y_m: float | None = None

    def __post_init__(self) -> None:
        values = {
            name: _strict_finite_number(getattr(self, name), name)
            for name in (
                "robot_radius_m",
                "collision_margin_m",
                "fixed_odom_tracking_bound_m",
                "empirical_odom_drift_bound_m",
                "braking_latency_distance_m",
                "localization_sigma_multiplier",
                "heading_sigma_rad",
                "heading_lever_arm_m",
                "sampling_spacing_m",
            )
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

        for name in (
            "robot_radius_m",
            "localization_sigma_multiplier",
            "sampling_spacing_m",
        ):
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "collision_margin_m",
            "fixed_odom_tracking_bound_m",
            "empirical_odom_drift_bound_m",
            "braking_latency_distance_m",
            "heading_sigma_rad",
            "heading_lever_arm_m",
        ):
            if values[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")

        heading_contribution = (
            values["localization_sigma_multiplier"]
            * values["heading_sigma_rad"]
            * values["heading_lever_arm_m"]
        )
        if not math.isfinite(heading_contribution):
            raise ValueError("heading uncertainty contribution must be finite")
        reference_values = (
            self.heading_reference_x_m,
            self.heading_reference_y_m,
        )
        if (reference_values[0] is None) != (reference_values[1] is None):
            raise ValueError(
                "heading reference x/y must be provided together"
            )
        if reference_values[0] is not None:
            reference_x = _strict_finite_number(
                reference_values[0], "heading_reference_x_m"
            )
            reference_y = _strict_finite_number(
                reference_values[1], "heading_reference_y_m"
            )
            object.__setattr__(self, "heading_reference_x_m", reference_x)
            object.__setattr__(self, "heading_reference_y_m", reference_y)

    @property
    def heading_contribution_m(self) -> float:
        return (
            self.localization_sigma_multiplier
            * self.heading_sigma_rad
            * self.heading_lever_arm_m
        )

    def to_evidence_dict(self) -> dict[str, float | None]:
        return {
            "robot_radius_m": self.robot_radius_m,
            "collision_margin_m": self.collision_margin_m,
            "fixed_odom_tracking_bound_m": self.fixed_odom_tracking_bound_m,
            "empirical_odom_drift_bound_m": self.empirical_odom_drift_bound_m,
            "braking_latency_distance_m": self.braking_latency_distance_m,
            "localization_sigma_multiplier": self.localization_sigma_multiplier,
            "heading_sigma_rad": self.heading_sigma_rad,
            "heading_lever_arm_m": self.heading_lever_arm_m,
            "heading_contribution_m": self.heading_contribution_m,
            "sampling_spacing_m": self.sampling_spacing_m,
            "heading_reference_x_m": self.heading_reference_x_m,
            "heading_reference_y_m": self.heading_reference_y_m,
        }


@dataclass(frozen=True)
class RouteUncertaintyAdmissionResult:
    """Sampled clearance profile plus the delegated budget decision."""

    segments: tuple[RouteClearanceSegment, ...]
    decision: RouteUncertaintyBudgetDecision
    evidence: dict[str, object]

    def to_evidence_dict(self) -> dict[str, object]:
        return dict(self.evidence)


@dataclass(frozen=True)
class _SampledRouteSegment:
    route_index: int
    start: Pose2D
    end: Pose2D
    length_m: float
    normal_x: float
    normal_y: float
    interval_count: int
    actual_spacing_m: float
    samples: tuple[tuple[float, float, float], ...]
    minimum_sampled_clearance_m: float
    lipschitz_deduction_m: float
    clearance_lower_bound_m: float


def evaluate_route_uncertainty_admission(
    costmap: Costmap,
    map_route: Sequence[Pose2D],
    covariance: PlanarCovariance,
    config: RouteUncertaintyAdmissionConfig,
) -> RouteUncertaintyAdmissionResult:
    """Build a static clearance profile and require every budget to pass.

    Invalid or ambiguous route/map inputs return a rejected aggregate decision
    with finite deterministic evidence.  A malformed config is rejected by
    ``RouteUncertaintyAdmissionConfig`` before this function is called.
    """

    validation_errors: list[str] = []
    if not isinstance(config, RouteUncertaintyAdmissionConfig):
        validation_errors.append("admission_config_missing_or_ambiguous")
    costmap_for_evidence: Costmap | None = None
    if not isinstance(costmap, Costmap):
        validation_errors.append("costmap_missing_or_ambiguous")
    else:
        costmap_errors = _validate_costmap_geometry(costmap)
        validation_errors.extend(costmap_errors)
        if not costmap_errors:
            costmap_for_evidence = costmap
    if not isinstance(covariance, PlanarCovariance):
        validation_errors.append("planar_covariance_missing_or_ambiguous")

    poses = _route_poses_or_empty(map_route, validation_errors)
    sampled_segments: list[_SampledRouteSegment] = []
    if not validation_errors:
        assert isinstance(costmap, Costmap)
        assert isinstance(config, RouteUncertaintyAdmissionConfig)
        sampled_segments, sampling_error = _sample_route_segments(
            costmap, poses, config.sampling_spacing_m
        )
        if sampling_error is not None:
            validation_errors.append(sampling_error)

    if validation_errors:
        segments: tuple[RouteClearanceSegment, ...] = ()
    else:
        assert isinstance(covariance, PlanarCovariance)
        assert isinstance(config, RouteUncertaintyAdmissionConfig)
        segments = _budget_segments(sampled_segments, covariance, config)

    decision = evaluate_route_uncertainty_budget(segments)
    evidence = _admission_evidence(
        costmap=costmap_for_evidence,
        poses=poses,
        covariance=covariance if isinstance(covariance, PlanarCovariance) else None,
        config=(
            config if isinstance(config, RouteUncertaintyAdmissionConfig) else None
        ),
        sampled_segments=sampled_segments,
        segments=segments,
        decision=decision,
        validation_errors=tuple(dict.fromkeys(validation_errors)),
    )
    return RouteUncertaintyAdmissionResult(
        segments=segments,
        decision=decision,
        evidence=evidence,
    )


def route_uncertainty_admission_evidence_sha256(
    value: RouteUncertaintyAdmissionResult | Mapping[str, object],
) -> str:
    """Hash the complete clearance-profile evidence canonically."""

    if isinstance(value, RouteUncertaintyAdmissionResult):
        evidence = value.evidence
    elif isinstance(value, Mapping):
        evidence = value
    else:
        raise ValueError("admission evidence must be a result or mapping")
    return payload_sha256(evidence)


def _route_poses_or_empty(
    map_route: object,
    errors: list[str],
) -> tuple[Pose2D, ...]:
    if isinstance(map_route, (str, bytes)) or not isinstance(map_route, Sequence):
        errors.append("map_route_missing_or_ambiguous")
        return ()
    poses = tuple(map_route)
    if len(poses) < 2:
        errors.append("map_route_requires_at_least_two_poses")
    for index, pose in enumerate(poses):
        if not isinstance(pose, Pose2D):
            errors.append(f"map_route_pose_{index}_missing_or_ambiguous")
            continue
        if not all(_is_finite_number(value) for value in (pose.x_m, pose.y_m)):
            errors.append(f"map_route_pose_{index}_nonfinite")
        if not (
            _is_finite_number(pose.yaw_rad)
            or _is_nan_number(pose.yaw_rad)
        ):
            errors.append(f"map_route_pose_{index}_yaw_invalid")
    if any(not isinstance(pose, Pose2D) for pose in poses):
        return ()
    return poses


def _validate_costmap_geometry(costmap: Costmap) -> tuple[str, ...]:
    errors: list[str] = []
    if (
        not isinstance(costmap.width, int)
        or isinstance(costmap.width, bool)
        or costmap.width <= 0
        or not isinstance(costmap.height, int)
        or isinstance(costmap.height, bool)
        or costmap.height <= 0
    ):
        errors.append("costmap_dimensions_invalid")
    if not _is_finite_number(costmap.resolution) or costmap.resolution <= 0.0:
        errors.append("costmap_resolution_invalid")
    origin = costmap.metadata.origin
    if len(origin) != 3 or not all(_is_finite_number(value) for value in origin):
        errors.append("costmap_origin_invalid")
    if len(costmap.cells) != costmap.height or any(
        len(row) != costmap.width for row in costmap.cells
    ):
        errors.append("costmap_cells_shape_invalid")
    if any(not costmap.in_bounds(cell) for cell in costmap.blocked_cells):
        errors.append("costmap_blocked_cell_out_of_bounds")
    return tuple(errors)


def _sample_route_segments(
    costmap: Costmap,
    poses: tuple[Pose2D, ...],
    maximum_spacing_m: float,
) -> tuple[list[_SampledRouteSegment], str | None]:
    result: list[_SampledRouteSegment] = []
    for index, (start, end) in enumerate(zip(poses, poses[1:])):
        dx = end.x_m - start.x_m
        dy = end.y_m - start.y_m
        length_m = math.hypot(dx, dy)
        if not math.isfinite(length_m):
            return [], f"map_route_segment_{index}_length_nonfinite"
        if length_m <= 0.0:
            return [], f"map_route_segment_{index}_zero_length_ambiguous"

        ratio = length_m / maximum_spacing_m
        if not math.isfinite(ratio):
            return [], f"map_route_segment_{index}_sample_count_nonfinite"
        interval_count = max(1, int(math.ceil(ratio)))
        if interval_count > _MAX_SAMPLE_INTERVALS_PER_SEGMENT:
            return [], f"map_route_segment_{index}_sample_count_excessive"
        actual_spacing_m = length_m / interval_count
        samples: list[tuple[float, float, float]] = []
        try:
            for sample_index in range(interval_count + 1):
                fraction = sample_index / interval_count
                x_m = start.x_m + fraction * dx
                y_m = start.y_m + fraction * dy
                clearance_m = point_clearance_to_blocked_m(
                    costmap, Pose2D(x_m=x_m, y_m=y_m)
                )
                if not math.isfinite(clearance_m) or clearance_m < 0.0:
                    return [], (
                        f"map_route_segment_{index}_sample_clearance_invalid"
                    )
                samples.append((x_m, y_m, clearance_m))
        except (TypeError, ValueError, OverflowError):
            return [], f"map_route_segment_{index}_sampling_failed"

        minimum_sampled = min(item[2] for item in samples)
        lipschitz_deduction = actual_spacing_m / 2.0
        clearance_lower_bound = max(
            0.0, minimum_sampled - lipschitz_deduction
        )
        result.append(
            _SampledRouteSegment(
                route_index=index,
                start=start,
                end=end,
                length_m=length_m,
                normal_x=-dy / length_m,
                normal_y=dx / length_m,
                interval_count=interval_count,
                actual_spacing_m=actual_spacing_m,
                samples=tuple(samples),
                minimum_sampled_clearance_m=minimum_sampled,
                lipschitz_deduction_m=lipschitz_deduction,
                clearance_lower_bound_m=clearance_lower_bound,
            )
        )
    return result, None


def _budget_segments(
    sampled: Sequence[_SampledRouteSegment],
    covariance: PlanarCovariance,
    config: RouteUncertaintyAdmissionConfig,
) -> tuple[RouteClearanceSegment, ...]:
    result: list[RouteClearanceSegment] = []
    for index, item in enumerate(sampled):
        segment_heading_contribution_m = _heading_contribution_for_points(
            (item.start, item.end),
            config,
        )
        result.append(
            _budget_segment(
                segment_id=f"segment:{item.route_index:04d}",
                raw_clearance_m=item.clearance_lower_bound_m,
                normal_x=item.normal_x,
                normal_y=item.normal_y,
                is_corner=False,
                covariance=covariance,
                config=config,
                heading_contribution_m=segment_heading_contribution_m,
            )
        )
        if index + 1 < len(sampled):
            following = sampled[index + 1]
            result.append(
                _budget_segment(
                    segment_id=f"corner:{index + 1:04d}",
                    raw_clearance_m=min(
                        item.clearance_lower_bound_m,
                        following.clearance_lower_bound_m,
                    ),
                    # The budget validates a normal even in corner mode.  It
                    # then deliberately ignores this direction and uses the
                    # covariance's largest eigen-axis.
                    normal_x=item.normal_x,
                    normal_y=item.normal_y,
                    is_corner=True,
                    covariance=covariance,
                    config=config,
                    heading_contribution_m=(
                        _heading_contribution_for_points(
                            (item.end,),
                            config,
                        )
                    ),
                )
            )
    return tuple(result)


def _budget_segment(
    *,
    segment_id: str,
    raw_clearance_m: float,
    normal_x: float,
    normal_y: float,
    is_corner: bool,
    covariance: PlanarCovariance,
    config: RouteUncertaintyAdmissionConfig,
    heading_contribution_m: float,
) -> RouteClearanceSegment:
    return RouteClearanceSegment(
        segment_id=segment_id,
        raw_centerline_clearance_m=raw_clearance_m,
        robot_radius_m=config.robot_radius_m,
        collision_margin_m=config.collision_margin_m,
        fixed_odom_tracking_bound_m=config.fixed_odom_tracking_bound_m,
        empirical_odom_drift_bound_m=config.empirical_odom_drift_bound_m,
        braking_latency_distance_m=config.braking_latency_distance_m,
        localization_sigma_multiplier=config.localization_sigma_multiplier,
        heading_contribution_m=heading_contribution_m,
        covariance=covariance,
        segment_normal_x=normal_x,
        segment_normal_y=normal_y,
        is_corner=is_corner,
    )


def _heading_contribution_for_points(
    points: Sequence[Pose2D],
    config: RouteUncertaintyAdmissionConfig,
) -> float:
    if config.heading_reference_x_m is None:
        lever_arm_m = config.heading_lever_arm_m
    else:
        assert config.heading_reference_y_m is not None
        lever_arm_m = max(
            config.heading_lever_arm_m,
            max(
                math.hypot(
                    point.x_m - config.heading_reference_x_m,
                    point.y_m - config.heading_reference_y_m,
                )
                + config.robot_radius_m
                for point in points
            ),
        )
    contribution_m = (
        config.localization_sigma_multiplier
        * config.heading_sigma_rad
        * lever_arm_m
    )
    if not math.isfinite(contribution_m):
        raise ValueError("segment heading contribution must be finite")
    return contribution_m


def _admission_evidence(
    *,
    costmap: Costmap | None,
    poses: Sequence[Pose2D],
    covariance: PlanarCovariance | None,
    config: RouteUncertaintyAdmissionConfig | None,
    sampled_segments: Sequence[_SampledRouteSegment],
    segments: Sequence[RouteClearanceSegment],
    decision: RouteUncertaintyBudgetDecision,
    validation_errors: Sequence[str],
) -> dict[str, object]:
    route_payload = {
        "frame": "map",
        "poses": [
            _route_pose_evidence(pose)
            for pose in poses
        ],
    }
    return {
        "schema_version": ROUTE_UNCERTAINTY_ADMISSION_SCHEMA_VERSION,
        "scope": {
            "admission_only": True,
            "generates_commands": False,
            "mutates_route": False,
            "probability_guarantee": False,
            "clearance_costmap_contract": "uninflated_static_map",
        },
        "validation": {
            "ok": not validation_errors,
            "errors": list(validation_errors),
        },
        "costmap": _costmap_evidence(costmap),
        "route": {
            **route_payload,
            "route_sha256": payload_sha256(route_payload),
        },
        "covariance_m2": (
            covariance.to_evidence_dict() if covariance is not None else None
        ),
        "config": config.to_evidence_dict() if config is not None else None,
        "sampling": {
            "method": "endpoint_inclusive_equal_spacing",
            "clearance_lower_bound": (
                "minimum_sampled_clearance_m - actual_spacing_m / 2"
            ),
            "distance_property": "1_lipschitz",
            "segments": [_sampled_segment_evidence(item) for item in sampled_segments],
        },
        "budget_profile": [
            {
                "profile_index": index,
                "segment_id": segment.segment_id,
                "raw_centerline_clearance_m": (
                    segment.raw_centerline_clearance_m
                ),
                "segment_normal": {
                    "x": segment.segment_normal_x,
                    "y": segment.segment_normal_y,
                },
                "is_corner": segment.is_corner,
                "heading_contribution_m": segment.heading_contribution_m,
            }
            for index, segment in enumerate(segments)
        ],
        "decision": decision.to_evidence_dict(),
        "decision_evidence_sha256": uncertainty_budget_evidence_sha256(decision),
    }


def _costmap_evidence(costmap: Costmap | None) -> dict[str, object] | None:
    if costmap is None:
        return None
    blocked_geometry = {
        "width": costmap.width,
        "height": costmap.height,
        "resolution_m": costmap.resolution,
        "origin": list(costmap.metadata.origin),
        "blocked_cells": [
            {"x": cell.x, "y": cell.y}
            for cell in sorted(costmap.blocked_cells, key=lambda cell: (cell.y, cell.x))
        ],
    }
    source_counts: dict[str, int] = {}
    for cell in costmap.blocked_cells:
        source = costmap.cell_sources.get(cell, "unspecified")
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "width": costmap.width,
        "height": costmap.height,
        "resolution_m": costmap.resolution,
        "origin": list(costmap.metadata.origin),
        "blocked_cell_count": len(costmap.blocked_cells),
        "blocked_source_counts": dict(sorted(source_counts.items())),
        "blocked_geometry_sha256": payload_sha256(blocked_geometry),
    }


def _sampled_segment_evidence(item: _SampledRouteSegment) -> dict[str, object]:
    return {
        "route_index": item.route_index,
        "start": {"x_m": item.start.x_m, "y_m": item.start.y_m},
        "end": {"x_m": item.end.x_m, "y_m": item.end.y_m},
        "length_m": item.length_m,
        "segment_normal": {"x": item.normal_x, "y": item.normal_y},
        "interval_count": item.interval_count,
        "sample_count": len(item.samples),
        "actual_spacing_m": item.actual_spacing_m,
        "samples": [
            {"x_m": x_m, "y_m": y_m, "clearance_m": clearance_m}
            for x_m, y_m, clearance_m in item.samples
        ],
        "minimum_sampled_clearance_m": item.minimum_sampled_clearance_m,
        "lipschitz_deduction_m": item.lipschitz_deduction_m,
        "clearance_lower_bound_m": item.clearance_lower_bound_m,
    }


def _route_pose_evidence(pose: Pose2D) -> dict[str, object]:
    if _is_nan_number(pose.yaw_rad):
        yaw_rad: float | None = None
        yaw_mode = "unconstrained_nan"
    elif _is_finite_number(pose.yaw_rad):
        yaw_rad = float(pose.yaw_rad)
        yaw_mode = "constrained"
    else:
        yaw_rad = None
        yaw_mode = "invalid_nonfinite"
    return {
        "x_m": _finite_number_or_none(pose.x_m),
        "y_m": _finite_number_or_none(pose.y_m),
        "yaw_rad": yaw_rad,
        "yaw_mode": yaw_mode,
    }


def _strict_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _is_nan_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_number_or_none(value: object) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


__all__ = [
    "ROUTE_UNCERTAINTY_ADMISSION_SCHEMA_VERSION",
    "RouteUncertaintyAdmissionConfig",
    "RouteUncertaintyAdmissionResult",
    "evaluate_route_uncertainty_admission",
    "route_uncertainty_admission_evidence_sha256",
]
