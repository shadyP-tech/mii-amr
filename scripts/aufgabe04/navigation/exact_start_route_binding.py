"""Strictly bind exact-start clearance evidence to persisted route vertices."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


RouteXY = Sequence[tuple[float, float]]


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{name} must be finite")
    return selected


def _metadata_xy(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return (
        _finite_float(value.get("x_m"), f"{name}.x_m"),
        _finite_float(value.get("y_m"), f"{name}.y_m"),
    )


def _validated_route_xy(route_xy: RouteXY) -> tuple[tuple[float, float], ...]:
    selected = []
    for index, point in enumerate(route_xy):
        if not isinstance(point, tuple) or len(point) != 2:
            raise ValueError(f"route waypoint {index} must be an (x, y) tuple")
        x_m = _finite_float(point[0], f"route waypoint {index}.x_m")
        y_m = _finite_float(point[1], f"route waypoint {index}.y_m")
        selected.append((x_m, y_m))
    if not selected:
        raise ValueError("exact-start route is empty")
    return tuple(selected)


def validate_exact_start_route_binding(
    metadata: Mapping[str, object],
    route_xy: RouteXY,
    *,
    tolerance_m: float = 1.0e-7,
) -> None:
    """Raise unless connector evidence is bound to route vertices and inflation.

    The validator intentionally does not reconstruct clearance from a map.  It
    establishes that the immutable route has not drifted from the persisted
    continuous-clearance proof that was created with the planning map.
    """

    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("exact-start binding tolerance must be non-negative")
    points = _validated_route_xy(route_xy)
    connector = metadata.get("exact_start_connector")
    if not isinstance(connector, Mapping):
        raise ValueError("source route lacks exact-start connector evidence")
    if connector.get("validated") is not True:
        raise ValueError("exact-start connector was not validated")
    required = connector.get("required")
    if not isinstance(required, bool):
        raise ValueError("exact-start connector required flag must be boolean")

    exact_xy = _metadata_xy(
        connector.get("exact_start"),
        "exact_start_connector.exact_start",
    )
    anchor_xy = _metadata_xy(
        connector.get("anchor"),
        "exact_start_connector.anchor",
    )
    anchor_index = 1 if required else 0
    if anchor_index >= len(points):
        raise ValueError("exact-start connector has no bound anchor waypoint")
    if (
        math.hypot(
            exact_xy[0] - points[0][0],
            exact_xy[1] - points[0][1],
        )
        > tolerance_m
    ):
        raise ValueError("exact-start evidence differs from route waypoint 0")
    if (
        math.hypot(
            anchor_xy[0] - points[anchor_index][0],
            anchor_xy[1] - points[anchor_index][1],
        )
        > tolerance_m
    ):
        raise ValueError("exact-start anchor differs from its route waypoint")

    length_m = math.hypot(anchor_xy[0] - exact_xy[0], anchor_xy[1] - exact_xy[1])
    recorded_length_m = _finite_float(
        connector.get("connector_length_m"),
        "exact_start_connector.connector_length_m",
    )
    if not math.isclose(
        length_m,
        recorded_length_m,
        rel_tol=0.0,
        abs_tol=tolerance_m,
    ):
        raise ValueError("exact-start connector length differs from route geometry")
    if required != (length_m > 1.0e-9):
        raise ValueError("exact-start required flag differs from route geometry")

    required_clearance_m = _finite_float(
        connector.get("required_clearance_m"),
        "exact_start_connector.required_clearance_m",
    )
    inflation_m = _finite_float(
        metadata.get("inflation_radius_m"),
        "inflation_radius_m",
    )
    if required_clearance_m < 0.0 or inflation_m < 0.0:
        raise ValueError("exact-start clearance and route inflation must be non-negative")
    if not math.isclose(
        required_clearance_m,
        inflation_m,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("exact-start required clearance differs from route inflation")

    continuous_m = _finite_float(
        connector.get("minimum_continuous_clearance_m"),
        "exact_start_connector.minimum_continuous_clearance_m",
    )
    margin_m = _finite_float(
        connector.get("minimum_margin_m"),
        "exact_start_connector.minimum_margin_m",
    )
    if continuous_m <= required_clearance_m or margin_m <= 0.0:
        raise ValueError("exact-start continuous-clearance margin is not positive")
    if not math.isclose(
        continuous_m - required_clearance_m,
        margin_m,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("exact-start clearance margin is internally inconsistent")
    sample_count = connector.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError("exact-start connector requires at least two clearance samples")
    sample_spacing_m = _finite_float(
        connector.get("sample_spacing_m"),
        "exact_start_connector.sample_spacing_m",
    )
    sampled_m = _finite_float(
        connector.get("minimum_sampled_clearance_m"),
        "exact_start_connector.minimum_sampled_clearance_m",
    )
    if sample_spacing_m < 0.0 or sampled_m < 0.0:
        raise ValueError("exact-start sampled-clearance evidence must be non-negative")
    if not math.isclose(
        sample_spacing_m * (sample_count - 1),
        length_m,
        rel_tol=0.0,
        abs_tol=tolerance_m,
    ):
        raise ValueError("exact-start sample spacing differs from connector geometry")
    expected_continuous_m = max(0.0, sampled_m - sample_spacing_m / 2.0)
    if not math.isclose(
        expected_continuous_m,
        continuous_m,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("exact-start sampled-clearance evidence is inconsistent")
