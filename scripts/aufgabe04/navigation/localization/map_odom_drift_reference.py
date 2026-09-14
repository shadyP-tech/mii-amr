"""Immutable, route-bound reference for map/odom displacement.

This value has no ROS or certificate-loader dependencies. A certificate binds
it to its first route point and frozen transform; a route revision keeps that
same reference so its yaw lever arms remain covered by the original budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


ROUTE_ANCHOR_METRIC = "certified_route_anchor_v1"
_FIELDS = frozenset({"metric", "map_anchor", "odom_anchor"})
_POINT_FIELDS = frozenset({"x_m", "y_m"})


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("drift reference coordinates must be finite numbers")
    return 0.0 if value == 0 else float(value)


def _point(value: object) -> tuple[float, float]:
    if not isinstance(value, Mapping) or set(value) != _POINT_FIELDS:
        raise ValueError("drift reference point fields mismatch")
    return _finite(value["x_m"]), _finite(value["y_m"])


@dataclass(frozen=True)
class RouteDriftAnchor:
    map_x_m: float
    map_y_m: float
    odom_x_m: float
    odom_y_m: float

    def __post_init__(self) -> None:
        for name in ("map_x_m", "map_y_m", "odom_x_m", "odom_y_m"):
            object.__setattr__(self, name, _finite(getattr(self, name)))

    @classmethod
    def from_route_start(cls, pose_map, frozen_transform) -> RouteDriftAnchor:
        x_m, y_m = _finite(pose_map.x_m), _finite(pose_map.y_m)
        tx, ty, yaw = (_finite(getattr(frozen_transform, name)) for name in ("x_m", "y_m", "yaw_rad"))
        dx, dy = x_m - tx, y_m - ty
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return cls(x_m, y_m, cosine * dx + sine * dy, -sine * dx + cosine * dy)

    @classmethod
    def from_evidence(cls, value: object) -> RouteDriftAnchor:
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("drift reference fields mismatch")
        if value["metric"] != ROUTE_ANCHOR_METRIC:
            raise ValueError("unsupported map/odom drift metric")
        return cls(*_point(value["map_anchor"]), *_point(value["odom_anchor"]))

    def to_evidence(self) -> dict[str, object]:
        # Reconstruct to detect corruption of a frozen value before publication.
        anchor = RouteDriftAnchor(self.map_x_m, self.map_y_m, self.odom_x_m, self.odom_y_m)
        return {
            "metric": ROUTE_ANCHOR_METRIC,
            "map_anchor": {"x_m": anchor.map_x_m, "y_m": anchor.map_y_m},
            "odom_anchor": {"x_m": anchor.odom_x_m, "y_m": anchor.odom_y_m},
        }

    def validate_transform(self, frozen_transform) -> None:
        anchor = RouteDriftAnchor.from_evidence(self.to_evidence())
        tx, ty, yaw = (_finite(getattr(frozen_transform, name)) for name in ("x_m", "y_m", "yaw_rad"))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        expected = (
            cosine * anchor.odom_x_m - sine * anchor.odom_y_m + tx,
            sine * anchor.odom_x_m + cosine * anchor.odom_y_m + ty,
        )
        # Numerical inverse/forward transform round-off only, not a drift gate.
        if not all(math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1.0e-9)
                   for actual, wanted in zip((anchor.map_x_m, anchor.map_y_m), expected)):
            raise ValueError("drift reference does not match frozen map_from_odom")

    def validate_route_start(self, pose_map) -> None:
        if (self.map_x_m, self.map_y_m) != (_finite(pose_map.x_m), _finite(pose_map.y_m)):
            raise ValueError("drift reference does not match certified route start")


__all__ = ["ROUTE_ANCHOR_METRIC", "RouteDriftAnchor"]
