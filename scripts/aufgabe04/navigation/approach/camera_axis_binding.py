"""Pure binding between a stand-axis observation and its opposite face."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    normalize_angle,
)


REAL_STAND_AXIS_OBSERVATION_KIND = "real_stand_axis_without_qr"


def opposite_face_normal_from_axis_observation(
    payload: Mapping[str, object],
) -> float:
    """Derive the axis-perpendicular face opposite the observing robot."""

    if not isinstance(payload, Mapping):
        raise ValueError("axis observation must be a mapping")
    if payload.get("observation_kind") != REAL_STAND_AXIS_OBSERVATION_KIND:
        raise ValueError("unexpected axis observation kind")
    axis_rad = _finite_number(payload.get("stand_axis_rad"), "stand_axis_rad")
    stand = _mapping(payload.get("stand_center"), "stand_center")
    robot = _mapping(payload.get("robot_pose"), "robot_pose")
    stand_x_m = _finite_number(stand.get("x_m"), "stand_center.x_m")
    stand_y_m = _finite_number(stand.get("y_m"), "stand_center.y_m")
    robot_x_m = _finite_number(robot.get("x_m"), "robot_pose.x_m")
    robot_y_m = _finite_number(robot.get("y_m"), "robot_pose.y_m")
    relative_x_m = robot_x_m - stand_x_m
    relative_y_m = robot_y_m - stand_y_m
    if math.hypot(relative_x_m, relative_y_m) <= 1.0e-9:
        raise ValueError("axis observation robot pose coincides with stand center")

    robot_side = math.atan2(relative_y_m, relative_x_m)
    normals = (
        normalize_angle(axis_rad + math.pi / 2.0),
        normalize_angle(axis_rad - math.pi / 2.0),
    )
    selected = min(normals, key=lambda normal: math.cos(normal - robot_side))
    if math.cos(selected - robot_side) > -0.5:
        raise ValueError(
            "stand axis does not resolve a sufficiently opposite inspection face"
        )
    return selected


def load_opposite_face_normal(axis_observation_path: Path) -> float:
    """Load an observation once and derive its motion-neutral face normal."""

    try:
        payload = json.loads(Path(axis_observation_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load axis observation: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("axis observation JSON root must be an object")
    return opposite_face_normal_from_axis_observation(payload)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"axis observation {name} must be a mapping")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"axis observation {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"axis observation {name} must be finite")
    return number


__all__ = [
    "REAL_STAND_AXIS_OBSERVATION_KIND",
    "load_opposite_face_normal",
    "opposite_face_normal_from_axis_observation",
]
