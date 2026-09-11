"""Shared wire contract for historical candidate planning-frame evidence.

Version-one projections permit the legacy four fields and a defined optional
pose_provenance extension. Reading evidence never refreshes localization or
grants motion authority. Capture metadata remains detached and lossless.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping

from scripts.aufgabe04.navigation.localization.candidate_planning_pose import (
    candidate_pose_from_captures, candidate_tf_capture_pose,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D, normalize_yaw,
)


_FRAME_FIELDS = frozenset({"current_pose", "map_from_odom", "map_frame", "odom_frame"})
_POSE_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})
_PROVENANCE_FIELDS = frozenset({"pose_basis", "map_from_odom_capture", "odom_pose_capture"})
_DIAGNOSTIC_FIELDS = frozenset({
    "diagnostic_chained_map_pose", "diagnostic_chained_map_capture",
    "chained_to_authoritative_translation_delta_m",
    "chained_to_authoritative_yaw_delta_rad",
})
_NUMERICAL_TOLERANCE = 1.0e-12


def _mapping(value: object, fields: frozenset[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields mismatch")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _frame_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip("/") != value or any(
        char.isspace() for char in value
    ):
        raise ValueError(f"{name} must be a non-prefixed frame identifier")
    return value


def _pose(value: object, name: str) -> tuple[float, float, float]:
    payload = _mapping(value, _POSE_FIELDS, name)
    return tuple(_finite(payload[key], f"{name}.{key}") for key in (
        "x_m", "y_m", "yaw_rad",
    ))


def _close(actual: float, expected: float, name: str, *, tolerance: float = _NUMERICAL_TOLERANCE) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} mismatch")


def _finite_json(value: object) -> None:
    """Preserve extensible capture diagnostics, but reject non-JSON/NaN data."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _finite(value, "capture metadata")
    elif isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    else:
        raise ValueError("capture metadata must be finite JSON")


def _capture_metadata(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("planning frame capture must be an object")
    _finite_json(value)
    if "age_sec" in value:
        # ROS records age from integer nanoseconds; epoch float subtraction
        # loses sub-microsecond precision. This is consistency, not freshness.
        _close(
            _finite(value["age_sec"], "capture age_sec"),
            _finite(value.get("capture_time_sec"), "capture_time_sec")
            - _finite(value.get("stamp_sec"), "stamp_sec"),
            "capture age", tolerance=1.0e-6,
        )
    if "max_future_sec" in value and _finite(value["max_future_sec"], "max_future_sec") < 0:
        raise ValueError("capture max_future_sec must be nonnegative")
    if "quaternion" in value:
        quaternion = value["quaternion"]
        if not isinstance(quaternion, Mapping) or not quaternion:
            raise ValueError("capture quaternion must be an object")
        for component in quaternion.values():
            _finite(component, "capture quaternion")
    return value


def _validate_provenance(
    value: object, *, pose: tuple[float, float, float],
    transform: PlanarTransform2D, map_frame: str, odom_frame: str,
) -> None:
    fields = _PROVENANCE_FIELDS
    if isinstance(value, Mapping) and set(value) & _DIAGNOSTIC_FIELDS:
        fields |= _DIAGNOSTIC_FIELDS
    provenance = _mapping(value, fields, "pose_provenance")
    if provenance["pose_basis"] != "direct_map_from_odom_times_observed_odom_pose":
        raise ValueError("unsupported planning frame pose_basis")
    direct = _capture_metadata(provenance["map_from_odom_capture"])
    odom = _capture_metadata(provenance["odom_pose_capture"])
    composed, captured_transform = candidate_pose_from_captures(
        direct, odom, map_frame=map_frame, odom_frame=odom_frame,
    )
    if captured_transform != transform:
        raise ValueError("planning frame map_from_odom capture mismatch")
    for actual, expected in zip(pose, (composed.x_m, composed.y_m, composed.yaw_rad)):
        _close(actual, expected, "planning frame composed current_pose")
    if fields == _PROVENANCE_FIELDS:
        return

    chained = _pose(provenance["diagnostic_chained_map_pose"], "diagnostic_chained_map_pose")
    capture = provenance["diagnostic_chained_map_capture"]
    if capture is not None:
        capture = _capture_metadata(capture)
        # A chained lookup is diagnostic. Check its internal identity and
        # values, without requiring it to equal the authoritative map pose.
        captured_pose = candidate_tf_capture_pose(capture, map_frame, odom["source_frame"])
        for actual, expected in zip(captured_pose, chained):
            _close(actual, expected, "diagnostic chained capture pose")
    for key, expected in (
        ("chained_to_authoritative_translation_delta_m", math.hypot(pose[0] - chained[0], pose[1] - chained[1])),
        ("chained_to_authoritative_yaw_delta_rad", abs((pose[2] - chained[2] + math.pi) % math.tau - math.pi)),
    ):
        _close(_finite(provenance[key], key), expected, key)


def validate_planning_frame_evidence(value: object) -> dict[str, object]:
    """Return a detached validated mapping, preserving legacy and capture data."""
    fields = _FRAME_FIELDS
    if isinstance(value, Mapping) and "pose_provenance" in value:
        fields |= {"pose_provenance"}
    payload = _mapping(value, fields, "planning_frame_admission")
    pose = _pose(payload["current_pose"], "current_pose")
    transform_values = _pose(payload["map_from_odom"], "map_from_odom")
    if transform_values[2] != normalize_yaw(transform_values[2]):
        raise ValueError("map_from_odom.yaw_rad must be normalized")
    transform = PlanarTransform2D(*transform_values)
    map_frame = _frame_id(payload["map_frame"], "map_frame")
    odom_frame = _frame_id(payload["odom_frame"], "odom_frame")
    if map_frame == odom_frame:
        raise ValueError("map_frame and odom_frame must be distinct")
    if "pose_provenance" in payload:
        _validate_provenance(
            payload["pose_provenance"], pose=pose, transform=transform,
            map_frame=map_frame, odom_frame=odom_frame,
        )
    return deepcopy(dict(payload))


__all__ = ["validate_planning_frame_evidence"]
