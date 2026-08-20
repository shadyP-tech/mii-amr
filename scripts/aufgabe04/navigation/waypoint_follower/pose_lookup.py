"""ROS-free TF pose parsing and lookup diagnostics for the follower runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.navigation.models import Pose2D


# Gazebo odometry quaternions are unit normalized. A 1e-3 norm tolerance
# admits only floating-point serialization drift; it is not a normalization
# or malformed-pose repair path.
SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE = 1.0e-3


def _yaw_from_quaternion(quaternion) -> float:
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def _normalized_tf_frame_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip("/"):
        raise ValueError("TF frame ID is missing")
    return value.strip("/")


def validated_planar_pose_from_tf(
    transform,
    *,
    expected_target_frame: str,
    expected_source_frame: str,
) -> Pose2D:
    """Extract one finite planar pose from an exact configured TF edge."""

    observed_target = _normalized_tf_frame_id(
        getattr(getattr(transform, "header", None), "frame_id", None)
    )
    observed_source = _normalized_tf_frame_id(
        getattr(transform, "child_frame_id", None)
    )
    expected_target = _normalized_tf_frame_id(expected_target_frame)
    expected_source = _normalized_tf_frame_id(expected_source_frame)
    if observed_target != expected_target or observed_source != expected_source:
        raise ValueError(
            "TF frame identity mismatch: "
            f"observed={observed_target}<-{observed_source}, "
            f"expected={expected_target}<-{expected_source}"
        )
    try:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        translation_values = tuple(
            float(value)
            for value in (translation.x, translation.y, translation.z)
        )
        quaternion = tuple(
            float(value)
            for value in (rotation.x, rotation.y, rotation.z, rotation.w)
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("TF pose payload is malformed") from exc
    if not all(
        math.isfinite(value) for value in (*translation_values, *quaternion)
    ):
        raise ValueError("TF pose payload is non-finite")
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    if (
        abs(quaternion_norm - 1.0)
        > SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE
    ):
        raise ValueError("TF pose quaternion is not normalized")
    yaw_rad = _yaw_from_quaternion(rotation)
    if not math.isfinite(yaw_rad):
        raise ValueError("TF pose yaw is non-finite")
    return Pose2D(translation_values[0], translation_values[1], yaw_rad)


def ros_stamp_sec(stamp) -> float | None:
    try:
        value = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class PoseLookupResult:
    pose: Pose2D | None
    details: dict[str, object] | None = None
    stamp_sec: float | None = None


def tf_lookup_failure_details(
    *,
    reason: str,
    target_frame: str,
    source_frame: str,
    max_age_sec: float,
    age_sec: float | None = None,
    exception: BaseException | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "map-to-base transform unavailable",
        "source": "tf_lookup",
        "reason": reason,
        "target_frame": target_frame,
        "source_frame": source_frame,
        "max_age_sec": max_age_sec,
    }
    if age_sec is not None:
        payload["age_sec"] = age_sec
    if exception is not None:
        payload["exception_type"] = exception.__class__.__name__
        payload["exception"] = str(exception)
    return payload


def pose_lookup_diagnostics(result: PoseLookupResult) -> dict[str, object]:
    details = dict(result.details or {})
    if not details:
        details = {
            "source": "tf_lookup",
            "reason": "fresh_transform",
        }
    if result.stamp_sec is not None:
        details["stamp_sec"] = result.stamp_sec
    return details


def stale_tf_recovery_failure_details(
    final_details: dict[str, object],
    *,
    first_lookup: PoseLookupResult,
    retry_lookup: PoseLookupResult,
    callback_drain: dict[str, object],
) -> dict[str, object]:
    first_details = pose_lookup_diagnostics(first_lookup)
    retry_details = pose_lookup_diagnostics(retry_lookup)
    combined = dict(final_details)
    combined.update(
        {
            "fail_closed": True,
            "recovery_attempted": True,
            "zero_published_before_retry": True,
            "first_lookup_age_sec": first_details.get("age_sec"),
            "retry_lookup_age_sec": retry_details.get("age_sec"),
            "first_lookup_stamp_sec": first_lookup.stamp_sec,
            "retry_lookup_stamp_sec": retry_lookup.stamp_sec,
            "first_lookup": first_details,
            "retry_lookup": retry_details,
            "callback_drain": callback_drain,
        }
    )
    return combined

