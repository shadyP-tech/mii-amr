"""Read-only, exact-edge TF samples for initial admission and live monitoring."""

from __future__ import annotations

import math

try:  # pragma: no cover - exercised on ROS hosts.
    from tf2_ros import TransformException
except ImportError:  # pragma: no cover - offline tests inject a buffer.
    TransformException = Exception

from scripts.aufgabe04.navigation.waypoint_follower.pose_lookup import (
    PoseLookupResult,
    tf_lookup_failure_details,
    validated_planar_pose_from_tf,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import RuntimeBindingProxy

Time = RuntimeBindingProxy("Time", None)
Duration = RuntimeBindingProxy("Duration", None)


def sample_tf_pose(node, *, target_frame: str, source_frame: str, max_future_sec: float) -> PoseLookupResult:
    """Sample this node's listener; never spin, refresh, reseal, or publish."""

    max_age = node.follower_config.max_tf_age_sec
    stamp_sec = None
    age = None

    def failed(reason, exception=None):
        details = tf_lookup_failure_details(
            reason=reason, target_frame=target_frame, source_frame=source_frame,
            max_age_sec=max_age, age_sec=age, exception=exception,
        )
        details.update({
            "stop_reason": f"TF transform unavailable: {target_frame} <- {source_frame}",
            "available": False, "max_future_sec": max_future_sec,
            "validation_passed": False,
        })
        return PoseLookupResult(None, details, stamp_sec)

    try:
        transform = node.tf_buffer.lookup_transform(
            target_frame, source_frame, Time(), timeout=Duration(seconds=0.1),
        )
    except TransformException as exc:
        return failed("lookup_exception", exc)
    try:
        stamp = Time.from_msg(transform.header.stamp)
        stamp_sec = stamp.nanoseconds / 1_000_000_000.0
        age = (node.get_clock().now() - stamp).nanoseconds / 1_000_000_000.0
        if not math.isfinite(stamp_sec) or not math.isfinite(age):
            raise ValueError("TF timestamp or age is non-finite")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        # Do not put non-finite values into JSON evidence.
        stamp_sec = None
        age = None
        return failed("malformed_transform_stamp", exc)
    if age < -max_future_sec:
        return failed("future_transform")
    if age > max_age:
        return failed("stale_transform")
    try:
        pose = validated_planar_pose_from_tf(
            transform, expected_target_frame=target_frame, expected_source_frame=source_frame,
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return failed("malformed_transform_pose", exc)
    return PoseLookupResult(pose, {
        "source": "tf_lookup", "reason": "fresh_transform",
        "target_frame": target_frame, "source_frame": source_frame,
        "available": True, "validation_passed": True,
        "age_sec": age, "max_age_sec": max_age, "max_future_sec": max_future_sec,
    }, stamp_sec)


def map_tf_monitor_warning(lookup: PoseLookupResult) -> str:
    """Preserve runtime warning vocabulary; eligibility uses structured data."""

    if lookup.pose is not None:
        return ""
    details = lookup.details or {}
    reason = details.get("reason")
    if reason == "future_transform":
        return "future_map_from_odom"
    if reason == "stale_transform":
        return "stale_map_from_odom"
    if reason == "malformed_transform_pose":
        return f"map_from_odom_malformed: {details.get('exception', '')}"
    return f"map_from_odom_lookup_failed: {details.get('exception', reason)}"


def refresh_tf_sample_age(node, lookup: PoseLookupResult) -> PoseLookupResult:
    """Recheck an already collected sample; never fetch a substitute transform."""

    if lookup.pose is None:
        return lookup
    details = refreshed_tf_sample_details(node, lookup.details or {}, lookup.stamp_sec)
    return PoseLookupResult(
        lookup.pose if details["available"] else None, details, lookup.stamp_sec,
    )


def refreshed_tf_sample_details(node, details, stamp_sec) -> dict[str, object]:
    result = dict(details)
    reason = ""
    age = None
    try:
        if type(stamp_sec) not in (int, float) or not math.isfinite(stamp_sec):
            raise ValueError("TF sample timestamp is missing or non-finite")
        age = node.get_clock().now().nanoseconds / 1_000_000_000.0 - stamp_sec
        maximum = result["max_age_sec"]
        future = result["max_future_sec"]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (age, maximum, future)):
            raise ValueError("TF sample freshness is malformed")
        if age < -future:
            reason = "future_transform"
        elif age > maximum:
            reason = "stale_transform"
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        reason = "malformed_transform_stamp"
        age = None
        result.update(exception_type=type(exc).__name__, exception=str(exc))
    result["age_sec"] = age
    if reason:
        result.update(
            reason=reason, available=False, validation_passed=False,
            stop_reason=f"TF transform unavailable: {result.get('target_frame')} <- {result.get('source_frame')}",
        )
    return result
