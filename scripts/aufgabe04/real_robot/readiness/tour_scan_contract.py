"""Finite, exact-time LiDAR evidence for a stopped stored-pose tour.

This contract never marks unobserved cells free. Invalid range readings remain
None, and scanner poses belong to each scan's own source timestamp in odom.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Mapping, Sequence


MAX_SCAN_AGE_SEC = .25
MAX_WINDOW_SEC = .50
MIN_SAMPLE_SEPARATION_SEC = .08
FUTURE_TOLERANCE_SEC = .02
HASH_FIELD = "tour_scan_capture_sha256"


def finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def stamp_seconds(stamp: object) -> float:
    sec, nanosec = getattr(stamp, "sec"), getattr(stamp, "nanosec")
    if (type(sec) is not int or type(nanosec) is not int
            or sec < 0 or not 0 <= nanosec < 1_000_000_000):
        raise ValueError("invalid ROS source timestamp")
    value = sec + nanosec / 1_000_000_000
    if value <= 0:
        raise ValueError("source timestamp must be positive")
    return value


def exact_transform_pose(transform: object, *, target_frame: str,
                         source_frame: str, stamp_sec: float) -> dict[str, float]:
    header = transform.header
    if header.frame_id != target_frame or transform.child_frame_id != source_frame:
        raise ValueError("exact-time transform frame identity mismatch")
    if abs(stamp_seconds(header.stamp) - stamp_sec) > 1e-7:
        raise ValueError("transform does not match the scan source timestamp")
    translation, rotation = transform.transform.translation, transform.transform.rotation
    x, y, _ = (finite(getattr(translation, name), f"translation.{name}") for name in "xyz")
    qx, qy, qz, qw = (finite(getattr(rotation, name), f"quaternion.{name}") for name in "xyzw")
    if abs(math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) - 1.0) > .01:
        raise ValueError("transform quaternion is not normalized")
    # A planar occupancy projection requires a horizontal scanner/base frame.
    roll = math.atan2(2*(qw*qx + qy*qz), 1-2*(qx*qx + qy*qy))
    pitch = math.asin(max(-1., min(1., 2*(qw*qy-qz*qx))))
    if abs(roll) > .05 or abs(pitch) > .05:
        raise ValueError("transform is not a planar scan pose")
    return {"x_m": x, "y_m": y,
            "yaw_rad": math.atan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))}


def scan_geometry(message: object) -> dict[str, object]:
    angle_min = finite(message.angle_min, "angle_min")
    increment = finite(message.angle_increment, "angle_increment")
    lower = finite(message.range_min, "range_min")
    upper = finite(message.range_max, "range_max")
    if increment == 0 or lower < 0 or upper <= lower:
        raise ValueError("invalid scan angle or range bounds")
    raw = tuple(message.ranges)
    if not 2 <= len(raw) <= 4096 or abs(increment) * (len(raw)-1) > 2*math.pi + .05:
        raise ValueError("invalid scan sample count or angular extent")
    ranges = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError("scan ranges must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scan ranges must be numeric") from exc
        ranges.append(number if math.isfinite(number) and number > 0 and lower <= number <= upper else None)
    if not any(value is not None for value in ranges):
        raise ValueError("scan has no valid finite returns")
    return {"angle_min": angle_min, "angle_increment": increment,
            "range_min": lower, "range_max": upper, "ranges": ranges}


def _validate_scan_record(scan: Mapping[str, object]) -> None:
    """Also validate persisted records; a correct hash alone is not admission."""
    for field in ("scan_pose_odom", "base_pose_odom"):
        pose = scan[field]
        if not isinstance(pose, Mapping) or set(pose) != {"x_m", "y_m", "yaw_rad"}:
            raise ValueError("capture pose must contain planar numeric coordinates")
        for name, value in pose.items():
            finite(value, name)
    finite(scan["angle_min"], "angle_min")
    increment = finite(scan["angle_increment"], "angle_increment")
    lower, upper = finite(scan["range_min"], "range_min"), finite(scan["range_max"], "range_max")
    ranges = scan["ranges"]
    if (not isinstance(ranges, (list, tuple)) or not 2 <= len(ranges) <= 4096
            or increment == 0 or lower < 0 or upper <= lower
            or abs(increment)*(len(ranges)-1) > 2*math.pi+.05):
        raise ValueError("invalid persisted scan geometry")
    any_valid = False
    for value in ranges:
        if value is None:
            continue
        number = finite(value, "range")
        if not (number > 0 and lower <= number <= upper):
            raise ValueError("persisted invalid range must be None")
        any_valid = True
    if not any_valid:
        raise ValueError("capture scan contains no valid finite returns")


def scan_evidence(message: object, *, received_at_unix_sec: float,
                  scan_transform: object, base_transform: object,
                  odom_frame: str, base_frame: str, scan_frame: str) -> dict[str, object]:
    if message.header.frame_id != scan_frame:
        raise ValueError("scan frame identity mismatch")
    stamp = stamp_seconds(message.header.stamp)
    receipt = finite(received_at_unix_sec, "received_at_unix_sec")
    if not -FUTURE_TOLERANCE_SEC <= receipt-stamp <= MAX_SCAN_AGE_SEC:
        raise ValueError("scan is stale or future-dated at receipt")
    return {"stamp_sec": stamp, "received_at_unix_sec": receipt,
            "scan_pose_stamp_sec": stamp, "base_pose_stamp_sec": stamp,
            "scan_pose_odom": exact_transform_pose(scan_transform, target_frame=odom_frame,
                source_frame=scan_frame, stamp_sec=stamp),
            "base_pose_odom": exact_transform_pose(base_transform, target_frame=odom_frame,
                source_frame=base_frame, stamp_sec=stamp), **scan_geometry(message)}


def capture_payload(scans: Sequence[Mapping[str, object]], *, tour_id: str,
                    odom_frame: str, base_frame: str, scan_frame: str,
                    captured_at_unix_sec: float) -> dict[str, object]:
    """Validate one bounded stationary cohort; earlier cohort scans may age .5 s."""
    if any(not isinstance(value, str) or not value.strip()
           for value in (tour_id, odom_frame, base_frame, scan_frame)):
        raise ValueError("capture identity fields must be nonempty strings")
    now = finite(captured_at_unix_sec, "captured_at_unix_sec")
    if len(scans) != 3:
        raise ValueError("capture requires exactly three distinct scans")
    stamps = [finite(scan["stamp_sec"], "stamp_sec") for scan in scans]
    receipts = [finite(scan["received_at_unix_sec"], "received_at_unix_sec") for scan in scans]
    if any(b-a < MIN_SAMPLE_SEPARATION_SEC-1e-6 for a, b in zip(stamps, stamps[1:])):
        raise ValueError("scan source stamps must be ordered and separated by .08 s")
    if stamps[-1]-stamps[0] > MAX_WINDOW_SEC+1e-6 or max(receipts)-min(receipts) > MAX_WINDOW_SEC+1e-6:
        raise ValueError("stationary scan window exceeds .5 s")
    if not -FUTURE_TOLERANCE_SEC <= now-stamps[-1] <= MAX_SCAN_AGE_SEC:
        raise ValueError("latest capture scan is stale or future-dated")
    for scan, stamp, receipt in zip(scans, stamps, receipts):
        _validate_scan_record(scan)
        if stamp <= 0:
            raise ValueError("capture source timestamp must be positive")
        if not -FUTURE_TOLERANCE_SEC <= receipt-stamp <= MAX_SCAN_AGE_SEC:
            raise ValueError("capture contains a scan stale at receipt")
        if receipt > now + FUTURE_TOLERANCE_SEC:
            raise ValueError("capture receipt is in the future")
        if scan["scan_pose_stamp_sec"] != stamp or scan["base_pose_stamp_sec"] != stamp:
            raise ValueError("capture transform timestamp mismatch")
    poses = [scan["base_pose_odom"] for scan in scans]
    for a, b in combinations(poses, 2):
        ax, ay, aa = (finite(a[key], key) for key in ("x_m", "y_m", "yaw_rad"))
        bx, by, ba = (finite(b[key], key) for key in ("x_m", "y_m", "yaw_rad"))
        if math.hypot(ax-bx, ay-by) > .015 or abs(math.atan2(math.sin(aa-ba), math.cos(aa-ba))) > math.radians(2):
            raise ValueError("robot moved during stopped scan capture")
    mounts = []
    for scan in scans:
        base, laser = scan["base_pose_odom"], scan["scan_pose_odom"]
        dx, dy = laser["x_m"]-base["x_m"], laser["y_m"]-base["y_m"]
        yaw = base["yaw_rad"]
        mounts.append((math.cos(yaw)*dx+math.sin(yaw)*dy,
                       -math.sin(yaw)*dx+math.cos(yaw)*dy,
                       laser["yaw_rad"]-yaw))
    for a, b in combinations(mounts, 2):
        yaw_change = math.atan2(math.sin(a[2]-b[2]), math.cos(a[2]-b[2]))
        if math.hypot(a[0]-b[0], a[1]-b[1]) > .005 or abs(yaw_change) > .01:
            raise ValueError("scanner mounting transform changed during capture")
    return {"schema_version": 1, "artifact_kind": "stored_pose_tour_scan_capture",
            "tour_id": tour_id, "odom_frame": odom_frame, "base_frame": base_frame,
            "scan_frame": scan_frame, "captured_at_unix_sec": now,
            "scans": [dict(scan) for scan in scans]}
