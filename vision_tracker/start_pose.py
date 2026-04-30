"""
Shared start-pose checking helpers for real-world experiment runs.

This module is deliberately pure: it does not open cameras, display windows,
or depend on ROS.  Both the live tracker overlay and the start gate use the
same acceptance calculation.
"""

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StartPoseReference:
    x: float
    y: float
    yaw_deg: float


@dataclass(frozen=True)
class TrackerPose:
    timestamp: str
    x: float
    y: float
    yaw_rad: float
    yaw_deg: float
    valid_pose: bool
    num_detected: int
    timestamp_epoch: float = None
    file_mtime: float = None


def default_reference():
    config = _config()
    return StartPoseReference(
        x=config.START_POSE_REF_X,
        y=config.START_POSE_REF_Y,
        yaw_deg=config.START_POSE_REF_YAW_DEG,
    )


def angle_error_deg(measured_deg, reference_deg):
    """Return signed shortest-angle error in degrees."""
    return (measured_deg - reference_deg + 180.0) % 360.0 - 180.0


def pose_age_sec(pose, now=None):
    """Return pose age from timestamp, falling back to file mtime."""
    if now is None:
        now = time.time()

    source_time = pose.timestamp_epoch
    if source_time is None:
        source_time = pose.file_mtime

    if source_time is None:
        return float("inf")

    return max(0.0, now - source_time)


def check_start_pose(
    pose,
    ref=None,
    position_tol_m=None,
    yaw_tol_deg=None,
    max_age_sec=None,
    required_markers=None,
    now=None,
):
    """Check whether a measured pose satisfies the configured start gate."""
    if ref is None:
        ref = default_reference()

    if (
        position_tol_m is None
        or yaw_tol_deg is None
        or max_age_sec is None
        or required_markers is None
    ):
        config = _config()
        position_tol_m = _default_if_none(
            position_tol_m,
            config.START_POSE_POSITION_TOLERANCE_M,
        )
        yaw_tol_deg = _default_if_none(
            yaw_tol_deg,
            config.START_POSE_YAW_TOLERANCE_DEG,
        )
        max_age_sec = _default_if_none(max_age_sec, config.START_POSE_MAX_AGE_SEC)
        required_markers = _default_if_none(
            required_markers,
            config.START_POSE_REQUIRED_MARKERS,
        )

    dx = pose.x - ref.x
    dy = pose.y - ref.y
    position_error_m = math.hypot(dx, dy)
    yaw_error_deg = angle_error_deg(pose.yaw_deg, ref.yaw_deg)
    age_sec = pose_age_sec(pose, now=now)

    finite_pose = all(
        math.isfinite(value)
        for value in (pose.x, pose.y, pose.yaw_rad, pose.yaw_deg)
    )
    fresh = age_sec <= max_age_sec
    enough_markers = pose.num_detected >= required_markers

    accepted = (
        finite_pose
        and pose.valid_pose
        and enough_markers
        and fresh
        and position_error_m <= position_tol_m
        and abs(yaw_error_deg) <= yaw_tol_deg
    )

    return {
        "accepted": accepted,
        "dx": dx,
        "dy": dy,
        "position_error_m": position_error_m,
        "yaw_error_deg": yaw_error_deg,
        "pose_age_sec": age_sec,
        "fresh": fresh,
        "valid_pose": pose.valid_pose,
        "num_detected": pose.num_detected,
        "finite_pose": finite_pose,
        "enough_markers": enough_markers,
        "position_tolerance_m": position_tol_m,
        "yaw_tolerance_deg": yaw_tol_deg,
    }


def read_latest_pose(path=None):
    """Read the latest tracker pose CSV written by ``pose_estimator``."""
    if path is None:
        path = _config().LATEST_TRACKER_POSE_FILE
    if not os.path.exists(path):
        return None

    try:
        file_mtime = os.path.getmtime(path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None

    if not rows:
        return None

    row = rows[-1]

    try:
        return TrackerPose(
            timestamp=row.get("timestamp", ""),
            x=float(row["x"]),
            y=float(row["y"]),
            yaw_rad=float(row["yaw_rad"]),
            yaw_deg=float(row["yaw_deg"]),
            valid_pose=_parse_bool(row.get("valid_pose"), default=False),
            num_detected=int(float(row.get("num_detected", 0) or 0)),
            timestamp_epoch=_parse_timestamp_epoch(row.get("timestamp", "")),
            file_mtime=file_mtime,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_bool(value, default=False):
    if value is None or value == "":
        return default

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _parse_timestamp_epoch(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _default_if_none(value, default):
    return default if value is None else value


def _config():
    import config

    return config
