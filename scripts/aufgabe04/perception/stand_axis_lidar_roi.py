"""ROS-free LiDAR ROI helpers for the Aufgabe 04 stand-axis debug viewer.

This module is observe-only. It does not subscribe to ROS topics, publish
motion, or feed station routing/navigation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROI_OBSERVATION_SCHEMA_VERSION = 1
ROI_OBSERVER_VERSION = "aufgabe04-stand-axis-lidar-roi-debug-v1"
DEFAULT_ROI_OBSERVATION_JSONL = Path(
    "results/aufgabe04/stand_axis_lidar_roi/stand_axis_lidar_roi_observations.jsonl"
)


@dataclass(frozen=True)
class PlainLaserScan:
    ranges: tuple[float, ...]
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    scan_frame_id: str
    scan_stamp_sec: float | None = None
    receipt_sec: float | None = None


@dataclass(frozen=True)
class ScanConeRangeQuery:
    distance_m: float | None
    selected_sample_count: int
    rejection_reason: str
    bearing_rad: float
    cone_half_angle_rad: float
    scan_frame_id: str
    scan_stamp_sec: float | None
    scan_age_sec: float | None


@dataclass(frozen=True)
class StandAxisLidarRoiObservation:
    schema_version: int
    observer_version: str
    observed_at_sec: float
    image_topic: str
    image_stamp_sec: float | None
    scan_topic: str
    scan_frame_id: str
    scan_stamp_sec: float | None
    scan_age_sec: float | None
    rect_center_x_px: float | None
    camera_fx_px: float | None
    camera_cx_px: float | None
    camera_bearing_rad: float | None
    lidar_bearing_rad: float | None
    bearing_source: str
    cone_half_angle_rad: float
    selected_sample_count: int
    used_distance_m: float | None
    fallback_source: str
    rejection_reason: str
    estimate_source: str
    estimate_usable: bool


def image_center_x_to_bearing_rad(
    rect_center_x_px: float,
    *,
    camera_fx_px: float,
    camera_cx_px: float,
    camera_to_lidar_yaw_offset_rad: float = 0.0,
) -> float:
    """Map an image x coordinate to an approximate LiDAR-frame bearing."""

    rect_center_x_px = _finite_float(rect_center_x_px, "rect_center_x_px")
    camera_fx_px = _finite_float(camera_fx_px, "camera_fx_px")
    camera_cx_px = _finite_float(camera_cx_px, "camera_cx_px")
    camera_to_lidar_yaw_offset_rad = _finite_float(
        camera_to_lidar_yaw_offset_rad,
        "camera_to_lidar_yaw_offset_rad",
    )
    if camera_fx_px <= 0.0:
        raise ValueError("camera_fx_px must be positive")
    camera_bearing = math.atan((rect_center_x_px - camera_cx_px) / camera_fx_px)
    return _normalize_angle(camera_bearing + camera_to_lidar_yaw_offset_rad)


def camera_bearing_rad(
    rect_center_x_px: float,
    *,
    camera_fx_px: float,
    camera_cx_px: float,
) -> float:
    return image_center_x_to_bearing_rad(
        rect_center_x_px,
        camera_fx_px=camera_fx_px,
        camera_cx_px=camera_cx_px,
        camera_to_lidar_yaw_offset_rad=0.0,
    )


def median_range_in_scan_cone(
    scan: PlainLaserScan | None,
    *,
    bearing_rad: float,
    cone_half_angle_rad: float,
    now_sec: float | None = None,
    max_scan_age_sec: float = 0.0,
    min_sample_count: int = 1,
) -> ScanConeRangeQuery:
    """Return a fail-closed median range query for a scan cone."""

    bearing_rad = _finite_float(bearing_rad, "bearing_rad")
    cone_half_angle_rad = max(0.0, _finite_float(cone_half_angle_rad, "cone_half_angle_rad"))
    min_sample_count = max(1, int(min_sample_count))
    if scan is None:
        return ScanConeRangeQuery(
            distance_m=None,
            selected_sample_count=0,
            rejection_reason="no_scan",
            bearing_rad=bearing_rad,
            cone_half_angle_rad=cone_half_angle_rad,
            scan_frame_id="",
            scan_stamp_sec=None,
            scan_age_sec=None,
        )

    scan_age_sec = None
    if now_sec is not None and scan.receipt_sec is not None:
        scan_age_sec = max(0.0, float(now_sec) - float(scan.receipt_sec))
        if max_scan_age_sec > 0.0 and scan_age_sec > max_scan_age_sec:
            return ScanConeRangeQuery(
                distance_m=None,
                selected_sample_count=0,
                rejection_reason="stale_scan",
                bearing_rad=bearing_rad,
                cone_half_angle_rad=cone_half_angle_rad,
                scan_frame_id=scan.scan_frame_id,
                scan_stamp_sec=scan.scan_stamp_sec,
                scan_age_sec=scan_age_sec,
            )

    selected = []
    for index, raw_range in enumerate(scan.ranges):
        try:
            distance = float(raw_range)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(distance) or distance < scan.range_min or distance > scan.range_max:
            continue
        sample_bearing = scan.angle_min + index * scan.angle_increment
        if abs(_normalize_angle(sample_bearing - bearing_rad)) <= cone_half_angle_rad:
            selected.append(distance)

    if len(selected) < min_sample_count:
        return ScanConeRangeQuery(
            distance_m=None,
            selected_sample_count=len(selected),
            rejection_reason="too_few_valid_samples",
            bearing_rad=bearing_rad,
            cone_half_angle_rad=cone_half_angle_rad,
            scan_frame_id=scan.scan_frame_id,
            scan_stamp_sec=scan.scan_stamp_sec,
            scan_age_sec=scan_age_sec,
        )

    selected.sort()
    middle = len(selected) // 2
    if len(selected) % 2:
        distance_m = selected[middle]
    else:
        distance_m = (selected[middle - 1] + selected[middle]) / 2.0
    return ScanConeRangeQuery(
        distance_m=distance_m,
        selected_sample_count=len(selected),
        rejection_reason="",
        bearing_rad=bearing_rad,
        cone_half_angle_rad=cone_half_angle_rad,
        scan_frame_id=scan.scan_frame_id,
        scan_stamp_sec=scan.scan_stamp_sec,
        scan_age_sec=scan_age_sec,
    )


def nearest_scan_to_stamp(
    scans: Sequence[PlainLaserScan],
    *,
    image_stamp_sec: float | None,
    tolerance_sec: float,
) -> PlainLaserScan | None:
    """Select a LaserScan synchronized to an image using ROS header stamps."""

    if not scans or not math.isfinite(tolerance_sec) or tolerance_sec <= 0.0:
        return None
    if image_stamp_sec is None or not math.isfinite(image_stamp_sec):
        return scans[-1]
    stamped = [
        scan
        for scan in scans
        if scan.scan_stamp_sec is not None and math.isfinite(scan.scan_stamp_sec)
    ]
    if not stamped:
        return None
    nearest = min(stamped, key=lambda scan: abs(scan.scan_stamp_sec - image_stamp_sec))
    if abs(nearest.scan_stamp_sec - image_stamp_sec) > tolerance_sec:
        return None
    return nearest


def observation_to_payload(observation: StandAxisLidarRoiObservation) -> dict[str, object]:
    return asdict(observation)


def observation_from_payload(payload: Mapping[str, object]) -> StandAxisLidarRoiObservation:
    return StandAxisLidarRoiObservation(
        schema_version=int(_require_number(payload, "schema_version")),
        observer_version=_require_str(payload, "observer_version"),
        observed_at_sec=_require_number(payload, "observed_at_sec"),
        image_topic=_require_str(payload, "image_topic"),
        image_stamp_sec=_optional_number(payload, "image_stamp_sec"),
        scan_topic=_require_str(payload, "scan_topic"),
        scan_frame_id=str(payload.get("scan_frame_id") or ""),
        scan_stamp_sec=_optional_number(payload, "scan_stamp_sec"),
        scan_age_sec=_optional_number(payload, "scan_age_sec"),
        rect_center_x_px=_optional_number(payload, "rect_center_x_px"),
        camera_fx_px=_optional_number(payload, "camera_fx_px"),
        camera_cx_px=_optional_number(payload, "camera_cx_px"),
        camera_bearing_rad=_optional_number(payload, "camera_bearing_rad"),
        lidar_bearing_rad=_optional_number(payload, "lidar_bearing_rad"),
        bearing_source=_require_str(payload, "bearing_source"),
        cone_half_angle_rad=_require_number(payload, "cone_half_angle_rad"),
        selected_sample_count=int(_require_number(payload, "selected_sample_count")),
        used_distance_m=_optional_number(payload, "used_distance_m"),
        fallback_source=_require_str(payload, "fallback_source"),
        rejection_reason=str(payload.get("rejection_reason") or ""),
        estimate_source=_require_str(payload, "estimate_source"),
        estimate_usable=bool(payload.get("estimate_usable")),
    )


def write_observation_jsonl(path: Path, observations: Iterable[StandAxisLidarRoiObservation]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        for observation in observations:
            file.write(json.dumps(observation_to_payload(observation), sort_keys=True) + "\n")


def load_observation_jsonl(path: Path) -> tuple[StandAxisLidarRoiObservation, ...]:
    observations = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("line payload must be an object")
            observations.append(observation_from_payload(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid stand-axis LiDAR ROI JSONL line {line_number}: {exc}") from exc
    return tuple(observations)


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_number(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric or null")
    return float(value)
