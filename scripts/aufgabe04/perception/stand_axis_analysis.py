"""Offline stand-axis analysis for exported Aufgabe 04 LiDAR samples.

This module is intentionally plain-data only. It does not parse ROS bags,
import ROS message types, subscribe to topics, publish motion, or feed route
planning. It is a measurement tool for exported scan samples.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from scripts.aufgabe04.perception.lidar_stand_detector import (
    cluster_scan_points,
    scan_points_from_ranges,
)
from scripts.aufgabe04.perception.models import BaseFramePoint, LidarStandDetectorConfig


_EPSILON = 1e-12


@dataclass(frozen=True)
class AxisEstimate:
    axis_rad: float | None
    confidence: float
    point_count: int
    length_m: float
    lateral_width_m: float
    centroid_x_m: float
    centroid_y_m: float
    linearity: float
    reason: str


@dataclass(frozen=True)
class AxisUsabilityThresholds:
    min_points: int = 4
    min_confidence: float = 0.60
    min_length_m: float = 0.05
    max_lateral_width_m: float = 0.12
    min_length_to_width_ratio: float = 2.0


@dataclass(frozen=True)
class AxisUsability:
    usable: bool
    reason: str


@dataclass(frozen=True)
class ScanSample:
    sample_id: str
    points: tuple[BaseFramePoint, ...]
    truth_axis_rad: float | None = None


@dataclass(frozen=True)
class AxisAnalysisRow:
    sample_id: str
    cluster_id: str
    point_count: int
    width_m: float
    estimated_axis_rad: float | None
    truth_axis_rad: float | None
    angular_error_rad: float | None
    confidence: float
    usable: bool
    reason: str

    def to_csv_row(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "cluster_id": self.cluster_id,
            "point_count": self.point_count,
            "width_m": _format_optional_float(self.width_m),
            "estimated_axis_rad": _format_optional_float(self.estimated_axis_rad),
            "truth_axis_rad": _format_optional_float(self.truth_axis_rad),
            "angular_error_rad": _format_optional_float(self.angular_error_rad),
            "confidence": f"{self.confidence:.6f}",
            "usable": self.usable,
            "reason": self.reason,
        }


def estimate_cluster_axis(points: Sequence[BaseFramePoint]) -> AxisEstimate:
    point_count = len(points)
    if point_count < 2:
        return AxisEstimate(None, 0.0, point_count, 0.0, 0.0, 0.0, 0.0, 0.0, "too_few_points")

    centroid_x = sum(point.x_m for point in points) / point_count
    centroid_y = sum(point.y_m for point in points) / point_count
    centered = [(point.x_m - centroid_x, point.y_m - centroid_y) for point in points]

    cov_xx = sum(x * x for x, _ in centered) / point_count
    cov_yy = sum(y * y for _, y in centered) / point_count
    cov_xy = sum(x * y for x, y in centered) / point_count
    trace = cov_xx + cov_yy
    discriminant = math.sqrt(max(0.0, (cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy))
    major_eigenvalue = (trace + discriminant) / 2.0
    minor_eigenvalue = (trace - discriminant) / 2.0

    if major_eigenvalue <= _EPSILON:
        return AxisEstimate(
            None,
            0.0,
            point_count,
            0.0,
            0.0,
            centroid_x,
            centroid_y,
            0.0,
            "degenerate_cluster",
        )

    axis_rad = _normalize_axis_angle(0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy))
    cos_axis = math.cos(axis_rad)
    sin_axis = math.sin(axis_rad)
    major_projection = [x * cos_axis + y * sin_axis for x, y in centered]
    minor_projection = [-x * sin_axis + y * cos_axis for x, y in centered]
    length_m = max(major_projection) - min(major_projection)
    lateral_width_m = max(minor_projection) - min(minor_projection)

    linearity = max(0.0, min(1.0, (major_eigenvalue - minor_eigenvalue) / major_eigenvalue))
    point_score = min(1.0, point_count / 8.0)
    confidence = max(0.0, min(1.0, 0.80 * linearity + 0.20 * point_score))
    reason = "axis_estimated" if confidence > 0.0 else "ambiguous_geometry"

    return AxisEstimate(
        axis_rad,
        confidence,
        point_count,
        length_m,
        lateral_width_m,
        centroid_x,
        centroid_y,
        linearity,
        reason,
    )


def angular_error_rad(estimated: AxisEstimate | float, truth_axis_rad: float) -> float:
    estimated_axis = estimated.axis_rad if isinstance(estimated, AxisEstimate) else estimated
    if estimated_axis is None:
        raise ValueError("estimated axis is not available")
    return _axis_angle_difference(float(estimated_axis), truth_axis_rad)


def classify_axis_usability(
    estimate: AxisEstimate,
    thresholds: AxisUsabilityThresholds | None = None,
) -> AxisUsability:
    cfg = thresholds or AxisUsabilityThresholds()
    if estimate.axis_rad is None:
        return AxisUsability(False, estimate.reason)
    if estimate.point_count < cfg.min_points:
        return AxisUsability(False, "too_few_points")
    if estimate.confidence < cfg.min_confidence:
        return AxisUsability(False, "low_confidence")
    if estimate.length_m < cfg.min_length_m:
        return AxisUsability(False, "cluster_too_short")
    if estimate.lateral_width_m > cfg.max_lateral_width_m:
        return AxisUsability(False, "cluster_too_wide")
    width_for_ratio = max(estimate.lateral_width_m, 0.001)
    if estimate.length_m / width_for_ratio < cfg.min_length_to_width_ratio:
        return AxisUsability(False, "ambiguous_aspect_ratio")
    return AxisUsability(True, "usable")


def analyze_scan_sample(
    sample: ScanSample,
    *,
    detector_config: LidarStandDetectorConfig | None = None,
    thresholds: AxisUsabilityThresholds | None = None,
) -> tuple[AxisAnalysisRow, ...]:
    rows = []
    for index, cluster in enumerate(cluster_scan_points(sample.points, config=detector_config), start=1):
        estimate = estimate_cluster_axis(cluster)
        usability = classify_axis_usability(estimate, thresholds)
        error = None
        if estimate.axis_rad is not None and sample.truth_axis_rad is not None:
            error = angular_error_rad(estimate, sample.truth_axis_rad)
        rows.append(
            AxisAnalysisRow(
                sample_id=sample.sample_id,
                cluster_id=f"cluster_{index:02d}",
                point_count=len(cluster),
                width_m=_cluster_span(cluster),
                estimated_axis_rad=estimate.axis_rad,
                truth_axis_rad=sample.truth_axis_rad,
                angular_error_rad=error,
                confidence=estimate.confidence,
                usable=usability.usable,
                reason=usability.reason,
            )
        )
    return tuple(rows)


def analyze_samples(
    samples: Iterable[ScanSample],
    *,
    detector_config: LidarStandDetectorConfig | None = None,
    thresholds: AxisUsabilityThresholds | None = None,
) -> tuple[AxisAnalysisRow, ...]:
    rows = []
    for sample in samples:
        rows.extend(
            analyze_scan_sample(
                sample,
                detector_config=detector_config,
                thresholds=thresholds,
            )
        )
    return tuple(rows)


def load_scan_samples(path: Path) -> tuple[ScanSample, ...]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_json_samples(path)
    if path.suffix.lower() == ".csv":
        return _load_csv_samples(path)
    raise ValueError(f"unsupported scan sample format: {path.suffix}")


def write_axis_analysis_csv(path: Path, rows: Iterable[AxisAnalysisRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AxisAnalysisRow("", "", 0, 0.0, None, None, None, 0.0, False, "").to_csv_row())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def _load_json_samples(path: Path) -> tuple[ScanSample, ...]:
    payload = json.loads(path.read_text())
    raw_samples = (
        payload
        if isinstance(payload, list)
        else payload.get("samples")
        if isinstance(payload, Mapping)
        else None
    )
    if not isinstance(raw_samples, list):
        raise ValueError("JSON scan samples must be a list or an object with a samples list")
    return tuple(_sample_from_mapping(item, index) for index, item in enumerate(raw_samples, start=1))


def _load_csv_samples(path: Path) -> tuple[ScanSample, ...]:
    grouped: dict[str, list[BaseFramePoint]] = {}
    truth_by_sample: dict[str, float | None] = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            sample_id = row.get("sample_id") or "sample_001"
            x_m = _required_float(row, "x_m")
            y_m = _required_float(row, "y_m")
            grouped.setdefault(sample_id, []).append(
                BaseFramePoint(
                    x_m=x_m,
                    y_m=y_m,
                    bearing_rad=float(row.get("bearing_rad") or 0.0),
                    range_m=float(row.get("range_m") or math.hypot(x_m, y_m)),
                    source_index=int(row.get("source_index") or row_index),
                )
            )
            if row.get("truth_axis_rad") not in (None, ""):
                truth_by_sample[sample_id] = float(row["truth_axis_rad"])
            else:
                truth_by_sample.setdefault(sample_id, None)
    return tuple(
        ScanSample(sample_id, tuple(points), truth_by_sample.get(sample_id))
        for sample_id, points in grouped.items()
    )


def _sample_from_mapping(payload: object, index: int) -> ScanSample:
    if not isinstance(payload, Mapping):
        raise ValueError("scan sample entries must be objects")
    sample_id = str(payload.get("sample_id") or f"sample_{index:03d}")
    truth_axis_rad = _optional_float(payload.get("truth_axis_rad"))

    if isinstance(payload.get("points"), list):
        points = tuple(
            _point_from_mapping(point, point_index)
            for point_index, point in enumerate(payload["points"])
        )
    elif isinstance(payload.get("ranges"), list):
        points = tuple(
            scan_points_from_ranges(
                [float(value) for value in payload["ranges"]],
                angle_min_rad=_required_payload_float(payload, "angle_min_rad"),
                angle_increment_rad=_required_payload_float(payload, "angle_increment_rad"),
            )
        )
    else:
        raise ValueError(f"{sample_id} must contain points or ranges")
    return ScanSample(sample_id, points, truth_axis_rad)


def _point_from_mapping(payload: object, source_index: int) -> BaseFramePoint:
    if not isinstance(payload, Mapping):
        raise ValueError("point entries must be objects")
    x_m = _required_payload_float(payload, "x_m")
    y_m = _required_payload_float(payload, "y_m")
    return BaseFramePoint(
        x_m=x_m,
        y_m=y_m,
        bearing_rad=float(payload.get("bearing_rad") or math.atan2(y_m, x_m)),
        range_m=float(payload.get("range_m") or math.hypot(x_m, y_m)),
        source_index=int(payload.get("source_index") or source_index),
    )


def _cluster_span(points: Sequence[BaseFramePoint]) -> float:
    if len(points) < 2:
        return 0.0
    return max(
        math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
        for a in points
        for b in points
    )


def _normalize_axis_angle(angle_rad: float) -> float:
    normalized = (angle_rad + math.pi / 2.0) % math.pi - math.pi / 2.0
    if normalized <= -math.pi / 2.0:
        return normalized + math.pi
    return normalized


def _axis_angle_difference(a_rad: float, b_rad: float) -> float:
    diff = abs(_normalize_axis_angle(a_rad) - _normalize_axis_angle(b_rad))
    return min(diff, math.pi - diff)


def _required_float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"CSV row missing {key}")
    return float(value)


def _required_payload_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("optional value must be numeric")
    return float(value)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"
