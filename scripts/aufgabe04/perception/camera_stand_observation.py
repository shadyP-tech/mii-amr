"""Validated camera evidence used to resolve a stand's QR-facing pose."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraStandObservation:
    schema_version: int
    observed_at_sec: float
    image_topic: str
    camera_frame: str
    map_frame: str
    robot_x_m: float
    robot_y_m: float
    stand_x_m: float
    stand_y_m: float
    stand_axis_rad: float
    axis_confidence: float
    side: str
    side_confidence: float
    qr_texts: tuple[str, ...] = ()


def validate_camera_observation(
    observation: CameraStandObservation,
    *,
    required_map_frame: str,
    min_axis_confidence: float = 0.60,
    min_side_confidence: float = 0.60,
    max_age_sec: float | None = None,
    now_sec: float | None = None,
) -> None:
    if observation.schema_version != 1:
        raise ValueError("unsupported camera observation schema_version")
    if observation.map_frame.strip("/") != required_map_frame.strip("/"):
        raise ValueError("camera observation map_frame mismatch")
    if not observation.image_topic or not observation.camera_frame:
        raise ValueError("camera observation is missing topic or frame provenance")
    values = (
        observation.robot_x_m, observation.robot_y_m, observation.stand_x_m,
        observation.stand_y_m, observation.stand_axis_rad,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera observation geometry must be finite")
    if observation.axis_confidence < min_axis_confidence:
        raise ValueError("camera stand-axis confidence is insufficient")
    if observation.side not in ("qr_code_side", "basic_color_side"):
        raise ValueError("camera stand side is unknown")
    if observation.side_confidence < min_side_confidence:
        raise ValueError("camera stand-side confidence is insufficient")
    if max_age_sec is not None:
        age = (time.time() if now_sec is None else now_sec) - observation.observed_at_sec
        if age < 0.0 or age > max_age_sec:
            raise ValueError("camera observation is stale")


def load_camera_observation(path: Path) -> CameraStandObservation:
    payload = json.loads(Path(path).read_text())
    payload["qr_texts"] = tuple(payload.get("qr_texts", ()))
    return CameraStandObservation(**payload)


def write_camera_observation(path: Path, observation: CameraStandObservation) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(observation), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stand_axis_from_camera_yaw(
    *,
    robot_x_m: float,
    robot_y_m: float,
    stand_x_m: float,
    stand_y_m: float,
    camera_yaw_rad: float,
    camera_heading_rad: float | None = None,
) -> float:
    """Convert a camera-relative visible-face normal into a map-frame stand axis.

    ``camera_heading_rad`` is the synchronized optical-axis heading in the map
    frame.  Callers that do not have camera pose retain the legacy centered-
    target fallback based on the robot-to-stand line of sight.
    """

    values = (robot_x_m, robot_y_m, stand_x_m, stand_y_m, camera_yaw_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera-to-map geometry must be finite")
    if camera_heading_rad is not None and not math.isfinite(camera_heading_rad):
        raise ValueError("camera heading must be finite when supplied")
    dx = stand_x_m - robot_x_m
    dy = stand_y_m - robot_y_m
    if math.hypot(dx, dy) <= 1e-6:
        raise ValueError("robot and stand positions must differ")
    optical_heading_rad = (
        math.atan2(dy, dx)
        if camera_heading_rad is None
        else camera_heading_rad
    )
    visible_normal_rad = optical_heading_rad + camera_yaw_rad
    return math.atan2(
        math.sin(visible_normal_rad - math.pi / 2.0),
        math.cos(visible_normal_rad - math.pi / 2.0),
    )
