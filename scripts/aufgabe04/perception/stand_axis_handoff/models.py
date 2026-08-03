"""Pure data models for calibrated stand-axis handoff."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RigidTransform:
    parent_frame: str
    child_frame: str
    translation_xyz_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class LidarAxisEstimate:
    usable: bool
    reason: str
    angle_rad: float | None = None
    confidence: float = 0.0
    sample_count: int = 0
    scan_count: int = 0
    target_range_m: float | None = None
    target_bearing_rad: float | None = None
    center_xy_m: tuple[float, float] | None = None
    length_m: float | None = None
    width_m: float | None = None
    linearity: float | None = None


@dataclass(frozen=True)
class CameraAxisEstimate:
    usable: bool
    reason: str
    angle_rad: float | None = None
    confidence: float = 0.0
    sample_count: int = 0
    max_deviation_rad: float | None = None
    source: str = ""
    center_xy_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class ApproachPose:
    x_m: float
    y_m: float
    yaw_rad: float
    stand_off_m: float


@dataclass(frozen=True)
class AxisHandoffConfig:
    max_axis_difference_rad: float = math.radians(15.0)
    max_center_difference_m: float = 0.10
    approach_stand_off_m: float = 0.45


@dataclass(frozen=True)
class AxisHandoffDecision:
    status: str
    accepted: bool
    reason: str
    lidar: LidarAxisEstimate
    camera: CameraAxisEstimate
    axial_difference_rad: float | None = None
    center_difference_m: float | None = None
    accepted_axis_rad: float | None = None
    approach_pose: ApproachPose | None = None
    observe_only: bool = True
    motion_authorized: bool = False
