"""Calibrated, observe-only LiDAR-to-camera stand-axis handoff."""

from .consensus import AxialConsensus, AxialConsensusAccumulator
from .decision import evaluate_axis_handoff
from .geometry import (
    axial_difference_rad,
    axial_normalize_rad,
    camera_axis_in_scan,
    camera_face_normal_axis_in_scan,
    rectified_pixel_bearing_in_scan,
    transform_point,
)
from .lidar_axis import estimate_pooled_lidar_axis
from .models import (
    ApproachPose,
    AxisHandoffConfig,
    AxisHandoffDecision,
    CameraAxisEstimate,
    LidarAxisEstimate,
    RigidTransform,
)

__all__ = [
    "ApproachPose",
    "AxialConsensus",
    "AxialConsensusAccumulator",
    "AxisHandoffConfig",
    "AxisHandoffDecision",
    "CameraAxisEstimate",
    "LidarAxisEstimate",
    "RigidTransform",
    "axial_difference_rad",
    "axial_normalize_rad",
    "camera_axis_in_scan",
    "camera_face_normal_axis_in_scan",
    "estimate_pooled_lidar_axis",
    "evaluate_axis_handoff",
    "rectified_pixel_bearing_in_scan",
    "transform_point",
]
