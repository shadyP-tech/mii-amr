"""Immutable values shared by the live exploration loop.

These values stay in one process. Sensor admission belongs to the ROS boundary;
path admission belongs to the planner. A consumed stopped epoch cannot be reused
to start another motion leg.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float = math.nan

    def __post_init__(self) -> None:
        finite(self.x_m, "pose x")
        finite(self.y_m, "pose y")
        if isinstance(self.yaw_rad, bool) or not isinstance(self.yaw_rad, (int, float)):
            raise ValueError("pose yaw must be numeric")
        if math.isinf(self.yaw_rad):
            raise ValueError("pose yaw cannot be infinite")


@dataclass(frozen=True)
class Transform2D:
    """Transform from local coordinates to parent coordinates."""

    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "yaw_rad"):
            finite(getattr(self, name), name)

    def apply(self, pose: Pose2D) -> Pose2D:
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return Pose2D(c * pose.x_m - s * pose.y_m + self.x_m,
                      s * pose.x_m + c * pose.y_m + self.y_m,
                      wrap_angle(pose.yaw_rad + self.yaw_rad))

    def inverse_apply(self, pose: Pose2D) -> Pose2D:
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        dx, dy = pose.x_m - self.x_m, pose.y_m - self.y_m
        return Pose2D(c * dx + s * dy, -s * dx + c * dy,
                      wrap_angle(pose.yaw_rad - self.yaw_rad))


@dataclass(frozen=True)
class Velocity:
    linear_x_mps: float = 0.0
    angular_z_radps: float = 0.0

    def __post_init__(self) -> None:
        finite(self.linear_x_mps, "linear velocity")
        finite(self.angular_z_radps, "angular velocity")


@dataclass(frozen=True)
class Candidate:
    uid: str
    x_m: float
    y_m: float
    radius_m: float
    uncertainty_m: float
    confidence: float = 1.0
    sightings: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.uid, str) or not self.uid:
            raise ValueError("candidate requires an identity")
        for name in ("x_m", "y_m", "radius_m", "uncertainty_m", "confidence"):
            finite(getattr(self, name), name)
        if self.radius_m <= 0 or self.uncertainty_m < 0 or not 0 <= self.confidence <= 1:
            raise ValueError("invalid candidate geometry or confidence")
        if type(self.sightings) is not int or self.sightings < 1:
            raise ValueError("candidate requires at least one sighting")


@dataclass(frozen=True)
class StoppedState:
    epoch: int
    captured_monotonic: float
    pose_map: Pose2D
    pose_odom: Pose2D
    map_from_odom: Transform2D
    position_covariance: tuple[float, float, float]
    yaw_variance: float
    map_id: str
    source_stamps: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch < 1 or not self.map_id:
            raise ValueError("stopped state requires an epoch and map identity")
        if finite(self.captured_monotonic, "capture time") < 0:
            raise ValueError("capture time cannot be negative")
        for pose in (self.pose_map, self.pose_odom):
            if not isinstance(pose, Pose2D):
                raise ValueError("stopped state requires typed poses")
            finite(pose.yaw_rad, "stopped yaw")
        if not isinstance(self.map_from_odom, Transform2D):
            raise ValueError("stopped state requires a typed map transform")
        if len(self.position_covariance) != 3:
            raise ValueError("position covariance requires xx, xy, yy")
        xx, xy, yy = (finite(v, "covariance") for v in self.position_covariance)
        if min(xx, yy, finite(self.yaw_variance, "yaw variance")) < 0 or xy * xy > xx * yy + 1e-15:
            raise ValueError("covariance must be positive semidefinite")
        for name, stamp in self.source_stamps:
            if not name or finite(stamp, "source stamp") <= 0:
                raise ValueError("invalid source timestamp")


@dataclass(frozen=True)
class PlannedLeg:
    kind: str
    target_id: str
    map_id: str
    source_epoch: int
    poses: tuple[Pose2D, ...]
    target_center: tuple[float, float] | None = None
    minimum_clearance_m: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("coverage", "front", "opposite"):
            raise ValueError("unsupported exploration leg kind")
        if not self.target_id or not self.map_id or type(self.source_epoch) is not int or self.source_epoch < 1:
            raise ValueError("planned leg requires target, map and stopped epoch")
        if not isinstance(self.poses, tuple) or len(self.poses) < 2:
            raise ValueError("planned motion requires an immutable route")
        if not all(isinstance(p, Pose2D) for p in self.poses):
            raise ValueError("route requires typed poses")
        finite(self.poses[-1].yaw_rad, "final heading")
        if finite(self.minimum_clearance_m, "route clearance") < 0:
            raise ValueError("route clearance cannot be negative")
        if self.target_center is not None:
            if len(self.target_center) != 2:
                raise ValueError("target center requires two coordinates")
            for v in self.target_center:
                finite(v, "target center")


@dataclass(frozen=True)
class InspectionResult:
    target_id: str
    status: str
    source_epoch: int
    source_pose: Pose2D
    source_stamp_sec: float
    target_x_m: float
    target_y_m: float
    target_uncertainty_m: float
    qr_id: str | None = None
    axis_normal_rad: float | None = None
    axis_half_width_rad: float | None = None
    accepted_frames: int = 0

    def __post_init__(self) -> None:
        if self.status not in ("identified", "unidentified", "unresolved"):
            raise ValueError("invalid inspection status")
        if not self.target_id or type(self.source_epoch) is not int or self.source_epoch < 1:
            raise ValueError("inspection requires a current target and epoch")
        if not isinstance(self.source_pose, Pose2D):
            raise ValueError("inspection requires a typed source pose")
        finite(self.source_pose.yaw_rad, "inspection yaw")
        for name in ("source_stamp_sec", "target_x_m", "target_y_m", "target_uncertainty_m"):
            finite(getattr(self, name), name)
        if self.source_stamp_sec <= 0 or self.target_uncertainty_m < 0:
            raise ValueError("invalid inspection time or uncertainty")
        if self.status == "identified" and (not isinstance(self.qr_id, str) or not self.qr_id.strip()):
            raise ValueError("identified inspection requires a QR payload")
        if self.status != "identified" and self.qr_id is not None:
            raise ValueError("unidentified inspection cannot claim a QR payload")
        if (self.axis_normal_rad is None) != (self.axis_half_width_rad is None):
            raise ValueError("orientation requires a center and interval")
        if self.axis_normal_rad is not None:
            finite(self.axis_normal_rad, "axis normal")
            if not 0 <= finite(self.axis_half_width_rad, "axis interval") < math.pi / 2:
                raise ValueError("orientation interval is not bounded")
        if type(self.accepted_frames) is not int or self.accepted_frames < 0:
            raise ValueError("invalid accepted frame count")
