"""Pure geometry for orientation-blind stand inspection and QR approach."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.navigation.models import Pose2D


QR_SIDE = "qr_code_side"
BASIC_SIDE = "basic_color_side"


@dataclass(frozen=True)
class TwoStageApproach:
    stand: Pose2D
    pre_approach: Pose2D
    final_qr_approach: Pose2D | None
    qr_normal_rad: float | None
    side: str = "unknown_side"


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def pre_approach_pose(
    stand: Pose2D,
    robot: Pose2D,
    *,
    offset_m: float,
) -> Pose2D:
    """Choose the robot-facing stand side without using stand/QR yaw."""

    if offset_m <= 0.0:
        raise ValueError("pre-approach offset must be positive")
    bearing = math.atan2(robot.y_m - stand.y_m, robot.x_m - stand.x_m)
    return Pose2D(
        stand.x_m + offset_m * math.cos(bearing),
        stand.y_m + offset_m * math.sin(bearing),
        normalize_angle(bearing + math.pi),
    )


def qr_facing_pose_from_camera(
    stand: Pose2D,
    observation_pose: Pose2D,
    *,
    stand_axis_rad: float,
    side: str,
    offset_m: float,
) -> TwoStageApproach:
    """Resolve the axis ambiguity with camera side evidence and face the QR."""

    if side not in (QR_SIDE, BASIC_SIDE):
        raise ValueError("camera side must be qr_code_side or basic_color_side")
    if offset_m <= 0.0:
        raise ValueError("QR approach offset must be positive")
    observation_bearing = math.atan2(
        observation_pose.y_m - stand.y_m,
        observation_pose.x_m - stand.x_m,
    )
    normals = (
        normalize_angle(stand_axis_rad + math.pi / 2.0),
        normalize_angle(stand_axis_rad - math.pi / 2.0),
    )
    visible_normal = min(normals, key=lambda value: abs(normalize_angle(value - observation_bearing)))
    qr_normal = visible_normal if side == QR_SIDE else normalize_angle(visible_normal + math.pi)
    final_pose = Pose2D(
        stand.x_m + offset_m * math.cos(qr_normal),
        stand.y_m + offset_m * math.sin(qr_normal),
        normalize_angle(qr_normal + math.pi),
    )
    return TwoStageApproach(
        stand=stand,
        pre_approach=observation_pose,
        final_qr_approach=final_pose,
        qr_normal_rad=qr_normal,
        side=side,
    )
