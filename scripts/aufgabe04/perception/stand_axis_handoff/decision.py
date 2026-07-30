"""Fail-closed handoff policy between coarse LiDAR and camera refinement."""

from __future__ import annotations

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import (
    approach_pose_from_axis,
    axial_difference_rad,
)
from scripts.aufgabe04.perception.stand_axis_handoff.models import (
    AxisHandoffConfig,
    AxisHandoffDecision,
    CameraAxisEstimate,
    LidarAxisEstimate,
)


def evaluate_axis_handoff(
    *,
    lidar: LidarAxisEstimate,
    camera: CameraAxisEstimate,
    config: AxisHandoffConfig = AxisHandoffConfig(),
) -> AxisHandoffDecision:
    if not lidar.usable or lidar.angle_rad is None:
        return AxisHandoffDecision(
            "lidar_rejected",
            False,
            lidar.reason,
            lidar,
            camera,
        )
    approach = (
        None
        if lidar.center_xy_m is None
        else approach_pose_from_axis(
            stand_center_xy_m=lidar.center_xy_m,
            stand_axis_rad=lidar.angle_rad,
            stand_off_m=config.approach_stand_off_m,
        )
    )
    if not camera.usable or camera.angle_rad is None:
        return AxisHandoffDecision(
            "camera_collecting",
            False,
            camera.reason,
            lidar,
            camera,
            approach_pose=approach,
        )
    difference = axial_difference_rad(camera.angle_rad, lidar.angle_rad)
    if difference > config.max_axis_difference_rad:
        return AxisHandoffDecision(
            "axis_inconsistent",
            False,
            "camera_lidar_axis_difference_above_gate",
            lidar,
            camera,
            axial_difference_rad=difference,
            approach_pose=approach,
        )
    return AxisHandoffDecision(
        "camera_refined",
        True,
        "camera_lidar_consistent",
        lidar,
        camera,
        axial_difference_rad=difference,
        accepted_axis_rad=camera.angle_rad,
        approach_pose=approach,
    )
