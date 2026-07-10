"""Final approach target generation for station visits."""

import math

from .models import ApproachTarget, Station, StationPose


def approach_target_for_station(station: Station) -> ApproachTarget:
    offset_x = math.cos(station.pose.yaw_rad) * station.approach_offset_m
    offset_y = math.sin(station.pose.yaw_rad) * station.approach_offset_m
    pose = StationPose(
        x_m=station.pose.x_m - offset_x,
        y_m=station.pose.y_m - offset_y,
        yaw_rad=station.pose.yaw_rad,
    )
    return ApproachTarget(
        station_id=station.station_id,
        pose=pose,
        stop_distance_m=station.approach_offset_m,
    )
