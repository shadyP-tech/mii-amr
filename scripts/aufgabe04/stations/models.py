from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StationPose:
    x_m: float
    y_m: float
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class ApproachTarget:
    station_id: str
    pose: StationPose
    stop_distance_m: float


@dataclass(frozen=True)
class Station:
    station_id: str
    pose: StationPose
    approach_offset_m: float = 0.25
    keepout_radius_m: float = 0.20


@dataclass(frozen=True)
class StationVisit:
    station_id: str
    target: ApproachTarget
    waypoint_ids: Tuple[str, ...] = ()

