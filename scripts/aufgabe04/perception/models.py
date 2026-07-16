from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class BaseFramePoint:
    x_m: float
    y_m: float
    bearing_rad: float
    range_m: float
    source_index: int


@dataclass(frozen=True)
class StandCandidate:
    candidate_id: str
    bearing_rad: float
    distance_m: float
    approximate_width_m: float
    center_x_m: float
    center_y_m: float
    point_count: int
    confidence: float


@dataclass(frozen=True)
class LidarStandDetectorConfig:
    min_range_m: float = 0.08
    max_range_m: float = 3.5
    max_cluster_gap_m: float = 0.08
    # The TurtleBot LDS has roughly one-degree angular resolution.  A 7 cm
    # stand head at 1.2--1.7 m commonly occupies only two adjacent beams.
    min_cluster_points: int = 2
    min_width_m: float = 0.03
    max_width_m: float = 0.45


@dataclass(frozen=True)
class ColorRange:
    label: str
    lower_hsv: Tuple[int, int, int]
    upper_hsv: Tuple[int, int, int]


@dataclass(frozen=True)
class ColorClassification:
    label: str
    confidence: float
    matched_pixels: int
    total_pixels: int
    timestamp_sec: Optional[float] = None


@dataclass(frozen=True)
class ColorClassifierConfig:
    min_confidence: float = 0.20
    unknown_label: str = "unknown"
