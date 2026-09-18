"""Reject incompatible head locations without supplying measured image corners.

The current candidate projection supplies only a conservative search screen.
Full-image edge detection and raw border refinement stay unchanged. An optional
LiDAR volume region limits contour and rail discovery before the work quotas.
Current camera/LiDAR association remains required after the 3D fit.
"""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.head_proposal import _extent
from scripts.aufgabe04.perception.stand_axis.head_outer_border import OUTER_HEAD_SEARCH_GROWTH_FACTORS
from scripts.aufgabe04.perception.stand_axis.geometry import _distance, order_corners
from scripts.aufgabe04.perception.stand_axis.lidar_head_edge_region import LidarHeadEdgeRegion


# Match observer.head_model_admission.head_scale_gate. Tests compare these
# decisions directly so changing final association cannot silently narrow here.
MIN_MEASURED_HEIGHT_RATIO = .60
MAX_MEASURED_HEIGHT_RATIO = 1.35
MIN_MEASURED_SIDE_BALANCE = .65


@dataclass(frozen=True)
class CandidateHeadSearch:
    center: tuple[float, float]
    height: float
    max_center_offset_ratio: float = 1.5
    edge_region: LidarHeadEdgeRegion | None = None

    def __post_init__(self):
        if (len(self.center) != 2
                or not all(math.isfinite(value) for value in (*self.center, self.height,
                                                             self.max_center_offset_ratio))
                or self.height <= 0. or not 0. < self.max_center_offset_ratio <= 1.5):
            raise ValueError("candidate head search requires a finite current projection")

    @classmethod
    def optional(cls, u, v, height, *, max_center_offset_ratio=1.5):
        """Invalid projections retain the ordinary full-image geometry path."""
        try:
            return cls((float(u), float(v)), float(height), float(max_center_offset_ratio))
        except (TypeError, ValueError, OverflowError):
            return None

    def accepts_hint(self, corners):
        """Keep every hint whose permitted raw refinement could meet final bounds.

        Each raw refinement admits height scale [.85, 1.15] and center shift at
        most corridor + 2 pixels; metric corridors are capped at eight pixels.
        Outer searches grow about the same hint center by at most 1.25. Their
        additional center gate can only narrow this ten-pixel envelope.
        """
        _width, height, center = _extent(corners)
        largest_growth = max((1., *OUTER_HEAD_SEARCH_GROWTH_FACTORS))
        # Coordinates reconstructed at an envelope endpoint can lose a few ulps.
        roundoff = 1.e-12 * max(height, self.height, 1.)
        return (.85 * height <= MAX_MEASURED_HEIGHT_RATIO * self.height + roundoff
                and 1.15 * largest_growth * height + roundoff >= MIN_MEASURED_HEIGHT_RATIO * self.height
                and math.dist(center, self.center)
                <= self.max_center_offset_ratio * self.height + 10. + roundoff)

    def accepts_measurement(self, corners):
        """Apply the existing candidate size/center bounds to measured pixels."""
        _width, height, center = _extent(corners)
        tl, tr, br, bl = order_corners(corners)
        left, right = _distance(tl, bl), _distance(tr, br)
        return ((self.edge_region is None or self.edge_region.contains(corners))
                and MIN_MEASURED_HEIGHT_RATIO <= height / self.height <= MAX_MEASURED_HEIGHT_RATIO
                and min(left, right) / max(left, right, 1.e-9) >= MIN_MEASURED_SIDE_BALANCE
                and math.dist(center, self.center) / self.height <= self.max_center_offset_ratio)

    def diagnostics(self):
        return {"policy": "conservative_candidate_head_screen", "image_scope": "full_image",
                "edge_region": None if self.edge_region is None else self.edge_region.diagnostics(),
                "center_u_px": self.center[0], "center_v_px": self.center[1],
                "height_px": self.height, "max_center_offset_ratio": self.max_center_offset_ratio,
                "hint_center_refinement_allowance_px": 10.,
                "measured_height_ratio": (MIN_MEASURED_HEIGHT_RATIO, MAX_MEASURED_HEIGHT_RATIO),
                "minimum_measured_side_balance": MIN_MEASURED_SIDE_BALANCE,
                "supplies_corners": False, "qr_used": False, "motion_authorized": False}
