"""Candidate search constraints; these never supply measured head pixels."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.head_proposal import _extent


@dataclass(frozen=True)
class HeadSearchBounds:
    center: tuple[float, float]
    height: float
    center_offset_ratio: float
    height_tolerance_ratio: float

    @classmethod
    def optional(cls, u, v, height, center_offset_ratio, height_tolerance_ratio):
        if all(value is None for value in (u, v, height)):
            return None
        if (any(value is None for value in (u, v, height))
                or not all(math.isfinite(value) for value in
                           (u, v, height, center_offset_ratio, height_tolerance_ratio))
                or height < 12. or not 0. < center_offset_ratio <= 1.5
                or not 0. < height_tolerance_ratio <= .5):
            raise ValueError("invalid candidate head search bounds")
        return cls((float(u), float(v)), float(height),
                   float(center_offset_ratio), float(height_tolerance_ratio))

    def accepts(self, corners):
        _width, height, center = _extent(corners)
        # Apply the observer's existing vertical association bound before
        # unrelated background rectangles can consume the comparison budget.
        return (abs(height / self.height - 1.) <= self.height_tolerance_ratio
                and math.dist(center, self.center) <= self.center_offset_ratio * self.height
                and abs(center[1] - self.center[1])
                <= min(.75, self.center_offset_ratio) * self.height)

    def image_bounds(self, shape):
        # Include every head admitted by the center and height limits, including
        # rotated corners and the existing 1.35 width/height allowance.
        extent = .85 * (1. + self.height_tolerance_ratio)
        radius = self.height * (self.center_offset_ratio + extent) + 6.
        vertical_radius = self.height * (min(.75, self.center_offset_ratio) + extent) + 6.
        u, v = self.center
        return (max(0, int(math.floor(u - radius))), max(0, int(math.floor(v - vertical_radius))),
                min(shape[1], int(math.ceil(u + radius)) + 1),
                min(shape[0], int(math.ceil(v + vertical_radius)) + 1))

    def diagnostics(self):
        return {"center_u_px": self.center[0], "center_v_px": self.center[1],
                "height_px": self.height, "max_center_offset_ratio": self.center_offset_ratio,
                "max_vertical_center_offset_ratio": min(.75, self.center_offset_ratio),
                "height_tolerance_ratio": self.height_tolerance_ratio}
