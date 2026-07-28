"""Data contracts shared by the stand-axis estimator modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.aufgabe04.perception.stand_structure_hypothesis import (
        StandStructureEvidence,
    )


@dataclass(frozen=True)
class ImagePoint:
    u_px: float
    v_px: float


@dataclass(frozen=True)
class StandAxisImageEstimate:
    usable: bool
    reason: str
    mode: str
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None
    axis_line: tuple[ImagePoint, ImagePoint] | None
    left_height_px: float
    right_height_px: float
    height_ratio: float | None
    yaw_proxy: float | None
    yaw_deg: float | None
    closer_side: str | None
    contour_area_px: float
    source: str = "unknown"


@dataclass(frozen=True)
class StandAxisEdgeDebugArtifacts:
    edges: object
    face_mask: object | None = None
    rectangle_mask: object | None = None
    rectangle_overlay: object | None = None
    # Immutable pre-morphology Canny evidence. ``edges`` is allowed to contain
    # small gap closures used to discover topology; raw_edges is the only edge
    # domain allowed to validate and refit the measured head rectangle.
    raw_edges: object | None = None
    structure_evidence: StandStructureEvidence | None = None


@dataclass(frozen=True)
class _QuadrilateralEdgeSupport:
    """Per-side evidence that a quadrilateral follows a real edge cutout."""

    top: float
    right: float
    bottom_left: float
    bottom_right: float
    left: float
    tolerance_px: float

    @property
    def bottom(self) -> float:
        return (self.bottom_left + self.bottom_right) / 2.0

    @property
    def mean(self) -> float:
        return (self.top + self.right + self.bottom + self.left) / 4.0

    @property
    def accepted(self) -> bool:
        # The lower middle of a real head can be hidden by the stand stem.
        # Both outer bottom segments must nevertheless be visible; otherwise
        # a U-shaped or unrelated cutout must not become a closed rectangle.
        return (
            self.top >= 0.55
            and self.right >= 0.55
            and self.left >= 0.55
            and self.bottom_left >= 0.45
            and self.bottom_right >= 0.45
            and self.mean >= 0.60
        )


@dataclass(frozen=True)
class _SilhouetteFaceCandidate:
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    face_mask: object
    rectangle_fit_reliable: bool = True
    rectangle_fit_reason: str = "rectangle_fit_supported"
    structure_evidence: StandStructureEvidence | None = None

@dataclass(frozen=True)
class _LineSegment:
    start: ImagePoint
    end: ImagePoint
    length_px: float
    angle_deg: float

    @property
    def y_min(self) -> float:
        return min(self.start.v_px, self.end.v_px)

    @property
    def y_max(self) -> float:
        return max(self.start.v_px, self.end.v_px)

    @property
    def x_min(self) -> float:
        return min(self.start.u_px, self.end.u_px)

    @property
    def x_max(self) -> float:
        return max(self.start.u_px, self.end.u_px)

    @property
    def x_mid(self) -> float:
        return (self.start.u_px + self.end.u_px) / 2.0

    def top_point(self) -> ImagePoint:
        return self.start if self.start.v_px <= self.end.v_px else self.end

    def bottom_point(self) -> ImagePoint:
        return self.start if self.start.v_px > self.end.v_px else self.end
