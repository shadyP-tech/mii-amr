"""Current-frame refinement of a model-predicted physical head border."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _corners_inside_image,
    _well_formed_quadrilateral,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    _QuadrilateralEdgeSupport,
)
from scripts.aufgabe04.perception.stand_axis.raw_support import (
    _quadrilateral_edge_support,
    _raw_side_evidence_and_corners,
)


@dataclass(frozen=True)
class RefinedHeadMeasurement:
    accepted: bool
    reason: str
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None
    evidence_mask: object
    support: _QuadrilateralEdgeSupport | None


def refine_projected_head_border(
    cv2,
    raw_edges,
    projected_corners: Sequence[ImagePoint],
    *,
    maximum_parallel_side_length_ratio: float = 1.30,
) -> RefinedHeadMeasurement:
    """Fit the four physical rails using only current raw-Canny pixels."""

    import numpy

    empty = numpy.zeros(raw_edges.shape[:2], dtype=numpy.uint8)
    if not _well_formed_quadrilateral(projected_corners):
        return RefinedHeadMeasurement(
            False, "projected_head_not_well_formed", None, empty, None
        )
    if not _corners_inside_image(projected_corners, raw_edges.shape):
        return RefinedHeadMeasurement(
            False, "projected_head_outside_image", None, empty, None
        )
    evidence, corners = _raw_side_evidence_and_corners(
        cv2,
        raw_edges,
        projected_corners,
        fixed_parallel_side_direction=None,
        real_camera_endpoint_fraction=0.18,
        maximum_parallel_side_length_ratio=maximum_parallel_side_length_ratio,
    )
    if corners is None:
        return RefinedHeadMeasurement(
            False, "model_corridor_refinement_unavailable", None, evidence, None
        )
    support = _quadrilateral_edge_support(cv2, evidence, corners)
    if not support.accepted:
        return RefinedHeadMeasurement(
            False, "model_corridor_support_insufficient", None, evidence, support
        )
    return RefinedHeadMeasurement(
        True,
        "fresh_current_frame_refinement",
        corners,
        evidence,
        support,
    )
