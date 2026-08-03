"""Current-frame refinement of a model-predicted physical head border."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _corners_inside_image,
    _distance,
    _well_formed_quadrilateral,
    order_corners,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    _QuadrilateralEdgeSupport,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    StandModelProfile,
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


def model_corridor_half_width_px(
    projected_corners: Sequence[ImagePoint],
    *,
    model_profile: StandModelProfile,
    pose_reprojection_rmse_px: float,
) -> float:
    """Bound model search using projected size and independent uncertainties.

    QR reprojection residual describes only the QR plane fit. It cannot cover
    CAD-to-outer-rail error, Canny localization, or provisional measurements.
    Those terms scale with projected head size. The result remains narrowly
    bounded because current-frame geometry and four-side support are checked
    after fitting.
    """

    import math

    if not math.isfinite(pose_reprojection_rmse_px) or pose_reprojection_rmse_px < 0.0:
        raise ValueError("pose_reprojection_rmse_px must be finite and non-negative")
    ordered = order_corners(projected_corners)
    projected_width = (
        _distance(ordered[0], ordered[1]) + _distance(ordered[3], ordered[2])
    ) / 2.0
    projected_height = (
        _distance(ordered[0], ordered[3]) + _distance(ordered[1], ordered[2])
    ) / 2.0
    minimum_extent_px = min(projected_width, projected_height)
    if not math.isfinite(minimum_extent_px) or minimum_extent_px <= 0.0:
        raise ValueError("projected head extent must be finite and positive")

    profile_fraction = (
        0.04 if model_profile.measurement_status == "provisional" else 0.03
    )
    dimensional_tolerance_px = minimum_extent_px * max(
        model_profile.tolerance_m / model_profile.head_width_m,
        model_profile.tolerance_m / model_profile.head_height_m,
    )
    return min(
        8.0,
        max(
            4.0,
            2.0 + profile_fraction * minimum_extent_px,
            2.0 + dimensional_tolerance_px,
            2.5 + 0.75 * pose_reprojection_rmse_px,
        ),
    )


def refine_projected_head_border(
    cv2,
    raw_edges,
    projected_corners: Sequence[ImagePoint],
    *,
    maximum_parallel_side_length_ratio: float = 1.30,
    corridor_half_width_px: float | None = None,
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
        prefer_prediction=True,
        maximum_band_px=corridor_half_width_px,
        # A radiator rail can continue beyond a real head corner. The metric
        # model already predicts the four corner neighbourhoods, so intersect
        # the independently fitted top/bottom and side lines instead of growing
        # side runs along aligned background structure.
        recover_parallel_endpoints=False,
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
    projected = order_corners(projected_corners)
    refined = order_corners(corners)
    projected_width = (
        _distance(projected[0], projected[1])
        + _distance(projected[3], projected[2])
    ) / 2.0
    projected_height = (
        _distance(projected[0], projected[3])
        + _distance(projected[1], projected[2])
    ) / 2.0
    refined_width = (
        _distance(refined[0], refined[1]) + _distance(refined[3], refined[2])
    ) / 2.0
    refined_height = (
        _distance(refined[0], refined[3]) + _distance(refined[1], refined[2])
    ) / 2.0
    projected_center = (
        sum(point.u_px for point in projected) / 4.0,
        sum(point.v_px for point in projected) / 4.0,
    )
    refined_center = (
        sum(point.u_px for point in refined) / 4.0,
        sum(point.v_px for point in refined) / 4.0,
    )
    corridor = 6.0 if corridor_half_width_px is None else corridor_half_width_px
    center_shift = (
        (refined_center[0] - projected_center[0]) ** 2
        + (refined_center[1] - projected_center[1]) ** 2
    ) ** 0.5
    maximum_corner_shift = max(
        _distance(projected_point, refined_point)
        for projected_point, refined_point in zip(projected, refined)
    )
    width_scale = refined_width / max(projected_width, 1.0e-6)
    height_scale = refined_height / max(projected_height, 1.0e-6)
    if (
        not 0.85 <= width_scale <= 1.15
        or not 0.85 <= height_scale <= 1.15
        or center_shift > corridor + 2.0
        or maximum_corner_shift > 2.0 * corridor + 2.0
    ):
        return RefinedHeadMeasurement(
            False,
            "model_refinement_geometry_inconsistent",
            None,
            evidence,
            support,
        )
    return RefinedHeadMeasurement(
        True,
        "fresh_current_frame_refinement",
        corners,
        evidence,
        support,
    )
