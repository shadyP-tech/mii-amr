"""Crop-local 2D search seeds for strict current-pixel head refinement.

A proposal carries no pose or measurement authority. It only positions the
ordinary bounded rail search; the caller must still validate the current raw
head structure, calibrated head fit and observability before using its axis.
The proposal itself carries neither a view-side label nor QR identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _corners_inside_image,
    _well_formed_quadrilateral,
    order_corners,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    model_corridor_half_width_px,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


@dataclass(frozen=True)
class HeadBorderSeed:
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    corridor_half_width_px: float
    source: str


def validate_current_head_proposal(
    corners: Sequence[ImagePoint] | None,
    *,
    frame_shape,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    """Require a complete finite quadrilateral in this exact crop's pixels."""

    if corners is None:
        return None
    try:
        points = tuple(ImagePoint(float(p.u_px), float(p.v_px)) for p in corners)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("current head proposal must contain four crop-local corners") from exc
    if (
        len(points) != 4
        or not all(math.isfinite(v) for p in points for v in (p.u_px, p.v_px))
        or not _corners_inside_image(points, frame_shape)
        or not _well_formed_quadrilateral(points)
    ):
        raise ValueError("current head proposal must be finite, well-formed and inside the crop")
    return order_corners(points)


def select_head_border_seed(
    *,
    model_profile: StandModelProfile,
    projected_corners: Sequence[ImagePoint] | None,
    pose_reprojection_rmse_px: float | None,
    current_head_proposal_corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None,
) -> HeadBorderSeed:
    """Select a search location without widening refinement acceptance gates.

    An independent image proposal has no pose residual. Zero here omits that
    uncertainty term; the same measured-dimension and pixel-width bounds apply.
    Proposal coordinates are already validated against the current crop.
    """

    if current_head_proposal_corners is not None:
        corners = current_head_proposal_corners
        residual = 0.0
        source = "current_head_proposal"
    else:
        if projected_corners is None or pose_reprojection_rmse_px is None:
            raise ValueError("head refinement needs a current proposal or pose projection")
        corners = order_corners(projected_corners)
        residual = pose_reprojection_rmse_px
        source = "pose_projection"
    return HeadBorderSeed(
        corners=corners,
        corridor_half_width_px=model_corridor_half_width_px(
            corners, model_profile=model_profile,
            pose_reprojection_rmse_px=residual,
        ),
        source=source,
    )
