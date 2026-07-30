"""Pure geometry for real-camera head-candidate temporal association."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _well_formed_quadrilateral,
    order_corners,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    StandAxisImageEstimate,
)


@dataclass(frozen=True)
class HeadCandidateSignature:
    center_x_px: float
    center_y_px: float
    extent_px: float
    width_px: float
    height_px: float
    side_direction_rad: float
    corners_px: tuple[tuple[float, float], ...]
    yaw_deg: float | None
    source: str


@dataclass(frozen=True)
class _BoundaryCoordinates:
    direction_u: float
    direction_v: float
    left_offset: float
    right_offset: float
    top_left_along: float
    top_right_along: float
    bottom_right_along: float
    bottom_left_along: float


def head_candidate_signature(
    estimate: StandAxisImageEstimate,
) -> HeadCandidateSignature | None:
    if not estimate.usable or estimate.corners is None:
        return None
    if not _well_formed_quadrilateral(estimate.corners):
        return None
    top_left, top_right, bottom_right, bottom_left = estimate.corners
    xs = [point.u_px for point in estimate.corners]
    ys = [point.v_px for point in estimate.corners]
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        return None
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    if extent <= 0.0:
        return None
    width_px = (
        math.dist(
            (top_left.u_px, top_left.v_px),
            (top_right.u_px, top_right.v_px),
        )
        + math.dist(
            (bottom_left.u_px, bottom_left.v_px),
            (bottom_right.u_px, bottom_right.v_px),
        )
    ) / 2.0
    left_side = (
        bottom_left.u_px - top_left.u_px,
        bottom_left.v_px - top_left.v_px,
    )
    right_side = (
        bottom_right.u_px - top_right.u_px,
        bottom_right.v_px - top_right.v_px,
    )
    left_height = math.hypot(*left_side)
    right_height = math.hypot(*right_side)
    height_px = (left_height + right_height) / 2.0
    if width_px <= 0.0 or height_px <= 0.0:
        return None
    left_direction = (
        left_side[0] / left_height,
        left_side[1] / left_height,
    )
    right_direction = (
        right_side[0] / right_height,
        right_side[1] / right_height,
    )
    if (
        left_direction[0] * right_direction[0]
        + left_direction[1] * right_direction[1]
        < 0.0
    ):
        right_direction = (-right_direction[0], -right_direction[1])
    yaw_deg = estimate.yaw_deg
    if yaw_deg is not None and not math.isfinite(yaw_deg):
        yaw_deg = None
    return HeadCandidateSignature(
        center_x_px=(min(xs) + max(xs)) / 2.0,
        center_y_px=(min(ys) + max(ys)) / 2.0,
        extent_px=extent,
        width_px=width_px,
        height_px=height_px,
        side_direction_rad=math.atan2(
            left_direction[1] + right_direction[1],
            left_direction[0] + right_direction[0],
        ),
        corners_px=tuple(
            (point.u_px, point.v_px) for point in estimate.corners
        ),
        yaw_deg=yaw_deg,
        source=estimate.source,
    )


def _boundary_coordinates(
    signature: HeadCandidateSignature,
) -> _BoundaryCoordinates:
    direction_u = math.cos(signature.side_direction_rad)
    direction_v = math.sin(signature.side_direction_rad)
    normal_u = -direction_v
    normal_v = direction_u
    top_left, top_right, bottom_right, bottom_left = signature.corners_px
    left_center = (
        (top_left[0] + bottom_left[0]) / 2.0,
        (top_left[1] + bottom_left[1]) / 2.0,
    )
    right_center = (
        (top_right[0] + bottom_right[0]) / 2.0,
        (top_right[1] + bottom_right[1]) / 2.0,
    )
    if (
        (right_center[0] - left_center[0]) * normal_u
        + (right_center[1] - left_center[1]) * normal_v
        < 0.0
    ):
        normal_u = -normal_u
        normal_v = -normal_v

    def normal_offset(point):
        return point[0] * normal_u + point[1] * normal_v

    def along(point):
        return point[0] * direction_u + point[1] * direction_v

    return _BoundaryCoordinates(
        direction_u=direction_u,
        direction_v=direction_v,
        left_offset=(
            normal_offset(top_left) + normal_offset(bottom_left)
        )
        / 2.0,
        right_offset=(
            normal_offset(top_right) + normal_offset(bottom_right)
        )
        / 2.0,
        top_left_along=along(top_left),
        top_right_along=along(top_right),
        bottom_right_along=along(bottom_right),
        bottom_left_along=along(bottom_left),
    )


def _polygon_area_px(corners: tuple[ImagePoint, ...]) -> float:
    return abs(
        sum(
            first.u_px * second.v_px - second.u_px * first.v_px
            for first, second in zip(corners, corners[1:] + corners[:1])
        )
    ) / 2.0


def _replace_estimate_corners(
    template: StandAxisImageEstimate,
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint],
) -> StandAxisImageEstimate:
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    left_height = math.dist(
        (top_left.u_px, top_left.v_px),
        (bottom_left.u_px, bottom_left.v_px),
    )
    right_height = math.dist(
        (top_right.u_px, top_right.v_px),
        (bottom_right.u_px, bottom_right.v_px),
    )
    ratio = left_height / max(right_height, 1.0e-9)
    ordered = (top_left, top_right, bottom_right, bottom_left)
    return replace(
        template,
        reason="axis_estimated_temporal_filtered",
        corners=ordered,
        left_height_px=left_height,
        right_height_px=right_height,
        height_ratio=ratio,
        yaw_proxy=(ratio - 1.0) / (ratio + 1.0),
        closer_side=(
            "left"
            if left_height > right_height
            else "right"
            if right_height > left_height
            else "equal"
        ),
        contour_area_px=_polygon_area_px(ordered),
    )


def blend_parallel_head_estimates(
    previous: StandAxisImageEstimate,
    current: StandAxisImageEstimate,
    *,
    alpha: float,
) -> StandAxisImageEstimate:
    """Low-pass a trapezoid while keeping the two side rails parallel."""

    previous_signature = head_candidate_signature(previous)
    current_signature = head_candidate_signature(current)
    if previous_signature is None or current_signature is None:
        return current
    alpha = min(1.0, max(0.0, float(alpha)))
    angle_delta = (
        current_signature.side_direction_rad
        - previous_signature.side_direction_rad
        + math.pi / 2.0
    ) % math.pi - math.pi / 2.0
    angle = previous_signature.side_direction_rad + alpha * angle_delta
    direction_u = math.cos(angle)
    direction_v = math.sin(angle)
    normal_u = -direction_v
    normal_v = direction_u

    previous_corners = previous_signature.corners_px
    current_corners = current_signature.corners_px
    previous_left_center = (
        (previous_corners[0][0] + previous_corners[3][0]) / 2.0,
        (previous_corners[0][1] + previous_corners[3][1]) / 2.0,
    )
    previous_right_center = (
        (previous_corners[1][0] + previous_corners[2][0]) / 2.0,
        (previous_corners[1][1] + previous_corners[2][1]) / 2.0,
    )
    if (
        (previous_right_center[0] - previous_left_center[0]) * normal_u
        + (previous_right_center[1] - previous_left_center[1]) * normal_v
        < 0.0
    ):
        normal_u = -normal_u
        normal_v = -normal_v

    def decompose(corners_px):
        top_left, top_right, bottom_right, bottom_left = corners_px

        def normal_offset(point):
            return point[0] * normal_u + point[1] * normal_v

        def along(point):
            return point[0] * direction_u + point[1] * direction_v

        return (
            (normal_offset(top_left) + normal_offset(bottom_left)) / 2.0,
            (normal_offset(top_right) + normal_offset(bottom_right)) / 2.0,
            along(top_left),
            along(top_right),
            along(bottom_right),
            along(bottom_left),
        )

    values = tuple(
        previous_value + alpha * (current_value - previous_value)
        for previous_value, current_value in zip(
            decompose(previous_corners),
            decompose(current_corners),
        )
    )
    (
        left_offset,
        right_offset,
        top_left_along,
        top_right_along,
        bottom_right_along,
        bottom_left_along,
    ) = values

    def compose(normal_offset, along):
        return ImagePoint(
            normal_offset * normal_u + along * direction_u,
            normal_offset * normal_v + along * direction_v,
        )

    return _replace_estimate_corners(
        current,
        order_corners(
            (
                compose(left_offset, top_left_along),
                compose(right_offset, top_right_along),
                compose(right_offset, bottom_right_along),
                compose(left_offset, bottom_left_along),
            )
        ),
    )


def structural_geometry_compatible(
    previous: HeadCandidateSignature,
    current: HeadCandidateSignature,
    *,
    max_center_jump_scale: float,
    max_height_ratio: float,
    max_side_direction_jump_deg: float,
) -> bool:
    """Match one physical head without treating projected width as identity."""

    minimum_height = max(1.0, min(previous.height_px, current.height_px))
    center_jump = math.hypot(
        current.center_x_px - previous.center_x_px,
        current.center_y_px - previous.center_y_px,
    )
    if center_jump > max(6.0, max_center_jump_scale * minimum_height):
        return False
    height_ratio = max(previous.height_px, current.height_px) / minimum_height
    if height_ratio > max_height_ratio:
        return False
    side_delta = abs(
        (
            current.side_direction_rad
            - previous.side_direction_rad
            + math.pi / 2.0
        )
        % math.pi
        - math.pi / 2.0
    )
    return side_delta <= math.radians(max_side_direction_jump_deg)


def outer_candidate_area(estimate: StandAxisImageEstimate) -> float:
    if estimate.corners is None:
        return 0.0
    return _polygon_area_px(order_corners(estimate.corners))


def single_boundary_inset_index(
    previous: HeadCandidateSignature,
    current: HeadCandidateSignature,
    *,
    minimum_inset_px: float,
) -> int | None:
    """Detect an inner Canny band replacing one otherwise stable outer side."""

    reference = _boundary_coordinates(previous)
    direction_u = reference.direction_u
    direction_v = reference.direction_v
    normal_u = -direction_v
    normal_v = direction_u
    previous_left = previous.corners_px[0]
    previous_right = previous.corners_px[1]
    if (
        (previous_right[0] - previous_left[0]) * normal_u
        + (previous_right[1] - previous_left[1]) * normal_v
        < 0.0
    ):
        normal_u = -normal_u
        normal_v = -normal_v

    top_left, top_right, bottom_right, bottom_left = current.corners_px

    def normal_offset(point):
        return point[0] * normal_u + point[1] * normal_v

    def along(point):
        return point[0] * direction_u + point[1] * direction_v

    current_values = (
        (normal_offset(top_left) + normal_offset(bottom_left)) / 2.0,
        (normal_offset(top_right) + normal_offset(bottom_right)) / 2.0,
        (along(top_left) + along(top_right)) / 2.0,
        (along(bottom_left) + along(bottom_right)) / 2.0,
    )
    previous_values = (
        reference.left_offset,
        reference.right_offset,
        (reference.top_left_along + reference.top_right_along) / 2.0,
        (
            reference.bottom_left_along
            + reference.bottom_right_along
        )
        / 2.0,
    )
    inward = (
        current_values[0] - previous_values[0],
        previous_values[1] - current_values[1],
        current_values[2] - previous_values[2],
        previous_values[3] - current_values[3],
    )
    large_inward = [delta >= minimum_inset_px for delta in inward]
    stable_limit = max(2.0, 0.5 * minimum_inset_px)
    is_single_inset = (
        sum(large_inward) == 1
        and all(
            abs(delta) <= stable_limit
            for delta, is_large in zip(inward, large_inward)
            if not is_large
        )
    )
    return large_inward.index(True) if is_single_inset else None
