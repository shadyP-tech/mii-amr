"""Current-pixel head geometry. Locators never certify their own borders.

The retained calibrated algorithms fit raw image rails, exclude competing
frames and paper insets, and keep every plausible positive-depth IPPE pose.
No historical receipt or persisted image proof enters this module.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import wraps
from itertools import product
from types import SimpleNamespace
from typing import Sequence, Tuple
import math
import time
import numpy as np

@dataclass(frozen=True)
class ModelPoint3D:
    x_m: float
    y_m: float
    z_m: float = 0.0

@dataclass(frozen=True)
class HeadModel:
    head_width_m: float
    head_height_m: float
    head_depth_m: float
    qr_panel_width_m: float
    qr_panel_height_m: float
    tolerance_m: float
    head_top_height_m: float
    # These are startup-admitted physical dimensions; no alternative profile
    # or persisted evidence can be substituted during a session.

    @property
    def head_corners(self):
        w,h = self.head_width_m/2., self.head_height_m/2.
        return (ModelPoint3D(-w,-h),ModelPoint3D(w,-h),ModelPoint3D(w,h),ModelPoint3D(-w,h))

@dataclass(frozen=True)
class CurrentHeadRefinement:
    refinement: object
    outer_recovery: object
    seed: object
    alternative_corners: tuple = ()


def capture_current_head_refinement(frame_bgr, raw_edges, *, model_profile,
                                    refinement, outer_recovery, seed):
    # Private result within one image call. The immutable source never leaves
    # this call, so there is no serialized cache to revalidate or rebind.
    return CurrentHeadRefinement(refinement, outer_recovery, seed)

@dataclass(frozen=True)
class ColorRange:
    label: str
    lower_hsv: Tuple[int, int, int]
    upper_hsv: Tuple[int, int, int]


class ImageSourceSupport:
    """Immutable current image-domain mask, independent of colour or target.

    The complete four borders must lie inside real source pixels after allowing
    for Gaussian blur, Canny's gradient and nonmaximum-suppression footprint.
    Neither image pixels nor raw edges are altered or supplied by this filter.
    """

    def __init__(self, cv2, valid_pixels, *, blur_kernel=5):
        import numpy as np
        if (not isinstance(valid_pixels, np.ndarray) or valid_pixels.ndim != 2
                or not valid_pixels.size or valid_pixels.dtype not in (np.uint8, np.bool_)
                or type(blur_kernel) is not int or blur_kernel < 0):
            raise ValueError("source support requires a nonempty camera-domain mask")
        self.radius = (blur_kernel // 2 if blur_kernel > 1 else 0) + 2
        self._valid = cv2.erode((valid_pixels != 0).astype(np.uint8),
            np.ones((2*self.radius+1, 2*self.radius+1), np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        self._valid.flags.writeable = False
        self.shape = self._valid.shape

    def accepts(self, corners):
        import numpy as np
        if corners is None or len(corners) != 4:
            return False
        points = np.asarray([(p.u_px, p.v_px) for p in corners], dtype=float)
        height, width = self.shape
        if (not np.isfinite(points).all() or np.any(points < 0.)
                or np.any(points[:, 0] >= width - 1.) or np.any(points[:, 1] >= height - 1.)):
            return False
        return all(self.segment(a, b) for a, b in zip(points, np.roll(points, -1, axis=0)))

    def segment(self, a, b):
        import numpy as np
        a, b = np.asarray(a, float), np.asarray(b, float)
        height, width = self.shape
        if (not np.isfinite((a, b)).all() or np.any(a < 0.) or np.any(b < 0.)
                or max(a[0], b[0]) >= width-1. or max(a[1], b[1]) >= height-1.):
            return False
        count = int(math.ceil(float(np.max(np.abs(b-a))))) + 1
        pixels = np.rint(np.linspace(a, b, count)).astype(np.int32)
        return bool(self._valid[pixels[:, 1], pixels[:, 0]].all())

    def filter(self, other=None):
        domain = self
        class Filter:
            source_support = domain

            def preview(self, proposal):
                return domain.accepts(proposal.corners) and (
                    other is None or getattr(other, "preview", other)(proposal))

            def __call__(self, proposal):
                return domain.accepts(proposal.corners) and (other is None or other(proposal))
        return Filter()


STAND_EDGE_PALETTE = (
    ColorRange("light-grey", (105, 30, 65), (140, 85, 255)),
    ColorRange("blue", (96, 150, 40), (127, 255, 255)),
    ColorRange("green", (38, 65, 40), (95, 255, 255)),
    ColorRange("red", (0, 75, 40), (12, 255, 255)),
    ColorRange("red", (166, 75, 40), (179, 255, 255)),
    ColorRange("purple", (128, 100, 40), (165, 255, 255)),
)


def color_edge_support(cv2, numpy, frame, *, color="all", ranges=None):
    """Allow real edges within two pixels of colored exterior contours.

    Canny often places an outline just outside its colored surface. A small
    dilation keeps that outline without opening away thin colored rails.
    A small closing bridges segmentation pinholes; external contours omit QR
    texture enclosed by a colored border. Never generate measured edges from
    this binary mask: it only selects existing Canny pixels.
    Explicit ranges support the viewer's live HSV trackbars.
    """
    if ranges is None:
        ranges = tuple(item for item in STAND_EDGE_PALETTE
                       if color == "all" or item.label == color)
        if not ranges:
            raise ValueError(f"unknown edge color: {color}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask_for_ranges(cv2, numpy, hsv, ranges)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    outline = numpy.zeros_like(mask)
    cv2.drawContours(outline, contours, -1, 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Closing can bridge gaps, but those bridges cannot license distant edges.
    return cv2.bitwise_and(cv2.dilate(outline, kernel, iterations=1),
                           cv2.dilate(mask, kernel, iterations=1))


def build_mask_for_ranges(cv2, numpy, hsv_frame, color_ranges: Sequence[ColorRange]):
    mask = numpy.zeros(hsv_frame.shape[:2], dtype=numpy.uint8)
    for color_range in color_ranges:
        lower = numpy.array(color_range.lower_hsv, dtype=numpy.uint8)
        upper = numpy.array(color_range.upper_hsv, dtype=numpy.uint8)
        range_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.bitwise_or(mask, range_mask)
    return mask


def _canny_edges_from_frame(
    cv2,
    frame,
    *,
    edge_preprocess: str,
    blur_kernel: int,
    canny_low: int,
    canny_high: int,
):
    """Extract edges without assigning semantic meaning to any color.

    ``channel_union`` applies identical blur/Canny operations independently to
    B, G, and R and takes their logical union. It is invariant to channel
    permutation and therefore remains color-agnostic, while retaining borders
    whose foreground/background luminance happens to be almost identical.
    """

    if edge_preprocess != "channel_union":
        raise ValueError("physical head acquisition requires channel_union edges")
    edge_frame = frame
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        edge_frame = cv2.GaussianBlur(edge_frame, (blur_kernel, blur_kernel), 0)
    channels = cv2.split(edge_frame)
    if not channels:
        raise ValueError("frame must contain at least one image channel")
    edges = cv2.Canny(channels[0], canny_low, canny_high)
    for channel in channels[1:]:
        edges = cv2.bitwise_or(edges, cv2.Canny(channel, canny_low, canny_high))
    return edges


class HeadAcquisitionDeadlineExceeded(RuntimeError):
    def __init__(self, stage, *, raw_verifications=0):
        super().__init__(stage)
        self.stage = stage
        self.raw_verifications = raw_verifications


def check_head_acquisition_deadline(deadline_monotonic_sec, stage, **progress):
    if deadline_monotonic_sec is None:
        return
    if not math.isfinite(deadline_monotonic_sec):
        raise ValueError("head acquisition deadline must be finite")
    if time.monotonic() >= deadline_monotonic_sec:
        raise HeadAcquisitionDeadlineExceeded(stage, **progress)


def bounded_head_acquisition(acquire):
    """Translate expiry into a diagnostic result with no proposal authority."""
    @wraps(acquire)
    def bounded(*args, **kwargs):
        try:
            return acquire(*args, **kwargs)
        except HeadAcquisitionDeadlineExceeded as error:

            return HeadProposalResult(
                None, "head_acquisition_deadline_exceeded", error.raw_verifications,
            )
    return bounded


@dataclass(frozen=True)
class ImagePoint:
    u_px: float
    v_px: float


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


def _corners_inside_image(corners: Sequence[ImagePoint], image_shape) -> bool:
    height, width = image_shape[:2]
    return all(0.0 <= point.u_px < width and 0.0 <= point.v_px < height for point in corners)


def _polygon_area(corners: Sequence[ImagePoint]) -> float:
    ordered = order_corners(corners)
    area = 0.0
    for current, following in zip(ordered, ordered[1:] + ordered[:1]):
        area += current.u_px * following.v_px - following.u_px * current.v_px
    return abs(area) / 2.0


def _well_formed_quadrilateral(
    corners: Sequence[ImagePoint],
    *,
    min_bbox_fill_ratio: float = 0.60,
    min_side_length_ratio: float = 0.20,
    min_corner_sine: float = 0.12,
) -> bool:
    """Reject collapsed or triangular four-point detector output.

    OpenCV may occasionally report four QR points even though two points have
    converged onto one corner. Area and average aspect ratio alone do not catch
    that case. This gate is deliberately projection-tolerant, but requires a
    convex polygon with four meaningful sides and enough area inside its axis
    aligned bounding box to represent a visible stand face.
    """

    if len(corners) != 4:
        return False
    ordered = order_corners(corners)
    coordinates = tuple(
        (float(point.u_px), float(point.v_px)) for point in ordered
    )
    if not all(math.isfinite(value) for point in coordinates for value in point):
        return False

    xs = tuple(point[0] for point in coordinates)
    ys = tuple(point[1] for point in coordinates)
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if bbox_area <= 0.0:
        return False
    if _polygon_area(ordered) / bbox_area < min_bbox_fill_ratio:
        return False

    edges = tuple(
        (
            following[0] - current[0],
            following[1] - current[1],
        )
        for current, following in zip(
            coordinates,
            coordinates[1:] + coordinates[:1],
        )
    )
    lengths = tuple(math.hypot(*edge) for edge in edges)
    maximum_length = max(lengths)
    if maximum_length <= 0.0:
        return False
    if min(lengths) / maximum_length < min_side_length_ratio:
        return False

    corner_crosses = []
    for previous_edge, current_edge in zip(edges[-1:] + edges[:-1], edges):
        cross = (
            previous_edge[0] * current_edge[1]
            - previous_edge[1] * current_edge[0]
        )
        denominator = math.hypot(*previous_edge) * math.hypot(*current_edge)
        if denominator <= 0.0 or abs(cross) / denominator < min_corner_sine:
            return False
        corner_crosses.append(cross)
    return all(cross > 0.0 for cross in corner_crosses) or all(
        cross < 0.0 for cross in corner_crosses
    )


def order_corners(points: Sequence[ImagePoint]) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]:
    if len(points) != 4:
        raise ValueError("exactly four points are required")
    ordered_by_y = sorted(points, key=lambda point: (point.v_px, point.u_px))
    top = sorted(ordered_by_y[:2], key=lambda point: point.u_px)
    bottom = sorted(ordered_by_y[2:], key=lambda point: point.u_px)
    top_left, top_right = top
    bottom_left, bottom_right = bottom
    return top_left, top_right, bottom_right, bottom_left


def _distance(first: ImagePoint, second: ImagePoint) -> float:
    return math.hypot(second.u_px - first.u_px, second.v_px - first.v_px)


@dataclass(frozen=True)
class RigidTransform:
    parent_frame: str
    child_frame: str
    translation_xyz_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def _normalized_quaternion(
    rotation_xyzw: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(rotation_xyzw) != 4:
        raise ValueError("quaternion must contain four values")
    values = tuple(float(value) for value in rotation_xyzw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def rotate_vector(
    vector_xyz: Sequence[float],
    rotation_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    if len(vector_xyz) != 3:
        raise ValueError("vector must contain three values")
    x, y, z = (float(value) for value in vector_xyz)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("vector must be finite")
    qx, qy, qz, qw = _normalized_quaternion(rotation_xyzw)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def transform_point(
    point_xyz: Sequence[float],
    transform: RigidTransform,
) -> tuple[float, float, float]:
    rotated = rotate_vector(point_xyz, transform.rotation_xyzw)
    return tuple(
        value + offset
        for value, offset in zip(rotated, transform.translation_xyz_m)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class RectifiedCameraMatrix:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def validate(self) -> None:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rectified camera matrix values must be finite")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("rectified focal lengths must be positive")


@dataclass(frozen=True)
class PlanarPoseHypothesis:
    rotation_vector: tuple[float, float, float]
    translation_xyz_m: tuple[float, float, float]
    face_normal_xyz: tuple[float, float, float]
    yaw_deg: float
    reprojection_rmse_px: float
    positive_depth: bool


@dataclass(frozen=True)
class PlanarPoseResult:
    accepted: bool
    reason: str
    hypotheses: tuple[PlanarPoseHypothesis, ...]
    ambiguity_gap_px: float | None

    @property
    def best(self) -> PlanarPoseHypothesis | None:
        return self.hypotheses[0] if self.accepted and self.hypotheses else None

    def axis_ambiguous(
        self,
        *,
        max_residual_gap_px: float = 0.10,
        min_axis_separation_deg: float = 5.0,
    ) -> bool:
        """Whether similarly fitting planar solutions imply different axes."""

        if len(self.hypotheses) < 2 or self.ambiguity_gap_px is None:
            return False
        first, second = self.hypotheses[:2]
        axial_difference = abs(
            (first.yaw_deg - second.yaw_deg + 90.0) % 180.0 - 90.0
        )
        return (
            self.ambiguity_gap_px <= max_residual_gap_px
            and axial_difference >= min_axis_separation_deg
        )


def estimate_planar_pose_ippe(
    cv2,
    image_points: Sequence[ImagePoint],
    model_points: Sequence[ModelPoint3D],
    camera: RectifiedCameraMatrix,
    *,
    max_reprojection_rmse_px: float = 2.0,
) -> PlanarPoseResult:
    """Return all physically visible IPPE hypotheses ordered by residual."""

    import numpy

    camera.validate()
    if len(image_points) != len(model_points) or len(image_points) < 4:
        raise ValueError("matching image/model point lists need at least four points")
    if not math.isfinite(max_reprojection_rmse_px) or max_reprojection_rmse_px <= 0.0:
        raise ValueError("max_reprojection_rmse_px must be finite and positive")
    model_z = [point.z_m for point in model_points]
    if max(model_z) - min(model_z) > 1.0e-9:
        raise ValueError("IPPE seed points must be coplanar")

    object_points = numpy.asarray(
        [[point.x_m, point.y_m, point.z_m] for point in model_points],
        dtype=numpy.float64,
    )
    pixels = numpy.asarray(
        [[point.u_px, point.v_px] for point in image_points],
        dtype=numpy.float64,
    )
    if not numpy.isfinite(object_points).all() or not numpy.isfinite(pixels).all():
        raise ValueError("pose points must be finite")
    camera_matrix = numpy.asarray(
        (
            (camera.fx_px, 0.0, camera.cx_px),
            (0.0, camera.fy_px, camera.cy_px),
            (0.0, 0.0, 1.0),
        ),
        dtype=numpy.float64,
    )
    distortion = numpy.zeros((4, 1), dtype=numpy.float64)
    try:
        result = cv2.solvePnPGeneric(
            object_points,
            pixels,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except Exception:
        result = ()
    if not result or not bool(result[0]):
        return PlanarPoseResult(False, "ippe_failed", (), None)

    hypotheses = []
    for rotation_vector, translation_vector in zip(result[1], result[2]):
        rotation_vector = numpy.asarray(rotation_vector, dtype=numpy.float64).reshape(3, 1)
        translation_vector = numpy.asarray(translation_vector, dtype=numpy.float64).reshape(3, 1)
        try:
            projected, _jacobian = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation_vector,
                camera_matrix,
                distortion,
            )
            rotation, _jacobian = cv2.Rodrigues(rotation_vector)
        except Exception:
            continue
        residual = projected.reshape(-1, 2) - pixels
        rmse = math.sqrt(float(numpy.mean(numpy.sum(residual * residual, axis=1))))
        normal = rotation @ numpy.asarray(((0.0,), (0.0,), (1.0,)), dtype=numpy.float64)
        normal_xyz = tuple(float(value) for value in normal.reshape(3))
        translation_xyz = tuple(float(value) for value in translation_vector.reshape(3))
        finite = all(math.isfinite(value) for value in (*normal_xyz, *translation_xyz, rmse))
        if not finite:
            continue
        hypotheses.append(
            PlanarPoseHypothesis(
                rotation_vector=tuple(float(value) for value in rotation_vector.reshape(3)),
                translation_xyz_m=translation_xyz,
                face_normal_xyz=normal_xyz,
                # Match the public stand-axis convention: positive image-left.
                yaw_deg=-math.degrees(math.atan2(normal_xyz[0], abs(normal_xyz[2]))),
                reprojection_rmse_px=rmse,
                positive_depth=translation_xyz[2] > 0.0,
            )
        )
    visible = tuple(
        sorted(
            (hypothesis for hypothesis in hypotheses if hypothesis.positive_depth),
            key=lambda hypothesis: hypothesis.reprojection_rmse_px,
        )
    )
    if not visible:
        return PlanarPoseResult(False, "no_positive_depth_pose", (), None)
    gap = (
        visible[1].reprojection_rmse_px - visible[0].reprojection_rmse_px
        if len(visible) >= 2
        else None
    )
    accepted = visible[0].reprojection_rmse_px <= max_reprojection_rmse_px
    return PlanarPoseResult(
        accepted,
        "pose_estimated" if accepted else "reprojection_error_too_high",
        visible,
        gap,
    )


def coherent_metric_rail_points(
    cv2,
    candidates,
    candidate_bins,
    start,
    end,
    *,
    band_px: float,
    expected_length_px: float,
    minimum_coverage: float,
    fixed_direction=None,
    outward_sign=0.,
    color_support_mask=None,
):
    """Associate whole raw-edge rails before selecting one pixel per bin.

    Every returned point comes from ``candidates``, already clipped to the
    caller's side corridor and sampling intervals. Hough segments only propose
    lines: their interpolated pixels are never used as measurement evidence.
    Coverage of a coherent rail takes precedence over isolated pixels closer
    to the prediction. Comparable rails use projection distance to break ties;
    indistinguishable, spatially separate rails remain unobservable.
    A position-bounded physical head may instead request the enclosing rail,
    requiring at least 80 percent raw coverage in the same finite corridor.
    Colour support is a soft ranking cue worth at most one pixel, never a mask
    on returned evidence or a replacement for the raw corner checks.
    """

    import numpy

    if len(candidates) < 5:
        return candidates[:0]
    origin = numpy.floor(candidates.min(axis=0)).astype(numpy.int32)
    pixels = numpy.rint(candidates - origin).astype(numpy.int32)
    local_edges = numpy.zeros(
        (int(pixels[:, 1].max()) + 1, int(pixels[:, 0].max()) + 1),
        dtype=numpy.uint8,
    )
    local_edges[pixels[:, 1], pixels[:, 0]] = 255
    segments = cv2.HoughLinesP(
        local_edges,
        1.0,
        numpy.pi / 180.0,
        threshold=max(5, int(0.18 * expected_length_px)),
        minLineLength=max(4.0, 0.20 * expected_length_px),
        maxLineGap=max(2.0, min(5.0, 0.04 * expected_length_px)),
    )
    start_xy = numpy.array((start.u_px, start.v_px), dtype=numpy.float64)
    end_xy = numpy.array((end.u_px, end.v_px), dtype=numpy.float64)
    midpoint = (start_xy + end_xy) / 2.0
    tangent = end_xy - start_xy
    tangent /= numpy.linalg.norm(tangent)
    reference_direction = (
        tangent
        if fixed_direction is None
        else numpy.asarray(fixed_direction, dtype=numpy.float64)
    )
    reference_direction = reference_direction / numpy.linalg.norm(reference_direction)
    proposals = []
    if segments is not None:
        for segment in segments.reshape(-1, 4):
            proposals.append(
                (
                    segment[:2].astype(numpy.float64) + origin,
                    (segment[2:] - segment[:2]).astype(numpy.float64),
                )
            )
    # The lower head edge has two separate visible intervals around the stem.
    # Short Hough segments can quantize their slopes differently and lose one
    # interval on extrapolation. Also test the predicted/common direction at
    # offsets actually observed in this corridor; both intervals still need
    # independent raw support before the final quadrilateral can be accepted.
    reference_normal = numpy.array(
        (-reference_direction[1], reference_direction[0])
    )
    offsets = (candidates - midpoint) @ reference_normal
    offset_bins = numpy.floor(offsets + 0.5).astype(numpy.int32)
    for offset_bin in numpy.unique(offset_bins):
        offset = float(numpy.median(offsets[offset_bins == offset_bin]))
        if abs(offset) <= band_px:
            proposals.append(
                (midpoint + offset * reference_normal, reference_direction)
            )
    hypotheses = []
    for point, proposal_direction in proposals:
        direction = proposal_direction.copy()
        length = float(numpy.linalg.norm(direction))
        if length <= 0.0:
            continue
        direction /= length
        if float(direction @ tangent) < 0.0:
            direction *= -1.0
        angle = math.degrees(
            math.acos(min(1.0, abs(float(direction @ reference_direction))))
        )
        if angle > (12.0 if fixed_direction is not None else 20.0):
            continue
        normal = numpy.array((-direction[1], direction[0]))
        distance = abs(float((midpoint - point) @ normal))
        if distance > band_px:
            continue
        # A tight proposal band avoids combining two different parallel rails.
        # The existing robust fit and its coverage gates run on these raw
        # samples afterwards; this does not replace their acceptance checks.
        residuals = numpy.abs((candidates - point) @ normal)
        selected = residuals <= 1.25
        coverage = len(numpy.unique(candidate_bins[selected])) / max(
            expected_length_px, 1.0
        )
        if coverage < minimum_coverage:
            continue
        # Ignore a few rasterization bins when comparing complete alternatives.
        coverage_rank = round(min(1.0, coverage) * 20.0) / 20.0
        colour = 0.
        if color_support_mask is not None:
            locations = numpy.rint(candidates[selected]).astype(numpy.int32)
            colour = float(numpy.mean(color_support_mask[locations[:, 1], locations[:, 0]] > 0))
        if outward_sign and coverage >= .80:
            signed_offset = float((point-midpoint) @ reference_normal)*outward_sign
            # A soft hint worth at most one pixel of outward displacement.
            # Missing grey/overexposed mask support cannot delete a raw rail.
            rank = (1., signed_offset+colour, coverage_rank, -float(numpy.median(residuals[selected])))
        elif outward_sign:
            continue
        else:
            rank = (coverage_rank, -distance+colour, -float(numpy.median(residuals[selected])))
        hypotheses.append((rank, point, normal, selected))

    if not hypotheses:
        return candidates[:0]
    hypotheses.sort(key=lambda item: item[0], reverse=True)
    best_rank, best_point, best_normal, best_selected = hypotheses[0]
    for rank, point, normal, _selected in hypotheses[1:]:
        if rank[0] != best_rank[0] or abs(rank[1] - best_rank[1]) > 0.75:
            continue
        # Do not invent a measured rail halfway between equally plausible
        # parallel edges. Nearby Canny bands on the same rail may co-exist.
        best_nearest = midpoint + float((best_point - midpoint) @ best_normal) * best_normal
        other_nearest = midpoint + float((point - midpoint) @ normal) * normal
        # Two 2.5 px localization bands can describe the same thick Canny rail.
        if float(numpy.linalg.norm(best_nearest - other_nearest)) > 5.0:
            return candidates[:0]
    return candidates[best_selected]


@dataclass(frozen=True)
class MetricCornerArmSupport:
    """Raw support for the two incident rails at every semantic head corner."""

    radius_px: float
    minimum_bins: int
    bins_by_corner: dict[str, dict[str, int]]

    @property
    def accepted(self) -> bool:
        return all(
            count >= self.minimum_bins
            for arms in self.bins_by_corner.values()
            for count in arms.values()
        )


def metric_corner_arm_support(cv2, raw_edges, corners) -> MetricCornerArmSupport:
    """Count observed bins on both arms without discarding failed evidence.

    Long window/radiator segments may satisfy the trimmed side tests while
    their infinite-line intersections have no physical head corner nearby.
    Check the untouched image here: the side evidence mask deliberately omits
    endpoints. A corner pixel itself is not required, since heads are rounded.
    Counts use the original 1.5 px minimum longitudinal distance, bounded
    4–6 px radius, 2.5 px perpendicular allowance, and rounded pixel bins.
    """

    import numpy


    shortest_side = min(
        _distance(first, second)
        for first, second in zip(corners, corners[1:] + corners[:1])
    )
    radius = max(4.0, min(6.0, 0.08 * shortest_side))
    margin = int(math.ceil(radius + 2.0))
    height, width = raw_edges.shape[:2]
    names_and_arms = (
        ("head_top_left", ("left", "top")),
        ("head_top_right", ("top", "right")),
        ("head_bottom_right", ("right", "bottom")),
        ("head_bottom_left", ("bottom", "left")),
    )
    bins_by_corner = {}
    for index, corner in enumerate(corners):
        x0 = max(0, int(math.floor(corner.u_px)) - margin)
        y0 = max(0, int(math.floor(corner.v_px)) - margin)
        x1 = min(width, int(math.ceil(corner.u_px)) + margin + 1)
        y1 = min(height, int(math.ceil(corner.v_px)) + margin + 1)
        points = cv2.findNonZero(raw_edges[y0:y1, x0:x1])
        relative = (
            numpy.empty((0, 2), dtype=numpy.float64) if points is None
            else points.reshape(-1, 2).astype(numpy.float64)
        )
        relative += numpy.array((x0 - corner.u_px, y0 - corner.v_px))
        corner_name, arm_names = names_and_arms[index]
        bins_by_corner[corner_name] = {}
        for arm_name, neighbor in zip(
            arm_names, (corners[index - 1], corners[(index + 1) % 4])
        ):
            direction = numpy.array(
                (neighbor.u_px - corner.u_px, neighbor.v_px - corner.v_px)
            )
            direction /= numpy.linalg.norm(direction)
            along = relative @ direction
            normal = relative @ numpy.array((-direction[1], direction[0]))
            selected = (
                (along >= 1.5) & (along <= radius) & (numpy.abs(normal) <= 2.5)
            )
            bins = numpy.unique(
                numpy.floor(along[selected] + 0.5).astype(numpy.int32)
            )
            bins_by_corner[corner_name][arm_name] = len(bins)
    return MetricCornerArmSupport(radius, 2, bins_by_corner)


def _quadrilateral_edge_support(
    cv2,
    edge_mask,
    corners: Sequence[ImagePoint],
    *,
    tolerance_px: float | None = None,
) -> _QuadrilateralEdgeSupport:
    """Measure how much of every proposed side is backed by cutout pixels.

    Sampling uses a distance transform so a fitted sub-pixel line may be a
    few pixels away from a thick Canny edge.  End points are trimmed because
    rounded/dilated corners are noisy.  The bottom is sampled in two outer
    intervals so the central stem attachment is an allowed notch, not false
    evidence against an otherwise real head border.
    """

    import numpy

    ordered = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = ordered
    mean_width = (
        _distance(top_left, top_right) + _distance(bottom_left, bottom_right)
    ) / 2.0
    mean_height = (
        _distance(top_left, bottom_left) + _distance(top_right, bottom_right)
    ) / 2.0
    if tolerance_px is None:
        tolerance_px = min(5.0, max(2.0, 0.03 * min(mean_width, mean_height)))
    tolerance_px = float(tolerance_px)

    if edge_mask is None or edge_mask.size == 0 or cv2.countNonZero(edge_mask) == 0:
        return _QuadrilateralEdgeSupport(0.0, 0.0, 0.0, 0.0, 0.0, tolerance_px)

    height, width = edge_mask.shape[:2]
    x0, y0, x1, y1 = 0, 0, width, height
    if (math.isfinite(tolerance_px) and tolerance_px >= 0.
            and _corners_inside_image(ordered, edge_mask.shape)):
        # Support only asks whether distance <= tolerance, not its exact value
        # farther away. DIST_L2/mask3 has minimum step cost .955: any qualifying
        # path fits inside this conservative margin, including sample rounding.
        # Outside-image/invalid inputs retain the original full-image behavior.
        margin = 2 * math.ceil(tolerance_px) + 2
        x0 = max(0, math.floor(min(p.u_px for p in ordered)) - margin)
        y0 = max(0, math.floor(min(p.v_px for p in ordered)) - margin)
        x1 = min(width, math.ceil(max(p.u_px for p in ordered)) + margin + 1)
        y1 = min(height, math.ceil(max(p.v_px for p in ordered)) + margin + 1)
    edge_pixels = numpy.asarray(edge_mask[y0:y1, x0:x1]) > 0
    distance_input = numpy.where(edge_pixels, 0, 255).astype(numpy.uint8)
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)

    def segment_support(
        start: ImagePoint,
        end: ImagePoint,
        start_fraction: float,
        end_fraction: float,
    ) -> float:
        segment_length = _distance(start, end)
        sample_count = max(
            8,
            int(math.ceil(segment_length * max(0.0, end_fraction - start_fraction)))
            + 1,
        )
        fractions = numpy.linspace(start_fraction, end_fraction, sample_count)
        xs = numpy.rint(
            start.u_px + fractions * (end.u_px - start.u_px)
        ).astype(numpy.int32)
        ys = numpy.rint(
            start.v_px + fractions * (end.v_px - start.v_px)
        ).astype(numpy.int32)
        xs = numpy.clip(xs, 0, max(0, width - 1))
        ys = numpy.clip(ys, 0, max(0, height - 1))
        return float(numpy.mean(distance[ys - y0, xs - x0] <= tolerance_px))

    return _QuadrilateralEdgeSupport(
        top=segment_support(top_left, top_right, 0.08, 0.92),
        right=segment_support(top_right, bottom_right, 0.08, 0.92),
        bottom_left=segment_support(bottom_left, bottom_right, 0.08, 0.40),
        bottom_right=segment_support(bottom_left, bottom_right, 0.60, 0.92),
        left=segment_support(top_left, bottom_left, 0.08, 0.92),
        tolerance_px=tolerance_px,
    )


def _fit_raw_edge_side_in_band(
    cv2,
    edge_points,
    start: ImagePoint,
    end: ImagePoint,
    *,
    band_px: float,
    intervals: Sequence[tuple[float, float]],
    outward_sign: float = 0.0,
    fixed_direction: tuple[float, float] | None = None,
    minimum_coverage: float = 0.45,
    color_support_mask=None,
):
    """Fit one proposed side from nearby raw pixels without connectivity.

    The topology side supplies only a search band. For every pixel-sized bin
    along that side, one untouched Canny pixel is selected by the requested
    outer/nearest policy and a robust line is fitted through those samples.
    The physical fitter first associates one coherent rail within that same band,
    so nearer fragments cannot discard its pixels before the robust fit.
    This prevents a closed topology mask from measuring its own proposal.
    """

    import numpy

    dx = end.u_px - start.u_px
    dy = end.v_px - start.v_px
    length = math.hypot(dx, dy)
    if length <= 1e-6 or len(edge_points) == 0:
        return None, edge_points[:0]
    tangent = numpy.array([dx / length, dy / length], dtype=numpy.float64)
    normal = numpy.array([-tangent[1], tangent[0]], dtype=numpy.float64)
    relative = edge_points - numpy.array(
        [start.u_px, start.v_px],
        dtype=numpy.float64,
    )
    along_px = relative @ tangent
    along_fraction = along_px / length
    normal_offset = relative @ normal
    interval_mask = numpy.zeros(len(edge_points), dtype=bool)
    expected_length_px = 0.0
    for interval_start, interval_end in intervals:
        interval_mask |= (
            (along_fraction >= interval_start)
            & (along_fraction <= interval_end)
        )
        expected_length_px += max(0.0, interval_end - interval_start) * length
    candidate_mask = interval_mask & (numpy.abs(normal_offset) <= band_px)
    candidate_indices = numpy.flatnonzero(candidate_mask)
    candidates = edge_points[candidate_indices]
    if len(candidates) == 0:
        return None, candidates

    # A QR edge parallel to the outer border can enter the wider search band.
    # For known silhouette sides, retain the outermost raw pixel in each
    # tangent bin. This makes the topology proposal a search corridor only;
    # an interior QR edge cannot win merely because it lies closer to the
    # proposal. Legacy callers may still request closest-to-proposal samples
    # with outward_sign=0.
    # Do not use numpy.rint here: its ties-to-even rule collapses adjacent
    # integer pixels whenever the fitted line centre lies on a half pixel.
    candidate_bins = numpy.floor(
        along_px[candidate_indices] + 0.5
    ).astype(numpy.int32)
    candidate_offsets = normal_offset[candidate_indices]
    pass

    candidates = coherent_metric_rail_points(
        cv2,
        candidates,
        candidate_bins,
        start,
        end,
        band_px=band_px,
        expected_length_px=expected_length_px,
        minimum_coverage=minimum_coverage,
        fixed_direction=fixed_direction,
        outward_sign=outward_sign,
        color_support_mask=color_support_mask,
    )
    if len(candidates) == 0:
        return None, candidates
    relative_candidates = candidates - numpy.array((start.u_px, start.v_px))
    candidate_bins = numpy.floor(
        relative_candidates @ tangent + 0.5
    ).astype(numpy.int32)
    candidate_offsets = relative_candidates @ normal
    nearest_by_bin: dict[int, int] = {}
    if outward_sign:
        sample_order = numpy.argsort(-outward_sign * candidate_offsets)
    else:
        sample_order = numpy.argsort(numpy.abs(candidate_offsets))
    for local_index in sample_order:
        nearest_by_bin.setdefault(
            int(candidate_bins[local_index]),
            int(local_index),
        )
    selected = candidates[list(nearest_by_bin.values())]
    minimum_points = max(5, int(math.ceil(0.18 * expected_length_px)))
    if len(selected) < minimum_points:
        return None, candidates

    def robust_fit(points):
        fitted = cv2.fitLine(
            points.astype(numpy.float32).reshape(-1, 1, 2),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        direction = numpy.array(
            [float(fitted[0]), float(fitted[1])],
            dtype=numpy.float64,
        )
        direction_norm = float(numpy.linalg.norm(direction))
        if direction_norm <= 1e-9:
            return None
        direction /= direction_norm
        if float(direction @ tangent) < 0.0:
            direction *= -1.0
        point = numpy.array(
            [float(fitted[2]), float(fitted[3])],
            dtype=numpy.float64,
        )
        return point, direction

    def fixed_direction_fit(points):
        direction = numpy.asarray(fixed_direction, dtype=numpy.float64).reshape(2)
        direction_norm = float(numpy.linalg.norm(direction))
        if direction_norm <= 1e-9:
            return None
        direction /= direction_norm
        if float(direction @ tangent) < 0.0:
            direction *= -1.0
        angle_cosine = max(-1.0, min(1.0, float(direction @ tangent)))
        if math.degrees(math.acos(angle_cosine)) > 22.0:
            return None
        fit_normal = numpy.array(
            [-direction[1], direction[0]],
            dtype=numpy.float64,
        )
        offsets = points @ fit_normal
        offset = float(numpy.median(offsets))
        residuals = numpy.abs(offsets - offset)
        inlier_limit_px = max(1.25, min(2.5, 0.50 * band_px))
        inliers = points[residuals <= inlier_limit_px]
        if len(inliers) < minimum_points:
            return None
        offset = float(numpy.median(inliers @ fit_normal))
        along_center = float(numpy.median(inliers @ direction))
        point = along_center * direction + offset * fit_normal
        return point, direction

    first_fit = (
        fixed_direction_fit(selected)
        if fixed_direction is not None
        else robust_fit(selected)
    )
    if first_fit is None:
        return None, candidates
    fit_point, fit_direction = first_fit
    angle_cosine = max(-1.0, min(1.0, float(fit_direction @ tangent)))
    # A minimum-area/topology rectangle can be noticeably more axis-aligned
    # than the true perspective edge.  Keep enough angular freedom to recover
    # that raw edge while the narrow spatial band still rejects unrelated
    # structure.
    if fixed_direction is None and math.degrees(math.acos(angle_cosine)) > 22.0:
        return None, candidates

    fit_normal = numpy.array(
        [-fit_direction[1], fit_direction[0]],
        dtype=numpy.float64,
    )
    residuals = numpy.abs((selected - fit_point) @ fit_normal)
    inlier_limit_px = max(1.25, min(2.5, 0.50 * band_px))
    inliers = selected[residuals <= inlier_limit_px]
    if len(inliers) < minimum_points:
        return None, candidates
    final_fit = (
        fixed_direction_fit(inliers)
        if fixed_direction is not None
        else robust_fit(inliers)
    )
    if final_fit is None:
        return None, candidates
    fit_point, fit_direction = final_fit
    angle_cosine = max(-1.0, min(1.0, float(fit_direction @ tangent)))
    if fixed_direction is None and math.degrees(math.acos(angle_cosine)) > 20.0:
        return None, candidates

    fit_normal = numpy.array(
        [-fit_direction[1], fit_direction[0]],
        dtype=numpy.float64,
    )
    midpoint = numpy.array(
        [(start.u_px + end.u_px) / 2.0, (start.v_px + end.v_px) / 2.0],
        dtype=numpy.float64,
    )
    if abs(float((midpoint - fit_point) @ fit_normal)) > band_px + 1.0:
        return None, candidates

    candidate_residuals = numpy.abs((candidates - fit_point) @ fit_normal)
    evidence = candidates[candidate_residuals <= inlier_limit_px]
    projected_bins = numpy.unique(
        numpy.floor(
            (evidence - fit_point) @ fit_direction + 0.5
        ).astype(numpy.int32)
    )
    coverage = len(projected_bins) / max(expected_length_px, 1.0)
    if coverage < minimum_coverage:
        return None, evidence
    return (
        (
            float(fit_point[0]),
            float(fit_point[1]),
            float(fit_direction[0]),
            float(fit_direction[1]),
        ),
        evidence,
    )


def _image_line_intersection(first, second) -> ImagePoint | None:
    first_x, first_y, first_dx, first_dy = first
    second_x, second_y, second_dx, second_dy = second
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) < 0.15:
        return None
    offset_x = second_x - first_x
    offset_y = second_y - first_y
    scale = (offset_x * second_dy - offset_y * second_dx) / denominator
    x = first_x + scale * first_dx
    y = first_y + scale * first_dy
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return ImagePoint(float(x), float(y))


def _parallel_side_lengths_comparable(
    corners: Sequence[ImagePoint],
    *,
    maximum_ratio: float = 1.45,
) -> bool:
    """Keep a learned real-camera trapezoid from growing one background rail.

    Both side fits already use one shared direction, so they are parallel by
    construction. This remaining check prevents endpoint recovery from making
    one physical side implausibly longer than the other.
    """

    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    left_length = _distance(top_left, bottom_left)
    right_length = _distance(top_right, bottom_right)
    shorter_length = min(left_length, right_length)
    if shorter_length <= 1e-6:
        return False
    return max(left_length, right_length) / shorter_length <= maximum_ratio


def _raw_edge_points_in_side_bounds(
    cv2, raw_edges, start, end, *, band_px, intervals,
):
    """Keep an enclosing box of the unchanged finite side corridor.

    This is only a workload filter. The fitter still applies its exact
    tangent intervals and normal band to these original-coordinate pixels.
    A one-pixel guard contains rounding at the rectangular envelope; disjoint
    intervals retain their intervening gap until the original interval test.
    """
    import numpy

    dx, dy = end.u_px - start.u_px, end.v_px - start.v_px
    length = math.hypot(dx, dy)
    if length <= 1e-6 or not intervals:
        return numpy.empty((0, 2), dtype=numpy.float64)
    fractions = [value for interval in intervals for value in interval]
    first, last = min(fractions), max(fractions)
    xs = (start.u_px + first * dx, start.u_px + last * dx)
    ys = (start.v_px + first * dy, start.v_px + last * dy)
    x_margin, y_margin = band_px * abs(dy) / length, band_px * abs(dx) / length
    x0, x1 = math.floor(min(xs) - x_margin) - 1, math.ceil(max(xs) + x_margin) + 2
    y0, y1 = math.floor(min(ys) - y_margin) - 1, math.ceil(max(ys) + y_margin) + 2
    rows, cols = raw_edges.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(cols, x1), min(rows, y1)
    if x0 >= x1 or y0 >= y1:
        return numpy.empty((0, 2), dtype=numpy.float64)
    locations = cv2.findNonZero(raw_edges[y0:y1, x0:x1])
    if locations is None:
        return numpy.empty((0, 2), dtype=numpy.float64)
    # findNonZero keeps the same row-major order as full-image extraction.
    return locations.reshape(-1, 2).astype(numpy.float64) + (x0, y0)


def _raw_side_evidence_and_corners(
    cv2,
    raw_edges,
    rough_corners: Sequence[ImagePoint],
    *,
    maximum_parallel_side_length_ratio: float | None = None,
    maximum_band_px: float | None = None,
    prefer_outer_metric_rail: bool = False,
    color_support_mask=None,
):
    """Refit a common-sided head trapezoid from outer raw-Canny evidence.

    The left and right sides share a direction measured from both raw sides.
    Top and bottom retain their perspective slopes. Corners are intersections
    of these four measured rails; background runs cannot extend their ends.
    """

    import numpy

    evidence_mask = numpy.zeros(raw_edges.shape[:2], dtype=numpy.uint8)
    # Extract each finite corridor without scanning the whole room image.
    edge_points = numpy.empty((0, 2), dtype=numpy.float64)
    top_left, top_right, bottom_right, bottom_left = order_corners(rough_corners)
    widths = (
        _distance(top_left, top_right),
        _distance(bottom_left, bottom_right),
    )
    heights = (
        _distance(top_left, bottom_left),
        _distance(top_right, bottom_right),
    )
    minimum_extent = min((*widths, *heights))
    band_px = float(
        max(
            3,
            min(
                8,
                int(round(0.08 * minimum_extent)),
            ),
        )
    )
    if maximum_band_px is not None:
        if not math.isfinite(maximum_band_px) or maximum_band_px <= 0.0:
            raise ValueError("maximum_band_px must be finite and positive")
        band_px = min(band_px, float(maximum_band_px))
    side_specs = {
        # Top and bottom both run left-to-right, so their fitted normal points
        # down in image coordinates. The silhouette exterior is therefore the
        # negative-normal side for the top and the positive-normal side for the
        # bottom. Reversing these signs selects inner frame/QR edges instead.
        "top": (top_left, top_right, ((0.08, 0.92),), -1.0, 0.50),
        "right": (top_right, bottom_right, ((0.08, 0.92),), -1.0, 0.55),
        # The stand stem legitimately interrupts the middle of the lower edge.
        "bottom": (
            bottom_left,
            bottom_right,
            ((0.08, 0.40), (0.60, 0.92)),
            1.0,
            0.40,
        ),
        "left": (top_left, bottom_left, ((0.08, 0.92),), 1.0, 0.55),
    }
    side_points = {}

    def fit_side(name: str, *, fixed_direction=None):
        start, end, intervals, outward_sign, minimum_coverage = side_specs[name]
        if not prefer_outer_metric_rail:
            # A metric projection identifies which of several nearby parallel
            # rails belongs to the stand. Rank coherent raw rails first, then
            # select their nearest pixels instead of outermost heater/QR edges.
            outward_sign = 0.0
        if name not in side_points:
            side_points[name] = _raw_edge_points_in_side_bounds(
                cv2, raw_edges, start, end, band_px=band_px, intervals=intervals)
        return _fit_raw_edge_side_in_band(
            cv2,
            side_points[name],
            start,
            end,
            band_px=band_px,
            intervals=intervals,
            outward_sign=outward_sign,
            fixed_direction=fixed_direction,
            minimum_coverage=minimum_coverage,
            color_support_mask=color_support_mask,
        )

    top, top_evidence = fit_side("top")
    bottom, bottom_evidence = fit_side("bottom")

    shared_direction = None
    preliminary_right, _right_evidence = fit_side("right")
    preliminary_left, _left_evidence = fit_side("left")
    if preliminary_right is not None and preliminary_left is not None:
        right_direction = numpy.array(preliminary_right[2:4], dtype=numpy.float64)
        left_direction = numpy.array(preliminary_left[2:4], dtype=numpy.float64)
        if float(right_direction @ left_direction) < 0.0:
            left_direction *= -1.0
        direction_cosine = max(
            -1.0,
            min(1.0, float(right_direction @ left_direction)),
        )
        if math.degrees(math.acos(direction_cosine)) <= 12.0:
            combined = right_direction + left_direction
            combined_norm = float(numpy.linalg.norm(combined))
            if combined_norm > 1e-9:
                combined /= combined_norm
                shared_direction = (float(combined[0]), float(combined[1]))

    if shared_direction is None:
        right = left = None
        right_evidence = left_evidence = edge_points[:0]
    else:
        right, right_evidence = fit_side(
            "right",
            fixed_direction=shared_direction,
        )
        left, left_evidence = fit_side(
            "left",
            fixed_direction=shared_direction,
        )

    lines = (top, right, bottom, left)
    evidences = (top_evidence, right_evidence, bottom_evidence, left_evidence)
    for line, evidence in zip(lines, evidences):
        if len(evidence):
            xs = numpy.rint(evidence[:, 0]).astype(numpy.int32)
            ys = numpy.rint(evidence[:, 1]).astype(numpy.int32)
            valid = (
                (xs >= 0)
                & (xs < evidence_mask.shape[1])
                & (ys >= 0)
                & (ys < evidence_mask.shape[0])
            )
            evidence_mask[ys[valid], xs[valid]] = 255

    if any(line is None for line in lines):
        return evidence_mask, None
    corners = (
        _image_line_intersection(top, left),
        _image_line_intersection(top, right),
        _image_line_intersection(bottom, right),
        _image_line_intersection(bottom, left),
    )
    if any(point is None for point in corners):
        return evidence_mask, None
    ordered_corners = order_corners(corners)
    if (
        maximum_parallel_side_length_ratio is not None
        and not _parallel_side_lengths_comparable(
            ordered_corners,
            maximum_ratio=maximum_parallel_side_length_ratio,
        )
    ):
        return evidence_mask, None
    return evidence_mask, ordered_corners


@dataclass(frozen=True)
class HeadBorderSeed:
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    corridor_half_width_px: float
    source: str


def select_head_border_seed(
    *, model_profile: HeadModel,
    current_head_proposal_corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint],
) -> HeadBorderSeed:
    """Bound refinement around the caller's validated current-image proposal."""
    if current_head_proposal_corners is None:
        raise ValueError("head refinement requires a current image proposal")
    return HeadBorderSeed(
        corners=current_head_proposal_corners,
        corridor_half_width_px=model_corridor_half_width_px(
            current_head_proposal_corners, model_profile=model_profile),
        source="current_head_proposal",
    )


@dataclass(frozen=True)
class RefinedHeadMeasurement:
    accepted: bool
    reason: str
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None
    evidence_mask: object
    support: _QuadrilateralEdgeSupport | None
    # Rejected intersections are diagnostics, never accepted measurements.
    candidate_corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None = None
    corner_arm_support: MetricCornerArmSupport | None = None


def model_corridor_half_width_px(
    projected_corners: Sequence[ImagePoint],
    *,
    model_profile: HeadModel,
) -> float:
    """Bound current-pixel search by measured dimensions and localization error.

    The proposal is a search hint. Four measured rails and current corner arms
    still determine whether the resulting border is usable.
    """
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

    dimensional_tolerance_px = minimum_extent_px * max(
        model_profile.tolerance_m / model_profile.head_width_m,
        model_profile.tolerance_m / model_profile.head_height_m,
    )
    return min(
        8.0,
        max(
            4.0,
            2.0 + 0.03 * minimum_extent_px,
            2.0 + dimensional_tolerance_px,
        ),
    )


def refine_projected_head_border(
    cv2,
    raw_edges,
    projected_corners: Sequence[ImagePoint],
    *,
    maximum_parallel_side_length_ratio: float = 1.30,
    corridor_half_width_px: float | None = None,
    prefer_outer_metric_rail: bool = False,
    color_support_mask=None,
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
        maximum_parallel_side_length_ratio=maximum_parallel_side_length_ratio,
        maximum_band_px=corridor_half_width_px,
        # A radiator rail can continue beyond a real head corner. The metric
        # model already predicts the four corner neighbourhoods, so intersect
        # the independently fitted top/bottom and side lines instead of growing
        # side runs along aligned background structure.
        prefer_outer_metric_rail=prefer_outer_metric_rail,
        color_support_mask=color_support_mask,
    )
    if corners is None:
        return RefinedHeadMeasurement(
            False, "model_corridor_refinement_unavailable", None, evidence, None
        )
    if not _well_formed_quadrilateral(corners) or not _corners_inside_image(
        corners, raw_edges.shape
    ):
        return RefinedHeadMeasurement(
            False, "model_refinement_geometry_inconsistent", None, evidence, None,
            candidate_corners=tuple(corners),
        )
    support = _quadrilateral_edge_support(cv2, evidence, corners)
    if not support.accepted:
        return RefinedHeadMeasurement(
            False, "model_corridor_support_insufficient", None, evidence, support,
            candidate_corners=tuple(corners),
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
            candidate_corners=refined,
        )
    corner_support = metric_corner_arm_support(cv2, raw_edges, refined)
    if not corner_support.accepted:
        return RefinedHeadMeasurement(
            False,
            "model_corner_evidence_insufficient",
            None,
            evidence,
            support,
            candidate_corners=refined,
            corner_arm_support=corner_support,
        )
    return RefinedHeadMeasurement(
        True,
        "fresh_current_frame_refinement",
        corners,
        evidence,
        support,
        candidate_corners=refined,
        corner_arm_support=corner_support,
    )


OUTER_HEAD_SEARCH_GROWTH_FACTORS = (1.10, 1.25)


@dataclass(frozen=True)
class HeadOuterBorderEvidence:
    accepted: bool
    reason: str
    original_corners: tuple[ImagePoint, ...] | None
    recovered_corners: tuple[ImagePoint, ...] | None
    attempted_growth_factors: tuple[float, ...] = ()
    recovered: bool = False
    selection_source: str = "current_enclosing_raw_borders"
    scale_identifiability: str = "conditional_on_candidate_and_measured_profile"
    neutral_proposal_corners: tuple[ImagePoint, ...] | None = None
    independently_resolved: bool = False
    proposal_inward_shift_px: float = 0.0
    attempted_raw_refinements: int = 0
    current_raw_alternatives: tuple[tuple[ImagePoint, ...], ...] = ()
    head_size_m: tuple[float, float] | None = None
    largest_inset_size_m: tuple[float, float] | None = None
    model_tolerance_m: float | None = None


def _encloses(outer, inner, tolerance=1.5):
    """Convex containment using signed edge distances, without fitted poses."""
    area = sum(a.u_px*b.v_px-b.u_px*a.v_px for a, b in zip(outer, outer[1:]+outer[:1]))
    sign = 1. if area >= 0 else -1.
    return all(sign*((b.u_px-a.u_px)*(p.v_px-a.v_px)-(b.v_px-a.v_px)*(p.u_px-a.u_px))
               >= -tolerance*max(_distance(a, b), 1.)
               for a, b in zip(outer, outer[1:]+outer[:1]) for p in inner)


def _inward_shift(selected, reference):
    """Largest perpendicular inward displacement of an original current rail."""
    selected, reference = tuple(order_corners(selected)), tuple(order_corners(reference))
    area = sum(a.u_px*b.v_px-b.u_px*a.v_px for a, b in zip(selected, selected[1:]+selected[:1]))
    sign = 1. if area >= 0. else -1.
    return max(0., max(-sign*((b.u_px-a.u_px)*(point.v_px-a.v_px)
                      -(b.v_px-a.v_px)*(point.u_px-a.u_px))/max(_distance(a, b), 1.)
                      for a, b, p, q in zip(selected, selected[1:]+selected[:1],
                                             reference, reference[1:]+reference[:1])
                      for point in (p, q)))


def _span(corners):
    a, b, c, d = order_corners(corners)
    return ((_distance(a, b)+_distance(c, d))/2.,
            (_distance(a, d)+_distance(b, c))/2.)


def _independent_enclosure(selected, original, head_size, inset_size, tolerance):
    """Reject the two edges of a paper rail as apparent head-frame recovery.

    Both dimensions must grow by at least the measured head/paper separation,
    reduced only by the profile's recorded metrology tolerance. A tiny larger
    contour is not independent evidence for the physical head boundary.
    """
    try:
        if (len(head_size) != 2 or len(inset_size) != 2
                or not all(math.isfinite(value) and value > 0. for value in (*head_size, *inset_size))
                or not math.isfinite(tolerance) or tolerance < 0.
                or any(inset+tolerance >= head for head, inset in zip(head_size, inset_size))):
            return False
        return all(outer/inner >= head/(inset+tolerance)
                   for outer, inner, head, inset in zip(
                       _span(selected), _span(original), head_size, inset_size))
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def validated_current_head_boundary(evidence, *, corners,
                                    require_independent=False):
    """Bind an outer-boundary decision to its exact current fit and profile.

    This is a producer consistency check, not a substitute for candidate
    association, raw pixel verification, freshness or metric pose quality.
    """
    if not isinstance(evidence, HeadOuterBorderEvidence):
        return False
    try:
        selected = tuple(corners)
        original = tuple(evidence.original_corners)
        proposal = tuple(evidence.neutral_proposal_corners)
        for quad in (selected, original, proposal, *evidence.current_raw_alternatives):
            if (len(quad) != 4 or not all(math.isfinite(value) for point in quad
                    for value in (point.u_px, point.v_px))
                    or not _well_formed_quadrilateral(quad)):
                return False
        if not (evidence.accepted is True
                and evidence.selection_source == "current_enclosing_raw_borders"
                and evidence.reason in {"maximal_current_head_border", "current_raw_outer_head_recovered"}
                and evidence.recovered_corners == selected
                and original in evidence.current_raw_alternatives
                and selected in evidence.current_raw_alternatives):
            return False
        independent = bool(evidence.recovered is True and evidence.independently_resolved is True
            and selected != original and _encloses(selected, original)
            and 1.03*_polygon_area(original) <= _polygon_area(selected)
            <= 1.70*_polygon_area(original)
            and _independent_enclosure(selected, original, evidence.head_size_m,
                                       evidence.largest_inset_size_m, evidence.model_tolerance_m))
        return not require_independent or independent
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return False


def select_current_outer_head_border(cv2, raw_edges, *, model_profile, refinement,
                                     corridor_half_width_px, neutral_proposal_corners=None,
                                     deadline_monotonic_sec=None, prefer_outer_metric_rail=False,
                                     color_support_mask=None):
    """Prefer an enclosing complete border; missing outward pixels stay missing."""
    original = refinement.corners
    if not refinement.accepted or original is None:
        return refinement, HeadOuterBorderEvidence(
            False, "current_head_border_unavailable", original, None)
    original = tuple(order_corners(original))
    proposal = tuple(order_corners(neutral_proposal_corners or original))
    center = (sum(p.u_px for p in proposal)/4, sum(p.v_px for p in proposal)/4)
    height = (_distance(proposal[0], proposal[3])+_distance(proposal[1], proposal[2]))/2
    # The two-pixel rail localization allowance is already used by the joint
    # current-gradient locator. Physical metrology adds its existing tolerance.
    inward_allowance = 2. + height*model_profile.tolerance_m/model_profile.head_height_m
    inward = _inward_shift(original, proposal)
    head_size = (model_profile.head_width_m, model_profile.head_height_m)
    inset_size = (model_profile.qr_panel_width_m, model_profile.qr_panel_height_m)
    selected, area = refinement, _polygon_area(original)
    tried, alternatives = [], [original]
    attempted = 0
    # Do not lose the original independently acquired outer frame when a wide
    # raw corridor selects an inner paper rail. The reference only locates a
    # narrower current-pixel fit; it never supplies accepted corners itself.
    if inward > inward_allowance:
        check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_anchored_refinement")
        attempted += 1
        anchored = refine_projected_head_border(
            cv2, raw_edges, proposal,
            corridor_half_width_px=min(2., corridor_half_width_px))
        check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_anchored_refinement")
        if anchored.accepted and anchored.corners is not None:
            anchored_corners = tuple(order_corners(anchored.corners))
            if (_encloses(anchored_corners, original)
                    and 1.03*area <= _polygon_area(anchored_corners) <= 1.70*area):
                selected, area = anchored, _polygon_area(anchored_corners)
                alternatives.append(anchored_corners)
    for growth in OUTER_HEAD_SEARCH_GROWTH_FACTORS:
        check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_outer_refinement")
        proposed = tuple(ImagePoint(center[0]+(p.u_px-center[0])*growth,
                                    center[1]+(p.v_px-center[1])*growth) for p in proposal)
        tried.append(growth)
        attempted += 1
        current = refine_projected_head_border(
            cv2, raw_edges, proposed, corridor_half_width_px=corridor_half_width_px)
        check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_outer_refinement")
        if not current.accepted or current.corners is None:
            continue
        corners = tuple(order_corners(current.corners))
        larger = _polygon_area(corners)
        shifted = math.hypot(sum(p.u_px for p in corners)/4-center[0],
                             sum(p.v_px for p in corners)/4-center[1])
        if (1.03*area <= larger <= 1.70*_polygon_area(original)
                and shifted <= max(2., .08*height) and _encloses(corners, original)
                and _encloses(corners, tuple(selected.corners))):
            selected, area = current, larger
            alternatives.append(corners)
    if prefer_outer_metric_rail and attempted < 3:
        # Resolve mixed rim sides with one coherent enclosing fit. A failed
        # outward fit must NOT discard an already verified original: doing so
        # can hide a competing boundary and falsely promote an inset frame.
        check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_enclosing_rails")
        attempted += 1
        current = refine_projected_head_border(cv2, raw_edges, original,
            corridor_half_width_px=min(8., corridor_half_width_px+2.), prefer_outer_metric_rail=True,
            color_support_mask=color_support_mask)
        check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_enclosing_rails")
        if (current.accepted and current.corners is not None
                and _encloses(tuple(current.corners), tuple(selected.corners))
                and 1.03*area <= _polygon_area(current.corners) <= 1.70*_polygon_area(original)):
            selected, area = current, _polygon_area(current.corners)
            alternatives.append(tuple(current.corners))
    recovered = selected is not refinement
    remaining_inward = _inward_shift(tuple(selected.corners), proposal)
    accepted = remaining_inward <= inward_allowance
    return selected, HeadOuterBorderEvidence(
        accepted, ("current_physical_head_boundary_unresolved" if not accepted else
                   "current_raw_outer_head_recovered" if recovered else "maximal_current_head_border"),
        original, tuple(selected.corners), tuple(tried), recovered,
        neutral_proposal_corners=proposal,
        independently_resolved=bool(accepted and recovered and _independent_enclosure(
            tuple(selected.corners), original, head_size, inset_size, model_profile.tolerance_m)),
        proposal_inward_shift_px=inward,
        attempted_raw_refinements=attempted,
        current_raw_alternatives=tuple(alternatives), head_size_m=head_size,
        largest_inset_size_m=inset_size, model_tolerance_m=model_profile.tolerance_m)


def refine_current_physical_head(
    cv2, raw_edges, *, model_profile, proposal_corners,
    deadline_monotonic_sec=None,
    prefer_outer_metric_rail=False,
    color_support_mask=None,
) -> tuple[RefinedHeadMeasurement, HeadOuterBorderEvidence, HeadBorderSeed]:
    """Apply the metric fitter's unchanged raw-border policy to a valid proposal.

    The caller owns proposal/crop validation and candidate association. Optional
    work deadlines stop between raw fits. Evidence has no 3D angle or admission authority;
    the usual fit, source freshness and observer gates remain separate.
    """
    seed = select_head_border_seed(
        model_profile=model_profile, current_head_proposal_corners=proposal_corners,
    )
    check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_raw_refinement")
    refinement = refine_projected_head_border(
        cv2, raw_edges, seed.corners, corridor_half_width_px=seed.corridor_half_width_px,
        color_support_mask=color_support_mask,
    )
    check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_raw_refinement")
    refinement, outer_recovery = select_current_outer_head_border(
        cv2, raw_edges, model_profile=model_profile, refinement=refinement,
        corridor_half_width_px=seed.corridor_half_width_px,
        neutral_proposal_corners=seed.corners,
        prefer_outer_metric_rail=prefer_outer_metric_rail,
        color_support_mask=color_support_mask,
        deadline_monotonic_sec=deadline_monotonic_sec,
    )
    return refinement, outer_recovery, seed


@dataclass(frozen=True)
class ProjectedHeadSize:
    depth_m: float
    depth_uncertainty_m: float
    height_px: float
    frontal_width_px: float
    min_height_px: float
    max_height_px: float
    max_width_px: float

    def accepts(self, width, height, *, refinement_allowance=False):
        # Rough locators can move during the existing raw-border refinement.
        low = .85 * height if refinement_allowance else height
        high = 1.15 * 1.25 * height if refinement_allowance else height
        width_low = .85 * width if refinement_allowance else width
        return (low <= self.max_height_px and high >= self.min_height_px
                and width_low <= self.max_width_px)


def projected_head_size(*, model_profile, depth_m, fx, fy,
                        depth_uncertainty_m=.02, center_normalized=(0., 0.),
                        camera_vertical=(0., 1., 0.), pixel_margin=4., position_uncertainty_m=None):
    """Bound projected upright side lengths using calibrated camera depth.

    The vertical vector is the stand's world/base up direction expressed in
    optical-camera coordinates. Unknown yaw is covered by the head radius;
    distance/position error and model tolerance expand the search interval.
    A frontal width is reported, but no lower width is imposed on oblique heads.
    """
    values = (depth_m, fx, fy, depth_uncertainty_m, pixel_margin,
              *center_normalized, *camera_vertical)
    if (any(not math.isfinite(v) for v in values) or min(depth_m, fx, fy) <= 0
            or min(depth_uncertainty_m, pixel_margin) < 0):
        raise ValueError("metric head search requires finite physical camera depth and calibration")
    norm = math.sqrt(sum(v*v for v in camera_vertical))
    if position_uncertainty_m is not None and (
            not math.isfinite(position_uncertainty_m) or position_uncertainty_m < 0.):
        raise ValueError("position uncertainty must be finite and nonnegative")
    if not .99 <= norm <= 1.01:
        raise ValueError("camera vertical must be a unit vector")
    ux, uy, uz = (v/norm for v in camera_vertical)
    x, y = (v*depth_m for v in center_normalized)
    h, w, d, tol = (model_profile.head_height_m, model_profile.head_width_m,
                    model_profile.head_depth_m, model_profile.tolerance_m)
    radius = math.hypot(w/2., d) + depth_uncertainty_m + tol
    xy_radius = math.hypot(w/2., d) + max(depth_uncertainty_m, position_uncertainty_m or 0.) + tol
    near, far = depth_m-radius, depth_m+radius
    hmin, hmax = h-tol, h+tol
    if hmin <= 0 or near <= .5*hmax*abs(uz) + .01:
        raise ValueError("possible head volume crosses the camera plane")
    # Exact projected length of a vertical segment at its nominal centre:
    # H ||(fx(ux Z-X uz), fy(uy Z-Y uz))|| / (Z²-(H uz/2)²).
    numerator = math.hypot(fx*(ux*depth_m-x*uz), fy*(uy*depth_m-y*uz))
    nominal = h*numerator/(depth_m**2-(h*uz/2.)**2)
    # Divide numerator and denominator by Z before bounding. Bounding their
    # shared Z independently makes a slightly tilted camera admit ~75--245 px
    # for a 137 px head. For |dX|, |dY|, |dZ| <= radius, the normalized-ray
    # error is <= radius * (1 + |X/Z|) / near. The triangle inequality then
    # bounds the projected vertical vector for every allowed side centre.
    xn, yn = center_normalized
    vertical_scale = math.hypot(fx*(ux-xn*uz), fy*(uy-yn*uz))
    scale_error = abs(uz)/near*math.hypot(fx*(xy_radius+radius*abs(xn)), fy*(xy_radius+radius*abs(yn)))
    lower = max(0., hmin*max(0., vertical_scale-scale_error)/far-pixel_margin)
    upper = hmax*(vertical_scale+scale_error)/(near*(1.-(hmax*uz/(2.*near))**2))+pixel_margin
    # Bound every horizontal yaw using the maximum projective magnification.
    max_ray = (math.hypot(x, y)+xy_radius+.5*hmax)/near
    ray_scale = math.sqrt(1.+max_ray**2)
    max_width = max(fx, fy)*math.hypot(w+tol, d+tol)*ray_scale/near+pixel_margin
    return ProjectedHeadSize(depth_m, depth_uncertainty_m, nominal, fx*w/depth_m,
                             lower, upper, max_width)


def metric_head_search(*, model_profile, depth_m, fx, fy, cx, cy, image_shape,
                       center=None, depth_uncertainty_m=.02,
                       camera_vertical=(0., 1., 0.), max_center_offset_ratio=1.5,
                       edge_region=None, position_uncertainty_m=None):
    """Build size bounds and, with declared position error, a head-volume prior.

    Position error bounds the camera X/Y centre independently of optical depth.
    Callers without that registration contract retain the broad legacy centre
    allowance; a depth measurement alone never establishes lateral accuracy.
    """
    pass

    location_known = center is not None
    center = (cx, cy) if center is None else center
    size = projected_head_size(model_profile=model_profile, depth_m=depth_m,
        fx=fx, fy=fy, depth_uncertainty_m=depth_uncertainty_m,
        center_normalized=((center[0]-cx)/fx, (center[1]-cy)/fy),
        camera_vertical=camera_vertical, position_uncertainty_m=position_uncertainty_m)
    center_bounds = None
    if position_uncertainty_m is not None and (
            not math.isfinite(position_uncertainty_m) or position_uncertainty_m < 0.):
        raise ValueError("position uncertainty must be finite and nonnegative")
    if location_known and position_uncertainty_m is not None:
        # A circular horizontal envelope covers every unknown head yaw. Project
        # its conservative camera-axis box, including floor/registration error.
        # Unlike a scalar centre-radius-plus-full-height window, this uses the
        # known upright construction to bound both image axes before rail quotas.
        from itertools import product
        pass
        norm = math.sqrt(sum(v*v for v in camera_vertical))
        vertical = tuple(v/norm for v in camera_vertical)
        radius = math.hypot(model_profile.head_width_m/2., model_profile.head_depth_m)
        radius += max(depth_uncertainty_m, position_uncertainty_m) + model_profile.tolerance_m
        half_height = model_profile.head_height_m/2. + position_uncertainty_m + model_profile.tolerance_m
        extents = tuple(radius*math.sqrt(max(0., 1.-v*v))+half_height*abs(v) for v in vertical)
        point = ((center[0]-cx)*depth_m/fx, (center[1]-cy)*depth_m/fy, depth_m)
        # Centre uncertainty is independent of the head's half-width/height.
        # Adding the object's extent to centre tolerance admits another object
        # immediately beside the selected stand. Preserve projective depth error.
        errors = (position_uncertainty_m+model_profile.tolerance_m,)*2 + (
            depth_uncertainty_m+model_profile.tolerance_m,)
        centres = tuple(product(*((p-e, p+e) for p,e in zip(point,errors))))
        center_us, center_vs = zip(*((fx*x/z+cx, fy*y/z+cy) for x, y, z in centres))
        center_bounds = (min(center_us)-4., min(center_vs)-4., max(center_us)+4., max(center_vs)+4.)
        vertices = tuple(product(*( (p-e, p+e) for p, e in zip(point, extents))))
        if min(p[2] for p in vertices) <= .01:
            raise ValueError("possible upright head volume crosses the camera plane")
        us, vs = zip(*((fx*x/z+cx, fy*y/z+cy) for x, y, z in vertices))
        rows, cols = image_shape[:2]
        bounds = (max(0, math.floor(min(us)-6.)), max(0, math.floor(min(vs)-6.)),
                  min(cols, math.ceil(max(us)+6.)+1), min(rows, math.ceil(max(vs)+6.)+1))
        if edge_region is not None:
            a, b, c, d = edge_region.bounds
            bounds = (max(a, bounds[0]), max(b, bounds[1]), min(c, bounds[2]), min(d, bounds[3]))
        edge_region = LidarHeadEdgeRegion((rows, cols), bounds)
    return CandidateHeadSearch(center, size.height_px, max_center_offset_ratio,
        edge_region=edge_region, pixel_size=size,
        center_tolerance_px=(None if location_known else math.hypot(*image_shape[:2])),
        center_bounds_px=center_bounds)


@dataclass(frozen=True)
class LidarHeadEdgeRegion:
    shape: tuple[int, int]
    bounds: tuple[int, int, int, int]  # Exclusive upper bounds, full-image pixels.

    def __post_init__(self):
        h, w = self.shape
        x0, y0, x1, y1 = self.bounds
        if (any(type(v) is not int for v in (*self.shape, *self.bounds))
                or not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h):
            raise ValueError("invalid LiDAR head edge region")

    def locator_edges(self, raw_edges):
        import numpy as np

        if raw_edges.shape != self.shape:
            raise ValueError("LiDAR head region must match the exact processing image")
        x0, y0, x1, y1 = self.bounds
        selected = np.zeros_like(raw_edges)
        selected[y0:y1, x0:x1] = raw_edges[y0:y1, x0:x1]
        return selected

    def contains(self, corners):
        x0, y0, x1, y1 = self.bounds
        return bool(corners) and all(
            x0 <= p.u_px < x1 and y0 <= p.v_px < y1 for p in corners)


_MAX_LINES_PER_DIRECTION = 32


@dataclass(frozen=True)
class HeadProposal:
    """Current 2D corners and crop bounds, relative to the input search image."""

    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    bounds_xyxy: tuple[int, int, int, int]
    head_bounds_xyxy: tuple[float, float, float, float]
    center_u_px: float
    center_v_px: float
    observed_height_px: float
    expected_height_ratio: float
    center_offset_head_heights: float
    raw_edge_support: float
    geometry_quality: float


@dataclass(frozen=True)
class HeadProposalResult:
    proposal: HeadProposal | None
    reason: str
    raw_verifications: int = 0


def _extent(corners):
    left, right, lower_right, lower_left = order_corners(corners)
    width = (_distance(left, right) + _distance(lower_left, lower_right)) / 2.0
    height = (_distance(left, lower_left) + _distance(right, lower_right)) / 2.0
    center = (
        sum(point.u_px for point in corners) / 4.0,
        sum(point.v_px for point in corners) / 4.0,
    )
    return width, height, center


def _line_groups(lines, *, expected_height, expected_center):
    groups = ([], [])
    if lines is None:
        return groups
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        first, last = (float(x1), float(y1)), (float(x2), float(y2))
        dx, dy = last[0] - first[0], last[1] - first[1]
        length = math.hypot(dx, dy)
        if not 0.55 * expected_height <= length <= 1.4 * expected_height:
            continue
        if (
            abs((first[0] + last[0]) / 2.0 - expected_center[0]) > 2.0 * expected_height
            or abs((first[1] + last[1]) / 2.0 - expected_center[1]) > 1.6 * expected_height
        ):
            continue
        if abs(dy) <= 0.30 * abs(dx):
            group, coordinate = groups[0], 0
        elif abs(dx) <= 0.30 * abs(dy):
            group, coordinate = groups[1], 1
        else:
            continue
        if first[coordinate] > last[coordinate]:
            first, last = last, first
        group.append((first, last))
    # Size, not proximity to the nominal center, orders the cheap locator.
    # Thus an off-center physical head is not displaced by tiny QR rectangles.
    for group in groups:
        group.sort(key=lambda pair: abs(math.log(math.dist(*pair) / expected_height)))
        del group[_MAX_LINES_PER_DIRECTION:]
    return groups


def _rough_proposals(groups, *, expected_height, expected_center, max_center_offset,
                     max_vertical_center_offset=None, deadline_monotonic_sec=None):
    ranked = []

    def add(points):
        corners = order_corners(tuple(ImagePoint(*point) for point in points))
        width, height, center = _extent(corners)
        if (not 0.70 <= height / expected_height <= 1.30
                or not 0.35 <= width / max(height, 1.0) <= 1.35
                or math.dist(center, expected_center) > max_center_offset * expected_height):
            return
        if (max_vertical_center_offset is not None
                and abs(center[1] - expected_center[1]) > max_vertical_center_offset * expected_height):
            return
        score = abs(math.log(height / expected_height)) + 0.2 * math.dist(center, expected_center) / expected_height
        if any(max(_distance(a, b) for a, b in zip(corners, existing[1])) < 2.0 for existing in ranked):
            return
        ranked.append((score, corners))

    for direction, group in enumerate(groups):
        for index, first in enumerate(group):
            check_head_acquisition_deadline(deadline_monotonic_sec, "endpoint_hypotheses")
            for second in group[index + 1:]:
                cross = 1 - direction
                first_mean = (first[0][cross] + first[1][cross]) / 2.0
                second_mean = (second[0][cross] + second[1][cross]) / 2.0
                upper, lower = (first, second) if first_mean < second_mean else (second, first)
                if max(abs(upper[i][direction] - lower[i][direction]) for i in (0, 1)) > 0.25 * expected_height:
                    continue
                points = (upper[0], upper[1], lower[1], lower[0])
                add(points)
                # A low-contrast rail can have a truncated locator segment
                # although its original Canny pixels continue to the corner.
                # One union-extent seed corrects that endpoint bias. These
                # extrapolated positions are never measurement evidence.
                start = min(upper[0][direction], lower[0][direction])
                end = max(upper[1][direction], lower[1][direction])
                def extended(segment, value):
                    point = list(segment[0])
                    fraction = (value-segment[0][direction]) / (segment[1][direction]-segment[0][direction])
                    point[cross] += fraction*(segment[1][cross]-segment[0][cross])
                    point[direction] = value
                    return tuple(point)
                add((extended(upper, start), extended(upper, end),
                     extended(lower, end), extended(lower, start)))
    return sorted(ranked, key=lambda item: item[0])


def _proposal(measurement, shape, *, expected_height, expected_center):
    corners = order_corners(measurement.corners)
    _width, height, center = _extent(corners)
    x0 = min(point.u_px for point in corners)
    y0 = min(point.v_px for point in corners)
    x1 = max(point.u_px for point in corners)
    y1 = max(point.v_px for point in corners)
    margin = max(4.0, 0.12 * height)
    bounds = (
        max(0, int(math.floor(x0 - margin))),
        max(0, int(math.floor(y0 - margin))),
        min(shape[1], int(math.ceil(x1 + margin)) + 1),
        min(shape[0], int(math.ceil(y1 + 0.45 * height)) + 1),
    )
    support = float(measurement.support.mean)
    return HeadProposal(
        corners=corners,
        bounds_xyxy=bounds,
        head_bounds_xyxy=(x0, y0, x1, y1),
        center_u_px=center[0], center_v_px=center[1],
        observed_height_px=height,
        expected_height_ratio=height / expected_height,
        center_offset_head_heights=math.dist(center, expected_center) / expected_height,
        raw_edge_support=support,
        geometry_quality=support,
    )


MIN_MEASURED_HEIGHT_RATIO = .60


MAX_MEASURED_HEIGHT_RATIO = 1.35


MIN_MEASURED_SIDE_BALANCE = .65


@dataclass(frozen=True)
class CandidateHeadSearch:
    center: tuple[float, float]
    height: float
    max_center_offset_ratio: float = 1.5
    edge_region: LidarHeadEdgeRegion | None = None
    pixel_size: ProjectedHeadSize | None = None
    center_tolerance_px: float | None = None
    center_bounds_px: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if (len(self.center) != 2
                or not all(math.isfinite(value) for value in (*self.center, self.height,
                                                             self.max_center_offset_ratio))
                or self.height <= 0. or not 0. < self.max_center_offset_ratio <= 1.5):
            raise ValueError("candidate head search requires a finite current projection")
        if self.center_tolerance_px is not None and (
                not math.isfinite(self.center_tolerance_px) or self.center_tolerance_px <= 0):
            raise ValueError("candidate centre tolerance must be finite and positive")
        if self.center_bounds_px is not None:
            x0, y0, x1, y1 = self.center_bounds_px
            if not all(math.isfinite(v) for v in self.center_bounds_px) or x0 >= x1 or y0 >= y1:
                raise ValueError("invalid metric centre bounds")

    def accepts_center(self, center, margin=0.):
        if self.center_bounds_px is None:
            return True
        x0, y0, x1, y1 = self.center_bounds_px
        return x0-margin <= center[0] <= x1+margin and y0-margin <= center[1] <= y1+margin

    @staticmethod
    def projected_center(corners):
        """The physical rectangle centre projects to its diagonal intersection.

        Averaging image corners is biased toward the nearer side under yaw;
        it can reject a correct close head at the position-uncertainty bound.
        """
        a, b, c, d = order_corners(corners)
        vx, vy = c.u_px-a.u_px, c.v_px-a.v_px
        wx, wy = d.u_px-b.u_px, d.v_px-b.v_px
        denominator = vx*wy-vy*wx
        if abs(denominator) < 1.e-9:
            return (math.nan, math.nan)
        fraction = ((b.u_px-a.u_px)*wy-(b.v_px-a.v_px)*wx)/denominator
        return (a.u_px+fraction*vx, a.v_px+fraction*vy)

    @property
    def center_limit_px(self):
        return (self.max_center_offset_ratio*self.height if self.center_tolerance_px is None
                else self.center_tolerance_px)

    def image_bounds(self, shape):
        if self.edge_region is not None:
            return self.edge_region.bounds
        extent = self.height*1.35 if self.pixel_size is None else max(
            self.pixel_size.max_height_px, self.pixel_size.max_width_px)
        radius = self.center_limit_px + extent + 10.
        u, v = self.center
        return (max(0, math.floor(u-radius)), max(0, math.floor(v-radius)),
                min(shape[1], math.ceil(u+radius)+1), min(shape[0], math.ceil(v+radius)+1))

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
        width, height, center = _extent(corners)
        if self.pixel_size is not None:
            return (self.pixel_size.accepts(width, height, refinement_allowance=True)
                    and self.accepts_center(self.projected_center(corners), 10.)
                    and math.dist(center, self.center) <= self.center_limit_px+10.)
        largest_growth = max((1., *OUTER_HEAD_SEARCH_GROWTH_FACTORS))
        # Coordinates reconstructed at an envelope endpoint can lose a few ulps.
        roundoff = 1.e-12 * max(height, self.height, 1.)
        return (.85 * height <= MAX_MEASURED_HEIGHT_RATIO * self.height + roundoff
                and 1.15 * largest_growth * height + roundoff >= MIN_MEASURED_HEIGHT_RATIO * self.height
                and math.dist(center, self.center)
                <= self.max_center_offset_ratio * self.height + 10. + roundoff)

    def accepts_measurement(self, corners):
        """Apply the existing candidate size/center bounds to measured pixels."""
        width, height, center = _extent(corners)
        tl, tr, br, bl = order_corners(corners)
        left, right = _distance(tl, bl), _distance(tr, br)
        return ((self.edge_region is None or self.edge_region.contains(corners))
                and self.accepts_center(self.projected_center(corners))
                # These bounds describe each upright physical side. Averaging
                # them can conceal one short rail ending in QR/background texture.
                and (all(self.pixel_size.accepts(width, side) for side in (left, right))
                     if self.pixel_size is not None
                     else MIN_MEASURED_HEIGHT_RATIO <= height / self.height <= MAX_MEASURED_HEIGHT_RATIO)
                and min(left, right) / max(left, right, 1.e-9) >= MIN_MEASURED_SIDE_BALANCE
                and math.dist(center, self.center) <= self.center_limit_px)


class HeadSearchInputs:
    """Reuse extraction only within one current-frame cold search and its retries."""

    NAMES = frozenset({"locator_edges", "contours", "gray", "lsd", "hough", "distance"})

    def __init__(self, frame, raw_edges, edge_region):
        self._context = (frame, raw_edges, edge_region)
        self._values = {}

    def check_context(self, frame, raw_edges, edge_region):
        if any(old is not new for old, new in zip(
                self._context, (frame, raw_edges, edge_region))):
            raise ValueError("head search inputs require the same frame, edges and region")

    def get(self, name, extract):
        if name not in self.NAMES:
            raise ValueError("unsupported head search extraction")
        if name not in self._values:
            self._values[name] = extract()
        return self._values[name]


def _inverse_point(point, transform):
    x, y, z, w = transform.rotation_xyzw
    return rotate_vector(tuple(a-b for a, b in zip(point, transform.translation_xyz_m)),
                         (-x, -y, -z, w))


def _rail_pixels(raw_edges, a, b):
    import numpy as np

    x0 = max(0, int(min(a.u_px, b.u_px)) - 2)
    x1 = min(raw_edges.shape[1], int(math.ceil(max(a.u_px, b.u_px))) + 3)
    y0 = max(0, int(min(a.v_px, b.v_px)) - 2)
    y1 = min(raw_edges.shape[0], int(math.ceil(max(a.v_px, b.v_px))) + 3)
    y, x = np.nonzero(raw_edges[y0:y1, x0:x1])
    points = np.column_stack((x + x0, y + y0))
    delta = np.array((b.u_px - a.u_px, b.v_px - a.v_px))
    length = float(np.linalg.norm(delta))
    tangent = delta / max(length, 1.)
    relative = points - (a.u_px, a.v_px)
    along = relative @ tangent
    offset = abs(relative @ np.array((-tangent[1], tangent[0])))
    keep = (along >= .12 * length) & (along <= .88 * length) & (offset <= 1.5)
    return frozenset((points[keep, 1] * raw_edges.shape[1] + points[keep, 0]).tolist())


def rail_signature(raw_edges, corners):
    return tuple(_rail_pixels(raw_edges, a, b)
                 for a, b in zip(corners, corners[1:] + corners[:1]))


def same_current_border_family(first, second, *, raw_edges=None, frame_bgr=None,
                               first_signature=None, second_signature=None):
    """Only measured pixel overlap or one evidenced thin ridge identifies a rail."""
    if max(_distance(a, b) for a, b in zip(first, second)) > 9.:
        return False
    if raw_edges is None:
        return max(_distance(a, b) for a, b in zip(first, second)) <= 1.5
    first_signature = first_signature or rail_signature(raw_edges, first)
    second_signature = second_signature or rail_signature(raw_edges, second)
    for index, (a, b) in enumerate(zip(first_signature, second_signature)):
        if max(_distance(first[index], second[index]),
               _distance(first[(index + 1) % 4], second[(index + 1) % 4])) <= 1.5 and a and b:
            continue
        if a and b and len(a & b) >= .70 * max(len(a), len(b)):
            continue
        if frame_bgr is None or not single_thin_stripe(
                frame_bgr, first[index], first[(index + 1) % 4],
                second[index], second[(index + 1) % 4]):
            return False
    return True


class CurrentBorderFamilies:
    """Per-image cache; nothing survives into another camera observation."""

    def __init__(self, raw_edges=None, frame_bgr=None):
        self.raw_edges = raw_edges
        self.frame_bgr = frame_bgr
        self._signatures = {}
        self._comparisons = {}
        self._rail_signatures = {}
        self._rail_comparisons = {}

    def signature(self, corners):
        key = tuple(corners)
        if key not in self._signatures:
            self._signatures[key] = tuple(self._rail_signature(a, b)
                for a, b in zip(key, key[1:]+key[:1]))
        return self._signatures[key]

    def _rail_signature(self, a, b):
        key = (a, b)
        if key not in self._rail_signatures:
            self._rail_signatures[key] = _rail_pixels(self.raw_edges, a, b)
        return self._rail_signatures[key]

    def _same_rail(self, a, b, c, d):
        key = (a, b, c, d)
        if key not in self._rail_comparisons:
            first, second = self._rail_signature(a, b), self._rail_signature(c, d)
            result = bool(first and second and (
                max(_distance(a, c), _distance(b, d)) <= 1.5
                or len(first & second) >= .70 * max(len(first), len(second))))
            if not result and self.frame_bgr is not None:
                result = single_thin_stripe(self.frame_bgr, a, b, c, d)
            self._rail_comparisons[key] = result
        return self._rail_comparisons[key]

    def same(self, first, second):
        key = (tuple(first), tuple(second))
        if key not in self._comparisons:
            if self.raw_edges is None:
                result = same_current_border_family(*key)
            elif max(_distance(a, b) for a, b in zip(*key)) > 9.:
                result = False
            else:
                result = all(self._same_rail(first[i], first[(i+1)%4],
                                             second[i], second[(i+1)%4]) for i in range(4))
            self._comparisons[key] = result
            self._comparisons[key[::-1]] = result
        return self._comparisons[key]


def _sample(frame, positions, *, allow_partial=False):
    low = np.floor(positions).astype(int)
    x, y = low[..., 0], low[..., 1]
    valid = np.all((x >= 0) & (y >= 0) & (x+1 < frame.shape[1])
                   & (y+1 < frame.shape[0]), axis=1)
    if not allow_partial and not valid.all():
        return None
    if allow_partial:
        x = np.clip(x, 0, frame.shape[1]-2)
        y = np.clip(y, 0, frame.shape[0]-2)
    fraction = positions-low
    fx, fy = fraction[..., 0, None], fraction[..., 1, None]
    values = ((1-fx)*(1-fy)*frame[y,x]+fx*(1-fy)*frame[y,x+1]
              +(1-fx)*fy*frame[y+1,x]+fx*fy*frame[y+1,x+1])
    return (values, valid) if allow_partial else values


def single_thin_stripe(frame, a, b, c, d):
    """Keep opposite-gradient/single-ridge proof, including multi-ridge veto."""
    first, last, other, end = (np.array((p.u_px, p.v_px), dtype=float) for p in (a,b,c,d))
    tangent = last-first
    tangent /= max(float(np.linalg.norm(tangent)), 1.)
    normal = np.array((-tangent[1],tangent[0]))
    if max(abs(float((other-first)@normal)),abs(float((end-last)@normal))) > 6.:
        return False
    fraction = np.array((.2,.35,.5,.65,.8))[:,None]
    p, q = first+fraction*(last-first), other+fraction*(end-other)
    values, valid = _sample(frame,np.stack((p-.75*normal,p+.75*normal,q-.75*normal,q+.75*normal),axis=1),allow_partial=True)
    g1,g2=values[:,1]-values[:,0],values[:,3]-values[:,2]
    magnitude=np.linalg.norm(g1,axis=1)*np.linalg.norm(g2,axis=1)
    opposite=valid & (magnitude>=400.) & (np.sum(g1*g2,axis=1)<=-.8*magnitude)
    radius=np.abs((q-p)@normal)/2+2.5
    positions=(p+q)[:,None,:]/2 + radius[:,None,None]*np.linspace(-1,1,25)[None,:,None]*normal
    values=_sample(frame,positions)
    if values is None:
        return False
    channels=np.argmax(np.ptp(values,axis=1),axis=1)
    profiles=values[np.arange(5),:,channels]
    baseline=(profiles[:,0]+profiles[:,-1])/2
    eligible=(np.abs(profiles[:,0]-profiles[:,-1])<=15.) & (np.ptp(profiles,axis=1)>=25.)
    positive=profiles.max(axis=1)-baseline>=baseline-profiles.min(axis=1)
    excursion=(profiles-baseline[:,None])*np.where(positive,1.,-1.)[:,None]
    eligible &= excursion.min(axis=1)>=-15.
    active=excursion>=np.maximum(12.,.25*excursion.max(axis=1))[:,None]
    starts=np.sum(active & ~np.roll(active,1,axis=1),axis=1)
    multiple=np.count_nonzero(eligible & (starts>1))
    supported=np.count_nonzero(eligible & (starts==1) & ~active[:,0] & ~active[:,-1])
    return bool(multiple<2 and (supported>=4 or np.count_nonzero(opposite)>=4))


@dataclass(frozen=True)
class CurrentFrameResolutionRecord:
    """One completed current-pixel refinement, local to one acquisition call."""

    hint_corners: tuple
    original_corners: tuple
    proposal: object
    recovery_evidence: object = None


def resolve_measured_current_frames(measured, resolutions, families):
    """Apply completed raw recoveries to their already measured original rails.

    Different location hints can measure the same initial frame, while only one
    hint's outward search finds its enclosing border. That completed search can
    resolve the other *measured* original. Its final proposal must already have
    passed the caller's candidate association; no new corners are constructed.

    This is a single substitution pass, not a transitive enclosure relation.
    Conflicting originals or final frames stay ambiguous. Untested hints still
    use the stricter neutral-hint policy in ``resolved_tested_head_hint``.
    """
    accepted_ids = {id(item) for item in measured}
    recoveries = []
    for entry in resolutions:
        evidence = entry.recovery_evidence
        if (id(entry.proposal) not in accepted_ids or not isinstance(evidence, HeadOuterBorderEvidence)
                or evidence.recovered is not True
                or evidence.original_corners != entry.original_corners
                or not validated_current_head_boundary(evidence,
                    corners=entry.proposal.corners)):
            continue
        original, recovered = entry.original_corners, entry.proposal.corners
        if (not 1.03 * _polygon_area(original) <= _polygon_area(recovered)
                <= 1.70 * _polygon_area(original) or not _encloses(recovered, original)):
            continue
        recoveries.append(entry)
    resolved = []
    for proposal in measured:
        matches = [entry for entry in recoveries
                   if families.same(proposal.corners, entry.original_corners)]
        compatible = bool(matches) and all(
            families.same(first.original_corners, second.original_corners)
            and families.same(first.proposal.corners, second.proposal.corners)
            for index, first in enumerate(matches) for second in matches[index + 1:])
        resolved.append(max(matches, key=lambda entry: _polygon_area(entry.proposal.corners)).proposal
                        if compatible else proposal)
    return resolved


def resolved_tested_head_hint(corners, resolutions, families):
    """Reuse a tested origin's canonical frame only on the same observed rails.

    A strict current refinement can resolve an inset locator to its physical
    frame. Cover only aliases of both its tested hint and its initial measured
    rails: the outer search still depends on the neutral hint. Matching only
    the initial measurement would permit a different outward search. No
    untested enclosure creates a mapping, and conflicting current rails or
    final physical frames remain unresolved.
    """
    matches = [entry for entry in resolutions if all(
        _polygon_area(corners) <= 1.03 * _polygon_area(origin)
        and families.same(corners, origin)
        for origin in (entry.hint_corners, entry.original_corners))
        and families.same(entry.hint_corners, entry.original_corners)]
    if not matches:
        return None
    for index, first in enumerate(matches):
        for second in matches[index + 1:]:
            if (not families.same(first.original_corners, second.original_corners)
                    or not families.same(first.hint_corners, second.hint_corners)
                    or not families.same(first.proposal.corners, second.proposal.corners)):
                return None
    return max(matches, key=lambda entry: _polygon_area(entry.proposal.corners)).proposal


def resolved_current_frame(corners, measured, families):
    """Cover a hint only by the same observed rails and the existing area guard.

    An untested outward rectangle cannot inherit verification from an inset
    frame. Agreement among matches must also be pairwise: one intermediate
    hint cannot merge two otherwise different current border families.
    """
    matches = [item for item in measured
               if _polygon_area(corners) <= 1.03 * _polygon_area(item.corners)
               and families.same(corners, item.corners)]
    if not matches or any(not families.same(first.corners, second.corners)
                          for index, first in enumerate(matches)
                          for second in matches[index + 1:]):
        return None
    return max(matches, key=lambda item: _polygon_area(item.corners))


def distinct_current_frames(measured, families):
    """Collapse only isolated, fully pairwise-agreeing measured-frame aliases.

    Current rail agreement need not be transitive. If A agrees with B and C,
    but B and C disagree, discarding B could hide that ambiguity from the next
    selection stage. Preserve every proposal in such a component; connectivity
    is used solely to find these conflicts, never to authorize a merge.
    """
    ordered = sorted(measured, key=lambda item: -_polygon_area(item.corners))
    neighbours = [{index} for index in range(len(ordered))]
    for index, first in enumerate(ordered):
        for other, second in enumerate(ordered[index + 1:], index + 1):
            if families.same(first.corners, second.corners):
                neighbours[index].add(other)
                neighbours[other].add(index)
    remaining = set(range(len(ordered)))
    selected = []
    while remaining:
        component, pending = set(), {min(remaining)}
        while pending:
            current = pending.pop()
            component.add(current)
            pending.update(neighbours[current] - component)
        remaining.difference_update(component)
        if all(component <= neighbours[index] for index in component):
            selected.append(ordered[min(component)])
        else:
            selected.extend(ordered[index] for index in sorted(component))
    return selected


def rank_current_head_hypotheses(cv2, raw_edges, hypotheses, *, corridor_half_width_px=4.,
                                 deadline_monotonic_sec=None, frame_bgr=None, border_families=None):
    """Cover spatial/border families before spending spare work on variants.

    The inexpensive locator asks for one current pixel bin on each corner arm;
    the subsequent unchanged strict refinement still requires two, on its fitted
    corners. Infinite rail intersections without any local corner pixels cannot
    consume the verification budget. Neither check synthesizes missing borders.
    """
    directly_supported = []
    for item in hypotheses:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_corner_support")
        evidence = metric_corner_arm_support(cv2, raw_edges, item[1])
        if all(count >= 1 for arms in evidence.bins_by_corner.values() for count in arms.values()):
            directly_supported.append(item)
    # Paired segment endpoints can be a few pixels short of a current corner.
    # Keep such alternate corridors when all four endpoints are within the
    # strict fitter's maximum shift of another current-corner-supported hint.
    # This is a locator rescue; only a subsequent raw fit supplies corners.
    maximum_shift = 2. * corridor_half_width_px + 2.
    supported = [item for item in hypotheses if any(
        max(_distance(a, b) for a, b in zip(item[1], other[1])) <= maximum_shift
        for other in directly_supported)]
    # Complete enclosing borders are tried before their inset alternatives.
    # Selection still compares verified independent families; area is not an
    # authority to choose one stand over another.
    supported.sort(key=lambda item: (-_polygon_area(item[1]), -item[0],
                                    tuple((p.u_px, p.v_px) for p in item[1])))
    groups = []
    families = border_families or CurrentBorderFamilies(raw_edges, frame_bgr)
    for item in supported:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_border_families")
        for group in groups:
            if all(families.same(item[1], member[1]) for member in group):
                group.append(item)
                break
        else:
            groups.append([item])
    ordered = [group[index] for index in range(max(map(len, groups), default=0))
               for group in groups if index < len(group)]
    return ordered


def uncovered_head_hypotheses(hypotheses, verified, *, cv2=None, raw_edges=None,
                              frame_bgr=None, all_hypotheses=(), border_families=None,
                              deadline_monotonic_sec=None):
    """A failed representative cannot erase its untested border variants."""
    families = border_families or CurrentBorderFamilies(raw_edges, frame_bgr)
    uncovered = [item for item in hypotheses if not any(
        families.same(item[1], proposal.corners)
        and _polygon_area(item[1]) <= 1.03 * _polygon_area(proposal.corners)
        for proposal in verified)]
    if not uncovered or cv2 is None:
        return uncovered
    # Texture classification can use the already current-border/corner-supported
    # locator graph. Those hints never provide accepted physical head corners.
    # Exact duplicates carry no extra texture topology. Keep different raw
    # borders, even when they belong to one family: projective texture tests
    # can differ between those measurements.
    unique_corners = dict.fromkeys(tuple(item[1]) for item in all_hypotheses)
    pool = [SimpleNamespace(corners=corners) for corners in unique_corners]
    covered = set()
    unique_verified = {tuple(item.corners): item for item in verified}
    for outer in unique_verified.values():
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_uncovered_families")
        others = [item for item in pool if not families.same(outer.corners, item.corners)]
        covered.update(id(item) for item in texture_members(cv2, outer, others,
            deadline_monotonic_sec=deadline_monotonic_sec))
    covered_corners = {tuple(item.corners) for item in pool if id(item) in covered}
    return [item for item in uncovered if not any(
        families.same(item[1], corners) for corners in covered_corners)]


def texture_members(cv2, outer, others, *, deadline_monotonic_sec=None):
    """Find repeated inset boxes and rectangles assembled from their same rails.

    Three disjoint, aligned small boxes establish a distributed texture family.
    Additional overlapping/tall rectangles are not automatically competing heads
    when *each* of their four rails belongs to this same inset rail graph. A lone
    inset, a new unexplained rail, or another enclosing border stays ambiguous.
    This is a generic image-topology test with no QR size or identity input.
    """
    import numpy as np

    transform = cv2.getPerspectiveTransform(
        np.asarray([(p.u_px, p.v_px) for p in outer.corners], np.float32),
        np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), np.float32))
    normalized = []
    for proposal in others:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_texture_topology")
        points = cv2.perspectiveTransform(np.asarray(
            [(p.u_px, p.v_px) for p in proposal.corners], np.float32).reshape(1, 4, 2), transform)[0]
        if not np.isfinite(points).all() or points.min() < .02 or points.max() > .98:
            continue
        low, high = points.min(axis=0), points.max(axis=0)
        size, center = high - low, points.mean(axis=0)
        if size.min() <= 0.:
            continue
        edges = np.roll(points, -1, axis=0) - points
        if any(abs(edge[1 - axis]) > .20 * abs(edge[axis])
               for edge, axis in zip(edges, (0, 1, 0, 1))):
            continue
        normalized.append((proposal, low, high, center, size))
    anchors = []
    for item in sorted(normalized, key=lambda item: (-float(np.prod(item[4])),
                                                    tuple(item[1]))):
        if item[4].max() > .42 or np.linalg.norm(item[3] - .5) < .20:
            continue
        # Contour/raster variants of a small texture rail do not create extra
        # independent boxes. Physical-head alternatives are never merged here.
        if any(np.max(np.abs(item[1] - old[1])) < .035
               and np.max(np.abs(item[2] - old[2])) < .035 for old in anchors):
            continue
        anchors.append(item)
    if len(anchors) < 3 or len(anchors) > 32:
        return []
    best = []
    for start in anchors:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_texture_rail_graph")
        group = [start]
        for item in anchors:
            if item is start:
                continue
            if any(np.any(np.maximum(item[4], old[4]) > 1.5 * np.minimum(item[4], old[4]))
                   or np.all(np.minimum(item[2], old[2]) > np.maximum(item[1], old[1]))
                   for old in group):
                continue
            group.append(item)
        centers = np.asarray([item[3] for item in group])
        if (len(group) < 3 or np.any(np.ptp(centers, axis=0) < .35)
                or len({tuple(center >= .5) for center in centers}) < 3):
            continue
        def explained(item):
            for axis in range(2):
                along = 1 - axis
                for value in (item[1][axis], item[2][axis]):
                    intervals = sorted((max(item[1][along], anchor[1][along]),
                                        min(item[2][along], anchor[2][along]))
                        for anchor in group if min(abs(value - anchor[1][axis]),
                                                  abs(value - anchor[2][axis])) <= .035)
                    coverage, previous = 0., -float("inf")
                    for low, high in intervals:
                        coverage += max(0., high - max(low, previous))
                        previous = max(previous, high)
                    # Collinearity alone is insufficient: the observed texture
                    # segments must explain most of every composite rail. A
                    # separate central box between them remains independent.
                    if coverage < .5 * float(item[2][along] - item[1][along]):
                        return False
            return True
        def concentric_inset(item):
            # Repeated printed motifs can contain concentric rails. Requiring
            # collinearity with the outer motif mistakes its inner ring for a
            # separate head. Only the already established three distributed,
            # disjoint anchors can own such an inset; ordinary containment in
            # the head (including a lone central rectangle) is insufficient.
            for anchor in group:
                scale = item[4] / anchor[4]
                center_error = np.abs(item[3] - anchor[3]) / anchor[4]
                if (np.all(item[1] > anchor[1]) and np.all(item[2] < anchor[2])
                        and np.all((.35 <= scale) & (scale <= .90))
                        and abs(float(scale[0] - scale[1])) <= .10
                        and np.all(center_error <= .08)):
                    return True
            return False
        members = [item[0] for item in normalized if explained(item) or concentric_inset(item)]
        if len(members) > len(best):
            best = members
    return best


def connected_border_family(selected, proposals, families):
    """Collect mutually evidenced aliases without transitive rail bridges."""
    members = [selected]
    remaining = [item for item in proposals if item is not selected]
    while remaining:
        new = []
        for item in remaining:
            if all(families.same(item.corners, member.corners) for member in members + new):
                new.append(item)
        if not new:
            break
        members.extend(new)
        remaining = [item for item in remaining if all(item is not added for added in new)]
    return members


def select_verified_head(cv2, verified, *, raw_edges=None, frame_bgr=None,
                         texture_hypotheses=(), border_families=None, deadline_monotonic_sec=None):
    """Return a complete outer frame or retain genuine current-head ambiguity."""
    if not verified:
        return None, "head_proposal_unavailable"
    selected = max(verified, key=lambda proposal: _polygon_area(proposal.corners))
    families = border_families or CurrentBorderFamilies(raw_edges, frame_bgr)
    same_frame = connected_border_family(selected, verified, families)
    others = [proposal for proposal in verified if all(proposal is not member for member in same_frame)]
    pool = others + [SimpleNamespace(corners=item[1]) for item in texture_hypotheses
                    if not families.same(selected.corners, item[1])]
    texture = texture_members(cv2, selected, pool,
        deadline_monotonic_sec=deadline_monotonic_sec) if others else []
    if any(all(proposal is not member for member in texture) for proposal in others):
        return None, "head_proposal_ambiguous"
    return selected, "current_head_proposal"


MAX_IMAGE_PIXELS = 1920 * 1080


MAX_CONTOURS = 1024


MAX_RAILS_PER_DIRECTION = 24


MAX_LOCATOR_HYPOTHESES = 256


MAX_RAW_VERIFICATIONS = 12


MAX_HEAD_IMAGE_AREA_FRACTION = .80


MIN_LOCATOR_BORDER_SUPPORT = .80


def _distinct_rails(group, direction, *, deadline_monotonic_sec=None):
    """Keep the same longest-first representatives using bounded array work."""
    import numpy as np

    ordered = sorted(group, key=lambda item: (-item[0], item[1:]))
    kept = np.empty((len(ordered), 5), dtype=np.float64)
    distinct = []
    for item in ordered:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_rail_merge")
        length, a, b = item
        old = kept[:len(distinct)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        # Compare only retained representatives. A chain of overlapping
        # fragments must not merge two rails that fail the original predicate.
        parallel = (np.abs(dx * (old[:, 4] - old[:, 2])
                           - dy * (old[:, 3] - old[:, 1])) / (length * old[:, 0])) < .02
        nearby = np.maximum(np.abs(dx * (old[:, 2] - a[1]) - dy * (old[:, 1] - a[0])),
                            np.abs(dx * (old[:, 4] - a[1]) - dy * (old[:, 3] - a[0]))) / length < 1.5
        overlap = np.minimum(b[direction], old[:, 3 + direction]) - np.maximum(a[direction], old[:, 1 + direction])
        if np.any(parallel & nearby & (overlap > .5 * np.minimum(length, old[:, 0]))):
            continue
        kept[len(distinct)] = (length, *a, *b)
        distinct.append(item)
    return distinct


def _bounded_quad(points, shape):
    corners = order_corners(tuple(ImagePoint(float(x), float(y)) for x, y in points))
    if not _well_formed_quadrilateral(corners):
        return None
    rows, cols = shape[:2]
    if not all(3 <= p.u_px < cols - 3 and 3 <= p.v_px < rows - 3 for p in corners):
        return None
    width, height, _center = _extent(corners)
    if (min(_distance(a, b) for a, b in zip(corners, corners[1:] + corners[:1]))
            < MIN_HEAD_EDGE_PX or not 0. < width / height <= 1.35):
        return None
    # Rectification introduces a strong closed border around the whole image.
    # A cold locator needs visible context, so that canvas boundary is excluded.
    if _polygon_area(corners) > MAX_HEAD_IMAGE_AREA_FRACTION * rows * cols:
        return None
    return corners


def _rail_endpoint_hints(cv2, gray, raw_edges, *, deadline_monotonic_sec=None,
                         preferred_head_height_px=None, source_support=None,
                         edge_region=None, metric_search=None, frame_inputs=None):
    """Pair current line segments, using candidate scale only to order hints."""
    import numpy as np

    preferred_height = preferred_head_height_px
    # Preserve LSD's original sampling and endpoints: cropping gray changes
    # even interior subpixel lines. Screen its results before the rail quota.
    extract = lambda name, operation: operation() if frame_inputs is None else frame_inputs.get(name, operation)
    detected = extract("lsd", lambda: cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0])
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_hough")
    # Equal-luminance colour boundaries can be absent from grayscale LSD while
    # the caller's channel-union Canny has complete current evidence.
    raw_lines = extract("hough", lambda: cv2.HoughLinesP(raw_edges, 1., np.pi / 180., threshold=8,
                               minLineLength=16, maxLineGap=2))
    groups = ([], [])
    segments = [segment for lines in (detected, raw_lines) if lines is not None
                for segment in lines.reshape(-1, 4)]
    for x0, y0, x1, y1 in segments:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_rails")
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = math.hypot(dx, dy)
        if length < MIN_HEAD_EDGE_PX * .70:
            continue
        direction = 0 if abs(dy) <= .50 * abs(dx) else 1 if abs(dx) <= .50 * abs(dy) else None
        if direction is None:
            continue
        first, last = (float(x0), float(y0)), (float(x1), float(y1))
        if (getattr(metric_search, "center_bounds_px", None) is None
                and edge_region is not None and not edge_region.contains(
                (ImagePoint(*first), ImagePoint(*last)))):
            continue
        if source_support is not None and not source_support.segment(first, last):
            continue
        if metric_search is not None:
            bx0, by0, bx1, by1 = metric_search.image_bounds(gray.shape)
            midpoint = ((first[0]+last[0])/2., (first[1]+last[1])/2.)
            size = metric_search.pixel_size
            upper = size.max_width_px if direction == 0 else size.max_height_px
            if (length > 1.8*upper or (metric_search.center_bounds_px is None
                    and (not bx0 <= midpoint[0] < bx1 or not by0 <= midpoint[1] < by1))):
                continue
        if first[direction] > last[direction]:
            first, last = last, first
        groups[direction].append((length, first, last))
    distinct_groups = tuple(_distinct_rails(group, direction,
        deadline_monotonic_sec=deadline_monotonic_sec) for direction, group in enumerate(groups))
    hints = []
    for direction, group in enumerate(groups):
        # LSD and Hough often locate the same rail with different endpoints.
        # Collapse those search duplicates before allocating the fixed budget;
        # the retained longest fragment still supplies only a locator line.
        distinct = distinct_groups[direction]
        if preferred_height is not None:
            # The projected physical size already bounds this candidate. Give
            # its complete rails a slot before small printed-texture fragments;
            # position/scale remain hints, not measured corners or an angle.
            groups[direction][:] = sorted(distinct, key=lambda item: (
                abs(math.log(item[0] / preferred_height)), item[1:]))[:MAX_RAILS_PER_DIRECTION]
        else:
            # Without a projected candidate, spread the fixed budget across
            # line scales and image thirds so room boundaries cannot monopolize it.
            strata = {}
            for item in distinct:
                length, a, b = item
                key = (max(0, int(math.log2(length / MIN_HEAD_EDGE_PX))),
                       min(2, int(3. * (a[0] + b[0]) / (2. * gray.shape[1]))))
                strata.setdefault(key, []).append(item)
            groups[direction][:] = [bucket[index] for index in range(max(map(len, strata.values()), default=0))
                                    for _key, bucket in sorted(strata.items()) if index < len(bucket)][:MAX_RAILS_PER_DIRECTION]
        if getattr(metric_search, "center_bounds_px", None) is not None and edge_region is not None:
            # Do not refill slots vacated by the volume gate with ever more
            # inset/printed rails. That makes a tighter mask increase the
            # number of competing rectangles despite fewer input edge pixels.
            retained = [item for item in groups[direction] if edge_region.contains(
                (ImagePoint(*item[1]), ImagePoint(*item[2])))]
            groups[direction][:] = retained
        group = groups[direction]
        cross = 1 - direction
        for index, (length, first, last) in enumerate(group):
            check_head_acquisition_deadline(deadline_monotonic_sec, "cold_endpoint_hypotheses")
            for other_length, other_first, other_last in group[index + 1:]:
                if min(length, other_length) < .20 * max(length, other_length):
                    continue
                separation = abs((first[cross] + last[cross] - other_first[cross] - other_last[cross]) / 2.)
                if not MIN_HEAD_EDGE_PX * .70 <= separation <= max(gray.shape):
                    continue
                # A neck touching the lower rail splits its locator into two
                # short segments. Use their union extent only as a search hint;
                # unchanged raw rail/corner checks must establish every pixel.
                start = min(first[direction], other_first[direction])
                end = max(last[direction], other_last[direction])
                overlap = min(last[direction], other_last[direction]) - max(first[direction], other_first[direction])
                if overlap < .15 * min(length, other_length) or end - start > 1.4 * max(length, other_length):
                    continue
                hints.append((first, last, other_last, other_first))
                def extended(a, b, value):
                    fraction = (value - a[direction]) / (b[direction] - a[direction])
                    point = [0., 0.]
                    point[direction] = value
                    point[cross] = a[cross] + fraction * (b[cross] - a[cross])
                    return tuple(point)
                hints.append((extended(first, last, start), extended(first, last, end),
                              extended(other_first, other_last, end), extended(other_first, other_last, start)))
    return tuple(hints)


@bounded_head_acquisition
def acquire_cold_head_proposal(
    cv2, frame_bgr, *, raw_edges=None, edge_preprocess="channel_union",
    canny_low=20, canny_high=60,
    deadline_monotonic_sec=None,
    proposal_filter=None,
    model_profile, refinement_out=None, candidate_search=None,
    color_support_mask=None,
    _verification_limit=None,
) -> HeadProposalResult:
    """Locate a unique complete head in a bounded image without a prior pose.

    Contours and line hints locate, but never certify, boundaries. A clipped or
    open rectangle cannot become a measurement through locator interpolation.
    Current corner evidence and border families allocate the fixed work budget.
    Unverified independent families remain unresolved on budget exhaustion.
    """
    import numpy as np

    if refinement_out is not None:
        refinement_out.clear()

    verification_limit = MAX_RAW_VERIFICATIONS if _verification_limit is None else _verification_limit
    if type(verification_limit) is not int or not 0 <= verification_limit <= MAX_RAW_VERIFICATIONS:
        raise ValueError("head verification limit must remain within the original work budget")
    raw_verifications = 0
    candidate_association_rejections = 0

    def result(reason, proposal=None):
        return HeadProposalResult(proposal, reason, raw_verifications)

    if (frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
            or min(frame_bgr.shape[:2]) < MIN_HEAD_EDGE_PX + 6):
        return result("head_proposal_input_invalid")
    if frame_bgr.shape[0] * frame_bgr.shape[1] > MAX_IMAGE_PIXELS:
        return result("head_cold_acquisition_image_budget_exceeded")
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_preprocessing")
    if raw_edges is None:
        raw_edges = _canny_edges_from_frame(
            cv2, frame_bgr, edge_preprocess=edge_preprocess, blur_kernel=5,
            canny_low=canny_low, canny_high=canny_high,
        )
    if raw_edges.ndim != 2 or raw_edges.shape != frame_bgr.shape[:2]:
        raise ValueError("raw_edges must match the cold search image")
    if color_support_mask is not None and (
            color_support_mask.ndim != 2 or color_support_mask.shape != raw_edges.shape
            or str(color_support_mask.dtype) != "uint8"):
        raise ValueError("colour support must be a uint8 mask of the exact processing image")
    edge_region = getattr(candidate_search, "edge_region", None)
    frame_inputs = HeadSearchInputs(frame_bgr, raw_edges, edge_region)
    frame_inputs.check_context(frame_bgr, raw_edges, edge_region)
    locator_edges = frame_inputs.get("locator_edges",
        lambda: raw_edges if edge_region is None else edge_region.locator_edges(raw_edges))
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_contours")
    contours = frame_inputs.get("contours",
        lambda: cv2.findContours(locator_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2])
    metric_search = candidate_search if getattr(candidate_search, "pixel_size", None) is not None else None
    if metric_search is not None:
        bx0, by0, bx1, by1 = metric_search.image_bounds(raw_edges.shape)
        def metric_relevant(contour):
            x, y, w, h = cv2.boundingRect(contour)
            size = metric_search.pixel_size
            # A contour is only a locator. Leave refinement margin and keep
            # short texture loops available in the untouched raw evidence.
            return (x >= bx0 and y >= by0 and x+w <= bx1 and y+h <= by1
                    and h >= .85*size.min_height_px/(1.15*1.25)
                    and h <= 1.5*size.max_height_px
                    and w <= 1.5*size.max_width_px)
        contours = [contour for contour in contours if metric_relevant(contour)]
    eligible_contours = []
    for contour in contours:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_contour_hypotheses")
        if edge_region is not None:
            x0, y0, x1, y1 = edge_region.bounds
            x, y, w, h = cv2.boundingRect(contour)
            # Mask clipping is not a closed observed contour.
            if x <= x0 or y <= y0 or x+w >= x1 or y+h >= y1:
                continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 4 * MIN_HEAD_EDGE_PX:
            continue
        eligible_contours.append((contour, perimeter))
        if len(eligible_contours) > MAX_CONTOURS:
            return result("head_cold_acquisition_contour_budget_exceeded")
    # A contour shorter than the minimum four sides could never supply a
    # valid quad. Tiny blind/radiator/printed-texture loops do not spend the
    # bounded quadrilateral search budget, but remain in untouched raw edges.
    seeds = []
    for contour, perimeter in eligible_contours:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_contour_hypotheses")
        for fraction in (.015, .025, .04):
            quad = cv2.approxPolyDP(contour, fraction * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                seeds.append((quad.reshape(-1, 2), "closed_contour"))
    gray = frame_inputs.get("gray", lambda: cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_line_detection")
    hints = _rail_endpoint_hints(cv2, gray, raw_edges,
                                            deadline_monotonic_sec=deadline_monotonic_sec,
                                            preferred_head_height_px=(metric_search.height if metric_search is not None
                                                                      else None),
                                            metric_search=metric_search,
                                            source_support=getattr(proposal_filter, "source_support", None),
                                            edge_region=edge_region,
                                            frame_inputs=frame_inputs)
    seeds.extend((hint, "paired_current_rails") for hint in hints)
    distance = frame_inputs.get("distance", lambda: cv2.distanceTransform(
        np.where(raw_edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3))
    fractions = np.linspace(.10, .90, 24)
    hypotheses = {}
    texture_hints = {}
    association_previews = {}
    seen_exact_seeds = set()
    for points, locator in seeds:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_hypothesis_support")
        corners = _bounded_quad(points, raw_edges.shape)
        if corners is None:
            continue
        if corners in seen_exact_seeds:
            continue
        seen_exact_seeds.add(corners)
        texture_only = False
        points = np.asarray([(p.u_px, p.v_px) for p in corners])
        pixels = np.rint(points[:, None, :] + fractions[None, :, None]
                         * (np.roll(points, -1, axis=0) - points)[:, None, :]).astype(np.int32)
        support = np.mean(distance[pixels[:, :, 1], pixels[:, :, 0]] <= 4., axis=1)
        # Long paired hints can bridge separate texture boxes. Each of the
        # four borders must already be substantially present, rather than
        # relying on the downstream mean support to hide a missing interval.
        if min(support) < MIN_LOCATOR_BORDER_SUPPORT:
            continue
        key = tuple(round(float(value), 2) for value in points.ravel())
        score = .70 * float(min(support)) + .30 * float(np.mean(support))
        if candidate_search is not None and not candidate_search.accepts_hint(corners):
            # Keep current supported texture for the unchanged topology graph.
            # Screening a head location cannot erase its smaller inset anchors.
            texture_only = True
        if proposal_filter is not None and not texture_only:
            if key not in association_previews:
                if len(association_previews) >= MAX_LOCATOR_HYPOTHESES:
                    return result("head_cold_acquisition_locator_budget_exceeded")
                _width, height, center = _extent(corners)
                # The callback sees a current supported 2D locator, with no
                # pose/angle authority. It is rechecked on measured corners
                # after strict refinement; projection never certifies pixels.
                preview = _proposal(SimpleNamespace(corners=corners,
                    support=SimpleNamespace(mean=float(np.mean(support)))), raw_edges.shape,
                    expected_height=height,
                    expected_center=center)
                association_previews[key] = bool(getattr(proposal_filter, "preview", proposal_filter)(preview))
            if not association_previews[key]:
                candidate_association_rejections += 1
                continue
        destination = texture_hints if texture_only else hypotheses
        if key not in destination or score > destination[key][0]:
            destination[key] = (score, corners, locator)
    if len(texture_hints) > MAX_LOCATOR_HYPOTHESES:
        return result("head_cold_acquisition_locator_budget_exceeded")
    if len(hypotheses) > MAX_LOCATOR_HYPOTHESES:
        return result("head_cold_acquisition_locator_budget_exceeded")
    # Group duplicate rail contours before spending the strict verification
    # budget. Retain distinct border locations: choosing inward versus outward
    # rails remains a measurement decision, never a proximity-to-centre prior.
    ranked = sorted(hypotheses.values(), key=lambda item: (-item[0], -_polygon_area(item[1]),
                                                         tuple((p.u_px, p.v_px) for p in item[1])))
    unique = []
    border_families = CurrentBorderFamilies(raw_edges, frame_bgr)
    for item in ranked:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_hypothesis_deduplication")
        if not any(max(_distance(a, b) for a, b in zip(item[1], other[1])) < 3.
                   and border_families.same(item[1], other[1])
                   for other in unique):
            unique.append(item)
    ordered = rank_current_head_hypotheses(cv2, raw_edges, unique,
        frame_bgr=frame_bgr, border_families=border_families,
        deadline_monotonic_sec=deadline_monotonic_sec)
    texture_pool = ordered + list(texture_hints.values())
    accepted = []
    verified_proposals = []
    untested = []
    tested_hints = set()
    tested_frame_resolutions = []
    measured_frames = {}
    def verify(items):
        nonlocal raw_verifications, candidate_association_rejections
        for score, corners, locator in items:
            check_head_acquisition_deadline(deadline_monotonic_sec, "cold_strict_verification",
                raw_verifications=raw_verifications)
            hint_key = tuple(corners)
            if hint_key in tested_hints:
                continue
            if resolved_current_frame(
                    corners, verified_proposals, border_families) is not None:
                continue
            if resolved_tested_head_hint(
                    corners, tested_frame_resolutions, border_families) is not None:
                continue
            if raw_verifications >= verification_limit:
                untested.append((score, corners, locator))
                continue
            tested_hints.add(hint_key)
            outer = None
            measured, outer, _seed = refine_current_physical_head(
                cv2, raw_edges, model_profile=model_profile, proposal_corners=corners,
                prefer_outer_metric_rail=getattr(metric_search, "center_bounds_px", None) is not None,
                color_support_mask=color_support_mask,
                deadline_monotonic_sec=deadline_monotonic_sec)
            raw_verifications += 1
            if measured.accepted and (outer is None or outer.accepted):
                _width, height, center = _extent(measured.corners)
                proposal = _proposal(measured, raw_edges.shape,
                    expected_height=height,
                    expected_center=center)
                verified_proposals.append(proposal)
                if outer is not None:
                    tested_frame_resolutions.append(CurrentFrameResolutionRecord(
                        tuple(corners), tuple(outer.original_corners), proposal, outer))
                if candidate_search is not None and not candidate_search.accepts_measurement(measured.corners):
                    continue
                if proposal_filter is not None and not proposal_filter(proposal):
                    candidate_association_rejections += 1
                    continue
                accepted.append(proposal)
                if outer is not None:
                    measured_frames[proposal.corners] = (measured, outer, _seed)
    verify(ordered)
    untested = [item for item in untested if resolved_current_frame(
        item[1], verified_proposals, border_families) is None
        and resolved_tested_head_hint(item[1], tested_frame_resolutions, border_families) is None]
    resolved = resolve_measured_current_frames(
        accepted, tested_frame_resolutions, border_families)
    accepted = distinct_current_frames(resolved, border_families)
    uncovered = uncovered_head_hypotheses(untested, verified_proposals,
        cv2=cv2, raw_edges=raw_edges, frame_bgr=frame_bgr, all_hypotheses=texture_pool,
        border_families=border_families, deadline_monotonic_sec=deadline_monotonic_sec)
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_head_selection",
        raw_verifications=raw_verifications)
    if uncovered:
        return result("head_cold_acquisition_verification_budget_exceeded")
    selected, reason = select_verified_head(cv2, accepted,
        raw_edges=raw_edges, frame_bgr=frame_bgr, texture_hypotheses=texture_pool,
        border_families=border_families, deadline_monotonic_sec=deadline_monotonic_sec)
    if selected is None and not accepted and candidate_association_rejections:
        reason = "head_proposal_candidate_association_rejected"
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_complete_head_selection",
        raw_verifications=raw_verifications)
    if selected is not None and refinement_out is not None:
        measured, outer, seed = measured_frames[selected.corners]
        refinement_out["selected"] = capture_current_head_refinement(
            frame_bgr, raw_edges, model_profile=model_profile,
            refinement=measured, outer_recovery=outer, seed=seed)
    return result(reason, selected)


MAX_RIM_COMPONENTS = 8


@dataclass(frozen=True)
class MaterialRim:
    label: str
    support: object
    holes: tuple


def _box(x, y, w, h):
    return tuple(ImagePoint(float(u), float(v)) for u, v in
                 ((x,y),(x+w,y),(x+w,y+h),(x,y+h)))


def material_rims(cv2, frame, search, *, deadline=None):
    """Require an enclosed, head-scale hole in one chromatic material.

    Hole coordinates are only locators. No contour/morphology coordinate is
    returned as a physical measurement. Achromatic or broken rims use the
    ordinary raw-edge acquisition path.
    """
    import numpy as np
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    palette = {}
    for item in STAND_EDGE_PALETTE:
        if item.label != 'light-grey':
            palette.setdefault(item.label, []).append(item)
    result = []
    for label, ranges in palette.items():
        check_head_acquisition_deadline(deadline, 'material_rim_components')
        mask = np.zeros(frame.shape[:2], np.uint8)
        for item in ranges:
            # ColorRange's field names are shared with build_mask_for_ranges.
            mask |= build_mask_for_ranges(cv2, np, hsv, (item,))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if len(contours) > 1024:
            raise ValueError('material rim contour budget exceeded')
        if hierarchy is None:
            continue
        parents = {}
        for index, contour in enumerate(contours):
            check_head_acquisition_deadline(deadline, 'material_rim_holes')
            parent = int(hierarchy[0,index,3])
            if parent < 0:
                continue
            x,y,w,h = cv2.boundingRect(contour)
            if min(w,h) < 24 or not search.accepts_hint(_box(x,y,w,h)):
                continue
            if search.edge_region is not None and not search.edge_region.contains(_box(x,y,w,h)):
                continue
            parents.setdefault(parent, []).append((x,y,w,h))
        if not parents:
            continue
        if sum(len(r.holes) for r in result)+sum(map(len,parents.values())) > MAX_RIM_COMPONENTS:
            raise ValueError('material rim component budget exceeded')
        exterior = np.zeros_like(mask)
        selected = np.zeros_like(mask)
        for parent in parents:
            cv2.drawContours(exterior, contours, parent, 255, 1)
            cv2.drawContours(selected, contours, parent, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        support = cv2.bitwise_and(cv2.dilate(exterior,kernel),cv2.dilate(mask & selected,kernel))
        result.append(MaterialRim(label, support, tuple(hole for holes in parents.values() for hole in holes)))
    return tuple(result)


def unexplained_raw_rectangle(cv2, raw, selected, search, *, frame, model_profile,
                              proposal_filter=None, deadline=None):
    """Retain complete raw contours outside the selected rim, including gray.

    This is a conservative competitor veto, not a geometry producer. Printed
    contours inside this complete material-owned head cannot donate corners to
    the selected measurement. Independent material components are compared by
    the caller before this check.
    """
    pass
    source = raw if search.edge_region is None else search.edge_region.locator_edges(raw)
    contours = cv2.findContours(source,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) > 1024:
        return True
    for contour in contours:
        check_head_acquisition_deadline(deadline, 'material_rim_raw_competitors')
        length = cv2.arcLength(contour,True)
        if length < 96:
            continue
        for fraction in (.015,.025,.04):
            quad = cv2.approxPolyDP(contour,fraction*length,True)
            if len(quad)!=4 or not cv2.isContourConvex(quad):
                continue
            corners = _bounded_quad(quad.reshape(-1,2),raw.shape)
            if corners is None or not search.accepts_hint(corners):
                continue
            source_support = getattr(proposal_filter, 'source_support', None)
            if source_support is not None and not source_support.accepts(corners):
                continue
            if _encloses(selected,corners,tolerance=10.):
                continue
            if _quadrilateral_edge_support(cv2,raw,corners).accepted:
                return True
    # Open outer contours (a stem may join the rim) still have paired raw
    # rails. Run the ordinary bounded locator with zero verification slots:
    # any unowned hypothesis vetoes the shortcut, including achromatic heads.
    pass
    class Unowned:
        source_support = getattr(proposal_filter, 'source_support', None)

        def __call__(self, proposal):
            return (not _encloses(selected, proposal.corners, tolerance=10.)
                    and (proposal_filter is None or
                         getattr(proposal_filter,'preview',proposal_filter)(proposal)))
        preview = __call__
    # The locator expects source_support to be absent rather than None.
    gate = Unowned()
    if gate.source_support is None:
        del Unowned.source_support
    competitors = acquire_cold_head_proposal(cv2,frame,raw_edges=raw,
        candidate_search=search,model_profile=model_profile,proposal_filter=gate,
        _verification_limit=0,deadline_monotonic_sec=deadline)
    return ('budget' in competitors.reason or 'deadline' in competitors.reason
            or 'ambiguous' in competitors.reason or competitors.proposal is not None)


@bounded_head_acquisition
def acquire_material_rim(cv2, frame, *, raw_edges, candidate_search, model_profile,
                         proposal_filter=None, refinement_out=None, budget_out=None, deadline_monotonic_sec=None):
    if candidate_search is None or candidate_search.pixel_size is None:
        return None
    deadline = deadline_monotonic_sec
    used = 0
    if budget_out is not None:
        budget_out["used"] = 0
    def result(reason, proposal=None):
        return HeadProposalResult(proposal, reason, used)
    if frame.shape[0]*frame.shape[1] > MAX_IMAGE_PIXELS:
        return result("head_material_rim_budget_exceeded")
    try:
        rims = material_rims(cv2,frame,candidate_search,deadline=deadline)
    except ValueError:
        return result('head_material_rim_budget_exceeded')
    if not rims:
        return None
    verified = []
    families = CurrentBorderFamilies(raw_edges,frame)
    for rim in rims:
        check_head_acquisition_deadline(deadline,'material_rim_acquisition')
        remaining = MAX_RAW_VERIFICATIONS-used-1  # Reserve one full-raw validation.
        if remaining < 1:
            return result('head_material_rim_budget_exceeded')
        located = acquire_cold_head_proposal(cv2,frame,
            raw_edges=cv2.bitwise_and(raw_edges,rim.support), candidate_search=candidate_search,
            model_profile=model_profile,proposal_filter=proposal_filter,
            color_support_mask=rim.support,_verification_limit=remaining,
            deadline_monotonic_sec=deadline)
        used += located.raw_verifications
        if budget_out is not None:
            budget_out["used"] = used
        if located.proposal is None:
            if 'ambiguous' in located.reason or 'budget' in located.reason or 'deadline' in located.reason:
                return result(located.reason)
            continue
        # Never reuse the masked acquisition's transient refinement proof.
        check_head_acquisition_deadline(deadline,'material_rim_full_raw_validation')
        measured,outer,seed = refine_current_physical_head(cv2,raw_edges,
            model_profile=model_profile,proposal_corners=located.proposal.corners,
            deadline_monotonic_sec=deadline)
        used += 1
        if budget_out is not None:
            budget_out["used"] = used
        if (not measured.accepted or not outer.accepted or measured.corners is None
                or not families.same(located.proposal.corners,measured.corners)
                or not candidate_search.accepts_measurement(measured.corners)):
            return result('head_material_rim_full_raw_disagrees')
        _,height,center = _extent(measured.corners)
        proposal = _proposal(measured,raw_edges.shape,expected_height=height,expected_center=center)
        if proposal_filter is not None and not proposal_filter(proposal):
            return result('head_proposal_candidate_association_rejected')
        verified.append((proposal,measured,outer,seed))
    if not verified:
        return None
    # Pairwise evidence, not transitive grouping or color priority.
    if any(not families.same(a[0].corners,b[0].corners)
           for i,a in enumerate(verified) for b in verified[i+1:]):
        return result('head_proposal_ambiguous')
    proposal,measured,outer,seed = verified[0]
    if unexplained_raw_rectangle(cv2,raw_edges,proposal.corners,candidate_search,
            frame=frame,model_profile=model_profile,proposal_filter=proposal_filter,deadline=deadline):
        return result('head_proposal_ambiguous')
    check_head_acquisition_deadline(deadline,'material_rim_complete')
    if refinement_out is not None:
        refinement_out['selected'] = capture_current_head_refinement(frame,raw_edges,
            model_profile=model_profile,refinement=measured,outer_recovery=outer,seed=seed)
    return result('current_head_proposal',proposal)


MIN_HEAD_EDGE_PX = 24.0


MAX_SCALED_JACOBIAN_CONDITION = 1.0e6


def _local_yaw_uncertainty(cv2, *, profile, camera, pose, sigma):
    import numpy

    points = numpy.asarray([(p.x_m, p.y_m, p.z_m) for p in profile.head_corners])
    rotation = numpy.asarray(pose.rotation_vector, dtype=numpy.float64)
    translation = numpy.asarray(pose.translation_xyz_m, dtype=numpy.float64)
    matrix = numpy.asarray(((camera.fx_px, 0.0, camera.cx_px),
                            (0.0, camera.fy_px, camera.cy_px), (0.0, 0.0, 1.0)))
    rotation_matrix = cv2.Rodrigues(rotation)[0]
    depths = (points @ rotation_matrix.T + translation)[:, 2]
    if not numpy.isfinite(depths).all() or not numpy.all(depths > 1.0e-6):
        return None, None, False
    if abs(float(rotation_matrix[2, 2])) < 1.0e-6:
        return None, None, True
    _projected, jacobian = cv2.projectPoints(points, rotation, translation, matrix, numpy.zeros(4))
    # Scale translation parameters by measured head size, so rank/conditioning
    # does not compare radians with arbitrary meter-valued column magnitudes.
    scale = numpy.diag((1.0, 1.0, 1.0, profile.head_width_m,
                        profile.head_height_m, max(profile.head_width_m, profile.head_height_m)))
    design = numpy.asarray(jacobian[:, :6], dtype=numpy.float64) @ scale
    if design.shape != (8, 6) or not numpy.isfinite(design).all():
        return None, None, True
    _u, singular, vt = numpy.linalg.svd(design, full_matrices=False)
    if singular[-1] <= 0.0:
        return None, None, True
    condition = float(singular[0] / singular[-1])
    if not math.isfinite(condition) or condition > MAX_SCALED_JACOBIAN_CONDITION:
        return None, condition, True
    covariance = (vt.T * (sigma / singular) ** 2) @ vt

    def yaw(rvec):
        normal = cv2.Rodrigues(rvec)[0][:, 2]
        return -math.atan2(float(normal[0]), abs(float(normal[2])))

    gradient = numpy.zeros(6)
    for index in range(3):
        delta = numpy.zeros(3)
        delta[index] = 1.0e-6
        difference = (yaw(rotation + delta) - yaw(rotation - delta) + math.pi / 2.0) % math.pi - math.pi / 2.0
        gradient[index] = difference / 2.0e-6
    variance = float(gradient @ covariance @ gradient)
    return (None if not math.isfinite(variance) or variance < 0.0 else
            math.degrees(math.sqrt(variance))), condition, True


NOISE_ALLOWANCE_MULTIPLIER = 3.0


@dataclass(frozen=True)
class HeadOrientationHypothesis:
    yaw_rad: float
    yaw_std_rad: float
    reprojection_rmse_px: float
    corner_sigma_px: float
    jacobian_condition_number: float
    all_corners_positive_depth: bool
    rotation_vector: tuple[float, float, float]
    translation_xyz_m: tuple[float, float, float]
    face_normal_xyz: tuple[float, float, float]


def _finite(*values):
    try:
        return all(not isinstance(value, bool) and math.isfinite(value) for value in values)
    except (TypeError, ValueError):
        return False


def _complete_border(corners, shape):
    # A rail coinciding with the crop boundary cannot prove the whole head is
    # visible. Require one current pixel outside every extreme border.
    return bool(corners is not None and all(
        1.0 <= p.u_px <= shape[1] - 2.0 and 1.0 <= p.v_px <= shape[0] - 2.0
        for p in corners
    ))


def enclosing_axial_interval(hypotheses):
    """Smallest connected modulo-pi hull of all expanded hypotheses.

    Splitting and merging arcs before selecting a complementary gap avoids
    incorrectly narrowing intervals whose noise allowance crosses the wrap.
    A full-circle union has no useful bounded undirected orientation.
    """

    intervals = []
    for hypothesis in hypotheses:
        center = hypothesis.yaw_rad
        radius = NOISE_ALLOWANCE_MULTIPLIER * hypothesis.yaw_std_rad
        if not _finite(center, radius) or radius < 0.0 or radius >= math.pi / 2.0:
            return None
        start = (center - radius) % math.pi
        end = start + 2.0 * radius
        if end <= math.pi:
            intervals.append((start, end))
        else:
            intervals.extend(((start, math.pi), (0.0, end - math.pi)))
    if not intervals:
        return None
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    gaps = [(following[0] - current[1], following[0])
            for current, following in zip(merged, merged[1:])]
    gaps.append((merged[0][0] + math.pi - merged[-1][1], merged[0][0]))
    gap, start = max(gaps)
    if gap <= 1.0e-12:
        return None
    width = math.pi - gap
    return ((start + width / 2.0 + math.pi / 2.0) % math.pi - math.pi / 2.0,
            width / 2.0)


