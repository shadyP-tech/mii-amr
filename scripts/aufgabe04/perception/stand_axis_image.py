from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


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

    edge_pixels = numpy.asarray(edge_mask) > 0
    distance_input = numpy.where(edge_pixels, 0, 255).astype(numpy.uint8)
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    height, width = distance.shape[:2]

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
        return float(numpy.mean(distance[ys, xs] <= tolerance_px))

    return _QuadrilateralEdgeSupport(
        top=segment_support(top_left, top_right, 0.08, 0.92),
        right=segment_support(top_right, bottom_right, 0.08, 0.92),
        bottom_left=segment_support(bottom_left, bottom_right, 0.08, 0.40),
        bottom_right=segment_support(bottom_left, bottom_right, 0.60, 0.92),
        left=segment_support(top_left, bottom_left, 0.08, 0.92),
        tolerance_px=tolerance_px,
    )


def _validated_refitted_head_corners(
    rough_corners: Sequence[ImagePoint],
    refitted_corners: Sequence[ImagePoint],
    *,
    image_shape,
    stem_center_x: float,
    stem_top_y: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    """Validate a border refit without silently substituting rough geometry."""

    rough = order_corners(rough_corners)
    refitted = order_corners(refitted_corners)
    if not _corners_inside_image(refitted, image_shape):
        return None

    rough_area = _polygon_area(rough)
    refitted_area = _polygon_area(refitted)
    # A close simulation contour can seed on the 53.147 mm white panel while
    # raw Canny correctly refits the 69.930 mm outer board. Their nominal area
    # ratio is about 1.73, so the former 1.60 ceiling rejected the true head.
    # The subsequent local-corner and four-side support gates still prevent a
    # connected arena wall from becoming a rectangle.
    if rough_area <= 0.0 or not 0.55 * rough_area <= refitted_area <= 2.10 * rough_area:
        return None

    aspect_ratio = quadrilateral_aspect_ratio(refitted)
    if not min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
        return None

    top_left, top_right, bottom_right, bottom_left = refitted
    top_width = _distance(top_left, top_right)
    bottom_width = _distance(bottom_left, bottom_right)
    if min(top_width, bottom_width) < 0.45 * max(top_width, bottom_width, 1e-6):
        return None

    rough_xs = [point.u_px for point in rough]
    rough_ys = [point.v_px for point in rough]
    rough_width = max(rough_xs) - min(rough_xs)
    rough_height = max(rough_ys) - min(rough_ys)
    rough_extent = max(rough_width, rough_height, 1.0)
    margin = max(4.0, 0.30 * rough_extent)
    if any(
        point.u_px < min(rough_xs) - margin
        or point.u_px > max(rough_xs) + margin
        or point.v_px < min(rough_ys) - margin
        or point.v_px > max(rough_ys) + margin
        for point in refitted
    ):
        return None

    rough_center_x = (min(rough_xs) + max(rough_xs)) / 2.0
    rough_center_y = (min(rough_ys) + max(rough_ys)) / 2.0
    refitted_xs = [point.u_px for point in refitted]
    refitted_ys = [point.v_px for point in refitted]
    refitted_center_x = (min(refitted_xs) + max(refitted_xs)) / 2.0
    refitted_center_y = (min(refitted_ys) + max(refitted_ys)) / 2.0
    if math.hypot(
        refitted_center_x - rough_center_x,
        refitted_center_y - rough_center_y,
    ) > 0.25 * rough_extent:
        return None

    candidate_left = min(refitted_xs)
    candidate_right = max(refitted_xs)
    candidate_width = candidate_right - candidate_left
    if not (
        candidate_left + 0.12 * candidate_width
        <= stem_center_x
        <= candidate_right - 0.12 * candidate_width
    ):
        return None
    candidate_top = min(refitted_ys)
    candidate_bottom = max(refitted_ys)
    candidate_height = max(candidate_bottom - candidate_top, 1.0)
    if stem_top_y < candidate_bottom - 0.15 * candidate_height:
        return None
    if stem_top_y > candidate_bottom + 0.50 * candidate_height:
        return None
    return refitted


def _select_supported_head_corners(
    cv2,
    face_mask,
    rough_corners: Sequence[ImagePoint],
    refitted_corners: Sequence[ImagePoint] | None,
    *,
    image_shape,
    stem_center_x: float,
    stem_top_y: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    allow_rough_fallback: bool = True,
) -> tuple[
    tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None,
    str,
    _QuadrilateralEdgeSupport | None,
]:
    """Choose only geometry independently supported by the real cutout."""

    best_support: _QuadrilateralEdgeSupport | None = None
    if refitted_corners is not None:
        refitted = _validated_refitted_head_corners(
            rough_corners,
            refitted_corners,
            image_shape=image_shape,
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        if refitted is not None:
            support = _quadrilateral_edge_support(cv2, face_mask, refitted)
            best_support = support
            if support.accepted:
                return refitted, "refitted_rectangle_edge_supported", support

    if allow_rough_fallback:
        # A topology-localized proposal may be retained only after the separate
        # evidence mask confirms all four proposed sides.  In the dual-edge
        # pipeline that mask contains untouched raw-Canny pixels selected in
        # narrow side bands, so morphology still cannot create pose evidence.
        rough = _validated_refitted_head_corners(
            rough_corners,
            rough_corners,
            image_shape=image_shape,
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        if rough is not None:
            rough_support = _quadrilateral_edge_support(cv2, face_mask, rough)
            if best_support is None or rough_support.mean > best_support.mean:
                best_support = rough_support
            if rough_support.accepted:
                return rough, "rough_rectangle_edge_supported", rough_support

    return None, "head_rectangle_fit_unreliable", best_support


def estimate_stand_axis_from_mask(
    cv2,
    mask,
    *,
    min_area_px: float = 250.0,
    min_edge_height_px: float = 8.0,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    stand_depth_m: float | None = None,
    stand_head_bottom_height_m: float | None = None,
) -> StandAxisImageEstimate:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _unusable("no_contour", source="color_mask")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area_px:
        return _unusable("contour_too_small", contour_area_px=area, source="color_mask")

    corners = _quadrilateral_corners(cv2, contour)
    if corners is None:
        return _unusable("no_four_corner_contour", contour_area_px=area, source="color_mask")

    return estimate_stand_axis_from_corners(
        corners,
        min_edge_height_px=min_edge_height_px,
        stand_width_m=stand_width_m,
        stand_distance_m=stand_distance_m,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
        stand_depth_m=stand_depth_m,
        stand_head_bottom_height_m=stand_head_bottom_height_m,
        cv2=cv2,
        contour_area_px=area,
        source="color_mask",
    )


def estimate_stand_axis_from_edges(
    cv2,
    frame,
    *,
    edge_preprocess: str = "outer_border",
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_iterations: int = 1,
    close_kernel: int = 5,
    close_iterations: int = 1,
    hough_threshold: int = 20,
    hough_min_line_length_px: int = 12,
    hough_max_line_gap_px: int = 8,
    min_boundary_line_length_px: float = 35.0,
    face_width_fraction: float = 0.60,
    min_face_area_fraction: float = 0.25,
    min_area_px: float = 250.0,
    min_edge_height_px: float = 8.0,
    min_aspect_ratio: float = 0.45,
    max_aspect_ratio: float = 1.80,
    front_face_to_qr_width_ratio: float | None = None,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    stand_depth_m: float | None = None,
    stand_head_bottom_height_m: float | None = None,
    silhouette_only: bool = False,
    edge_exclusion_mask=None,
) -> tuple[StandAxisImageEstimate, StandAxisEdgeDebugArtifacts]:
    frame_height, frame_width = frame.shape[:2]
    effective_camera_fy_px = camera_fy_px if camera_fy_px is not None else camera_fx_px
    effective_camera_cx_px = camera_cx_px if camera_cx_px is not None else (frame_width - 1.0) / 2.0
    effective_camera_cy_px = camera_cy_px if camera_cy_px is not None else (frame_height - 1.0) / 2.0
    raw_edges = _canny_edges_from_frame(
        cv2,
        frame,
        edge_preprocess=edge_preprocess,
        blur_kernel=blur_kernel,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    topology_edges = _topology_edges_from_frame(
        cv2,
        frame,
        edge_preprocess=edge_preprocess,
        canny_low=canny_low,
        canny_high=canny_high,
        fallback_edges=raw_edges,
    )
    if edge_exclusion_mask is not None:
        if edge_exclusion_mask.shape[:2] != raw_edges.shape[:2]:
            raise ValueError("edge_exclusion_mask must match the processed frame size")
        raw_edges = cv2.bitwise_and(raw_edges, cv2.bitwise_not(edge_exclusion_mask))
        topology_edges = cv2.bitwise_and(
            topology_edges,
            cv2.bitwise_not(edge_exclusion_mask),
        )

    topology_seed = topology_edges.copy()
    if dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        topology_seed = cv2.dilate(
            topology_seed,
            kernel,
            iterations=dilate_iterations,
        )

    topology_hypotheses = _edge_topology_hypotheses(
        cv2,
        topology_seed,
        close_kernel=close_kernel,
        close_iterations=close_iterations,
        include_gap_recovery=silhouette_only,
        edge_exclusion_mask=edge_exclusion_mask,
    )
    edges = topology_hypotheses[0]

    adaptive_min_area_px = max(
        min_area_px,
        _largest_external_bounding_area(cv2, edges) * min_face_area_fraction,
    )

    if silhouette_only:
        # Match the real-camera silhouette pipeline: use connected topology to
        # locate the stem and propose a head quadrilateral, then independently
        # fit its four sides from untouched raw Canny pixels. Simulation
        # forbids QR geometry and synthetic outlines as orientation sources.
        silhouette_face = None
        debug_edges = edges
        for localization_edges in topology_hypotheses:
            hypothesis_min_area_px = max(
                min_area_px,
                _largest_external_bounding_area(cv2, localization_edges)
                * min_face_area_fraction,
            )
            candidate = _plain_face_from_stem_cropped_edges(
                cv2,
                localization_edges,
                measurement_edges=raw_edges,
                min_area_px=hypothesis_min_area_px,
                min_edge_height_px=min_edge_height_px,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                _fixed_parallel_side_direction=(0.0, 1.0),
            )
            if candidate is None:
                continue
            if silhouette_face is None or candidate.rectangle_fit_reliable:
                silhouette_face = candidate
                debug_edges = localization_edges
            if candidate.rectangle_fit_reliable:
                break
        estimate_source = "edge_plain_face_stem_anchor"
        # The edge window shows the connected morphology used for topology;
        # face_mask below contains only selected pre-morphology side evidence.
        if silhouette_face is None:
            return (
                _unusable("silhouette_head_unavailable", source=estimate_source),
                StandAxisEdgeDebugArtifacts(
                    edges=debug_edges,
                    raw_edges=raw_edges,
                ),
            )
        if not silhouette_face.rectangle_fit_reliable:
            return (
                _unusable(
                    silhouette_face.rectangle_fit_reason,
                    corners=silhouette_face.corners,
                    contour_area_px=_polygon_area(silhouette_face.corners),
                    source=estimate_source,
                ),
                StandAxisEdgeDebugArtifacts(
                    edges=debug_edges,
                    face_mask=silhouette_face.face_mask,
                    raw_edges=raw_edges,
                ),
            )
        estimate = estimate_stand_axis_from_corners(
            silhouette_face.corners,
            min_edge_height_px=min_edge_height_px,
            stand_width_m=stand_width_m,
            stand_distance_m=stand_distance_m,
            camera_fx_px=camera_fx_px,
            camera_fy_px=effective_camera_fy_px,
            camera_cx_px=effective_camera_cx_px,
            camera_cy_px=effective_camera_cy_px,
            stand_depth_m=stand_depth_m,
            stand_head_bottom_height_m=stand_head_bottom_height_m,
            cv2=cv2,
            contour_area_px=_polygon_area(silhouette_face.corners),
            source=estimate_source,
        )
        return estimate, StandAxisEdgeDebugArtifacts(
            edges=debug_edges,
            face_mask=silhouette_face.face_mask,
            rectangle_mask=_debug_rectangle_image(cv2, debug_edges.shape, silhouette_face.corners),
            rectangle_overlay=_debug_rectangle_overlay_image(
                cv2,
                debug_edges.shape,
                silhouette_face.corners,
                silhouette_face.face_mask,
            ),
            raw_edges=raw_edges,
        )

    qr_front_face = _front_face_from_qr_geometry(
        cv2,
        frame,
        edges,
        width_ratio=front_face_to_qr_width_ratio,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if qr_front_face is not None:
        return (
            estimate_stand_axis_from_corners(
                qr_front_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(qr_front_face.corners),
                source="edge_qr_scaled_front",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=qr_front_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, qr_front_face.corners),
                rectangle_overlay=_debug_rectangle_overlay_image(
                    cv2,
                    edges.shape,
                    qr_front_face.corners,
                    qr_front_face.face_mask,
                ),
                raw_edges=raw_edges,
            ),
        )

    plain_face = _plain_face_from_stem_cropped_edges(
        cv2,
        edges,
        measurement_edges=raw_edges,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if plain_face is not None:
        if not plain_face.rectangle_fit_reliable:
            return (
                _unusable(
                    plain_face.rectangle_fit_reason,
                    corners=plain_face.corners,
                    contour_area_px=_polygon_area(plain_face.corners),
                    source="edge_plain_face_stem_anchor",
                ),
                StandAxisEdgeDebugArtifacts(
                    edges=edges,
                    face_mask=plain_face.face_mask,
                    raw_edges=raw_edges,
                ),
            )
        return (
            estimate_stand_axis_from_corners(
                plain_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(plain_face.corners),
                source="edge_plain_face_stem_anchor",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=plain_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, plain_face.corners),
                rectangle_overlay=_debug_rectangle_overlay_image(
                    cv2,
                    edges.shape,
                    plain_face.corners,
                    plain_face.face_mask,
                ),
                raw_edges=raw_edges,
            ),
        )

    stem_face = _stem_anchored_face_from_edges(
        cv2,
        edges,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if stem_face is not None:
        return (
            estimate_stand_axis_from_corners(
                stem_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(stem_face.corners),
                source="edge_stem_anchor",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=stem_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, stem_face.corners),
                rectangle_overlay=_debug_rectangle_overlay_image(
                    cv2,
                    edges.shape,
                    stem_face.corners,
                    stem_face.face_mask,
                ),
                raw_edges=raw_edges,
            ),
        )

    silhouette_face = _face_quadrilateral_from_silhouette(
        cv2,
        edges,
        min_area_px=adaptive_min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        face_width_fraction=face_width_fraction,
    )
    if silhouette_face is not None:
        return (
            estimate_stand_axis_from_corners(
                silhouette_face.corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(silhouette_face.corners),
                source="edge_silhouette",
            ),
            StandAxisEdgeDebugArtifacts(
                edges=edges,
                face_mask=silhouette_face.face_mask,
                rectangle_mask=_debug_rectangle_image(cv2, edges.shape, silhouette_face.corners),
                rectangle_overlay=_debug_rectangle_overlay_image(
                    cv2,
                    edges.shape,
                    silhouette_face.corners,
                    silhouette_face.face_mask,
                ),
                raw_edges=raw_edges,
            ),
        )

    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: StandAxisImageEstimate | None = None
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < adaptive_min_area_px:
            continue
        corners = _quadrilateral_corners(cv2, contour)
        if corners is None or not cv2.isContourConvex(_points_to_cv2(corners)):
            continue
        if _contour_has_lower_appendage(cv2, contour, corners):
            continue
        aspect_ratio = quadrilateral_aspect_ratio(corners)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        estimate = estimate_stand_axis_from_corners(
            corners,
            min_edge_height_px=min_edge_height_px,
            stand_width_m=stand_width_m,
            stand_distance_m=stand_distance_m,
            camera_fx_px=camera_fx_px,
            camera_fy_px=effective_camera_fy_px,
            camera_cx_px=effective_camera_cx_px,
            camera_cy_px=effective_camera_cy_px,
            stand_depth_m=stand_depth_m,
            stand_head_bottom_height_m=stand_head_bottom_height_m,
            cv2=cv2,
            contour_area_px=area,
            source="edges",
        )
        if not estimate.usable:
            continue
        score = score_quadrilateral_candidate(corners, area)
        if score > best_score:
            best = estimate
            best_score = score

    if best is not None:
        return best, StandAxisEdgeDebugArtifacts(edges=edges, raw_edges=raw_edges)

    line_corners = _quadrilateral_from_line_segments(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
        min_boundary_line_length_px=min_boundary_line_length_px,
        min_edge_height_px=min_edge_height_px,
        min_area_px=adaptive_min_area_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if line_corners is not None:
        return (
            estimate_stand_axis_from_corners(
                line_corners,
                min_edge_height_px=min_edge_height_px,
                stand_width_m=stand_width_m,
                stand_distance_m=stand_distance_m,
                camera_fx_px=camera_fx_px,
                camera_fy_px=effective_camera_fy_px,
                camera_cx_px=effective_camera_cx_px,
                camera_cy_px=effective_camera_cy_px,
                stand_depth_m=stand_depth_m,
                stand_head_bottom_height_m=stand_head_bottom_height_m,
                cv2=cv2,
                contour_area_px=_polygon_area(line_corners),
                source="edge_lines",
            ),
            StandAxisEdgeDebugArtifacts(edges=edges, raw_edges=raw_edges),
        )

    edge_on = _edge_on_from_line_segments(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
        min_boundary_line_length_px=min_boundary_line_length_px,
        min_edge_height_px=min_edge_height_px,
    )
    if edge_on is not None:
        return edge_on, StandAxisEdgeDebugArtifacts(edges=edges, raw_edges=raw_edges)

    if best is None:
        return (
            _unusable("no_edge_quadrilateral", source="edges"),
            StandAxisEdgeDebugArtifacts(edges=edges, raw_edges=raw_edges),
        )


def estimate_stand_axis_from_corners(
    corners: Sequence[ImagePoint],
    *,
    min_edge_height_px: float = 8.0,
    stand_width_m: float | None = None,
    stand_distance_m: float | None = None,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    stand_depth_m: float | None = None,
    stand_head_bottom_height_m: float | None = None,
    cv2=None,
    contour_area_px: float = 0.0,
    source: str = "corners",
) -> StandAxisImageEstimate:
    corners = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = corners
    left_height = _distance(top_left, bottom_left)
    right_height = _distance(top_right, bottom_right)
    if left_height < min_edge_height_px or right_height < min_edge_height_px:
        return _unusable("edge_too_short", corners=corners, contour_area_px=contour_area_px, source=source)

    ratio = left_height / right_height
    yaw_proxy = (ratio - 1.0) / (ratio + 1.0)
    closer_side = "left" if left_height > right_height else "right" if right_height > left_height else "equal"
    yaw_deg = _yaw_deg_from_square_pnp(
        cv2,
        corners,
        stand_width_m=stand_width_m,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
    )
    if yaw_deg is None:
        yaw_deg = _yaw_deg_from_projected_width(corners, stand_width_m, stand_distance_m, camera_fx_px)
    if yaw_deg is None:
        yaw_deg = _yaw_deg_from_ratio(ratio, stand_width_m, stand_distance_m)

    return StandAxisImageEstimate(
        usable=True,
        reason="axis_estimated",
        mode="face_visible",
        corners=corners,
        axis_line=None,
        left_height_px=left_height,
        right_height_px=right_height,
        height_ratio=ratio,
        yaw_proxy=yaw_proxy,
        yaw_deg=yaw_deg,
        closer_side=closer_side,
        contour_area_px=contour_area_px,
        source=source,
    )


def quadrilateral_aspect_ratio(corners: Sequence[ImagePoint]) -> float:
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    width = (_distance(top_left, top_right) + _distance(bottom_left, bottom_right)) / 2.0
    height = (_distance(top_left, bottom_left) + _distance(top_right, bottom_right)) / 2.0
    if height <= 0.0:
        return 0.0
    return width / height


def score_quadrilateral_candidate(corners: Sequence[ImagePoint], area_px: float) -> float:
    aspect_ratio = quadrilateral_aspect_ratio(corners)
    aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect_ratio, 1e-6))))
    return area_px * (0.5 + 0.5 * aspect_score)


def wide_row_band(row_widths: Sequence[int], *, width_fraction: float = 0.60, max_gap: int = 3) -> tuple[int, int] | None:
    if not row_widths:
        return None
    max_width = max(row_widths)
    if max_width <= 0:
        return None
    threshold = max_width * width_fraction
    best = None
    best_length = -1
    start = None
    last_wide = None
    gap = 0
    for index, width in enumerate(row_widths):
        if width >= threshold:
            if start is None:
                start = index
            last_wide = index
            gap = 0
            continue
        if start is not None:
            gap += 1
            if gap > max_gap:
                end = last_wide if last_wide is not None else index - gap
                length = end - start + 1
                if length > best_length:
                    best = (start, end)
                    best_length = length
                start = None
                last_wide = None
                gap = 0
    if start is not None:
        end = last_wide if last_wide is not None else len(row_widths) - 1
        length = end - start + 1
        if length > best_length:
            best = (start, end)
    return best


def _front_face_from_qr_geometry(
    cv2,
    frame,
    edges,
    *,
    width_ratio: float | None,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate | None:
    if width_ratio is None or width_ratio <= 1.0:
        return None
    qr_corners = _detect_qr_quad_corners(cv2, frame)
    if qr_corners is None:
        return None
    corners = _scale_quadrilateral_about_center(qr_corners, width_ratio)
    area = _polygon_area(corners)
    if area < min_area_px:
        return None
    if not _corners_inside_image(corners, edges.shape):
        return None
    aspect_ratio = quadrilateral_aspect_ratio(corners)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return None
    estimate = estimate_stand_axis_from_corners(
        corners,
        min_edge_height_px=min_edge_height_px,
        contour_area_px=area,
        source="edge_qr_scaled_front",
    )
    if not estimate.usable:
        return None

    return _SilhouetteFaceCandidate(
        corners=corners,
        face_mask=_debug_outline_image(cv2, edges.shape, corners),
    )


def _detect_qr_quad_corners(cv2, frame) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    detector = cv2.QRCodeDetector()
    try:
        ok, points = detector.detect(frame)
    except Exception:
        ok, points = False, None
    if not ok or points is None:
        try:
            multi_result = detector.detectAndDecodeMulti(frame)
        except Exception:
            multi_result = ()
        points = multi_result[2] if len(multi_result) > 2 else None
        if points is None or len(points) == 0:
            return None
        points = points[0]
    try:
        flat_points = points.reshape(-1, 2)
    except Exception:
        return None
    if len(flat_points) < 4:
        return None
    return order_corners(tuple(ImagePoint(float(point[0]), float(point[1])) for point in flat_points[:4]))


def _scale_quadrilateral_about_center(
    corners: Sequence[ImagePoint],
    scale: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]:
    ordered = order_corners(corners)
    center_u = sum(point.u_px for point in ordered) / 4.0
    center_v = sum(point.v_px for point in ordered) / 4.0
    return order_corners(
        tuple(
            ImagePoint(
                center_u + (point.u_px - center_u) * scale,
                center_v + (point.v_px - center_v) * scale,
            )
            for point in ordered
        )
    )


def _corners_inside_image(corners: Sequence[ImagePoint], image_shape) -> bool:
    height, width = image_shape[:2]
    return all(0.0 <= point.u_px < width and 0.0 <= point.v_px < height for point in corners)


def _debug_outline_image(cv2, image_shape, corners: Sequence[ImagePoint]):
    import numpy

    outline = numpy.zeros(image_shape[:2], dtype=numpy.uint8)
    polygon = numpy.array(
        [[(int(round(point.u_px)), int(round(point.v_px))) for point in order_corners(corners)]],
        dtype=numpy.int32,
    )
    cv2.polylines(outline, polygon, isClosed=True, color=255, thickness=2)
    return outline


def _debug_rectangle_image(cv2, image_shape, corners: Sequence[ImagePoint]):
    return _debug_outline_image(cv2, image_shape, corners)


def _debug_rectangle_overlay_image(
    cv2,
    image_shape,
    corners: Sequence[ImagePoint],
    face_mask,
):
    """Show the accepted rectangle and the cutout that supports it together."""

    import numpy

    overlay = numpy.zeros(image_shape[:2], dtype=numpy.uint8)
    if face_mask is not None:
        overlay[numpy.asarray(face_mask) > 0] = 96
    rectangle = _debug_rectangle_image(cv2, image_shape, corners)
    overlay[rectangle > 0] = 255
    return overlay


def _debug_polygon_edge_cutout_image(
    cv2,
    edges,
    corners: Sequence[ImagePoint],
    *,
    thickness_px: int,
):
    import numpy

    cutout = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    boundary = _debug_outline_image(cv2, edges.shape, corners)
    if thickness_px > 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness_px, thickness_px))
        boundary = cv2.dilate(boundary, kernel, iterations=1)
    return cv2.bitwise_and(edges, boundary, dst=cutout)


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
):
    """Fit one proposed side from nearby raw pixels without connectivity.

    The topology side supplies only a search band. For every pixel-sized bin
    along that side, one untouched Canny pixel is selected by the requested
    outer/nearest policy and a robust line is fitted through those samples.
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


def _parallel_side_run_endpoints(
    edge_points,
    fitted_line,
    rough_start: ImagePoint,
    rough_end: ImagePoint,
    side_evidence,
    *,
    band_px: float,
):
    """Recover one head side's real endpoints from its aligned raw-edge run.

    A row-envelope proposal can be horizontally topped even when perspective
    makes the real head top strongly sloped. Once an outer side has been fitted
    with the calibrated parallel direction, its contiguous raw-Canny run gives
    the two corner heights without relying on that inaccurate top proposal.
    The search stays local to the rough side and must overlap the evidence that
    supported the fitted line, preventing a separate stem/base run from winning.
    """

    import numpy

    line_x, line_y, direction_x, direction_y = fitted_line
    direction = numpy.array([direction_x, direction_y], dtype=numpy.float64)
    direction_norm = float(numpy.linalg.norm(direction))
    if direction_norm <= 1e-9:
        return None
    direction /= direction_norm
    rough_tangent = numpy.array(
        [
            rough_end.u_px - rough_start.u_px,
            rough_end.v_px - rough_start.v_px,
        ],
        dtype=numpy.float64,
    )
    rough_length = float(numpy.linalg.norm(rough_tangent))
    if rough_length <= 1e-6:
        return None
    if float(direction @ rough_tangent) < 0.0:
        direction *= -1.0

    fit_point = numpy.array([line_x, line_y], dtype=numpy.float64)
    fit_normal = numpy.array([-direction[1], direction[0]], dtype=numpy.float64)
    along = (edge_points - fit_point) @ direction
    residuals = numpy.abs((edge_points - fit_point) @ fit_normal)
    rough_along = numpy.array(
        [
            (numpy.array([rough_start.u_px, rough_start.v_px]) - fit_point)
            @ direction,
            (numpy.array([rough_end.u_px, rough_end.v_px]) - fit_point)
            @ direction,
        ],
        dtype=numpy.float64,
    )
    search_margin = max(4.0, 0.35 * rough_length)
    residual_limit_px = max(1.25, min(2.5, 0.50 * band_px))
    local_mask = (
        (residuals <= residual_limit_px)
        & (along >= float(rough_along.min()) - search_margin)
        & (along <= float(rough_along.max()) + search_margin)
    )
    local_points = edge_points[local_mask]
    local_along = along[local_mask]
    if len(local_points) < 5:
        return None

    local_bins = numpy.floor(local_along + 0.5).astype(numpy.int32)
    unique_bins = numpy.unique(local_bins)
    if len(unique_bins) < 5:
        return None
    # Arena-wall suppression can cross an otherwise continuous outer side at
    # the wall/stand depth discontinuity.  In the live simulation capture it
    # removed roughly one quarter of the near side while leaving both the
    # upper run and the real bottom corner.  Keep those fragments in one local
    # side run.  The search is still constrained to the fitted line, to the
    # rough head neighbourhood, and to a run overlapping the original side
    # evidence, so this cannot jump across to the stem or another stand.
    maximum_gap_bins = max(3, int(math.ceil(0.30 * rough_length)))
    runs: list[tuple[int, int]] = []
    run_start = run_end = int(unique_bins[0])
    for raw_bin in unique_bins[1:]:
        current_bin = int(raw_bin)
        if current_bin - run_end > maximum_gap_bins:
            runs.append((run_start, run_end))
            run_start = current_bin
        run_end = current_bin
    runs.append((run_start, run_end))

    if len(side_evidence):
        evidence_bins = numpy.floor(
            (side_evidence - fit_point) @ direction + 0.5
        ).astype(numpy.int32)
    else:
        evidence_bins = numpy.empty((0,), dtype=numpy.int32)

    def run_rank(run: tuple[int, int]) -> tuple[int, int, int]:
        start_bin, end_bin = run
        overlap = int(
            numpy.count_nonzero(
                (evidence_bins >= start_bin) & (evidence_bins <= end_bin)
            )
        )
        span = end_bin - start_bin
        support = int(
            numpy.count_nonzero(
                (local_bins >= start_bin) & (local_bins <= end_bin)
            )
        )
        return overlap, span, support

    selected_start, selected_end = max(runs, key=run_rank)
    selected_overlap, selected_span, _selected_support = run_rank(
        (selected_start, selected_end)
    )
    minimum_overlap = max(5, int(math.ceil(0.25 * len(evidence_bins))))
    if selected_overlap < minimum_overlap:
        return None
    if selected_span < max(8.0, 0.55 * rough_length):
        return None

    run_mask = (local_bins >= selected_start) & (local_bins <= selected_end)
    run_points = local_points[run_mask]
    start_point = fit_point + float(selected_start) * direction
    end_point = fit_point + float(selected_end) * direction
    return (
        ImagePoint(float(start_point[0]), float(start_point[1])),
        ImagePoint(float(end_point[0]), float(end_point[1])),
        run_points,
    )


def _level_camera_endpoint_perspective_consistent(
    left_top: ImagePoint,
    left_bottom: ImagePoint,
    right_top: ImagePoint,
    right_bottom: ImagePoint,
) -> bool:
    """Reject a wall crossing masquerading as one parallel-side endpoint.

    The simulated head is vertical and its bottom remains above the fixed
    camera.  Both horizontal board edges therefore share a slope direction,
    while the upper edge (farther from camera height) has at least as much
    perspective slope as the lower edge.  A truncated side ending on the arena
    seam reverses that relationship.
    """

    top_delta_y = right_top.v_px - left_top.v_px
    bottom_delta_y = right_bottom.v_px - left_bottom.v_px
    mean_height = (
        _distance(left_top, left_bottom)
        + _distance(right_top, right_bottom)
    ) / 2.0
    tolerance_px = max(2.0, 0.04 * mean_height)
    if (
        top_delta_y * bottom_delta_y < 0.0
        and min(abs(top_delta_y), abs(bottom_delta_y)) > tolerance_px
    ):
        return False
    return abs(bottom_delta_y) <= abs(top_delta_y) + tolerance_px


def _raw_side_evidence_and_corners(
    cv2,
    raw_edges,
    rough_corners: Sequence[ImagePoint],
    *,
    fixed_parallel_side_direction: tuple[float, float] | None = None,
):
    """Refit a vertical-sided head trapezoid from outer raw-Canny evidence.

    The left and right sides share one image direction. Simulation supplies
    the calibrated vertical direction exactly; other callers estimate one
    robust common direction from both raw sides. Top and bottom retain their
    own perspective slopes.
    """

    import numpy

    evidence_mask = numpy.zeros(raw_edges.shape[:2], dtype=numpy.uint8)
    locations = cv2.findNonZero(raw_edges)
    if locations is None:
        return evidence_mask, None
    edge_points = locations.reshape(-1, 2).astype(numpy.float64)
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
    band_px = float(max(3, min(6, int(round(0.08 * minimum_extent)))))
    parallel_side_band_px = band_px
    if fixed_parallel_side_direction is not None:
        # At close simulation viewpoints the connected topology can describe
        # the white-panel boundary rather than the board silhouette.  The
        # panel is 53.147 mm wide inside a 69.930 mm board, so either real
        # outer side may be about 15.8% of the panel width beyond that rough
        # proposal. Rasterized topology can sit another 2-3 px inside the panel
        # edge, so retain a 24% corridor for that localization error. Scale
        # only the calibrated parallel-side search; top/bottom retain the
        # narrow wall-safe corridor.
        parallel_side_band_px = float(
            max(
                band_px,
                min(24, int(round(0.24 * minimum_extent))),
            )
        )
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

    def fit_side(name: str, *, fixed_direction=None):
        start, end, intervals, outward_sign, minimum_coverage = side_specs[name]
        side_band_px = (
            parallel_side_band_px
            if name in ("left", "right")
            else band_px
        )
        return _fit_raw_edge_side_in_band(
            cv2,
            edge_points,
            start,
            end,
            band_px=side_band_px,
            intervals=intervals,
            outward_sign=outward_sign,
            fixed_direction=fixed_direction,
            minimum_coverage=minimum_coverage,
        )

    top, top_evidence = fit_side("top")
    bottom, bottom_evidence = fit_side("bottom")

    shared_direction = fixed_parallel_side_direction
    if shared_direction is None:
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

    parallel_endpoint_corners = None
    if (
        fixed_parallel_side_direction is not None
        and right is not None
        and left is not None
    ):
        # In the level simulation camera, the two fitted outer sides are the
        # most stable head evidence. Recover their complete raw runs and use
        # the endpoint pairs as accurate proposals for the perspective-sloped
        # top and bottom. This repairs row-envelope proposals without accepting
        # topology-only geometry or inferring an unsupported fourth side.
        left_run = _parallel_side_run_endpoints(
            edge_points,
            left,
            top_left,
            bottom_left,
            left_evidence,
            band_px=parallel_side_band_px,
        )
        right_run = _parallel_side_run_endpoints(
            edge_points,
            right,
            top_right,
            bottom_right,
            right_evidence,
            band_px=parallel_side_band_px,
        )
        if left_run is not None and right_run is not None:
            left_top, left_bottom, recovered_left_evidence = left_run
            right_top, right_bottom, recovered_right_evidence = right_run
            if not _level_camera_endpoint_perspective_consistent(
                left_top,
                left_bottom,
                right_top,
                right_bottom,
            ):
                left_run = right_run = None
            else:
                recovered_top, recovered_top_evidence = _fit_raw_edge_side_in_band(
                    cv2,
                    edge_points,
                    left_top,
                    right_top,
                    band_px=band_px,
                    intervals=((0.04, 0.96),),
                    outward_sign=0.0,
                    minimum_coverage=0.50,
                )
                recovered_bottom, recovered_bottom_evidence = (
                    _fit_raw_edge_side_in_band(
                        cv2,
                        edge_points,
                        left_bottom,
                        right_bottom,
                        band_px=band_px,
                        intervals=((0.04, 0.42), (0.58, 0.96)),
                        outward_sign=0.0,
                        minimum_coverage=0.40,
                    )
                )
                if recovered_top is not None and recovered_bottom is not None:
                    top = recovered_top
                    bottom = recovered_bottom
                    top_evidence = recovered_top_evidence
                    bottom_evidence = recovered_bottom_evidence
                    left_evidence = recovered_left_evidence
                    right_evidence = recovered_right_evidence
                    # The side-run endpoints are directly observed corner pixels.
                    # Keep them as the final corners; intersecting an infinite top
                    # fit extrapolated from only its middle fragment can otherwise
                    # overshoot a rounded outer corner by several pixels.
                    parallel_endpoint_corners = (
                        left_top,
                        right_top,
                        right_bottom,
                        left_bottom,
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
    if parallel_endpoint_corners is not None:
        ordered_endpoint_corners = order_corners(parallel_endpoint_corners)
        if _quadrilateral_edge_support(
            cv2,
            evidence_mask,
            ordered_endpoint_corners,
        ).accepted:
            return evidence_mask, ordered_endpoint_corners
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
        fixed_parallel_side_direction is not None
        and not _level_camera_endpoint_perspective_consistent(
            ordered_corners[0],
            ordered_corners[3],
            ordered_corners[1],
            ordered_corners[2],
        )
    ):
        return evidence_mask, None
    return evidence_mask, ordered_corners


def _expanded_head_edge_roi(
    cv2,
    edges,
    rough_corners: Sequence[ImagePoint],
    *,
    margin_px: int,
    stem_center_x: float | None = None,
    stem_top_y: float | None = None,
    min_edge_height_px: float = 8.0,
):
    import numpy

    ordered = order_corners(rough_corners)
    min_x = min(point.u_px for point in ordered)
    max_x = max(point.u_px for point in ordered)
    min_y = min(point.v_px for point in ordered)
    max_y = max(point.v_px for point in ordered)
    width = max_x - min_x
    height = max_y - min_y
    horizontal_margin = max(float(margin_px), 0.18 * width)
    top_margin = max(float(margin_px), 0.35 * height)
    bottom_margin = max(float(margin_px), 0.14 * height)

    x_min = max(0, int(math.floor(min_x - horizontal_margin)))
    x_max = min(edges.shape[1], int(math.ceil(max_x + horizontal_margin)) + 1)
    y_min = max(0, int(math.floor(min_y - top_margin)))
    y_max = min(edges.shape[0], int(math.ceil(max_y + bottom_margin)) + 1)

    edge_roi = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    if x_max <= x_min or y_max <= y_min:
        return edge_roi
    edge_roi[y_min:y_max, x_min:x_max] = edges[y_min:y_max, x_min:x_max]

    if stem_center_x is None or stem_top_y is None:
        return edge_roi

    rough_bottom_y = max_y
    erase_from_y = int(round(min(rough_bottom_y + max(2.0, 0.25 * min_edge_height_px), stem_top_y)))
    if erase_from_y >= edge_roi.shape[0]:
        return edge_roi

    stem_half_width = max(5, int(round(min_edge_height_px * 1.2)))
    stem_x = int(round(stem_center_x))
    erase_left = max(0, stem_x - stem_half_width)
    erase_right = min(edge_roi.shape[1], stem_x + stem_half_width + 1)
    if erase_left < erase_right:
        edge_roi[erase_from_y:, erase_left:erase_right] = 0
    return edge_roi


def _connected_border_mask_and_corners(
    cv2,
    edges,
    edge_cutout,
    *,
    fallback_corners: Sequence[ImagePoint],
    min_edge_height_px: float,
):
    import numpy

    # ``fallback_corners`` remains part of this private call contract because
    # callers also use it for the final independent support check.  It must
    # never be returned here merely because every border fit failed.
    _ = fallback_corners

    line_fit_corners = _cutout_outer_border_line_corners(
        cv2,
        edge_cutout,
        min_edge_height_px=min_edge_height_px,
    )
    if line_fit_corners is not None:
        return edge_cutout, line_fit_corners

    cutout_rect_corners = _cutout_min_area_rect_corners(
        cv2,
        edge_cutout,
        min_edge_height_px=min_edge_height_px,
    )
    if cutout_rect_corners is not None:
        return edge_cutout, cutout_rect_corners

    hull_corners = _outer_hull_corners(cv2, edge_cutout, min_edge_height_px=min_edge_height_px)
    if hull_corners is not None:
        return edge_cutout, hull_corners

    row_bounds = []
    max_span = 0.0
    for y_px in range(edge_cutout.shape[0]):
        columns = numpy.flatnonzero(edge_cutout[y_px, :])
        if len(columns) < 2:
            continue
        left = float(columns[0])
        right = float(columns[-1])
        span = right - left
        if span <= 0.0:
            continue
        row_bounds.append((float(y_px), left, right, span))
        max_span = max(max_span, span)

    if len(row_bounds) < 4 or max_span < min_edge_height_px:
        return edge_cutout, None

    side_rows = [row for row in row_bounds if row[3] >= max(min_edge_height_px, 0.38 * max_span)]
    if len(side_rows) < 4:
        return edge_cutout, None

    ys = numpy.array([row[0] for row in side_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in side_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in side_rows], dtype=numpy.float64)
    spans = numpy.array([row[3] for row in side_rows], dtype=numpy.float64)
    broad_ys = ys[spans >= max(min_edge_height_px, 0.62 * max_span)]
    if len(broad_ys) >= 2:
        top_y = float(broad_ys.min())
        bottom_y = float(broad_ys.max())
    else:
        top_y = float(ys.min())
        bottom_y = float(ys.max())
    if bottom_y - top_y < min_edge_height_px:
        return edge_cutout, None

    left_line = _fit_x_line_at_ys(ys, lefts)
    right_line = _fit_x_line_at_ys(ys, rights)
    top_line, bottom_line = _fit_top_bottom_y_lines(edge_cutout, min_edge_height_px=min_edge_height_px)
    if left_line is not None and right_line is not None and top_line is not None and bottom_line is not None:
        top_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, top_line)
        top_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, top_line)
        bottom_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, bottom_line)
        bottom_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, bottom_line)
        if None not in (top_left, top_right, bottom_right, bottom_left):
            corners = order_corners((top_left, top_right, bottom_right, bottom_left))
        else:
            corners = _outer_row_envelope_corners(
                left_line,
                right_line,
                top_y=top_y,
                bottom_y=bottom_y,
                min_edge_height_px=min_edge_height_px,
            )
    else:
        corners = _outer_row_envelope_corners(
            left_line,
            right_line,
            top_y=top_y,
            bottom_y=bottom_y,
            min_edge_height_px=min_edge_height_px,
        )
    if corners is None:
        return edge_cutout, None

    return edge_cutout, corners


def _edge_pixels_inside_polygon(cv2, edges, corners: Sequence[ImagePoint], *, margin_px: int = 0):
    import numpy

    mask = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    polygon = numpy.array(
        [[(int(round(point.u_px)), int(round(point.v_px))) for point in corners]],
        dtype=numpy.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    if margin_px > 0:
        kernel_size = max(1, int(margin_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.bitwise_and(edges, mask)


def _cutout_outer_border_line_corners(
    cv2,
    edge_cutout,
    *,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    import numpy

    row_bounds = []
    max_span = 0.0
    for y_px in range(edge_cutout.shape[0]):
        columns = numpy.flatnonzero(edge_cutout[y_px, :])
        if len(columns) < 2:
            continue
        left = float(columns[0])
        right = float(columns[-1])
        span = right - left
        if span <= 0.0:
            continue
        row_bounds.append((float(y_px), left, right, span))
        max_span = max(max_span, span)
    if len(row_bounds) < 4 or max_span < min_edge_height_px:
        return None

    side_rows = [row for row in row_bounds if row[3] >= max(min_edge_height_px, 0.36 * max_span)]
    if len(side_rows) < 4:
        return None

    ys = numpy.array([row[0] for row in side_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in side_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in side_rows], dtype=numpy.float64)
    spans = numpy.array([row[3] for row in side_rows], dtype=numpy.float64)
    broad_ys = ys[spans >= max(min_edge_height_px, 0.58 * max_span)]
    if len(broad_ys) < 2:
        return None
    top_y = float(broad_ys.min())
    bottom_y = float(broad_ys.max())
    if bottom_y - top_y < min_edge_height_px:
        return None

    top_band_px = max(3.0, min_edge_height_px * 0.55)
    bottom_band_px = max(3.0, min_edge_height_px * 0.65)
    min_horizontal_length_px = max(min_edge_height_px, 0.16 * max_span)
    top_line = _fit_y_line_from_border_segments(
        cv2,
        edge_cutout,
        target_y=top_y,
        band_px=top_band_px,
        min_length_px=min_horizontal_length_px,
        prefer_lower=False,
    )
    if top_line is None:
        top_line = _fit_y_line_from_extreme_column_points(
            edge_cutout,
            target_y=top_y,
            band_px=top_band_px,
            use_top=True,
        )
    bottom_line = _fit_y_line_from_border_segments(
        cv2,
        edge_cutout,
        target_y=bottom_y,
        band_px=bottom_band_px,
        min_length_px=max(6.0, 0.11 * max_span),
        prefer_lower=True,
    )
    if bottom_line is None:
        bottom_line = _fit_y_line_from_extreme_column_points(
            edge_cutout,
            target_y=bottom_y,
            band_px=bottom_band_px,
            use_top=False,
        )
    vertical_margin = max(3.0, 0.16 * (bottom_y - top_y))
    side_fit_rows = [
        row
        for row in side_rows
        if top_y + vertical_margin <= row[0] <= bottom_y - vertical_margin
    ]
    if len(side_fit_rows) < 4:
        side_fit_rows = side_rows
    side_ys = numpy.array([row[0] for row in side_fit_rows], dtype=numpy.float64)
    side_lefts = numpy.array([row[1] for row in side_fit_rows], dtype=numpy.float64)
    side_rights = numpy.array([row[2] for row in side_fit_rows], dtype=numpy.float64)
    left_line = _fit_x_line_at_ys(side_ys, side_lefts)
    right_line = _fit_x_line_at_ys(side_ys, side_rights)
    if left_line is None or right_line is None or top_line is None or bottom_line is None:
        return None

    top_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, top_line)
    top_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, top_line)
    bottom_right = _intersect_x_of_y_line_with_y_of_x_line(right_line, bottom_line)
    bottom_left = _intersect_x_of_y_line_with_y_of_x_line(left_line, bottom_line)
    if None in (top_left, top_right, bottom_right, bottom_left):
        return None
    corners = order_corners((top_left, top_right, bottom_right, bottom_left))
    top_left, top_right, bottom_right, bottom_left = corners
    if min(_distance(top_left, top_right), _distance(bottom_left, bottom_right)) < min_edge_height_px:
        return None
    if min(_distance(top_left, bottom_left), _distance(top_right, bottom_right)) < min_edge_height_px:
        return None
    return corners


def _fit_y_line_from_border_segments(
    cv2,
    edge_cutout,
    *,
    target_y: float,
    band_px: float,
    min_length_px: float,
    prefer_lower: bool,
) -> tuple[float, float] | None:
    import numpy

    if cv2 is None:
        return None
    hough_min_length = max(5, int(round(min_length_px)))
    segments = _line_segments_from_edges(
        cv2,
        edge_cutout,
        hough_threshold=8,
        hough_min_line_length_px=hough_min_length,
        hough_max_line_gap_px=max(4, int(round(band_px))),
    )
    if not segments:
        return None

    candidates = []
    for segment in segments:
        if abs(segment.angle_deg) > 38.0:
            continue
        y_mid = (segment.start.v_px + segment.end.v_px) / 2.0
        distance_from_target = abs(y_mid - target_y)
        if distance_from_target > band_px:
            continue
        if segment.length_px < min_length_px:
            continue
        y_bias = y_mid - target_y if prefer_lower else target_y - y_mid
        score = segment.length_px - 2.0 * distance_from_target + 0.35 * max(0.0, y_bias)
        candidates.append((score, segment))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [segment for _score, segment in candidates[:4]]
    points = []
    for segment in selected:
        points.append((segment.start.u_px, segment.start.v_px))
        points.append((segment.end.u_px, segment.end.v_px))
    if len(points) < 4:
        return None

    xs = numpy.array([point[0] for point in points], dtype=numpy.float64)
    ys = numpy.array([point[1] for point in points], dtype=numpy.float64)
    if float(xs.max() - xs.min()) < max(3.0, min_length_px):
        return None
    slope, intercept = numpy.polyfit(xs, ys, 1)
    if not math.isfinite(float(slope)) or not math.isfinite(float(intercept)):
        return None
    return float(slope), float(intercept)


def _fit_y_line_from_extreme_column_points(
    edge_cutout,
    *,
    target_y: float,
    band_px: float,
    use_top: bool,
) -> tuple[float, float] | None:
    import numpy

    points = []
    lower = target_y - band_px
    upper = target_y + band_px
    for x_px in range(edge_cutout.shape[1]):
        rows = numpy.flatnonzero(edge_cutout[:, x_px])
        if len(rows) == 0:
            continue
        y_px = float(rows[0] if use_top else rows[-1])
        if lower <= y_px <= upper:
            points.append((float(x_px), y_px))
    if len(points) < 4:
        return None
    xs = numpy.array([point[0] for point in points], dtype=numpy.float64)
    ys = numpy.array([point[1] for point in points], dtype=numpy.float64)
    if float(xs.max() - xs.min()) < 1e-6:
        return None
    slope, intercept = numpy.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _cutout_min_area_rect_corners(
    cv2,
    edge_cutout,
    *,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    points = cv2.findNonZero(edge_cutout)
    if points is None or len(points) < 4:
        return None
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    corners = order_corners(tuple(ImagePoint(float(point[0]), float(point[1])) for point in box))
    top_left, top_right, bottom_right, bottom_left = corners
    if min(_distance(top_left, top_right), _distance(bottom_left, bottom_right)) < min_edge_height_px:
        return None
    if min(_distance(top_left, bottom_left), _distance(top_right, bottom_right)) < min_edge_height_px:
        return None
    return corners


def _outer_hull_corners(
    cv2,
    edge_cutout,
    *,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    points = cv2.findNonZero(edge_cutout)
    if points is None or len(points) < 4:
        return None
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0.0:
        return None
    for epsilon_fraction in (0.01, 0.015, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon_fraction * perimeter, True)
        if len(approx) == 4:
            corners = order_corners(
                tuple(ImagePoint(float(point[0][0]), float(point[0][1])) for point in approx)
            )
            top_left, top_right, bottom_right, bottom_left = corners
            if min(_distance(top_left, top_right), _distance(bottom_left, bottom_right)) < min_edge_height_px:
                continue
            if min(_distance(top_left, bottom_left), _distance(top_right, bottom_right)) < min_edge_height_px:
                continue
            return corners
    return None


def _outer_row_envelope_corners(
    left_line: tuple[float, float] | None,
    right_line: tuple[float, float] | None,
    *,
    top_y: float,
    bottom_y: float,
    min_edge_height_px: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    if left_line is None or right_line is None:
        return None
    left_at_top = left_line[0] * top_y + left_line[1]
    left_at_bottom = left_line[0] * bottom_y + left_line[1]
    right_at_top = right_line[0] * top_y + right_line[1]
    right_at_bottom = right_line[0] * bottom_y + right_line[1]
    if min(right_at_top - left_at_top, right_at_bottom - left_at_bottom) < min_edge_height_px:
        return None
    return order_corners(
        (
            ImagePoint(left_at_top, top_y),
            ImagePoint(right_at_top, top_y),
            ImagePoint(right_at_bottom, bottom_y),
            ImagePoint(left_at_bottom, bottom_y),
        )
    )


def _fit_x_line_at_ys(ys, xs) -> tuple[float, float] | None:
    import numpy

    if len(ys) < 2:
        return None
    if abs(float(ys[-1]) - float(ys[0])) < 1e-6:
        return 0.0, float(xs.mean())
    slope, intercept = numpy.polyfit(ys, xs, 1)
    return float(slope), float(intercept)


def _fit_top_bottom_y_lines(edge_cutout, *, min_edge_height_px: float):
    import numpy

    column_bounds = []
    max_span = 0.0
    for x_px in range(edge_cutout.shape[1]):
        rows = numpy.flatnonzero(edge_cutout[:, x_px])
        if len(rows) < 2:
            continue
        top = float(rows[0])
        bottom = float(rows[-1])
        span = bottom - top
        if span <= 0.0:
            continue
        column_bounds.append((float(x_px), top, bottom, span))
        max_span = max(max_span, span)
    if len(column_bounds) < 4 or max_span < min_edge_height_px:
        return None, None

    boundary_columns = [column for column in column_bounds if column[3] >= max(min_edge_height_px, 0.35 * max_span)]
    if len(boundary_columns) < 4:
        return None, None

    xs = numpy.array([column[0] for column in boundary_columns], dtype=numpy.float64)
    tops = numpy.array([column[1] for column in boundary_columns], dtype=numpy.float64)
    bottoms = numpy.array([column[2] for column in boundary_columns], dtype=numpy.float64)
    if abs(float(xs[-1]) - float(xs[0])) < 1e-6:
        return None, None
    top_slope, top_intercept = numpy.polyfit(xs, tops, 1)
    bottom_slope, bottom_intercept = numpy.polyfit(xs, bottoms, 1)
    return (float(top_slope), float(top_intercept)), (float(bottom_slope), float(bottom_intercept))


def _intersect_x_of_y_line_with_y_of_x_line(
    x_of_y_line: tuple[float, float],
    y_of_x_line: tuple[float, float],
) -> ImagePoint | None:
    x_slope, x_intercept = x_of_y_line
    y_slope, y_intercept = y_of_x_line
    denominator = 1.0 - x_slope * y_slope
    if abs(denominator) < 1e-6:
        return None
    x = (x_slope * y_intercept + x_intercept) / denominator
    y = y_slope * x + y_intercept
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return ImagePoint(float(x), float(y))


def _debug_contour_edge_cutout_image(
    cv2,
    edges,
    contour,
    *,
    x_offset: int,
    y_offset: int,
    roi_width: int,
    roi_height: int,
    stem_center_x: float | None = None,
    stem_top_y: float | None = None,
    min_edge_height_px: float = 8.0,
):
    import numpy

    cutout = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
    y_end = min(edges.shape[0], y_offset + roi_height)
    x_end = min(edges.shape[1], x_offset + roi_width)
    if y_offset < 0 or x_offset < 0 or y_end <= y_offset or x_end <= x_offset:
        return cutout

    roi_edges = edges[y_offset:y_end, x_offset:x_end]
    boundary = numpy.zeros(roi_edges.shape[:2], dtype=numpy.uint8)
    # The connected localization contour may sit several pixels outside the
    # pre-morphology Canny border.  Keep the band wide enough to bridge that
    # known morphology offset, while still far narrower than the head interior
    # so label/QR texture cannot become rectangle evidence.
    boundary_thickness = max(
        5,
        min(11, int(round(min_edge_height_px * 1.10))),
    )
    cv2.drawContours(boundary, [contour], -1, 255, thickness=boundary_thickness)
    selected = cv2.bitwise_and(roi_edges, boundary)

    if stem_center_x is not None and stem_top_y is not None:
        local_stem_x = int(round(stem_center_x - x_offset))
        local_stem_top = int(round(stem_top_y - y_offset))
        stem_half_width = max(4, int(round(min_edge_height_px * 0.85)))
        erase_y = max(0, local_stem_top - max(2, int(round(min_edge_height_px * 0.20))))
        erase_left = max(0, local_stem_x - stem_half_width)
        erase_right = min(selected.shape[1], local_stem_x + stem_half_width + 1)
        if erase_left < erase_right and erase_y < selected.shape[0]:
            selected[erase_y:, erase_left:erase_right] = 0

    cutout[y_offset:y_end, x_offset:x_end] = selected
    return cutout


def _stem_local_x_bounds(
    frame_width: int,
    *,
    stem_center_x: float,
    min_edge_height_px: float,
) -> tuple[int, int]:
    """Bound a stem hypothesis to its own stand in a full camera frame."""

    # At close range 6.75 minimum-edge units still cover a ~100 px head. In a
    # 640 px full frame the fractional floor gives roughly 54 px on each side,
    # enough for the target head but not the neighbouring smaller stand seen in
    # the standalone-viewer flicker captures.
    x_radius = max(
        35.0,
        6.75 * min_edge_height_px,
        0.085 * frame_width,
    )
    x_min = max(0, int(math.floor(stem_center_x - x_radius)))
    x_max = min(frame_width, int(math.ceil(stem_center_x + x_radius)) + 1)
    return x_min, x_max


def _plain_face_from_stem_cropped_edges(
    cv2,
    edges,
    *,
    measurement_edges=None,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    _raster_stem_anchor: tuple[float, float] | None = None,
    _fixed_parallel_side_direction: tuple[float, float] | None = None,
) -> _SilhouetteFaceCandidate | None:
    """Localize with connected edges and measure with independent edge pixels.

    ``edges`` may be dilated/closed so fragmented head and stem boundaries form
    usable topology. ``measurement_edges`` is never morphologically expanded;
    only its pixels may enter the returned cutout or support the rectangle.
    Keeping the inputs separate makes the silhouette path independent of stand
    hue while preventing morphology from inventing orientation evidence.
    """

    import numpy

    uses_independent_measurement = (
        measurement_edges is not None and measurement_edges is not edges
    )
    if measurement_edges is None:
        measurement_edges = edges
    if measurement_edges.shape[:2] != edges.shape[:2]:
        raise ValueError("measurement_edges must match localization edges")

    if _raster_stem_anchor is None:
        stem_anchors = _stem_anchor_candidates_from_edges(
            cv2,
            edges,
            min_edge_height_px=min_edge_height_px,
        )
        if not stem_anchors:
            return None

        diagnostic_candidate = None
        for stem_center_x, stem_top_y in stem_anchors:
            # Hough pairs commonly yield half-pixel centers. All following
            # evidence is rasterized, so carrying a subpixel center into
            # rounded ROI bounds made one seam column appear/disappear between
            # adjacent frames. Try both neighboring pixel anchors for each
            # ranked stem hypothesis. A candidate is returned only after the
            # untouched-Canny four-side gate accepts its rectangle.
            raster_centers = []
            for center in (
                round(stem_center_x),
                math.floor(stem_center_x),
                math.ceil(stem_center_x),
            ):
                center = float(center)
                if center not in raster_centers:
                    raster_centers.append(center)

            for raster_center_x in raster_centers:
                candidate = _plain_face_from_stem_cropped_edges(
                    cv2,
                    edges,
                    measurement_edges=measurement_edges,
                    min_area_px=min_area_px,
                    min_edge_height_px=min_edge_height_px,
                    min_aspect_ratio=min_aspect_ratio,
                    max_aspect_ratio=max_aspect_ratio,
                    _raster_stem_anchor=(raster_center_x, stem_top_y),
                    _fixed_parallel_side_direction=(
                        _fixed_parallel_side_direction
                    ),
                )
                if candidate is None:
                    continue
                if candidate.rectangle_fit_reliable:
                    return candidate
                if diagnostic_candidate is None:
                    diagnostic_candidate = candidate
        return diagnostic_candidate

    stem_center_x, stem_top_y = _raster_stem_anchor
    contour_face = _plain_face_from_stem_head_contour(
        cv2,
        edges,
        measurement_edges=measurement_edges,
        stem_center_x=stem_center_x,
        stem_top_y=stem_top_y,
        min_area_px=min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        fixed_parallel_side_direction=_fixed_parallel_side_direction,
    )
    if contour_face is not None and contour_face.rectangle_fit_reliable:
        return contour_face

    frame_height, frame_width = edges.shape[:2]
    search_height = max(35, int(round(0.60 * frame_height)))
    y_min = max(0, int(round(stem_top_y - search_height)))
    y_max = min(frame_height, int(round(stem_top_y + max(3.0, 0.5 * min_edge_height_px))))
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=stem_center_x,
        min_edge_height_px=min_edge_height_px,
    )
    if y_max <= y_min or x_max <= x_min:
        return contour_face

    roi = edges[y_min:y_max, x_min:x_max]
    row_bounds = []
    max_span = 0.0
    for local_y in range(roi.shape[0]):
        global_y = y_min + local_y
        columns = numpy.flatnonzero(roi[local_y, :])
        if len(columns) < 2:
            continue
        left = float(columns[0] + x_min)
        right = float(columns[-1] + x_min)
        span = right - left
        if span <= 0.0:
            continue
        # A sloped arena-wall edge can touch one corner of the head and make
        # the row span extend far to one side.  The stand stem must sit well
        # inside a genuine head row; reject one-sided background branches
        # before they establish the maximum face width.
        if not (
            left + 0.12 * span
            <= stem_center_x
            <= right - 0.12 * span
        ):
            continue
        row_bounds.append((float(global_y), left, right, span))
        max_span = max(max_span, span)

    if len(row_bounds) < 4:
        return contour_face

    robust_span = float(
        numpy.percentile(
            numpy.array([row[3] for row in row_bounds], dtype=numpy.float64),
            75.0,
        )
    )
    if robust_span < min_edge_height_px:
        return contour_face
    broad_rows = [
        row
        for row in row_bounds
        if row[3] >= max(min_edge_height_px, 0.55 * robust_span)
        and row[3] <= 1.50 * robust_span
    ]
    if len(broad_rows) < 4:
        return contour_face

    ys = numpy.array([row[0] for row in broad_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in broad_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in broad_rows], dtype=numpy.float64)
    top = float(ys.min())
    bottom = float(ys.max())
    if bottom - top < min_edge_height_px:
        return contour_face

    left_at_top, left_at_bottom = _fit_boundary_x_at_ys(ys, lefts, top, bottom)
    right_at_top, right_at_bottom = _fit_boundary_x_at_ys(ys, rights, top, bottom)
    width_top = right_at_top - left_at_top
    width_bottom = right_at_bottom - left_at_bottom
    avg_width = (width_top + width_bottom) / 2.0
    height = bottom - top
    if width_top < min_edge_height_px or width_bottom < min_edge_height_px or avg_width < min_edge_height_px:
        return contour_face
    left = min(left_at_top, left_at_bottom)
    right = max(right_at_top, right_at_bottom)
    width = right - left
    if stem_center_x < left + 0.18 * width or stem_center_x > right - 0.18 * width:
        return contour_face

    aspect_ratio = avg_width / max(height, 1e-6)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return contour_face
    corners = order_corners(
        (
            ImagePoint(left_at_top, top),
            ImagePoint(right_at_top, top),
            ImagePoint(right_at_bottom, bottom),
            ImagePoint(left_at_bottom, bottom),
        )
    )
    area = _polygon_area(corners)
    if area < min_area_px:
        return contour_face

    if uses_independent_measurement:
        # The closed/dilated edge image proposed this rectangle.  Measure each
        # of its four sides independently from immutable raw Canny pixels; no
        # connected face border is required, and internal QR/label edges are
        # prevented from becoming a global rectangle fit.
        face_mask, border_corners = _raw_side_evidence_and_corners(
            cv2,
            measurement_edges,
            corners,
            fixed_parallel_side_direction=_fixed_parallel_side_direction,
        )
    else:
        edge_cutout = _expanded_head_edge_roi(
            cv2,
            measurement_edges,
            corners,
            margin_px=max(10, int(round(min_edge_height_px * 1.8))),
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_edge_height_px=min_edge_height_px,
        )
        boundary_cutout = _debug_polygon_edge_cutout_image(
            cv2,
            measurement_edges,
            corners,
            thickness_px=max(
                5,
                min(11, int(round(1.10 * min_edge_height_px))),
            ),
        )
        edge_cutout = cv2.bitwise_and(edge_cutout, boundary_cutout)
        # Legacy same-domain callers still use the connected-border refit.
        head_gate = _edge_pixels_inside_polygon(
            cv2,
            measurement_edges,
            corners,
            margin_px=max(2, int(round(0.25 * min_edge_height_px))),
        )
        edge_cutout = cv2.bitwise_and(edge_cutout, head_gate)
        face_mask, border_corners = _connected_border_mask_and_corners(
            cv2,
            measurement_edges,
            edge_cutout,
            fallback_corners=corners,
            min_edge_height_px=min_edge_height_px,
        )
    selected_corners, fit_reason, _support = _select_supported_head_corners(
        cv2,
        face_mask,
        corners,
        border_corners,
        image_shape=measurement_edges.shape,
        stem_center_x=stem_center_x,
        stem_top_y=stem_top_y,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        allow_rough_fallback=not uses_independent_measurement,
    )
    row_face = _SilhouetteFaceCandidate(
        corners=selected_corners if selected_corners is not None else corners,
        face_mask=face_mask,
        rectangle_fit_reliable=selected_corners is not None,
        rectangle_fit_reason=fit_reason,
    )
    if row_face.rectangle_fit_reliable or contour_face is None:
        return row_face
    # Preserve the original cutout/reason when neither independent refit is
    # supported. This keeps fail-closed diagnostics stable while allowing the
    # robust row envelope to recover from a seam-corrupted contour.
    return contour_face


def _plain_face_from_stem_head_contour(
    cv2,
    edges,
    *,
    measurement_edges=None,
    stem_center_x: float,
    stem_top_y: float,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    fixed_parallel_side_direction: tuple[float, float] | None = None,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    uses_independent_measurement = (
        measurement_edges is not None and measurement_edges is not edges
    )
    if measurement_edges is None:
        measurement_edges = edges
    if measurement_edges.shape[:2] != edges.shape[:2]:
        raise ValueError("measurement_edges must match localization edges")

    frame_height, frame_width = edges.shape[:2]
    search_height = max(35, int(round(0.62 * frame_height)))
    y_min = max(0, int(round(stem_top_y - search_height)))
    y_max = min(frame_height, int(round(stem_top_y + max(4.0, 0.45 * min_edge_height_px))))
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=stem_center_x,
        min_edge_height_px=min_edge_height_px,
    )
    if y_max <= y_min or x_max <= x_min:
        return None

    roi = edges[y_min:y_max, x_min:x_max].copy()
    if cv2.countNonZero(roi) == 0:
        return None

    close_kernel_size = max(3, int(round(min_edge_height_px * 0.45)))
    if close_kernel_size % 2 == 0:
        close_kernel_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: _SilhouetteFaceCandidate | None = None
    best_rank = (-1, -1.0)
    for contour in contours:
        local_x, local_y, width, height = cv2.boundingRect(contour)
        if width < min_edge_height_px or height < min_edge_height_px:
            continue
        global_left = float(x_min + local_x)
        global_right = float(x_min + local_x + width)
        global_top = float(y_min + local_y)
        global_bottom = float(y_min + local_y + height)
        if not (global_left + 0.15 * width <= stem_center_x <= global_right - 0.15 * width):
            continue
        if global_bottom < stem_top_y - 0.25 * height:
            continue
        if global_top > stem_top_y - 0.35 * height:
            continue

        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area_px:
            continue
        corners = _quadrilateral_corners(cv2, contour)
        if corners is None:
            hull = cv2.convexHull(contour)
            corners = _quadrilateral_corners(cv2, hull)
        if corners is None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            corners = tuple(ImagePoint(float(point[0]), float(point[1])) for point in box)
        corners = order_corners(
            tuple(
                ImagePoint(point.u_px + x_min, point.v_px + y_min)
                for point in corners
            )
        )
        aspect_ratio = quadrilateral_aspect_ratio(corners)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        top_left, top_right, bottom_right, bottom_left = corners
        top_width = _distance(top_left, top_right)
        bottom_width = _distance(bottom_left, bottom_right)
        if (
            min(top_width, bottom_width)
            < 0.40 * max(top_width, bottom_width, 1e-6)
        ):
            # A wall branch joined to one head corner produces a trapezoid
            # hundreds of pixels wide on only one edge.  It is not a plausible
            # projected square face, even though its average aspect ratio can
            # accidentally fall inside the accepted range.
            continue
        area = _polygon_area(corners)
        if area < min_area_px:
            continue
        rough_area = area
        bottom_y = max(point.v_px for point in corners)
        top_y = min(point.v_px for point in corners)
        if bottom_y > stem_top_y + max(8.0, 0.12 * (bottom_y - top_y)):
            continue

        rough_corners = corners
        if uses_independent_measurement:
            face_mask, border_corners = _raw_side_evidence_and_corners(
                cv2,
                measurement_edges,
                rough_corners,
                fixed_parallel_side_direction=fixed_parallel_side_direction,
            )
        else:
            # Keep only original Canny edges near the selected head contour.
            # This connected-border path remains for same-domain callers; the
            # dual-edge path above deliberately does not depend on connectivity.
            edge_cutout = _debug_contour_edge_cutout_image(
                cv2,
                measurement_edges,
                contour,
                x_offset=x_min,
                y_offset=y_min,
                roi_width=roi.shape[1],
                roi_height=roi.shape[0],
                stem_center_x=stem_center_x,
                stem_top_y=stem_top_y,
                min_edge_height_px=min_edge_height_px,
            )
            head_gate = _edge_pixels_inside_polygon(
                cv2,
                measurement_edges,
                corners,
                margin_px=max(2, int(round(0.25 * min_edge_height_px))),
            )
            edge_cutout = cv2.bitwise_and(edge_cutout, head_gate)
            face_mask, border_corners = _connected_border_mask_and_corners(
                cv2,
                measurement_edges,
                edge_cutout,
                fallback_corners=rough_corners,
                min_edge_height_px=min_edge_height_px,
            )
        selected_corners, fit_reason, _support = _select_supported_head_corners(
            cv2,
            face_mask,
            rough_corners,
            border_corners,
            image_shape=measurement_edges.shape,
            stem_center_x=stem_center_x,
            stem_top_y=stem_top_y,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            allow_rough_fallback=not uses_independent_measurement,
        )
        candidate_corners = (
            selected_corners if selected_corners is not None else rough_corners
        )

        # Rank candidates by the robust pre-cutout quadrilateral.  Otherwise a
        # thin connected wall branch can enlarge the refitted border and beat
        # the actual stand head solely because its corrupted area is larger.
        score = rough_area * (
            1.0
            + max(
                0.0,
                1.0
                - abs(stem_center_x - (global_left + global_right) / 2.0)
                / max(width, 1),
            )
        )
        rank = (int(selected_corners is not None), score)
        if rank > best_rank:
            best = _SilhouetteFaceCandidate(
                corners=candidate_corners,
                face_mask=face_mask,
                rectangle_fit_reliable=selected_corners is not None,
                rectangle_fit_reason=fit_reason,
            )
            best_rank = rank
    return best


def _stem_anchored_face_from_edges(
    cv2,
    edges,
    *,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    stem = _stem_anchor_from_edges(cv2, edges, min_edge_height_px=min_edge_height_px)
    if stem is None:
        return None
    stem_center_x, stem_top_y = stem
    frame_height, frame_width = edges.shape[:2]
    search_height = max(35, int(round(0.65 * frame_height)))
    y_min = max(0, int(round(stem_top_y - search_height)))
    y_max = min(frame_height, int(round(stem_top_y + max(8.0, 0.08 * search_height))))
    x_radius = max(35, int(round(0.33 * frame_width)))
    x_min = max(0, int(round(stem_center_x - x_radius)))
    x_max = min(frame_width, int(round(stem_center_x + x_radius)))
    if y_max <= y_min or x_max <= x_min:
        return None

    roi = edges[y_min:y_max, x_min:x_max]
    row_bounds = []
    max_span = 0.0
    stem_top_limit = stem_top_y + max(4.0, min_edge_height_px)
    for local_y in range(roi.shape[0]):
        global_y = y_min + local_y
        if global_y > stem_top_limit:
            continue
        columns = numpy.flatnonzero(roi[local_y, :])
        if len(columns) < 2:
            continue
        left = float(columns[0] + x_min)
        right = float(columns[-1] + x_min)
        span = right - left
        if span <= 0.0:
            continue
        if not (
            left + 0.12 * span
            <= stem_center_x
            <= right - 0.12 * span
        ):
            continue
        row_bounds.append((float(global_y), left, right, span))
        max_span = max(max_span, span)

    if len(row_bounds) < 4:
        return None

    robust_span = float(
        numpy.percentile(
            numpy.array([row[3] for row in row_bounds], dtype=numpy.float64),
            75.0,
        )
    )
    if robust_span < min_edge_height_px:
        return None
    min_span = max(min_edge_height_px, 0.45 * robust_span)
    broad_rows = [
        row
        for row in row_bounds
        if min_span <= row[3] <= 1.50 * robust_span
    ]
    if len(broad_rows) < 4:
        return None

    ys = numpy.array([row[0] for row in broad_rows], dtype=numpy.float64)
    lefts = numpy.array([row[1] for row in broad_rows], dtype=numpy.float64)
    rights = numpy.array([row[2] for row in broad_rows], dtype=numpy.float64)
    top = float(ys.min())
    bottom = float(ys.max())
    if bottom - top < min_edge_height_px:
        return None

    left_at_top, left_at_bottom = _fit_boundary_x_at_ys(ys, lefts, top, bottom)
    right_at_top, right_at_bottom = _fit_boundary_x_at_ys(ys, rights, top, bottom)
    width_top = right_at_top - left_at_top
    width_bottom = right_at_bottom - left_at_bottom
    avg_width = (width_top + width_bottom) / 2.0
    height = bottom - top
    if width_top < min_edge_height_px or width_bottom < min_edge_height_px or avg_width < min_edge_height_px:
        return None
    left = min(left_at_top, left_at_bottom)
    right = max(right_at_top, right_at_bottom)
    width = right - left
    if stem_center_x < left + 0.20 * width or stem_center_x > right - 0.20 * width:
        return None

    aspect_ratio = avg_width / max(height, 1e-6)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return None
    corners = order_corners(
        (
            ImagePoint(left_at_top, top),
            ImagePoint(right_at_top, top),
            ImagePoint(right_at_bottom, bottom),
            ImagePoint(left_at_bottom, bottom),
        )
    )
    area = _polygon_area(corners)
    if area < min_area_px:
        return None

    return _SilhouetteFaceCandidate(
        corners=corners,
        face_mask=_debug_outline_image(cv2, edges.shape, corners),
    )


def _fit_boundary_x_at_ys(ys, xs, top_y: float, bottom_y: float) -> tuple[float, float]:
    import numpy

    if len(ys) < 2 or abs(float(ys[-1]) - float(ys[0])) < 1e-6:
        value = float(xs.mean()) if len(xs) else 0.0
        return value, value
    slope, intercept = numpy.polyfit(ys, xs, 1)
    return float(slope * top_y + intercept), float(slope * bottom_y + intercept)


def _stem_anchor_from_edges(cv2, edges, *, min_edge_height_px: float) -> tuple[float, float] | None:
    """Return the highest-ranked stem hypothesis for diagnostic callers."""

    candidates = _stem_anchor_candidates_from_edges(
        cv2,
        edges,
        min_edge_height_px=min_edge_height_px,
    )
    return candidates[0] if candidates else None


def _stem_anchor_candidates_from_edges(
    cv2,
    edges,
    *,
    min_edge_height_px: float,
) -> list[tuple[float, float]]:
    """Rank plausible stem anchors without treating Hough rank as truth.

    A wall seam can connect a head side to label texture and make that false
    vertical pair score slightly above the real, lower stem pair.  Consumers
    that derive orientation may therefore try these anchors in rank order and
    let independent rectangle edge support select the first valid head.
    """

    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=12,
        # The stem is narrower/shorter than a close-range head edge.  Requiring
        # three face-edge heights made the detector discard the actual stem
        # and then anchor on a long vertical side of the head instead.
        hough_min_line_length_px=max(8, int(round(min_edge_height_px * 1.5))),
        hough_max_line_gap_px=10,
    )
    frame_height, frame_width = edges.shape[:2]
    # The Gazebo full-frame stem occupies only about 10-13% of image height;
    # the earlier 15% minimum was appropriate for a tight projected ROI but
    # removed both real stem edges before pairing in a 640x480 frame.
    min_length = max(min_edge_height_px * 1.5, frame_height * 0.10)
    verticals = [
        segment
        for segment in segments
        if segment.length_px >= min_length
        and abs(abs(segment.angle_deg) - 90.0) <= 12.0
        and (
            segment.y_min >= 0.30 * frame_height
            # In a tight oblique ROI, morphology can merge a true stem edge
            # through head/label texture all the way to the upper head. Keep
            # that early-starting line only when it also reaches the bottom
            # portion where a real stand stem must continue; the sustained
            # transition gate below still has to validate the junction.
            or segment.y_max >= 0.80 * frame_height
        )
        # In a full 640x480 camera frame the visible stem terminates at the
        # base around 55-65% image height, while the head borders terminate
        # substantially higher.  The former 70% ROI-specific threshold
        # discarded the real full-frame Gazebo stem entirely.
        and segment.y_max >= 0.55 * frame_height
    ]
    if not verticals:
        return []

    scored_pairs = []
    for index, left in enumerate(verticals):
        for right in verticals[index + 1 :]:
            separation = abs(left.x_mid - right.x_mid)
            if separation < 3.0 or separation > 0.16 * frame_width:
                continue
            overlap = _overlap_length(left.y_min, left.y_max, right.y_min, right.y_max)
            if overlap < 0.35 * min(left.length_px, right.length_px):
                continue
            top_y = min(left.y_min, right.y_min)
            center_x = (left.x_mid + right.x_mid) / 2.0
            lower_reach = (left.y_max + right.y_max) / max(2.0 * frame_height, 1.0)
            score = (
                overlap
                + 0.25 * (left.length_px + right.length_px)
                + 2.0 * lower_reach
                + max(0.0, frame_height - top_y) / frame_height
            )
            scored_pairs.append((score, center_x, top_y))

    candidates = []

    def append_distinct(anchor: tuple[float, float]) -> None:
        center_x, top_y = anchor
        if any(
            abs(center_x - known_x) <= 1.0 and abs(top_y - known_y) <= 2.0
            for known_x, known_y in candidates
        ):
            return
        candidates.append(anchor)

    for _score, center_x, line_top_y in sorted(scored_pairs, reverse=True):
        append_distinct(
            _resolved_stem_anchor(
                edges,
                center_x=center_x,
                line_top_y=line_top_y,
                min_edge_height_px=min_edge_height_px,
            )
        )

    # Preserve the former single-line fallback, but keep the other plausible
    # lower-reaching lines available when a false pair happened to exist.
    for segment in sorted(
        verticals,
        key=lambda item: (
            item.y_max,
            item.length_px,
            -item.y_min,
        ),
        reverse=True,
    ):
        append_distinct(
            _resolved_stem_anchor(
                edges,
                center_x=segment.x_mid,
                line_top_y=segment.y_min,
                min_edge_height_px=min_edge_height_px,
            )
        )
    return candidates


def _resolved_stem_anchor(
    edges,
    *,
    center_x: float,
    line_top_y: float,
    min_edge_height_px: float,
) -> tuple[float, float]:
    """Resolve a line top to a sustained head-to-stem transition when valid."""

    frame_height = edges.shape[0]
    transition_y = _stem_top_from_row_width_transition(
        edges,
        center_x=center_x,
        min_edge_height_px=min_edge_height_px,
    )
    line_anchor_y = min(
        frame_height - 1.0,
        line_top_y + min(2.0, 0.25 * min_edge_height_px),
    )
    use_late_transition = bool(
        transition_y is not None
        and transition_y > line_top_y + max(6.0, 0.08 * frame_height)
        and _has_sustained_stem_below_transition(
            edges,
            center_x=center_x,
            transition_y=transition_y,
            min_edge_height_px=min_edge_height_px,
        )
    )
    # Thick localization edges start paired Hough lines slightly inside the
    # lower head border, so advance their top by at most two pixels. Only a
    # much later sustained narrow run may override that line-derived anchor.
    return center_x, transition_y if use_late_transition else line_anchor_y


def _stem_top_from_row_width_transition(
    edges,
    *,
    center_x: float,
    min_edge_height_px: float,
) -> float | None:
    """Locate the broad-head to narrow-stem transition below a stem line pair."""

    import numpy

    frame_height, frame_width = edges.shape[:2]
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=center_x,
        min_edge_height_px=min_edge_height_px,
    )
    local_width = max(1, x_max - x_min)
    # The head in the far-view flicker is only about 34 px wide. The old 80 px
    # full-frame broad threshold skipped it entirely. Scale against this stem's
    # local corridor and keep a clear margin over the expected narrow stem.
    broad_span = max(
        2.0 * min_edge_height_px,
        min(0.35 * local_width, 4.0 * min_edge_height_px),
    )
    narrow_span = max(
        1.25 * min_edge_height_px,
        min(0.18 * local_width, 2.5 * min_edge_height_px),
    )
    last_broad_y = None
    narrow_run_start = None
    narrow_run_length = 0
    for y_px in range(int(0.30 * frame_height), frame_height):
        columns = numpy.flatnonzero(edges[y_px, x_min:x_max])
        if len(columns) < 2:
            narrow_run_start = None
            narrow_run_length = 0
            continue
        left = float(columns[0] + x_min)
        right = float(columns[-1] + x_min)
        span = right - left
        if span >= broad_span and left <= center_x <= right:
            last_broad_y = y_px
            narrow_run_start = None
            narrow_run_length = 0
            continue
        if (
            last_broad_y is not None
            and span <= narrow_span
            and left - min_edge_height_px <= center_x <= right + min_edge_height_px
        ):
            if narrow_run_start is None:
                narrow_run_start = y_px
            narrow_run_length += 1
            if narrow_run_length >= 3:
                return float(narrow_run_start)
        else:
            narrow_run_start = None
            narrow_run_length = 0
    return None


def _has_sustained_stem_below_transition(
    edges,
    *,
    center_x: float,
    transition_y: float,
    min_edge_height_px: float,
) -> bool:
    """Confirm that a late width transition is a real lower stem.

    Hough segments on an oblique head can merge a head side, label edge, and
    stem into one long line whose reported top is far above the physical
    head/stem junction.  A genuine junction is still followed by a sustained
    narrow run around the same center.  A single arena edge crossing below the
    stand is not, so this check lets us trust the late transition without
    reviving the old floor/wall false anchor.
    """

    import numpy

    frame_height, frame_width = edges.shape[:2]
    start_y = max(0, int(math.floor(transition_y)))
    if start_y >= frame_height:
        return False
    x_min, x_max = _stem_local_x_bounds(
        frame_width,
        stem_center_x=center_x,
        min_edge_height_px=min_edge_height_px,
    )
    local_width = max(1, x_max - x_min)
    narrow_span = max(
        1.25 * min_edge_height_px,
        min(0.18 * local_width, 2.5 * min_edge_height_px),
    )
    required_run = max(
        4,
        int(math.ceil(min_edge_height_px)),
        int(math.ceil(0.08 * frame_height)),
    )
    max_gap = max(1, int(round(0.20 * min_edge_height_px)))
    supported_run = 0
    gap = 0
    search_end = min(
        frame_height,
        start_y + max(3 * required_run, int(math.ceil(0.40 * frame_height))),
    )
    for y_px in range(start_y, search_end):
        columns = numpy.flatnonzero(edges[y_px, x_min:x_max])
        supported = False
        if len(columns) >= 2:
            left = float(columns[0] + x_min)
            right = float(columns[-1] + x_min)
            span = right - left
            supported = (
                span <= narrow_span
                and left - min_edge_height_px <= center_x <= right + min_edge_height_px
            )
        if supported:
            supported_run += 1
            gap = 0
            if supported_run >= required_run:
                return True
            continue
        gap += 1
        if gap > max_gap:
            supported_run = 0
            gap = 0
    return False


def _face_quadrilateral_from_silhouette(
    cv2,
    edges,
    *,
    min_area_px: float,
    min_edge_height_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    face_width_fraction: float,
) -> _SilhouetteFaceCandidate | None:
    import numpy

    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_candidate = None
    best_score = -1.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < min_area_px:
            continue
        if width < min_edge_height_px or height < min_edge_height_px:
            continue

        component = numpy.zeros(edges.shape[:2], dtype=numpy.uint8)
        cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
        crop = component[y : y + height, x : x + width]
        row_widths = [int(cv2.countNonZero(crop[row_index, :])) for row_index in range(crop.shape[0])]
        band = wide_row_band(row_widths, width_fraction=face_width_fraction)
        if band is None:
            continue
        band_start, band_end = band
        if band_start > 0.30 * height:
            continue
        if band_end - band_start + 1 < min_edge_height_px:
            continue

        band_mask = crop[band_start : band_end + 1, :]
        band_mask, band_x_offset = _expand_band_mask_to_nearby_edges(
            cv2,
            edges,
            band_mask,
            x_offset=x,
            y_offset=y + band_start,
        )
        corners = _quadrilateral_from_mask_component(
            cv2,
            band_mask,
            x_offset=float(band_x_offset),
            y_offset=float(y + band_start),
        )
        if corners is None:
            continue
        aspect_ratio = quadrilateral_aspect_ratio(corners)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        if _polygon_area(corners) < min_area_px:
            continue
        stem_score = _lower_stem_support_score(cv2, crop, band_start, band_end)
        score = _polygon_area(corners) * (1.0 + 2.0 * stem_score)
        if score > best_score:
            best_candidate = _SilhouetteFaceCandidate(
                corners=corners,
                face_mask=_debug_outline_image(cv2, edges.shape, corners),
            )
            best_score = score
    return best_candidate


def _expand_band_mask_to_nearby_edges(cv2, edges, band_mask, *, x_offset: int, y_offset: int):
    import numpy

    height, width = band_mask.shape[:2]
    all_columns = numpy.flatnonzero(band_mask.max(axis=0))
    if len(all_columns) == 0:
        return band_mask, x_offset

    band_left = int(all_columns[0])
    band_right = int(all_columns[-1])
    band_width = max(1, band_right - band_left + 1)
    search_margin = max(8, int(round(0.35 * band_width)))
    global_left = max(0, x_offset + band_left - search_margin)
    global_right = min(edges.shape[1], x_offset + band_right + search_margin + 1)
    if global_right <= global_left:
        return band_mask, x_offset

    expanded = numpy.zeros((height, global_right - global_left), dtype=band_mask.dtype)
    for row_index in range(height):
        columns = numpy.flatnonzero(band_mask[row_index, :])
        if len(columns) == 0:
            continue
        left = x_offset + int(columns[0]) - global_left
        right = x_offset + int(columns[-1]) - global_left
        global_y = y_offset + row_index
        if global_y < 0 or global_y >= edges.shape[0]:
            continue
        support = numpy.flatnonzero(edges[global_y, global_left:global_right])
        if len(support) > 0:
            left = min(left, int(support[0]))
            right = max(right, int(support[-1]))
        left = max(0, left)
        right = min(expanded.shape[1] - 1, right)
        expanded[row_index, left : right + 1] = 255

    if cv2.countNonZero(expanded) == 0:
        return band_mask, x_offset
    return expanded, global_left


def _lower_stem_support_score(cv2, crop, band_start: int, band_end: int) -> float:
    import numpy

    face = crop[band_start : band_end + 1, :]
    face_columns = numpy.flatnonzero(face.max(axis=0))
    if len(face_columns) == 0:
        return 0.0

    face_width = max(1.0, float(face_columns[-1] - face_columns[0] + 1))
    face_center = (float(face_columns[0]) + float(face_columns[-1])) / 2.0
    face_height = max(1, band_end - band_start + 1)
    lower_start = band_end + 1
    lower_end = min(crop.shape[0], lower_start + max(8, int(round(face_height * 0.8))))
    if lower_start >= lower_end:
        return 0.0

    supported_rows = 0
    for row_index in range(lower_start, lower_end):
        row = crop[row_index, :]
        row_width = int(cv2.countNonZero(row))
        if row_width <= 0 or row_width > 0.45 * face_width:
            continue
        row_columns = numpy.flatnonzero(row)
        if len(row_columns) == 0:
            continue
        row_center = (float(row_columns[0]) + float(row_columns[-1])) / 2.0
        if abs(row_center - face_center) <= 0.28 * face_width:
            supported_rows += 1

    required_rows = max(4.0, face_height * 0.18)
    return min(1.0, supported_rows / required_rows)


def _quadrilateral_from_mask_component(
    cv2,
    mask,
    *,
    x_offset: float,
    y_offset: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    corners = _quadrilateral_corners(cv2, contour)
    if corners is None:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = tuple(ImagePoint(float(point[0]), float(point[1])) for point in box)

    return order_corners(
        tuple(
            ImagePoint(point.u_px + x_offset, point.v_px + y_offset)
            for point in corners
        )
    )


def _largest_external_bounding_area(cv2, edges) -> float:
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = 0.0
    for contour in contours:
        _x, _y, width, height = cv2.boundingRect(contour)
        largest = max(largest, float(width * height))
    return largest


def _edge_topology_hypotheses(
    cv2,
    topology_seed,
    *,
    close_kernel: int,
    close_iterations: int,
    include_gap_recovery: bool,
    edge_exclusion_mask=None,
):
    """Return bounded morphology variants used only to locate edge topology.

    The configured variant stays first. In silhouette mode, two conservative
    alternatives bridge the one- or two-pixel head-border gaps seen in Gazebo
    without ever changing the raw Canny evidence used for rectangle support.
    """

    specifications = [(close_kernel, close_iterations)]
    if include_gap_recovery:
        specifications.extend(((3, 2), (5, 1)))

    hypotheses = []
    seen = set()
    for kernel_size, iterations in specifications:
        specification = (
            (int(kernel_size), int(iterations))
            if kernel_size > 1 and iterations > 0
            else (1, 0)
        )
        if specification in seen:
            continue
        seen.add(specification)

        edges = topology_seed.copy()
        if specification[1] > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (specification[0], specification[0]),
            )
            edges = cv2.morphologyEx(
                edges,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=specification[1],
            )
        # Apply the synchronized wall mask after closing as well. Otherwise a
        # removed wall line can be painted back across the exclusion band by a
        # neighbouring foreground edge.
        if edge_exclusion_mask is not None:
            edges = cv2.bitwise_and(
                edges,
                cv2.bitwise_not(edge_exclusion_mask),
            )
        hypotheses.append(edges)
    return hypotheses


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

    if edge_preprocess == "channel_union":
        edge_frame = frame
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            edge_frame = cv2.GaussianBlur(
                edge_frame,
                (blur_kernel, blur_kernel),
                0,
            )
        channels = cv2.split(edge_frame)
        if not channels:
            raise ValueError("frame must contain at least one image channel")
        edges = cv2.Canny(channels[0], canny_low, canny_high)
        for channel in channels[1:]:
            edges = cv2.bitwise_or(
                edges,
                cv2.Canny(channel, canny_low, canny_high),
            )
        return edges

    edge_input = _edge_input_image(
        cv2,
        frame,
        edge_preprocess=edge_preprocess,
        blur_kernel=blur_kernel,
    )
    return cv2.Canny(edge_input, canny_low, canny_high)


def _topology_edges_from_frame(
    cv2,
    frame,
    *,
    edge_preprocess: str,
    canny_low: int,
    canny_high: int,
    fallback_edges,
):
    """Build a texture-suppressed locator without changing measurement edges."""

    if edge_preprocess != "channel_union":
        return fallback_edges.copy()

    # Apply the existing low-pass outer-border preparation to each channel
    # independently. The operation is symmetric under any B/G/R permutation,
    # but QR modules are attenuated before the small morphology hypotheses can
    # accidentally connect them to the head outline.
    channels = cv2.split(frame)
    if not channels:
        raise ValueError("frame must contain at least one image channel")
    edges = cv2.Canny(
        _outer_border_edge_input(cv2, channels[0]),
        canny_low,
        canny_high,
    )
    for channel in channels[1:]:
        edges = cv2.bitwise_or(
            edges,
            cv2.Canny(
                _outer_border_edge_input(cv2, channel),
                canny_low,
                canny_high,
            ),
        )
    return edges


def _edge_input_image(cv2, frame, *, edge_preprocess: str, blur_kernel: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if edge_preprocess == "outer_border":
        return _outer_border_edge_input(cv2, gray)
    if edge_preprocess == "gray":
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        return gray
    raise ValueError(f"unsupported edge preprocess mode: {edge_preprocess}")


def _outer_border_edge_input(cv2, gray):
    # Suppress QR-code texture before Canny. The square outline and stem are
    # low-frequency structure; QR modules are high-frequency interior texture.
    smoothed = cv2.GaussianBlur(gray, (9, 9), 0)
    smoothed = cv2.medianBlur(smoothed, 7)
    return cv2.bilateralFilter(smoothed, 9, 50, 50)


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


def estimate_edge_on_axis_from_line(
    start: ImagePoint,
    end: ImagePoint,
    *,
    min_edge_height_px: float = 8.0,
    source: str = "edge_on_line",
) -> StandAxisImageEstimate:
    top = start if start.v_px <= end.v_px else end
    bottom = end if top is start else start
    length_px = _distance(top, bottom)
    if length_px < min_edge_height_px:
        return _unusable("edge_on_line_too_short", source=source, axis_line=(top, bottom))
    return StandAxisImageEstimate(
        usable=True,
        reason="edge_on_approx_90_deg",
        mode="edge_on",
        corners=None,
        axis_line=(top, bottom),
        left_height_px=length_px,
        right_height_px=0.0,
        height_ratio=None,
        yaw_proxy=None,
        yaw_deg=None,
        closer_side="side_on",
        contour_area_px=0.0,
        source=source,
    )


def _quadrilateral_from_line_segments(
    cv2,
    edges,
    *,
    hough_threshold: int,
    hough_min_line_length_px: int,
    hough_max_line_gap_px: int,
    min_boundary_line_length_px: float,
    min_edge_height_px: float,
    min_area_px: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
    )
    if not segments:
        return None

    verticals = [
        segment
        for segment in segments
        if segment.length_px >= max(min_edge_height_px, min_boundary_line_length_px)
        and abs(abs(segment.angle_deg) - 90.0) <= 25.0
    ]
    horizontals = [
        segment
        for segment in segments
        if segment.length_px >= max(min_edge_height_px, min_boundary_line_length_px * 0.55)
        and abs(segment.angle_deg) <= 25.0
    ]

    best_corners = None
    best_score = -1.0
    for left in verticals:
        for right in verticals:
            if left.x_mid >= right.x_mid:
                continue
            width = right.x_mid - left.x_mid
            avg_height = (left.length_px + right.length_px) / 2.0
            if width < min_edge_height_px or avg_height < min_edge_height_px:
                continue
            if abs(left.length_px - right.length_px) > 0.55 * avg_height:
                continue
            if abs(left.y_min - right.y_min) > 0.45 * avg_height:
                continue
            if abs(left.y_max - right.y_max) > 0.45 * avg_height:
                continue

            corners = order_corners((left.top_point(), right.top_point(), right.bottom_point(), left.bottom_point()))
            aspect_ratio = quadrilateral_aspect_ratio(corners)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            area = _polygon_area(corners)
            if area < min_area_px:
                continue
            support = _horizontal_support_score(horizontals, corners)
            score = score_quadrilateral_candidate(corners, area) * (1.0 + 0.25 * support)
            if score > best_score:
                best_corners = corners
                best_score = score
    return best_corners


def _edge_on_from_line_segments(
    cv2,
    edges,
    *,
    hough_threshold: int,
    hough_min_line_length_px: int,
    hough_max_line_gap_px: int,
    min_boundary_line_length_px: float,
    min_edge_height_px: float,
) -> StandAxisImageEstimate | None:
    segments = _line_segments_from_edges(
        cv2,
        edges,
        hough_threshold=hough_threshold,
        hough_min_line_length_px=hough_min_line_length_px,
        hough_max_line_gap_px=hough_max_line_gap_px,
    )
    verticals = [
        segment
        for segment in segments
        if segment.length_px >= max(min_edge_height_px * 2.0, hough_min_line_length_px, min_boundary_line_length_px)
        and abs(abs(segment.angle_deg) - 90.0) <= 15.0
    ]
    if not verticals:
        return None
    frame_height = float(edges.shape[0])
    best = max(
        verticals,
        key=lambda segment: segment.length_px * (1.0 + max(0.0, (frame_height - segment.y_min) / frame_height)),
    )
    return estimate_edge_on_axis_from_line(
        best.top_point(),
        best.bottom_point(),
        min_edge_height_px=min_edge_height_px,
        source="edge_on_line",
    )


def _line_segments_from_edges(
    cv2,
    edges,
    *,
    hough_threshold: int,
    hough_min_line_length_px: int,
    hough_max_line_gap_px: int,
) -> tuple[_LineSegment, ...]:
    # Probabilistic Hough sampling otherwise depends on whatever OpenCV RNG
    # state an earlier frame/test left behind.  A fixed seed makes identical
    # edge masks yield identical stem/head candidates and prevents the live
    # diagnostic ROI from flickering between valid and unrelated rectangles.
    if hasattr(cv2, "setRNGSeed"):
        cv2.setRNGSeed(0)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length_px,
        maxLineGap=hough_max_line_gap_px,
    )
    if raw_lines is None:
        return ()
    return tuple(_line_segment_from_hough(line[0]) for line in raw_lines)


def _line_segment_from_hough(values) -> _LineSegment:
    x1, y1, x2, y2 = (float(value) for value in values)
    start = ImagePoint(x1, y1)
    end = ImagePoint(x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return _LineSegment(start, end, math.hypot(dx, dy), angle)


def _horizontal_support_score(horizontals: Sequence[_LineSegment], corners: Sequence[ImagePoint]) -> int:
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    x_min = min(top_left.u_px, bottom_left.u_px)
    x_max = max(top_right.u_px, bottom_right.u_px)
    width = max(1.0, x_max - x_min)
    top_y = (top_left.v_px + top_right.v_px) / 2.0
    bottom_y = (bottom_left.v_px + bottom_right.v_px) / 2.0
    tolerance = max(5.0, 0.18 * width)
    support = 0
    for target_y in (top_y, bottom_y):
        if any(
            abs(((line.start.v_px + line.end.v_px) / 2.0) - target_y) <= tolerance
            and _overlap_length(line.x_min, line.x_max, x_min, x_max) >= 0.35 * width
            for line in horizontals
        ):
            support += 1
    return support


def _overlap_length(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _contour_has_lower_appendage(cv2, contour, corners: Sequence[ImagePoint]) -> bool:
    _x, y, _w, h = cv2.boundingRect(contour)
    contour_bottom = y + h
    candidate_bottom = max(point.v_px for point in corners)
    candidate_height = max(point.v_px for point in corners) - min(point.v_px for point in corners)
    if candidate_height <= 0.0:
        return False
    return contour_bottom > candidate_bottom + 0.25 * candidate_height


def _polygon_area(corners: Sequence[ImagePoint]) -> float:
    ordered = order_corners(corners)
    area = 0.0
    for current, following in zip(ordered, ordered[1:] + ordered[:1]):
        area += current.u_px * following.v_px - following.u_px * current.v_px
    return abs(area) / 2.0


def _quadrilateral_corners(cv2, contour) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    perimeter = cv2.arcLength(contour, True)
    for epsilon_fraction in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
        if len(approx) == 4:
            points = [ImagePoint(float(point[0][0]), float(point[0][1])) for point in approx]
            return order_corners(points)
    return None


def order_corners(points: Sequence[ImagePoint]) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]:
    if len(points) != 4:
        raise ValueError("exactly four points are required")
    ordered_by_y = sorted(points, key=lambda point: (point.v_px, point.u_px))
    top = sorted(ordered_by_y[:2], key=lambda point: point.u_px)
    bottom = sorted(ordered_by_y[2:], key=lambda point: point.u_px)
    top_left, top_right = top
    bottom_left, bottom_right = bottom
    return top_left, top_right, bottom_right, bottom_left


def _yaw_deg_from_ratio(
    ratio: float,
    stand_width_m: float | None,
    stand_distance_m: float | None,
) -> float | None:
    if stand_width_m is None or stand_distance_m is None:
        return None
    if stand_width_m <= 0.0 or stand_distance_m <= 0.0:
        return None
    sin_yaw = (2.0 * stand_distance_m / stand_width_m) * ((ratio - 1.0) / (ratio + 1.0))
    if abs(sin_yaw) > 1.0:
        return None
    return math.degrees(math.asin(sin_yaw))


def _yaw_deg_from_square_pnp(
    cv2,
    corners: Sequence[ImagePoint],
    *,
    stand_width_m: float | None,
    camera_fx_px: float | None,
    camera_fy_px: float | None,
    camera_cx_px: float | None,
    camera_cy_px: float | None,
) -> float | None:
    if cv2 is None:
        return None
    if (
        stand_width_m is None
        or camera_fx_px is None
        or camera_fy_px is None
        or camera_cx_px is None
        or camera_cy_px is None
    ):
        return None
    if stand_width_m <= 0.0 or camera_fx_px <= 0.0 or camera_fy_px <= 0.0:
        return None
    try:
        import numpy
    except ImportError:
        return None

    half = stand_width_m / 2.0
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    camera_matrix = numpy.array(
        [
            [camera_fx_px, 0.0, camera_cx_px],
            [0.0, camera_fy_px, camera_cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=numpy.float64,
    )
    distortion = numpy.zeros((4, 1), dtype=numpy.float64)

    def yaw_from_rotation_vector(rvec) -> float | None:
        rotation, _jacobian = cv2.Rodrigues(rvec)
        normal = rotation @ numpy.array(
            [[0.0], [0.0], [1.0]],
            dtype=numpy.float64,
        )
        normal_x = float(normal[0, 0])
        normal_z = float(normal[2, 0])
        if not math.isfinite(normal_x) or not math.isfinite(normal_z):
            return None
        # OpenCV optical +x points to image-right.  The public stand-axis
        # convention used by yaw_proxy, the metric fallbacks, ROS map yaw, and
        # the viewpoint planner is positive image-left / counterclockwise.
        # Convert handedness here so every estimator branch exposes the same
        # signed quantity to both the real viewer and simulation observer.
        return -math.degrees(math.atan2(normal_x, abs(normal_z)))

    # IPPE_SQUARE is specialized for a four-point coplanar square and returns
    # both planar-pose solutions.  Its object/image point order is fixed; the
    # image order below is therefore BL, BR, TR, TL rather than our normal
    # display order.  Select the physically visible solution with the lowest
    # measured reprojection error.
    if (
        hasattr(cv2, "solvePnPGeneric")
        and hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
    ):
        square_object_points = numpy.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=numpy.float64,
        )
        square_image_points = numpy.array(
            [
                [bottom_left.u_px, bottom_left.v_px],
                [bottom_right.u_px, bottom_right.v_px],
                [top_right.u_px, top_right.v_px],
                [top_left.u_px, top_left.v_px],
            ],
            dtype=numpy.float64,
        )
        try:
            generic_result = cv2.solvePnPGeneric(
                square_object_points,
                square_image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except Exception:
            generic_result = ()
        if generic_result and bool(generic_result[0]):
            best_pose = None
            rvecs = generic_result[1]
            tvecs = generic_result[2]
            for rvec, tvec in zip(rvecs, tvecs):
                tvec = numpy.asarray(tvec, dtype=numpy.float64).reshape(3, 1)
                if not math.isfinite(float(tvec[2, 0])) or float(tvec[2, 0]) <= 0.0:
                    continue
                try:
                    projected, _jacobian = cv2.projectPoints(
                        square_object_points,
                        rvec,
                        tvec,
                        camera_matrix,
                        distortion,
                    )
                except Exception:
                    continue
                residual = projected.reshape(-1, 2) - square_image_points
                reprojection_rmse = math.sqrt(
                    float(numpy.mean(numpy.sum(residual * residual, axis=1)))
                )
                yaw_deg = yaw_from_rotation_vector(rvec)
                if yaw_deg is None or not math.isfinite(reprojection_rmse):
                    continue
                if best_pose is None or reprojection_rmse < best_pose[0]:
                    best_pose = (reprojection_rmse, yaw_deg)
            if best_pose is not None:
                return best_pose[1]

    # Compatibility fallback for OpenCV builds without IPPE_SQUARE.
    object_points = numpy.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=numpy.float64,
    )
    image_points = numpy.array(
        [
            [top_left.u_px, top_left.v_px],
            [top_right.u_px, top_right.v_px],
            [bottom_right.u_px, bottom_right.v_px],
            [bottom_left.u_px, bottom_left.v_px],
        ],
        dtype=numpy.float64,
    )
    try:
        ok, rvec, _tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        return None
    if not ok:
        return None
    return yaw_from_rotation_vector(rvec)


def _yaw_deg_from_projected_width(
    corners: Sequence[ImagePoint],
    stand_width_m: float | None,
    stand_distance_m: float | None,
    camera_fx_px: float | None,
) -> float | None:
    if stand_width_m is None or stand_distance_m is None or camera_fx_px is None:
        return None
    if stand_width_m <= 0.0 or stand_distance_m <= 0.0 or camera_fx_px <= 0.0:
        return None
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    top_width = _distance(top_left, top_right)
    bottom_width = _distance(bottom_left, bottom_right)
    observed_width_px = (top_width + bottom_width) / 2.0
    expected_front_width_px = camera_fx_px * stand_width_m / stand_distance_m
    if observed_width_px <= 0.0 or expected_front_width_px <= 0.0:
        return None
    cos_yaw = observed_width_px / expected_front_width_px
    if cos_yaw > 1.0:
        return None
    cos_yaw = max(0.0, cos_yaw)
    magnitude_deg = math.degrees(math.acos(cos_yaw))

    left_height = _distance(top_left, bottom_left)
    right_height = _distance(top_right, bottom_right)
    height_proxy = (left_height - right_height) / max(left_height + right_height, 1e-6)
    top_center_x = (top_left.u_px + top_right.u_px) / 2.0
    bottom_center_x = (bottom_left.u_px + bottom_right.u_px) / 2.0
    shear_proxy = (bottom_center_x - top_center_x) / max(observed_width_px, 1e-6)
    sign_cue = height_proxy if abs(height_proxy) >= 0.01 else shear_proxy
    sign = -1.0 if sign_cue < 0.0 else 1.0
    return sign * magnitude_deg


def _distance(first: ImagePoint, second: ImagePoint) -> float:
    return math.hypot(second.u_px - first.u_px, second.v_px - first.v_px)


def _points_to_cv2(corners: Sequence[ImagePoint]):
    import numpy

    return numpy.array([[[point.u_px, point.v_px]] for point in corners], dtype=numpy.float32)


def _unusable(
    reason: str,
    *,
    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None = None,
    axis_line: tuple[ImagePoint, ImagePoint] | None = None,
    contour_area_px: float = 0.0,
    source: str = "unknown",
) -> StandAxisImageEstimate:
    return StandAxisImageEstimate(
        usable=False,
        reason=reason,
        mode="unavailable",
        corners=corners,
        axis_line=axis_line,
        left_height_px=0.0,
        right_height_px=0.0,
        height_ratio=None,
        yaw_proxy=None,
        yaw_deg=None,
        closer_side=None,
        contour_area_px=contour_area_px,
        source=source,
    )
