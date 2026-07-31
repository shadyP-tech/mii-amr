from __future__ import annotations

import math
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _corners_inside_image,
    _debug_outline_image,
    _debug_polygon_edge_cutout_image,
    _debug_rectangle_image,
    _debug_rectangle_overlay_image,
    _distance,
    _largest_qr_quad,
    _points_to_cv2,
    _polygon_area,
    _quadrilateral_corners,
    _scale_quadrilateral_about_center,
    _unusable,
    _well_formed_quadrilateral,
    _yaw_deg_from_projected_width,
    _yaw_deg_from_ratio,
    _yaw_deg_from_square_pnp,
    estimate_edge_on_axis_from_line,
    estimate_stand_axis_from_corners,
    order_corners,
    quadrilateral_aspect_ratio,
    score_quadrilateral_candidate,
    wide_row_band,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
    _LineSegment,
    _QuadrilateralEdgeSupport,
    _SilhouetteFaceCandidate,
)
from scripts.aufgabe04.perception.stand_axis.head_candidates import (
    _head_first_face_from_edges,
)
from scripts.aufgabe04.perception.stand_axis.preprocessing import (
    _canny_edges_from_frame,
    _edge_input_image,
    _edge_topology_hypotheses,
    _largest_external_bounding_area,
    _outer_border_edge_input,
    _topology_supported_measurement_edges,
    _topology_edges_from_frame,
)
from scripts.aufgabe04.perception.stand_axis.raw_support import (
    _fit_raw_edge_side_in_band,
    _image_line_intersection,
    _level_camera_endpoint_perspective_consistent,
    _parallel_side_lengths_comparable,
    _parallel_side_run_endpoints,
    _quadrilateral_edge_support,
    _raw_side_evidence_and_corners,
    _select_supported_head_corners,
    _validated_refitted_head_corners,
)
from scripts.aufgabe04.perception.stand_axis.stem_candidates import (
    _attach_structure_evidence,
    _connected_border_mask_and_corners,
    _contour_has_lower_appendage,
    _cutout_min_area_rect_corners,
    _cutout_outer_border_line_corners,
    _debug_contour_edge_cutout_image,
    _edge_on_from_line_segments,
    _edge_pixels_inside_polygon,
    _expand_band_mask_to_nearby_edges,
    _expanded_head_edge_roi,
    _fit_boundary_x_at_ys,
    _fit_top_bottom_y_lines,
    _fit_x_line_at_ys,
    _fit_y_line_from_border_segments,
    _fit_y_line_from_extreme_column_points,
    _has_sustained_stem_below_transition,
    _horizontal_support_score,
    _intersect_x_of_y_line_with_y_of_x_line,
    _line_segment_from_hough,
    _line_segment_x_at_y,
    _line_segments_from_edges,
    _lower_stem_support_score,
    _outer_hull_corners,
    _outer_row_envelope_corners,
    _overlap_length,
    _plain_face_from_stem_head_contour,
    _quadrilateral_from_line_segments,
    _quadrilateral_from_mask_component,
    _resolved_stem_anchor,
    _stem_anchor_candidates_from_edges,
    _stem_anchor_from_edges,
    _stem_local_x_bounds,
    _stem_owned_head_candidate_score,
    _stem_owned_head_from_line_segments,
    _stem_top_from_row_width_transition,
)
from scripts.aufgabe04.perception.stand_axis.model_pipeline import (
    estimate_stand_axis_from_metric_model,
)


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
    parallel_side_direction: tuple[float, float] = (0.0, 1.0),
    silhouette_only: bool = False,
    structural_diagnostic: bool = False,
    edge_exclusion_mask=None,
    topology_edge_exclusion_mask=None,
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

    # A colour-adaptive foreground gate narrows the topology proposal image.
    # Keep ``raw_edges`` immutable for diagnostics, then derive a second,
    # pre-morphology measurement image below. Its pixels remain real Canny
    # evidence but must belong to the gated low-frequency topology corridor.
    effective_topology_exclusion_mask = edge_exclusion_mask
    if topology_edge_exclusion_mask is not None:
        if topology_edge_exclusion_mask.shape[:2] != raw_edges.shape[:2]:
            raise ValueError("topology_edge_exclusion_mask must match the processed frame size")
        topology_edges = cv2.bitwise_and(
            topology_edges,
            cv2.bitwise_not(topology_edge_exclusion_mask),
        )
        effective_topology_exclusion_mask = (
            topology_edge_exclusion_mask
            if effective_topology_exclusion_mask is None
            else cv2.bitwise_or(
                effective_topology_exclusion_mask,
                topology_edge_exclusion_mask,
            )
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
        edge_exclusion_mask=effective_topology_exclusion_mask,
    )
    edges = topology_hypotheses[0]

    adaptive_min_area_px = max(
        min_area_px,
        _largest_external_bounding_area(cv2, edges) * min_face_area_fraction,
    )

    if silhouette_only:
        # Match the real-camera silhouette pipeline: use connected topology to
        # locate the stem and propose a head quadrilateral, then independently
        # fit its four sides from pre-morphology Canny pixels. With a real
        # adaptive gate those pixels must also remain inside the gated
        # low-frequency topology corridor, preventing background/QR escape.
        # Simulation forbids QR geometry and synthetic outlines as orientation
        # sources and retains its legacy unrestricted raw measurement image.
        silhouette_face = None
        debug_edges = edges
        for localization_edges in topology_hypotheses:
            measurement_edges = raw_edges
            if topology_edge_exclusion_mask is not None:
                # Bind the final raw-pixel fit to the exact gated topology
                # hypothesis being displayed and evaluated in this pass.
                # Later gap-recovery hypotheses can widen the corridor only
                # locally; the foreground exclusion is reapplied after every
                # morphology operation above.
                measurement_edges = _topology_supported_measurement_edges(
                    cv2,
                    raw_edges,
                    localization_edges,
                    min_edge_height_px=min_edge_height_px,
                )
            hypothesis_min_area_px = max(
                min_area_px,
                _largest_external_bounding_area(cv2, localization_edges)
                * min_face_area_fraction,
            )
            candidate = _plain_face_from_stem_cropped_edges(
                cv2,
                localization_edges,
                measurement_edges=measurement_edges,
                head_first_proposal_edges=(
                    localization_edges
                    if topology_edge_exclusion_mask is not None
                    else None
                ),
                min_area_px=hypothesis_min_area_px,
                min_edge_height_px=min_edge_height_px,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                _fixed_parallel_side_direction=parallel_side_direction,
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
        raw_edges,
        width_ratio=front_face_to_qr_width_ratio,
        # A valid QR plane is already strong target-specific evidence. Do not
        # let an unrelated heater/radiator contour inflate its area threshold.
        min_area_px=min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
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
    plain_structure_supported = (
        plain_face is not None
        and plain_face.rectangle_fit_reliable
        and plain_face.structure_evidence is not None
        and plain_face.structure_evidence.tracking_supported
    )
    if qr_front_face is not None and not plain_structure_supported:
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
                rectangle_mask=_debug_rectangle_image(
                    cv2,
                    edges.shape,
                    qr_front_face.corners,
                ),
                rectangle_overlay=_debug_rectangle_overlay_image(
                    cv2,
                    edges.shape,
                    qr_front_face.corners,
                    qr_front_face.face_mask,
                ),
                raw_edges=raw_edges,
            ),
        )

    if plain_face is not None:
        structure = plain_face.structure_evidence
        if structural_diagnostic and (
            structure is None or not structure.tracking_supported
        ):
            reason = (
                "structure_evidence_unavailable"
                if structure is None
                else structure.reason
            )
            return (
                _unusable(
                    reason,
                    corners=plain_face.corners,
                    contour_area_px=_polygon_area(plain_face.corners),
                    source="edge_structure_diagnostic",
                ),
                StandAxisEdgeDebugArtifacts(
                    edges=edges,
                    face_mask=plain_face.face_mask,
                    raw_edges=raw_edges,
                    structure_evidence=structure,
                ),
            )
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
                    structure_evidence=structure,
                ),
            )
        estimate_source = "edge_raw_head_geometry"
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
                source=estimate_source,
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
                structure_evidence=structure,
            ),
        )

    if structural_diagnostic:
        # Generic contour and line fallbacks remain diagnostic proposal tools;
        # they cannot independently accept a target in structure-owned mode.
        return (
            _unusable(
                "structure_head_unavailable",
                source="edge_structure_diagnostic",
            ),
            StandAxisEdgeDebugArtifacts(edges=edges, raw_edges=raw_edges),
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
    if qr_corners is None or not _well_formed_quadrilateral(qr_corners):
        return None
    corners = _scale_quadrilateral_about_center(qr_corners, width_ratio)
    if not _well_formed_quadrilateral(corners):
        return None
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
    if not _quadrilateral_edge_support(cv2, edges, corners).accepted:
        return None

    return _SilhouetteFaceCandidate(
        corners=corners,
        face_mask=_debug_outline_image(cv2, edges.shape, corners),
    )


def _detect_qr_quad_corners(cv2, frame) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    detector = cv2.QRCodeDetector()
    try:
        ok, points = detector.detectMulti(frame)
    except Exception:
        ok, points = False, None
    candidates = _qr_quad_candidates(points) if ok else ()
    if candidates:
        return _largest_qr_quad(candidates)

    try:
        ok, points = detector.detect(frame)
    except Exception:
        ok, points = False, None
    candidates = _qr_quad_candidates(points) if ok else ()
    if candidates:
        return _largest_qr_quad(candidates)

    if not candidates:
        try:
            multi_result = detector.detectAndDecodeMulti(frame)
        except Exception:
            multi_result = ()
        points = multi_result[2] if len(multi_result) > 2 else None
        candidates = _qr_quad_candidates(points)
    return _largest_qr_quad(candidates)


def _qr_quad_candidates(
    points,
) -> tuple[tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint], ...]:
    if points is None:
        return ()
    try:
        quadrilaterals = points.reshape(-1, 4, 2)
    except Exception:
        return ()
    return tuple(
        order_corners(
            tuple(
                ImagePoint(float(point[0]), float(point[1]))
                for point in quadrilateral
            )
        )
        for quadrilateral in quadrilaterals
    )








def _plain_face_from_stem_cropped_edges(
    cv2,
    edges,
    *,
    measurement_edges=None,
    head_first_proposal_edges=None,
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
    if (
        head_first_proposal_edges is not None
        and head_first_proposal_edges.shape[:2] != edges.shape[:2]
    ):
        raise ValueError("head_first_proposal_edges must match localization edges")

    if _raster_stem_anchor is None:
        head_first = _head_first_face_from_edges(
            cv2,
            (
                measurement_edges
                if head_first_proposal_edges is None
                else head_first_proposal_edges
            ),
            measurement_edges=measurement_edges,
            min_edge_height_px=min_edge_height_px,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            fixed_parallel_side_direction=_fixed_parallel_side_direction,
            bounded_endpoint_recovery=(
                head_first_proposal_edges is not None
                and _fixed_parallel_side_direction is None
            ),
        )
        if head_first is not None:
            return head_first
        stem_anchors = _stem_anchor_candidates_from_edges(
            cv2,
            edges,
            min_edge_height_px=min_edge_height_px,
        )
        if not stem_anchors:
            return None

        diagnostic_candidate = None
        best_reliable_candidate = None
        best_reliable_rank = (-1, -math.inf)
        for stem_center_x, stem_top_y in stem_anchors:
            # Hough pairs commonly yield half-pixel centers. All following
            # evidence is rasterized, so carrying a subpixel center into
            # rounded ROI bounds made one seam column appear/disappear between
            # adjacent frames. Try both neighboring pixel anchors for each
            # ranked stem hypothesis. A candidate is returned only after
            # untouched Canny accepts all four sides or a current raw
            # head/stem/base structure owns the missing-bottom recovery.
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
                    head_first_proposal_edges=head_first_proposal_edges,
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
                candidate = _attach_structure_evidence(
                    cv2,
                    candidate,
                    measurement_edges=measurement_edges,
                    stem_center_x=raster_center_x,
                    stem_top_y=stem_top_y,
                    min_aspect_ratio=min_aspect_ratio,
                    max_aspect_ratio=max_aspect_ratio,
                )
                if candidate.rectangle_fit_reliable:
                    score = _stem_owned_head_candidate_score(
                        cv2,
                        candidate,
                        stem_center_x=raster_center_x,
                        stem_top_y=stem_top_y,
                    )
                    # Structure evidence is annotation only. Raw head
                    # geometry must decide which candidate is accepted.
                    rank = (1, score)
                    if rank > best_reliable_rank:
                        best_reliable_candidate = candidate
                        best_reliable_rank = rank
                    continue
                if diagnostic_candidate is None:
                    diagnostic_candidate = candidate
        return (
            best_reliable_candidate
            if best_reliable_candidate is not None
            else diagnostic_candidate
        )

    stem_center_x, stem_top_y = _raster_stem_anchor
    line_face = _stem_owned_head_from_line_segments(
        cv2,
        edges,
        measurement_edges=measurement_edges,
        stem_center_x=stem_center_x,
        stem_top_y=stem_top_y,
        min_area_px=min_area_px,
        min_edge_height_px=min_edge_height_px,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
    if line_face is not None:
        return line_face

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
