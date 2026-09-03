"""Fail-closed no-QR acquisition of a measured stand's backside axis.

This module is intentionally ROS-free.  It does not infer front/back from the
6 mm head depth: that geometry is not resolvable reliably in one camera frame.
Instead, a candidate-centred metric projection supplies the expected head
location and scale, while immutable current-frame Canny pixels must prove the
outer four-sided head and its centred neck.  In the absence of a detected QR
quadrilateral, that evidence may be labelled only ``backside_candidate``.
"""

from __future__ import annotations

from dataclasses import replace
import math

from scripts.aufgabe04.artifacts.backside_axis_observation import (
    BACKSIDE_AXIS_SAMPLE_SOURCE,
    BACKSIDE_VISIBLE_FACE,
    MAXIMUM_HEAD_CENTER_ERROR_RATIO,
    MAXIMUM_HEAD_SCALE_RATIO,
    MINIMUM_BACKSIDE_FACE_CONFIDENCE,
    MINIMUM_HEAD_SCALE_RATIO,
)
from scripts.aufgabe04.perception.stand_axis.geometry import (
    _debug_rectangle_image,
    _debug_rectangle_overlay_image,
    _distance,
    _polygon_area,
    _unusable,
    estimate_stand_axis_from_corners,
    order_corners,
    quadrilateral_aspect_ratio,
)
from scripts.aufgabe04.perception.stand_axis.head_candidates import (
    _head_first_face_from_edges,
    _short_centered_neck_support,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    StandModelProfile,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis.preprocessing import (
    _edge_topology_hypotheses,
    _topology_edges_from_frame,
    _topology_supported_measurement_edges,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    RectifiedCameraMatrix,
    estimate_planar_pose_ippe,
)
from scripts.aufgabe04.perception.stand_axis.raw_support import (
    _quadrilateral_edge_support,
)
from scripts.aufgabe04.perception.stand_axis.stem_candidates import (
    _stem_anchor_candidates_from_edges,
)


MODEL_BACKSIDE_AXIS_SOURCE = BACKSIDE_AXIS_SAMPLE_SOURCE
MODEL_BACKSIDE_VISIBLE_FACE = BACKSIDE_VISIBLE_FACE

# The expected centre comes from the map-associated candidate projection, not
# from an image search.  Allow up to 0.55 expected head heights of projection
# error so modest map/localization error does not suppress a real stand, while
# still preventing a neighbouring square or radiator cell from being claimed.
MAX_HEAD_CENTER_ERROR_RATIO = MAXIMUM_HEAD_CENTER_ERROR_RATIO
MIN_HEAD_SCALE_RATIO = MINIMUM_HEAD_SCALE_RATIO
MAX_HEAD_SCALE_RATIO = MAXIMUM_HEAD_SCALE_RATIO
MIN_NORMALIZED_PROJECTED_ASPECT = math.cos(math.radians(70.0))
MAX_NORMALIZED_PROJECTED_ASPECT = 1.35
MIN_BACKSIDE_FACE_CONFIDENCE = MINIMUM_BACKSIDE_FACE_CONFIDENCE


def _global_corners(
    corners,
    *,
    x_offset: int,
    y_offset: int,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]:
    return order_corners(
        tuple(
            ImagePoint(
                point.u_px + float(x_offset),
                point.v_px + float(y_offset),
            )
            for point in corners
        )
    )


def _mean_head_height(corners) -> float:
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    return (
        _distance(top_left, bottom_left)
        + _distance(top_right, bottom_right)
    ) / 2.0


def _target_crop(
    frame_shape,
    *,
    center_u_px: float,
    center_v_px: float,
    expected_height_px: float,
    horizontal_half_width_ratio: float = 1.25,
) -> tuple[int, int, int, int] | None:
    frame_height, frame_width = frame_shape[:2]
    if (
        not math.isfinite(float(horizontal_half_width_ratio))
        or float(horizontal_half_width_ratio) < 1.25
        or float(horizontal_half_width_ratio) > 2.25
    ):
        raise ValueError(
            "horizontal_half_width_ratio must be within [1.25, 2.25]"
        )
    x0 = max(
        0,
        int(
            math.floor(
                center_u_px
                - float(horizontal_half_width_ratio) * expected_height_px
            )
        ),
    )
    x1 = min(
        frame_width,
        int(
            math.ceil(
                center_u_px
                + float(horizontal_half_width_ratio) * expected_height_px
            )
        )
        + 1,
    )
    y0 = max(0, int(math.floor(center_v_px - 1.20 * expected_height_px)))
    # Retain enough image below the expected head for the independent paired
    # neck-rail gate used by the low-level head detector.
    y1 = min(
        frame_height,
        int(math.ceil(center_v_px + 1.60 * expected_height_px)) + 1,
    )
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    return x0, y0, x1, y1


def _gate_score(value: float, minimum: float, maximum: float) -> float:
    if value <= 1.0:
        return max(0.0, min(1.0, (value - minimum) / (1.0 - minimum)))
    return max(0.0, min(1.0, (maximum - value) / (maximum - 1.0)))


def _failure(
    reason: str,
    *,
    model_profile: StandModelProfile,
    raw_edges,
    topology_edges=None,
    face_mask=None,
    corners=None,
    support_mean: float | None = None,
    scale_ratio: float | None = None,
    center_error_ratio: float | None = None,
    pose_reprojection_rmse_px: float | None = None,
    pose_ambiguity_gap_px: float | None = None,
    pose_fit_source: str | None = None,
) -> tuple[StandAxisImageEstimate, StandAxisEdgeDebugArtifacts]:
    estimate = replace(
        _unusable(
            reason,
            corners=corners,
            contour_area_px=(0.0 if corners is None else _polygon_area(corners)),
            source=MODEL_BACKSIDE_AXIS_SOURCE,
        ),
        evidence_state="unobservable",
        model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        pose_reprojection_rmse_px=pose_reprojection_rmse_px,
        pose_ambiguity_gap_px=pose_ambiguity_gap_px,
    )
    return estimate, StandAxisEdgeDebugArtifacts(
        edges=raw_edges if topology_edges is None else topology_edges,
        raw_edges=raw_edges,
        face_mask=face_mask,
        evidence_state="unobservable",
        model_profile_sha256=model_profile.sha256,
        pose_reprojection_rmse_px=pose_reprojection_rmse_px,
        pose_ambiguity_gap_px=pose_ambiguity_gap_px,
        refinement_support_mean=support_mean,
        model_pose_fit_source=pose_fit_source,
        pose_seed_source="none",
        model_reason=reason,
        model_measurement_status=model_profile.measurement_status,
        visible_face_reason=reason,
        head_scale_ratio=scale_ratio,
        head_center_error_ratio=center_error_ratio,
    )


def estimate_stand_axis_from_model_backside(
    cv2,
    frame,
    *,
    raw_edges,
    model_profile: StandModelProfile,
    expected_head_center_u_px: float,
    expected_head_center_v_px: float,
    expected_head_height_px: float,
    camera_fx_px: float,
    camera_fy_px: float,
    camera_cx_px: float,
    camera_cy_px: float,
    edge_preprocess: str,
    canny_low: int,
    canny_high: int,
    min_edge_height_px: float,
    max_reprojection_rmse_px: float,
    target_crop_horizontal_half_width_ratio: float = 1.25,
) -> tuple[StandAxisImageEstimate, StandAxisEdgeDebugArtifacts]:
    """Bootstrap an undirected stand axis from a QR-free current frame.

    The caller must already have established that neither a QR pose nor a
    tracked metric pose exists.  Acceptance requires a measured physical model,
    finite target projection, model-consistent scale/aspect and target centre,
    four independently supported raw-Canny head sides, a centred two-rail neck,
    and a finite PnP-derived axial yaw.
    """

    expected_values = (
        expected_head_center_u_px,
        expected_head_center_v_px,
        expected_head_height_px,
    )
    if (
        not model_profile.committable
        or model_profile.environment != "physical"
        or not all(math.isfinite(float(value)) for value in expected_values)
        or float(expected_head_height_px) <= 0.0
    ):
        return _failure(
            "model_backside_expected_geometry_invalid",
            model_profile=model_profile,
            raw_edges=raw_edges,
        )
    center_u = float(expected_head_center_u_px)
    center_v = float(expected_head_center_v_px)
    expected_height = float(expected_head_height_px)
    frame_height, frame_width = frame.shape[:2]
    if not (0.0 <= center_u < frame_width and 0.0 <= center_v < frame_height):
        return _failure(
            "model_backside_expected_center_outside_image",
            model_profile=model_profile,
            raw_edges=raw_edges,
        )
    crop = _target_crop(
        frame.shape,
        center_u_px=center_u,
        center_v_px=center_v,
        expected_height_px=expected_height,
        horizontal_half_width_ratio=target_crop_horizontal_half_width_ratio,
    )
    if crop is None:
        return _failure(
            "model_backside_target_crop_unavailable",
            model_profile=model_profile,
            raw_edges=raw_edges,
        )
    x0, y0, x1, y1 = crop
    crop_frame = frame[y0:y1, x0:x1]
    crop_raw_edges = raw_edges[y0:y1, x0:x1]
    topology_seed = _topology_edges_from_frame(
        cv2,
        crop_frame,
        edge_preprocess=edge_preprocess,
        canny_low=canny_low,
        canny_high=canny_high,
        fallback_edges=crop_raw_edges,
    )
    topology_hypotheses = _edge_topology_hypotheses(
        cv2,
        topology_seed,
        close_kernel=3,
        close_iterations=1,
        include_gap_recovery=True,
    )
    model_aspect = (
        float(camera_fx_px)
        / float(camera_fy_px)
        * model_profile.head_width_m
        / model_profile.head_height_m
    )
    if not math.isfinite(model_aspect) or model_aspect <= 0.0:
        return _failure(
            "model_backside_projected_aspect_invalid",
            model_profile=model_profile,
            raw_edges=raw_edges,
        )

    candidate_records = []
    for topology in topology_hypotheses:
        measurement = _topology_supported_measurement_edges(
            cv2,
            crop_raw_edges,
            topology,
            min_edge_height_px=min_edge_height_px,
        )
        candidate = _head_first_face_from_edges(
            cv2,
            topology,
            measurement_edges=measurement,
            min_edge_height_px=min_edge_height_px,
            min_aspect_ratio=(
                MIN_NORMALIZED_PROJECTED_ASPECT * model_aspect
            ),
            max_aspect_ratio=(
                MAX_NORMALIZED_PROJECTED_ASPECT * model_aspect
            ),
            fixed_parallel_side_direction=None,
            bounded_endpoint_recovery=True,
        )
        if candidate is None or not candidate.rectangle_fit_reliable:
            continue
        local_corners = order_corners(candidate.corners)
        support = _quadrilateral_edge_support(
            cv2,
            candidate.face_mask,
            local_corners,
        )
        short_neck_supported = _short_centered_neck_support(
            measurement,
            local_corners,
        )
        top_left, top_right, bottom_right, bottom_left = local_corners
        head_width = (
            _distance(top_left, top_right)
            + _distance(bottom_left, bottom_right)
        ) / 2.0
        head_height = _mean_head_height(local_corners)
        head_center_u = sum(point.u_px for point in local_corners) / 4.0
        head_bottom_v = (
            bottom_left.v_px + bottom_right.v_px
        ) / 2.0
        stem_anchor_supported = any(
            abs(stem_u - head_center_u) <= 0.20 * head_width
            and abs(stem_v - head_bottom_v) <= 0.25 * head_height
            for stem_u, stem_v in _stem_anchor_candidates_from_edges(
                cv2,
                topology,
                min_edge_height_px=min_edge_height_px,
            )
        )
        neck_supported = short_neck_supported and stem_anchor_supported
        global_corners = _global_corners(
            local_corners,
            x_offset=x0,
            y_offset=y0,
        )
        observed_height = _mean_head_height(global_corners)
        scale_ratio = observed_height / expected_height
        observed_center_u = sum(point.u_px for point in global_corners) / 4.0
        observed_center_v = sum(point.v_px for point in global_corners) / 4.0
        center_error_ratio = math.hypot(
            observed_center_u - center_u,
            observed_center_v - center_v,
        ) / expected_height
        normalized_aspect = (
            quadrilateral_aspect_ratio(global_corners) / model_aspect
        )
        rank = (
            int(support.accepted and neck_supported),
            -center_error_ratio,
            -abs(math.log(max(scale_ratio, 1.0e-9))),
            support.mean,
        )
        candidate_records.append(
            (
                rank,
                topology,
                measurement,
                candidate,
                global_corners,
                support,
                neck_supported,
                scale_ratio,
                center_error_ratio,
                normalized_aspect,
            )
        )

    if not candidate_records:
        return _failure(
            "model_backside_head_and_neck_unavailable",
            model_profile=model_profile,
            raw_edges=raw_edges,
        )
    (
        _rank,
        selected_topology,
        _selected_measurement,
        selected_candidate,
        corners,
        support,
        neck_supported,
        scale_ratio,
        center_error_ratio,
        normalized_aspect,
    ) = max(candidate_records, key=lambda record: record[0])

    import numpy

    full_topology = numpy.zeros(raw_edges.shape[:2], dtype=numpy.uint8)
    full_topology[y0:y1, x0:x1] = selected_topology
    full_face_mask = numpy.zeros(raw_edges.shape[:2], dtype=numpy.uint8)
    full_face_mask[y0:y1, x0:x1] = selected_candidate.face_mask

    failure_reason = None
    if not support.accepted:
        failure_reason = "model_backside_raw_four_side_support_insufficient"
    elif not neck_supported:
        failure_reason = "model_backside_neck_support_insufficient"
    elif not MIN_HEAD_SCALE_RATIO <= scale_ratio <= MAX_HEAD_SCALE_RATIO:
        failure_reason = "model_backside_head_scale_mismatch"
    elif center_error_ratio > MAX_HEAD_CENTER_ERROR_RATIO:
        failure_reason = "model_backside_target_center_mismatch"
    elif not (
        MIN_NORMALIZED_PROJECTED_ASPECT
        <= normalized_aspect
        <= MAX_NORMALIZED_PROJECTED_ASPECT
    ):
        failure_reason = "model_backside_projected_aspect_mismatch"
    if failure_reason is not None:
        return _failure(
            failure_reason,
            model_profile=model_profile,
            raw_edges=raw_edges,
            topology_edges=full_topology,
            face_mask=full_face_mask,
            corners=corners,
            support_mean=support.mean,
            scale_ratio=scale_ratio,
            center_error_ratio=center_error_ratio,
        )

    planar_pose = estimate_planar_pose_ippe(
        cv2,
        corners,
        tuple(model_profile.head_corners),
        RectifiedCameraMatrix(
            float(camera_fx_px),
            float(camera_fy_px),
            float(camera_cx_px),
            float(camera_cy_px),
        ),
        max_reprojection_rmse_px=max_reprojection_rmse_px,
    )
    planar_axis_ambiguous = planar_pose.axis_ambiguous()
    selected_pose = planar_pose.best
    if not planar_pose.accepted or selected_pose is None:
        rejected_rmse = (
            None
            if not planar_pose.hypotheses
            else planar_pose.hypotheses[0].reprojection_rmse_px
        )
        return _failure(
            "model_backside_planar_pose_unavailable",
            model_profile=model_profile,
            raw_edges=raw_edges,
            topology_edges=full_topology,
            face_mask=full_face_mask,
            corners=corners,
            support_mean=support.mean,
            scale_ratio=scale_ratio,
            center_error_ratio=center_error_ratio,
            pose_reprojection_rmse_px=rejected_rmse,
            pose_ambiguity_gap_px=planar_pose.ambiguity_gap_px,
            pose_fit_source="head_only_backside_rejected",
        )
    if planar_axis_ambiguous:
        return _failure(
            "model_backside_planar_pose_axis_ambiguous",
            model_profile=model_profile,
            raw_edges=raw_edges,
            topology_edges=full_topology,
            face_mask=full_face_mask,
            corners=corners,
            support_mean=support.mean,
            scale_ratio=scale_ratio,
            center_error_ratio=center_error_ratio,
            pose_reprojection_rmse_px=selected_pose.reprojection_rmse_px,
            pose_ambiguity_gap_px=planar_pose.ambiguity_gap_px,
            pose_fit_source="head_only_backside_ambiguous",
        )

    estimate = estimate_stand_axis_from_corners(
        corners,
        min_edge_height_px=min_edge_height_px,
        stand_width_m=model_profile.head_width_m,
        contour_area_px=_polygon_area(corners),
        source=MODEL_BACKSIDE_AXIS_SOURCE,
    )
    if not math.isfinite(selected_pose.yaw_deg):
        return _failure(
            "model_backside_axial_yaw_unavailable",
            model_profile=model_profile,
            raw_edges=raw_edges,
            topology_edges=full_topology,
            face_mask=full_face_mask,
            corners=corners,
            support_mean=support.mean,
            scale_ratio=scale_ratio,
            center_error_ratio=center_error_ratio,
            pose_reprojection_rmse_px=selected_pose.reprojection_rmse_px,
            pose_ambiguity_gap_px=planar_pose.ambiguity_gap_px,
            pose_fit_source="head_only_backside_yaw_unavailable",
        )

    scale_score = _gate_score(
        scale_ratio,
        MIN_HEAD_SCALE_RATIO,
        MAX_HEAD_SCALE_RATIO,
    )
    center_score = max(
        0.0,
        1.0 - center_error_ratio / MAX_HEAD_CENTER_ERROR_RATIO,
    )
    aspect_score = _gate_score(
        normalized_aspect,
        MIN_NORMALIZED_PROJECTED_ASPECT,
        MAX_NORMALIZED_PROJECTED_ASPECT,
    )
    face_confidence = max(
        0.0,
        min(
            1.0,
            0.50 * support.mean
            + 0.20 * scale_score
            + 0.20 * center_score
            + 0.10 * aspect_score,
        ),
    )
    if face_confidence < MIN_BACKSIDE_FACE_CONFIDENCE:
        return _failure(
            "model_backside_face_confidence_insufficient",
            model_profile=model_profile,
            raw_edges=raw_edges,
            topology_edges=full_topology,
            face_mask=full_face_mask,
            corners=corners,
            support_mean=support.mean,
            scale_ratio=scale_ratio,
            center_error_ratio=center_error_ratio,
            pose_reprojection_rmse_px=selected_pose.reprojection_rmse_px,
            pose_ambiguity_gap_px=planar_pose.ambiguity_gap_px,
            pose_fit_source="head_only_backside_confidence_rejected",
        )

    success_reason = "qr_absent_model_head_and_neck_supported"
    estimate = replace(
        estimate,
        reason="axis_estimated_model_backside_current_frame",
        yaw_deg=selected_pose.yaw_deg,
        # PnP supplies only the undirected axial angle here.  Do not publish a
        # directed camera face normal from a QR-free symmetric plane.
        camera_face_normal_xyz=None,
        camera_face_center_xyz_m=None,
        evidence_state="fresh_backside",
        model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        pose_reprojection_rmse_px=selected_pose.reprojection_rmse_px,
        pose_ambiguity_gap_px=planar_pose.ambiguity_gap_px,
        visible_face=MODEL_BACKSIDE_VISIBLE_FACE,
        visible_face_confidence=face_confidence,
    )
    return estimate, StandAxisEdgeDebugArtifacts(
        edges=full_topology,
        raw_edges=raw_edges,
        face_mask=full_face_mask,
        rectangle_mask=_debug_rectangle_image(cv2, raw_edges.shape, corners),
        rectangle_overlay=_debug_rectangle_overlay_image(
            cv2,
            raw_edges.shape,
            corners,
            full_face_mask,
        ),
        evidence_state="fresh_backside",
        model_profile_sha256=model_profile.sha256,
        pose_reprojection_rmse_px=selected_pose.reprojection_rmse_px,
        pose_ambiguity_gap_px=planar_pose.ambiguity_gap_px,
        refinement_support_mean=support.mean,
        model_pose_fit_source="undirected_head_axis_only",
        qr_detected=False,
        pose_seed_source="backside_geometry",
        model_reason=estimate.reason,
        model_measurement_status=model_profile.measurement_status,
        visible_face_reason=success_reason,
        head_scale_ratio=scale_ratio,
        head_center_error_ratio=center_error_ratio,
    )
