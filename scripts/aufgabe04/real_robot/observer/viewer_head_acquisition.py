"""Viewer-equivalent geometry followed by bounded, head-local marker work.

The original image and intrinsics reach the shared fitter unchanged. QR work
uses a separate crop and cannot select, refit or rescale the physical head.
Candidate association and receipt freshness remain the observer's responsibility.
"""

from dataclasses import replace
import math
import time

from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
from scripts.aufgabe04.perception.stand_axis.metric_head_search import metric_head_search
from scripts.aufgabe04.perception.stand_axis.head_frame_detection import head_frame_detection
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import classify_current_head_backside
from scripts.aufgabe04.perception.stand_axis.marker_work_schedule import (
    MIN_NATIVE_MARKER_BUDGET_SEC, current_head_available_for_markers,
)
from scripts.aufgabe04.perception.stand_axis.qr_marker_validation import validate_qr_marker
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix, detect_qr_quad
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi, validate_intrinsics
from scripts.aufgabe04.real_robot.observer.camera_target_registration import HeadRoiEvaluation
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import merge_current_qr_observations


VIEWER_HEAD_SOURCE = "viewer_full_frame_head_search"


def _finite_or_zero(value):
    return float(value) if value is not None and math.isfinite(value) else 0.


def _marker_bounds(frame, estimate, complete_head, fallback_attempt):
    height, width = frame.shape[:2]
    if complete_head:
        u, v = zip(*((p.u_px, p.v_px) for p in estimate.corners))
        margin = max(8., .15 * max(max(u)-min(u), max(v)-min(v)))
        bounds = (max(0, math.floor(min(u)-margin)), max(0, math.floor(min(v)-margin)),
                  min(width, math.ceil(max(u)+margin)+1), min(height, math.ceil(max(v)+margin)+1))
        return bounds, "current_complete_head"
    if fallback_attempt is None:
        return None, "no_bounded_identity_crop"
    roi = fallback_attempt.roi
    bounds = (max(0, roi.x0), max(0, roi.y0), min(width, roi.x1), min(height, roi.y1))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None, "no_bounded_identity_crop"
    return bounds, fallback_attempt.source


def _local_observations(observations, crop):
    if observations is None:
        return None
    validated = []
    for observation in observations:
        # Validate in the decoder's actual input before translation. A backend's
        # complete input rectangle must never become apparently valid QR corners.
        corners = validated_qr_corners(observation.corners, image_shape=crop.shape)
        validated.append(replace(observation, corners=corners))
    return tuple(validated)


def _full_image_observations(observations, bounds):
    return None if observations is None else tuple(replace(
        item, corners=None if item.corners is None else tuple(
            (u + bounds[0], v + bounds[1]) for u, v in item.corners)) for item in observations)


def _acquire_identity(cv2, frame, *, bounds, scope, complete_head, cache, budget,
        native_decoder, full_decoder, now):
    observations, qr_detected, marker_verified, detection_scale = None, None, None, None
    marker_reason = "head_unavailable_marker_unchecked" if not complete_head else "qr_marker_processing_budget_exhausted"
    metadata = dict(performed=False, identity_scope=scope,
        identity_roi=None if bounds is None else list(bounds), marker_refresh_performed=False)
    identity_started = now()
    if bounds is None:
        metadata["reason"] = scope
    elif budget.remaining_work_sec(now()) < MIN_NATIVE_MARKER_BUDGET_SEC:
        metadata["reason"] = "identity_deferred_for_source_freshness"
    else:
        x0, y0, x1, y1 = bounds
        crop = frame[y0:y1, x0:x1]
        native = cache.decode(roi=bounds, mode="native", frame=crop, decoder=native_decoder)
        observations = _local_observations(native.observations, crop)
        metadata.update(performed=True, native_check=native.metadata())
        if not observations and complete_head and budget.remaining_work_sec(now()) >= MIN_NATIVE_MARKER_BUDGET_SEC:
            marker_started = now()
            detection = detect_qr_quad(cv2, crop, scales=(1.,),
                allow_decode_fallback=False, allow_native_decode_fallback=False)
            marker = validate_qr_marker(cv2, crop, detection)
            metadata.update(marker_refresh_performed=True,
                native_marker_elapsed_ms=(now()-marker_started)*1000.)
            detection_scale = None if detection is None else detection.scale
            if budget.remaining_work_sec(now()) > 0.:
                qr_detected, marker_verified, marker_reason = detection is not None, marker.verified, marker.reason
            else:
                marker_reason = "qr_marker_completion_deadline_exceeded"
        decision = budget.request(roi=bounds, roi_source=scope, now_monotonic_sec=now(),
            current_qr_signal=bool(observations) or qr_detected is True,
            identity_geometry_available=bool(observations) and (
                len(observations) > 1 or observations[0].corners is not None),
            complete_head_available=complete_head, selected_crop=True)
        metadata["acquisition"] = decision.metadata()
        if decision.allowed:
            if decision.cache_only and budget._full_result is not None:
                acquired = replace(budget._full_result, cache_hit=True, elapsed_ms=0.)
            else:
                provenance = {}
                acquired = cache.decode(roi=bounds, mode="full", frame=crop,
                    decoder=lambda image: full_decoder(image, decision.max_elapsed_sec, provenance),
                    decoder_provenance=provenance)
                budget._full_result = acquired
            observations = merge_current_qr_observations(observations, _local_observations(acquired.observations, crop))
            metadata["full_decode"] = acquired.metadata()
        if observations:
            qr_detected, marker_verified = True, True
            marker_reason = "multiple_decoded_qr_identities" if len(observations) > 1 else "decoded_qr_identity"
        observations = _full_image_observations(observations, bounds)
    identity_ms = (now()-identity_started)*1000.
    metadata["elapsed_ms"] = identity_ms
    return observations, qr_detected, marker_verified, detection_scale, marker_reason, metadata


def evaluate_viewer_head(
    cv2, frame, *, model_profile, intrinsics, pose_hint, projection,
    expected_head_height_px, fallback_attempt, cache, budget, native_decoder,
    full_decoder, deadline_monotonic_sec, edge_preprocess="channel_union",
    canny_low=20, canny_high=60, estimator=None, now=None,
    max_center_offset_ratio=1.5,
    proposal_filter=None,
    source_support=None,
    lidar_edge_region=None,
    lidar_edge_region_diagnostics=None,
    depth_uncertainty_m=.02,
    position_uncertainty_m=None,
    camera_vertical=(0., 1., 0.),
    previous_head_miss=False, identity_search_attempt=None, search_decoder=None,
):
    """Measure full-frame geometry once, returning neutral side classification.

    A decoded payload supplies positive marker evidence. Empty decoding supplies
    none: only a completed current native finder check on a complete head may
    provide marker absence. Skipped or late checks retain unknown side.
    """
    now = time.monotonic if now is None else now
    validate_intrinsics(intrinsics)
    if frame.shape[:2] != (intrinsics.height_px, intrinsics.width_px):
        raise ValueError("viewer head frame must match its full-image intrinsics")
    started = now()
    candidate_search = CandidateHeadSearch.optional(
        getattr(projection, "u_px", None), getattr(projection, "v_px", None),
        expected_head_height_px, max_center_offset_ratio=max_center_offset_ratio)
    depth = getattr(projection, "depth_m", None)
    if depth is None or not math.isfinite(depth) or depth <= 0.:
        candidate_search = None
    if candidate_search is not None and lidar_edge_region is not None:
        candidate_search = replace(candidate_search, edge_region=lidar_edge_region)
    if candidate_search is not None:
        try:
            candidate_search = metric_head_search(model_profile=model_profile, depth_m=depth,
                fx=intrinsics.fx_px, fy=intrinsics.fy_px, cx=intrinsics.cx_px, cy=intrinsics.cy_px,
                image_shape=frame.shape, center=candidate_search.center,
                depth_uncertainty_m=depth_uncertainty_m, camera_vertical=camera_vertical,
                position_uncertainty_m=position_uncertainty_m,
                max_center_offset_ratio=max_center_offset_ratio, edge_region=lidar_edge_region)
        except ValueError:
            candidate_search = None
    # One periodic image services identity first after failed head fitting.
    # No previous pixels, payloads or head corners cross this scheduling boundary.
    identity_fallback = identity_search_attempt or fallback_attempt
    early_identity = None
    if budget.identity_first_due(now_monotonic_sec=now(), previous_head_miss=previous_head_miss):
        early_bounds, early_scope = _marker_bounds(frame, None, False, identity_fallback)
        early_identity = _acquire_identity(cv2, frame, bounds=early_bounds, scope=early_scope,
            complete_head=False, cache=cache, budget=budget, native_decoder=native_decoder,
            full_decoder=search_decoder if identity_search_attempt is not None and search_decoder else full_decoder,
            now=now)
    geometry_started = now()
    estimate, debug = estimate_current_head_geometry(
        cv2, frame, model_profile=model_profile,
        camera_fx_px=intrinsics.fx_px, camera_fy_px=intrinsics.fy_px,
        camera_cx_px=intrinsics.cx_px, camera_cy_px=intrinsics.cy_px,
        pose_hint=pose_hint, edge_preprocess=edge_preprocess,
        canny_low=canny_low, canny_high=canny_high,
        deadline_monotonic_sec=deadline_monotonic_sec, estimator=estimator,
        candidate_search=candidate_search,
        proposal_filter=proposal_filter,
        source_support=source_support,
    )
    geometry_completed = now()
    geometry_ms = (geometry_completed-geometry_started)*1000.
    height = _finite_or_zero(expected_head_height_px)
    attempt = HeadRoiAttempt(
        ImageRoi(0, 0, intrinsics.width_px, intrinsics.height_px, height),
        VIEWER_HEAD_SOURCE, 1., _finite_or_zero(getattr(projection, "u_px", None)),
        _finite_or_zero(getattr(projection, "v_px", None)), height,
    )
    complete_head = current_head_available_for_markers((estimate, debug, None))
    bounds, scope = _marker_bounds(frame, estimate, complete_head, identity_fallback)
    if early_identity is None:
        identity = _acquire_identity(cv2, frame, bounds=bounds, scope=scope,
            complete_head=complete_head, cache=cache, budget=budget,
            native_decoder=native_decoder,
            full_decoder=(search_decoder if scope == "current_unique_scan_qr_search" and search_decoder else full_decoder),
            now=now)
    else:
        identity = early_identity
    observations, qr_detected, marker_verified, detection_scale, marker_reason, identity_metadata = identity
    identity_ms = identity_metadata["elapsed_ms"]
    metadata = dict(geometry_first=early_identity is None, geometry_scope="full_image",
        head_frame_detection=head_frame_detection(estimate, debug),
        candidate_screen=None if candidate_search is None else candidate_search.diagnostics(),
        current_scan_proposal_filter_applied=proposal_filter is not None,
        lidar_edge_region=lidar_edge_region_diagnostics,
        geometry_completed_monotonic_sec=geometry_completed,
        current_image_geometry_refit=False, **identity_metadata)
    timings = {**(debug.stage_timings_ms or {}), "initial_geometry_pass_ms": geometry_ms,
               "qr_identity": identity_ms, "total": (now()-started)*1000.}
    debug = replace(debug, qr_detected=qr_detected, qr_marker_verified=marker_verified,
        qr_marker_reason=marker_reason, qr_detection_scale=detection_scale, stage_timings_ms=timings)
    return HeadRoiEvaluation(attempt, frame, estimate, debug, observations, metadata)


def classify_viewer_head(evaluation, *, model_profile, intrinsics, expected_head_height_px):
    """Attach current side evidence without changing the measured angle.

    Its current complete head defines the classification center; the original
    projected height remains a scale check. This is not candidate association:
    the observer must still associate these original full-image corners with
    the intended map candidate and current scan before creating any receipt.
    """
    corners = evaluation.estimate.corners
    if corners is None or len(corners) != 4:
        return evaluation
    center = tuple(sum(getattr(p, name) for p in corners)/len(corners) for name in ("u_px", "v_px"))
    debug = replace(evaluation.debug, head_acquisition_diagnostics={
        **(evaluation.debug.head_acquisition_diagnostics or {}),
        "side_projection_source": "current_verified_head_pixels",
        "side_projection_center_px": center, "original_candidate_association_required": True,
    })
    estimate, debug = classify_current_head_backside(
        evaluation.estimate, debug, model_profile=model_profile,
        camera=RectifiedCameraMatrix(intrinsics.fx_px, intrinsics.fy_px, intrinsics.cx_px, intrinsics.cy_px),
        expected_center_u_px=center[0], expected_center_v_px=center[1],
        expected_height_px=expected_head_height_px,
    )
    return replace(evaluation, estimate=estimate, debug=debug)


__all__ = ["VIEWER_HEAD_SOURCE", "evaluate_viewer_head", "classify_viewer_head"]
