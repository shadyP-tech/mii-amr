"""Shared automatic head acquisition for the viewer and camera exploration.

Callers supply the full rectified image and its corresponding intrinsics. A
previous verified pose is only a bounded search hint; the metric pipeline fits
every observation from current pixels. An optional candidate screen excludes
incompatible cold-search hints without changing the image or supplying corners.
Current candidate association and QR identity binding remain separate gates.
"""

from __future__ import annotations

from dataclasses import replace

from scripts.aufgabe04.perception.stand_axis.current_image_head_fit import CurrentImageHeadFit
from scripts.aufgabe04.perception.stand_axis.model_input_cache import MetricModelInputCache, RoiBounds
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.pose_tracking import MetricPoseTracker
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


DEFAULT_MIN_EDGE_HEIGHT_PX = 8.0


def create_head_geometry_tracker() -> MetricPoseTracker:
    """Use the viewer's freshness, hint lifetime and brief-miss policy."""

    return MetricPoseTracker(
        prediction_ttl_sec=0.25, search_hint_ttl_sec=2.0, max_soft_misses=2,
    )


def estimate_current_head_geometry(
    cv2,
    frame,
    *,
    model_profile: StandModelProfile,
    camera_fx_px: float,
    camera_fy_px: float,
    camera_cx_px: float,
    camera_cy_px: float,
    pose_hint: PlanarPoseHypothesis | None = None,
    edge_preprocess: str = "channel_union",
    blur_kernel: int = 5,
    canny_low: int = 20,
    canny_high: int = 60,
    min_edge_height_px: float = DEFAULT_MIN_EDGE_HEIGHT_PX,
    deadline_monotonic_sec: float | None = None,
    qr_marker_policy: str = "disabled",
    qr_observations: tuple[DecodedQrObservation, ...] | None = None,
    input_cache: MetricModelInputCache | None = None,
    input_cache_roi: RoiBounds | None = None,
    current_image_head_fit: CurrentImageHeadFit | None = None,
    candidate_search=None,
    proposal_filter=None,
    source_support=None,
    edge_exclusion_mask=None,
    color_support_mask=None,
    use_color_prior=True,
    estimator=None,
):
    """Run the same full-image cold/tracked metric path in both consumers.

    The optional candidate screen rejects incompatible cold-search locations;
    metric pixel bounds also check final cold and tracked measurements.
    it supplies no rectangle and does not crop or alter the input pixels.
    A proposal filter may preview rough locators conservatively and associate
    completed current borders before selection. It supplies no border pixels.
    Optional QR observations decorate the current physical geometry with marker
    evidence; they do not seed its pose. Exact-image caches may avoid repeating
    work when decoding adds identity to this same image. A supplied proposal
    filter disables geometry reuse because scan/clock context can change.
    An optional uint8 edge exclusion removes Canny evidence before acquisition
    and fitting, while leaving source pixels available for QR decoding.
    Colour support is separate: it only ranks existing coherent rails, and
    leaves all raw edge pixels intact. Metric searches use the shared stand
    palette by default; callers can override the mask or disable this hint.
    ``estimator`` is an injection seam for consumer tests; production uses the
    shared metric fitter.
    """

    fit = estimate_stand_axis_from_metric_model if estimator is None else estimator
    if source_support is not None and source_support.shape != frame.shape[:2]:
        raise ValueError("source support must match the exact processing image")
    edge_region = getattr(candidate_search, "edge_region", None)
    if edge_region is not None and edge_region.shape != frame.shape[:2]:
        raise ValueError("LiDAR head region must match the exact processing image")
    if (use_color_prior and color_support_mask is None
            and getattr(candidate_search, "pixel_size", None) is not None):
        import numpy as np
        from scripts.aufgabe04.perception.stand_color_support import color_edge_support
        color_support_mask = color_edge_support(cv2, np, frame)
    if not use_color_prior:
        color_support_mask = None
    result = fit(
        cv2,
        frame,
        model_profile=model_profile,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
        pose_hint=pose_hint,
        edge_preprocess=edge_preprocess,
        blur_kernel=blur_kernel,
        canny_low=canny_low,
        canny_high=canny_high,
        min_edge_height_px=min_edge_height_px,
        deadline_monotonic_sec=deadline_monotonic_sec,
        qr_marker_policy=qr_marker_policy,
        qr_observations=qr_observations,
        input_cache=input_cache,
        input_cache_roi=input_cache_roi,
        current_image_head_fit=current_image_head_fit,
        candidate_search=candidate_search,
        proposal_filter=(proposal_filter if source_support is None else
                         source_support.filter(proposal_filter)),
        **({"edge_exclusion_mask": edge_exclusion_mask}
           if edge_exclusion_mask is not None else {}),
        **({"color_support_mask": color_support_mask}
           if color_support_mask is not None else {}),
    )
    metric_search = candidate_search if getattr(candidate_search, "pixel_size", None) is not None else None
    if source_support is None and edge_region is None and metric_search is None:
        return result
    estimate, debug = result
    diagnostics = dict(debug.head_acquisition_diagnostics or {})
    if source_support is not None:
        diagnostics["source_support"] = source_support.diagnostics()
    if edge_region is not None:
        diagnostics["lidar_edge_region"] = edge_region.diagnostics()
    if metric_search is not None:
        diagnostics["candidate_screen"] = metric_search.diagnostics()
    debug = replace(debug, head_acquisition_diagnostics=diagnostics)
    # Tracked fits bypass cold proposal callbacks, and raw refinement can move
    # a border. Recheck final pixels before any geometry can leave this facade.
    outside_source = (source_support is not None and estimate.corners is not None
                      and not source_support.accepts(estimate.corners))
    outside_candidate = (edge_region is not None and estimate.corners is not None
                         and not edge_region.contains(estimate.corners))
    outside_metric = (metric_search is not None and estimate.corners is not None
                      and not metric_search.accepts_measurement(estimate.corners))
    if outside_source or outside_candidate or outside_metric:
        from scripts.aufgabe04.perception.stand_axis.geometry import _unusable
        reason = ("head_border_outside_source_image" if outside_source
                  else "head_border_outside_lidar_candidate_region" if outside_candidate
                  else "head_border_outside_metric_pixel_bounds")
        estimate = replace(_unusable(reason, source=estimate.source),
            evidence_state="unobservable", model_profile_sha256=estimate.model_profile_sha256,
            model_measurement_status=estimate.model_measurement_status)
        debug = replace(debug, model_pose=None, head_model_quality=None,
            head_orientation_bounds=None, head_outer_recovery=None,
            refined_corners=None, rectangle_mask=None, projected_landmarks=None,
            model_reason=reason, evidence_state="unobservable")
    return estimate, debug
