"""Shared automatic head acquisition for the viewer and camera exploration.

Callers supply the full rectified image and its corresponding intrinsics. A
previous verified pose is only a bounded search hint; the metric pipeline fits
every observation from current pixels. Candidate projection and QR identity
binding belong after acquisition and cannot select a different input border.
"""

from __future__ import annotations

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
    estimator=None,
):
    """Run the same full-image cold/tracked metric path in both consumers.

    No candidate center, expected scale or preselected rectangle is supplied.
    Optional QR observations decorate the current physical geometry with marker
    evidence; they do not seed its pose. Exact-image caches may avoid repeating
    work when decoding adds identity to this same image. ``estimator`` is an
    injection seam for consumer tests; production uses the shared metric fitter.
    """

    fit = estimate_stand_axis_from_metric_model if estimator is None else estimator
    return fit(
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
    )
