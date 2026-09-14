"""Schedule QR acquisition without starving current head geometry.

Every evaluated crop still receives a current native QR check and the metric
pipeline's current marker/finder validation. Expensive payload recovery is
limited to one crop per image. Positive current QR evidence takes priority;
otherwise a periodic probe alternates nominal and wider acquisition crops.
Only scheduling state crosses frames, never pixels, identity or geometry.
"""

from dataclasses import dataclass, replace
import math


QR_REACQUISITION_INTERVAL_SEC = 1.0
MAX_FULL_QR_WORK_SEC = 0.12
MIN_FULL_QR_WORK_SEC = 0.04
QR_PUBLICATION_RESERVE_SEC = 0.05


@dataclass(frozen=True)
class QrAcquisitionDecision:
    allowed: bool
    reason: str
    max_elapsed_sec: float
    cache_only: bool = False

    def metadata(self):
        return dict(allowed=self.allowed, reason=self.reason,
                    max_elapsed_sec=self.max_elapsed_sec, cache_only=self.cache_only)


class QrAcquisitionPolicy:
    """Create image budgets using bounded, target-local scheduling state."""

    def __init__(self):
        self._target_key = None
        self._epoch_stamp = None
        self._last_stamp = None
        self._last_probe_bucket = None

    def begin_frame(self, *, target_key, image_stamp_sec, started_ros_sec,
                    started_monotonic_sec, max_sensor_age_sec):
        values = (image_stamp_sec, started_ros_sec, started_monotonic_sec, max_sensor_age_sec)
        if (not isinstance(target_key, str) or not target_key
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value < 0 for value in values) or max_sensor_age_sec <= 0):
            raise ValueError("QR acquisition requires a target and finite image timing")
        if (self._target_key != target_key or self._last_stamp is None
                or image_stamp_sec < self._last_stamp):
            self._target_key, self._epoch_stamp = target_key, image_stamp_sec
            self._last_probe_bucket = None
        self._last_stamp = image_stamp_sec
        bucket = int((image_stamp_sec - self._epoch_stamp) / QR_REACQUISITION_INTERVAL_SEC)
        remaining = max(0., max_sensor_age_sec - max(0., started_ros_sec - image_stamp_sec))
        return QrFrameAcquisitionBudget(
            self, bucket=bucket, deadline_monotonic_sec=started_monotonic_sec + remaining,
        )


class QrFrameAcquisitionBudget:
    """One cooperative full-decode allowance; backend calls remain atomic."""

    def __init__(self, policy, *, bucket, deadline_monotonic_sec):
        self._policy = policy
        self._bucket = bucket
        self._deadline = deadline_monotonic_sec
        self._full_roi = None
        self._full_result = None  # Exact crop of this image; never held by the cross-frame policy.
        self._decisions = []

    def request(self, *, roi, roi_source, now_monotonic_sec,
                current_qr_signal, identity_geometry_available):
        if (type(now_monotonic_sec) not in (int, float)
                or not math.isfinite(now_monotonic_sec)):
            raise ValueError("QR acquisition clock must be finite")
        decision = None
        if self._full_roi == roi:
            decision = QrAcquisitionDecision(True, "same_image_exact_crop_cache", 0., True)
        elif identity_geometry_available:
            decision = QrAcquisitionDecision(False, "native_identity_geometry_available", 0.)
        elif self._full_roi is not None:
            decision = QrAcquisitionDecision(False, "one_full_crop_per_image", 0.)
        elif not current_qr_signal and (
            self._policy._last_probe_bucket == self._bucket
            or (roi_source == "nominal_projection") != (self._bucket % 2 == 0)
        ):
            decision = QrAcquisitionDecision(False, "periodic_empty_search_deferred", 0.)
        else:
            available = min(MAX_FULL_QR_WORK_SEC,
                            self._deadline - now_monotonic_sec - QR_PUBLICATION_RESERVE_SEC)
            if available < MIN_FULL_QR_WORK_SEC:
                decision = QrAcquisitionDecision(False, "image_processing_budget_exhausted", 0.)
            else:
                self._full_roi = roi
                self._policy._last_probe_bucket = self._bucket
                decision = QrAcquisitionDecision(
                    True, "current_qr_identity_recovery" if current_qr_signal
                    else "periodic_current_image_search", available,
                )
        self._decisions.append({"roi": list(roi), "roi_source": roi_source,
                                **decision.metadata()})
        return decision

    def metadata(self):
        return {
            "policy": "native_each_crop_bounded_full_reacquisition_v1",
            "reacquisition_interval_sec": QR_REACQUISITION_INTERVAL_SEC,
            "maximum_full_crops_per_image": 1,
            "maximum_full_work_sec": MAX_FULL_QR_WORK_SEC,
            "publication_reserve_sec": QR_PUBLICATION_RESERVE_SEC,
            "periodic_roi_scope": "nominal" if self._bucket % 2 == 0 else "expanded",
            "deadline_monotonic_sec": self._deadline,
            "native_current_image_checks_required": True,
            "cached_measurement_reuse": False,
            "cooperative_backend_budget": True,
            "decisions": list(self._decisions),
        }


def merge_current_qr_observations(native, acquired):
    """Keep current-frame conflicts and prefer a symbol's own recovered quad."""
    if not acquired:
        return native
    if not native:
        return acquired
    if len(native) == len(acquired) == 1 and native[0].text == acquired[0].text:
        return acquired if acquired[0].corners is not None else native
    # Different payloads or symbol multiplicity must survive acquisition.
    if len(native) > 1:
        return native + tuple(item for item in acquired
                              if item.text not in {old.text for old in native})
    return acquired + tuple(item for item in native
                            if item.text not in {old.text for old in acquired})


def evaluate_roi_with_qr_acquisition(
    *, frame, roi, roi_source, cache, budget, native_decoder, full_decoder,
    estimate, now,
):
    """Keep current geometry fast, then recover this frame's own QR if needed.

    Decoder callbacks and the metric estimator are injected. The estimator is
    rerun on these same pixels only when acquisition adds new QR evidence;
    geometry and QR identity never migrate between images or crop coordinates.
    """
    native = cache.decode(roi=roi, mode="native", frame=frame, decoder=native_decoder)
    native_observations = native.observations
    started = now()
    result, debug = estimate(native_observations)
    first_fit_ms = (now() - started) * 1000.
    decision = budget.request(
        roi=roi, roi_source=roi_source, now_monotonic_sec=now(),
        current_qr_signal=bool(native_observations) or debug.qr_detected or debug.qr_marker_verified,
        identity_geometry_available=bool(native_observations) and (
            len(native_observations) > 1 or native_observations[0].corners is not None
        ),
    )
    observations = native_observations
    metadata = native.metadata()
    qr_elapsed_ms = native.elapsed_ms
    repeated_fit = False
    if decision.allowed:
        provenance = {}
        if decision.cache_only and budget._full_result is not None:
            # The shared three-entry cache can be full of native crops. Keep
            # this one allowed full result in its image budget even then.
            acquired = replace(budget._full_result, cache_hit=True, elapsed_ms=0.)
        else:
            acquired = cache.decode(
                roi=roi, mode="full", frame=frame,
                decoder=lambda crop: full_decoder(crop, decision.max_elapsed_sec, provenance),
                decoder_provenance=provenance,
            )
            budget._full_result = acquired
        observations = merge_current_qr_observations(native_observations, acquired.observations)
        metadata = {**acquired.metadata(), "native_check": native.metadata()}
        qr_elapsed_ms += acquired.elapsed_ms
        if observations != native_observations:
            result, debug = estimate(observations)
            repeated_fit = True
    timings = dict(debug.stage_timings_ms or {})
    if repeated_fit:
        timings["initial_geometry_pass_ms"] = first_fit_ms
        timings["total"] = timings.get("total", 0.) + first_fit_ms
    timings["qr_identity"] = qr_elapsed_ms
    metadata.update(elapsed_ms=qr_elapsed_ms, acquisition=decision.metadata(),
                    current_image_geometry_refit=repeated_fit)
    return result, replace(debug, stage_timings_ms=timings), observations, metadata
