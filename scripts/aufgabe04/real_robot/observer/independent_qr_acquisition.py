"""Bound identity work independently of successful physical-head acquisition.

Geometry is evaluated before optional decoding. Its exact-image fit is reused
once when decorating the result, so decoding cannot trigger another acquisition
or fit. A failed locator may still collect a bounded identity probe; that text
has no authority until the ordinary current scan/QR binding accepts it.
"""

from dataclasses import replace

from scripts.aufgabe04.perception.stand_axis.marker_work_schedule import MIN_NATIVE_MARKER_BUDGET_SEC
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import (
    merge_current_qr_observations,
)


def evaluate_geometry_then_identity(*, frame, roi, roi_source, cache, budget,
                                    native_decoder, full_decoder, geometry_only,
                                    decorate, now, current_image_head_fit):
    started = now()
    estimate, debug = geometry_only()
    geometry_ms = (now() - started) * 1000.
    first_timings = dict(debug.stage_timings_ms or {})
    if budget.remaining_work_sec(now()) < MIN_NATIVE_MARKER_BUDGET_SEC:
        return estimate, debug, None, {
            "performed": False, "reason": "identity_deferred_for_source_freshness",
            "geometry_first": True, "elapsed_ms": 0.,
            "current_image_geometry_refit": False,
        }
    native = cache.decode(roi=roi, mode="native", frame=frame, decoder=native_decoder)
    observations = native.observations
    decision = budget.request(
        roi=roi, roi_source=roi_source, now_monotonic_sec=now(),
        current_qr_signal=bool(observations),
        identity_geometry_available=bool(observations) and (
            len(observations) > 1 or observations[0].corners is not None),
        complete_head_available=bool(estimate.usable),
    )
    elapsed_ms = native.elapsed_ms
    metadata = {"native_check": native.metadata()}
    if decision.allowed:
        if decision.cache_only and budget._full_result is not None:
            acquired = replace(budget._full_result, cache_hit=True, elapsed_ms=0.)
        else:
            provenance = {}
            acquired = cache.decode(
                roi=roi, mode="full", frame=frame,
                decoder=lambda crop: full_decoder(crop, decision.max_elapsed_sec, provenance),
                decoder_provenance=provenance)
            budget._full_result = acquired
        observations = merge_current_qr_observations(observations, acquired.observations)
        elapsed_ms += acquired.elapsed_ms
        metadata["full_decode"] = acquired.metadata()
    refreshed = False
    # Publication still checks image AND scan age. Do not start a marker pass
    # when an atomic decoder has already consumed the work allowance.
    if budget.remaining_work_sec(now()) > .005:
        estimate, debug = decorate(observations)
        refreshed = True
    timings = dict(debug.stage_timings_ms or {})
    total_ms = (now() - started) * 1000.
    timings = {**first_timings, **timings, "initial_geometry_pass_ms": geometry_ms,
               "qr_identity": elapsed_ms, "total": total_ms}
    metadata.update(performed=True, geometry_first=True, elapsed_ms=elapsed_ms,
        acquisition=decision.metadata(), current_image_geometry_refit=(
            refreshed and not current_image_head_fit.reused),
        current_image_geometry_reused=refreshed and current_image_head_fit.reused,
        marker_refresh_performed=refreshed)
    return estimate, replace(debug, stage_timings_ms=timings), observations, metadata


def probe_identity_after_head_miss(selection, *, cache, budget, full_decoder, now):
    """Decode one current crop without demanding a successful head or angle.

    A skipped/empty probe never claims marker absence or a backside. The QR's
    original crop coordinates are preserved for normal camera/LiDAR binding.
    """
    current = selection.selected
    if current.qr_decode_metadata is None or current.qr_decode_metadata.get("performed") is not False:
        return selection
    roi = current.attempt.roi
    bounds = (roi.x0, roi.y0, roi.x1, roi.y1)
    decision = budget.request(roi=bounds, roi_source=current.attempt.source,
        now_monotonic_sec=now(), current_qr_signal=False, identity_geometry_available=False,
        selected_crop=True)
    metadata = {**current.qr_decode_metadata, "independent_identity_probe": decision.metadata()}
    if not decision.allowed:
        updated = replace(current, qr_decode_metadata=metadata)
    else:
        provenance = {}
        if decision.cache_only and budget._full_result is not None:
            acquired = replace(budget._full_result, cache_hit=True, elapsed_ms=0.)
        else:
            acquired = cache.decode(roi=bounds, mode="full", frame=current.frame,
                decoder=lambda crop: full_decoder(crop, decision.max_elapsed_sec, provenance),
                decoder_provenance=provenance)
        budget._full_result = acquired
        observations = acquired.observations
        debug = replace(current.debug,
            qr_detected=True if observations else None,
            qr_marker_verified=True if observations else None,
            qr_marker_reason=("decoded_qr_identity" if observations else "head_unavailable_marker_unchecked"))
        updated = replace(current, debug=debug, qr_observations=observations,
            qr_decode_metadata={**metadata, **acquired.metadata(), "performed": True,
                                "geometry_required": False})
    return replace(selection, selected=updated,
        evaluations=tuple(updated if item is current else item for item in selection.evaluations))
