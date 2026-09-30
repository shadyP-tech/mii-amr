"""Lifetime camera activity and processed-frame outcomes, without authority."""

from collections import Counter
from typing import Mapping


CAMERA_PIPELINE_COUNTERS = frozenset({
    "received_images", "received_scans", "received_camera_infos",
    "invalid_image_headers", "invalid_scan_headers", "invalid_camera_info_headers",
    "ingress_overwritten_images", "ingress_overwritten_scans",
    "ingress_overwritten_camera_infos", "scan_witness_ingress_gap",
    "scan_witness_expired_before_tf", "scan_witness_context_rejected",
    "scan_witness_tf_pending", "scan_witness_ingested", "scan_witness_ingestion_rejected",
    "capture_metadata_failures", "publication_rejections", "committed_artifacts",
    "associated_frames", "axis_sample_frames", "qr_sample_frames", "unpaired_images",
    "synchronized_tuples", "tf_ready_tuples", "stale_input_images", "processed_images",
    "fresh_detector_results", "verified_geometry_results", "obsolete_detector_results",
    "consensus_frames",
})
OUTCOME_FIELDS = ("state", "reason", "estimator_reason", "association_reason")
MAX_OUTCOME_LABELS = 64
MAX_LABEL_LENGTH = 256


def _label(value):
    return value.strip()[:MAX_LABEL_LENGTH] if isinstance(value, str) and value.strip() else None


def _counts(value, allowed=None):
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, count in value.items():
        if (_label(key) is not None and len(key) <= MAX_LABEL_LENGTH
                and (allowed is None or key in allowed)
                and isinstance(count, int) and not isinstance(count, bool) and count >= 0):
            result[key] = count
            if len(result) >= MAX_OUTCOME_LABELS + 1:
                break
    return result


def validate_camera_pipeline_counts(value):
    """Ignore malformed or unknown diagnostic counters, never coerce them."""
    return _counts(value, CAMERA_PIPELINE_COUNTERS)


def validate_camera_processing_outcomes(value):
    if not isinstance(value, Mapping):
        return None
    last = value.get("last_frame")
    if not isinstance(last, Mapping) or _label(last.get("state")) is None:
        return None
    return {
        "diagnostic_only": True,
        "last_frame": {key: text for key in OUTCOME_FIELDS
                       if (text := _label(last.get(key))) is not None},
        **{key + "_counts": _counts(value.get(key + "_counts"))
           for key in OUTCOME_FIELDS},
    }


class CameraProcessingDiagnostics:
    """Count one outcome per processed image, excluding TF/sensor statuses.

    The owner starts this only after exact-time TF, immediately before image
    decoding. The first outcome consumes the marker, independently of optional
    capture storage, so subsequent pending/exhausted TF snapshots retain it.
    """

    def __init__(self):
        self._pending = False
        self._last = None
        self._counts = {key: Counter() for key in OUTCOME_FIELDS}

    def begin_frame(self):
        self._pending = True

    def discard_pending(self):
        # A suppressed/failed publication from the preceding iteration must
        # never turn a later sensor or TF status into a processed outcome.
        self._pending = False

    def record_outcome(self, state, details):
        if not self._pending:
            return
        self._pending = False
        debug = details.get("stand_axis_debug")
        debug = debug if isinstance(debug, Mapping) else {}
        association = debug.get("current_head_candidate_association")
        association = association if isinstance(association, Mapping) else {}
        values = dict(
            state=state, reason=details.get("reason"),
            estimator_reason=details.get("estimator_reason", debug.get("estimator_reason")),
            association_reason=association.get("reason"),
        )
        self._last = {key: text for key, value in values.items()
                      if (text := _label(value)) is not None}
        for key, value in self._last.items():
            counts = self._counts[key]
            if value not in counts and len(counts) >= MAX_OUTCOME_LABELS:
                value = "other_labels"
            counts[value] += 1

    def snapshot(self):
        if self._last is None:
            return None
        return {"diagnostic_only": True, "last_frame": dict(self._last),
                **{key + "_counts": dict(counts) for key, counts in self._counts.items()}}
