"""Bounded, stationary evidence that the current target has no scan support.

This policy only produces a no-motion deferral. The caller supplies a current
raw registration association after validating the image, scan and exact TF.
Successful current reconciliation or head association always clears the window.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping


STATE = "target_reconciliation_required"
REASON = "persistent_target_support_missing"
POLICY = "stationary_fresh_target_support_failure"
MIN_SAMPLES = 7
MIN_SAMPLE_SPAN_SEC = 2.0
MIN_ELAPSED_SEC = 5.0
MAX_HISTORY_SEC = 15.0
MAX_SENSOR_AGE_SEC = 0.5
MAX_SYNC_DELTA_SEC = 0.1
MAX_TRANSLATION_M = 0.02
MAX_ROTATION_RAD = math.radians(2.0)
_HASH_FIELDS = ("candidate_snapshot_sha256", "robot_profile_sha256",
                "calibration_profile_sha256", "stand_model_profile_sha256")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _binding(value):
    if not isinstance(value, Mapping):
        raise ValueError("target binding must be a mapping")
    result = {}
    for name in ("candidate_uid", "stream_id", "target_key", "planning_frame"):
        text = value.get(name)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"target binding lacks {name}")
        result[name] = text
    center = value.get("stand_center")
    if not isinstance(center, Mapping):
        raise ValueError("target binding lacks stand_center")
    result["stand_center"] = {key: _finite(center.get(key), key) for key in ("x_m", "y_m")}
    for name in _HASH_FIELDS:
        digest = value.get(name)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"target binding has invalid {name}")
        result[name] = digest
    if set(value) != set(result) or set(center) != {"x_m", "y_m"}:
        raise ValueError("target binding has unsupported fields")
    return result


def target_support_binding_sha256(binding):
    """Return the canonical digest of the fully validated target binding."""
    raw = json.dumps(_binding(binding), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _frame(value, now):
    if not isinstance(value, Mapping) or value.get("tf_validated") is not True:
        raise ValueError("current exact TF is not validated")
    for name in ("poisoned", "motion_epoch_reset", "frame_accepted"):
        if value.get(name) is not False:
            raise ValueError(f"frame is not an unaccepted stationary observation: {name}")
    image = _finite(value.get("frame_stamp_sec"), "frame_stamp_sec")
    scan = _finite(value.get("scan_stamp_sec"), "scan_stamp_sec")
    if (not 0 <= now-image <= MAX_SENSOR_AGE_SEC
            or not 0 <= now-scan <= MAX_SENSOR_AGE_SEC
            or abs(image-scan) > MAX_SYNC_DELTA_SEC):
        raise ValueError("target support tuple is stale or unsynchronized")
    pose = value.get("robot_pose")
    if not isinstance(pose, Mapping):
        raise ValueError("target support frame lacks robot pose")
    return dict(frame_stamp_sec=image, scan_stamp_sec=scan,
                robot_pose={key: _finite(pose.get(key), key) for key in ("x_m", "y_m", "yaw_rad")},
                motion_epoch=_nonnegative_int(value.get("motion_epoch"), "motion_epoch"),
                tf_validated=True, poisoned=False, motion_epoch_reset=False, frame_accepted=False)


def _association(value, frame, now):
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("target support association schema is invalid")
    if type(value.get("associated")) is not bool:
        raise ValueError("association decision is invalid")
    result = {"schema_version": 1, "associated": value["associated"]}
    for name in ("rejection_reason", "scan_frame_id"):
        if not isinstance(value.get(name), str) or (name == "scan_frame_id" and not value[name]):
            raise ValueError(f"association lacks {name}")
        result[name] = value[name]
    for name in ("scan_stamp_sec", "scan_age_sec", "map_bearing_rad", "cone_half_angle_rad"):
        result[name] = _finite(value.get(name), name)
    if (result["scan_stamp_sec"] != frame["scan_stamp_sec"]
            or not 0 <= result["scan_age_sec"] <= MAX_SENSOR_AGE_SEC
            or result["scan_age_sec"] > now-frame["scan_stamp_sec"]+1e-6
            or not 0 < result["cone_half_angle_rad"] <= math.radians(15)+1e-12):
        raise ValueError("association is not bound to a fresh candidate scan cone")
    limits = value.get("accepted_range_m")
    if not isinstance(limits, (list, tuple)) or len(limits) != 2:
        raise ValueError("association range is invalid")
    result["accepted_range_m"] = [_finite(x, "accepted_range_m") for x in limits]
    if not 0 <= result["accepted_range_m"][0] < result["accepted_range_m"][1]:
        raise ValueError("association range is invalid")
    for name in ("cone_valid_sample_count", "in_range_sample_count", "candidate_cluster_count",
                 "eligible_cluster_count", "selected_cluster_sample_count"):
        result[name] = _nonnegative_int(value.get(name), name)
    if (result["in_range_sample_count"] > result["cone_valid_sample_count"]
            or result["eligible_cluster_count"] > result["candidate_cluster_count"]
            or result["candidate_cluster_count"] > result["in_range_sample_count"]):
        raise ValueError("association sample counts are inconsistent")
    for name in ("nearest_cone_distance_m", "nearest_range_delta_m"):
        raw = value.get(name)
        result[name] = None if raw is None else _finite(raw, name)
    return result


def _negative(association):
    """Empty/invalid scan rays and ambiguous clusters are never absence proof."""
    lower, upper = association["accepted_range_m"]
    nearest = association["nearest_cone_distance_m"]
    delta = association["nearest_range_delta_m"]
    return (association["associated"] is False
            and association["rejection_reason"] == "no_samples_in_accepted_range"
            and association["cone_valid_sample_count"] > 0
            and all(association[name] == 0 for name in
                    ("in_range_sample_count", "candidate_cluster_count", "eligible_cluster_count",
                     "selected_cluster_sample_count"))
            and nearest is not None and nearest > 0
            and (nearest < lower or nearest > upper)
            and delta is not None and delta > 0
            and math.isclose(delta, max(lower-nearest, nearest-upper, 0.), abs_tol=1e-9))


def _stationary(first, current):
    p, q = first["robot_pose"], current["robot_pose"]
    angle = math.atan2(math.sin(q["yaw_rad"]-p["yaw_rad"]), math.cos(q["yaw_rad"]-p["yaw_rad"]))
    return (first["motion_epoch"] == current["motion_epoch"]
            and math.hypot(q["x_m"]-p["x_m"], q["y_m"]-p["y_m"]) <= MAX_TRANSLATION_M
            and abs(angle) <= MAX_ROTATION_RAD)


def validate_target_support_failure(payload, *, target_binding=None):
    """Validate a receipt's binding and every fresh negative tuple; return a copy.

    This checks the producer's explicit scan association evidence. It does not
    invent raw scan measurements or authorize any subsequent recovery motion.
    The containing artifact's content hash must also be checked by its reader.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("target support receipt must be a mapping")
    if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
            or payload.get("policy") != POLICY or payload.get("state") != STATE
            or payload.get("disposition") != STATE or payload.get("reason") != REASON):
        raise ValueError("target support receipt policy is invalid")
    for flag in ("motion_authorized", "candidate_geometry_updated", "completion_authorized"):
        if payload.get(flag) is not False:
            raise ValueError(f"target support receipt cannot grant {flag}")
    binding = _binding(payload.get("target_binding"))
    if (payload.get("target_binding_sha256") != target_support_binding_sha256(binding)
            or (target_binding is not None and binding != _binding(target_binding))):
        raise ValueError("target support receipt binding mismatch")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not MIN_SAMPLES <= len(samples) <= 1024:
        raise ValueError("target support receipt sample count is invalid")
    previous = None
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("target support sample must be a mapping")
        now = _finite(sample.get("observed_at_sec"), "observed_at_sec")
        frame = _frame(sample.get("frame"), now)
        association = _association(sample.get("association"), frame, now)
        if not _negative(association):
            raise ValueError("target support sample is not an explicit range miss")
        if not _stationary(samples[0]["frame"], frame):
            raise ValueError("target support samples changed stationary epoch")
        if previous is not None and (now <= previous["observed_at_sec"]
                or frame["frame_stamp_sec"] <= previous["frame"]["frame_stamp_sec"]
                or frame["scan_stamp_sec"] <= previous["frame"]["scan_stamp_sec"]):
            raise ValueError("target support samples must have distinct increasing tuples")
        previous = sample
    start, end = samples[0]["observed_at_sec"], samples[-1]["observed_at_sec"]
    span = samples[-1]["frame"]["frame_stamp_sec"]-samples[0]["frame"]["frame_stamp_sec"]
    if not MIN_ELAPSED_SEC <= end-start <= MAX_HISTORY_SEC or span < MIN_SAMPLE_SPAN_SEC:
        raise ValueError("target support receipt does not satisfy the observation window")
    expected = dict(sample_count=len(samples), observed_start_sec=start, observed_end_sec=end,
                    elapsed_sec=end-start, sample_span_sec=span)
    for name, expected_value in expected.items():
        actual = _finite(payload.get(name), name)
        if not math.isclose(actual, expected_value, abs_tol=1e-9):
            raise ValueError(f"target support receipt has inconsistent {name}")
    return copy.deepcopy(dict(payload))


class TargetSupportFailureWindow:
    """Accumulate only fresh, stopped, distinct raw target range misses."""

    def __init__(self):
        self.context = None
        self.samples = []
        self.metadata = {"ready": False, "reason": "no_samples", "sample_count": 0}

    def reset(self, reason="reset"):
        self.context = None
        self.samples = []
        self.metadata = {"ready": False, "reason": reason, "sample_count": 0}

    def observe(self, *, association, frame, target_binding, now_sec,
                reconciliation_validated=False, associated_head=False):
        """Return a validated deferral receipt when the opportunity expires.

        ``reconciliation_validated`` means the caller successfully replayed a
        proof bound to this exact image/scan tuple, target and stationary epoch.
        It is not a metadata ``ready`` hint. ``associated_head`` similarly means
        the final current head association succeeded.
        """
        try:
            now = _finite(now_sec, "now_sec")
            binding = _binding(target_binding)
            current = _frame(frame, now)
            if type(reconciliation_validated) is not bool or type(associated_head) is not bool:
                raise ValueError("current support decisions must be boolean")
            if reconciliation_validated or associated_head:
                self.reset("current_reconciliation_supported" if reconciliation_validated else "current_head_supported")
                return None
            current_association = _association(association, current, now)
        except (TypeError, ValueError, KeyError) as exc:
            self.reset(str(exc))
            return None
        if current_association["associated"] or current_association["in_range_sample_count"] > 0:
            self.reset("current_raw_target_supported")
            return None
        if not _negative(current_association):
            self.reset("current_scan_does_not_prove_target_absence")
            return None
        context = (target_support_binding_sha256(binding), current["motion_epoch"])
        if context != self.context:
            self.reset("target_binding_changed")
            self.context = context
        self.samples = [item for item in self.samples if 0 <= now-item["observed_at_sec"] <= MAX_HISTORY_SEC]
        if self.samples:
            previous = self.samples[-1]
            if (now <= previous["observed_at_sec"]
                    or current["frame_stamp_sec"] <= previous["frame"]["frame_stamp_sec"]
                    or current["scan_stamp_sec"] <= previous["frame"]["scan_stamp_sec"]):
                self.metadata = dict(ready=False, reason="duplicate_or_out_of_order_tuple", sample_count=len(self.samples))
                return None
            if not _stationary(self.samples[0]["frame"], current):
                self.samples = []
        self.samples.append(dict(observed_at_sec=now, frame=current, association=current_association))
        self.samples = self.samples[-1024:]
        start = self.samples[0]["observed_at_sec"]
        span = current["frame_stamp_sec"]-self.samples[0]["frame"]["frame_stamp_sec"]
        self.metadata = dict(ready=False, reason="collecting_fresh_target_support_failures",
                             sample_count=len(self.samples), elapsed_sec=now-start, sample_span_sec=span)
        if len(self.samples) < MIN_SAMPLES or now-start < MIN_ELAPSED_SEC or span < MIN_SAMPLE_SPAN_SEC:
            return None
        receipt = dict(schema_version=1, policy=POLICY, state=STATE, disposition=STATE, reason=REASON,
                       target_binding=binding, target_binding_sha256=context[0],
                       samples=copy.deepcopy(self.samples), sample_count=len(self.samples),
                       observed_start_sec=start, observed_end_sec=now, elapsed_sec=now-start,
                       sample_span_sec=span, motion_authorized=False,
                       candidate_geometry_updated=False, completion_authorized=False)
        receipt = validate_target_support_failure(receipt)
        self.metadata = dict(ready=True, reason=REASON, sample_count=len(self.samples), elapsed_sec=now-start)
        return receipt
