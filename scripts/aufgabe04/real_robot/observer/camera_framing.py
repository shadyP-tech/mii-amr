"""Current-image framing hints, without pose, identity or motion authority.

The caller must establish fresh, stationary, candidate-associated front
evidence before invoking the builder. The envelope uses the observed head's
angular size and optical depth; it is a search constraint, not a calibration
measurement or a replacement for the next observation's pixel gates.
"""

from collections.abc import Mapping
import math


FRAMING_REASONS = frozenset({"crop_clipped", "head_qr_geometry_mismatch"})
MIN_HEAD_SIZE_PX = 18.0
FRAME_USABLE_FRACTION = 0.8
HEAD_AND_NECK_HEIGHT_FACTOR = 1.5


def _positive(value: object, *, zero_allowed: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if zero_allowed else value > 0))


def validate_camera_framing_hint(value: object) -> dict | None:
    """Drop incomplete hints; they must never change timeout admission."""
    if not isinstance(value, Mapping):
        return None
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or not isinstance(value.get("target_key"), str) or not value["target_key"].strip()
            or not isinstance(value.get("reason"), str) or value["reason"] not in FRAMING_REASONS
            or value.get("front_evidence_verified") is not True
            or value.get("candidate_associated") is not True
            or value.get("framing_limited") is not True
            or value.get("motion_authorized") is not False
            or value.get("completion_authorized") is not False
            or not _positive(value.get("source_image_stamp_sec"), zero_allowed=True)):
        return None
    if not all(_positive(value.get(key)) for key in (
        "range_m", "optical_depth_m", "minimum_range_m", "maximum_range_m",
    )) or value["minimum_range_m"] >= value["maximum_range_m"]:
        return None
    return dict(value)


def build_camera_framing_hint(
    *, target_key: str, source_image_stamp_sec: float,
    front_evidence_verified: bool, candidate_associated: bool, reason: str,
    range_m: float, optical_depth_m: float, image_size: tuple[int, int],
    head_bounds: tuple[float, float, float, float], fx_px: float, fy_px: float,
) -> dict | None:
    """Suggest a bounded framing search after a strict current-pixel fit fails.

    Nominal ROI clipping by itself is not a trigger: callers first recenter
    the head and attempt strict fitting. The hint cannot certify a stand
    angle. Focal lengths and measured angular extent retain the current
    camera/depth relationship instead of introducing a global standoff floor.
    """
    if front_evidence_verified is not True or candidate_associated is not True:
        return None
    if (not isinstance(reason, str) or reason not in FRAMING_REASONS or not isinstance(target_key, str)
            or not target_key.strip()
            or not _positive(source_image_stamp_sec, zero_allowed=True)
            or not all(_positive(v) for v in (range_m, optical_depth_m, fx_px, fy_px))
            or not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or not all(type(v) is int and v > 0 for v in image_size)
            or not isinstance(head_bounds, (tuple, list)) or len(head_bounds) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in head_bounds)):
        return None
    width, height = image_size
    x0, y0, x1, y1 = head_bounds
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    angular_width, angular_height = (x1 - x0) / fx_px, (y1 - y0) / fy_px
    head_width_m, head_height_m = angular_width * optical_depth_m, angular_height * optical_depth_m
    minimum_depth = max(
        fx_px * head_width_m / (width * FRAME_USABLE_FRACTION),
        fy_px * head_height_m * HEAD_AND_NECK_HEIGHT_FACTOR / (height * FRAME_USABLE_FRACTION),
    )
    maximum_depth = min(fx_px * head_width_m, fy_px * head_height_m) / MIN_HEAD_SIZE_PX
    # Preserve the measured base-range/optical-depth offset for this short
    # same-bearing search. Reprojection and full gates run again on arrival.
    depth_offset = range_m - optical_depth_m
    payload = {
        "schema_version": 1, "target_key": target_key,
        "source_image_stamp_sec": source_image_stamp_sec,
        "front_evidence_verified": True, "candidate_associated": True,
        "framing_limited": True, "reason": reason,
        "range_m": range_m, "optical_depth_m": optical_depth_m,
        "minimum_range_m": max(1.0e-6, minimum_depth + depth_offset),
        "maximum_range_m": maximum_depth + depth_offset,
        "image_size": [width, height], "head_bounds": list(head_bounds),
        "focal_lengths_px": [fx_px, fy_px],
        "minimum_head_size_px": MIN_HEAD_SIZE_PX,
        "motion_authorized": False, "completion_authorized": False,
    }
    return validate_camera_framing_hint(payload)
