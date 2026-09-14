"""Observer face binding for an independently measured head plane."""

import math

from scripts.aufgabe04.perception.stand_axis.head_model_admission import (
    MEASURED_HEAD_AXIS_SOURCE, HeadModelAdmission, admit_measured_head_model,
    requires_measured_head_admission,
)


def measured_head_front_is_current(*, qr_binding, marker_verified: bool,
                                   resolved_qr_id: str | None) -> bool:
    """A head plane gains front meaning only from its independent live QR path."""
    return (
        marker_verified is True and isinstance(resolved_qr_id, str) and bool(resolved_qr_id)
        and qr_binding.accepted is True and qr_binding.symbol_count == 1
        and qr_binding.reason == "decoded_qr_target_associated"
        and qr_binding.qr_texts_for_evidence == (resolved_qr_id,)
    )


def measured_head_needs_full_qr_decode(*, previous_axis_source: str | None,
                                     previous_bound_qr_stamp_sec: float | None,
                                     image_stamp_sec: float, max_age_sec: float) -> bool:
    """A head track cannot replace acquiring this target's QR identity.

    Native tracking is allowed only after the immediately preceding accepted
    frame refreshed a bound temporal identity. One missing/invalid refresh
    clears that qualification, so the next image gets the full decoder again.
    """
    if previous_axis_source != MEASURED_HEAD_AXIS_SOURCE:
        return False
    if (type(previous_bound_qr_stamp_sec) not in (int, float)
            or not math.isfinite(previous_bound_qr_stamp_sec)):
        return True
    return not 0.0 < image_stamp_sec - previous_bound_qr_stamp_sec <= max_age_sec


def measured_head_lidar_rejection(association, *, registered: bool,
                                  cone_half_angle_rad: float, registered_association=None) -> str | None:
    """Require one cluster at the current fitted head's ray in either ROI mode."""
    if association.associated is not True:
        return "measured_head_lidar_unassociated"
    if registered_association is not None and registered_association.witnessed_fragmentation is not None:
        from scripts.aufgabe04.real_robot.observer.scan_target_persistence import registered_target_is_unique
        if (not registered or registered_association.search_association != association
                or not registered_target_is_unique(registered_association)):
            return "measured_head_lidar_fragmentation_unverified"
    if association.eligible_cluster_count != 1:
        from scripts.aufgabe04.real_robot.observer.scan_target_persistence import registered_target_is_unique
        if (not registered or registered_association is None
                or registered_association.search_association != association
                or not registered_target_is_unique(registered_association)):
            return "measured_head_lidar_clusters_ambiguous"
    delta = (association.selected_cluster_bearing_delta_from_map_rad if registered
             else association.selected_cluster_bearing_delta_from_camera_rad)
    if type(delta) not in (int, float) or not math.isfinite(delta) or not 0. <= delta <= cone_half_angle_rad:
        return "measured_head_bearing_outside_target_cluster"
    return None


def head_scale_gate(
    *,
    expected_size_px: float,
    left_height_px: float,
    right_height_px: float,
) -> dict[str, object]:
    """Check that accepted head sides have the calibrated physical scale."""

    expected = float(expected_size_px)
    heights = (float(left_height_px), float(right_height_px))
    measured = sum(heights) / 2.0
    ratio = measured / max(expected, 1.0e-9)
    balance = min(heights) / max(max(heights), 1.0e-9)
    accepted = (
        all(math.isfinite(value) and value > 0.0 for value in (*heights, expected))
        and 0.60 <= ratio <= 1.35
        and balance >= 0.65
    )
    return {
        "accepted": accepted,
        "expected_size_px": expected,
        "measured_height_px": measured,
        "left_height_px": heights[0],
        "right_height_px": heights[1],
        "height_ratio": ratio,
        "side_balance": balance,
        "reason": "ok" if accepted else "head_size_projection_mismatch",
    }
