"""Keep a current selected physical border attached to its current metric fit.

Acquisition already compared physical-border alternatives. Refinement may move
along those same measured rails, but cannot silently switch to an independent
inset or enclosing rectangle after that comparison. Historical hints and manual
unverified seeds do not supply this binding proof.
"""

from dataclasses import replace

from scripts.aufgabe04.perception.stand_axis.head_border_families import same_current_border_family


BORDER_BINDING_REJECTED = "current_head_border_binding_rejected"


def bind_selected_current_head(result, selected_corners, *, raw_edges, frame_bgr):
    """Return the current fit and diagnostic-only selected/fitted rail binding."""
    estimate, debug, pose = result
    diagnostics = {"performed": False, "accepted": False,
                   "policy": "same_current_supporting_border_family",
                   "historical_measurement_reused": False}
    quality = debug.head_model_quality
    complete_border = (quality is not None and quality.outer_border_verified is True
                       and quality.raw_corner_support_accepted is True)
    if ((not estimate.usable and not complete_border)
            or estimate.corners is None or selected_corners is None):
        diagnostics["reason"] = "current_complete_head_corners_unavailable"
        return result, diagnostics
    matched = same_current_border_family(
        tuple(selected_corners), tuple(estimate.corners),
        raw_edges=raw_edges, frame_bgr=frame_bgr)
    diagnostics.update(performed=True, accepted=matched,
        reason="current_selected_border_preserved" if matched else BORDER_BINDING_REJECTED,
        selected_corners=[(p.u_px, p.v_px) for p in selected_corners],
        fitted_corners=[(p.u_px, p.v_px) for p in estimate.corners])
    if matched:
        return result, diagnostics
    rejected = replace(estimate, usable=False, reason=BORDER_BINDING_REJECTED,
        mode="unavailable", evidence_state="unobservable", axis_line=None,
        yaw_deg=None, yaw_proxy=None, closer_side=None,
        camera_face_normal_xyz=None, camera_face_center_xyz_m=None,
        visible_face=None, visible_face_confidence=None)
    if quality is not None:
        quality = replace(quality, accepted=False, reason=BORDER_BINDING_REJECTED,
                          outer_border_verified=False)
    rejected_debug = replace(debug, evidence_state="unobservable",
        model_reason=BORDER_BINDING_REJECTED, model_pose=None, projected_landmarks=None,
        head_model_quality=quality, head_outer_recovery=None,
        head_orientation_bounds=None, head_pose_hypotheses=None,
        head_backside_classification=None, head_backside_appearance=None)
    return (rejected, rejected_debug, None), diagnostics
