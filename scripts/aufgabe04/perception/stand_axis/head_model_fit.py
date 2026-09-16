"""Fit only the measured physical head from current neutral image evidence."""

from dataclasses import replace
import math

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _debug_rectangle_image, _polygon_area, _unusable,
    estimate_stand_axis_from_corners,
)
from scripts.aufgabe04.perception.stand_axis.head_border_seed import (
    validate_current_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.current_head_refinement import refine_current_physical_head
from scripts.aufgabe04.perception.stand_axis.current_head_refinement_proof import CurrentHeadRefinement
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    MEASURED_HEAD_AXIS_SOURCE, MAX_HEAD_REPROJECTION_RMSE_PX,
    evaluate_head_model_quality, validated_head_model_quality,
)
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import evaluate_current_head_orientation_bounds
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import estimate_planar_pose_ippe


def fit_current_measured_head(
    cv2, raw_edges, *, model_profile, camera, proposal_corners,
    max_reprojection_rmse_px=2.0, min_edge_height_px=8.0,
    current_head_refinement=None, frame_bgr=None,
):
    """Recheck enclosing raw head rails/corners and solve without a pose seed.

    The caller supplies a current crop-local neutral proposal associated with
    the candidate. Proposal coordinates locate a bounded search and supply no
    accepted pose, identity, side label, or historical ambiguity reference.
    """

    def unavailable(reason):
        estimate = replace(
            _unusable(reason, source=MEASURED_HEAD_AXIS_SOURCE),
            evidence_state="unobservable", model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status,
        )
        return estimate, StandAxisEdgeDebugArtifacts(
            edges=raw_edges, raw_edges=raw_edges,
            evidence_state="unobservable", model_reason=estimate.reason,
            model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status,
            model_pose_fit_source=MEASURED_HEAD_AXIS_SOURCE,
        ), None
    camera.validate()
    try:
        proposal_corners = validate_current_head_proposal(proposal_corners, frame_shape=raw_edges.shape)
    except ValueError:
        proposal_corners = None
    if proposal_corners is None:
        return unavailable("head_model_proposal_invalid")
    if (not math.isfinite(max_reprojection_rmse_px) or max_reprojection_rmse_px <= 0.0
            or not math.isfinite(min_edge_height_px) or min_edge_height_px <= 0.0):
        raise ValueError("head fit gates must be finite and positive")
    if current_head_refinement is None:
        refinement, outer_recovery, seed = refine_current_physical_head(
            cv2, raw_edges, model_profile=model_profile, proposal_corners=proposal_corners,
        )
    else:
        try:
            if not isinstance(current_head_refinement, CurrentHeadRefinement):
                raise ValueError("current head refinement proof is invalid")
            refinement, outer_recovery, seed = current_head_refinement.resolve(
                frame_bgr, raw_edges, model_profile=model_profile,
                proposal_corners=proposal_corners)
        except (AttributeError, TypeError, ValueError):
            return unavailable("current_head_refinement_invalid")
    corners = refinement.corners
    pose = None
    if refinement.accepted and corners is not None and outer_recovery.accepted:
        pose = estimate_planar_pose_ippe(
            cv2, corners, model_profile.head_corners, camera,
            max_reprojection_rmse_px=min(max_reprojection_rmse_px, MAX_HEAD_REPROJECTION_RMSE_PX),
        )
    quality = evaluate_head_model_quality(
        cv2, profile=model_profile, camera=camera, corners=corners, pose_result=pose,
        raw_border_support_mean=None if refinement.support is None else refinement.support.mean,
        raw_corner_support_accepted=bool(refinement.corner_arm_support is not None
                                         and refinement.corner_arm_support.accepted),
        centered_neck_supported=False, neck_junction_verified=False,
        outer_border_verified=outer_recovery.accepted,
    )
    orientation_bounds = evaluate_current_head_orientation_bounds(
        cv2, profile=model_profile, camera=camera, corners=corners, pose_result=pose,
        frame_shape=raw_edges.shape,
        raw_border_support_mean=None if refinement.support is None else refinement.support.mean,
        raw_corner_support_accepted=bool(refinement.accepted and refinement.corner_arm_support is not None
                                         and refinement.corner_arm_support.accepted),
        outer_border_verified=outer_recovery.accepted,
    )
    reason = (refinement.reason if not refinement.accepted else
              outer_recovery.reason if not outer_recovery.accepted else quality.reason)
    estimate = replace(
        _unusable(reason, corners=corners,
                  contour_area_px=0.0 if corners is None else _polygon_area(corners),
                  source=MEASURED_HEAD_AXIS_SOURCE),
        evidence_state="unobservable", model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        pose_reprojection_rmse_px=quality.reprojection_rmse_px,
        pose_ambiguity_gap_px=quality.ambiguity_gap_px,
    )
    debug = StandAxisEdgeDebugArtifacts(
        edges=raw_edges, raw_edges=raw_edges, face_mask=refinement.evidence_mask,
        predicted_corners=seed.corners, refined_corners=corners,
        candidate_corners=refinement.candidate_corners,
        corner_arm_support=refinement.corner_arm_support,
        model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        model_pose_fit_source=MEASURED_HEAD_AXIS_SOURCE,
        pose_seed_source="current_head_proposal", model_reason=reason,
        evidence_state="unobservable", head_model_quality=quality,
        head_orientation_bounds=orientation_bounds,
        head_neck_junction=None,
        head_outer_recovery=outer_recovery,
        refinement_support_mean=quality.raw_border_support_mean,
        model_corridor_half_width_px=seed.corridor_half_width_px,
        pose_reprojection_rmse_px=quality.reprojection_rmse_px,
        pose_ambiguity_gap_px=quality.ambiguity_gap_px,
    )
    if not validated_head_model_quality(quality):
        return estimate, debug, pose
    best = pose.best
    estimate = estimate_stand_axis_from_corners(
        corners, min_edge_height_px=min_edge_height_px,
        contour_area_px=_polygon_area(corners), source=MEASURED_HEAD_AXIS_SOURCE,
    )
    if not estimate.usable:
        return replace(estimate, evidence_state="unobservable"), replace(
            debug, model_reason=estimate.reason,
        ), pose
    estimate = replace(
        estimate, usable=True, reason="axis_estimated_current_measured_head",
        yaw_deg=best.yaw_deg, camera_face_normal_xyz=best.face_normal_xyz,
        camera_face_center_xyz_m=best.translation_xyz_m,
        evidence_state="fresh_refined", model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        pose_reprojection_rmse_px=best.reprojection_rmse_px,
        pose_ambiguity_gap_px=pose.ambiguity_gap_px,
        visible_face=None, visible_face_confidence=None,
    )
    projection = project_stand_model(cv2, model_profile, best, camera)
    return estimate, replace(
        debug, rectangle_mask=_debug_rectangle_image(cv2, raw_edges.shape, corners),
        model_pose=best, model_reason=estimate.reason, evidence_state="fresh_refined",
        projected_landmarks=dict(projection.landmarks),
    ), pose
