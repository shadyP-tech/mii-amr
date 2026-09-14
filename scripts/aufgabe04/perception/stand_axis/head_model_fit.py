"""Fit only the measured physical head from current neutral image evidence."""

from dataclasses import replace
import math

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _debug_rectangle_image, _polygon_area, _unusable,
    estimate_stand_axis_from_corners,
)
from scripts.aufgabe04.perception.stand_axis.geometry_contract import classify_joint_geometry_contract
from scripts.aufgabe04.perception.stand_axis.head_border_seed import (
    select_head_border_seed, validate_current_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    MEASURED_HEAD_AXIS_SOURCE, MAX_HEAD_REPROJECTION_RMSE_PX,
    evaluate_head_model_quality, validated_head_model_quality,
)
from scripts.aufgabe04.perception.stand_axis.head_outer_border import select_current_outer_head_border
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.perception.stand_axis.pose_fit_diagnostics import collect_metric_model_diagnostics
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import estimate_planar_pose_ippe


def fit_current_measured_head(
    cv2, raw_edges, *, model_profile, camera, proposal_corners,
    max_reprojection_rmse_px=2.0, min_edge_height_px=8.0,
):
    """Recheck enclosing raw head rails/corners and solve without a pose seed.

    The caller supplies a current crop-local neutral proposal associated with
    the candidate. Proposal coordinates locate a bounded search and supply no
    accepted pose, identity, side label, or historical ambiguity reference.
    """

    camera.validate()
    try:
        proposal_corners = validate_current_head_proposal(proposal_corners, frame_shape=raw_edges.shape)
    except ValueError:
        proposal_corners = None
    if proposal_corners is None:
        estimate = replace(
            _unusable("head_model_proposal_invalid", source=MEASURED_HEAD_AXIS_SOURCE),
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
    if (not math.isfinite(max_reprojection_rmse_px) or max_reprojection_rmse_px <= 0.0
            or not math.isfinite(min_edge_height_px) or min_edge_height_px <= 0.0):
        raise ValueError("head fit gates must be finite and positive")
    seed = select_head_border_seed(
        model_profile=model_profile, projected_corners=None,
        pose_reprojection_rmse_px=None, current_head_proposal_corners=proposal_corners,
    )
    refinement = refine_projected_head_border(
        cv2, raw_edges, seed.corners, corridor_half_width_px=seed.corridor_half_width_px,
    )
    refinement, outer_recovery = select_current_outer_head_border(
        cv2, raw_edges, model_profile=model_profile, refinement=refinement,
        corridor_half_width_px=seed.corridor_half_width_px,
    )
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
    reason = quality.reason if refinement.accepted else refinement.reason
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


def attach_independent_qr_diagnostics(
    cv2, *, estimate, debug, head_pose, qr_corners, marker_verified,
    model_profile, camera, max_reprojection_rmse_px,
):
    """Explain QR/model disagreement without changing the head estimate/pose."""

    qr_pose = None
    if qr_corners is not None:
        qr_pose = estimate_planar_pose_ippe(
            cv2, qr_corners, model_profile.qr_corners, camera,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
        )
    joint = None
    if qr_corners is not None and debug.refined_corners is not None:
        joint = estimate_planar_pose_ippe(
            cv2, debug.refined_corners + tuple(qr_corners),
            model_profile.head_corners + model_profile.qr_corners, camera,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
        )
    diagnostic_pose = (None if joint is None or not joint.hypotheses else joint.hypotheses[0])
    diagnostics = collect_metric_model_diagnostics(
        cv2, profile=model_profile, camera=camera,
        head_corners=debug.refined_corners, qr_corners=qr_corners,
        diagnostic_pose=diagnostic_pose or debug.model_pose,
        qr_pose=qr_pose, head_pose=head_pose,
        max_reprojection_rmse_px=max_reprojection_rmse_px,
    )
    if joint is not None and not joint.accepted:
        contract = classify_joint_geometry_contract(
            profile=model_profile, diagnostics=diagnostics, joint_reason=joint.reason,
            joint_reprojection_rmse_px=None if diagnostic_pose is None else diagnostic_pose.reprojection_rmse_px,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
            qr_marker_verified=marker_verified,
        )
        if contract is not None:
            diagnostics = replace(diagnostics, geometry_contract=contract)
    return estimate, replace(debug, model_diagnostics=diagnostics)
