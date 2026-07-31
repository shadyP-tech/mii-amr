"""Model-seeded acquisition, projection, and current-frame refinement."""

from __future__ import annotations

from dataclasses import replace

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _debug_rectangle_image,
    _debug_rectangle_overlay_image,
    _polygon_area,
    _unusable,
    estimate_stand_axis_from_corners,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    refine_projected_head_border,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis.preprocessing import (
    _canny_edges_from_frame,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    PlanarPoseHypothesis,
    RectifiedCameraMatrix,
    detect_qr_quad_corners,
    estimate_planar_pose_ippe,
)


def estimate_stand_axis_from_metric_model(
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
    min_edge_height_px: float = 8.0,
    max_reprojection_rmse_px: float = 2.0,
) -> tuple[StandAxisImageEstimate, StandAxisEdgeDebugArtifacts]:
    """Project from QR/tracking, then accept only current-frame rail support."""

    camera = RectifiedCameraMatrix(
        float(camera_fx_px),
        float(camera_fy_px),
        float(camera_cx_px),
        float(camera_cy_px),
    )
    camera.validate()
    raw_edges = _canny_edges_from_frame(
        cv2,
        frame,
        edge_preprocess=edge_preprocess,
        blur_kernel=blur_kernel,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    qr_corners = detect_qr_quad_corners(cv2, frame)
    qr_pose = None
    if qr_corners is not None:
        qr_pose = estimate_planar_pose_ippe(
            cv2,
            qr_corners,
            model_profile.qr_corners,
            camera,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
        )
    seed_pose = (
        qr_pose.best
        if qr_pose is not None and qr_pose.best is not None
        else pose_hint
    )
    if seed_pose is None:
        estimate = replace(
            _unusable("model_pose_seed_unavailable", source="model_seed"),
            evidence_state="unobservable",
            model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status,
        )
        return estimate, StandAxisEdgeDebugArtifacts(
            edges=raw_edges,
            raw_edges=raw_edges,
            evidence_state="unobservable",
            model_profile_sha256=model_profile.sha256,
            qr_detected=qr_corners is not None,
        )

    projected = project_stand_model(cv2, model_profile, seed_pose, camera)
    refinement = refine_projected_head_border(
        cv2,
        raw_edges,
        projected.head_corners,
    )
    seed_rmse = (
        None
        if qr_pose is None or qr_pose.best is None
        else qr_pose.best.reprojection_rmse_px
    )
    seed_gap = None if qr_pose is None else qr_pose.ambiguity_gap_px
    if not refinement.accepted or refinement.corners is None:
        estimate = replace(
            _unusable(
                refinement.reason,
                corners=projected.head_corners,
                contour_area_px=_polygon_area(projected.head_corners),
                source="model_projection",
            ),
            evidence_state="predicted_only",
            model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status,
            pose_reprojection_rmse_px=seed_rmse,
            pose_ambiguity_gap_px=seed_gap,
        )
        return estimate, StandAxisEdgeDebugArtifacts(
            edges=raw_edges,
            raw_edges=raw_edges,
            face_mask=refinement.evidence_mask,
            predicted_corners=projected.head_corners,
            evidence_state="predicted_only",
            model_profile_sha256=model_profile.sha256,
            pose_reprojection_rmse_px=seed_rmse,
            pose_ambiguity_gap_px=seed_gap,
            refinement_support_mean=(
                None if refinement.support is None else refinement.support.mean
            ),
            model_pose=seed_pose,
            qr_detected=qr_corners is not None,
        )

    refined_pose = estimate_planar_pose_ippe(
        cv2,
        refinement.corners,
        model_profile.head_corners,
        camera,
        max_reprojection_rmse_px=max_reprojection_rmse_px,
    )
    refined_axis_ambiguous = refined_pose.axis_ambiguous()
    if (
        not refined_pose.accepted
        or refined_pose.best is None
        or refined_axis_ambiguous
    ):
        estimate = replace(
            _unusable(
                (
                    "planar_pose_axis_ambiguous"
                    if refined_axis_ambiguous
                    else refined_pose.reason
                ),
                corners=refinement.corners,
                contour_area_px=_polygon_area(refinement.corners),
                source="model_refined_head",
            ),
            evidence_state="ambiguous",
            model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status,
            pose_reprojection_rmse_px=(
                None
                if not refined_pose.hypotheses
                else refined_pose.hypotheses[0].reprojection_rmse_px
            ),
            pose_ambiguity_gap_px=refined_pose.ambiguity_gap_px,
        )
        return estimate, StandAxisEdgeDebugArtifacts(
            edges=raw_edges,
            raw_edges=raw_edges,
            face_mask=refinement.evidence_mask,
            predicted_corners=projected.head_corners,
            evidence_state="ambiguous",
            model_profile_sha256=model_profile.sha256,
            pose_reprojection_rmse_px=estimate.pose_reprojection_rmse_px,
            pose_ambiguity_gap_px=refined_pose.ambiguity_gap_px,
            refinement_support_mean=(
                None if refinement.support is None else refinement.support.mean
            ),
            model_pose=seed_pose,
            qr_detected=qr_corners is not None,
        )

    best = refined_pose.best
    estimate = estimate_stand_axis_from_corners(
        refinement.corners,
        min_edge_height_px=min_edge_height_px,
        stand_width_m=model_profile.head_width_m,
        camera_fx_px=camera.fx_px,
        camera_fy_px=camera.fy_px,
        camera_cx_px=camera.cx_px,
        camera_cy_px=camera.cy_px,
        cv2=cv2,
        contour_area_px=_polygon_area(refinement.corners),
        source="model_current_frame_refined",
    )
    estimate = replace(
        estimate,
        reason="axis_estimated_model_current_frame_refined",
        yaw_deg=best.yaw_deg,
        camera_face_normal_xyz=best.face_normal_xyz,
        camera_face_center_xyz_m=best.translation_xyz_m,
        evidence_state="fresh_refined",
        model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        pose_reprojection_rmse_px=best.reprojection_rmse_px,
        pose_ambiguity_gap_px=refined_pose.ambiguity_gap_px,
    )
    return estimate, StandAxisEdgeDebugArtifacts(
        edges=raw_edges,
        raw_edges=raw_edges,
        face_mask=refinement.evidence_mask,
        rectangle_mask=_debug_rectangle_image(
            cv2, raw_edges.shape, refinement.corners
        ),
        rectangle_overlay=_debug_rectangle_overlay_image(
            cv2,
            raw_edges.shape,
            refinement.corners,
            refinement.evidence_mask,
        ),
        predicted_corners=projected.head_corners,
        evidence_state="fresh_refined",
        model_profile_sha256=model_profile.sha256,
        pose_reprojection_rmse_px=best.reprojection_rmse_px,
        pose_ambiguity_gap_px=refined_pose.ambiguity_gap_px,
        refinement_support_mean=(
            None if refinement.support is None else refinement.support.mean
        ),
        model_pose=best,
        qr_detected=qr_corners is not None,
    )
