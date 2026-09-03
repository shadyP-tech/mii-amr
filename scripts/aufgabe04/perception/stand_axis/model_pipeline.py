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
from scripts.aufgabe04.perception.stand_axis.model_backside_acquisition import (
    estimate_stand_axis_from_model_backside,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    model_corridor_half_width_px,
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
    detect_qr_quad,
    estimate_planar_pose_ippe,
    select_temporally_consistent_pose,
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
    expected_head_center_u_px: float | None = None,
    expected_head_center_v_px: float | None = None,
    expected_head_height_px: float | None = None,
    backside_target_crop_horizontal_half_width_ratio: float = 1.25,
) -> tuple[StandAxisImageEstimate, StandAxisEdgeDebugArtifacts]:
    """Acquire from QR/tracking or a gated no-QR backside candidate.

    Every successful branch remains bound to current-frame rail support.  The
    no-QR branch is available only with a measured physical model and a full
    candidate-centred expected-head projection.
    """

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
    # A tracked pose already constrains the narrow refinement corridors.  In
    # that state, avoid paying for the 4x acquisition pyramid on every frame;
    # a native QR observation may still refresh the seed.  Once tracking
    # expires, the full pyramid reacquires the stand.
    qr_detection = detect_qr_quad(
        cv2,
        frame,
        scales=((1.0,) if pose_hint is not None else (1.0, 2.0, 4.0)),
    )
    qr_corners = None if qr_detection is None else qr_detection.corners
    qr_pose = None
    if qr_corners is not None:
        qr_pose = estimate_planar_pose_ippe(
            cv2,
            qr_corners,
            model_profile.qr_corners,
            camera,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
        )
    qr_seed = None if qr_pose is None else qr_pose.best
    if (
        qr_pose is not None
        and qr_pose.axis_ambiguous()
        and pose_hint is not None
    ):
        qr_seed = select_temporally_consistent_pose(qr_pose, pose_hint)
    seed_pose = qr_seed if qr_seed is not None else pose_hint
    pose_seed_source = (
        f"qr_pyramid_{qr_detection.scale:g}x"
        if qr_seed is not None and qr_detection is not None
        else ("tracked_pose" if pose_hint is not None else "none")
    )
    if seed_pose is None:
        expected_geometry = (
            expected_head_center_u_px,
            expected_head_center_v_px,
            expected_head_height_px,
        )
        if (
            qr_corners is None
            and pose_hint is None
            and model_profile.committable
            and model_profile.environment == "physical"
            and all(value is not None for value in expected_geometry)
        ):
            return estimate_stand_axis_from_model_backside(
                cv2,
                frame,
                raw_edges=raw_edges,
                model_profile=model_profile,
                expected_head_center_u_px=float(expected_head_center_u_px),
                expected_head_center_v_px=float(expected_head_center_v_px),
                expected_head_height_px=float(expected_head_height_px),
                camera_fx_px=camera.fx_px,
                camera_fy_px=camera.fy_px,
                camera_cx_px=camera.cx_px,
                camera_cy_px=camera.cy_px,
                edge_preprocess=edge_preprocess,
                canny_low=canny_low,
                canny_high=canny_high,
                min_edge_height_px=min_edge_height_px,
                max_reprojection_rmse_px=max_reprojection_rmse_px,
                target_crop_horizontal_half_width_ratio=(
                    backside_target_crop_horizontal_half_width_ratio
                ),
            )
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
            qr_detection_scale=(
                None if qr_detection is None else qr_detection.scale
            ),
            pose_seed_source=pose_seed_source,
            model_reason=estimate.reason,
            model_measurement_status=model_profile.measurement_status,
        )

    projected = project_stand_model(cv2, model_profile, seed_pose, camera)
    corridor_half_width_px = model_corridor_half_width_px(
        projected.head_corners,
        model_profile=model_profile,
        pose_reprojection_rmse_px=seed_pose.reprojection_rmse_px,
    )
    refinement = refine_projected_head_border(
        cv2,
        raw_edges,
        projected.head_corners,
        corridor_half_width_px=corridor_half_width_px,
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
            model_corridor_half_width_px=corridor_half_width_px,
            model_pose=seed_pose,
            qr_detected=qr_corners is not None,
            qr_detection_scale=(
                None if qr_detection is None else qr_detection.scale
            ),
            pose_seed_source=pose_seed_source,
            model_reason=estimate.reason,
            model_measurement_status=model_profile.measurement_status,
            projected_landmarks=dict(projected.landmarks),
        )

    pose_image_points = tuple(refinement.corners)
    pose_model_points = tuple(model_profile.head_corners)
    pose_fit_source = (
        "head_only_provisional"
        if not model_profile.committable
        else "head_only_qr_unavailable"
    )
    if qr_corners is not None and model_profile.committable:
        # QR corners and outer-head corners are independent semantic
        # observations of one measured plane. Their joint fit is much harder
        # for a background rail to satisfy than the head rectangle alone.
        pose_image_points += tuple(qr_corners)
        pose_model_points += tuple(model_profile.qr_corners)
        pose_fit_source = "joint_qr_head"
    refined_pose = estimate_planar_pose_ippe(
        cv2,
        pose_image_points,
        pose_model_points,
        camera,
        max_reprojection_rmse_px=max_reprojection_rmse_px,
    )
    refined_axis_ambiguous = refined_pose.axis_ambiguous()
    selected_pose = refined_pose.best
    ambiguity_resolved = False
    ambiguity_reference = pose_hint
    if (
        ambiguity_reference is None
        and qr_pose is not None
        and qr_seed is not None
        and not qr_pose.axis_ambiguous()
    ):
        # On acquisition, an unambiguous direct QR pose can resolve the refined
        # head pose. If QR itself is ambiguous, no same-frame model prediction
        # is allowed to manufacture certainty.
        ambiguity_reference = qr_seed
    if refined_axis_ambiguous and ambiguity_reference is not None:
        selected_pose = select_temporally_consistent_pose(
            refined_pose,
            ambiguity_reference,
        )
        ambiguity_resolved = selected_pose is not None
    if (
        not refined_pose.accepted
        or selected_pose is None
        or (refined_axis_ambiguous and not ambiguity_resolved)
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
            model_corridor_half_width_px=corridor_half_width_px,
            model_pose_fit_source=pose_fit_source,
            model_pose=seed_pose,
            qr_detected=qr_corners is not None,
            qr_detection_scale=(
                None if qr_detection is None else qr_detection.scale
            ),
            pose_seed_source=pose_seed_source,
            model_reason=estimate.reason,
            model_measurement_status=model_profile.measurement_status,
            projected_landmarks=dict(projected.landmarks),
        )

    best = selected_pose
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
        model_corridor_half_width_px=corridor_half_width_px,
        model_pose_fit_source=pose_fit_source,
        model_pose=best,
        qr_detected=qr_corners is not None,
        qr_detection_scale=(
            None if qr_detection is None else qr_detection.scale
        ),
        pose_seed_source=pose_seed_source,
        model_reason=estimate.reason,
        model_measurement_status=model_profile.measurement_status,
        projected_landmarks=dict(projected.landmarks),
    )
