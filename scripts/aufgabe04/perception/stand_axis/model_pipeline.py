"""Independent current measured-head angles and separate QR/model diagnostics."""

from __future__ import annotations

from dataclasses import replace

from scripts.aufgabe04.perception.stand_axis.geometry_contract import (
    classify_joint_geometry_contract,
)
from scripts.aufgabe04.perception.stand_axis.head_model_fit import (
    attach_independent_qr_diagnostics, fit_current_measured_head,
)
from scripts.aufgabe04.perception.stand_axis.head_proposal import acquire_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import classify_current_head_backside
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
from scripts.aufgabe04.perception.stand_axis.head_border_seed import (
    select_head_border_seed,
    validate_current_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.pose_fit_diagnostics import (
    collect_metric_model_diagnostics,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.model_input_cache import (
    MetricModelInputCache,
    RoiBounds,
)
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    refine_projected_head_border,
)
from scripts.aufgabe04.perception.stand_axis.model_stage_timing import ModelStageTiming
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
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
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.perception.stand_axis.qr_marker_validation import (
    QrMarkerEvidence,
    validate_qr_marker,
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
    qr_observations: tuple[DecodedQrObservation, ...] | None = None,
    input_cache: MetricModelInputCache | None = None,
    input_cache_roi: RoiBounds | None = None,
    current_head_proposal_corners: tuple[ImagePoint, ...] | None = None,
) -> tuple[StandAxisImageEstimate, StandAxisEdgeDebugArtifacts]:
    """Fit physical head angles from current pixels independently of QR.

    A candidate projection permits bounded QR-neutral head acquisition before
    any pose exists. QR/tracker seeds may position the legacy viewer search,
    but every physical angle is refitted from raw head rails/corners and neck,
    with independent head-only ambiguity and uncertainty gates. QR/joint fits
    remain separate diagnostics. An exact-image cache reuses preprocessing,
    never geometry. Legacy backside search can propose current corners when
    neutral acquisition fails, but physical angle admission still uses the
    same independent head-only checks. A separate structural/marker classifier
    may attach a backside-candidate label without changing that head angle.
    """

    timing = ModelStageTiming()
    camera = RectifiedCameraMatrix(
        float(camera_fx_px),
        float(camera_fy_px),
        float(camera_cx_px),
        float(camera_cy_px),
    )
    camera.validate()
    head_proposal = validate_current_head_proposal(
        current_head_proposal_corners, frame_shape=frame.shape,
    )
    if (input_cache is None) != (input_cache_roi is None):
        raise ValueError("model input cache and exact ROI must be supplied together")
    cached_inputs = None if input_cache is None else input_cache.begin(
        frame, roi=input_cache_roi, cv2=cv2, edge_preprocess=edge_preprocess,
        blur_kernel=blur_kernel, canny_low=canny_low, canny_high=canny_high,
        pose_hint_present=pose_hint is not None, qr_observations=qr_observations,
    )

    def preprocess_edges():
        return _canny_edges_from_frame(
            cv2, frame, edge_preprocess=edge_preprocess, blur_kernel=blur_kernel,
            canny_low=canny_low, canny_high=canny_high,
        )

    raw_edges = (preprocess_edges() if cached_inputs is None else
                 cached_inputs.compute("edge_preprocessing", preprocess_edges))
    timing.mark("edge_preprocessing")
    expected_geometry = (
        expected_head_center_u_px, expected_head_center_v_px, expected_head_height_px,
    )
    independent_head_requested = bool(
        model_profile.committable and model_profile.environment == "physical"
        and (head_proposal is not None or all(value is not None for value in expected_geometry))
    )
    head_result = None
    if independent_head_requested:
        if head_proposal is None:
            acquisition = acquire_head_proposal(
                cv2, frame, raw_edges=raw_edges,
                expected_head_center_u_px=expected_head_center_u_px,
                expected_head_center_v_px=expected_head_center_v_px,
                expected_head_height_px=expected_head_height_px,
            )
            head_proposal = None if acquisition.proposal is None else acquisition.proposal.corners
        timing.mark("independent_head_acquisition")
        if head_proposal is not None:
            head_result = fit_current_measured_head(
                cv2, raw_edges, model_profile=model_profile, camera=camera,
                proposal_corners=head_proposal,
                max_reprojection_rmse_px=max_reprojection_rmse_px,
                min_edge_height_px=min_edge_height_px,
            )
        timing.mark("independent_head_fit")
    if qr_observations is not None:
        if any(not isinstance(item, DecodedQrObservation) for item in qr_observations):
            raise ValueError("metric model QR observations have an invalid type")
        if len(qr_observations) > 1 and not independent_head_requested:
            estimate = replace(
                _unusable("model_qr_identity_ambiguous", source="model_seed"),
                evidence_state="unobservable", model_profile_sha256=model_profile.sha256,
                model_measurement_status=model_profile.measurement_status,
            )
            return estimate, StandAxisEdgeDebugArtifacts(
                edges=raw_edges, raw_edges=raw_edges, qr_detected=True,
                qr_marker_verified=True, qr_marker_reason="multiple_decoded_qr_identities",
                evidence_state="unobservable", model_reason=estimate.reason,
                model_profile_sha256=model_profile.sha256,
                model_measurement_status=model_profile.measurement_status,
                stage_timings_ms=timing.snapshot(),
            )
    # A tracked pose already constrains the narrow refinement corridors.  In
    # that state, avoid paying for the 4x acquisition pyramid on every frame;
    # a native QR observation may still refresh the seed.  Once tracking
    # expires, the full pyramid reacquires the stand.
    def acquire_qr_quad():
        return detect_qr_quad(
            cv2, frame,
            scales=((1.0,) if pose_hint is not None or independent_head_requested else (1.0, 2.0, 4.0)),
            allow_decode_fallback=(pose_hint is None and not independent_head_requested),
            **({"decoded_observations": qr_observations} if qr_observations is not None else {}),
        )

    qr_detection = (acquire_qr_quad() if cached_inputs is None else
                    cached_inputs.compute("qr_detection", acquire_qr_quad))
    timing.mark("qr_detection")
    qr_corners = None if qr_detection is None else qr_detection.corners
    qr_marker_detected = qr_corners is not None or bool(qr_observations)
    marker = (
        QrMarkerEvidence(True, "decoded_qr_identity")
        if qr_observations else validate_qr_marker(cv2, frame, qr_detection)
    )
    timing.mark("qr_marker_validation")
    def finish_independent_head(result):
        estimate, artifacts, head_pose = result
        estimate, artifacts = attach_independent_qr_diagnostics(
            cv2, estimate=estimate, debug=artifacts, head_pose=head_pose,
            qr_corners=qr_corners, marker_verified=marker.verified,
            model_profile=model_profile, camera=camera,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
        )
        timing.mark("independent_qr_diagnostics")
        artifacts = replace(
            artifacts, qr_detected=qr_marker_detected,
            qr_marker_verified=marker.verified,
            qr_marker_reason=("multiple_decoded_qr_identities"
                              if qr_observations and len(qr_observations) > 1 else marker.reason),
            qr_detection_scale=None if qr_detection is None else qr_detection.scale,
            stage_timings_ms=timing.snapshot(),
        )
        return classify_current_head_backside(
            estimate, artifacts, model_profile=model_profile, camera=camera,
            expected_center_u_px=expected_head_center_u_px,
            expected_center_v_px=expected_head_center_v_px,
            expected_height_px=expected_head_height_px,
        )
    if head_result is not None:
        return finish_independent_head(head_result)
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
    timing.mark("qr_seed_pose")
    pose_seed_source = (
        (f"qr_pyramid_{qr_detection.scale:g}x"
         if qr_detection.detector == "opencv_native"
         else f"qr_{qr_detection.detector}_{qr_detection.scale:g}x")
        if qr_seed is not None and qr_detection is not None
        else ("tracked_pose" if pose_hint is not None else "none")
    )
    proposal_can_seed_joint_fit = (
        head_proposal is not None and qr_corners is not None
        and marker.verified and model_profile.committable
    )
    if seed_pose is None and not proposal_can_seed_joint_fit:
        expected_geometry = (
            expected_head_center_u_px,
            expected_head_center_v_px,
            expected_head_height_px,
        )
        if (
            not qr_marker_detected
            and pose_hint is None
            and model_profile.committable
            and model_profile.environment == "physical"
            and all(value is not None for value in expected_geometry)
        ):
            estimate, artifacts = estimate_stand_axis_from_model_backside(
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
            timing.mark("backside_acquisition")
            if estimate.corners is not None and (estimate.usable or estimate.reason ==
                                                 "model_backside_neck_support_insufficient"):
                # Retain the alternate locator, not its former angle or
                # side authority. A rejected legacy stem anchor may still
                # locate corners. The independent physical-head fit rechecks
                # current raw borders, neck structure and planar ambiguity.
                return finish_independent_head(fit_current_measured_head(
                    cv2, raw_edges, model_profile=model_profile, camera=camera,
                    proposal_corners=estimate.corners,
                    max_reprojection_rmse_px=max_reprojection_rmse_px,
                    min_edge_height_px=min_edge_height_px,
                ))
            return estimate, replace(
                artifacts, stage_timings_ms=timing.snapshot(),
                qr_marker_verified=False, qr_marker_reason=marker.reason,
            )
        estimate = replace(
            _unusable(
                "model_qr_text_without_geometry"
                if qr_observations and qr_corners is None
                else "model_pose_seed_unavailable",
                source="model_seed",
            ),
            evidence_state="unobservable",
            model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status,
        )
        return estimate, StandAxisEdgeDebugArtifacts(
            edges=raw_edges,
            raw_edges=raw_edges,
            evidence_state="unobservable",
            model_profile_sha256=model_profile.sha256,
            qr_detected=qr_marker_detected,
            qr_marker_verified=marker.verified,
            qr_marker_reason=marker.reason,
            qr_detection_scale=(
                None if qr_detection is None else qr_detection.scale
            ),
            pose_seed_source=pose_seed_source,
            model_reason=estimate.reason,
            model_measurement_status=model_profile.measurement_status,
            stage_timings_ms=timing.snapshot(),
        )

    projected = (
        None if seed_pose is None else project_stand_model(cv2, model_profile, seed_pose, camera)
    )
    if model_profile.committable and model_profile.environment == "physical" and projected is not None:
        # Viewer/tracker callers may have no candidate projection. A seed can
        # position current raw-border searches, but it never selects an IPPE
        # branch or contributes an angle to this independent measurement.
        return finish_independent_head(fit_current_measured_head(
            cv2, raw_edges, model_profile=model_profile, camera=camera,
            proposal_corners=projected.head_corners,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
            min_edge_height_px=min_edge_height_px,
        ))
    border_seed = select_head_border_seed(
        model_profile=model_profile,
        projected_corners=None if projected is None else projected.head_corners,
        pose_reprojection_rmse_px=None if seed_pose is None else seed_pose.reprojection_rmse_px,
        # A proposal never replaces the QR-free branch's ordinary evidence
        # gates or supplies a directed pose by itself.
        current_head_proposal_corners=(head_proposal if proposal_can_seed_joint_fit else None),
    )
    corridor_half_width_px = border_seed.corridor_half_width_px
    timing.mark("seed_projection")
    refinement = refine_projected_head_border(
        cv2,
        raw_edges,
        border_seed.corners,
        corridor_half_width_px=corridor_half_width_px,
    )
    timing.mark("border_refinement")
    seed_rmse = (
        None
        if qr_pose is None or qr_pose.best is None
        else qr_pose.best.reprojection_rmse_px
    )
    seed_gap = None if qr_pose is None else qr_pose.ambiguity_gap_px
    base_artifacts = StandAxisEdgeDebugArtifacts(
        edges=raw_edges,
        raw_edges=raw_edges,
        face_mask=refinement.evidence_mask,
        predicted_corners=border_seed.corners,
        refined_corners=refinement.corners,
        candidate_corners=refinement.candidate_corners,
        corner_arm_support=refinement.corner_arm_support,
        model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        refinement_support_mean=(
            None if refinement.support is None else refinement.support.mean
        ),
        model_corridor_half_width_px=corridor_half_width_px,
        model_pose=seed_pose,
        qr_detected=qr_marker_detected,
        qr_marker_verified=marker.verified,
        qr_marker_reason=marker.reason,
        qr_detection_scale=(None if qr_detection is None else qr_detection.scale),
        pose_seed_source=(border_seed.source if proposal_can_seed_joint_fit else pose_seed_source),
        projected_landmarks=None if projected is None else dict(projected.landmarks),
    )
    if not refinement.accepted or refinement.corners is None:
        estimate = replace(
            _unusable(
                refinement.reason,
                corners=border_seed.corners,
                contour_area_px=_polygon_area(border_seed.corners),
                source="model_projection",
            ),
            evidence_state="predicted_only",
            model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status,
            pose_reprojection_rmse_px=seed_rmse,
            pose_ambiguity_gap_px=seed_gap,
        )
        diagnostics = collect_metric_model_diagnostics(
            cv2, profile=model_profile, camera=camera, head_corners=None,
            qr_corners=qr_corners, diagnostic_pose=seed_pose, qr_pose=qr_pose,
        )
        timing.mark("diagnostics")
        return estimate, replace(
            base_artifacts,
            evidence_state="predicted_only",
            pose_reprojection_rmse_px=seed_rmse,
            pose_ambiguity_gap_px=seed_gap,
            model_reason=estimate.reason,
            model_diagnostics=diagnostics,
            stage_timings_ms=timing.snapshot(),
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
    timing.mark("pose_fit")
    pose_rejected = (
        not refined_pose.accepted
        or selected_pose is None
        or (refined_axis_ambiguous and not ambiguity_resolved)
    )
    diagnostics = collect_metric_model_diagnostics(
        cv2, profile=model_profile, camera=camera,
        head_corners=refinement.corners, qr_corners=qr_corners,
        diagnostic_pose=(
            selected_pose if selected_pose is not None else
            (refined_pose.hypotheses[0] if refined_pose.hypotheses else None)
        ),
        qr_pose=qr_pose,
        head_pose=(refined_pose if pose_fit_source != "joint_qr_head" else None),
        diagnose_head_only=(pose_rejected and pose_fit_source == "joint_qr_head"),
        max_reprojection_rmse_px=max_reprojection_rmse_px,
    )
    contract = (
        classify_joint_geometry_contract(
            profile=model_profile, diagnostics=diagnostics,
            joint_reason=refined_pose.reason,
            joint_reprojection_rmse_px=(None if not refined_pose.hypotheses else
                                        refined_pose.hypotheses[0].reprojection_rmse_px),
            max_reprojection_rmse_px=max_reprojection_rmse_px,
            qr_marker_verified=marker.verified,
        )
        if pose_rejected and pose_fit_source == "joint_qr_head"
        else None
    )
    if contract is not None:
        diagnostics = replace(diagnostics, geometry_contract=contract)
    timing.mark("diagnostics")
    base_artifacts = replace(
        base_artifacts, model_diagnostics=diagnostics,
        model_pose_fit_source=pose_fit_source,
    )
    if pose_rejected:
        estimate = replace(
            _unusable(
                (
                    contract.reason if contract is not None else
                    ("planar_pose_axis_ambiguous"
                     if refined_axis_ambiguous else refined_pose.reason)
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
        return estimate, replace(
            base_artifacts,
            evidence_state="ambiguous",
            pose_reprojection_rmse_px=estimate.pose_reprojection_rmse_px,
            pose_ambiguity_gap_px=refined_pose.ambiguity_gap_px,
            model_reason=estimate.reason,
            # Only the accepted joint model may be a directed metric pose.
            # Independent QR/head fits remain diagnostic after disagreement.
            model_pose=None if contract is not None else base_artifacts.model_pose,
            stage_timings_ms=timing.snapshot(),
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
    refined_projection = project_stand_model(cv2, model_profile, best, camera)
    timing.mark("refined_projection")
    return estimate, replace(
        base_artifacts,
        rectangle_mask=_debug_rectangle_image(
            cv2, raw_edges.shape, refinement.corners
        ),
        rectangle_overlay=_debug_rectangle_overlay_image(
            cv2,
            raw_edges.shape,
            refinement.corners,
            refinement.evidence_mask,
        ),
        evidence_state="fresh_refined",
        pose_reprojection_rmse_px=best.reprojection_rmse_px,
        pose_ambiguity_gap_px=refined_pose.ambiguity_gap_px,
        model_pose=best,
        model_reason=estimate.reason,
        projected_landmarks=dict(refined_projection.landmarks),
        stage_timings_ms=timing.snapshot(),
    )
