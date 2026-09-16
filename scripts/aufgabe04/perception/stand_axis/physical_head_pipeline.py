"""QR-free acquisition and fitting of the measured physical head.

Candidate projections and a previous head pose locate current pixels only.
A viewer with neither can acquire a complete head directly from the image.
Identity and view-side evidence are attached separately by the caller.
"""

from dataclasses import asdict, replace
import math
import time

from scripts.aufgabe04.perception.stand_axis.geometry import _unusable
from scripts.aufgabe04.perception.stand_axis.current_head_border_binding import bind_selected_current_head
from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.head_model_quality import MEASURED_HEAD_AXIS_SOURCE
from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import classify_current_head_backside
from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts


def classify_physical_head_in_frame(
    estimate, artifacts, *, model_profile, camera,
    expected_center_u_px, expected_center_v_px, expected_height_px,
):
    """Classify the complete current tracked head within its selected crop.

    A tracked crop already follows an associated head, so its current verified
    borders locate the side-classification window, just as a current registered
    proposal does. The original map projection remains the observer's separate
    association bound. Neither historical pixels nor the pose hint supply side
    evidence; current marker checks and a new complete-head/scan proof still
    govern backside use.
    """
    diagnostics = artifacts.head_acquisition_diagnostics or {}
    if (diagnostics.get("source") == "candidate_tracked_head_search"
            and admit_measured_head_model(estimate=estimate, debug=artifacts,
                yaw_rad=(math.radians(estimate.yaw_deg)
                         if type(estimate.yaw_deg) in (int, float) else math.nan)).accepted):
        expected_center_u_px = sum(p.u_px for p in estimate.corners) / 4
        expected_center_v_px = sum(p.v_px for p in estimate.corners) / 4
        artifacts = replace(artifacts, head_acquisition_diagnostics={
            **diagnostics, "side_projection_source": "current_verified_head_pixels",
            "side_projection_center_px": (expected_center_u_px, expected_center_v_px),
            "original_candidate_association_required": True,
        })
    return classify_current_head_backside(
        estimate, artifacts, model_profile=model_profile, camera=camera,
        expected_center_u_px=expected_center_u_px,
        expected_center_v_px=expected_center_v_px, expected_height_px=expected_height_px)


def fit_physical_head_in_frame(
    cv2, frame, raw_edges, *, model_profile, camera, timing,
    current_head_proposal_corners=None, current_head_proposal_verified=False, pose_hint=None,
    current_head_refinement=None,
    expected_head_center_u_px=None, expected_head_center_v_px=None,
    expected_head_height_px=None, max_reprojection_rmse_px=2., min_edge_height_px=8.,
    deadline_monotonic_sec=None,
):
    """Return a current head fit without consulting any QR observation.

    A named mission candidate may use its associated prior to locate current
    borders inside the caller's bounded crop. External projection/association
    gates still apply. Only an unprojected viewer may use full-image cold search.
    A missed search hint may reacquire within the same bounds and deadline.
    Verified pose ambiguity never retries another locator or borrows an angle.
    """
    expected = (expected_head_center_u_px, expected_head_center_v_px, expected_head_height_px)
    diagnostics = {"policy": "current_head_pixels_only", "qr_used_for_geometry": False,
                   "neck_required": False, "source": None, "acquisition": None,
                   "tracked_fit_reason": None}

    def fitted(corners, *, selected_current_border=False, boundary=None):
        if expired():
            return unavailable("head_acquisition_deadline_exceeded")
        result = fit_current_measured_head(
            cv2, raw_edges, model_profile=model_profile, camera=camera, proposal_corners=corners,
            max_reprojection_rmse_px=max_reprojection_rmse_px, min_edge_height_px=min_edge_height_px,
            frame_bgr=frame, current_head_refinement=boundary)
        diagnostics["current_boundary_reused"] = bool(
            boundary is not None and result[1].head_outer_recovery is not None)
        if selected_current_border:
            result, diagnostics["selected_border_binding"] = bind_selected_current_head(
                result, corners, raw_edges=raw_edges, frame_bgr=frame)
        timing.mark("independent_head_fit")
        if expired():
            return unavailable("head_acquisition_deadline_exceeded")
        return result

    def expired():
        return deadline_monotonic_sec is not None and time.monotonic() >= deadline_monotonic_sec

    def finished(result):
        estimate, artifacts, pose = result
        return estimate, replace(artifacts, head_acquisition_diagnostics=dict(diagnostics)), pose

    def unavailable(reason):
        estimate = replace(_unusable(reason, source=MEASURED_HEAD_AXIS_SOURCE),
            evidence_state="unobservable", model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status)
        return finished((estimate, StandAxisEdgeDebugArtifacts(
            edges=raw_edges, raw_edges=raw_edges, model_reason=reason,
            model_pose_fit_source=MEASURED_HEAD_AXIS_SOURCE, evidence_state="unobservable",
            model_profile_sha256=model_profile.sha256,
            model_measurement_status=model_profile.measurement_status), None))

    if expired():
        return unavailable("head_acquisition_deadline_exceeded")
    if current_head_refinement is not None and current_head_proposal_corners is None:
        return unavailable("current_head_refinement_invalid")
    if current_head_proposal_corners is not None:
        diagnostics["source"] = "current_candidate_proposal"
        return finished(fitted(current_head_proposal_corners,
            selected_current_border=current_head_proposal_verified is True,
            boundary=current_head_refinement))
    if any(value is not None for value in expected) and not all(value is not None for value in expected):
        return unavailable("head_candidate_projection_incomplete")
    if all(value is not None for value in expected) and (
            not all(type(value) in (int, float) and math.isfinite(value) for value in expected)
            or expected_head_height_px <= 0):
        return unavailable("head_candidate_projection_invalid")
    acquisition_expected = expected
    if pose_hint is not None:
        diagnostics["source"] = ("candidate_tracked_head_search"
                                 if all(value is not None for value in expected)
                                 else "tracked_head_search")
        projected = project_stand_model(cv2, model_profile, pose_hint, camera)
        tracked = fitted(projected.head_corners)
        diagnostics["tracked_fit_reason"] = tracked[0].reason
        quality = tracked[1].head_model_quality
        if quality is not None and quality.outer_border_verified:
            # Current verified borders retain their current angle decision;
            # another search must not select a more convenient hypothesis.
            return finished(tracked)
        if not all(value is not None for value in expected):
            corners = projected.head_corners
            acquisition_expected = (
                sum(p.u_px for p in corners) / 4., sum(p.v_px for p in corners) / 4.,
                (math.dist((corners[0].u_px, corners[0].v_px), (corners[3].u_px, corners[3].v_px))
                 + math.dist((corners[1].u_px, corners[1].v_px), (corners[2].u_px, corners[2].v_px))) / 2.)
    refinement_out = {}
    if all(value is not None for value in acquisition_expected):
        diagnostics["source"] = ("candidate_projection" if all(value is not None for value in expected)
                                 else "tracked_head_reacquisition")
        acquisition = acquire_cold_head_proposal(
            cv2, frame, raw_edges=raw_edges,
            model_profile=model_profile,
            refinement_out=refinement_out,
            expected_head_center_u_px=acquisition_expected[0],
            expected_head_center_v_px=acquisition_expected[1],
            expected_head_height_px=acquisition_expected[2],
            expected_head_height_tolerance_ratio=.30, max_center_offset_ratio=1.5,
            deadline_monotonic_sec=deadline_monotonic_sec)
    else:
        diagnostics["source"] = "cold_current_head_search"
        acquisition = acquire_cold_head_proposal(cv2, frame, raw_edges=raw_edges,
                                               model_profile=model_profile,
                                               refinement_out=refinement_out,
                                               deadline_monotonic_sec=deadline_monotonic_sec)
    diagnostics["acquisition"] = asdict(acquisition)
    timing.mark("independent_head_acquisition")
    if expired() or acquisition.reason == "head_acquisition_deadline_exceeded":
        return unavailable("head_acquisition_deadline_exceeded")
    if acquisition.proposal is None:
        # Keep the producer detail alongside the stable observer-facing reason.
        return unavailable("head_proposal_ambiguous" if "ambiguous" in acquisition.reason
                           else "model_current_head_border_unavailable")
    return finished(fitted(acquisition.proposal.corners, selected_current_border=True,
                           boundary=refinement_out.get("selected")))
