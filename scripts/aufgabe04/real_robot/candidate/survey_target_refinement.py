"""Optional stopped-point refinement before observing a retained survey target."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.artifacts.current_target_estimate import planning_target_geometry
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
    CandidateLidarCaptureUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.target_admission import (
    _require_retained_survey_geometry, bind_current_lidar_target, evaluate_target,
)
from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import TourScanCaptureError


HASH_FIELD = "survey_target_refinement_sha256"


def refine_survey_observation_target(frame, *, effects, output_dir, attempt_index):
    """Try one cohort; missing or unsuitable geometry leaves the survey view intact.

    The caller owns the first-arrival restriction. This helper grants no motion
    or alignment authority, and malformed source/configuration evidence remains
    an error rather than being treated as missing visibility.
    """
    from .current_lidar_targets import capture_current_lidar_targets

    if (type(attempt_index) is not int or attempt_index != 0
            or getattr(frame, "retained_survey_target", None) is None
            or getattr(frame, "retained_backside_axis_path", None) is not None):
        raise ValueError("survey refinement requires the first survey-only observation")
    _require_retained_survey_geometry(frame)
    root = Path(output_dir)
    path = root / "survey_target_refinement.json"
    if root.is_symlink() or path.exists() or path.is_symlink():
        raise ValueError("survey refinement output must be fresh")
    uid = frame.candidate.candidate_uid
    binding = frame.retained_survey_target
    evidence = {
        "schema_version": 1, "candidate_uid": uid,
        "observation_attempt_index": attempt_index,
        "maximum_cohort_count": 1, "accepted": False,
        "retained_survey_target_path": str(binding.evidence_path),
        "retained_survey_target_sha256": binding.evidence_sha256,
        "current_lidar_support": None, "candidate_target_admission": None,
        "motion_authorized": False, "stand_axis_authorized": False,
        "head_alignment_verified": False, "camera_centered": False,
        "keepouts_changed": False,
    }

    def finish(result, reason):
        evidence["reason"] = reason
        digest = write_content_hashed_json(path, evidence, hash_field=HASH_FIELD)
        return result, {**evidence, "evidence_path": str(path), "evidence_sha256": digest}

    try:
        estimates, support = capture_current_lidar_targets(
            frame.config, effects, frame.planning_frame, {uid}, root / "current_lidar_support")
    except (CandidateLidarCaptureUnavailableError, TourScanCaptureError) as exc:
        if isinstance(exc, TourScanCaptureError) and not exc.retryable:
            raise
        evidence["capture_unavailable"] = {
            "exception_type": type(exc).__name__, "detail": str(exc),
            "reason_code": getattr(exc, "reason_code", "fresh_lidar_cohort_unavailable"),
            "diagnostics": getattr(exc, "diagnostics", {}),
        }
        return finish(frame, "fresh_lidar_refinement_unavailable")

    decision = support["candidate_decisions"][uid]
    if (set(estimates)-{uid} or decision["candidate_uid"] != uid
            or decision["accepted"] is not (uid in estimates)):
        raise ValueError("survey refinement current-target identity/decision changed")
    evidence["current_lidar_support"] = support
    if uid not in estimates:
        return finish(frame, "fresh_lidar_refinement_not_supported")

    geometry = planning_target_geometry(frame.candidate, estimates[uid])
    admission = evaluate_target(frame.config, frame.candidate, target_geometry=geometry)
    evidence["candidate_target_admission"] = admission.to_evidence()
    if not admission.accepted:
        return finish(frame, "fresh_lidar_refinement_target_rejected")

    refined = bind_current_lidar_target(replace(frame,
        camera_target_geometry=None, camera_target_geometry_evidence_path=None,
        camera_alignment=None, current_lidar_target_path=None,
        retained_lidar_target=None, retained_survey_target=None),
        evidence_path=Path(support["evidence_path"]))
    evidence["accepted"] = True
    return finish(refined, "fresh_lidar_refinement_retained")
