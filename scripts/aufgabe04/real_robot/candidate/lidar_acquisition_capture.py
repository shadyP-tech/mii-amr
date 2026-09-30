"""Candidate-bound adapter for passive, exact-time stationary scan cohorts.

Local scans are separate from the completed survey. A cohort proves stationary
geometry for an advisory view; it does not certify a head axis or permit motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import time
from typing import Callable, Mapping

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan, coverage_survey_plan_sha256, validate_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    LidarVisibilityReceipt, lidar_visibility_receipt_from_scan, visibility_receipts_sha256,
)
from scripts.aufgabe04.perception.lidar_visibility_frames import LidarVisibilityFrameProvenance
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import (
    HASH_FIELD as COHORT_HASH_FIELD, MAX_SCAN_AGE_SEC, FUTURE_TOLERANCE_SEC,
    capture_payload, finite,
)


HASH_FIELD = "candidate_lidar_view_sha256"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CandidateLidarCaptureUnavailableError(ValueError):
    """A valid cohort expired or predates this stopped observation epoch."""


@dataclass(frozen=True)
class CandidateLidarCaptureRequest:
    plan: CoverageSurveyPlan
    candidate_snapshot_sha256: str
    candidate_uid: str
    viewpoint_id: str
    output_dir: Path
    observation_not_before_sec: float
    planning_frame: CandidatePlanningFrame
    base_frame: str
    scan_frame: str
    scan_topic: str


@dataclass(frozen=True)
class CandidateLidarView:
    candidate_uid: str
    candidate_snapshot_sha256: str
    viewpoint_id: str
    receipts: tuple[LidarVisibilityReceipt, ...]
    latest_base_pose_odom: Pose2D
    pose_stamp_sec: float
    captured_at_unix_sec: float
    evidence_path: Path
    evidence_sha256: str
    capture_path: Path
    capture_sha256: str
    mount_evidence: tuple[Mapping[str, object], ...] = ()


def _regular(path, *, root=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("candidate LiDAR source must be a regular non-symlink file")
    if root is not None and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("candidate LiDAR source is outside the capture directory")
    return path


def capture_candidate_lidar_view(
    request: CandidateLidarCaptureRequest,
    *,
    capture_cohort: Callable[[CandidateLidarCaptureRequest], Path],
    clock: Callable[[], float] = time.time,
) -> CandidateLidarView:
    """Revalidate all three source scans and bind them to one candidate.

    ``capture_cohort`` returns the hashed artifact from ``capture_tour_scan``.
    Its tour-named format is reused with separate candidate-local lineage;
    no viewpoints are added to the frozen survey.
    """
    if not isinstance(request, CandidateLidarCaptureRequest):
        raise TypeError("candidate LiDAR capture requires a typed request")
    validate_coverage_survey_plan(request.plan)
    for value in (request.candidate_uid, request.viewpoint_id):
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise ValueError("candidate LiDAR identity must be a safe identifier")
    if (not isinstance(request.candidate_snapshot_sha256, str)
            or _SHA256.fullmatch(request.candidate_snapshot_sha256) is None):
        raise ValueError("candidate LiDAR snapshot binding must be SHA-256")
    frame = request.planning_frame
    if not isinstance(frame, CandidatePlanningFrame) or frame.map_frame != request.plan.planning_frame:
        raise ValueError("candidate LiDAR planning frame differs from plan")
    frame_evidence = frame.to_evidence()
    floor = finite(request.observation_not_before_sec, "observation timestamp floor")
    started = finite(clock(), "capture start")
    if floor < 0 or floor > started + FUTURE_TOLERANCE_SEC:
        raise ValueError("candidate LiDAR observation timestamp floor is invalid")
    root = Path(request.output_dir)
    if root.is_symlink():
        raise ValueError("candidate LiDAR output directory must not be a symlink")
    evidence_path = root / "candidate_lidar_view.json"
    if evidence_path.exists() or evidence_path.is_symlink():
        raise ValueError("candidate LiDAR view output must be fresh")
    capture_path = _regular(capture_cohort(request), root=root)
    now = finite(clock(), "capture end")
    if not 0 <= now - started <= 30.0:
        raise ValueError("candidate LiDAR capture exceeded the bounded epoch")
    raw = load_content_hashed_json(capture_path, hash_field=COHORT_HASH_FIELD)
    capture_hash = payload_sha256(raw)
    expected = {
        "schema_version": 1, "artifact_kind": "stored_pose_tour_scan_capture",
        "tour_id": request.viewpoint_id, "odom_frame": frame.odom_frame,
        "base_frame": request.base_frame, "scan_frame": request.scan_frame,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("candidate LiDAR cohort identity differs from request")
    validated = capture_payload(raw["scans"], tour_id=request.viewpoint_id,
        odom_frame=frame.odom_frame, base_frame=request.base_frame, scan_frame=request.scan_frame,
        captured_at_unix_sec=raw["captured_at_unix_sec"])
    if validated != raw:
        raise ValueError("candidate LiDAR cohort has unsupported fields")
    scans = raw["scans"]
    captured = finite(raw["captured_at_unix_sec"], "capture timestamp")
    if captured > now + FUTURE_TOLERANCE_SEC or now-scans[-1]["stamp_sec"] < -FUTURE_TOLERANCE_SEC:
        raise ValueError("candidate LiDAR cohort is future-dated")
    if captured < floor or now-scans[-1]["stamp_sec"] > MAX_SCAN_AGE_SEC:
        raise CandidateLidarCaptureUnavailableError("candidate LiDAR cohort is stale at capture completion")
    if any(scan["stamp_sec"] < floor - 1e-6 for scan in scans):
        raise CandidateLidarCaptureUnavailableError(
            "candidate LiDAR cohort contains a scan before the observation floor")
    transform = frame.map_from_odom
    cosine, sine = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    observer_config = {
        "schema_version": 1, "source_kind": "candidate_stationary_scan_cohort",
        "capture_sha256": capture_hash, "planning_frame": frame_evidence,
        "candidate_snapshot_sha256": request.candidate_snapshot_sha256,
        "candidate_uid": request.candidate_uid, "scan_topic": request.scan_topic,
    }
    config_hash = payload_sha256(observer_config)
    receipts = []
    for index, scan in enumerate(scans):
        pose = Pose2D(**scan["scan_pose_odom"])
        map_pose = Pose2D(cosine*pose.x_m-sine*pose.y_m+transform.x_m,
                         sine*pose.x_m+cosine*pose.y_m+transform.y_m,
                         math.remainder(pose.yaw_rad+transform.yaw_rad, math.tau))
        receipts.append(lidar_visibility_receipt_from_scan(
            receipt_id=f"local_{index}_{capture_hash[:24]}", survey_id=request.plan.survey_id,
            viewpoint_id=request.viewpoint_id, planning_frame=frame.map_frame,
            scan_frame=request.scan_frame, scan_topic=request.scan_topic,
            map_bundle_sha256=request.plan.map_bundle_sha256, observer_config_sha256=config_hash,
            scan_stamp_sec=scan["stamp_sec"], pose_stamp_sec=scan["scan_pose_stamp_sec"],
            observer_clock_sec=scan["received_at_unix_sec"], scan_pose_map=map_pose,
            angle_min_rad=scan["angle_min"], angle_increment_rad=scan["angle_increment"],
            range_min_m=scan["range_min"], range_max_m=scan["range_max"], ranges_m=scan["ranges"],
            frame_provenance=LidarVisibilityFrameProvenance(
                frame.map_frame, frame.odom_frame, transform, pose, capture_hash),
        ))
    receipts = tuple(receipts)
    base_pose = Pose2D(**scans[-1]["base_pose_odom"])
    mount_evidence = tuple({**scan["head_plane_mount"], "stamp_sec": scan["stamp_sec"]}
                          for scan in scans if scan.get("head_plane_mount") is not None)
    payload = {
        "schema_version": 1, "artifact_kind": "candidate_local_lidar_view",
        "candidate_uid": request.candidate_uid,
        "candidate_snapshot_sha256": request.candidate_snapshot_sha256,
        "survey_id": request.plan.survey_id, "viewpoint_id": request.viewpoint_id,
        "coverage_plan_sha256": coverage_survey_plan_sha256(request.plan),
        "map_bundle_sha256": request.plan.map_bundle_sha256, "planning_frame": frame_evidence,
        "observation_not_before_sec": floor, "captured_at_unix_sec": captured,
        "capture_path": str(capture_path), "capture_sha256": capture_hash,
        "source_evidence_kind": "stationary_scan_cohort_sha256",
        "observer_config": observer_config, "observer_config_sha256": config_hash,
        "receipts": [receipt.to_evidence_dict() for receipt in receipts],
        "receipt_set_sha256": visibility_receipts_sha256(receipts),
        "latest_base_pose_odom": scans[-1]["base_pose_odom"], "pose_stamp_sec": scans[-1]["stamp_sec"],
        "mount_evidence": list(mount_evidence),
        "scan_count": len(receipts), "motion_authorized": False, "stand_axis_authorized": False,
    }
    if payload_sha256(load_content_hashed_json(capture_path, hash_field=COHORT_HASH_FIELD)) != capture_hash:
        raise ValueError("candidate LiDAR source changed during validation")
    digest = write_content_hashed_json(evidence_path, payload, hash_field=HASH_FIELD)
    return CandidateLidarView(request.candidate_uid, request.candidate_snapshot_sha256,
        request.viewpoint_id, receipts, base_pose, scans[-1]["stamp_sec"], captured,
        evidence_path, digest, capture_path, capture_hash, mount_evidence)
