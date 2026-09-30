"""Recomputable scan-boundary sampling advice for a bounded stationary turn.

This requests better measurements, never a head normal or camera centering.
The child owns current-target, odometry, clearance and single-use motion gates.
"""
from dataclasses import dataclass
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
from scripts.aufgabe04.navigation.approach.lidar_head_geometry import HEAD_MODEL
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import _scan_surface_analysis
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.lidar_visibility_evidence import _receipt_from_hashed_payload
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import HASH_FIELD as VIEW_HASH
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot, candidate_snapshot_sha256


PURPOSE = "candidate_lidar_sampling"
MAX_TURN_RAD = math.radians(25.)
MAX_TOTAL_TRAVEL_RAD = math.radians(26.)


def predicted_head_support(*, distance_m, incidence_rad, angular_step_rad):
    """Noise-free broad-face sampling estimate, not a success probability."""
    if not all(math.isfinite(v) for v in (distance_m, incidence_rad, angular_step_rad)) or distance_m <= 0 or not 0 < abs(angular_step_rad) < math.pi/4:
        raise ValueError("invalid LiDAR support prediction geometry")
    width = HEAD_MODEL.width_m * max(0., math.cos(min(math.pi/2, abs(incidence_rad))))
    intervals = 2*math.atan2(width/2, distance_m) / abs(angular_step_rad)
    return {"angular_width_rad": 2*math.atan2(width/2, distance_m),
            "expected_return_count": intervals, "minimum_phase_return_count": math.floor(intervals),
            "noise_free": True, "physical_success_probability": None}


def _point_odom(point, transform):
    x, y = point.x_m-transform.x_m, point.y_m-transform.y_m
    c, s = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    return Pose2D(c*x+s*y, -s*x+c*y)


def _source_geometry(source_view_path, snapshot_path):
    from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
    view = load_content_hashed_json(Path(source_view_path), hash_field=VIEW_HASH)
    snapshot = load_candidate_snapshot(Path(snapshot_path))
    if (view["candidate_snapshot_sha256"] != candidate_snapshot_sha256(snapshot)
            or view["map_bundle_sha256"] != snapshot.map_bundle_sha256):
        raise ValueError("sampling source snapshot/map binding mismatch")
    candidate = snapshot.candidate_for(view["candidate_uid"])
    if candidate is None:
        raise ValueError("sampling source candidate unavailable")
    frame = CandidatePlanningFrame.from_evidence(view["planning_frame"])
    if frame.map_frame != snapshot.planning_frame:
        raise ValueError("sampling source frame mismatch")
    receipts = tuple(_receipt_from_hashed_payload(r) for r in view["receipts"])
    if not receipts or any(r.scan_metadata is None for r in receipts):
        raise ValueError("sampling source lacks original scan metadata")
    if any(r.scan_metadata.scan_topology_profile != "full_rotation" for r in receipts):
        raise ValueError("sampling source requires declared full-rotation scanner")
    from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD as CAPTURE_HASH, head_capture_payload
    from scripts.aufgabe04.perception.lidar_visibility_evidence import visibility_receipts_sha256
    capture = load_content_hashed_json(Path(view["capture_path"]), hash_field=CAPTURE_HASH)
    expected_capture = head_capture_payload(capture["scans"], tour_id=view["viewpoint_id"],
        odom_frame=frame.odom_frame, base_frame=capture["base_frame"], scan_frame=capture["scan_frame"],
        captured_at_unix_sec=capture["captured_at_unix_sec"])
    if (capture != expected_capture or payload_sha256(capture) != view["capture_sha256"]
            or visibility_receipts_sha256(receipts) != view["receipt_set_sha256"]
            or len(receipts) != len(capture["scans"])
            or view["latest_base_pose_odom"] != capture["scans"][-1]["base_pose_odom"]
            or view["pose_stamp_sec"] != capture["scans"][-1]["stamp_sec"]):
        raise ValueError("sampling source capture binding mismatch")
    for receipt, raw in zip(receipts, capture["scans"]):
        if (receipt.scan_stamp_sec != raw["stamp_sec"] or receipt.viewpoint_id != view["viewpoint_id"]
                or receipt.survey_id != view["survey_id"]
                or receipt.map_bundle_sha256 != snapshot.map_bundle_sha256
                or receipt.observer_config_sha256 != payload_sha256(view["observer_config"])
                or receipt.frame_provenance.odom_frame != frame.odom_frame
                or receipt.frame_provenance.map_frame != frame.map_frame
                or receipt.frame_provenance.map_from_odom != frame.map_from_odom
                or receipt.ranges_m != tuple(raw["ranges"])
                or receipt.angle_min_rad != raw["angle_min"] or receipt.angle_increment_rad != raw["angle_increment"]
                or receipt.scan_metadata.to_mapping() != raw["scan_metadata"]
                or receipt.scan_frame != capture["scan_frame"]
                or receipt.frame_provenance.source_evidence_id != view["capture_sha256"]
                or receipt.frame_provenance.canonical_scan_pose_odom != Pose2D(**raw["scan_pose_odom"])):
            raise ValueError("sampling receipt differs from original source scan")
    config = view["observer_config"]
    if (config["capture_sha256"] != view["capture_sha256"]
            or config["candidate_uid"] != view["candidate_uid"]
            or config["candidate_snapshot_sha256"] != view["candidate_snapshot_sha256"]
            or config["planning_frame"] != view["planning_frame"]
            or payload_sha256(config) != view["observer_config_sha256"]):
        raise ValueError("sampling source observer binding mismatch")
    center = _point_odom(candidate.geometry, frame.map_from_odom)
    radius = candidate.geometry.radius_m+candidate.geometry.uncertainty_m
    if not 0 < radius <= .15:
        raise ValueError("sampling candidate envelope is too wide")
    others = []
    for other in snapshot.candidates:
        if other.candidate_uid != candidate.candidate_uid:
            p = _point_odom(other.geometry, frame.map_from_odom)
            others.append((p.x_m, p.y_m, other.geometry.radius_m+other.geometry.uncertainty_m))
    diagnostics = [_scan_surface_analysis(r, center, radius, others)[1] for r in receipts]
    if any(d["reason"] == "competing_candidate" for d in diagnostics):
        raise ValueError("sampling source contains competing candidates")
    if sum(d["boundary_fragmented"] for d in diagnostics) < math.ceil(len(receipts)/2):
        raise ValueError("sampling source is not persistently boundary fragmented")
    last = receipts[-1]
    base = Pose2D(**view["latest_base_pose_odom"])
    scan = last.frame_provenance.canonical_scan_pose_odom
    dx, dy = scan.x_m-base.x_m, scan.y_m-base.y_m
    c, s = math.cos(base.yaw_rad), math.sin(base.yaw_rad)
    extrinsic = Pose2D(c*dx+s*dy, -s*dx+c*dy, math.remainder(scan.yaw_rad-base.yaw_rad, math.tau))
    return view, snapshot, receipts, center, radius, others, base, extrinsic


def select_sampling_yaw(*, base, scan_pose_robot, center, radius, receipts):
    """Place the entire candidate envelope inside all observed scan intervals."""
    def clear(turn):
        yaw = base.yaw_rad+turn
        c, s = math.cos(yaw), math.sin(yaw)
        x = base.x_m+c*scan_pose_robot.x_m-s*scan_pose_robot.y_m
        y = base.y_m+s*scan_pose_robot.x_m+c*scan_pose_robot.y_m
        distance = math.hypot(center.x_m-x, center.y_m-y)
        if distance <= radius:
            return False
        bearing = math.atan2(center.y_m-y, center.x_m-x)
        for receipt in receipts:
            step = abs(receipt.angle_increment_rad)
            sign = math.copysign(1., receipt.angle_increment_rad)
            span = min((len(receipt.ranges_m)-1)*step,
                       sign*(receipt.scan_metadata.angle_max_rad-receipt.angle_min_rad))
            if not math.radians(350) <= span < math.tau:
                return False
            margin = math.asin(radius/distance)+2*step+math.radians(2.)
            offset = (sign*(bearing-yaw-scan_pose_robot.yaw_rad-receipt.angle_min_rad)) % math.tau
            if not margin <= offset <= span-margin:
                return False
        return True
    if clear(0.):
        return None
    choices = [math.radians(sign*degree/2) for degree in range(30, 51) for sign in (-1, 1)]
    return next((turn for turn in choices if clear(turn)), None)


def build_sampling_advisory(*, source_view_path, snapshot_path, session_id,
                            robot_profile_sha256, calibration_profile_sha256,
                            base_frame, scan_frame, now_sec):
    view, snapshot, receipts, center, radius, others, base, extrinsic = _source_geometry(source_view_path, snapshot_path)
    if type(now_sec) not in (int, float) or not math.isfinite(now_sec):
        raise ValueError("sampling creation time must be finite")
    from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import HASH_FIELD as CAPTURE_HASH
    capture = load_content_hashed_json(Path(view["capture_path"]), hash_field=CAPTURE_HASH)
    if capture["base_frame"] != base_frame or capture["scan_frame"] != scan_frame:
        raise ValueError("sampling source profile frame mismatch")
    if not 0 <= now_sec-receipts[-1].scan_stamp_sec <= .5:
        raise ValueError("sampling source is not current")
    turn = select_sampling_yaw(base=base, scan_pose_robot=extrinsic, center=center,
                               radius=radius, receipts=receipts)
    if turn is None:
        raise ValueError("no useful bounded sampling yaw")
    if any(r.scan_frame != scan_frame or r.frame_provenance.odom_frame != view["planning_frame"]["odom_frame"]
           for r in receipts):
        raise ValueError("sampling source scanner frame mismatch")
    from dataclasses import asdict
    return {"schema_version": 1, "purpose": PURPOSE, "candidate_uid": view["candidate_uid"],
        "session_id": session_id, "robot_profile_sha256": robot_profile_sha256,
        "calibration_profile_sha256": calibration_profile_sha256,
        "source_view_path": str(Path(source_view_path).resolve()), "source_view_sha256": payload_sha256(view),
        "candidate_snapshot_path": str(Path(snapshot_path).resolve()),
        "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
        "created_at_sec": now_sec, "scan_stamp_sec": receipts[-1].scan_stamp_sec,
        "odom_stamp_sec": view["pose_stamp_sec"], "requested_yaw_rad": turn,
        "anchor_odom_pose": asdict(base), "scan_pose_robot": asdict(extrinsic),
        "target_center_odom": {"x_m": center.x_m, "y_m": center.y_m},
        "candidate_envelope_radius_m": radius, "competing_candidate_envelopes": [list(x) for x in others],
        "base_frame": base_frame, "scan_frame": scan_frame,
        "odom_frame": view["planning_frame"]["odom_frame"],
        "motion_authorized": False, "stand_axis_authorized": False,
        "head_alignment_verified": False, "camera_centered": False}


@dataclass(frozen=True)
class SamplingAdvice:
    requested_yaw_rad: float
    arrival_recovery: bool = False


def validate_sampling_advisory(payload, *, candidate_uid=None, session_id=None,
                               robot_profile_sha256=None, calibration_profile_sha256=None,
                               now_sec=None, **unused):
    if not isinstance(payload, dict) or payload.get("purpose") != PURPOSE:
        raise ValueError("invalid LiDAR sampling advisory")
    expected = build_sampling_advisory(source_view_path=payload["source_view_path"],
        snapshot_path=payload["candidate_snapshot_path"], session_id=payload["session_id"],
        robot_profile_sha256=payload["robot_profile_sha256"],
        calibration_profile_sha256=payload["calibration_profile_sha256"],
        base_frame=payload["base_frame"], scan_frame=payload["scan_frame"], now_sec=payload["created_at_sec"])
    if payload != expected:
        raise ValueError("LiDAR sampling advisory derivation changed")
    for value, field in ((candidate_uid, "candidate_uid"), (session_id, "session_id"),
                         (robot_profile_sha256, "robot_profile_sha256"),
                         (calibration_profile_sha256, "calibration_profile_sha256")):
        if value is not None and payload[field] != value:
            raise ValueError(f"LiDAR sampling {field} mismatch")
    if now_sec is not None and not 0 <= now_sec-payload["created_at_sec"] <= 5.:
        raise ValueError("LiDAR sampling advisory expired")
    return SamplingAdvice(payload["requested_yaw_rad"])


def fresh_sampling_target(scan, advisory, now_sec, *, base_pose=None):
    """Require compact real current support inside the same frozen envelope."""
    from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import stamp_seconds
    if scan.header.frame_id != advisory["scan_frame"] or not 0 <= now_sec-stamp_seconds(scan.header.stamp) <= .5:
        return None
    base = base_pose if base_pose is not None else Pose2D(**advisory["anchor_odom_pose"])
    relative = Pose2D(**advisory["scan_pose_robot"])
    c, s = math.cos(base.yaw_rad), math.sin(base.yaw_rad)
    x, y = base.x_m+c*relative.x_m-s*relative.y_m, base.y_m+s*relative.x_m+c*relative.y_m
    center, radius = advisory["target_center_odom"], advisory["candidate_envelope_radius_m"]
    points = []
    for index, distance in enumerate(scan.ranges):
        if not math.isfinite(distance) or not max(0., scan.range_min) < distance <= scan.range_max:
            continue
        angle = base.yaw_rad+relative.yaw_rad+scan.angle_min+index*scan.angle_increment
        p = x+distance*math.cos(angle), y+distance*math.sin(angle)
        if math.hypot(p[0]-center["x_m"], p[1]-center["y_m"]) > radius:
            continue
        if any(math.hypot(p[0]-ox, p[1]-oy) <= r for ox, oy, r in advisory["competing_candidate_envelopes"]):
            return None
        points.append(p)
    if len(points) < 3 or any(math.dist(a, b) > HEAD_MODEL.width_m+HEAD_MODEL.tolerance_m+2*HEAD_MODEL.point_noise_m
                              for a in points for b in points):
        return None
    return {"real_return_count": len(points), "candidate_uid": advisory["candidate_uid"],
            "head_alignment_verified": False, "motion_authorized": False}
