"""Fresh stopped LiDAR support for camera targets, without changing keepouts.

Whole scan clusters are formed before candidate association. A historical
candidate envelope must never cut a wall into an apparently compact stand.
The measured surface centroid is an uncertain local target, not a head axis
or a decoded identity. Source artifacts remain replayable and immutable.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.lidar_head_observability import verify_lidar_head_observability
from scripts.aufgabe04.perception.lidar_stand_detector import cluster_scan_points, scan_points_from_ranges
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    _receipt_from_hashed_payload, validate_lidar_visibility_receipt, visibility_receipts_sha256,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
    HASH_FIELD as VIEW_HASH_FIELD, CandidateLidarCaptureRequest, CandidateLidarView,
)
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import (
    HASH_FIELD as CAPTURE_HASH_FIELD, SCAN_COUNT, MAX_WINDOW_SEC, head_capture_payload,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256, validate_candidate_snapshot,
)

POLICY = "current_stopped_lidar_surface"
HASH_FIELD = "current_lidar_targets_sha256"
MAX_LATEST_AGE_SEC = 1.0
MAX_CENTER_SCATTER_M = .03
MIN_SUPPORTED_FRACTION = .75


def _odom_point(x, y, transform):
    dx, dy = x-transform.x_m, y-transform.y_m
    c, s = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    return c*dx+s*dy, -s*dx+c*dy


def _map_point(x, y, transform):
    c, s = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    return transform.x_m+c*x-s*y, transform.y_m+s*x+c*y


def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"current LiDAR {name} must be finite")
    return float(value)


def _validate_cohort(receipts, *, snapshot, planning_frame, now_sec, not_before_sec):
    if len(receipts) != SCAN_COUNT:
        raise ValueError("current LiDAR support requires a complete eight-scan cohort")
    now, floor = _finite(now_sec, "clock"), _finite(not_before_sec, "observation floor")
    if not 0 <= floor <= now:
        raise ValueError("current LiDAR observation floor is invalid")
    if not isinstance(planning_frame, CandidatePlanningFrame) or planning_frame.map_frame != snapshot.planning_frame:
        raise ValueError("current LiDAR planning frame differs from snapshot")
    stamps = []
    for receipt in receipts:
        validate_lidar_visibility_receipt(receipt)
        frame = receipt.frame_provenance
        if (frame is None or receipt.scan_metadata is None
                or receipt.map_bundle_sha256 != snapshot.map_bundle_sha256
                or frame.map_frame != planning_frame.map_frame
                or frame.odom_frame != planning_frame.odom_frame
                or frame.map_from_odom != planning_frame.map_from_odom):
            raise ValueError("current LiDAR receipt frame/map binding mismatch")
        if receipt.scan_stamp_sec < floor-1e-6 or receipt.observer_clock_sec > now+.02:
            raise ValueError("current LiDAR receipt predates stopped epoch or is future dated")
        if not -.02 <= receipt.observer_clock_sec-receipt.scan_stamp_sec <= MAX_LATEST_AGE_SEC:
            raise ValueError("current LiDAR scan was stale at receipt")
        stamps.append(receipt.scan_stamp_sec)
    if (len({r.receipt_id for r in receipts}) != SCAN_COUNT
            or any(b-a < .08-1e-6 for a, b in zip(stamps, stamps[1:]))
            or stamps[-1]-stamps[0] > MAX_WINDOW_SEC+1e-6
            or not -.02 <= now-stamps[-1] <= MAX_LATEST_AGE_SEC
            or len({(r.viewpoint_id, r.scan_frame, r.scan_topic, r.observer_config_sha256,
                     r.frame_provenance.source_evidence_id) for r in receipts}) != 1):
        raise ValueError("current LiDAR scan cohort is stale, replayed, or inconsistent")
    poses = [r.frame_provenance.canonical_scan_pose_odom for r in receipts]
    if any(math.hypot(a.x_m-b.x_m, a.y_m-b.y_m) > .02
           or abs(math.remainder(a.yaw_rad-b.yaw_rad, math.tau)) > math.radians(2)
           for a in poses for b in poses):
        raise ValueError("current LiDAR scanner moved during stopped capture")


def _clusters(receipt, maximum_width):
    cfg = LidarStandDetectorConfig(min_range_m=receipt.range_min_m,
        max_range_m=receipt.range_max_m+1e-9, max_cluster_gap_m=.08,
        min_cluster_points=2, min_width_m=.01, max_width_m=maximum_width)
    points = scan_points_from_ranges(receipt.ranges_m, angle_min_rad=receipt.angle_min_rad,
                                    angle_increment_rad=receipt.angle_increment_rad, config=cfg)
    topology = receipt.scan_metadata.topology(sample_count=len(receipt.ranges_m),
        angle_min_rad=receipt.angle_min_rad, angle_increment_rad=receipt.angle_increment_rad)
    pose = receipt.frame_provenance.canonical_scan_pose_odom
    clusters = []
    for cluster in cluster_scan_points(points, config=cfg, topology=topology):
        width = max((math.hypot(a.x_m-b.x_m, a.y_m-b.y_m)
                     for a in cluster for b in cluster), default=0.)
        local = tuple(sum(getattr(p, k) for p in cluster)/len(cluster) for k in ("x_m", "y_m"))
        center = _map_point(*local, pose)
        clusters.append({"center_odom": list(center), "width_m": width,
            "point_count": len(cluster), "source_indices": [p.source_index for p in cluster],
            "plausible": len(cluster) >= 2 and .01 <= width <= maximum_width})
    return clusters


def assess_current_lidar_targets(*, snapshot, planning_frame, receipts, candidate_uids,
                                stand_model, now_sec, not_before_sec, mount_evidence, base_frame):
    """Assess a complete cohort; invalid sensor evidence is a systemic error.

    Unsupported candidates are deferred, never deleted. Every scan remains in
    the support denominator, and ambiguous correspondence vetoes an estimate.
    """
    validate_candidate_snapshot(snapshot)
    uids = tuple(sorted(candidate_uids))
    if not uids or len(set(uids)) != len(uids) or not set(uids).issubset(snapshot.candidate_uids):
        raise ValueError("current LiDAR targets require unique known candidate IDs")
    receipts = tuple(receipts)
    _validate_cohort(receipts, snapshot=snapshot, planning_frame=planning_frame,
                     now_sec=now_sec, not_before_sec=not_before_sec)
    tf = planning_frame.map_from_odom
    centers = {c.candidate_uid: _odom_point(c.geometry.x_m, c.geometry.y_m, tf)
               for c in snapshot.candidates}
    limits = {c.candidate_uid: min(.16, 2*(c.geometry.radius_m+c.geometry.uncertainty_m))
              for c in snapshot.candidates}
    mount_reviews = {}
    for uid in uids:
        range_bound = max(math.dist(centers[uid], (r.frame_provenance.canonical_scan_pose_odom.x_m,
            r.frame_provenance.canonical_scan_pose_odom.y_m))+limits[uid]+snapshot.candidate_for(uid).geometry.radius_m
            for r in receipts)
        mount_review = verify_lidar_head_observability(stand_model=stand_model, base_frame=base_frame,
            mount_evidence=mount_evidence, target_range_m=range_bound,
            source_scan_stamps_sec=[r.scan_stamp_sec for r in receipts])
        if not mount_review["accepted"] and mount_review["reason"] != "laser_plane_not_inside_measured_head":
            raise ValueError("current LiDAR head cross-section unavailable: " + mount_review["reason"])
        mount_reviews[uid] = mount_review
    maximum_width = math.hypot(stand_model.head_width_m+stand_model.tolerance_m,
                               stand_model.head_depth_m+stand_model.tolerance_m)+.006
    scan_clusters = [_clusters(r, maximum_width) for r in receipts]
    estimates, decisions = {}, {}
    for uid in uids:
        candidate, center = snapshot.candidate_for(uid), centers[uid]
        scans, supported = [], []
        for receipt, clusters in zip(receipts, scan_clusters):
            nearby = [c for c in clusters if math.dist(c["center_odom"], center) <= limits[uid]+1e-9]
            plausible = [c for c in nearby if c["plausible"]]
            competing = [c for c in plausible if any(math.dist(c["center_odom"], other) <= limits[other_uid]+1e-9
                         for other_uid, other in centers.items() if other_uid != uid)]
            reason = ("competing_candidate" if competing else "ambiguous_clusters" if len(plausible) > 1
                      else "supported" if len(plausible) == 1 else "non_stand_cluster" if nearby
                      else "unsupported")
            if reason == "unsupported":
                pose = receipt.frame_provenance.canonical_scan_pose_odom
                distance = math.dist(center, (pose.x_m, pose.y_m))
                bearing = math.atan2(center[1]-pose.y_m, center[0]-pose.x_m)-pose.yaw_rad
                cone = math.asin(min(1., limits[uid]/max(distance, .001)))
                ranges = [r for index, r in enumerate(receipt.ranges_m) if r is not None
                          and abs(math.remainder(receipt.angle_min_rad+index*receipt.angle_increment_rad-bearing,
                                                 math.tau)) <= cone]
                reason = ("insufficient_visible_returns" if not ranges else
                          "occluded" if min(ranges) < distance-limits[uid] else "unsupported")
            selected = plausible[0] if reason == "supported" else None
            scans.append({"scan_stamp_sec": receipt.scan_stamp_sec, "reason": reason,
                "nearby_cluster_count": len(nearby), "plausible_cluster_count": len(plausible),
                "selected_cluster": selected})
            if selected is not None:
                supported.append(selected["center_odom"])
        reasons = []
        if not mount_reviews[uid]["accepted"]:
            reasons.append(mount_reviews[uid]["reason"])
        if any(s["reason"] in ("competing_candidate", "ambiguous_clusters") for s in scans):
            reasons.append("ambiguous_current_target_correspondence")
        if len(supported) < 3 or len(supported)/len(receipts) < MIN_SUPPORTED_FRACTION:
            reasons.append("insufficient_current_lidar_support")
        mean = None if not supported else tuple(sum(p[k] for p in supported)/len(supported) for k in (0, 1))
        scatter = None if mean is None else max(math.dist(p, mean) for p in supported)
        if scatter is not None and scatter > MAX_CENTER_SCATTER_M+1e-9:
            reasons.append("current_cluster_centers_unstable")
        estimate = None
        if not reasons:
            x, y = _map_point(*mean, tf)
            uncertainty = candidate.geometry.radius_m+scatter+.02
            if uncertainty > .14 or math.dist(mean, center) > limits[uid]+1e-9:
                reasons.append("current_target_geometry_exceeds_bound")
            else:
                estimate = {"x_m": x, "y_m": y, "uncertainty_m": uncertainty, "policy": POLICY}
                estimates[uid] = estimate
        decisions[uid] = {"candidate_uid": uid, "accepted": not reasons, "reasons": reasons,
            "estimate": estimate, "center_odom": None if mean is None else {"x_m": mean[0], "y_m": mean[1]},
            "displacement_m": None if mean is None else math.dist(mean, center),
            "maximum_displacement_m": limits[uid], "center_scatter_m": scatter,
            "supported_scan_count": len(supported), "scan_count": len(receipts), "scans": scans}
    evidence = {"schema_version": 1, "policy": POLICY, "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
        "map_bundle_sha256": snapshot.map_bundle_sha256, "planning_frame": planning_frame.to_evidence(),
        "candidate_uids": list(uids), "eligible_candidate_uids": sorted(estimates),
        "excluded_candidate_uids": sorted(set(uids)-set(estimates)), "candidate_decisions": decisions,
        "assessed_at_unix_sec": now_sec, "observation_not_before_sec": not_before_sec,
        "receipt_set_sha256": visibility_receipts_sha256(receipts), "scan_count": len(receipts),
        "maximum_cluster_width_m": maximum_width, "minimum_supported_scan_fraction": MIN_SUPPORTED_FRACTION,
        "stand_model": asdict(stand_model), "head_observability_by_candidate": mount_reviews, "base_frame": base_frame,
        "motion_authorized": False, "stand_axis_authorized": False, "candidate_rejection_authorized": False,
        "keepouts_changed": False}
    return estimates, evidence


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("current LiDAR evidence must be a regular non-symlink file")
    return path


def _load_sources(view_path, view_sha, capture_path, capture_sha):
    view = load_content_hashed_json(_regular(view_path), hash_field=VIEW_HASH_FIELD)
    capture = load_content_hashed_json(_regular(capture_path), hash_field=CAPTURE_HASH_FIELD)
    if (payload_sha256(view) != view_sha or payload_sha256(capture) != capture_sha
            or view["capture_path"] != str(capture_path) or view["capture_sha256"] != capture_sha):
        raise ValueError("current LiDAR source hash/path mismatch")
    rebuilt = head_capture_payload(capture["scans"], tour_id=capture["viewpoint_id"],
        odom_frame=capture["odom_frame"], base_frame=capture["base_frame"], scan_frame=capture["scan_frame"],
        captured_at_unix_sec=capture["captured_at_unix_sec"])
    if rebuilt != capture:
        raise ValueError("current LiDAR capture contract changed")
    if (view["artifact_kind"] != "candidate_local_lidar_view" or view["scan_count"] != SCAN_COUNT
            or view["viewpoint_id"] != capture["viewpoint_id"]
            or view["latest_base_pose_odom"] != capture["scans"][-1]["base_pose_odom"]
            or view["pose_stamp_sec"] != capture["scans"][-1]["stamp_sec"]
            or view["captured_at_unix_sec"] != capture["captured_at_unix_sec"]):
        raise ValueError("current LiDAR view capture identity/pose changed")
    receipts = tuple(_receipt_from_hashed_payload(r) for r in view["receipts"])
    if len(receipts) != len(capture["scans"]) or view["receipt_set_sha256"] != visibility_receipts_sha256(receipts):
        raise ValueError("current LiDAR view receipt binding changed")
    for receipt, scan in zip(receipts, capture["scans"]):
        frame = receipt.frame_provenance
        if (receipt.scan_stamp_sec != scan["stamp_sec"] or list(receipt.ranges_m) != scan["ranges"]
                or frame is None or asdict(frame.canonical_scan_pose_odom) != scan["scan_pose_odom"]
                or frame.source_evidence_id != capture_sha
                or receipt.scan_frame != capture["scan_frame"] or frame.odom_frame != capture["odom_frame"]
                or receipt.range_min_m != scan["range_min"] or receipt.range_max_m != scan["range_max"]
                or receipt.observer_clock_sec != scan["received_at_unix_sec"]
                or receipt.angle_min_rad != scan["angle_min"] or receipt.angle_increment_rad != scan["angle_increment"]
                or receipt.scan_metadata.to_mapping() != scan["scan_metadata"]):
            raise ValueError("current LiDAR receipts differ from original capture")
    expected_mounts = [{**s["head_plane_mount"], "stamp_sec": s["stamp_sec"]}
                       for s in capture["scans"] if "head_plane_mount" in s]
    if view["mount_evidence"] != expected_mounts:
        raise ValueError("current LiDAR mount records differ from original capture")
    return view, receipts


def _require_planning_start(view, planning_frame):
    expected = _odom_point(planning_frame.current_pose.x_m, planning_frame.current_pose.y_m,
                           planning_frame.map_from_odom)
    base = view["latest_base_pose_odom"]
    if (math.dist(expected, (base["x_m"], base["y_m"])) > .015+1e-9
            or abs(math.remainder(base["yaw_rad"]-(planning_frame.current_pose.yaw_rad-
                planning_frame.map_from_odom.yaw_rad), math.tau)) > math.radians(2)+1e-9):
        raise ValueError("current LiDAR stopped base pose differs from planning start")


def capture_current_lidar_targets(config, effects, planning_frame, candidate_uids, output_dir):
    """Capture once at the current stopped pose and bind all assessed targets."""
    if effects.capture_lidar_view is None or config.camera_calibration is None:
        raise RuntimeError("current LiDAR target capture dependencies are unavailable")
    uids = tuple(sorted(candidate_uids))
    if not uids:
        raise ValueError("current LiDAR target population is empty")
    root, floor = Path(output_dir), effects.clock()
    path = root / "current_lidar_targets.json"
    if root.is_symlink() or path.exists() or path.is_symlink():
        raise ValueError("current LiDAR support output must be fresh")
    snapshot_sha = candidate_snapshot_sha256(config.snapshot)
    viewpoint = "current_" + payload_sha256({"uids": list(uids), "snapshot": snapshot_sha,
                                            "floor": floor, "path": str(root)})[:32]
    request = CandidateLidarCaptureRequest(config.plan, snapshot_sha, uids[0], viewpoint,
        root / "capture", floor, planning_frame, config.camera_calibration.base_frame,
        config.lidar_scan_frame, config.lidar_scan_topic)
    captured = effects.capture_lidar_view(request)
    if (not isinstance(captured, CandidateLidarView) or captured.candidate_uid != request.candidate_uid
            or captured.candidate_snapshot_sha256 != snapshot_sha or captured.viewpoint_id != viewpoint):
        raise ValueError("current LiDAR capture request binding changed")
    view, receipts = _load_sources(captured.evidence_path, captured.evidence_sha256,
                                  captured.capture_path, captured.capture_sha256)
    if (view["candidate_snapshot_sha256"] != snapshot_sha or view["candidate_uid"] != uids[0]
            or view["viewpoint_id"] != viewpoint or view["planning_frame"] != planning_frame.to_evidence()
            or view["observation_not_before_sec"] != floor or receipts != captured.receipts
            or view["latest_base_pose_odom"] != asdict(captured.latest_base_pose_odom)
            or view["pose_stamp_sec"] != captured.pose_stamp_sec):
        raise ValueError("current LiDAR view request binding changed")
    _require_planning_start(view, planning_frame)
    estimates, evidence = assess_current_lidar_targets(snapshot=config.snapshot, planning_frame=planning_frame,
        receipts=receipts, candidate_uids=uids, stand_model=config.measured_stand_model,
        now_sec=effects.clock(), not_before_sec=floor, mount_evidence=view["mount_evidence"],
        base_frame=request.base_frame)
    evidence.update(candidate_lidar_view_path=str(captured.evidence_path),
        candidate_lidar_view_sha256=captured.evidence_sha256, capture_path=str(captured.capture_path),
        capture_sha256=captured.capture_sha256)
    digest = write_content_hashed_json(path, evidence, hash_field=HASH_FIELD)
    return estimates, {**evidence, "evidence_path": str(path), "evidence_sha256": digest}


def load_current_lidar_assessment(path, *, snapshot):
    """Replay the original complete assessment, including rejected targets."""
    evidence = load_content_hashed_json(_regular(path), hash_field=HASH_FIELD)
    if (evidence.get("policy") != POLICY or evidence.get("schema_version") != 1
            or evidence.get("candidate_snapshot_sha256") != candidate_snapshot_sha256(snapshot)
            or evidence.get("map_bundle_sha256") != snapshot.map_bundle_sha256):
        raise ValueError("current LiDAR target snapshot/candidate binding mismatch")
    view, receipts = _load_sources(evidence["candidate_lidar_view_path"], evidence["candidate_lidar_view_sha256"],
                                   evidence["capture_path"], evidence["capture_sha256"])
    if (view["candidate_snapshot_sha256"] != evidence["candidate_snapshot_sha256"]
            or view["planning_frame"] != evidence["planning_frame"]
            or view["observation_not_before_sec"] != evidence["observation_not_before_sec"]):
        raise ValueError("current LiDAR source snapshot changed")
    planning_frame = CandidatePlanningFrame.from_evidence(evidence["planning_frame"])
    _require_planning_start(view, planning_frame)
    estimates, replay = assess_current_lidar_targets(snapshot=snapshot,
        planning_frame=planning_frame,
        receipts=receipts, candidate_uids=evidence["candidate_uids"], stand_model=StandModelProfile(**evidence["stand_model"]),
        now_sec=evidence["assessed_at_unix_sec"], not_before_sec=evidence["observation_not_before_sec"],
        mount_evidence=view["mount_evidence"], base_frame=evidence["base_frame"])
    if any(evidence.get(key) != value for key, value in replay.items()):
        raise ValueError("current LiDAR support assessment failed source replay")
    return estimates, evidence


def load_current_lidar_target(path, *, candidate_uid, snapshot):
    """Replay source-bound support before using a local target estimate."""
    estimates, evidence = load_current_lidar_assessment(path, snapshot=snapshot)
    if candidate_uid not in estimates:
        raise ValueError("current LiDAR target snapshot/candidate binding mismatch")
    return dict(estimates[candidate_uid])
