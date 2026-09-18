"""Conservative multi-view LiDAR surface hints for the first camera view.

These are viewing suggestions, never certified head axes or front/back labels.
Fit each scan independently in canonical odom: repeated beams cannot manufacture
spatial support, and localization corrections cannot manufacture agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import reproject_candidate_point
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import StandSurveyRegistry, stand_survey_registry_sha256
from scripts.aufgabe04.perception.lidar_visibility_evidence import LidarVisibilityReceipt
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import axial_difference_rad, axial_normalize_rad
from scripts.aufgabe04.stations.candidate_snapshot import CandidateSnapshot, candidate_snapshot_sha256


@dataclass(frozen=True)
class LidarInspectionHint:
    candidate_uid: str
    snapshot_sha256: str
    tangent_rad: float
    evidence: Mapping[str, object]

    def normals(self, snapshot: CandidateSnapshot, candidate_uid: str) -> tuple[float, float]:
        if (self.candidate_uid != candidate_uid
                or self.snapshot_sha256 != candidate_snapshot_sha256(snapshot)
                or not math.isfinite(self.tangent_rad)):
            raise ValueError("LiDAR inspection hint candidate/snapshot binding mismatch")
        normal = math.remainder(self.tangent_rad + math.pi / 2, 2 * math.pi)
        return normal, math.remainder(normal + math.pi, 2 * math.pi)


def _line(points):
    if len(points) < 4:
        return None
    cx, cy = (sum(p[k] for p in points) / len(points) for k in (0, 1))
    xx = sum((x - cx) ** 2 for x, y in points) / len(points)
    yy = sum((y - cy) ** 2 for x, y in points) / len(points)
    xy = sum((x - cx) * (y - cy) for x, y in points) / len(points)
    angle = .5 * math.atan2(2 * xy, xx - yy)
    along = [(x - cx) * math.cos(angle) + (y - cy) * math.sin(angle) for x, y in points]
    across = [-(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) for x, y in points]
    span = max(along) - min(along)
    major = sum(v * v for v in along) / len(points)
    minor = sum(v * v for v in across) / len(points)
    if (not .04 <= span <= .18 or major <= 0 or minor / major > .05
            or max(abs(v) for v in across) > .006):
        return None
    return axial_normalize_rad(angle)


def _scan_axis(receipt, center, radius, other_centers):
    pose = receipt.frame_provenance.canonical_scan_pose_odom
    clusters = []
    current = []
    previous_index = None
    for index, distance in enumerate(receipt.ranges_m):
        if distance is None:
            continue
        angle = pose.yaw_rad + receipt.angle_min_rad + index * receipt.angle_increment_rad
        point = pose.x_m + distance * math.cos(angle), pose.y_m + distance * math.sin(angle)
        if math.hypot(point[0] - center.x_m, point[1] - center.y_m) > radius:
            continue
        if any(math.hypot(point[0] - x, point[1] - y) <= r for x, y, r in other_centers):
            return None  # A return is compatible with another candidate.
        if current and (index != previous_index + 1 or math.dist(point, current[-1]) > .04):
            clusters.append(current)
            current = []
        current.append(point)
        previous_index = index
    if current:
        clusters.append(current)
    # Do not join a seam without the original topology proof, or select the
    # nicest line from multiple surfaces in the candidate envelope.
    if len(clusters) != 1:
        return None
    return _line(clusters[0])


def derive_lidar_inspection_hints(
    *, snapshot: CandidateSnapshot, registry: StandSurveyRegistry,
    planning_frame: CandidatePlanningFrame, receipts: Sequence[LidarVisibilityReceipt],
) -> tuple[dict[str, LidarInspectionHint], dict[str, object]]:
    """Require >=3 distinct scans/view and two separated, consistent views.

    Thresholds are conservative proposal heuristics, not calibrated angle
    confidence. A circular support or a square base still cannot certify the
    head orientation; the camera must independently resolve it.
    """
    if (registry.map_bundle_sha256 != snapshot.map_bundle_sha256
            or registry.planning_frame != snapshot.planning_frame
            or planning_frame.map_frame != snapshot.planning_frame):
        raise ValueError("LiDAR hint frame/map binding mismatch")
    hints, diagnostics = {}, {}
    snapshot_hash = candidate_snapshot_sha256(snapshot)
    registry_hash = stand_survey_registry_sha256(registry)
    for candidate in snapshot.candidates:
        uid = candidate.candidate_uid
        source = registry.candidate_for(uid)
        frame = None if source is None else source.frame_provenance
        reason = "insufficient_independent_views"
        if frame is None or (frame.map_frame, frame.odom_frame) != (
                planning_frame.map_frame, planning_frame.odom_frame):
            diagnostics[uid] = {"reason": "candidate_frame_unavailable", "usable": False}
            continue
        projected = reproject_candidate_point(frame, planning_frame.map_from_odom).current_map_point
        if (math.hypot(projected.x_m - candidate.geometry.x_m,
                       projected.y_m - candidate.geometry.y_m) > 1e-8
                or set(source.source_observation_ids) != set(candidate.source.observation_ids)
                or candidate.source.source_artifact_sha256 != registry_hash):
            raise ValueError("LiDAR hint candidate source/projection mismatch")
        center = frame.canonical_odom_point
        radius = candidate.geometry.radius_m + candidate.geometry.uncertainty_m
        if not 0 < radius <= .15:
            diagnostics[uid] = {"reason": "association_envelope_too_wide", "usable": False}
            continue
        others = [(c.frame_provenance.canonical_odom_point.x_m,
                   c.frame_provenance.canonical_odom_point.y_m, c.radius_m + c.uncertainty_m)
                  for c in registry.candidates if c.candidate_uid != uid
                  and c.frame_provenance is not None]
        groups, counts = {}, {}
        seen = set()
        for receipt in receipts:
            rf = receipt.frame_provenance
            if (receipt.survey_id != registry.survey_id
                    or receipt.map_bundle_sha256 != snapshot.map_bundle_sha256
                    or receipt.viewpoint_id not in source.viewpoint_ids or rf is None
                    or (rf.map_frame, rf.odom_frame) != (frame.map_frame, frame.odom_frame)):
                continue
            stamp_key = (receipt.scan_frame, receipt.scan_stamp_sec)
            if stamp_key in seen:
                continue
            seen.add(stamp_key)
            counts[receipt.viewpoint_id] = counts.get(receipt.viewpoint_id, 0) + 1
            axis = _scan_axis(receipt, center, radius, others)
            if axis is not None:
                groups.setdefault(receipt.viewpoint_id, []).append((axis, receipt))
        views = []
        conflict = False
        for view_id, samples in sorted(groups.items()):
            if len(samples) < 3 or len(samples) / counts[view_id] < .75:
                continue
            angles = [a for a, _ in samples]
            poses = [r.frame_provenance.canonical_scan_pose_odom for _, r in samples]
            if any(math.hypot(p.x_m - poses[0].x_m, p.y_m - poses[0].y_m) > .02
                   or abs(math.remainder(p.yaw_rad - poses[0].yaw_rad, 2 * math.pi)) > math.radians(3)
                   for p in poses):
                continue
            if any(axial_difference_rad(a, b) > math.radians(8) for a in angles for b in angles):
                conflict = True
            mean = .5 * math.atan2(sum(math.sin(2*a) for a in angles), sum(math.cos(2*a) for a in angles))
            bearing = math.atan2(poses[0].y_m - center.y_m, poses[0].x_m - center.x_m)
            views.append((mean, bearing, view_id, samples))
        if conflict or any(axial_difference_rad(a[0], b[0]) > math.radians(8) for a in views for b in views):
            reason = "surface_axis_conflict"
        elif any(axial_difference_rad(a[1], b[1]) >= math.radians(20) for a in views for b in views):
            # Equal weighting per viewpoint, regardless of scan count.
            axis = .5 * math.atan2(sum(math.sin(2*v[0]) for v in views),
                                  sum(math.cos(2*v[0]) for v in views))
            evidence = {
                "usable": True, "reason": "multi_view_surface_axis",
                "candidate_uid": uid, "candidate_snapshot_sha256": snapshot_hash,
                "source_registry_sha256": registry_hash,
                "tangent_odom_rad": axis,
                "tangent_planning_rad": axial_normalize_rad(axis + planning_frame.map_from_odom.yaw_rad),
                "viewpoint_ids": [v[2] for v in views],
                "scan_count_by_view": {v[2]: len(v[3]) for v in views},
                "source_receipt_sha256s": [r.receipt_sha256 for v in views for _, r in v[3]],
                "planning_frame": planning_frame.to_evidence(),
                "stand_axis_authorized": False, "motion_authorized": False,
                "head_alignment_verified": False,
                "observed_view_axis_spread_rad": max(
                    axial_difference_rad(a[0], b[0]) for a in views for b in views),
                "angle_accuracy_calibrated": False,
            }
            hints[uid] = LidarInspectionHint(uid, snapshot_hash, evidence["tangent_planning_rad"], evidence)
            diagnostics[uid] = evidence
            continue
        diagnostics[uid] = {"usable": False, "reason": reason,
                            "usable_view_count": len(views)}
    return hints, diagnostics
