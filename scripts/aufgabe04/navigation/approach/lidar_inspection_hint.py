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
from scripts.aufgabe04.navigation.approach.lidar_head_geometry import (
    HEAD_MODEL, HeadSurfaceFit, adjacent_returns, combine_head_surfaces, fit_head_surface,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import StandSurveyRegistry, stand_survey_registry_sha256
from scripts.aufgabe04.perception.lidar_visibility_evidence import LidarVisibilityReceipt
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import axial_difference_rad, axial_normalize_rad
from scripts.aufgabe04.stations.candidate_snapshot import CandidateSnapshot, candidate_snapshot_sha256
from scripts.aufgabe04.perception.scan_topology import ScanTopology


@dataclass(frozen=True)
class LidarInspectionHint:
    candidate_uid: str
    snapshot_sha256: str
    tangent_rad: float
    evidence: Mapping[str, object]
    center_x_m: float | None = None
    center_y_m: float | None = None
    center_uncertainty_m: float | None = None
    angle_uncertainty_rad: float | None = None

    def normals(self, snapshot: CandidateSnapshot, candidate_uid: str) -> tuple[float, float]:
        if (self.candidate_uid != candidate_uid
                or self.snapshot_sha256 != candidate_snapshot_sha256(snapshot)
                or not math.isfinite(self.tangent_rad)):
            raise ValueError("LiDAR inspection hint candidate/snapshot binding mismatch")
        normal = math.remainder(self.tangent_rad + math.pi / 2, 2 * math.pi)
        return normal, math.remainder(normal + math.pi, 2 * math.pi)

    def fit_geometry(self, snapshot: CandidateSnapshot, candidate_uid: str) -> dict[str, float] | None:
        """Return bounded planning geometry; legacy tangent-only hints have none."""
        self.normals(snapshot, candidate_uid)
        values = {name: getattr(self, name) for name in (
            "center_x_m", "center_y_m", "center_uncertainty_m", "angle_uncertainty_rad")}
        if any(value is None for value in values.values()):
            return None
        if (not all(math.isfinite(v) for v in values.values())
                or self.center_uncertainty_m <= 0 or self.angle_uncertainty_rad <= 0):
            raise ValueError("LiDAR inspection hint geometry must have finite positive uncertainty")
        return values


def _scan_surface_analysis(receipt, center, radius, other_centers):
    diagnostics = {"scan_stamp_sec": receipt.scan_stamp_sec, "candidate_indices": [],
                   "cluster_indices": [], "boundary_fragmented": False,
                   "motion_authorized": False, "stand_axis_authorized": False}
    if receipt.frame_provenance is None:
        return None, {**diagnostics, "reason": "scan_frame_unavailable"}
    pose = receipt.frame_provenance.canonical_scan_pose_odom
    clusters = []
    current = []
    previous_index = None
    previous_distance = None
    indices = []
    cluster_indices = []
    current_indices = []
    for index, distance in enumerate(receipt.ranges_m):
        if distance is None:
            continue
        angle = pose.yaw_rad + receipt.angle_min_rad + index * receipt.angle_increment_rad
        point = pose.x_m + distance * math.cos(angle), pose.y_m + distance * math.sin(angle)
        if math.hypot(point[0] - center.x_m, point[1] - center.y_m) > radius:
            continue
        if any(math.hypot(point[0] - x, point[1] - y) <= r for x, y, r in other_centers):
            return None, {**diagnostics, "reason": "competing_candidate"}
        if current and (index != previous_index + 1 or not adjacent_returns(
                previous_distance, distance, receipt.angle_increment_rad)):
            clusters.append(current)
            cluster_indices.append(current_indices)
            current = []
            current_indices = []
        current.append(point)
        current_indices.append(index)
        indices.append(index)
        previous_index = index
        previous_distance = distance
    if current:
        clusters.append(current)
        cluster_indices.append(current_indices)
    metadata = receipt.scan_metadata
    topology = (ScanTopology(len(receipt.ranges_m), receipt.angle_min_rad, receipt.angle_increment_rad)
                if metadata is None else metadata.topology(sample_count=len(receipt.ranges_m),
                    angle_min_rad=receipt.angle_min_rad, angle_increment_rad=receipt.angle_increment_rad))
    boundary = len(clusters) >= 2 and indices[0] == 0 and indices[-1] == len(receipt.ranges_m)-1
    diagnostics.update(candidate_indices=indices, cluster_indices=cluster_indices,
                       boundary_fragmented=boundary, scan_topology=topology.evidence(),
                       candidate_return_count=len(indices),
                       angle_increment_rad=receipt.angle_increment_rad,
                       sensor_range_m=math.hypot(pose.x_m-center.x_m, pose.y_m-center.y_m))
    # Only the original, validated full-rotation geometry may join endpoints.
    # Internal missing bins remain separate even when the seam itself is valid.
    if (boundary and topology.joins_endpoints(indices[-1], indices[0])
            and adjacent_returns(receipt.ranges_m[indices[-1]], receipt.ranges_m[indices[0]],
                                 math.tau-(len(receipt.ranges_m)-1)*abs(receipt.angle_increment_rad))):
        clusters = [clusters[-1] + clusters[0], *clusters[1:-1]]
        cluster_indices = [cluster_indices[-1] + cluster_indices[0], *cluster_indices[1:-1]]
        diagnostics.update(cluster_indices=cluster_indices, seam_joined=True, boundary_fragmented=False)
    if len(clusters) != 1:
        return None, {**diagnostics, "reason": "fragmented_candidate_returns" if clusters else "no_candidate_returns"}
    surface = fit_head_surface(clusters[0], sensor_position=(pose.x_m, pose.y_m))
    return surface, {**diagnostics, "reason": ("accepted" if surface is not None else
        "fewer_than_four_returns" if len(clusters[0]) < 4 else "surface_geometry_rejected")}


def _scan_surface(receipt, center, radius, other_centers) -> HeadSurfaceFit | None:
    return _scan_surface_analysis(receipt, center, radius, other_centers)[0]


def analyze_candidate_lidar_support(*, registry, candidate_uid, receipts):
    """Explain all examined scans without selecting or authorizing a head axis."""
    from collections import Counter
    source = registry.candidate_for(candidate_uid)
    if source is None or source.frame_provenance is None:
        raise ValueError("LiDAR support candidate frame unavailable")
    center = source.frame_provenance.canonical_odom_point
    others = [(c.frame_provenance.canonical_odom_point.x_m,
               c.frame_provenance.canonical_odom_point.y_m, c.radius_m+c.uncertainty_m)
              for c in registry.candidates if c.candidate_uid != candidate_uid and c.frame_provenance is not None]
    scans = [_scan_surface_analysis(r, center, source.radius_m+source.uncertainty_m, others)[1]
             for r in receipts]
    return {"candidate_uid": candidate_uid, "examined_scan_count": len(scans), "scans": scans,
            "reason_counts": dict(Counter(s["reason"] for s in scans)),
            "boundary_fragmented_scan_count": sum(s["boundary_fragmented"] for s in scans),
            "motion_authorized": False, "stand_axis_authorized": False}


def _scan_axis(receipt, center, radius, other_centers):
    """Compatibility helper for scan-support replay tools."""
    surface = _scan_surface(receipt, center, radius, other_centers)
    return None if surface is None else surface.tangent_rad


def derive_lidar_inspection_hints(
    *, snapshot: CandidateSnapshot, registry: StandSurveyRegistry,
    planning_frame: CandidatePlanningFrame, receipts: Sequence[LidarVisibilityReceipt],
    additional_viewpoint_ids: Sequence[str] = (),
    candidate_uids: Sequence[str] | None = None,
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
    selected_uids = snapshot.candidate_uids if candidate_uids is None else tuple(candidate_uids)
    if (len(set(selected_uids)) != len(selected_uids)
            or any(uid not in snapshot.candidate_uids for uid in selected_uids)):
        raise ValueError("LiDAR hint candidate selection must contain unique known IDs")
    selected_uids = frozenset(selected_uids)
    hints, diagnostics = {}, {}
    additional_views = frozenset(additional_viewpoint_ids)
    snapshot_hash = candidate_snapshot_sha256(snapshot)
    registry_hash = stand_survey_registry_sha256(registry)
    for candidate in snapshot.candidates:
        uid = candidate.candidate_uid
        if uid not in selected_uids:
            continue
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
        groups, counts, observed_poses = {}, {}, {}
        seen = set()
        for receipt in receipts:
            rf = receipt.frame_provenance
            if (receipt.survey_id != registry.survey_id
                    or receipt.map_bundle_sha256 != snapshot.map_bundle_sha256
                    or receipt.viewpoint_id not in (*source.viewpoint_ids, *additional_views) or rf is None
                    or (rf.map_frame, rf.odom_frame) != (frame.map_frame, frame.odom_frame)):
                continue
            stamp_key = (receipt.scan_frame, receipt.scan_stamp_sec)
            if stamp_key in seen:
                continue
            seen.add(stamp_key)
            counts[receipt.viewpoint_id] = counts.get(receipt.viewpoint_id, 0) + 1
            observed_poses.setdefault(receipt.viewpoint_id, []).append(rf.canonical_scan_pose_odom)
            surface = _scan_surface(receipt, center, radius, others)
            if surface is not None:
                groups.setdefault(receipt.viewpoint_id, []).append((surface, receipt))
        views = []
        conflict = False
        for view_id, samples in sorted(groups.items()):
            if len(samples) < 3 or len(samples) / counts[view_id] < .75:
                continue
            angles = [s.tangent_rad for s, _ in samples]
            poses = observed_poses[view_id]
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
            fitted = combine_head_surfaces([s for v in views for s, _ in v[3]], tangent_rad=axis)
            if fitted is None:
                diagnostics[uid] = {"usable": False, "reason": "head_center_or_angle_uncertain",
                                    "usable_view_count": len(views)}
                continue
            transform = planning_frame.map_from_odom
            cp, sp = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
            center_x = transform.x_m + cp * fitted.center_x_m - sp * fitted.center_y_m
            center_y = transform.y_m + sp * fitted.center_x_m + cp * fitted.center_y_m
            evidence = {
                "usable": True, "reason": "multi_view_head_geometry",
                "candidate_uid": uid, "candidate_snapshot_sha256": snapshot_hash,
                "source_registry_sha256": registry_hash,
                "tangent_odom_rad": axis,
                "tangent_planning_rad": axial_normalize_rad(axis + planning_frame.map_from_odom.yaw_rad),
                "center_odom": {"x_m": fitted.center_x_m, "y_m": fitted.center_y_m},
                "center_planning": {"x_m": center_x, "y_m": center_y},
                "center_uncertainty_m": fitted.center_uncertainty_m,
                "angle_uncertainty_rad": fitted.angle_uncertainty_rad,
                "minimum_observed_span_m": fitted.observed_span_m,
                "minimum_beam_count": fitted.point_count,
                "head_model": {"width_m": HEAD_MODEL.width_m, "depth_m": HEAD_MODEL.depth_m,
                               "tolerance_m": HEAD_MODEL.tolerance_m},
                "assumed_point_noise_m": HEAD_MODEL.point_noise_m,
                "uncertainty_kind": "engineering_bounds_not_calibrated_confidence",
                "head_cross_section_intersection_assumed": True,
                "head_endpoints_observed": False,
                "viewpoint_ids": [v[2] for v in views],
                "scan_count_by_view": {v[2]: len(v[3]) for v in views},
                "source_receipt_sha256s": [r.receipt_sha256 for v in views for _, r in v[3]],
                "planning_frame": planning_frame.to_evidence(),
                "independent_view_requirement_met": True,
                "stand_axis_authorized": False, "motion_authorized": False,
                "head_alignment_verified": False,
                "observed_view_axis_spread_rad": max(
                    axial_difference_rad(a[0], b[0]) for a in views for b in views),
                "angle_accuracy_calibrated": False,
            }
            hints[uid] = LidarInspectionHint(
                uid, snapshot_hash, evidence["tangent_planning_rad"], evidence,
                center_x, center_y, fitted.center_uncertainty_m, fitted.angle_uncertainty_rad)
            diagnostics[uid] = evidence
            continue
        diagnostics[uid] = {"usable": False, "reason": reason,
                            "usable_view_count": len(views),
                            "scan_count_by_view": counts,
                            "surface_fit_count_by_view": {k: len(v) for k, v in groups.items()}}
    return hints, diagnostics


def fit_current_lidar_view(
    *, snapshot: CandidateSnapshot, registry: StandSurveyRegistry,
    planning_frame: CandidatePlanningFrame, candidate_uid: str,
    receipts: Sequence[LidarVisibilityReceipt],
) -> LidarInspectionHint | None:
    """Reacquire one stopped view for arrival verification, never initial authority.

    This deliberately does not establish the two independent views required to
    choose an initial normal. The arrival verifier must compare it to that
    previously supported hint and independently enforce sensor freshness.
    """
    candidate = next((c for c in snapshot.candidates if c.candidate_uid == candidate_uid), None)
    source = registry.candidate_for(candidate_uid)
    if candidate is None or source is None or source.frame_provenance is None or not receipts:
        return None
    frame = source.frame_provenance
    if (registry.map_bundle_sha256 != snapshot.map_bundle_sha256
            or registry.planning_frame != snapshot.planning_frame
            or planning_frame.map_frame != snapshot.planning_frame
            or (frame.map_frame, frame.odom_frame) != (planning_frame.map_frame, planning_frame.odom_frame)):
        return None
    projected = reproject_candidate_point(frame, planning_frame.map_from_odom).current_map_point
    if (math.hypot(projected.x_m - candidate.geometry.x_m, projected.y_m - candidate.geometry.y_m) > 1e-8
            or set(source.source_observation_ids) != set(candidate.source.observation_ids)
            or candidate.source.source_artifact_sha256 != stand_survey_registry_sha256(registry)):
        return None
    radius = candidate.geometry.radius_m + candidate.geometry.uncertainty_m
    if not 0 < radius <= .15:
        return None
    others = [(c.frame_provenance.canonical_odom_point.x_m,
               c.frame_provenance.canonical_odom_point.y_m, c.radius_m + c.uncertainty_m)
              for c in registry.candidates if c.candidate_uid != candidate_uid and c.frame_provenance is not None]
    fits, accepted, poses, seen = [], [], [], set()
    if len({r.viewpoint_id for r in receipts}) != 1:
        return None
    for receipt in receipts:
        rf = receipt.frame_provenance
        if (rf is None or receipt.survey_id != registry.survey_id
                or receipt.map_bundle_sha256 != snapshot.map_bundle_sha256
                or (rf.map_frame, rf.odom_frame) != (frame.map_frame, frame.odom_frame)):
            return None
        key = (receipt.scan_frame, receipt.scan_stamp_sec)
        if key in seen:
            return None
        seen.add(key)
        poses.append(rf.canonical_scan_pose_odom)
        fit = _scan_surface(receipt, frame.canonical_odom_point, radius, others)
        if fit is not None:
            fits.append(fit)
            accepted.append(receipt)
    if (len(fits) < 3 or len(fits) / len(receipts) < .75
            or any(math.hypot(p.x_m - poses[0].x_m, p.y_m - poses[0].y_m) > .02
                   or abs(math.remainder(p.yaw_rad - poses[0].yaw_rad, 2 * math.pi)) > math.radians(3)
                   for p in poses)
            or any(axial_difference_rad(a.tangent_rad, b.tangent_rad) > math.radians(8)
                   for a in fits for b in fits)):
        return None
    tangent = .5 * math.atan2(sum(math.sin(2 * f.tangent_rad) for f in fits),
                              sum(math.cos(2 * f.tangent_rad) for f in fits))
    fit = combine_head_surfaces(fits, tangent_rad=tangent)
    if fit is None:
        return None
    tf = planning_frame.map_from_odom
    cp, sp = math.cos(tf.yaw_rad), math.sin(tf.yaw_rad)
    x = tf.x_m + cp * fit.center_x_m - sp * fit.center_y_m
    y = tf.y_m + sp * fit.center_x_m + cp * fit.center_y_m
    snapshot_hash = candidate_snapshot_sha256(snapshot)
    evidence = {
        "usable": True, "reason": "current_stopped_head_geometry",
        "candidate_uid": candidate_uid, "candidate_snapshot_sha256": snapshot_hash,
        "source_registry_sha256": stand_survey_registry_sha256(registry),
        "source_receipt_sha256s": [r.receipt_sha256 for r in accepted],
        "examined_receipt_sha256s": [r.receipt_sha256 for r in receipts],
        "scan_count": len(accepted), "viewpoint_ids": [receipts[0].viewpoint_id],
        "tangent_odom_rad": tangent, "center_odom": {"x_m": fit.center_x_m, "y_m": fit.center_y_m},
        "center_planning": {"x_m": x, "y_m": y},
        "center_uncertainty_m": fit.center_uncertainty_m,
        "angle_uncertainty_rad": fit.angle_uncertainty_rad,
        "minimum_observed_span_m": fit.observed_span_m,
        "minimum_beam_count": fit.point_count,
        "planning_frame": planning_frame.to_evidence(),
        "independent_view_requirement_met": False,
        "stand_axis_authorized": False, "motion_authorized": False,
        "head_alignment_verified": False, "angle_accuracy_calibrated": False,
        "uncertainty_kind": "engineering_bounds_not_calibrated_confidence",
        "assumed_point_noise_m": HEAD_MODEL.point_noise_m,
    }
    return LidarInspectionHint(candidate_uid, snapshot_hash,
                               axial_normalize_rad(tangent + tf.yaw_rad), evidence,
                               x, y, fit.center_uncertainty_m, fit.angle_uncertainty_rad)
