import math
from dataclasses import replace
import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import CandidateFrameProvenance, CandidatePoint2D
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import derive_lidar_inspection_hints
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    StandSurveyRegistry, SurveyCandidate, STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    STATUS_PENDING_CAMERA, stand_survey_registry_sha256,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.perception.lidar_visibility_evidence import lidar_visibility_receipt_from_scan
from scripts.aufgabe04.perception.lidar_visibility_frames import LidarVisibilityFrameProvenance
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import axial_difference_rad
from scripts.aufgabe04.stations.candidate_snapshot import new_candidate_snapshot
from tests.aufgabe04 import test_candidate_preapproach_planning as planning_fixtures


def hint_fixture(transform=PlanarTransform2D(0., 0., 0.)):
    candidate = planning_fixtures.CandidatePreapproachPlanningTest._candidate("candidate_1", transform.x_m, transform.y_m)
    source = SurveyCandidate(
        candidate_uid=candidate.candidate_uid, x_m=0., y_m=0., radius_m=.06,
        uncertainty_m=.02, keepout_radius_m=.31, confidence=.9, hit_count=6,
        first_seen_sec=10., last_seen_sec=21., source_observation_ids=candidate.source.observation_ids,
        viewpoint_ids=("vp1", "vp2"), status=STATUS_PENDING_CAMERA,
        frame_provenance=CandidateFrameProvenance.from_frozen_map_observation(
            map_frame="map", odom_frame="odom", frozen_map_point=CandidatePoint2D(0., 0.),
            frozen_map_from_odom=PlanarTransform2D(0., 0., 0.), source_evidence_id="c" * 64,
        ),
    )
    registry = StandSurveyRegistry(STAND_SURVEY_REGISTRY_SCHEMA_VERSION, "survey", "map", "a" * 64, (source,))
    candidate = replace(candidate, source=replace(candidate.source, source_artifact_sha256=stand_survey_registry_sha256(registry)))
    snapshot = new_candidate_snapshot(snapshot_id="snapshot", created_unix_sec=30.,
        planning_frame="map", map_bundle_sha256="a" * 64, candidates=(candidate,))
    frame = CandidatePlanningFrame(Pose2D(-.6, 0., 0.), transform)
    return snapshot, registry, frame


def line_receipts(view=1, *, tangent=math.pi / 2, bearing=None, increment=.028,
                  circle=False, frozen=PlanarTransform2D(0., 0., 0.)):
    bearing = (math.pi if view == 1 else math.radians(210)) if bearing is None else bearing
    pose = Pose2D(.6 * math.cos(bearing), .6 * math.sin(bearing), math.remainder(bearing + math.pi, 2*math.pi))
    ranges = []
    normal = (-math.sin(tangent), math.cos(tangent))
    for i in range(31):
        theta = pose.yaw_rad + (i - 15) * increment
        ray = math.cos(theta), math.sin(theta)
        if circle:
            dot = pose.x_m * ray[0] + pose.y_m * ray[1]
            disc = dot*dot - (pose.x_m**2 + pose.y_m**2 - .04**2)
            r = -dot - math.sqrt(disc) if disc >= 0 else float("inf")
        else:
            denom = normal[0]*ray[0] + normal[1]*ray[1]
            r = -(normal[0]*pose.x_m + normal[1]*pose.y_m)/denom if abs(denom) > 1e-9 else float("inf")
            x, y = pose.x_m + r*ray[0], pose.y_m + r*ray[1]
            if r < 0 or abs(x*math.cos(tangent) + y*math.sin(tangent)) > .039:
                r = float("inf")
        ranges.append(r)
    cp, sp = math.cos(frozen.yaw_rad), math.sin(frozen.yaw_rad)
    map_pose = Pose2D(frozen.x_m + cp*pose.x_m - sp*pose.y_m,
                      frozen.y_m + sp*pose.x_m + cp*pose.y_m, pose.yaw_rad + frozen.yaw_rad)
    return tuple(lidar_visibility_receipt_from_scan(
        receipt_id=f"v{view}_r{i}", survey_id="survey", viewpoint_id=f"vp{view}",
        planning_frame="map", scan_frame="base_scan", scan_topic="/scan",
        map_bundle_sha256="a"*64, observer_config_sha256="b"*64,
        scan_stamp_sec=view*10+i*.1, pose_stamp_sec=view*10+i*.1, observer_clock_sec=view*10+i*.1+.01,
        scan_pose_map=map_pose, angle_min_rad=-15*increment, angle_increment_rad=increment,
        range_min_m=.1, range_max_m=3.5, ranges_m=ranges,
        frame_provenance=LidarVisibilityFrameProvenance("map", "odom", frozen, pose, "c"*64),
    ) for i in range(4))


class LidarInspectionHintTest(unittest.TestCase):
    def derive(self, receipts, fixture=None):
        snapshot, registry, frame = fixture or hint_fixture()
        return derive_lidar_inspection_hints(snapshot=snapshot, registry=registry,
                                             planning_frame=frame, receipts=receipts)

    def test_multiview_recovers_normal_pair_after_localization_rotation(self):
        fixture = hint_fixture(PlanarTransform2D(.3, -.2, .4))
        hints, evidence = self.derive(line_receipts() + line_receipts(2, frozen=PlanarTransform2D(-.1, .2, -.5)), fixture)
        hint = hints["candidate_1"]
        self.assertLess(axial_difference_rad(hint.tangent_rad, math.pi/2+.4), 1e-9)
        normals = hint.normals(fixture[0], "candidate_1")
        self.assertAlmostEqual(abs(math.remainder(normals[1]-normals[0], 2*math.pi)), math.pi)
        self.assertFalse(hint.evidence["head_alignment_verified"])
        self.assertFalse(hint.evidence["stand_axis_authorized"])
        self.assertEqual(evidence["candidate_1"]["scan_count_by_view"], {"vp1": 4, "vp2": 4})

    def test_single_view_repetition_never_supplies_independent_views(self):
        hints, evidence = self.derive(line_receipts() * 20)
        self.assertFalse(hints)
        self.assertEqual(evidence["candidate_1"]["usable_view_count"], 1)

    def test_same_scan_stamps_cannot_meet_minimum(self):
        self.assertFalse(self.derive((line_receipts()[0],)*10 + (line_receipts(2)[0],)*10)[0])

    def test_sparse_returns_do_not_gain_spatial_support_by_pooling(self):
        scans = line_receipts(increment=.06) + line_receipts(2, increment=.06)
        self.assertFalse(self.derive(scans * 10)[0])

    def test_nearly_same_view_and_opposite_view_are_insufficient(self):
        for bearing in (math.radians(185), 0.):
            self.assertFalse(self.derive(line_receipts() + line_receipts(2, bearing=bearing))[0])

    def test_conflicting_surfaces_rejected(self):
        hints, evidence = self.derive(line_receipts(increment=.01) + line_receipts(2, tangent=1., increment=.01))
        self.assertFalse(hints)
        self.assertEqual(evidence["candidate_1"]["reason"], "surface_axis_conflict")

    def test_round_support_does_not_supply_consistent_head_axis(self):
        self.assertFalse(self.derive(line_receipts(circle=True) + line_receipts(2, circle=True))[0])

    def test_multiple_clusters_and_intermittent_good_fits_are_insufficient(self):
        scans = line_receipts() + line_receipts(2)
        split = []
        for r in scans:
            ranges = list(r.ranges_m)
            ranges[15] = None
            split.append(replace(r, ranges_m=tuple(ranges)))
        self.assertFalse(self.derive(split)[0])
        # A few good scans among mostly missing returns do not establish a view.
        bad = [replace(r, receipt_id=r.receipt_id+"bad", scan_stamp_sec=r.scan_stamp_sec+1,
                       ranges_m=(None,)*len(r.ranges_m)) for r in scans]
        self.assertFalse(self.derive(scans + tuple(bad))[0])

    def test_moving_scan_window_is_not_treated_as_stopped_consensus(self):
        scans = list(line_receipts() + line_receipts(2))
        r = scans[0]
        pose = replace(r.scan_pose_map, x_m=r.scan_pose_map.x_m+.03)
        scans[0] = replace(r, scan_pose_map=pose,
                           frame_provenance=replace(r.frame_provenance, canonical_scan_pose_odom=pose))
        self.assertFalse(self.derive(scans)[0])

    def test_missing_or_other_frame_and_other_map_do_not_steer(self):
        for change in ({"frame_provenance": None}, {"map_bundle_sha256": "d"*64}, {"survey_id": "other"}):
            scans = tuple(replace(r, **change) for r in line_receipts() + line_receipts(2))
            self.assertFalse(self.derive(scans)[0])

    def test_binding_rejects_moved_snapshot_and_other_candidate(self):
        snapshot, registry, frame = hint_fixture()
        hints, _ = self.derive(line_receipts() + line_receipts(2))
        with self.assertRaises(ValueError):
            hints["candidate_1"].normals(snapshot, "other")
        changed = replace(snapshot, created_unix_sec=40.)
        with self.assertRaises(ValueError):
            hints["candidate_1"].normals(changed, "candidate_1")
        moved = replace(snapshot, candidates=(replace(snapshot.candidates[0], geometry=replace(snapshot.candidates[0].geometry, x_m=.1)),))
        with self.assertRaises(ValueError):
            self.derive(line_receipts(), (moved, registry, frame))

    def test_ambiguous_target_association_rejected(self):
        snapshot, registry, frame = hint_fixture()
        other = replace(registry.candidates[0], candidate_uid="other", source_observation_ids=("other_obs",))
        registry = replace(registry, candidates=registry.candidates + (other,))
        candidate = snapshot.candidates[0]
        snapshot = replace(snapshot, candidates=(replace(candidate, source=replace(candidate.source,
                           source_artifact_sha256=stand_survey_registry_sha256(registry))),))
        self.assertFalse(self.derive(line_receipts() + line_receipts(2), (snapshot, registry, frame))[0])


if __name__ == "__main__":
    unittest.main()
