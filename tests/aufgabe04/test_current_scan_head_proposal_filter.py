"""Rough head screening stays conservative; measured heads require live scans."""

import copy
from dataclasses import replace
import math
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
from scripts.aufgabe04.real_robot.observer.current_scan_head_proposal_filter import (
    CurrentScanHeadProposalFilter,
)
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext, StoppedScanTargetPersistence,
)


MODULE = "scripts.aufgabe04.real_robot.observer.current_scan_head_proposal_filter"
CAMERA_TO_SCAN = RigidTransform("scan", "camera", (0., 0., 0.), (.5, -.5, .5, -.5))


def proposal_at(u=320., v=240.):
    corners = tuple(ImagePoint(u + du, v + dv)
                    for du, dv in ((-50., -50.), (50., -50.), (50., 50.), (-50., 50.)))
    return HeadProposal(corners, (int(u-62), int(v-62), int(u+63), int(v+96)),
                        (u-50., v-50., u+50., v+50.), u, v, 100., 1., 0., .99, .99)


def centered_scan(bearing=0., stamp=10.):
    return PlainLaserScan((math.inf, math.inf, .60, .601, .60, math.inf, math.inf),
                          bearing-math.radians(3), math.radians(1), .1, 3.5,
                          "scan", stamp, stamp+.01)


class CurrentScanHeadProposalFilterTest(unittest.TestCase):
    def make_filter(self, **changes):
        options = dict(intrinsics=CameraIntrinsics(640, 480, 520., 515., 320., 240.),
                       scan_from_camera=CAMERA_TO_SCAN, scan=centered_scan(),
                       map_bearing_rad=0., cone_half_angle_rad=math.radians(3),
                       accepted_range_m=(.49, .71), now_sec=10.1, max_scan_age_sec=.5,
                       min_cluster_sample_count=2,
                       max_camera_map_bearing_delta_rad=math.radians(12))
        return CurrentScanHeadProposalFilter(**{**options, **changes})

    def test_measured_head_requires_valid_fresh_unique_current_scan(self):
        scan = centered_scan()
        cases = (
            (scan, True, None),
            (None, False, "no_scan"),
            (replace(scan, receipt_sec=9.), False, "stale_scan"),
            (replace(scan, angle_increment=0.), False, "invalid_scan_geometry"),
            (replace(scan, ranges=(math.inf,)*7), False, "no_valid_samples_in_map_cone"),
        )
        for current, accepted, reason in cases:
            with self.subTest(reason=reason):
                screen = self.make_filter(scan=current)
                self.assertTrue(screen.preview(proposal_at()))
                self.assertEqual(screen(proposal_at()), accepted)
                diagnostic = screen.metadata()
                self.assertEqual(diagnostic["measured_associations"][-1]["reason"], reason)
                self.assertEqual(diagnostic["measured_scan_previews"], 1)
                self.assertEqual(diagnostic["measured_scan_rejections"], int(not accepted))
                self.assertTrue(diagnostic["final_association_required"])
                self.assertFalse(diagnostic["motion_authorized"])

    def test_rough_preview_never_clusters_scan_reads_clock_or_calls_persistence(self):
        persistence = Mock(side_effect=AssertionError("rough hints cannot touch persistence"))
        clock = Mock(side_effect=AssertionError("rough hints do not need a scan clock"))
        screen = self.make_filter(preview_lidar_association=persistence, current_ros_sec=clock,
                                  scan=None, max_camera_map_bearing_delta_rad=math.radians(3))
        with patch(MODULE + ".associate_camera_registered_candidate_lidar_target",
                   side_effect=AssertionError("rough hints cannot query scan clusters")):
            self.assertTrue(screen.preview(proposal_at()))
            self.assertFalse(screen.preview(proposal_at(420.)))
        persistence.assert_not_called()
        clock.assert_not_called()
        self.assertEqual(screen.metadata()["rough_rejections"], 1)
        self.assertEqual(screen.metadata()["measured_scan_previews"], 0)

    def test_refinement_can_move_initially_rejected_center_inside_map_gate(self):
        # Eight pixels of actual refinement cross both a tighter configured
        # gate and the ordinary twelve-degree certified registration limit.
        for limit in (3., 12.):
            with self.subTest(limit_deg=limit):
                u = 320. + 520. * math.tan(math.radians(limit + .5))
                seed, measured = proposal_at(u), proposal_at(u-8.)
                bearing = -math.atan((measured.center_u_px-320.) / 520.)
                screen = self.make_filter(scan=centered_scan(bearing),
                    max_camera_map_bearing_delta_rad=math.radians(limit))
                self.assertFalse(screen(seed))
                self.assertTrue(screen.preview(seed))
                self.assertTrue(screen(measured), screen.metadata())
                self.assertEqual(screen.metadata()["rough_envelope_fallbacks"], 0)

    def test_rough_envelope_retains_every_admissible_sampled_refinement_center(self):
        # Independently check actual shifted rays throughout the continuous
        # center corridor, including a seed outside the certified map bound.
        seed = proposal_at(320. + 520. * math.tan(math.radians(12.5)))
        admitted = 0
        for du in (-10., -7.5, -2.5, 0., 2.5, 7.5, 10.):
            for dv in (-10., -3.5, 0., 3.5, 10.):
                measured = proposal_at(seed.center_u_px+du, seed.center_v_px+dv)
                bearing = -math.atan((measured.center_u_px-320.) / 520.)
                screen = self.make_filter(scan=centered_scan(bearing))
                if screen(measured):
                    admitted += 1
                    self.assertTrue(screen.preview(seed), (du, dv, screen.metadata()))
        self.assertGreater(admitted, 0)
        self.assertLess(admitted, 35)

    def test_new_clock_read_rejects_scan_that_expires_after_rough_preview(self):
        clock = Mock(return_value=10.1)
        screen = self.make_filter(current_ros_sec=clock)
        self.assertTrue(screen.preview(proposal_at()))
        clock.assert_not_called()
        self.assertTrue(screen(proposal_at()))
        clock.return_value = 10.6
        self.assertFalse(screen(proposal_at()))
        self.assertEqual(clock.call_count, 2)
        self.assertEqual(screen.metadata()["measured_associations"][-1]["reason"], "stale_scan")

    def test_angular_envelope_and_exact_scan_cross_the_pi_seam(self):
        # Turn the optical-to-scan transform 180 degrees about scan z.
        transform = replace(CAMERA_TO_SCAN, rotation_xyzw=(.5, .5, -.5, -.5))
        measured = proposal_at(310.)
        bearing = -math.pi + math.atan(10./520.)
        screen = self.make_filter(scan_from_camera=transform,
                                  scan=centered_scan(bearing), map_bearing_rad=math.pi-.01,
                                  max_camera_map_bearing_delta_rad=math.radians(3))
        self.assertTrue(screen.preview(measured))
        self.assertTrue(screen(measured), screen.metadata())
        self.assertFalse(screen.preview(proposal_at(220.)))
        self.assertEqual(screen.metadata()["rough_envelope_fallbacks"], 0)

    def test_singular_or_large_envelopes_preserve_rough_hypotheses(self):
        # Identity orientation points the central optical ray along scan z.
        # Nearby image points surround the singular ray; no angular interval
        # smaller than a half-plane safely encloses the ten-pixel square.
        transform = replace(CAMERA_TO_SCAN, rotation_xyzw=(0., 0., 0., 1.))
        for measured in (proposal_at(), proposal_at(321., 240.)):
            with self.subTest(center=measured.center_u_px):
                screen = self.make_filter(scan_from_camera=transform)
                self.assertTrue(screen.preview(measured))
                self.assertEqual(screen.metadata()["rough_envelope_fallbacks"], 1)
        singular = self.make_filter(scan_from_camera=transform)
        self.assertFalse(singular(proposal_at()))
        self.assertEqual(singular.metadata()["measured_associations"][-1]["reason"],
                         "current_head_projection_invalid")

    def test_read_only_persistence_retains_witnesses_across_competing_heads(self):
        state = StoppedScanTargetPersistence()
        pose = Pose2D(.8, -.2, math.pi)

        def context_at(stamp):
            return ScanPersistenceContext("candidate_1", "stopped_1", pose, pose, stamp,
                                          .2, -.2, .06, .02, .04, Pose2D(0., 0., 0.))

        def scan_at(stamp, fragmented=False):
            ranges = (math.inf, math.inf, math.inf, .6, .601,
                      math.nan if fragmented else .602, .603, math.inf, math.inf)
            return PlainLaserScan(ranges, -.05, .01, .1, 3., "scan", stamp, stamp+.01)

        for stamp in (10., 10.2, 10.4):
            self.assertTrue(state.ingest_scan(scan_at(stamp), context=context_at(stamp),
                                             now_sec=stamp+.1, max_scan_age_sec=.5))
        scan, context = scan_at(10.6, fragmented=True), context_at(10.6)
        history = copy.deepcopy(state._history)
        pending = copy.deepcopy(list(state._pending_scans._entries))
        metadata = copy.deepcopy(state.last_metadata)
        preview = Mock(side_effect=lambda association, current_scan: state.preview(
            association, current_scan, context=context, now_sec=10.7, max_scan_age_sec=.5))
        screen = self.make_filter(scan=scan, accepted_range_m=(.42, .64), now_sec=10.7,
                                  min_cluster_sample_count=1, preview_lidar_association=preview)
        self.assertTrue(screen.preview(proposal_at()))
        preview.assert_not_called()
        # This off-target measured head must not consume the scan-only witnesses
        # needed to establish that the actual head's split returns are one stand.
        self.assertFalse(screen(proposal_at(380.)))
        self.assertTrue(screen(proposal_at()), screen.metadata())
        self.assertEqual(preview.call_count, 2)
        self.assertEqual(state._history, history)
        self.assertEqual(list(state._pending_scans._entries), pending)
        self.assertEqual(state.last_metadata, metadata)
        self.assertIsNone(state._last_stamp)
        without_witnesses = self.make_filter(scan=scan, accepted_range_m=(.42, .64),
                                             now_sec=10.7, min_cluster_sample_count=1)
        self.assertFalse(without_witnesses(proposal_at()))


if __name__ == "__main__":
    unittest.main()
