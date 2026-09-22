"""A decoded neighbor inside a search crop cannot lend target identity."""

from dataclasses import replace
import math
import unittest

from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose, PassiveObserverEvidence
from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target


class QrTargetBindingTest(unittest.TestCase):
    def setUp(self):
        self.roi = ImageRoi(280, 200, 540, 400, 100)
        self.intrinsics = CameraIntrinsics(800, 600, 400, 400, 400, 300)
        self.transform = RigidTransform(
            "base_scan", "camera_optical", (0, 0, 0), (-.5, .5, -.5, .5),
        )
        self.window = PassiveObserverEvidence(
            target_key="candidate1/stream1/profile1", anchor_pose=EvidencePose(0, 0, 0),
            required_axis_samples=7, max_axis_deviation_rad=.1,
        )

    def observation(self, bearing_deg=0, *, text="Start", corners=True):
        u = 400 - 400 * math.tan(math.radians(bearing_deg)) - self.roi.x0
        v = 300 - self.roi.y0
        quad = ((u - 15, v - 15), (u + 15, v - 15),
                (u + 15, v + 15), (u - 15, v + 15)) if corners else None
        return DecodedQrObservation(text, quad, "fixture")

    def scan(self, bearing_deg=0, *, ranges=(.7, .7, .7), stamp=10):
        return PlainLaserScan(
            tuple(ranges), math.radians(bearing_deg - .5), math.radians(.5),
            .1, 3.5, "base_scan", stamp, stamp,
        )

    def bind(self, observations, *, scan=None, registered=False, now=10, **extra):
        options = dict(
            roi=self.roi, intrinsics=self.intrinsics, scan_from_camera=self.transform,
            scan=self.scan() if scan is None else scan,
            map_bearing_rad=0, cone_half_angle_rad=math.radians(3),
            accepted_range_m=(.5, .8), now_sec=now, max_scan_age_sec=.5,
            min_cluster_sample_count=2, camera_registration_accepted=registered,
            max_camera_map_bearing_delta_rad=math.radians(12),
        )
        options.update(extra)
        return bind_qr_observations_to_target(observations, **options)

    def record(self, binding, stamp, *, associated=True, axis=False, **extra):
        return self.window.record_frame(
            target_key=self.window.target_key, pose=EvidencePose(0, 0, 0),
            frame_stamp_sec=stamp, lidar_stamp_sec=stamp, observed_at_sec=stamp,
            lidar_associated=associated,
            qr_texts=binding.qr_texts_for_evidence, qr_symbol_count=binding.symbol_count,
            axis_yaw_rad=.1 if axis else None, axis_source="front" if axis else None,
            **extra,
        )

    def test_neighbor_qr_inside_crop_with_valid_map_lidar_never_latches(self):
        scan = self.scan()
        preliminary = associate_candidate_lidar_target(
            scan, map_bearing_rad=0, cone_half_angle_rad=math.radians(3),
            accepted_range_m=(.5, .8), min_cluster_sample_count=2,
        )
        self.assertTrue(preliminary.associated)
        binding = self.bind((self.observation(8),), scan=scan)
        self.assertFalse(binding.accepted)
        self.assertEqual(binding.reason, "camera_bearing_outside_map_cone")
        for i in range(8):
            update = self.record(binding, 10 + i / 3, axis=True)
            self.assertIsNone(update.resolved_qr_id)
            self.assertEqual(update.snapshot.current_qr_sample_count, 0)
        # Separately valid target axes can survive without borrowing its text.
        self.assertEqual(update.axis_consensus.sample_count, 7)

    def test_nominal_own_quad_restores_roi_offset_and_needs_two_frames(self):
        binding = self.bind((self.observation(),))
        self.assertTrue(binding.accepted)
        self.assertAlmostEqual(binding.camera_bearing_rad, 0)
        self.assertIsNone(self.record(binding, 10).resolved_qr_id)
        self.assertEqual(self.record(binding, 10.2).resolved_qr_id, "Start")

    def test_registered_own_quad_uses_bounded_narrow_cone_without_axis(self):
        observation = self.observation(9)
        scan = self.scan(9)
        self.assertFalse(self.bind((observation,), scan=scan).accepted)
        binding = self.bind((observation,), scan=scan, registered=True)
        self.assertTrue(binding.accepted)
        self.assertEqual(binding.association["search_bearing_source"], "registered_camera_bearing")
        self.record(binding, 10)
        update = self.record(binding, 10.2)
        self.assertEqual(update.resolved_qr_id, "Start")
        self.assertIsNone(update.axis_consensus)

    def test_registered_correction_cannot_exceed_existing_bound(self):
        binding = self.bind((self.observation(14),), scan=self.scan(14), registered=True)
        self.assertFalse(binding.accepted)
        self.assertEqual(binding.reason, "camera_map_bearing_delta_exceeds_limit")

    def test_range_staleness_and_unique_cluster_gates_remain_required(self):
        for registered in (False, True):
            with self.subTest(registered=registered):
                self.assertFalse(self.bind((self.observation(),), registered=registered,
                    scan=self.scan(ranges=(.9, .9, .9))).accepted)
                self.assertFalse(self.bind((self.observation(),), registered=registered,
                    now=11).accepted)
                self.assertFalse(self.bind((self.observation(),), registered=registered,
                    scan=self.scan(ranges=(.7, .7, math.inf, .7, .7))).accepted)

    def test_missing_or_out_of_crop_geometry_and_legacy_text_cannot_bind(self):
        outside = replace(self.observation(), corners=((0., 0.), (300., 0.), (300., 40.), (0., 40.)))
        for observations in (None, (), (self.observation(corners=False),), (outside,)):
            binding = self.bind(observations)
            self.assertFalse(binding.accepted)
            self.record(binding, 10)
            self.assertIsNone(self.record(binding, 10.2).resolved_qr_id)

    def test_same_text_multiple_symbols_poison_before_axis_or_qr_authority(self):
        binding = self.bind((self.observation(), self.observation(1)))
        self.assertEqual(binding.symbol_count, 2)
        self.assertFalse(binding.accepted)
        update = self.record(binding, 10, axis=True)
        self.assertFalse(update.frame_accepted)
        self.assertEqual(update.reason, "multiple_qr_symbols_in_associated_frame")
        self.assertTrue(update.snapshot.poisoned)
        self.assertIsNone(self.record(self.bind((self.observation(),)), 10.2).resolved_qr_id)

    def test_unassociated_multiple_symbols_do_not_poison_target_epoch(self):
        ambiguous = self.bind((self.observation(), self.observation(1)))
        update = self.record(ambiguous, 10, associated=False)
        self.assertFalse(update.snapshot.poisoned)
        valid = self.bind((self.observation(),))
        self.record(valid, 10.2)
        self.assertEqual(self.record(valid, 10.4).resolved_qr_id, "Start")

    def test_expected_identity_mismatch_poison_never_latches_wrong_identity(self):
        binding = self.bind((self.observation(text="QR_002"),))
        update = self.record(binding, 10, expected_qr_id="Start")
        self.assertEqual(update.reason, "associated_qr_identity_differs_from_expected")
        self.assertTrue(update.snapshot.poisoned)
        self.assertIsNone(update.resolved_qr_id)
        self.assertEqual(update.snapshot.current_qr_sample_count, 0)


if __name__ == "__main__":
    unittest.main()


class IndependentQrRegistrationTest(unittest.TestCase):
    setUp = QrTargetBindingTest.setUp
    observation = QrTargetBindingTest.observation
    scan = QrTargetBindingTest.scan
    bind = QrTargetBindingTest.bind

    def test_independent_quad_can_register_without_a_head(self):
        binding = self.bind((self.observation(9),), scan=self.scan(9), allow_independent_registration=True)
        self.assertTrue(binding.accepted)
        self.assertFalse(binding.independent_registration['head_geometry_required'])

    def test_competitor_outside_narrow_qr_cone_but_inside_envelope_vetoes(self):
        scan = self.scan(-.5, ranges=(.7, .7, math.inf, math.inf, math.inf, .7, .7))
        scan = replace(scan, angle_min=0., angle_increment=math.radians(1.5))
        binding = self.bind((self.observation(9),), scan=scan, allow_independent_registration=True)
        self.assertFalse(binding.accepted)
        self.assertEqual(binding.reason, 'independent_qr_registration_envelope_not_unique')
        self.assertEqual(binding.independent_registration['envelope']['eligible_cluster_count'], 2)

    def test_independent_registration_cannot_bind_a_neighbor_ray_to_nominal_scan(self):
        self.assertFalse(self.bind((self.observation(9),), allow_independent_registration=True).accepted)

    def test_independent_registration_keeps_range_and_bearing_bounds(self):
        for bearing, ranges in ((13., (.7, .7, .7)), (9., (.9, .9, .9))):
            self.assertFalse(self.bind((self.observation(bearing),), scan=self.scan(bearing, ranges=ranges),
                                      allow_independent_registration=True).accepted)

    def test_frame_mismatch_and_stale_source_with_fresh_receipt_rejected(self):
        scan = replace(self.scan(9), scan_frame_id='different')
        self.assertFalse(self.bind((self.observation(9),), scan=scan, allow_independent_registration=True).accepted)
        scan = replace(self.scan(9, stamp=9.), receipt_sec=10.)
        self.assertFalse(self.bind((self.observation(9),), scan=scan, allow_independent_registration=True).accepted)
