"""A decoded neighbor inside a search crop cannot lend target identity."""

from dataclasses import replace
import json
import math
import unittest

from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose, PassiveObserverEvidence
from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext, StoppedScanTargetPersistence, registered_target_metadata_is_unique,
)


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

    def witnessed_resolver(self):
        state = StoppedScanTargetPersistence()
        context = ScanPersistenceContext('candidate1', '0', Pose2D(0, 0, 0),
            Pose2D(0, 0, 0), 10., .76, 0., .1, .02, .04, Pose2D(0, 0, 0))
        for stamp in (9.4, 9.6, 9.8):
            state.ingest_scan(self.scan(ranges=(.7,) * 5, stamp=stamp),
                context=replace(context, image_stamp_sec=stamp), now_sec=stamp + .05,
                max_scan_age_sec=.5)
        return lambda a, s: state.resolve(a, s, context=context, now_sec=10., max_scan_age_sec=.5)

    def test_registered_qr_uses_witnessed_internal_fragments_without_parallax(self):
        scan = self.scan(ranges=(.7, .7, math.nan, .7, .7))
        self.assertFalse(self.bind((self.observation(),), scan=scan, registered=True).accepted)
        binding = self.bind((self.observation(),), scan=scan, registered=True,
                            resolve_lidar_association=self.witnessed_resolver())
        self.assertTrue(binding.accepted, binding.reason)
        self.assertEqual(binding.qr_texts_for_evidence, ('Start',))
        self.assertTrue(registered_target_metadata_is_unique(binding.association))
        cluster = binding.association['search_association']
        self.assertEqual(cluster['eligible_cluster_count'], 2)
        self.assertEqual(cluster['selected_cluster_source_indices'], (0, 1, 3, 4))
        from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
            build_qr_verified_observation_pose, validate_qr_verified_observation_pose, SOURCE_GATES,
        )
        receipt = build_qr_verified_observation_pose(candidate_uid='candidate1', stream_id='test',
            qr_id='Start', planning_frame='map', stand_center=dict(x_m=.76, y_m=0.),
            robot_pose=dict(x_m=0., y_m=0., yaw_rad=0.), sensor_stamp_sec=10., scan_stamp_sec=10.,
            checked_at_sec=10., robot_profile_sha256='a'*64, calibration_profile_sha256='b'*64,
            stand_model_profile_sha256='c'*64, target_key='candidate1', motion_epoch=0,
            camera_signature=(400., 400., 400., 300.),
            qr_corners_px=tuple((u+self.roi.x0, v+self.roi.y0) for u,v in self.observation().corners),
            image_shape=(600, 800), qr_binding=binding.metadata(), source_gates={k:True for k in SOURCE_GATES},
            localization_provenance=dict(map_frame='map', base_frame='base_footprint',
                scan_frame='base_scan', camera_frame='camera_optical',
                exact_image_transform_stamp_sec=10., exact_scan_transform_stamp_sec=10.))
        persisted = validate_qr_verified_observation_pose(json.loads(json.dumps(receipt, allow_nan=False)))
        self.assertTrue(persisted['completion_authorized'])
        self.assertFalse(persisted['facing_ready'])
        self.assertFalse(persisted['motion_authorized'])

    def test_qr_witnesses_do_not_override_a_finite_competitor_or_independent_envelope(self):
        for intervening in (.9, .55):
            with self.subTest(intervening=intervening):
                binding = self.bind((self.observation(),), registered=True,
                    scan=self.scan(ranges=(.7, .7, intervening, .7, .7)),
                    resolve_lidar_association=self.witnessed_resolver())
                self.assertFalse(binding.accepted)
        binding = self.bind((self.observation(),), allow_independent_registration=True,
            scan=self.scan(ranges=(.7, .7, math.nan, .7, .7)),
            resolve_lidar_association=self.witnessed_resolver())
        self.assertFalse(binding.accepted)
        self.assertEqual(binding.reason, 'independent_qr_registration_envelope_not_unique')

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
