"""Measured-center yaw geometry and motion-neutral receipt boundaries."""
from dataclasses import replace
import hashlib
import json
import math
import unittest

from scripts.aufgabe04.perception.stand_axis_handoff.models import RigidTransform
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
from scripts.aufgabe04.real_robot.observer.candidate_centering import (
    MAX_CENTERING_STEP_RAD, MAX_CENTERING_TRAVEL_RAD,
    build_camera_centering_advisory, center_point_in_base, project_center_after_turn,
    solve_camera_centering, validate_camera_centering_advisory,
)
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
from tests.aufgabe04 import test_current_head_association as association_fixtures


class CenteringGeometryTests(unittest.TestCase):
    def options(self, center=(300., 300.)):
        optical = RigidTransform('base_scan', 'camera', (0., 0., 0.), (-.5, .5, -.5, .5))
        return dict(center_px=center, intrinsics=CameraIntrinsics(800, 600, 600., 600., 400., 300.),
                    distance_m=.55, scan_from_camera=optical,
                    base_from_camera=replace(optical, parent_frame='base_footprint'))

    def test_turn_sign_and_analytic_projection(self):
        for u in (300., 400., 500.):
            with self.subTest(u=u):
                options = self.options((u, 300.))
                yaw = solve_camera_centering(**options)
                self.assertAlmostEqual(yaw, math.atan2(400.-u, 600.), places=12)
                point = center_point_in_base(**options)
                pixels = project_center_after_turn(point_base=point, yaw_rad=yaw,
                    intrinsics=options['intrinsics'], base_from_camera=options['base_from_camera'])
                self.assertAlmostEqual(pixels[0], 400., places=10)

    def test_image_center_not_principal_point_and_deadband(self):
        options = self.options((380., 300.))
        options['intrinsics'] = replace(options['intrinsics'], cx_px=405.87)
        yaw = solve_camera_centering(**options)
        self.assertGreater(yaw, 0.)
        point = center_point_in_base(**options)
        self.assertAlmostEqual(project_center_after_turn(point_base=point, yaw_rad=yaw,
            intrinsics=options['intrinsics'], base_from_camera=options['base_from_camera'])[0], 400.)
        for u in (390., 400., 410.):
            self.assertEqual(solve_camera_centering(**self.options((u, 300.))), 0.)

    def test_recorded_mount_requires_10_point_54_not_ray_only_12_degrees(self):
        q = (-.4671136171146707, .4831412431691032, -.5243683107682287, .5228931846153062)
        scan = RigidTransform('base_scan', 'camera',
                             (.07753724525972719, -.004971307208605384, -.05625336468095905), q)
        base = replace(scan, parent_frame='base_footprint',
                       translation_xyz_m=(.04553724525972719, -.004971307208605384, .12574663531904096))
        intrinsics = CameraIntrinsics(800, 600, 640.977317, 641.1612206, 405.8708878, 300.7059997)
        options = dict(center_px=(270.0519349202243, 287.48097054835625),
                       intrinsics=intrinsics, distance_m=.5590000152587891,
                       scan_from_camera=scan, base_from_camera=base)
        yaw = solve_camera_centering(**options)
        self.assertAlmostEqual(math.degrees(yaw), 10.53857197, places=6)
        point = center_point_in_base(**options)
        after = project_center_after_turn(point_base=point, yaw_rad=MAX_CENTERING_STEP_RAD,
                                         intrinsics=intrinsics, base_from_camera=base)
        self.assertAlmostEqual(after[0], 344.6483818, places=5)
        # q and -q are the same mount, not a second calibration.
        changed = {**options, 'scan_from_camera': replace(scan, rotation_xyzw=tuple(-x for x in q)),
                   'base_from_camera': replace(base, rotation_xyzw=tuple(-x for x in q))}
        self.assertAlmostEqual(solve_camera_centering(**changed), yaw, places=12)

    def test_invalid_or_unreachable_geometry_rejected(self):
        for changes in ({'center_px': (100., 300.)}, {'center_px': (math.nan, 300.)},
                {'center_px': (-1., 300.)}, {'center_px': (300., 600.)}, {'distance_m': 0.},
                {'distance_m': math.inf}, {'remaining_rotation_rad': 0.},
                {'remaining_rotation_rad': math.radians(13)},
                {'scan_from_camera': RigidTransform('base_scan', 'other_camera', (0., 0., 0.), (-.5,.5,-.5,.5))},
                {'base_from_camera': RigidTransform('base', 'camera', (0.,0.,0.), (0.,0.,0.,0.))}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                solve_camera_centering(**{**self.options(), **changes})

    def test_remaining_budget_is_a_hard_bound(self):
        with self.assertRaises(ValueError):
            solve_camera_centering(**self.options(), remaining_rotation_rad=math.radians(5.))
        self.assertLess(abs(solve_camera_centering(**self.options((360.,300.)),
            remaining_rotation_rad=math.radians(6.))), math.radians(6.))


class CenteringAdvisoryTests(unittest.TestCase):
    def options(self):
        fixture = association_fixtures.CurrentHeadAssociationTests().options()
        association = associate_current_measured_head(**fixture)
        self.assertTrue(association.accepted)
        return dict(association=association, intrinsics=fixture['intrinsics'],
            scan_from_camera=fixture['scan_from_camera'],
            base_from_camera=replace(fixture['scan_from_camera'], parent_frame='base_footprint'),
            candidate_uid='candidate_1', target_key='session:candidate_1', stream_id='session',
            planning_frame='map', motion_epoch=2, anchor_pose=EvidencePose(1., 2., .3),
            anchor_odom_pose=EvidencePose(.1, .2, .4), odom_stamp_sec=10.,
            image_stamp_sec=10.01, now_sec=10.1, robot_profile_sha256='a'*64,
            calibration_profile_sha256='b'*64, stand_model_profile_sha256='c'*64)

    def test_json_round_trip_hash_and_candidate_bindings(self):
        advisory = build_camera_centering_advisory(**self.options())
        self.assertIsNotNone(advisory)
        payload = json.loads(json.dumps(advisory.metadata()))
        validated = validate_camera_centering_advisory(payload, candidate_uid='candidate_1',
            target_key='session:candidate_1', stream_id='session', calibration_profile_sha256='b'*64,
            now_sec=12., min_image_stamp_sec=9.9, min_scan_stamp_sec=9.9)
        self.assertEqual(validated, advisory)
        self.assertFalse(payload['motion_authorized'])
        self.assertFalse(payload['completion_authorized'])
        self.assertEqual(payload['eligible_cluster_count'], 1)
        self.assertGreater(advisory.requested_yaw_rad, 0.)
        self.assertEqual(advisory.required_yaw_rad, advisory.requested_yaw_rad)
        for changes in ({'candidate_uid':'other'}, {'target_key':'other'}, {'stream_id':'other'},
                {'robot_profile_sha256':'d'*64}, {'calibration_profile_sha256':'d'*64},
                {'stand_model_profile_sha256':'d'*64}, {'now_sec':16.}, {'now_sec':9.},
                {'min_image_stamp_sec':10.01}, {'min_scan_stamp_sec':10.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_camera_centering_advisory(payload, **changes)

    def test_required_yaw_capped_per_step_without_erasing_total(self):
        options = self.options()
        options['association'] = replace(options['association'], full_image_center_px=(280.,300.))
        advice = build_camera_centering_advisory(**options)
        self.assertIsNotNone(advice)
        self.assertAlmostEqual(advice.requested_yaw_rad, MAX_CENTERING_STEP_RAD)
        self.assertGreater(advice.required_yaw_rad, advice.requested_yaw_rad)
        self.assertLess(advice.required_yaw_rad, MAX_CENTERING_TRAVEL_RAD)
        self.assertIsNone(build_camera_centering_advisory(**options,
            consumed_rotation_rad=MAX_CENTERING_STEP_RAD, completed_turn_count=1))

    def test_no_angle_needed_when_independent_head_bounds_are_available(self):
        options = self.options()
        association = options['association']
        # The upstream association has already validated the orientation-bound proof.
        bounded = replace(association, head_admission=replace(association.head_admission,
                          accepted=False), head_orientation_bounds=object())
        self.assertIsNotNone(build_camera_centering_advisory(**{**options, 'association':bounded}))
        self.assertIsNone(build_camera_centering_advisory(**{**options,
            'association':replace(bounded,head_orientation_bounds=None)}))

    def test_sensor_identity_epoch_and_budget_fail_closed(self):
        options = self.options()
        for changes in ({'image_stamp_sec':9.}, {'image_stamp_sec':10.2}, {'odom_stamp_sec':9.8},
                {'odom_stamp_sec':10.2}, {'now_sec':10.6}, {'candidate_uid':''},
                {'robot_profile_sha256':'not-a-profile'}, {'motion_epoch':-1},
                {'anchor_odom_pose':EvidencePose(math.nan,0.,0.)}, {'max_age_sec':1.},
                {'max_image_scan_skew_sec':1.}, {'completed_turn_count':2},
                {'completed_turn_count':True}, {'consumed_rotation_rad':MAX_CENTERING_TRAVEL_RAD},
                {'consumed_rotation_rad':-1.}):
            with self.subTest(changes=changes):
                self.assertIsNone(build_camera_centering_advisory(**{**options, **changes}))

    def test_no_advice_for_deadband_or_missing_ambiguous_head(self):
        options = self.options()
        association = options['association']
        search = association.lidar_association.search_association
        for changed in (replace(association, accepted=False),
                replace(association, full_image_center_px=(400.,300.)),
                replace(association, lidar_association=None),
                replace(association, lidar_association=replace(association.lidar_association,
                    search_association=replace(search, eligible_cluster_count=2))),
                replace(association, lidar_association=replace(association.lidar_association,
                    search_association=replace(search, scan_frame_id='other_scan')))):
            with self.subTest(association=changed):
                self.assertIsNone(build_camera_centering_advisory(**{**options,'association':changed}))

    def test_hash_and_rehashed_geometry_tampering_rejected(self):
        payload = build_camera_centering_advisory(**self.options()).metadata()
        with self.assertRaises(ValueError):
            validate_camera_centering_advisory({**payload,'candidate_uid':'changed'})
        for updates in ({'motion_authorized':True}, {'requested_yaw_rad':-.1},
                {'required_yaw_rad':.2}, {'eligible_cluster_count':2},
                {'deadband_px':100.}, {'created_at_sec':20.}, {'unexpected':'field'}):
            changed = {key:value for key,value in payload.items()
                       if key != 'camera_centering_advisory_sha256'}
            changed.update(updates)
            changed['camera_centering_advisory_sha256'] = hashlib.sha256(json.dumps(changed,
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_camera_centering_advisory(changed)


if __name__ == '__main__':
    unittest.main()
