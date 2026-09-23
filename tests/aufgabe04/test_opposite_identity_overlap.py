"""A background rectangle must not cut off the target's bottom QR modules."""
from dataclasses import replace
import unittest
import cv2
from tests.aufgabe04.opposite_overlap_fixture import recorded_options
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.opposite_target_support import detect_opposite_target_support
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop
from scripts.aufgabe04.qr_scanning.isolated_qr_views import rectify_isolated_qr_view, ISOLATED_QR_VIEWS
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr


class RecordedOppositeOverlapTests(unittest.TestCase):
    def setUp(self):
        self.frame,self.options,self.snapshot,self.tf,self.data=recorded_options()
        self.search=current_scan_qr_search(**self.options)
        keep=('intrinsics','model_profile','scan','image_stamp_sec','now_sec','map_bearing_rad',
              'cone_half_angle_rad','accepted_range_m','max_scan_age_sec','max_camera_map_bearing_delta_rad')
        self.support=detect_opposite_target_support(self.frame,cv2,attempt=self.search[0],
            scan_from_camera=self.tf('base_scan','camera'),**{k:self.options[k] for k in keep},max_elapsed_sec=.5)

    def crop(self, support=True):
        return exclusive_identity_crop(candidate_uid=self.data['candidate_uid'],snapshot=self.snapshot,
            support=self.support if support else None,search_result=self.search,**self.options)

    def test_real_background_overlap_keeps_entire_current_symbol(self):
        self.assertIsNotNone(self.support)
        attempt,proof=self.crop()
        self.assertIsNotNone(attempt)
        self.assertGreater(attempt.roi.y1,self.data['original_crop'][3])
        self.assertEqual(proof['sampling'],'isolated_current_qr_quad')
        self.assertTrue(any(p['occluded_by_target_symbol'] for p in proof['competitors']))

    @unittest.skipUnless(hasattr(cv2, 'wechat_qrcode_WeChatQRCode'), 'requires deployed WeChat backend')
    def test_complete_isolated_symbol_decodes_start(self):
        self.assertIsNotNone(self.support)
        image=rectify_isolated_qr_view(self.frame,self.support.corners_px,cv2,ISOLATED_QR_VIEWS[0])
        decoded=detect_qr_observations_bgr(image,cv2,max_elapsed_sec=.5)
        self.assertEqual([o.text for o in decoded],['Start'])

    def test_receipt_keeps_complete_outline_and_requires_separated_background(self):
        from copy import deepcopy
        from scripts.aufgabe04.artifacts.qr_verified_observation_pose import _validate_opposite_crop
        _,proof=self.crop()
        data=dict(candidate_uid=self.data['candidate_uid'],qr_binding=dict(association=proof['search']['envelope']))
        args=(data,self.data['image_stamp_sec'],self.data['scan_stamp_sec'],self.frame.shape[:2])
        _validate_opposite_crop(proof,*args)
        truncated=deepcopy(proof);truncated['bounds_xyxy'][3]=320
        with self.assertRaisesRegex(ValueError,'support differs'):
            _validate_opposite_crop(truncated,*args)
        rebound=deepcopy(proof)
        neighbor=next(p for p in rebound['competitors'] if p['occluded_by_target_symbol'])
        neighbor['depth_interval_m']=[.1,2.]
        with self.assertRaisesRegex(ValueError,'depth is not separated'):
            _validate_opposite_crop(rebound,*args)

    def test_current_outline_can_center_without_any_head_angle(self):
        from scripts.aufgabe04.real_robot.observer.candidate_centering import (
            build_camera_centering_advisory, validate_camera_centering_advisory, project_center_after_turn,
            center_point_in_base,
        )
        from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
        self.assertIsNotNone(self.support)
        advisory = build_camera_centering_advisory(association=self.support,
            intrinsics=self.options['intrinsics'], scan_from_camera=self.tf('base_scan','camera'),
            base_from_camera=self.tf('base_footprint','camera'),candidate_uid=self.data['candidate_uid'],
            target_key='target',stream_id='stream',planning_frame='map',motion_epoch=0,
            anchor_pose=EvidencePose(0.,0.,0.),anchor_odom_pose=EvidencePose(0.,0.,0.),
            odom_stamp_sec=self.data['image_stamp_sec'],image_stamp_sec=self.data['image_stamp_sec'],
            now_sec=self.options['now_sec'],robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,
            stand_model_profile_sha256=self.options['model_profile'].sha256)
        self.assertIsNotNone(advisory)
        self.assertLess(advisory.requested_yaw_rad, 0.)
        self.assertLessEqual(abs(advisory.requested_yaw_rad), .10472)
        self.assertEqual(validate_camera_centering_advisory(advisory.metadata()),advisory)
        point = center_point_in_base(center_px=self.support.full_image_center_px,
            intrinsics=advisory.intrinsics,distance_m=advisory.associated_range_m,
            scan_from_camera=advisory.scan_from_camera,base_from_camera=advisory.base_from_camera)
        u,_=project_center_after_turn(point_base=point,yaw_rad=advisory.required_yaw_rad,
            intrinsics=advisory.intrinsics,base_from_camera=advisory.base_from_camera)
        self.assertAlmostEqual(u,400.,places=6)

    def test_overlap_without_current_outline_is_recoverable_not_truncated(self):
        attempt,proof=self.crop(False)
        self.assertIsNone(attempt)
        self.assertEqual(proof['reason'],'target_crop_overlap_unresolved')

    def test_unseparated_neighbor_depth_cannot_claim_occlusion(self):
        self.assertIsNotNone(self.support)
        # Move only the support depth interval behind the recorded neighbor.
        self.support=replace(self.support,depth_m=2.)
        attempt,proof=self.crop()
        self.assertIsNone(attempt)
        self.assertEqual(proof['reason'],'target_crop_overlap_unresolved')
