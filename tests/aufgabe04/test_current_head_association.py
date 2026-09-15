"""Nominal and reacquired current heads share bounded candidate admission."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi, OpticalProjection
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.current_head_qr_binding import bind_qr_to_current_head
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target
from tests.aufgabe04.test_head_model_admission import head_estimate, head_debug, quality, outer_boundary


class CurrentHeadAssociationTests(unittest.TestCase):
    def options(self):
        corners = tuple(ImagePoint(80+x, 80+y) for x,y in ((-45,-45),(45,-45),(45,45),(-45,45)))
        return dict(
            estimate=head_estimate(corners=corners, left_height_px=90., right_height_px=90.),
            debug=head_debug(head_outer_recovery=outer_boundary(corners)), profile_sha256='a'*64,
            attempt=HeadRoiAttempt(ImageRoi(280,220,440,380,90.), 'nominal_projection', 1.8,400.,300.,90.),
            projection=OpticalProjection(400.,300.,.5,90.,True), expected_head_height_px=90.,
            intrinsics=CameraIntrinsics(800,600,640.,640.,400.,300.),
            scan_from_camera=RigidTransform('base_scan','camera',(0.,0.,0.),(-.5,.5,-.5,.5)),
            scan=PlainLaserScan((.55,)*9,math.radians(2),math.radians(.5),.1,3.,'base_scan',
                               scan_stamp_sec=10.,receipt_sec=10.),
            map_bearing_rad=0.,cone_half_angle_rad=math.radians(3),accepted_range_m=(.45,.65),
            now_sec=10.1,max_scan_age_sec=.5,min_cluster_sample_count=1,
            max_center_offset_ratio=1.5,max_camera_map_bearing_delta_rad=math.radians(12),
        )

    def test_same_full_image_head_has_same_association_without_reacquisition(self):
        o=self.options();nominal=associate_current_measured_head(**o)
        self.assertTrue(nominal.accepted,nominal.reason)
        self.assertGreater(nominal.lidar_association.camera_map_bearing_delta_rad,math.radians(3))
        self.assertEqual(nominal.full_image_center_px,(360.,300.))
        shifted=replace(o['attempt'],roi=ImageRoi(300,230,460,390,90.),
                        source='camera_registered_measured_head_reacquisition',
                        expected_center_u_px=360.,expected_center_v_px=300.)
        estimate=replace(o['estimate'],corners=tuple(ImagePoint(p.u_px-20,p.v_px-10) for p in o['estimate'].corners))
        reacquired=associate_current_measured_head(**{**o,'estimate':estimate,'attempt':shifted,
            'debug':head_debug(head_outer_recovery=outer_boundary(estimate.corners))})
        self.assertTrue(reacquired.accepted)
        self.assertEqual(nominal.lidar_association,reacquired.lidar_association)
        self.assertEqual(nominal.center_offset_ratio,reacquired.center_offset_ratio)
        self.assertEqual(nominal.roi_source,'nominal_projection')
        self.assertEqual(reacquired.roi_source,shifted.source)
        self.assertEqual(nominal.metadata()['projected_center_px'],(400.,300.))
        self.assertFalse(nominal.metadata()['motion_authorized'])

    def test_quality_profile_and_scale_cannot_lend_camera_registration(self):
        o=self.options()
        for changes in (
            {'debug':head_debug(head_model_quality=quality(outer_border_verified=False))},
            {'debug':head_debug(head_model_quality=quality(yaw_std_deg=3.1))},
            {'estimate':replace(o['estimate'],evidence_state='predicted_only')},
            {'estimate':replace(o['estimate'],source='model_current_frame_refined')},
            {'profile_sha256':'b'*64},
            {'estimate':replace(o['estimate'],left_height_px=20.,right_height_px=20.)},
            {'estimate':replace(o['estimate'],left_height_px=None)},
        ):
            with self.subTest(changes=changes), patch(
                'scripts.aufgabe04.real_robot.observer.current_head_association.associate_camera_registered_candidate_lidar_target',
                side_effect=AssertionError('unqualified geometry must not move the cone'),
            ):
                self.assertFalse(associate_current_measured_head(**{**o,**changes}).accepted)

    def test_original_projection_and_complete_current_crop_remain_required(self):
        o=self.options()
        cases=(
            {'projection':replace(o['projection'],u_px=510.)},
            {'projection':replace(o['projection'],depth_m=-.5)},
            {'projection':replace(o['projection'],u_px=math.nan)},
            {'attempt':replace(o['attempt'],roi=ImageRoi(-1,0,160,160,90.))},
            {'estimate':replace(o['estimate'],corners=tuple(ImagePoint(p.u_px+100,p.v_px) for p in o['estimate'].corners))},
            {'estimate':replace(o['estimate'],corners=(ImagePoint(math.nan,40),)*4)},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertFalse(associate_current_measured_head(**{**o,**changes}).accepted)
        # A recentered attempt cannot erase displacement from the map projection.
        moved=replace(o['attempt'],expected_center_u_px=360.,expected_center_v_px=300.)
        r=associate_current_measured_head(**{**o,'attempt':moved,'projection':replace(o['projection'],u_px=510.)})
        self.assertEqual(r.reason,'current_head_outside_registration_window')

    def test_existing_bearing_range_age_and_cluster_limits_remain_active(self):
        o=self.options()
        for changes,reason in (
            ({'map_bearing_rad':math.radians(20)},'camera_map_bearing_delta_exceeds_limit'),
            ({'now_sec':10.6},'stale_scan'),
            ({'accepted_range_m':(.8,1.)},'no_samples_in_accepted_range'),
            ({'scan':replace(o['scan'],ranges=(.55,.55,float('inf'),.55,.55,.55,.55,.55,.55))},'ambiguous_registered_camera_clusters'),
        ):
            with self.subTest(reason=reason):
                r=associate_current_measured_head(**{**o,**changes})
                self.assertFalse(r.accepted)
                self.assertEqual(r.reason,reason)

    def qr_binding(self, o, *, center=(80.,80.)):
        observations=(DecodedQrObservation('QR_003',tuple((center[0]+x,center[1]+y)
            for x,y in ((-20,-20),(20,-20),(20,20),(-20,20))),'synthetic'),)
        h=associate_current_measured_head(**o)
        fields={key:o[key] for key in ('intrinsics','scan_from_camera','scan','map_bearing_rad',
            'cone_half_angle_rad','accepted_range_m','now_sec','max_scan_age_sec','min_cluster_sample_count',
            'max_camera_map_bearing_delta_rad')}
        b=bind_qr_observations_to_target(observations,roi=o['attempt'].roi,
            camera_registration_accepted=h.accepted,**fields)
        return h,b,observations

    def test_own_symbol_and_overlapping_cluster_subsets_are_required(self):
        o=self.options();h,b,obs=self.qr_binding(o)
        self.assertTrue(b.accepted)
        accepted=bind_qr_to_current_head(b,obs,head_corners=o['estimate'].corners,head_association=h)
        self.assertTrue(accepted.accepted)
        nested={**b.association['search_association'],'selected_cluster_source_indices':[2,3,4]}
        subset=replace(b,association={**b.association,'search_association':nested})
        self.assertTrue(bind_qr_to_current_head(subset,obs,head_corners=o['estimate'].corners,head_association=h).accepted)
        for changed in (
            replace(subset,association={**subset.association,'search_association':{**nested,'selected_cluster_source_indices':[99]}}),
            replace(subset,association={**subset.association,'search_association':{**nested,'scan_stamp_sec':9.9}}),
        ):
            rejected=bind_qr_to_current_head(changed,obs,head_corners=o['estimate'].corners,head_association=h)
            self.assertFalse(rejected.accepted)
            self.assertEqual(rejected.qr_texts_for_evidence,())
        # A separately bound neighbouring symbol cannot become this head's identity.
        ranges=(.55,)*7+(float('inf'),)*6+(.55,)*7
        adjacent={**o,'scan':replace(o['scan'],ranges=ranges)}
        h,b,obs=self.qr_binding(adjacent,center=(20.,80.))
        self.assertTrue(b.accepted,b.reason)
        rejected=bind_qr_to_current_head(b,obs,head_corners=o['estimate'].corners,head_association=h)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason,'qr_outside_current_head')


if __name__=='__main__': unittest.main()
