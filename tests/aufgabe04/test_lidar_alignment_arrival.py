import math
from dataclasses import replace
import unittest

from scripts.aufgabe04.navigation.approach.lidar_alignment_arrival import verify_lidar_alignment_arrival
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import (
    derive_lidar_inspection_hints, fit_current_lidar_view,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.configuration.profile import CameraCalibrationProfile, RigidTransform
from tests.aufgabe04.test_lidar_inspection_hint import hint_fixture, line_receipts


def calibration(*, lateral=0., yaw=0.):
    # REP-103 optical axes, composed with the measured planar mount yaw.
    q = (.5, -.5, .5, -.5)
    s, c = math.sin(yaw/2), math.cos(yaw/2)
    rotation = (c*q[0]-s*q[1], c*q[1]+s*q[0], c*q[2]+s*q[3], c*q[3]-s*q[2])
    return CameraCalibrationProfile(
        1, "test_camera", "camera_optical", "base_link", 800, 600, "plumb_bob",
        (0.,)*5, (500.,0.,400.,0.,500.,300.,0.,0.,1.),
        (1.,0.,0.,0.,1.,0.,0.,0.,1.), (500.,0.,400.,0.,0.,500.,300.,0.,0.,0.,1.,0.),
        RigidTransform((.045, lateral, .125), rotation), 1., "measured_test_calibration")


def fixture(*, bearing=math.pi, pose_yaw=None, receipts=None):
    snapshot, registry, frame = hint_fixture()
    history = line_receipts(increment=.01) + line_receipts(2, increment=.01)
    hint = derive_lidar_inspection_hints(snapshot=snapshot, registry=registry,
        planning_frame=frame, receipts=history)[0]["candidate_1"]
    if receipts is None:
        receipts = tuple(replace(r, receipt_id=f"arrival_{i}", viewpoint_id="arrival",
                                 scan_stamp_sec=30.+i*.1, pose_stamp_sec=30.+i*.1,
                                 observer_clock_sec=30.+i*.1+.01)
                         for i,r in enumerate(line_receipts(bearing=bearing, increment=.01)))
    scan_pose = receipts[-1].frame_provenance.canonical_scan_pose_odom
    base = replace(scan_pose, yaw_rad=scan_pose.yaw_rad if pose_yaw is None else pose_yaw)
    frame = replace(frame, current_pose=base)
    current = fit_current_lidar_view(snapshot=snapshot, registry=registry,
        planning_frame=frame, candidate_uid="candidate_1", receipts=receipts)
    return dict(hint=hint, current_fit=current, current_receipts=receipts,
                snapshot=snapshot, candidate_uid="candidate_1", planning_frame=frame,
                calibration=calibration(), now_sec=30.35, not_before_sec=29.9,
                current_base_pose_odom=base, pose_stamp_sec=30.3,
                localization_position_bound_m=.001, localization_yaw_bound_rad=math.radians(.1))


class LidarAlignmentArrivalTest(unittest.TestCase):
    def verify(self, changes=None, **fixture_args):
        values = fixture(**fixture_args)
        values.update(changes or {})
        return verify_lidar_alignment_arrival(**values)

    def test_fresh_stopped_geometry_verifies_without_face_or_motion_authority(self):
        result = self.verify()
        self.assertTrue(result["accepted"], result)
        self.assertTrue(result["head_alignment_verified"])
        self.assertTrue(result["camera_centered_verified"])
        self.assertFalse(result["front_back_identity_resolved"])
        self.assertFalse(result["stand_axis_authorized"])
        self.assertFalse(result["motion_authorized"])
        self.assertFalse(result["angle_accuracy_calibrated"])

    def test_opposite_surface_normal_is_accepted_modulo_pi(self):
        result = self.verify(bearing=0.)
        self.assertTrue(result["accepted"], result)
        self.assertFalse(result["front_back_identity_resolved"])

    def test_normal_agreement_does_not_admit_camera_pointing_away(self):
        result = self.verify(pose_yaw=math.pi)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "arrival_target_behind_camera")

    def test_actual_pose_uses_calibrated_camera_origin_and_optical_yaw(self):
        result = self.verify({"calibration": calibration(lateral=.06, yaw=.08)})
        self.assertTrue(result["accepted"], result)
        self.assertAlmostEqual(result["camera_y_m"], .06, places=8)
        self.assertAlmostEqual(result["camera_optical_yaw_rad"], .08, places=8)
        self.assertGreater(result["camera_center_error_rad"], .15)
        self.assertFalse(result["camera_centered_verified"])

    def test_actual_yaw_residual_is_counted_once(self):
        result = self.verify(pose_yaw=math.radians(5))
        self.assertTrue(result["accepted"], result)
        expected = (max(result["position_side_error_rad"], result["optical_normal_error_rad"])
                    + result["fit_angle_uncertainty_rad"] + result["position_uncertainty_angle_rad"]
                    + result["localization_yaw_bound_rad"])
        self.assertAlmostEqual(result["total_alignment_bound_rad"], expected)
        self.assertFalse(result["planning_execution_tolerances_added"])

    def test_large_localization_budget_leaves_arrival_unverified(self):
        result = self.verify({"localization_position_bound_m": .12,
                              "localization_yaw_bound_rad": math.radians(5)})
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "arrival_alignment_budget_exceeded")

    def test_missing_fit_and_legacy_tangent_only_hint_do_not_verify(self):
        for field in ("hint", "current_fit"):
            with self.subTest(field=field):
                result = self.verify({field: None})
                self.assertFalse(result["head_alignment_verified"])
                self.assertEqual(result["fallback"], "ordinary_camera_acquisition_unverified")
        values = fixture()
        values["hint"] = replace(values["hint"], center_x_m=None)
        self.assertEqual(verify_lidar_alignment_arrival(**values)["reason"], "fitted_head_geometry_unavailable")

    def test_changed_raw_receipt_cannot_borrow_fitted_support(self):
        values = fixture()
        scan = values["current_receipts"][0]
        changed = replace(scan, ranges_m=(None,)*len(scan.ranges_m))
        values["current_receipts"] = (changed, *values["current_receipts"][1:])
        result = verify_lidar_alignment_arrival(**values)
        self.assertEqual(result["reason"], "fresh_fit_scan_binding_invalid")

    def test_duplicate_scans_and_old_stop_are_rejected(self):
        values = fixture()
        receipts = values["current_receipts"]
        for changes, reason in (({"current_receipts": (receipts[0],)*3}, "current_scan_order_or_identity_invalid"),
                                ({"not_before_sec": 30.05}, "arrival_scan_sources_not_fresh"),
                                ({"now_sec": 31.}, "arrival_scan_sources_not_fresh")):
            with self.subTest(changes=changes):
                self.assertEqual(self.verify(changes)["reason"], reason)

    def test_wrong_base_pose_and_stale_base_stamp_are_rejected(self):
        self.assertEqual(self.verify({"current_base_pose_odom": Pose2D(-.5, 0., 0.)})["reason"],
                         "arrival_base_pose_planning_frame_mismatch")
        self.assertEqual(self.verify({"pose_stamp_sec": 29.9})["reason"], "arrival_base_pose_not_current")

    def test_current_view_disagreement_blocks_historical_hint(self):
        for field, change, reason in (("center_x_m", .05, "fresh_head_center_conflicts_with_hint"),
                                     ("tangent_rad", .6, "fresh_head_normal_conflicts_with_hint")):
            values = fixture()
            hint = values["hint"]
            values["hint"] = replace(hint, **{field: getattr(hint, field)+change})
            with self.subTest(field=field):
                self.assertEqual(verify_lidar_alignment_arrival(**values)["reason"], reason)

    def test_missing_uncertainty_cannot_be_zero_by_default(self):
        for field in ("localization_position_bound_m", "localization_yaw_bound_rad"):
            result = self.verify({field: None})
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], f"invalid_{field}")

    def test_three_good_scans_plus_one_miss_still_allow_current_fit(self):
        values = fixture()
        receipts = list(values["current_receipts"])
        receipts[0] = replace(receipts[0], ranges_m=(None,)*len(receipts[0].ranges_m))
        result = self.verify(receipts=tuple(receipts))
        self.assertTrue(result["accepted"], result)
        self.assertEqual(len(result["accepted_fit_receipt_sha256s"]), 3)

    def test_current_view_cannot_replace_independent_survey_hint(self):
        values = fixture()
        values["hint"] = values["current_fit"]
        self.assertEqual(verify_lidar_alignment_arrival(**values)["reason"], "independent_survey_hint_required")

    def test_fresh_empty_scan_does_not_refresh_old_fitted_support(self):
        values = fixture()
        receipts = list(values["current_receipts"])
        receipts[-1] = replace(receipts[-1], scan_stamp_sec=31.2, pose_stamp_sec=31.2,
                               observer_clock_sec=31.21, ranges_m=(None,)*len(receipts[-1].ranges_m))
        values = fixture(receipts=tuple(receipts))
        values.update(now_sec=31.25, pose_stamp_sec=31.2)
        self.assertEqual(verify_lidar_alignment_arrival(**values)["reason"], "arrival_fitted_support_not_fresh")

    def test_invalid_calibration_is_unverified(self):
        result = self.verify({"calibration": None})
        self.assertFalse(result["accepted"])
        self.assertFalse(result["head_alignment_verified"])

    def test_moving_scan_window_is_not_arrival_evidence(self):
        values = fixture()
        receipts = list(values["current_receipts"])
        first = receipts[0]
        moved = replace(first.scan_pose_map, x_m=first.scan_pose_map.x_m+.04)
        receipts[0] = replace(first, scan_pose_map=moved,
            frame_provenance=replace(first.frame_provenance, canonical_scan_pose_odom=moved))
        self.assertIsNone(fixture(receipts=tuple(receipts))["current_fit"])
        self.assertFalse(self.verify(receipts=tuple(receipts))["accepted"])


if __name__ == "__main__":
    unittest.main()
