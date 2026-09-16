"""Real bounded acquisition, LiDAR association, recropping and same-frame 3D fit."""

import math
import time
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.current_image_head_fit import CurrentImageHeadFit
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import estimate_planar_pose_ippe
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.camera_target_registration import HeadRoiEvaluation
from scripts.aufgabe04.real_robot.observer.head_proposal_registration import acquire_registered_head_measurement
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt, TARGET_CENTERED_REACQUISITION_SOURCE
from tests.aufgabe04 import test_head_boundary_independence as fixtures


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class RegisteredHeadRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HeadBoundaryIndependenceTest.setUpClass()
        cls.fixture = fixtures.HeadBoundaryIndependenceTest()
        cls.profile = cls.fixture.profile

    def acquire(self, angle, *, scan_age=.1, deadline=None):
        head, camera = self.fixture.projection(angle_deg=angle)
        frame = np.zeros((600, 800, 3), np.uint8)
        cv2.fillConvexPoly(frame, np.rint([(p.u_px, p.v_px) for p in head]).astype(np.int32),
                          (200, 200, 200))
        height = (math.hypot(head[0].u_px-head[3].u_px, head[0].v_px-head[3].v_px)
                  + math.hypot(head[1].u_px-head[2].u_px, head[1].v_px-head[2].v_px))/2
        search = HeadRoiAttempt(ImageRoi(180, 100, 650, 500, height),
            TARGET_CENTERED_REACQUISITION_SOURCE, 4.5, 400., 300., height, 2.25)
        intrinsics = CameraIntrinsics(800, 600, camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px)
        scan = PlainLaserScan((float("inf"), .35, .351, .35, float("inf")),
            -.02, .01, .1, 3.5, "base_scan", scan_stamp_sec=10., receipt_sec=10.01)
        diagnostics, evaluations = {}, []

        def evaluate(attempt, corners, *, current_head_refinement=None):
            self.assertTrue(diagnostics["candidate_associated"])
            self.assertIsNotNone(current_head_refinement)
            roi = attempt.roi
            cropped = frame[roi.y0:roi.y1, roi.x0:roi.x1]
            holder = CurrentImageHeadFit()
            options = dict(model_profile=self.profile,
                camera_fx_px=camera.fx_px, camera_fy_px=camera.fy_px,
                camera_cx_px=camera.cx_px-roi.x0, camera_cy_px=camera.cy_px-roi.y0,
                current_head_proposal_corners=corners, current_head_proposal_verified=True,
                current_head_refinement=current_head_refinement, current_image_head_fit=holder,
                expected_head_center_u_px=attempt.expected_center_u_px-roi.x0,
                expected_head_center_v_px=attempt.expected_center_v_px-roi.y0,
                expected_head_height_px=attempt.expected_head_height_px,
                qr_marker_policy="disabled")
            first = estimate_stand_axis_from_metric_model(cv2, cropped, **options)
            decorated = estimate_stand_axis_from_metric_model(cv2, cropped, qr_observations=(), **options)
            self.assertTrue(holder.reused)
            self.assertEqual(first[0].corners, decorated[0].corners)
            self.assertEqual(first[0].yaw_deg, decorated[0].yaw_deg)
            evaluations.append((attempt, corners, first[0], first[1]))
            return HeadRoiEvaluation(attempt, cropped, *decorated)

        selection = acquire_registered_head_measurement(cv2, frame, search,
            intrinsics=intrinsics, scan_from_camera=RigidTransform("base_scan", "camera",
                (0., 0., 0.), (.5, -.5, .5, -.5)), scan=scan, map_bearing_rad=0.,
            cone_half_angle_rad=math.radians(3), accepted_range_m=(.25, .45),
            now_sec=10.+scan_age, max_scan_age_sec=.5, min_cluster_sample_count=2,
            max_camera_map_bearing_delta_rad=math.radians(12), max_center_offset_ratio=1.5,
            edge_preprocess="channel_union", canny_low=20, canny_high=60,
            evaluate=evaluate, diagnostics=diagnostics, model_profile=self.profile,
            deadline_monotonic_sec=time.monotonic()+2. if deadline is None else deadline)
        return selection, diagnostics, evaluations

    def test_cold_registered_crop_preserves_selected_boundary_and_solves_once(self):
        for angle in (20., 45.):
            with self.subTest(angle=angle), patch(
                "scripts.aufgabe04.perception.stand_axis.head_model_fit.refine_current_physical_head",
                side_effect=AssertionError("registered current boundaries cannot be searched again")), patch(
                "scripts.aufgabe04.perception.stand_axis.head_model_fit.estimate_planar_pose_ippe",
                wraps=estimate_planar_pose_ippe) as solver:
                selection, diagnostics, evaluations = self.acquire(angle)
                self.assertIsNotNone(selection, diagnostics)
                self.assertEqual(len(evaluations), 1)
                attempt, corners, estimate, debug = evaluations[0]
                self.assertTrue(estimate.usable, (estimate.reason, diagnostics))
                self.assertEqual(estimate.corners, corners)
                self.assertAlmostEqual(estimate.yaw_deg, -angle, delta=3.)
                self.assertGreater(attempt.roi.x0, 180)
                self.assertGreater(attempt.roi.y0, 100)
                self.assertTrue(debug.head_acquisition_diagnostics["current_boundary_reused"])
                self.assertTrue(debug.head_acquisition_diagnostics["selected_border_binding"]["accepted"])
                self.assertTrue(diagnostics["lidar_association"]["associated"])
                solver.assert_called_once()

    def test_stale_scan_or_expired_budget_cannot_use_the_registered_proof(self):
        for options in ({"scan_age": .6}, {"deadline": time.monotonic()-1.}):
            with self.subTest(options=options):
                selection, diagnostics, evaluations = self.acquire(45., **options)
                self.assertTrue(selection is None, diagnostics.get("reason"))
                self.assertEqual(evaluations, [])


if __name__ == "__main__":
    unittest.main()
