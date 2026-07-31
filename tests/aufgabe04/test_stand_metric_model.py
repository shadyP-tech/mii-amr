from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception.stand_axis.model_profile import (
    ModelPoint3D,
    StandModelProfile,
    load_stand_model,
    stand_model_from_payload,
    write_stand_model,
)
from scripts.aufgabe04.perception.stand_axis.model_projection import (
    project_stand_model,
)
from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    refine_projected_head_border,
)
from scripts.aufgabe04.perception.stand_axis.pose_tracking import (
    MetricPoseTracker,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    PlanarPoseHypothesis,
    PlanarPoseResult,
    RectifiedCameraMatrix,
    estimate_planar_pose_ippe,
)
from scripts.aufgabe04.perception.stand_axis_image import (
    estimate_stand_axis_from_metric_model,
)


def profile_payload(*, status: str = "measured") -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "real_stand_test_v1",
        "environment": "physical",
        "measurement_status": status,
        "head_width_m": 0.078,
        "head_height_m": 0.078,
        "head_depth_m": 0.007,
        "qr_symbol_width_m": 0.060,
        "qr_symbol_height_m": 0.060,
        "qr_center_x_m": 0.0,
        "qr_center_y_m": 0.0,
        "stem_width_m": 0.010,
        "stem_visible_height_m": 0.080,
        "tolerance_m": 0.001,
        "source": "test measurements",
    }


def frontal_pose() -> PlanarPoseHypothesis:
    return PlanarPoseHypothesis(
        rotation_vector=(0.0, 0.0, 0.0),
        translation_xyz_m=(0.0, 0.0, 0.40),
        face_normal_xyz=(0.0, 0.0, 1.0),
        yaw_deg=0.0,
        reprojection_rmse_px=0.0,
        positive_depth=True,
    )


def oblique_pose() -> PlanarPoseHypothesis:
    angle = math.radians(25.0)
    return PlanarPoseHypothesis(
        rotation_vector=(0.0, angle, 0.0),
        translation_xyz_m=(0.0, 0.0, 0.40),
        face_normal_xyz=(math.sin(angle), 0.0, math.cos(angle)),
        yaw_deg=-25.0,
        reprojection_rmse_px=0.0,
        positive_depth=True,
    )


class StandModelProfileTest(unittest.TestCase):
    def test_hashed_profile_round_trip_and_landmark_frame(self):
        profile = stand_model_from_payload(profile_payload())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stand.json"
            digest = write_stand_model(path, profile)
            loaded = load_stand_model(path)

        self.assertEqual(loaded.sha256, digest)
        self.assertTrue(loaded.committable)
        self.assertEqual(loaded.head_corners[0], ModelPoint3D(-0.039, -0.039, 0.0))
        self.assertIn("stem_junction_left", loaded.semantic_landmarks)

    def test_profile_rejects_qr_outside_head(self):
        payload = profile_payload()
        payload["qr_center_x_m"] = 0.020
        with self.assertRaisesRegex(ValueError, "exceeds"):
            stand_model_from_payload(payload)


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV and numpy are required")
class StandMetricGeometryTest(unittest.TestCase):
    def setUp(self):
        self.profile = stand_model_from_payload(profile_payload())
        self.camera = RectifiedCameraMatrix(400.0, 400.0, 320.0, 240.0)

    def _qr_pixels(self):
        projected = project_stand_model(cv2, self.profile, frontal_pose(), self.camera)
        return tuple(
            projected.landmarks[name]
            for name in (
                "qr_top_left",
                "qr_top_right",
                "qr_bottom_right",
                "qr_bottom_left",
            )
        )

    def test_ippe_preserves_planar_hypotheses_and_residual(self):
        result = estimate_planar_pose_ippe(
            cv2,
            self._qr_pixels(),
            self.profile.qr_corners,
            self.camera,
        )

        self.assertTrue(result.accepted)
        self.assertIsNotNone(result.best)
        self.assertLess(result.best.reprojection_rmse_px, 1.0e-4)
        self.assertGreater(result.best.translation_xyz_m[2], 0.0)

    def test_projection_corridor_refines_only_real_current_frame_edges(self):
        projected = project_stand_model(cv2, self.profile, frontal_pose(), self.camera)
        edges = numpy.zeros((480, 640), dtype=numpy.uint8)
        polygon = numpy.asarray(
            [[(round(point.u_px), round(point.v_px)) for point in projected.head_corners]],
            dtype=numpy.int32,
        )
        cv2.polylines(edges, polygon, True, 255, 2)

        refined = refine_projected_head_border(cv2, edges, projected.head_corners)
        blank = refine_projected_head_border(
            cv2,
            numpy.zeros_like(edges),
            projected.head_corners,
        )

        self.assertTrue(refined.accepted)
        self.assertIsNotNone(refined.corners)
        self.assertFalse(blank.accepted)
        self.assertIsNone(blank.corners)

    def test_high_level_pipeline_separates_prediction_from_measurement(self):
        projected = project_stand_model(cv2, self.profile, frontal_pose(), self.camera)
        measured_frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        polygon = numpy.asarray(
            [[(round(point.u_px), round(point.v_px)) for point in projected.head_corners]],
            dtype=numpy.int32,
        )
        cv2.polylines(measured_frame, polygon, True, (255, 255, 255), 2)
        blank_frame = numpy.zeros_like(measured_frame)
        options = dict(
            model_profile=self.profile,
            camera_fx_px=self.camera.fx_px,
            camera_fy_px=self.camera.fy_px,
            camera_cx_px=self.camera.cx_px,
            camera_cy_px=self.camera.cy_px,
            blur_kernel=1,
            canny_low=20,
            canny_high=60,
        )
        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_pipeline.detect_qr_quad_corners",
            return_value=self._qr_pixels(),
        ):
            measured, measured_debug = estimate_stand_axis_from_metric_model(
                cv2, measured_frame, **options
            )
            predicted, predicted_debug = estimate_stand_axis_from_metric_model(
                cv2, blank_frame, **options
            )

        self.assertTrue(measured.usable)
        self.assertEqual(measured.evidence_state, "fresh_refined")
        self.assertEqual(measured.source, "model_current_frame_refined")
        self.assertFalse(predicted.usable)
        self.assertEqual(predicted.evidence_state, "predicted_only")
        self.assertIsNotNone(predicted_debug.predicted_corners)
        self.assertEqual(measured_debug.model_profile_sha256, self.profile.sha256)

    def test_model_pipeline_recovers_oblique_current_frame_axis(self):
        pose = oblique_pose()
        projected = project_stand_model(cv2, self.profile, pose, self.camera)
        frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        polygon = numpy.asarray(
            [[(round(point.u_px), round(point.v_px)) for point in projected.head_corners]],
            dtype=numpy.int32,
        )
        cv2.polylines(frame, polygon, True, (255, 255, 255), 2)
        qr_pixels = tuple(
            projected.landmarks[name]
            for name in (
                "qr_top_left",
                "qr_top_right",
                "qr_bottom_right",
                "qr_bottom_left",
            )
        )
        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_pipeline.detect_qr_quad_corners",
            return_value=qr_pixels,
        ):
            estimate, _debug = estimate_stand_axis_from_metric_model(
                cv2,
                frame,
                model_profile=self.profile,
                camera_fx_px=self.camera.fx_px,
                camera_fy_px=self.camera.fy_px,
                camera_cx_px=self.camera.cx_px,
                camera_cy_px=self.camera.cy_px,
                blur_kernel=1,
            )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertAlmostEqual(estimate.yaw_deg, -25.0, delta=3.0)


class MetricPoseTrackerTest(unittest.TestCase):
    def test_similarly_fitting_different_axes_are_ambiguous(self):
        first = frontal_pose()
        second = PlanarPoseHypothesis(
            rotation_vector=(0.0, 0.0, 0.0),
            translation_xyz_m=(0.0, 0.0, 0.4),
            face_normal_xyz=(0.2, 0.0, 0.98),
            yaw_deg=12.0,
            reprojection_rmse_px=0.05,
            positive_depth=True,
        )
        result = PlanarPoseResult(
            accepted=True,
            reason="pose_estimated",
            hypotheses=(first, second),
            ambiguity_gap_px=0.05,
        )

        self.assertTrue(result.axis_ambiguous())

    def test_prediction_is_bounded_and_context_bound(self):
        tracker = MetricPoseTracker(prediction_ttl_sec=0.25)
        camera = (400.0, 400.0, 320.0, 240.0)
        tracker.accept(
            frontal_pose(),
            now_sec=10.0,
            profile_sha256="a" * 64,
            camera_signature=camera,
        )

        fresh = tracker.prediction(
            now_sec=10.2,
            profile_sha256="a" * 64,
            camera_signature=camera,
        )
        stale = tracker.prediction(
            now_sec=10.3,
            profile_sha256="a" * 64,
            camera_signature=camera,
        )

        self.assertEqual(fresh.state, "predicted_only")
        self.assertIsNotNone(fresh.pose)
        self.assertEqual(stale.state, "stale")
        self.assertIsNone(stale.pose)


if __name__ == "__main__":
    unittest.main()
