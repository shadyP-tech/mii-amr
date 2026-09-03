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
    STAND_MODEL_SCHEMA_VERSION,
    StandModelProfile,
    load_measured_physical_stand_model,
    load_stand_model,
    resolve_head_center_height_m,
    stand_model_from_payload,
    write_stand_model,
)
from scripts.aufgabe04.perception.stand_axis.model_projection import (
    project_stand_model,
)
from scripts.aufgabe04.perception.stand_axis.model_pipeline import (
    estimate_stand_axis_from_metric_model,
)
from scripts.aufgabe04.perception.stand_axis.model_backside_acquisition import (
    MODEL_BACKSIDE_AXIS_SOURCE,
)
from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    model_corridor_half_width_px,
    refine_projected_head_border,
)
from scripts.aufgabe04.perception.stand_axis.pose_tracking import (
    MetricPoseTracker,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    PlanarPoseHypothesis,
    PlanarPoseResult,
    QrQuadDetection,
    RectifiedCameraMatrix,
    detect_qr_quad,
    estimate_planar_pose_ippe,
    select_temporally_consistent_pose,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
)
from scripts.aufgabe04.perception.debug.stand_axis_viewer import (
    annotate_projected_model_landmarks,
)


def profile_payload(
    *,
    status: str = "measured",
    schema_version: int = STAND_MODEL_SCHEMA_VERSION,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "profile_id": f"real_stand_test_v{schema_version}",
        "environment": "physical",
        "measurement_status": status,
        "head_width_m": 0.078,
        "head_height_m": 0.078,
        "head_depth_m": 0.006,
        "qr_symbol_width_m": 0.062,
        "qr_symbol_height_m": 0.062,
        "qr_panel_width_m": 0.071,
        "qr_panel_height_m": 0.071,
        "qr_center_x_m": 0.0,
        "qr_center_y_m": 0.0,
        "head_top_height_m": 0.210,
        "base_width_m": 0.153,
        "base_depth_m": 0.153,
        "tolerance_m": 0.002,
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
        self.assertEqual(
            loaded.head_back_corners[0],
            ModelPoint3D(-0.039, -0.039, 0.006),
        )
        self.assertAlmostEqual(loaded.head_center_height_m, 0.171)
        self.assertAlmostEqual(
            loaded.base_circumscribed_radius_m,
            math.hypot(0.153, 0.153) / 2.0,
        )
        self.assertAlmostEqual(
            loaded.navigation_footprint_radius_m,
            math.hypot(0.153, 0.153) / 2.0 + 0.002,
        )
        self.assertEqual(loaded.qr_panel_width_m, 0.071)
        self.assertEqual(loaded.qr_panel_height_m, 0.071)
        self.assertNotIn("stem_junction_left", loaded.semantic_landmarks)
        self.assertIn("head_back_top_left", loaded.semantic_landmarks)

    def test_checked_in_legacy_v1_provisional_profile_remains_readable(self):
        repository_root = Path(__file__).resolve().parents[2]
        profile = load_stand_model(
            repository_root
            / "configs/aufgabe04/stand_models/physical_stand_assumptions_v1.json"
        )

        self.assertEqual(profile.schema_version, 1)
        self.assertEqual(profile.measurement_status, "provisional")
        self.assertIsNone(profile.head_center_height_m)
        self.assertIsNone(profile.base_circumscribed_radius_m)
        self.assertIsNone(profile.qr_panel_width_m)
        self.assertIn("stem_junction_left", profile.semantic_landmarks)

    def test_checked_in_measured_v2_profile_is_operational(self):
        repository_root = Path(__file__).resolve().parents[2]
        profile = load_measured_physical_stand_model(
            repository_root
            / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
        )

        self.assertEqual(profile.schema_version, STAND_MODEL_SCHEMA_VERSION)
        self.assertEqual(
            profile.sha256,
            "56fe19dcbfc8aa58682ea460e702a499c65cc719423940e6a892ca581e6d0b5f",
        )
        self.assertEqual(profile.qr_symbol_width_m, 0.062)
        self.assertEqual(profile.qr_panel_width_m, 0.071)
        self.assertAlmostEqual(profile.head_center_height_m, 0.171)

    def test_v2_requires_complete_geometry(self):
        payload = profile_payload()
        del payload["qr_panel_width_m"]

        with self.assertRaisesRegex(ValueError, "complete geometry"):
            stand_model_from_payload(payload)

    def test_head_center_height_defaults_to_exact_model_derivation(self):
        profile = stand_model_from_payload(profile_payload())

        self.assertAlmostEqual(resolve_head_center_height_m(profile, None), 0.171)
        self.assertEqual(
            resolve_head_center_height_m(profile, 0.172),
            profile.head_center_height_m,
        )
        self.assertEqual(
            resolve_head_center_height_m(profile, 0.173),
            profile.head_center_height_m,
        )

    def test_head_center_height_rejects_invalid_or_mismatched_override(self):
        profile = stand_model_from_payload(profile_payload())

        for requested in (True, float("nan"), float("inf"), 0.0, -0.1):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    resolve_head_center_height_m(profile, requested)
        with self.assertRaisesRegex(ValueError, "more than tolerance"):
            resolve_head_center_height_m(profile, 0.174)

    def test_head_center_height_requires_complete_model_geometry(self):
        payload = profile_payload(status="provisional", schema_version=1)
        for field in (
            "head_top_height_m",
            "base_width_m",
            "base_depth_m",
            "qr_panel_width_m",
            "qr_panel_height_m",
        ):
            del payload[field]
        profile = stand_model_from_payload(payload)

        with self.assertRaisesRegex(ValueError, "head_top_height_m"):
            resolve_head_center_height_m(profile, None)

    def test_profile_rejects_qr_outside_head(self):
        payload = profile_payload()
        payload["qr_center_x_m"] = 0.020
        with self.assertRaisesRegex(ValueError, "exceeds"):
            stand_model_from_payload(payload)

    def test_operational_loader_requires_measured_physical_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            measured_path = root / "measured.json"
            write_stand_model(
                measured_path,
                stand_model_from_payload(profile_payload()),
            )
            self.assertEqual(
                load_measured_physical_stand_model(measured_path).environment,
                "physical",
            )

            provisional_path = root / "provisional.json"
            write_stand_model(
                provisional_path,
                stand_model_from_payload(profile_payload(status="provisional")),
            )
            with self.assertRaisesRegex(ValueError, "measurement_status=measured"):
                load_measured_physical_stand_model(provisional_path)

            simulation_payload = profile_payload()
            simulation_payload["environment"] = "simulation"
            simulation_path = root / "simulation.json"
            write_stand_model(
                simulation_path,
                stand_model_from_payload(simulation_payload),
            )
            with self.assertRaisesRegex(ValueError, "environment=physical"):
                load_measured_physical_stand_model(simulation_path)

            legacy_payload = profile_payload(schema_version=1)
            for field in (
                "head_top_height_m",
                "base_width_m",
                "base_depth_m",
                "qr_panel_width_m",
                "qr_panel_height_m",
            ):
                del legacy_payload[field]
            legacy_path = root / "legacy.json"
            write_stand_model(
                legacy_path,
                stand_model_from_payload(legacy_payload),
            )
            with self.assertRaisesRegex(ValueError, "schema_version=2"):
                load_measured_physical_stand_model(legacy_path)


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

    def test_qr_acquisition_restores_four_x_corners_to_input_pixels(self):
        scaled_corners = (
            ImagePoint(40.0, 80.0),
            ImagePoint(120.0, 80.0),
            ImagePoint(120.0, 160.0),
            ImagePoint(40.0, 160.0),
        )
        frame = numpy.zeros((120, 160, 3), dtype=numpy.uint8)
        with patch(
            "scripts.aufgabe04.perception.stand_axis.qr_pose_seed."
            "_detect_qr_quad_corners_native",
            side_effect=(None, None, scaled_corners),
        ):
            detection = detect_qr_quad(cv2, frame)

        self.assertIsNotNone(detection)
        self.assertEqual(detection.scale, 4.0)
        self.assertEqual(detection.corners[0], ImagePoint(10.0, 20.0))
        self.assertEqual(detection.corners[2], ImagePoint(30.0, 40.0))

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

    def test_projection_corridor_prefers_predicted_rails_over_parallel_clutter(self):
        projected = project_stand_model(cv2, self.profile, oblique_pose(), self.camera)
        edges = numpy.zeros((480, 640), dtype=numpy.uint8)
        physical = numpy.asarray(
            [[(round(point.u_px), round(point.v_px)) for point in projected.head_corners]],
            dtype=numpy.int32,
        )
        distractor = physical + numpy.asarray([[[4, 0]]], dtype=numpy.int32)
        cv2.polylines(edges, distractor, True, 255, 1)
        cv2.polylines(edges, physical, True, 255, 2)

        refined = refine_projected_head_border(
            cv2,
            edges,
            projected.head_corners,
            corridor_half_width_px=4.5,
        )

        self.assertTrue(refined.accepted, refined.reason)
        self.assertLess(
            max(
                math.hypot(
                    measured.u_px - predicted.u_px,
                    measured.v_px - predicted.v_px,
                )
                for measured, predicted in zip(
                    refined.corners,
                    projected.head_corners,
                )
            ),
            3.0,
        )

    def test_provisional_model_corridor_recovers_recorded_outer_rail_bias(self):
        profile = stand_model_from_payload(profile_payload(status="provisional"))
        camera = RectifiedCameraMatrix(640.0, 640.0, 400.0, 300.0)
        projected = project_stand_model(cv2, profile, frontal_pose(), camera)
        corridor = model_corridor_half_width_px(
            projected.head_corners,
            model_profile=profile,
            pose_reprojection_rmse_px=0.28,
        )
        edges = numpy.zeros((600, 800), dtype=numpy.uint8)
        predicted = projected.head_corners
        # Reproduce the latest physical capture: the QR-seeded model is stable,
        # but the right and lower outer rails are roughly 4-6 px inside it.
        physical = numpy.asarray(
            [[
                (round(predicted[0].u_px + 1), round(predicted[0].v_px + 4)),
                (round(predicted[1].u_px - 5), round(predicted[1].v_px + 4)),
                (round(predicted[2].u_px - 5), round(predicted[2].v_px - 5)),
                (round(predicted[3].u_px + 1), round(predicted[3].v_px - 5)),
            ]],
            dtype=numpy.int32,
        )
        # Long heater rails remain present but do not form a model-consistent
        # four-sided head near the projection.
        left = round(predicted[0].u_px) - 55
        for x_px in range(left, left + 48, 12):
            cv2.line(
                edges,
                (x_px, round(predicted[0].v_px) - 25),
                (x_px, round(predicted[3].v_px) + 25),
                255,
                1,
            )
        cv2.polylines(edges, physical, True, 255, 2)

        refined = refine_projected_head_border(
            cv2,
            edges,
            projected.head_corners,
            corridor_half_width_px=corridor,
        )

        self.assertGreaterEqual(corridor, 6.0)
        self.assertLessEqual(corridor, 8.0)
        self.assertTrue(refined.accepted, refined.reason)
        self.assertIsNotNone(refined.corners)

    def test_model_corridor_rejects_heater_rails_without_four_head_sides(self):
        profile = stand_model_from_payload(profile_payload(status="provisional"))
        camera = RectifiedCameraMatrix(640.0, 640.0, 400.0, 300.0)
        projected = project_stand_model(cv2, profile, frontal_pose(), camera)
        corridor = model_corridor_half_width_px(
            projected.head_corners,
            model_profile=profile,
            pose_reprojection_rmse_px=0.28,
        )
        edges = numpy.zeros((600, 800), dtype=numpy.uint8)
        top = round(projected.head_corners[0].v_px) - 25
        bottom = round(projected.head_corners[3].v_px) + 25
        for x_px in range(300, 501, 12):
            cv2.line(edges, (x_px, top), (x_px, bottom), 255, 1)

        refined = refine_projected_head_border(
            cv2,
            edges,
            projected.head_corners,
            corridor_half_width_px=corridor,
        )

        self.assertFalse(refined.accepted)
        self.assertIsNone(refined.corners)

    def test_projected_depth_landmarks_render_as_diagnostics(self):
        projected = project_stand_model(
            cv2, self.profile, oblique_pose(), self.camera
        )
        frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)

        annotate_projected_model_landmarks(
            cv2,
            frame,
            projected.landmarks,
        )

        self.assertGreater(int(numpy.count_nonzero(frame)), 0)
        self.assertNotEqual(
            projected.head_corners[0],
            projected.head_back_corners[0],
        )

    def test_high_level_pipeline_separates_prediction_from_measurement(self):
        projected = project_stand_model(cv2, self.profile, oblique_pose(), self.camera)
        qr_pixels = tuple(
            projected.landmarks[name]
            for name in (
                "qr_top_left",
                "qr_top_right",
                "qr_bottom_right",
                "qr_bottom_left",
            )
        )
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
            "scripts.aufgabe04.perception.stand_axis.model_pipeline.detect_qr_quad",
            return_value=QrQuadDetection(qr_pixels, 4.0),
        ):
            measured, measured_debug = estimate_stand_axis_from_metric_model(
                cv2, measured_frame, **options
            )
            predicted, predicted_debug = estimate_stand_axis_from_metric_model(
                cv2, blank_frame, **options
            )

        self.assertTrue(measured.usable, measured.reason)
        self.assertEqual(measured.evidence_state, "fresh_refined")
        self.assertEqual(measured.source, "model_current_frame_refined")
        self.assertFalse(predicted.usable)
        self.assertEqual(predicted.evidence_state, "predicted_only")
        self.assertIsNotNone(predicted_debug.predicted_corners)
        self.assertEqual(measured_debug.model_profile_sha256, self.profile.sha256)
        self.assertEqual(measured_debug.qr_detection_scale, 4.0)
        self.assertEqual(measured_debug.pose_seed_source, "qr_pyramid_4x")
        self.assertIsNotNone(measured_debug.model_corridor_half_width_px)
        self.assertEqual(measured_debug.model_pose_fit_source, "joint_qr_head")
        self.assertIn("head_back_top_left", measured_debug.projected_landmarks)
        self.assertNotIn("stem_bottom_left", measured_debug.projected_landmarks)

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
            "scripts.aufgabe04.perception.stand_axis.model_pipeline.detect_qr_quad",
            return_value=QrQuadDetection(qr_pixels, 2.0),
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

    @staticmethod
    def _synthetic_backside_frame(*, include_neck: bool = True):
        frame = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
        cv2.rectangle(frame, (120, 50), (200, 130), (255, 255, 255), 2)
        if include_neck:
            cv2.line(frame, (153, 131), (153, 205), (255, 255, 255), 2)
            cv2.line(frame, (167, 131), (167, 205), (255, 255, 255), 2)
        return frame

    def _backside_options(self, **overrides):
        options = {
            "model_profile": self.profile,
            "camera_fx_px": self.camera.fx_px,
            "camera_fy_px": self.camera.fy_px,
            "camera_cx_px": self.camera.cx_px,
            "camera_cy_px": self.camera.cy_px,
            "blur_kernel": 1,
            "canny_low": 20,
            "canny_high": 60,
            "expected_head_center_u_px": 160.0,
            "expected_head_center_v_px": 90.0,
            "expected_head_height_px": 80.0,
        }
        options.update(overrides)
        return options

    def test_no_qr_measured_model_bootstraps_backside_axis(self):
        frame = self._synthetic_backside_frame()
        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_pipeline."
            "detect_qr_quad",
            return_value=None,
        ):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2,
                frame,
                **self._backside_options(),
            )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.source, MODEL_BACKSIDE_AXIS_SOURCE)
        self.assertEqual(estimate.evidence_state, "fresh_backside")
        self.assertEqual(estimate.visible_face, "backside_candidate")
        self.assertGreaterEqual(estimate.visible_face_confidence, 0.70)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertTrue(math.isfinite(estimate.yaw_deg))
        self.assertIsNone(estimate.camera_face_normal_xyz)
        self.assertEqual(
            debug.visible_face_reason,
            "qr_absent_model_head_and_neck_supported",
        )
        self.assertAlmostEqual(debug.head_scale_ratio, 1.0, delta=0.08)
        self.assertLess(debug.head_center_error_ratio, 0.05)
        self.assertEqual(debug.model_profile_sha256, self.profile.sha256)
        self.assertIsNotNone(estimate.pose_reprojection_rmse_px)
        self.assertEqual(
            debug.pose_reprojection_rmse_px,
            estimate.pose_reprojection_rmse_px,
        )
        self.assertEqual(
            debug.pose_ambiguity_gap_px,
            estimate.pose_ambiguity_gap_px,
        )

    def test_backside_bootstrap_rejects_ambiguous_planar_axis(self):
        frame = self._synthetic_backside_frame()
        best = PlanarPoseHypothesis(
            rotation_vector=(0.0, 0.0, 0.0),
            translation_xyz_m=(0.0, 0.0, 0.40),
            face_normal_xyz=(0.0, 0.0, 1.0),
            yaw_deg=0.0,
            reprojection_rmse_px=0.04,
            positive_depth=True,
        )
        competing = PlanarPoseHypothesis(
            rotation_vector=(0.0, -0.35, 0.0),
            translation_xyz_m=(0.0, 0.0, 0.40),
            face_normal_xyz=(-0.34, 0.0, 0.94),
            yaw_deg=20.0,
            reprojection_rmse_px=0.05,
            positive_depth=True,
        )
        ambiguous_pose = PlanarPoseResult(
            accepted=True,
            reason="pose_estimated",
            hypotheses=(best, competing),
            ambiguity_gap_px=0.01,
        )
        with (
            patch(
                "scripts.aufgabe04.perception.stand_axis.model_pipeline."
                "detect_qr_quad",
                return_value=None,
            ),
            patch(
                "scripts.aufgabe04.perception.stand_axis."
                "model_backside_acquisition.estimate_planar_pose_ippe",
                return_value=ambiguous_pose,
            ),
        ):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2,
                frame,
                **self._backside_options(),
            )

        self.assertFalse(estimate.usable)
        self.assertEqual(
            estimate.reason,
            "model_backside_planar_pose_axis_ambiguous",
        )
        self.assertEqual(estimate.evidence_state, "unobservable")
        self.assertIsNone(estimate.visible_face)
        self.assertEqual(estimate.pose_reprojection_rmse_px, 0.04)
        self.assertEqual(estimate.pose_ambiguity_gap_px, 0.01)
        self.assertEqual(
            debug.model_pose_fit_source,
            "head_only_backside_ambiguous",
        )
        self.assertEqual(debug.pose_reprojection_rmse_px, 0.04)
        self.assertEqual(debug.pose_ambiguity_gap_px, 0.01)

    def test_absent_expected_geometry_preserves_seed_unavailable_contract(self):
        frame = self._synthetic_backside_frame()
        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_pipeline."
            "detect_qr_quad",
            return_value=None,
        ):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2,
                frame,
                model_profile=self.profile,
                camera_fx_px=self.camera.fx_px,
                camera_fy_px=self.camera.fy_px,
                camera_cx_px=self.camera.cx_px,
                camera_cy_px=self.camera.cy_px,
                blur_kernel=1,
            )

        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "model_pose_seed_unavailable")
        self.assertEqual(estimate.source, "model_seed")
        self.assertEqual(debug.pose_seed_source, "none")

    def test_backside_bootstrap_rejects_wrong_scale_off_center_and_no_neck(self):
        cases = (
            (
                "wrong_scale",
                self._synthetic_backside_frame(),
                {"expected_head_height_px": 60.0},
                "model_backside_head_scale_mismatch",
            ),
            (
                "off_center",
                self._synthetic_backside_frame(),
                {"expected_head_center_u_px": 110.0},
                "model_backside_target_center_mismatch",
            ),
            (
                "no_neck",
                self._synthetic_backside_frame(include_neck=False),
                {},
                "model_backside_head_and_neck_unavailable",
            ),
        )
        for name, frame, overrides, expected_reason in cases:
            with self.subTest(name=name), patch(
                "scripts.aufgabe04.perception.stand_axis.model_pipeline."
                "detect_qr_quad",
                return_value=None,
            ):
                estimate, _debug = estimate_stand_axis_from_metric_model(
                    cv2,
                    frame,
                    **self._backside_options(**overrides),
                )

            self.assertFalse(estimate.usable)
            self.assertEqual(estimate.reason, expected_reason)
            self.assertNotEqual(estimate.evidence_state, "fresh_backside")
            self.assertIsNone(estimate.visible_face)

    def test_detected_qr_never_falls_through_to_backside_bootstrap(self):
        frame = self._synthetic_backside_frame()
        qr_detection = QrQuadDetection(
            (
                ImagePoint(130.0, 60.0),
                ImagePoint(190.0, 60.0),
                ImagePoint(190.0, 120.0),
                ImagePoint(130.0, 120.0),
            ),
            1.0,
        )
        rejected_pose = PlanarPoseResult(
            accepted=False,
            reason="pose_reprojection_error",
            hypotheses=(),
            ambiguity_gap_px=None,
        )
        with (
            patch(
                "scripts.aufgabe04.perception.stand_axis.model_pipeline."
                "detect_qr_quad",
                return_value=qr_detection,
            ),
            patch(
                "scripts.aufgabe04.perception.stand_axis.model_pipeline."
                "estimate_planar_pose_ippe",
                return_value=rejected_pose,
            ),
        ):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2,
                frame,
                **self._backside_options(),
            )

        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "model_pose_seed_unavailable")
        self.assertNotEqual(estimate.source, MODEL_BACKSIDE_AXIS_SOURCE)
        self.assertTrue(debug.qr_detected)


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

    def test_camera_history_resolves_planar_axis_ambiguity_without_lidar(self):
        reference = oblique_pose()
        flipped = PlanarPoseHypothesis(
            rotation_vector=(0.0, -0.30, 0.0),
            translation_xyz_m=(0.01, 0.0, 0.40),
            face_normal_xyz=(-0.30, 0.0, 0.95),
            yaw_deg=17.5,
            reprojection_rmse_px=0.04,
            positive_depth=True,
        )
        consistent = PlanarPoseHypothesis(
            rotation_vector=reference.rotation_vector,
            translation_xyz_m=(0.002, 0.0, 0.402),
            face_normal_xyz=reference.face_normal_xyz,
            yaw_deg=-24.5,
            reprojection_rmse_px=0.05,
            positive_depth=True,
        )
        result = PlanarPoseResult(
            accepted=True,
            reason="pose_estimated",
            hypotheses=(flipped, consistent),
            ambiguity_gap_px=0.01,
        )

        selected = select_temporally_consistent_pose(result, reference)

        self.assertEqual(selected, consistent)

    def test_camera_history_keeps_unclear_planar_pose_fail_closed(self):
        reference = frontal_pose()
        near_left = PlanarPoseHypothesis(
            rotation_vector=(0.0, 0.02, 0.0),
            translation_xyz_m=(0.002, 0.0, 0.40),
            face_normal_xyz=(0.02, 0.0, 1.0),
            yaw_deg=-1.0,
            reprojection_rmse_px=0.04,
            positive_depth=True,
        )
        near_right = PlanarPoseHypothesis(
            rotation_vector=(0.0, -0.02, 0.0),
            translation_xyz_m=(-0.002, 0.0, 0.40),
            face_normal_xyz=(-0.02, 0.0, 1.0),
            yaw_deg=1.0,
            reprojection_rmse_px=0.05,
            positive_depth=True,
        )
        result = PlanarPoseResult(
            accepted=True,
            reason="pose_estimated",
            hypotheses=(near_left, near_right),
            ambiguity_gap_px=0.01,
        )

        self.assertIsNone(select_temporally_consistent_pose(result, reference))

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
