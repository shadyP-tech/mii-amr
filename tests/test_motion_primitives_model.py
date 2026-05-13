import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_motion_primitives_model as builder  # noqa: E402
import predict_primitive_path_endpoint as predictor  # noqa: E402


FORWARD_HEADER = [
    "timestamp",
    "run_id",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",
    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",
    "notes",
]

ROTATION_HEADER = [
    "timestamp",
    "run_id",
    "run_mode",
    "command_angle_deg",
    "direction",
    "linear_x_mps",
    "angular_z_radps",
    "duration_sec",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "tracker_yaw_change_deg",
    "tracker_yaw_error_deg",
    "tracker_dx",
    "tracker_dy",
    "tracker_position_drift_m",
    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",
    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",
    "odom_yaw_change_deg",
    "odom_yaw_error_deg",
    "odom_dx",
    "odom_dy",
    "odom_position_drift_m",
    "notes",
]

VALIDATION_HEADER = [
    "timestamp",
    "run_id",
    "actions",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "notes",
]

SUPERVISOR_VALIDATION_HEADER = [
    "timestamp",
    "run_id",
    "prediction_file",
    "actions",
    "model_actions",
    "num_actions",
    "nominal_final_x",
    "nominal_final_y",
    "predicted_final_x",
    "predicted_final_y",
    "predicted_final_yaw_deg",
    "comparison_frame",
    "tracker_final_timestamp",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "tracker_final_x_model",
    "tracker_final_y_model",
    "tracker_final_yaw_model_deg",
    "tracker_error_dx",
    "tracker_error_dy",
    "tracker_error_m",
    "tracker_yaw_error_deg",
    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",
    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",
    "odom_final_x_model",
    "odom_final_y_model",
    "odom_final_yaw_model_deg",
    "odom_dx",
    "odom_dy",
    "odom_distance_m",
    "linear_speed_mps",
    "angular_speed_radps",
    "notes",
]


class MotionPrimitivesModelTest(unittest.TestCase):
    def test_pose_delta_wraps_yaw_before_sign_correction(self):
        row = {
            "tracker_start_x": "0.0",
            "tracker_start_y": "0.0",
            "tracker_start_yaw_deg": "179.0",
            "tracker_final_x": "0.0",
            "tracker_final_y": "0.0",
            "tracker_final_yaw_deg": "-176.0",
        }

        _delta, yaw_delta = builder.pose_local_delta_and_yaw_delta(
            row,
            "tracker",
            tracker_yaw_sign=1.0,
        )

        self.assertAlmostEqual(yaw_delta, 5.0)

    def test_local_displacement_uses_start_yaw(self):
        row = {
            "tracker_start_x": "0.0",
            "tracker_start_y": "0.0",
            "tracker_start_yaw_deg": "90.0",
            "tracker_final_x": "0.0",
            "tracker_final_y": "0.3",
            "tracker_final_yaw_deg": "90.0",
        }

        delta, _yaw_delta = builder.pose_local_delta_and_yaw_delta(
            row,
            "tracker",
            tracker_yaw_sign=1.0,
        )

        self.assertAlmostEqual(delta[0], 0.3, places=12)
        self.assertAlmostEqual(delta[1], 0.0, places=12)

    def test_builds_f30_from_actual_local_displacement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            forward_csv = root / "forward.csv"
            rotation_csv = root / "rotation.csv"
            self.write_rows(
                forward_csv,
                FORWARD_HEADER,
                [
                    self.forward_row("run_real_021", 0.31, 0.02, 1.0),
                    self.forward_row("run_real_022", 0.35, 0.04, 2.0),
                ],
            )
            self.write_rows(
                rotation_csv,
                ROTATION_HEADER,
                [
                    self.rotation_row("run_real_rot_cw90_018", 85.0, 0.01, 0.02),
                    self.rotation_row("run_real_rot_cw90_019", 83.0, 0.02, 0.03),
                    self.rotation_row("run_real_rot_ccw90_001", -86.0, -0.01, 0.02),
                    self.rotation_row("run_real_rot_ccw90_002", -84.0, -0.02, 0.03),
                ],
            )

            model = builder.build_motion_primitives_model(
                forward_csv=forward_csv,
                forward_run_range="21:22",
                rotation_csv=rotation_csv,
                cw_prefix="run_real_rot_cw90_",
                cw_run_range="18:19",
                ccw_prefix="run_real_rot_ccw90_",
                ccw_run_range="1:2",
                tracker_yaw_sign=-1.0,
            )

        f30 = model["primitives"]["F30"]
        self.assertAlmostEqual(f30["local_delta_mu"][0], 0.33)
        self.assertAlmostEqual(f30["local_delta_mu"][1], 0.03)
        self.assertAlmostEqual(f30["yaw_delta_mean_deg"], -1.5)
        self.assertLess(model["primitives"]["CW90"]["yaw_delta_mean_deg"], 0.0)
        self.assertGreater(model["primitives"]["CCW90"]["yaw_delta_mean_deg"], 0.0)

    def test_builds_optional_route_primitives_by_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            forward_csv = root / "forward.csv"
            rotation_csv = root / "rotation.csv"
            self.write_rows(
                forward_csv,
                FORWARD_HEADER,
                [
                    self.forward_row("run_real_021", 0.31, 0.02, 1.0),
                    self.forward_row("run_real_022", 0.35, 0.04, 2.0),
                    self.forward_row("run_real_fwd50_001", 0.51, 0.03, 0.5),
                    self.forward_row("run_real_fwd50_002", 0.53, 0.05, 0.8),
                ],
            )
            self.write_rows(
                rotation_csv,
                ROTATION_HEADER,
                [
                    self.rotation_row("run_real_rot_cw90_018", 85.0, 0.01, 0.02),
                    self.rotation_row("run_real_rot_cw90_019", 83.0, 0.02, 0.03),
                    self.rotation_row("run_real_rot_ccw90_001", -86.0, -0.01, 0.02),
                    self.rotation_row("run_real_rot_ccw90_002", -84.0, -0.02, 0.03),
                    self.rotation_row("run_real_rot_ccw45_001", -44.0, -0.01, 0.01),
                    self.rotation_row("run_real_rot_ccw45_002", -42.0, -0.02, 0.02),
                ],
            )

            model = builder.build_motion_primitives_model(
                forward_csv=forward_csv,
                forward_run_range="21:22",
                rotation_csv=rotation_csv,
                cw_prefix="run_real_rot_cw90_",
                cw_run_range="18:19",
                ccw_prefix="run_real_rot_ccw90_",
                ccw_run_range="1:2",
                tracker_yaw_sign=-1.0,
                extra_forward_specs=[
                    {
                        "name": "F50",
                        "run_id_prefix": "run_real_fwd50_",
                        "run_range": "1:2",
                    }
                ],
                extra_rotation_specs=[
                    {
                        "name": "CCW45",
                        "run_id_prefix": "run_real_rot_ccw45_",
                        "run_range": "1:2",
                    }
                ],
            )

        self.assertIn("F50", model["primitives"])
        self.assertIn("CCW45", model["primitives"])
        self.assertAlmostEqual(model["primitives"]["F50"]["local_delta_mu"][0], 0.52)
        self.assertGreater(model["primitives"]["CCW45"]["yaw_delta_mean_deg"], 0.0)
        self.assertEqual(
            model["data_selection"]["primitive_sources"]["F50"]["selected_run_ids"],
            ["run_real_fwd50_001", "run_real_fwd50_002"],
        )

    def test_covariance_validation_rejects_invalid_matrices(self):
        with self.assertRaises(builder.PrimitiveModelError):
            builder.validate_covariance("X", [[1.0, 0.1], [0.0, 1.0]])
        with self.assertRaises(builder.PrimitiveModelError):
            builder.validate_covariance("X", [[-1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(builder.PrimitiveModelError):
            builder.validate_covariance("X", [[1.0, 2.0], [2.0, 1.0]])

    def test_predictor_parses_and_rejects_actions(self):
        self.assertEqual(
            predictor.parse_actions("F30, ccw90, F30"),
            ["F30", "CCW90", "F30"],
        )
        with self.assertRaises(ValueError):
            predictor.parse_actions("")

    def test_predictor_rejects_unknown_action_and_invalid_samples(self):
        model = self.zero_cov_model()

        with self.assertRaises(ValueError):
            predictor.predict_action_sequence(model, ["UNKNOWN"], [0, 0, 0], 1, 7)
        with self.assertRaises(ValueError):
            predictor.predict_action_sequence(model, ["F30"], [0, 0, 0], 0, 7)

    def test_zero_covariance_prediction_is_deterministic(self):
        model = self.zero_cov_model()

        prediction = predictor.predict_action_sequence(
            model,
            ["F30", "CCW90", "F30"],
            [0.0, 0.0, 0.0],
            20,
            7,
        )

        self.assertAlmostEqual(prediction["endpoint_mu"][0], 1.0)
        self.assertAlmostEqual(prediction["endpoint_mu"][1], 1.0)
        self.assertAlmostEqual(prediction["endpoint_sigma"][0][0], 0.0)
        self.assertAlmostEqual(prediction["endpoint_sigma"][1][1], 0.0)
        self.assertAlmostEqual(prediction["yaw_summary"]["mean_deg"], 90.0)

    def test_fixed_seed_prediction_is_reproducible(self):
        model = self.noisy_model()

        first = predictor.predict_action_sequence(model, ["F30", "F30"], [0, 0, 0], 5, 3)
        second = predictor.predict_action_sequence(model, ["F30", "F30"], [0, 0, 0], 5, 3)

        self.assertEqual(first["final_poses"], second["final_poses"])

    def test_mirrored_execution_uses_physical_plot_frame(self):
        output = {
            "actions": ["F30", "CCW45", "F50"],
            "execution_actions": ["F30", "CW45", "F50"],
            "fixed_points": [[0.0, 0.0], [1.0, -0.5], [2.0, 0.25]],
            "validation": None,
        }

        self.assertEqual(predictor.output_plot_frame(output), "physical")
        self.assertEqual(predictor.fixed_points_frame(output), "physical")
        self.assertEqual(
            predictor.fixed_points_for_plot(output),
            [[0.0, 0.0], [1.0, -0.5], [2.0, 0.25]],
        )
        self.assertEqual(
            predictor.model_points_for_plot([[1.0, -0.5]], output),
            [[1.0, 0.5]],
        )
        self.assertEqual(
            predictor.model_sigma_for_plot([[1.0, 0.25], [0.25, 2.0]], output),
            [[1.0, -0.25], [-0.25, 2.0]],
        )

    def test_fixed_points_stay_unchanged_when_model_and_execution_actions_match(self):
        output = {
            "actions": ["F30", "CCW45", "F50"],
            "execution_actions": ["F30", "CCW45", "F50"],
            "fixed_points": [[0.0, 0.0], [1.0, -0.5]],
            "validation": None,
        }

        self.assertEqual(predictor.output_plot_frame(output), "model")
        self.assertEqual(predictor.fixed_points_frame(output), "model")
        self.assertEqual(
            predictor.fixed_points_for_plot(output),
            [[0.0, 0.0], [1.0, -0.5]],
        )

    def test_target_endpoint_uses_plotted_fixed_point_frame(self):
        output = {
            "actions": ["F30", "CCW45", "F50"],
            "execution_actions": ["F30", "CW45", "F50"],
            "fixed_points": [[0.0, 0.0], [1.0, -0.5]],
            "validation": None,
        }

        self.assertEqual(predictor.target_endpoint_for_plot(output), [1.0, -0.5])

    def test_physical_plot_uses_raw_validation_pose(self):
        output = {
            "actions": ["F30", "CCW45"],
            "execution_actions": ["F30", "CW45"],
            "fixed_points": [],
            "validation": None,
        }
        validation = {
            "tracker_final_pose": [3.7, 0.2, 174.0],
            "tracker_final_pose_raw": [3.7, -0.2, -174.0],
        }

        self.assertEqual(
            predictor.validation_pose_for_plot(validation, output),
            [3.7, -0.2, -174.0],
        )

    def test_confidence_contours_increase_to_95_percent(self):
        contours = predictor.confidence_contours(
            [0.0, 0.0],
            [[0.01, 0.0], [0.0, 0.04]],
        )
        chi2_values = [contour["ellipse"]["chi2_value"] for contour in contours]

        self.assertEqual([contour["label"] for contour in contours], ["50%", "68%", "80%", "95%"])
        self.assertEqual(chi2_values, sorted(chi2_values))
        self.assertAlmostEqual(chi2_values[-1], 5.991)

    def test_validation_csv_row_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "validation.csv"
            self.write_rows(
                path,
                VALIDATION_HEADER,
                [
                    {
                        "timestamp": "2026-05-04T15:00:00",
                        "run_id": "path_real_001",
                        "actions": "F30,CCW90,F30",
                        "tracker_start_x": "0.0",
                        "tracker_start_y": "0.0",
                        "tracker_start_yaw_deg": "0.0",
                        "tracker_final_x": "1.0",
                        "tracker_final_y": "1.0",
                        "tracker_final_yaw_deg": "90.0",
                        "notes": "validation",
                    }
                ],
            )

            row = predictor.load_validation_row(
                path,
                "path_real_001",
                ["F30", "CCW90", "F30"],
            )

        self.assertEqual(row["run_id"], "path_real_001")
        self.assertEqual(row["tracker_final_pose"], [1.0, 1.0, 90.0])
        self.assertIsNone(row["warning"])

    def test_supervisor_validation_csv_row_is_loaded_without_start_pose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "supervisor_validation.csv"
            self.write_rows(
                path,
                SUPERVISOR_VALIDATION_HEADER,
                [
                    {
                        "timestamp": "2026-05-07T14:32:50",
                        "run_id": "supervisor_validation_003",
                        "prediction_file": "results/supervisor_route_prediction.json",
                        "actions": "F30,CW45",
                        "model_actions": "F30,CCW45",
                        "num_actions": "2",
                        "nominal_final_x": "0.0",
                        "nominal_final_y": "0.0",
                        "predicted_final_x": "0.0",
                        "predicted_final_y": "0.0",
                        "predicted_final_yaw_deg": "0.0",
                        "comparison_frame": "model_mirror_y",
                        "tracker_final_timestamp": "2026-05-07T14:32:49",
                        "tracker_final_x": "3.7",
                        "tracker_final_y": "-0.2",
                        "tracker_final_yaw_deg": "-174.0",
                        "tracker_final_x_model": "3.7",
                        "tracker_final_y_model": "0.2",
                        "tracker_final_yaw_model_deg": "174.0",
                        "tracker_error_dx": "",
                        "tracker_error_dy": "",
                        "tracker_error_m": "",
                        "tracker_yaw_error_deg": "",
                        "odom_start_x": "",
                        "odom_start_y": "",
                        "odom_start_yaw_deg": "",
                        "odom_final_x": "",
                        "odom_final_y": "",
                        "odom_final_yaw_deg": "",
                        "odom_final_x_model": "",
                        "odom_final_y_model": "",
                        "odom_final_yaw_model_deg": "",
                        "odom_dx": "",
                        "odom_dy": "",
                        "odom_distance_m": "",
                        "linear_speed_mps": "0.1",
                        "angular_speed_radps": "0.3",
                        "notes": "supervisor_route_validation",
                    }
                ],
            )

            row = predictor.load_validation_row(
                path,
                "supervisor_validation_003",
                ["F30", "CCW45"],
            )

        self.assertEqual(row["run_id"], "supervisor_validation_003")
        self.assertIsNone(row["tracker_start_pose"])
        self.assertEqual(row["actions"], "F30,CW45")
        self.assertEqual(row["model_actions"], "F30,CCW45")
        self.assertEqual(row["comparison_frame"], "model_mirror_y")
        self.assertEqual(row["tracker_final_pose"], [3.7, 0.2, 174.0])
        self.assertEqual(row["tracker_final_pose_raw"], [3.7, -0.2, -174.0])
        self.assertIsNone(row["warning"])

    def write_rows(self, path, header, rows):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def forward_row(self, run_id, dx, dy, yaw_delta):
        return {
            "timestamp": "2026-05-04T15:00:00",
            "run_id": run_id,
            "tracker_start_x": "0.0",
            "tracker_start_y": "0.0",
            "tracker_start_yaw_deg": "0.0",
            "tracker_final_x": str(dx),
            "tracker_final_y": str(dy),
            "tracker_final_yaw_deg": str(yaw_delta),
            "odom_start_x": "0.0",
            "odom_start_y": "0.0",
            "odom_start_yaw_deg": "0.0",
            "odom_final_x": str(dx),
            "odom_final_y": str(dy),
            "odom_final_yaw_deg": str(yaw_delta),
            "notes": "real",
        }

    def rotation_row(self, run_id, raw_yaw_delta, dx, dy):
        is_ccw = "ccw90" in run_id
        return {
            "timestamp": "2026-05-04T15:00:00",
            "run_id": run_id,
            "run_mode": "rotate-in-place",
            "command_angle_deg": "90.0" if is_ccw else "-90.0",
            "direction": "counterclockwise" if is_ccw else "clockwise",
            "linear_x_mps": "0.0",
            "angular_z_radps": "0.3" if is_ccw else "-0.3",
            "duration_sec": "5.235987755982989",
            "tracker_start_x": "0.0",
            "tracker_start_y": "0.0",
            "tracker_start_yaw_deg": "0.0",
            "tracker_final_x": str(dx),
            "tracker_final_y": str(dy),
            "tracker_final_yaw_deg": str(raw_yaw_delta),
            "tracker_yaw_change_deg": str(raw_yaw_delta),
            "tracker_yaw_error_deg": "0.0",
            "tracker_dx": str(dx),
            "tracker_dy": str(dy),
            "tracker_position_drift_m": str((dx * dx + dy * dy) ** 0.5),
            "odom_start_x": "0.0",
            "odom_start_y": "0.0",
            "odom_start_yaw_deg": "0.0",
            "odom_final_x": "0.0",
            "odom_final_y": "0.0",
            "odom_final_yaw_deg": str(-raw_yaw_delta),
            "odom_yaw_change_deg": str(-raw_yaw_delta),
            "odom_yaw_error_deg": "0.0",
            "odom_dx": "0.0",
            "odom_dy": "0.0",
            "odom_position_drift_m": "0.0",
            "notes": "real_rotation",
        }

    def zero_cov_model(self):
        return {
            "primitives": {
                "F30": self.primitive([1.0, 0.0], 0.0),
                "CCW90": self.primitive([0.0, 0.0], 90.0),
            }
        }

    def noisy_model(self):
        return {
            "primitives": {
                "F30": {
                    "local_delta_mu": [1.0, 0.0],
                    "local_delta_sigma": [[0.01, 0.0], [0.0, 0.02]],
                    "yaw_delta_mean_deg": 0.0,
                    "yaw_delta_std_deg": 1.0,
                }
            }
        }

    def primitive(self, local_delta_mu, yaw_delta):
        return {
            "local_delta_mu": local_delta_mu,
            "local_delta_sigma": [[0.0, 0.0], [0.0, 0.0]],
            "yaw_delta_mean_deg": yaw_delta,
            "yaw_delta_std_deg": 0.0,
        }


if __name__ == "__main__":
    unittest.main()
