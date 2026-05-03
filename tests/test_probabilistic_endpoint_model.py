import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_probabilistic_endpoint_model as model  # noqa: E402
import predict_waypoint_endpoint_region as waypoint  # noqa: E402


REAL_HEADER = [
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


class ProbabilisticEndpointModelTest(unittest.TestCase):
    def test_real_run_range_parsing_selects_expected_ids(self):
        rows = [
            {"run_id": "run_real_020"},
            {"run_id": "run_real_021"},
            {"run_id": "run_real_050"},
            {"run_id": "run_real_051"},
        ]

        selected = model.filter_rows_by_run_range(rows, "21:50")

        self.assertEqual([row["run_id"] for row in selected], [
            "run_real_021",
            "run_real_050",
        ])

    def test_latest_n_filter_selects_final_rows(self):
        rows = [{"run_id": f"sim_{i:02d}"} for i in range(20)]

        selected = model.filter_latest_rows(rows, 15)

        self.assertEqual(selected[0]["run_id"], "sim_05")
        self.assertEqual(selected[-1]["run_id"], "sim_19")
        self.assertEqual(len(selected), 15)

    def test_required_column_validation_fails_clearly(self):
        with self.assertRaises(model.DataError) as ctx:
            model.require_columns(["run_id"], ["run_id", "tracker_final_x"], "x.csv")

        self.assertIn("tracker_final_x", str(ctx.exception))

    def test_missing_endpoint_values_are_skipped_and_recorded(self):
        rows = [
            {
                "_row_number": 2,
                "run_id": "run_real_021",
                "tracker_final_x": "0.2",
                "tracker_final_y": "0.0",
            },
            {
                "_row_number": 3,
                "run_id": "run_real_022",
                "tracker_final_x": "",
                "tracker_final_y": "0.0",
            },
        ]

        valid, skipped = model.valid_rows_with_columns(
            rows,
            ["run_id", "tracker_final_x", "tracker_final_y"],
        )

        self.assertEqual([row["run_id"] for row in valid], ["run_real_021"])
        self.assertEqual(skipped[0]["run_id"], "run_real_022")
        self.assertIn("tracker_final_x", skipped[0]["reason"])

    def test_absolute_endpoint_mean_and_covariance_match_known_data(self):
        points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        mu, sigma = model.empirical_mean_cov(points)

        self.assertAllClose(mu, [3.0, 4.0])
        self.assertAllClose(sigma, [[4.0, 4.0], [4.0, 4.0]])

    def test_motion_model_computes_world_delta(self):
        rows = [
            {
                "tracker_start_x": "0.0",
                "tracker_start_y": "0.0",
                "tracker_start_yaw_deg": "0.0",
                "tracker_final_x": "0.3",
                "tracker_final_y": "0.1",
            }
        ]

        delta = model.local_displacements(rows, "tracker")

        self.assertAllClose(delta, [[0.3, 0.1]])

    def test_local_frame_rotation_uses_start_yaw(self):
        rows = [
            {
                "tracker_start_x": "0.0",
                "tracker_start_y": "0.0",
                "tracker_start_yaw_deg": "90.0",
                "tracker_final_x": "0.0",
                "tracker_final_y": "0.3",
            }
        ]

        delta = model.local_displacements(rows, "tracker")

        self.assertAllClose(delta, [[0.3, 0.0]], places=12)

    def test_motion_error_subtracts_commanded_distance(self):
        error = model.motion_errors([[0.31, 0.02]], 0.30)

        self.assertAllClose(error, [[0.01, 0.02]])

    def test_circular_yaw_mean_handles_wraparound(self):
        summary = model.circular_yaw_summary_deg([179.0, -179.0])

        self.assertAlmostEqual(abs(summary["mean_deg"]), 180.0)
        self.assertLess(summary["std_deg"], 2.0)

    def test_ellipse_uses_2d_95_percent_chi_square_threshold(self):
        params = model.ellipse_parameters(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            chi2_value=model.CHI2_95_2D,
        )

        self.assertAlmostEqual(params["chi2_value"], 5.991)
        self.assertAlmostEqual(
            params["major_axis_length_m"],
            2.0 * math.sqrt(5.991),
        )

    def test_mahalanobis_outliers_handle_singular_covariance(self):
        points = [[0.0, 0.0], [1.0, 1.0]]
        sigma = [[1.0, 1.0], [1.0, 1.0]]

        distances = model.mahalanobis_squared(points, [0.0, 0.0], sigma)

        self.assertTrue(all(math.isfinite(distance) for distance in distances))

    def test_build_model_records_json_metadata_and_uses_local_error_bias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_csv = root / "real.csv"
            sim_csv = root / "sim.csv"
            self.write_rows(
                real_csv,
                [
                    self.row("run_real_021", 100.0, 100.0, 0.0, 100.31, 100.02, 0.0),
                    self.row("run_real_022", 200.0, 200.0, 0.0, 200.31, 200.02, 0.0),
                ],
            )
            self.write_rows(
                sim_csv,
                [
                    self.row("simulation_01", 0.0, 0.0, 0.0, 0.29, -0.01, 0.0),
                    self.row("simulation_02", 10.0, 10.0, 0.0, 10.29, 9.99, 0.0),
                ],
            )

            built, _ = model.build_analysis_model(
                real_csv=real_csv,
                real_run_range="21:22",
                sim_csv=sim_csv,
                sim_last_n=2,
                step_distance_m=0.30,
                compare_sim_real=True,
            )

        self.assertEqual(built["units"]["position"], "m")
        self.assertEqual(
            built["coordinate_frames"]["motion_primitive_error_model"],
            "robot local start frame",
        )
        self.assertEqual(
            built["data_selection"]["selected_real_run_ids"],
            ["run_real_021", "run_real_022"],
        )
        self.assertEqual(
            built["data_selection"]["selected_sim_run_ids"],
            ["simulation_01", "simulation_02"],
        )
        self.assertAlmostEqual(
            built["motion_primitive_error_model"]["mu_error"][0],
            0.01,
        )
        self.assertAlmostEqual(
            built["motion_primitive_error_model"]["mu_error"][1],
            0.02,
        )
        self.assertAlmostEqual(
            built["sim2real_displacement_bias"]["dx_m"],
            0.02,
        )
        self.assertAlmostEqual(
            built["sim2real_displacement_bias"]["dy_m"],
            0.03,
        )

    def test_written_json_contains_required_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            data = {
                "units": {"position": "m", "angle": "deg"},
                "coordinate_frames": {"motion_primitive_error_model": "robot local"},
                "data_selection": {"selected_real_run_ids": ["run_real_021"]},
            }

            model.write_json(path, data)

            loaded = json.loads(path.read_text())

        self.assertEqual(loaded["units"]["angle"], "deg")
        self.assertEqual(
            loaded["data_selection"]["selected_real_run_ids"],
            ["run_real_021"],
        )

    def test_waypoint_prediction_accumulates_mean_bias_and_covariance(self):
        waypoints = [[0.0, 0.0], [0.6, 0.0], [0.6, 0.3]]
        mu_error = [0.01, 0.02]
        sigma_error = [[0.0001, 0.0], [0.0, 0.0004]]

        prediction = waypoint.predict_endpoint_region(
            waypoints,
            mu_error,
            sigma_error,
            step_distance_m=0.30,
        )

        self.assertAllClose(prediction["predicted_mu"], [0.6, 0.35])
        self.assertAllClose(
            prediction["sigma"],
            [[0.0006, 0.0], [0.0, 0.0009]],
            places=12,
        )
        self.assertEqual(prediction["primitive_count"], 3)

    def test_waypoint_prediction_rejects_non_multiple_segments_by_default(self):
        waypoints = [[0.0, 0.0], [0.5, 0.0]]

        with self.assertRaises(ValueError):
            waypoint.predict_endpoint_region(
                waypoints,
                [0.01, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                step_distance_m=0.30,
            )

    def test_waypoint_prediction_allows_remainder_scaling_when_requested(self):
        waypoints = [[0.0, 0.0], [0.5, 0.0]]

        prediction = waypoint.predict_endpoint_region(
            waypoints,
            [0.03, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            step_distance_m=0.30,
            allow_remainder_scaling=True,
        )

        self.assertEqual(prediction["primitive_count"], 2)
        self.assertAlmostEqual(prediction["remainder_segments"][0]["scale"], 2.0 / 3.0)
        self.assertAlmostEqual(prediction["predicted_mu"][0], 0.55)

    @staticmethod
    def row(run_id, start_x, start_y, start_yaw, final_x, final_y, final_yaw):
        return [
            "2026-05-03T00:00:00",
            run_id,
            start_x,
            start_y,
            start_yaw,
            final_x,
            final_y,
            final_yaw,
            start_x,
            start_y,
            start_yaw,
            final_x,
            final_y,
            final_yaw,
            "test",
        ]

    @staticmethod
    def write_rows(path, rows):
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(REAL_HEADER)
            writer.writerows(rows)

    def assertAllClose(self, actual, expected, places=7):
        if isinstance(expected[0], list):
            self.assertEqual(len(actual), len(expected))
            for actual_row, expected_row in zip(actual, expected):
                self.assertAllClose(actual_row, expected_row, places=places)
            return

        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)


if __name__ == "__main__":
    unittest.main()
