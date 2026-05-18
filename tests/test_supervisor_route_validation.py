import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe02"))

import supervisor_route_validation as validation  # noqa: E402


class SupervisorRouteValidationTest(unittest.TestCase):
    def test_parse_forward_and_rotation_actions(self):
        forward = validation.parse_action("F30")
        clockwise = validation.parse_action("CW90")
        counterclockwise = validation.parse_action("CCW45")

        self.assertEqual(forward["kind"], "forward")
        self.assertAlmostEqual(forward["distance_m"], 0.30)

        self.assertEqual(clockwise["kind"], "rotate")
        self.assertAlmostEqual(clockwise["angle_deg"], -90.0)

        self.assertEqual(counterclockwise["kind"], "rotate")
        self.assertAlmostEqual(counterclockwise["angle_deg"], 45.0)

    def test_parse_action_rejects_invalid_action(self):
        with self.assertRaises(ValueError):
            validation.parse_action("LEFT90")

    def test_load_prediction_parses_actions_and_final_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prediction.json"
            path.write_text(
                json.dumps({
                    "actions": ["F30", "CCW45"],
                    "fixed_points": [[0.0, 0.0], [0.3, 0.0]],
                    "prediction": {
                        "endpoint_mu": [0.31, 0.02],
                        "final_yaw_mean_deg": 44.0,
                    },
                })
            )

            prediction = validation.load_prediction(path)

        self.assertEqual(prediction["actions"], ["F30", "CCW45"])
        self.assertEqual(len(prediction["parsed_actions"]), 2)
        self.assertEqual(prediction["nominal_final"], [0.3, 0.0])
        self.assertEqual(prediction["predicted_final"], [0.31, 0.02])
        self.assertAlmostEqual(prediction["predicted_final_yaw_deg"], 44.0)

    def test_build_result_row_computes_tracker_and_odom_errors(self):
        prediction = {
            "path": Path("results/example_prediction.json"),
            "actions": ["F30"],
            "nominal_final": [0.30, 0.0],
            "predicted_final": [0.31, 0.02],
            "predicted_final_yaw_deg": 10.0,
        }
        tracker_final = {
            "timestamp": "2026-05-07T12:00:00",
            "x": 0.34,
            "y": -0.02,
            "yaw_deg": 12.0,
        }
        odom_start = {"x": 1.0, "y": 2.0, "yaw_deg": 0.0}
        odom_final = {"x": 1.3, "y": 2.4, "yaw_deg": 5.0}

        row = validation.build_result_row(
            "run_a",
            prediction,
            tracker_final,
            odom_start,
            odom_final,
            0.1,
            0.3,
            "test",
        )
        values = dict(zip(validation.CSV_HEADER, row))

        self.assertEqual(values["run_id"], "run_a")
        self.assertEqual(values["actions"], "F30")
        self.assertEqual(values["model_actions"], "F30")
        self.assertEqual(values["comparison_frame"], "model_mirror_y")
        self.assertAlmostEqual(values["tracker_final_x"], 0.34)
        self.assertAlmostEqual(values["tracker_final_y"], -0.02)
        self.assertAlmostEqual(values["tracker_final_yaw_deg"], 12.0)
        self.assertAlmostEqual(values["tracker_final_x_model"], 0.34)
        self.assertAlmostEqual(values["tracker_final_y_model"], 0.02)
        self.assertAlmostEqual(values["tracker_final_yaw_model_deg"], -12.0)
        self.assertAlmostEqual(values["tracker_error_dx"], 0.03)
        self.assertAlmostEqual(values["tracker_error_dy"], 0.0)
        self.assertAlmostEqual(values["tracker_error_m"], 0.03)
        self.assertAlmostEqual(values["tracker_yaw_error_deg"], -22.0)
        self.assertAlmostEqual(values["odom_final_x_model"], 1.3)
        self.assertAlmostEqual(values["odom_final_y_model"], -2.4)
        self.assertAlmostEqual(values["odom_final_yaw_model_deg"], -5.0)
        self.assertAlmostEqual(values["odom_dx"], 0.3)
        self.assertAlmostEqual(values["odom_dy"], 0.4)
        self.assertAlmostEqual(values["odom_distance_m"], 0.5)

    def test_append_csv_row_migrates_legacy_supervisor_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            legacy_row = {
                field: ""
                for field in validation.LEGACY_CSV_HEADER
            }
            legacy_row.update({
                "timestamp": "2026-05-07T12:00:00",
                "run_id": "run_legacy",
                "prediction_file": "results/example_prediction.json",
                "actions": "F30",
                "num_actions": "1",
                "nominal_final_x": "0.3",
                "nominal_final_y": "0.0",
                "predicted_final_x": "0.31",
                "predicted_final_y": "0.02",
                "predicted_final_yaw_deg": "10.0",
                "tracker_final_timestamp": "2026-05-07T12:00:01",
                "tracker_final_x": "0.34",
                "tracker_final_y": "-0.02",
                "tracker_final_yaw_deg": "12.0",
                "odom_start_x": "0.0",
                "odom_start_y": "0.0",
                "odom_start_yaw_deg": "0.0",
                "odom_final_x": "0.3",
                "odom_final_y": "-0.1",
                "odom_final_yaw_deg": "5.0",
                "linear_speed_mps": "0.1",
                "angular_speed_radps": "0.3",
                "notes": "legacy",
            })
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=validation.LEGACY_CSV_HEADER)
                writer.writeheader()
                writer.writerow(legacy_row)

            new_row = [""] * len(validation.CSV_HEADER)
            validation.append_csv_row(path, validation.CSV_HEADER, new_row)

            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["comparison_frame"], "model_mirror_y")
        self.assertEqual(rows[0]["actions"], "F30")
        self.assertEqual(rows[0]["model_actions"], "F30")
        self.assertEqual(rows[0]["tracker_final_y"], "-0.02")
        self.assertEqual(float(rows[0]["tracker_final_y_model"]), 0.02)
        self.assertEqual(float(rows[0]["tracker_error_dy"]), 0.0)
        self.assertEqual(len(rows), 2)

    def test_legacy_migration_mirrors_rotation_labels_for_model_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            legacy_row = {
                field: ""
                for field in validation.LEGACY_CSV_HEADER
            }
            legacy_row.update({
                "run_id": "run_legacy",
                "prediction_file": "results/example_prediction.json",
                "actions": "F30,CW45,CCW90,CCW180,F180",
                "num_actions": "5",
                "predicted_final_x": "0.0",
                "predicted_final_y": "0.0",
                "predicted_final_yaw_deg": "0.0",
                "notes": "legacy",
            })
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=validation.LEGACY_CSV_HEADER)
                writer.writeheader()
                writer.writerow(legacy_row)

            validation.append_csv_row(path, validation.CSV_HEADER, [""] * len(validation.CSV_HEADER))

            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["actions"], "F30,CW45,CCW90,CCW180,F180")
        self.assertEqual(rows[0]["model_actions"], "F30,CCW45,CW90,CCW180,F180")

    def test_append_csv_row_rejects_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["unexpected"])

            with self.assertRaises(RuntimeError):
                validation.append_csv_row(path, validation.CSV_HEADER, [])


if __name__ == "__main__":
    unittest.main()
