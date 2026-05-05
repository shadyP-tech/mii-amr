import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_rotation_runs as rotation  # noqa: E402


HEADER = [
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


class RotationAnalysisTest(unittest.TestCase):
    def test_run_range_selects_suffix_numbers(self):
        rows = [
            {"run_id": "run_real_rot_cw90_017"},
            {"run_id": "run_real_rot_cw90_018"},
            {"run_id": "run_real_rot_cw90_032"},
            {"run_id": "run_real_rot_cw90_033"},
        ]

        selected = rotation.filter_rows(rows, run_range_text="18:32")

        self.assertEqual(
            [row["run_id"] for row in selected],
            ["run_real_rot_cw90_018", "run_real_rot_cw90_032"],
        )

    def test_run_id_prefix_avoids_mixed_direction_ranges(self):
        rows = [
            {"run_id": "run_real_rot_cw90_001"},
            {"run_id": "run_real_rot_cw90_002"},
            {"run_id": "run_real_rot_ccw90_001"},
            {"run_id": "run_real_rot_ccw90_002"},
            {"run_id": "run_real_rot_ccw90_016"},
        ]

        selected = rotation.filter_rows(
            rows,
            run_range_text="1:15",
            run_id_prefix="run_real_rot_ccw90_",
        )

        self.assertEqual(
            [row["run_id"] for row in selected],
            ["run_real_rot_ccw90_001", "run_real_rot_ccw90_002"],
        )

    def test_tracker_yaw_sign_correction_and_drift_covariance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "rotation.csv"
            self.write_rows(
                csv_path,
                [
                    self.row("run_real_rot_cw90_018", 88.0, -87.5, 0.01, 0.02),
                    self.row("run_real_rot_cw90_019", 86.0, -86.5, 0.03, 0.04),
                    self.row("run_real_rot_cw90_020", 84.0, -85.5, 0.05, 0.06),
                ],
            )

            model = rotation.build_rotation_analysis(
                csv_path=csv_path,
                run_range="18:20",
                tracker_yaw_sign=-1.0,
            )

        tracker = model["tracker_rotation_model"]
        self.assertEqual(tracker["n"], 3)
        self.assertAlmostEqual(
            tracker["corrected_yaw_change_deg"]["mean"],
            -86.0,
        )
        self.assertAlmostEqual(
            tracker["corrected_yaw_error_deg"]["mean"],
            4.0,
        )
        self.assertAlmostEqual(tracker["drift_mu"][0], 0.03)
        self.assertAlmostEqual(tracker["drift_mu"][1], 0.04)
        self.assertAlmostEqual(tracker["drift_sigma"][0][0], 0.0004)
        self.assertAlmostEqual(tracker["drift_sigma"][0][1], 0.0004)
        self.assertAlmostEqual(tracker["drift_sigma"][1][1], 0.0004)
        self.assertAlmostEqual(tracker["local_drift_mu"][0], 0.04)
        self.assertAlmostEqual(tracker["local_drift_mu"][1], -0.03)
        self.assertAlmostEqual(tracker["local_drift_sigma"][0][0], 0.0004)
        self.assertAlmostEqual(tracker["local_drift_sigma"][0][1], -0.0004)
        self.assertAlmostEqual(tracker["local_drift_sigma"][1][1], 0.0004)

    def test_missing_required_columns_fail_clearly(self):
        with self.assertRaises(rotation.DataError) as ctx:
            rotation.require_columns(["run_id"], rotation.REQUIRED_COLUMNS, "r.csv")

        self.assertIn("command_angle_deg", str(ctx.exception))

    def write_rows(self, path, rows):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def row(self, run_id, tracker_yaw, odom_yaw, dx, dy):
        drift = (dx * dx + dy * dy) ** 0.5
        return {
            "timestamp": "2026-05-04T14:00:00",
            "run_id": run_id,
            "run_mode": "rotate-in-place",
            "command_angle_deg": "-90",
            "direction": "clockwise",
            "linear_x_mps": "0.0",
            "angular_z_radps": "-0.3",
            "duration_sec": "5.235987755982989",
            "tracker_start_x": "0.0",
            "tracker_start_y": "0.0",
            "tracker_start_yaw_deg": "90.0",
            "tracker_final_x": str(dx),
            "tracker_final_y": str(dy),
            "tracker_final_yaw_deg": "180.0",
            "tracker_yaw_change_deg": str(tracker_yaw),
            "tracker_yaw_error_deg": "0.0",
            "tracker_dx": str(dx),
            "tracker_dy": str(dy),
            "tracker_position_drift_m": str(drift),
            "odom_start_x": "0.0",
            "odom_start_y": "0.0",
            "odom_start_yaw_deg": "90.0",
            "odom_final_x": "0.0",
            "odom_final_y": "0.0",
            "odom_final_yaw_deg": "0.0",
            "odom_yaw_change_deg": str(odom_yaw),
            "odom_yaw_error_deg": str(90 + odom_yaw),
            "odom_dx": "0.0",
            "odom_dy": "0.0",
            "odom_position_drift_m": "0.0",
            "notes": "real_rotation",
        }


if __name__ == "__main__":
    unittest.main()
