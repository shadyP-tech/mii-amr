from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.simulation.debug_bundle import (  # noqa: E402
    build_bundle,
    detect_telemetry_events,
    merge_timeline,
    validate_run_id,
)
from scripts.aufgabe04.simulation.debug_capture_node import (  # noqa: E402
    nearest_finite_range,
    quaternion_yaw,
)


WRAPPER = ROOT / "scripts" / "aufgabe04" / "simulation" / "run_with_debug_bundle.sh"


class SimulationDebugBundleTest(unittest.TestCase):
    def test_capture_helpers_handle_yaw_and_invalid_scan_ranges(self):
        self.assertAlmostEqual(quaternion_yaw(0.0, 0.0, 0.0, 1.0), 0.0)
        self.assertEqual(nearest_finite_range([float("inf"), float("nan"), 0.0, 0.42]), 0.42)

    def test_run_id_rejects_paths(self):
        with self.assertRaises(ValueError):
            validate_run_id("../escape")

    def test_timeline_merges_wall_and_iso_timestamps(self):
        telemetry = [{"source": "telemetry", "wall_time_sec": 100.0, "pose": {"x_m": 0.0}}]
        semantics = [
            {
                "source": "semantic",
                "timestamp": "1970-01-01T00:01:41+00:00",
                "event": "motion_started",
            }
        ]

        timeline = merge_timeline(telemetry, semantics)

        self.assertEqual([item["source"] for item in timeline], ["telemetry", "semantic"])
        self.assertEqual(timeline[0]["relative_time_sec"], 0.0)
        self.assertEqual(timeline[1]["relative_time_sec"], 1.0)

    def test_derived_events_report_threshold_and_oscillation_once(self):
        telemetry = []
        signs = [1, -1, 1, -1, 1, 1]
        for index, sign in enumerate(signs):
            telemetry.append(
                {
                    "wall_time_sec": 10.0 + index * 0.4,
                    "command": {"linear_x": 0.0, "angular_z": sign * 0.1},
                    "nearest_obstacle_m": 0.17 if index in (0, 1) else 0.5,
                }
            )

        events = detect_telemetry_events(telemetry)

        self.assertEqual(
            [event["event"] for event in events],
            ["obstacle_threshold_crossed", "angular_oscillation_candidate"],
        )

    def test_derived_events_report_sustained_command_without_progress(self):
        telemetry = [
            {
                "wall_time_sec": 20.0 + index * 0.5,
                "pose": {"x_m": 0.001 * index, "y_m": 0.0},
                "command": {"linear_x": 0.08, "angular_z": 0.0},
                "nearest_obstacle_m": 1.0,
            }
            for index in range(7)
        ]

        events = detect_telemetry_events(telemetry)

        self.assertEqual([event["event"] for event in events], ["no_progress_candidate"])

    def test_builder_writes_model_readable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = root / "bundle"
            telemetry = root / "telemetry.jsonl"
            semantic = root / "events.jsonl"
            telemetry.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "wall_time_sec": 100.0,
                                "pose": {"x_m": 0.0, "y_m": 0.0},
                                "ground_truth_pose": {"x_m": 0.0, "y_m": 0.0},
                                "command": {"linear_x": 0.1, "angular_z": 0.0},
                                "nearest_obstacle_m": 1.0,
                            }
                        ),
                        json.dumps(
                            {
                                "wall_time_sec": 101.0,
                                "pose": {"x_m": 0.1, "y_m": 0.0},
                                "ground_truth_pose": {"x_m": 0.1, "y_m": 0.0},
                                "command": {"linear_x": 0.0, "angular_z": 0.0},
                                "nearest_obstacle_m": 0.8,
                            }
                        ),
                    ]
                )
                + "\n"
            )
            semantic.write_text(
                json.dumps(
                    {
                        "timestamp": "1970-01-01T00:01:40.500000+00:00",
                        "event": "motion_started",
                    }
                )
                + "\n"
            )

            manifest = build_bundle(
                bundle_dir=bundle,
                run_id="sim_debug_001",
                telemetry_jsonl=telemetry,
                semantic_jsonl=semantic,
                perception_dirs=[],
                expected_behavior="drive to station B",
                observed_behavior="oscillated near the stand",
                world="test.world",
                command_exit_code=0,
                bag_path=bundle / "rosbag" / "run",
            )

            self.assertEqual(manifest["timeline_record_count"], 3)
            self.assertTrue((bundle / "manifest.json").is_file())
            self.assertTrue((bundle / "timeline.jsonl").is_file())
            self.assertTrue((bundle / "summary.md").is_file())
            self.assertTrue((bundle / "plots" / "trajectory.png").is_file())
            self.assertIn("first divergence", (bundle / "summary.md").read_text())

    def test_wrapper_help_documents_passive_simulation_gate(self):
        result = subprocess.run(
            ["bash", str(WRAPPER), "unused", "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("never publishes motion", result.stdout)
        self.assertIn("/clock", result.stdout)
        self.assertIn("--semantic-log", result.stdout)


if __name__ == "__main__":
    unittest.main()
