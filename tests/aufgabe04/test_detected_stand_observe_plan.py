import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.run_detected_stand_observe_plan import (
    _artifact_paths,
    _default_output_dir,
    _ensure_new_artifacts,
    _observer_args,
    _planner_argv,
    build_parser,
)


class DetectedStandObservePlanTest(unittest.TestCase):
    def test_defaults_are_real_robot_wait_and_one_pipeline_root(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.readiness_timeout_sec, 30.0)
        self.assertEqual(args.observation_duration_sec, 8.0)
        self.assertEqual(args.nomotion_refresh_sec, 2.0)
        self.assertEqual(args.localization_source, "amcl")
        self.assertEqual(args.order, "confidence")
        self.assertEqual(
            _default_output_dir(0.0),
            Path("results/aufgabe04/real_explore_19700101_000000"),
        )

    def test_observer_and_planner_share_runtime_contract(self):
        args = build_parser().parse_args(
            [
                "--map",
                "arena.yaml",
                "--output-dir",
                "run",
                "--namespace",
                "robot1",
                "--scan-topic",
                "lidar_scan",
                "--amcl-topic",
                "localization_pose",
            ]
        )
        paths = _artifact_paths(args.output_dir)
        observer = _observer_args(args, paths)
        planner = _planner_argv(args, paths, Pose2D(0.2, -0.1, 0.3))

        self.assertEqual(observer.namespace, "robot1")
        self.assertEqual(observer.scan_topic, "lidar_scan")
        self.assertEqual(observer.amcl_topic, "localization_pose")
        self.assertEqual(observer.output_jsonl, paths["observations"])
        self.assertIn("--start-x", planner)
        self.assertNotIn("--start-from-tf", planner)
        self.assertEqual(planner[planner.index("--start-x") + 1], "0.2")
        self.assertEqual(planner[planner.index("--max-stands") + 1], "1")

    def test_existing_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _artifact_paths(Path(tmpdir))
            paths["observations"].write_text("existing\n")

            with self.assertRaisesRegex(ValueError, "refusing"):
                _ensure_new_artifacts(paths)


if __name__ == "__main__":
    unittest.main()
