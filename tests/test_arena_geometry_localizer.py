import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import analyze_arena_geometry_from_bag as bag_analyzer  # noqa: E402
import arena_geometry_localizer as arena  # noqa: E402


def rotate_point(point, yaw_deg):
    yaw = math.radians(yaw_deg)
    x, y = point
    return (
        math.cos(yaw) * x - math.sin(yaw) * y,
        math.sin(yaw) * x + math.cos(yaw) * y,
    )


def transform_points(points, yaw_deg=0.0, lateral_offset_m=0.0):
    transformed = []
    for x, y in points:
        transformed.append(rotate_point((x, y - lateral_offset_m), yaw_deg))
    return transformed


def rectangular_points(
    include_clean=False,
    include_heater=False,
    yaw_deg=0.0,
    lateral_offset_m=0.0,
):
    half_length = 3.90 / 2.0
    half_width = 1.898 / 2.0
    points = []
    for index in range(61):
        x = -1.50 + index * 0.05
        points.append((x, -half_width))
        points.append((x, half_width))
    if include_clean:
        for index in range(39):
            y = -0.90 + index * 0.05
            points.append((-half_length, y))
    if include_heater:
        for index in range(39):
            y = -0.90 + index * 0.05
            points.append((half_length, y))
        for y_center in (-0.35, 0.35):
            for offset_index in range(10):
                y = y_center - 0.12 + offset_index * 0.025
                points.append((half_length - 0.16, y))
    return transform_points(points, yaw_deg=yaw_deg, lateral_offset_m=lateral_offset_m)


class ArenaGeometryLocalizerTest(unittest.TestCase):
    def test_long_walls_only_are_rejected_as_non_unique(self):
        result = arena.analyze_points(rectangular_points())

        self.assertFalse(result.success)
        self.assertTrue(result.long_wall_fit.ok)
        self.assertFalse(result.pose_unique)
        self.assertFalse(result.yaw_ambiguity_resolved)
        self.assertEqual(result.failure_reason, "pose_not_unique")
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_UNKNOWN)

    def test_clean_wall_resolves_pose_prior(self):
        result = arena.analyze_points(rectangular_points(include_clean=True))

        self.assertTrue(result.success)
        self.assertTrue(result.pose_unique)
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_CLEAN)
        self.assertEqual(result.short_wall_classification.observed_axis_side, "axis_negative")
        self.assertIsNotNone(result.estimated_pose_prior)
        self.assertAlmostEqual(result.estimated_pose_prior.x, 0.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.y, 0.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.yaw_deg, 0.0, delta=2.0)
        self.assertIsNotNone(result.estimated_covariance)

    def test_heater_wall_resolves_pose_prior(self):
        result = arena.analyze_points(rectangular_points(include_heater=True))

        self.assertTrue(result.success)
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_HEATER)
        self.assertEqual(result.short_wall_classification.observed_axis_side, "axis_positive")
        self.assertGreater(result.short_wall_classification.heater_feature_score, 0.75)
        self.assertAlmostEqual(result.estimated_pose_prior.x, 0.0, delta=0.12)

    def test_rotated_and_laterally_offset_scan_estimates_y_and_yaw(self):
        result = arena.analyze_points(
            rectangular_points(include_clean=True, yaw_deg=12.0, lateral_offset_m=0.18)
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.estimated_pose_prior.y, 0.18, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.yaw_deg, -12.0, delta=2.0)

    def test_map_frame_calibration_is_applied_to_pose_prior(self):
        config = arena.ArenaGeometryConfig(
            map_center_x=1.0,
            map_center_y=2.0,
            map_yaw_deg=90.0,
        )
        result = arena.analyze_points(rectangular_points(include_clean=True), config)

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.estimated_pose_prior.x, 1.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.y, 2.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.yaw_deg, 90.0, delta=2.0)

    def test_json_output_contains_required_commit_a_keys(self):
        result = arena.analyze_points(rectangular_points(include_clean=True))
        data = result.to_dict()

        for key in [
            "success",
            "failure_reason",
            "pose_unique",
            "yaw_ambiguity_resolved",
            "estimated_pose_prior",
            "estimated_covariance",
            "long_wall_fit",
            "short_wall_classification",
            "diagnostics",
        ]:
            self.assertIn(key, data)

    def test_new_commit_a_files_do_not_contain_live_motion_or_initialpose_code(self):
        for relative in [
            "scripts/aufgabe03/arena_geometry_localizer.py",
            "scripts/aufgabe03/analyze_arena_geometry_from_bag.py",
        ]:
            text = (ROOT / relative).read_text()
            self.assertNotIn("cmd_vel", text)
            self.assertNotIn("initialpose", text)
            self.assertNotIn("PoseWithCovariance", text)

    def test_analyzer_accepts_json_samples_for_non_ros_debugging(self):
        sample_data = {
            "scan_samples": [
                {
                    "ranges": [1.0, float("inf"), 2.0],
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2.0,
                    "range_min": 0.05,
                    "range_max": 3.0,
                    "odom_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.json"
            path.write_text(json.dumps(sample_data))
            samples = arena.load_scan_samples_json(path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(len(arena.finite_scan_points(samples[0])), 2)

    def test_analyzer_cli_writes_required_json_from_json_input(self):
        sample_data = {
            "scan_samples": [
                {
                    "ranges": [1.0, 2.0, 1.5],
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2.0,
                    "range_min": 0.05,
                    "range_max": 3.0,
                    "odom_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "samples.json"
            output_path = Path(tmpdir) / "diagnostics.json"
            input_path.write_text(json.dumps(sample_data))

            with contextlib.redirect_stdout(io.StringIO()):
                result = bag_analyzer.main(
                    [
                        "--input-json",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )
            written = json.loads(output_path.read_text())

        self.assertEqual(result, 0)
        self.assertIn("success", written)
        self.assertIn("long_wall_fit", written)
        self.assertEqual(written["source"]["type"], "json")

    def test_bag_cli_requires_exactly_one_input_source(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bag_analyzer.parse_args(["--output", "out.json"])
        args = bag_analyzer.parse_args(
            ["--input-json", "samples.json", "--output", "out.json"]
        )
        self.assertEqual(args.input_json, Path("samples.json"))


if __name__ == "__main__":
    unittest.main()
