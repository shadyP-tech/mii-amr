import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.models import BaseFramePoint  # noqa: E402
from scripts.aufgabe04.perception.stand_axis_analysis import (  # noqa: E402
    AxisUsabilityThresholds,
    ScanSample,
    analyze_scan_sample,
    angular_error_rad,
    classify_axis_usability,
    estimate_cluster_axis,
    load_scan_samples,
    write_axis_analysis_csv,
)


def line_points(axis_rad, *, count=9, length_m=0.24, center=(1.0, 0.5)):
    points = []
    cos_axis = math.cos(axis_rad)
    sin_axis = math.sin(axis_rad)
    for index in range(count):
        offset = -length_m / 2.0 + index * length_m / (count - 1)
        x_m = center[0] + offset * cos_axis
        y_m = center[1] + offset * sin_axis
        points.append(
            BaseFramePoint(
                x_m=x_m,
                y_m=y_m,
                bearing_rad=math.atan2(y_m, x_m),
                range_m=math.hypot(x_m, y_m),
                source_index=index,
            )
        )
    return points


def round_points(*, count=16, radius_m=0.08, center=(1.0, 0.5)):
    points = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        x_m = center[0] + radius_m * math.cos(angle)
        y_m = center[1] + radius_m * math.sin(angle)
        points.append(
            BaseFramePoint(
                x_m=x_m,
                y_m=y_m,
                bearing_rad=math.atan2(y_m, x_m),
                range_m=math.hypot(x_m, y_m),
                source_index=index,
            )
        )
    return points


class StandAxisAnalysisTest(unittest.TestCase):
    def assert_axis_close(self, actual, expected, *, delta=0.03):
        self.assertLessEqual(angular_error_rad(actual, expected), delta)

    def test_estimates_horizontal_cluster_axis(self):
        estimate = estimate_cluster_axis(line_points(0.0))

        self.assert_axis_close(estimate, 0.0)
        self.assertGreater(estimate.confidence, 0.80)
        self.assertTrue(classify_axis_usability(estimate).usable)

    def test_estimates_vertical_cluster_axis(self):
        estimate = estimate_cluster_axis(line_points(math.pi / 2.0))

        self.assert_axis_close(estimate, math.pi / 2.0)
        self.assertGreater(estimate.confidence, 0.80)
        self.assertTrue(classify_axis_usability(estimate).usable)

    def test_estimates_diagonal_cluster_axis(self):
        truth = math.radians(45.0)
        estimate = estimate_cluster_axis(line_points(truth))

        self.assert_axis_close(estimate, truth)
        self.assertGreater(estimate.confidence, 0.80)
        self.assertTrue(classify_axis_usability(estimate).usable)

    def test_classifies_round_cluster_as_not_usable(self):
        estimate = estimate_cluster_axis(round_points())

        usability = classify_axis_usability(
            estimate,
            AxisUsabilityThresholds(min_confidence=0.60, min_length_to_width_ratio=2.0),
        )

        self.assertFalse(usability.usable)
        self.assertIn(usability.reason, {"low_confidence", "ambiguous_aspect_ratio"})

    def test_angular_error_treats_axis_as_undirected(self):
        self.assertAlmostEqual(angular_error_rad(0.0, math.pi), 0.0)
        self.assertAlmostEqual(angular_error_rad(math.radians(10), math.radians(170)), math.radians(20))

    def test_analyze_scan_sample_outputs_metrics_row(self):
        sample = ScanSample(
            "synthetic_diagonal",
            tuple(line_points(math.radians(45.0))),
            truth_axis_rad=math.radians(45.0),
        )

        rows = analyze_scan_sample(sample)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sample_id, "synthetic_diagonal")
        self.assertTrue(rows[0].usable)
        self.assertLess(rows[0].angular_error_rad, 0.03)

    def test_plain_json_samples_can_be_loaded_and_written_to_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "samples.json"
            output_path = tmpdir_path / "axis_metrics.csv"
            points = [
                {
                    "x_m": point.x_m,
                    "y_m": point.y_m,
                    "source_index": point.source_index,
                }
                for point in line_points(0.0)
            ]
            input_path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "json_horizontal",
                                "truth_axis_rad": 0.0,
                                "points": points,
                            }
                        ]
                    }
                )
            )

            samples = load_scan_samples(input_path)
            rows = analyze_scan_sample(samples[0])
            write_axis_analysis_csv(output_path, rows)

            with output_path.open(newline="") as file:
                written = list(csv.DictReader(file))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["sample_id"], "json_horizontal")
            self.assertEqual(written[0]["usable"], "True")


if __name__ == "__main__":
    unittest.main()
