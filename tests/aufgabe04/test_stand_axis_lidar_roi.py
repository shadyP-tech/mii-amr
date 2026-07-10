import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.stand_axis_viewer import build_parser  # noqa: E402
from scripts.aufgabe04.perception.stand_axis_lidar_roi import (  # noqa: E402
    ROI_OBSERVATION_SCHEMA_VERSION,
    ROI_OBSERVER_VERSION,
    PlainLaserScan,
    StandAxisLidarRoiObservation,
    camera_bearing_rad,
    image_center_x_to_bearing_rad,
    load_observation_jsonl,
    median_range_in_scan_cone,
    write_observation_jsonl,
)


class StandAxisLidarRoiTest(unittest.TestCase):
    def make_scan(self) -> PlainLaserScan:
        return PlainLaserScan(
            ranges=(1.20, float("nan"), 0.80, 0.70, 1.30, 4.00),
            angle_min=math.radians(-25.0),
            angle_increment=math.radians(10.0),
            range_min=0.10,
            range_max=3.50,
            scan_frame_id="base_scan",
            scan_stamp_sec=10.0,
            receipt_sec=20.0,
        )

    def test_image_center_maps_to_zero_bearing(self):
        bearing = image_center_x_to_bearing_rad(320.0, camera_fx_px=600.0, camera_cx_px=320.0)

        self.assertAlmostEqual(bearing, 0.0)

    def test_left_and_right_image_centers_have_expected_signs(self):
        left = image_center_x_to_bearing_rad(220.0, camera_fx_px=500.0, camera_cx_px=320.0)
        right = image_center_x_to_bearing_rad(420.0, camera_fx_px=500.0, camera_cx_px=320.0)

        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(abs(left), abs(right))

    def test_camera_to_lidar_yaw_offset_is_added(self):
        bearing = image_center_x_to_bearing_rad(
            320.0,
            camera_fx_px=600.0,
            camera_cx_px=320.0,
            camera_to_lidar_yaw_offset_rad=0.12,
        )

        self.assertAlmostEqual(bearing, 0.12)

    def test_invalid_intrinsics_are_rejected(self):
        with self.assertRaises(ValueError):
            camera_bearing_rad(320.0, camera_fx_px=0.0, camera_cx_px=320.0)

    def test_scan_cone_uses_dynamic_bearing_and_ignores_invalid_ranges(self):
        query = median_range_in_scan_cone(
            self.make_scan(),
            bearing_rad=math.radians(0.0),
            cone_half_angle_rad=math.radians(11.0),
            now_sec=20.1,
            max_scan_age_sec=1.0,
        )

        self.assertAlmostEqual(query.distance_m, 0.75)
        self.assertEqual(query.selected_sample_count, 2)
        self.assertEqual(query.rejection_reason, "")
        self.assertEqual(query.scan_frame_id, "base_scan")
        self.assertAlmostEqual(query.scan_age_sec, 0.1)

    def test_scan_cone_fails_closed_when_no_samples_are_selected(self):
        query = median_range_in_scan_cone(
            self.make_scan(),
            bearing_rad=math.radians(90.0),
            cone_half_angle_rad=math.radians(2.0),
        )

        self.assertIsNone(query.distance_m)
        self.assertEqual(query.selected_sample_count, 0)
        self.assertEqual(query.rejection_reason, "too_few_valid_samples")

    def test_scan_cone_fails_closed_when_scan_is_stale(self):
        query = median_range_in_scan_cone(
            self.make_scan(),
            bearing_rad=0.0,
            cone_half_angle_rad=math.radians(10.0),
            now_sec=25.0,
            max_scan_age_sec=0.5,
        )

        self.assertIsNone(query.distance_m)
        self.assertEqual(query.selected_sample_count, 0)
        self.assertEqual(query.rejection_reason, "stale_scan")

    def test_observation_jsonl_round_trip_allows_missing_scan_distance(self):
        observation = StandAxisLidarRoiObservation(
            schema_version=ROI_OBSERVATION_SCHEMA_VERSION,
            observer_version=ROI_OBSERVER_VERSION,
            observed_at_sec=123.0,
            image_topic="/camera/image_raw/compressed",
            image_stamp_sec=122.9,
            scan_topic="/scan",
            scan_frame_id="base_scan",
            scan_stamp_sec=122.8,
            scan_age_sec=0.2,
            rect_center_x_px=300.0,
            camera_fx_px=610.0,
            camera_cx_px=320.0,
            camera_bearing_rad=-0.032775,
            lidar_bearing_rad=-0.022775,
            bearing_source="image-center",
            cone_half_angle_rad=math.radians(5.0),
            selected_sample_count=0,
            used_distance_m=None,
            fallback_source="image_center",
            rejection_reason="too_few_valid_samples",
            estimate_source="edge_plain_face_stem_anchor",
            estimate_usable=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.jsonl"
            write_observation_jsonl(path, (observation,))
            loaded = load_observation_jsonl(path)

        self.assertEqual(loaded, (observation,))

    def test_viewer_parser_keeps_fixed_bearing_default_and_opt_in_mode(self):
        parser = build_parser()

        fixed_args = parser.parse_args(["--compressed-image-topic", "/camera/image_raw/compressed"])
        dynamic_args = parser.parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--lidar-bearing-source",
                "image-center",
            ]
        )

        self.assertEqual(fixed_args.lidar_bearing_source, "fixed")
        self.assertEqual(dynamic_args.lidar_bearing_source, "image-center")


if __name__ == "__main__":
    unittest.main()
