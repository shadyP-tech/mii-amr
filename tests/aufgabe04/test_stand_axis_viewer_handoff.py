import unittest

from scripts.aufgabe04.perception.debug.stand_axis_viewer import (
    _validate_runtime_args,
    build_parser,
)


class StandAxisViewerHandoffTest(unittest.TestCase):
    def test_parser_exposes_safe_calibrated_handoff_defaults(self):
        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--scan-topic",
                "/scan",
                "--calibrated-handoff",
            ]
        )

        _validate_runtime_args(args)
        self.assertEqual(args.camera_info_topic, "/camera/camera_info")
        self.assertEqual(args.camera_optical_frame, "camera")
        self.assertEqual(args.scan_frame, "base_scan")
        self.assertEqual(args.handoff_lidar_window_scans, 20)
        self.assertEqual(args.handoff_max_axis_difference_deg, 15.0)

    def test_calibrated_handoff_requires_scan_topic(self):
        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--calibrated-handoff",
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires --scan-topic"):
            _validate_runtime_args(args)

    def test_calibrated_handoff_rejects_simulation_camera(self):
        args = build_parser().parse_args(
            [
                "--sim-raw-image-topic",
                "/camera/image_raw",
                "--scan-topic",
                "/scan",
                "--calibrated-handoff",
            ]
        )

        with self.assertRaisesRegex(ValueError, "real-camera-only"):
            _validate_runtime_args(args)


if __name__ == "__main__":
    unittest.main()
