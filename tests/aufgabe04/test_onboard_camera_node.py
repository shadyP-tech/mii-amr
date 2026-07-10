import importlib
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.onboard_camera_node import (  # noqa: E402
    OnboardQRScanner,
    build_parser,
)
from scripts.aufgabe04.qr_scanning.scan_processor import (  # noqa: E402
    QRScanProcessor,
    ScanProcessorConfig,
)


class FakeMsg:
    pass


class OnboardCameraNodeImportTest(unittest.TestCase):
    def test_import_does_not_require_ros_runtime_modules(self):
        module = importlib.import_module("scripts.aufgabe04.qr_scanning.onboard_camera_node")

        self.assertTrue(hasattr(module, "build_parser"))


class OnboardCameraNodeCliTest(unittest.TestCase):
    def test_defaults_are_passive_camera_scanner_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.compressed_image_topic, "camera/image_raw/compressed")
        self.assertEqual(args.robot_id, "Robot_Test_01")
        self.assertAlmostEqual(args.min_repeat_sec, 2.0)
        self.assertAlmostEqual(args.max_frame_age_sec, 1.0)

    def test_cli_has_no_motion_or_mission_arguments(self):
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--cmd-vel-topic", option_strings)
        self.assertNotIn("--nav2-goal", option_strings)
        self.assertNotIn("--server-base-url", option_strings)
        self.assertNotIn("--report-scan", option_strings)

    def test_raw_image_topic_argument_is_not_supported(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--ros-image-topic", "/camera/image_raw"])


class OnboardQRScannerCallbackTest(unittest.TestCase):
    def make_scanner(self, *, detector, converter=None, stamp_reader=None, times=None):
        rows = []
        messages = []
        time_values = iter(times or [10.0])

        def time_source():
            return next(time_values)

        def row_appender(path, row):
            rows.append((path, row))

        scanner = OnboardQRScanner(
            source="/camera/image_raw/compressed",
            processor=QRScanProcessor(
                ScanProcessorConfig(
                    robot_id="Robot_Test_01",
                    run_id="run-001",
                    min_repeat_sec=2.0,
                    max_frame_age_sec=1.0,
                )
            ),
            log_path=Path("qr_scans.csv"),
            cv2=object(),
            numpy=object(),
            frame_converter=converter or (lambda msg, cv2, numpy: object()),
            stamp_reader=stamp_reader or (lambda msg: None),
            detector=detector,
            time_source=time_source,
            printer=messages.append,
            row_appender=row_appender,
            once=True,
        )
        return scanner, rows, messages

    def test_callback_logs_and_prints_accepted_scan(self):
        scanner, rows, messages = self.make_scanner(detector=lambda frame, cv2: ("QR_001",))

        scanner.handle_compressed_image(FakeMsg())

        self.assertTrue(scanner.stop_requested)
        self.assertEqual(rows[0][1]["qr_id"], "QR_001")
        self.assertEqual(messages, ["QR scan: qr_id=QR_001 source=/camera/image_raw/compressed"])

    def test_callback_debounces_duplicate_scan(self):
        scanner, rows, messages = self.make_scanner(
            detector=lambda frame, cv2: ("QR_001",),
            times=[10.0, 11.0],
        )
        scanner.once = False

        scanner.handle_compressed_image(FakeMsg())
        scanner.handle_compressed_image(FakeMsg())

        self.assertEqual(len(rows), 1)
        self.assertEqual(messages, ["QR scan: qr_id=QR_001 source=/camera/image_raw/compressed"])

    def test_callback_logs_rejected_invalid_scan(self):
        scanner, rows, messages = self.make_scanner(detector=lambda frame, cv2: ("route: A -> B",))

        scanner.handle_compressed_image(FakeMsg())

        self.assertFalse(scanner.stop_requested)
        self.assertEqual(rows[0][1]["status"], "rejected")
        self.assertEqual(messages, [])

    def test_callback_logs_stale_scan_as_rejected(self):
        scanner, rows, messages = self.make_scanner(
            detector=lambda frame, cv2: ("QR_001",),
            stamp_reader=lambda msg: 8.0,
            times=[10.0],
        )

        scanner.handle_compressed_image(FakeMsg())

        self.assertFalse(scanner.stop_requested)
        self.assertEqual(rows[0][1]["reason"], "stale_frame")
        self.assertEqual(messages, [])

    def test_callback_throttles_conversion_warnings(self):
        def converter(msg, cv2, numpy):
            raise ValueError("bad frame")

        scanner, rows, messages = self.make_scanner(
            detector=lambda frame, cv2: ("QR_001",),
            converter=converter,
            times=[10.0, 10.5],
        )

        scanner.handle_compressed_image(FakeMsg())
        scanner.handle_compressed_image(FakeMsg())

        self.assertEqual(rows, [])
        self.assertEqual(messages, ["WARNING: bad frame"])


if __name__ == "__main__":
    unittest.main()
