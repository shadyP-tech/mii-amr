import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.color_mask_viewer import build_parser  # noqa: E402


class ColorMaskViewerCliTest(unittest.TestCase):
    def test_ros_image_topic_is_required(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

    def test_compressed_image_topic_parses(self):
        parser = build_parser()

        args = parser.parse_args(["--compressed-image-topic", "/image_raw/compressed", "--color", "green"])

        self.assertEqual(args.compressed_image_topic, "/image_raw/compressed")

    def test_raw_image_topic_is_not_supported(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--ros-image-topic", "/image_raw"])

    def test_max_frame_age_default_and_override(self):
        parser = build_parser()

        default_args = parser.parse_args(["--compressed-image-topic", "/image_raw/compressed"])
        disabled_args = parser.parse_args(
            ["--compressed-image-topic", "/image_raw/compressed", "--max-frame-age-sec", "0"]
        )

        self.assertAlmostEqual(default_args.max_frame_age_sec, 0.25)
        self.assertAlmostEqual(disabled_args.max_frame_age_sec, 0.0)

    def test_opencv_camera_and_video_sources_are_not_supported(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--camera-index", "0", "--ros-image-topic", "/image_raw"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--video", "sample.mp4", "--compressed-image-topic", "/image_raw/compressed"])

    def test_cli_has_no_motion_related_arguments(self):
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--cmd-vel-topic", option_strings)
        self.assertNotIn("--run", option_strings)
        self.assertNotIn("--nav2-goal", option_strings)


if __name__ == "__main__":
    unittest.main()
