import sys
import unittest
from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.color_mask_viewer import (  # noqa: E402
    build_parser,
    image_msg_to_bgr_frame,
)


try:
    import numpy
except ImportError:
    numpy = None


@dataclass
class FakeImage:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes


@unittest.skipIf(numpy is None, "numpy is required for image conversion tests")
class ImageMessageConversionTest(unittest.TestCase):
    def test_converts_bgr8_image_without_channel_swap(self):
        msg = FakeImage(
            height=1,
            width=2,
            encoding="bgr8",
            step=6,
            data=bytes([1, 2, 3, 4, 5, 6]),
        )

        frame = image_msg_to_bgr_frame(msg, numpy)

        self.assertEqual(frame.shape, (1, 2, 3))
        self.assertEqual(frame[0, 0].tolist(), [1, 2, 3])
        self.assertEqual(frame[0, 1].tolist(), [4, 5, 6])

    def test_converts_rgb8_image_to_bgr(self):
        msg = FakeImage(
            height=1,
            width=1,
            encoding="RGB888",
            step=3,
            data=bytes([10, 20, 30]),
        )

        frame = image_msg_to_bgr_frame(msg, numpy)

        self.assertEqual(frame[0, 0].tolist(), [30, 20, 10])

    def test_ignores_row_padding(self):
        msg = FakeImage(
            height=2,
            width=1,
            encoding="BGR888",
            step=5,
            data=bytes([1, 2, 3, 99, 99, 4, 5, 6, 88, 88]),
        )

        frame = image_msg_to_bgr_frame(msg, numpy)

        self.assertEqual(frame.shape, (2, 1, 3))
        self.assertEqual(frame[0, 0].tolist(), [1, 2, 3])
        self.assertEqual(frame[1, 0].tolist(), [4, 5, 6])

    def test_rejects_unsupported_encoding(self):
        msg = FakeImage(
            height=1,
            width=1,
            encoding="mono8",
            step=1,
            data=bytes([0]),
        )

        with self.assertRaisesRegex(ValueError, "unsupported"):
            image_msg_to_bgr_frame(msg, numpy)


class ColorMaskViewerCliTest(unittest.TestCase):
    def test_ros_image_topic_is_required(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

    def test_ros_image_topic_parses(self):
        parser = build_parser()

        args = parser.parse_args(["--ros-image-topic", "/image_raw", "--color", "green"])

        self.assertEqual(args.ros_image_topic, "/image_raw")

    def test_opencv_camera_and_video_sources_are_not_supported(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--camera-index", "0", "--ros-image-topic", "/image_raw"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--video", "sample.mp4", "--ros-image-topic", "/image_raw"])


if __name__ == "__main__":
    unittest.main()
