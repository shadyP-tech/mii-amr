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
    compressed_msg_stamp_sec,
    compressed_msg_to_bgr_frame,
)


try:
    import numpy
except ImportError:
    numpy = None


@dataclass
class FakeCompressedImage:
    format: str
    data: bytes


@dataclass
class FakeStamp:
    sec: int
    nanosec: int


@dataclass
class FakeHeader:
    stamp: FakeStamp


@dataclass
class FakeStampedCompressedImage(FakeCompressedImage):
    header: FakeHeader


try:
    import cv2
except ImportError:
    cv2 = None


@unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for image conversion tests")
class CompressedImageMessageConversionTest(unittest.TestCase):
    def test_decodes_jpeg_compressed_image_to_bgr(self):
        source = numpy.zeros((2, 2, 3), dtype=numpy.uint8)
        source[0, 0] = [0, 255, 0]
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)
        msg = FakeCompressedImage(format="jpeg", data=encoded.tobytes())

        frame = compressed_msg_to_bgr_frame(msg, cv2, numpy)

        self.assertEqual(frame.shape, (2, 2, 3))

    def test_rejects_empty_compressed_image(self):
        msg = FakeCompressedImage(format="jpeg", data=b"")

        with self.assertRaisesRegex(ValueError, "empty"):
            compressed_msg_to_bgr_frame(msg, cv2, numpy)

    def test_rejects_invalid_compressed_image_bytes(self):
        msg = FakeCompressedImage(format="jpeg", data=b"not a jpeg")

        with self.assertRaisesRegex(ValueError, "failed to decode"):
            compressed_msg_to_bgr_frame(msg, cv2, numpy)


class ImageMessageStampTest(unittest.TestCase):
    def test_extracts_header_stamp_seconds(self):
        msg = FakeStampedCompressedImage(
            format="jpeg",
            data=bytes([0, 0, 0]),
            header=FakeHeader(FakeStamp(sec=12, nanosec=345_000_000)),
        )

        self.assertAlmostEqual(compressed_msg_stamp_sec(msg), 12.345)

    def test_missing_header_stamp_returns_none(self):
        msg = FakeCompressedImage(
            format="jpeg",
            data=bytes([0, 0, 0]),
        )

        self.assertIsNone(compressed_msg_stamp_sec(msg))


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


if __name__ == "__main__":
    unittest.main()
