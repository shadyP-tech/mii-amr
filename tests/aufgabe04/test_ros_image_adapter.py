import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.ros_image_adapter import (  # noqa: E402
    compressed_msg_stamp_sec,
    compressed_msg_to_bgr_frame,
)


try:
    import cv2
    import numpy
except ImportError:
    cv2 = None
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
    stamp: object


@dataclass
class FakeStampedCompressedImage(FakeCompressedImage):
    header: FakeHeader


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

    def test_malformed_header_stamp_returns_none(self):
        msg = FakeStampedCompressedImage(
            format="jpeg",
            data=bytes([0, 0, 0]),
            header=FakeHeader(object()),
        )

        self.assertIsNone(compressed_msg_stamp_sec(msg))


if __name__ == "__main__":
    unittest.main()
