import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.aufgabe04.perception.debug.stand_axis_viewer import build_parser
from scripts.aufgabe04.perception.ros_image_adapter import raw_msg_to_bgr_frame
from scripts.aufgabe04.simulation.prepare_burger_camera_model import main


class SimCameraAdapterTest(unittest.TestCase):
    def test_sim_raw_topic_is_explicit_and_exclusive(self):
        args = build_parser().parse_args(["--sim-raw-image-topic", "/camera/image_raw"])
        self.assertEqual(args.sim_raw_image_topic, "/camera/image_raw")
        self.assertIsNone(args.compressed_image_topic)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                "--sim-raw-image-topic", "/camera/image_raw",
                "--compressed-image-topic", "/camera/image_raw/compressed",
            ])

    def test_rgb8_raw_image_becomes_bgr(self):
        import cv2
        import numpy

        message = SimpleNamespace(
            encoding="rgb8", width=1, height=1, step=3, data=bytes((1, 2, 3))
        )
        frame = raw_msg_to_bgr_frame(message, cv2, numpy)
        self.assertEqual(frame[0, 0].tolist(), [3, 2, 1])

    def test_generated_sdf_has_valid_simulation_fov(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.sdf"
            output = Path(tmp) / "generated.sdf"
            source.write_text("<sdf><camera><horizontal_fov>3.183</horizontal_fov></camera></sdf>")
            from unittest.mock import patch
            with patch("sys.argv", ["prepare", "--source", str(source), "--output", str(output)]):
                self.assertEqual(main(), 0)
            generated = output.read_text()
        self.assertIn("<horizontal_fov>1.3962634</horizontal_fov>", generated)
        self.assertNotIn("3.183", generated)


if __name__ == "__main__":
    unittest.main()
