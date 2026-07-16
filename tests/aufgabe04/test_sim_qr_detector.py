import unittest
from pathlib import Path

from scripts.aufgabe04.simulation.generate_gazebo_world import QR_SIZE, qr_matrix
from scripts.aufgabe04.simulation.sim_qr_detector import (
    detect_simulated_station_qr_bgr,
    detect_simulated_station_qr_texts_bgr,
)


class SimQrDetectorTest(unittest.TestCase):
    def test_sim_decoder_does_not_estimate_orientation(self):
        source = Path("scripts/aufgabe04/simulation/sim_qr_detector.py").read_text()
        viewer = Path("scripts/aufgabe04/perception/debug/stand_axis_viewer.py").read_text()
        self.assertNotIn("solvePnP", source)
        self.assertIn("always comes from the same head-silhouette estimate", viewer)
        self.assertNotIn("sim_qr_shared_square_pnp", viewer)

    def test_decodes_each_simulated_station_id(self):
        import cv2
        import numpy

        quiet_modules = 4
        scale = 12
        for station_id in ("A", "B", "C"):
            modules = numpy.asarray(qr_matrix(station_id), dtype=numpy.uint8)
            image = numpy.pad((1 - modules) * 255, quiet_modules, constant_values=255)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.assertEqual(
                detect_simulated_station_qr_texts_bgr(frame, cv2),
                (station_id,),
            )

    def test_returns_unknown_for_non_qr_image(self):
        import cv2
        import numpy

        frame = numpy.full((160, 160, 3), 255, dtype=numpy.uint8)
        self.assertEqual(detect_simulated_station_qr_texts_bgr(frame, cv2), ())

    def test_target_roi_selects_one_station_and_returns_full_frame_corners(self):
        import cv2
        import numpy

        quiet_modules = 4
        scale = 8
        canvas = numpy.full((300, 700, 3), 255, dtype=numpy.uint8)
        for station_id, x0 in (("A", 40), ("B", 390)):
            modules = numpy.asarray(qr_matrix(station_id), dtype=numpy.uint8)
            image = numpy.pad((1 - modules) * 255, quiet_modules, constant_values=255)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            height, width = image.shape[:2]
            canvas[30 : 30 + height, x0 : x0 + width] = image

        detection = detect_simulated_station_qr_bgr(
            canvas, cv2, roi=(360, 0, 700, 300)
        )
        self.assertIsNotNone(detection)
        self.assertEqual(detection.station_id, "B")
        self.assertGreater(min(point[0] for point in detection.corners_px), 390)

    def test_invalid_target_roi_fails_closed(self):
        import cv2
        import numpy

        frame = numpy.full((100, 100, 3), 255, dtype=numpy.uint8)
        self.assertIsNone(
            detect_simulated_station_qr_bgr(frame, cv2, roi=(50, 50, 20, 20))
        )


if __name__ == "__main__":
    unittest.main()
