import unittest

from scripts.aufgabe04.simulation.generate_gazebo_world import QR_SIZE, qr_matrix
from scripts.aufgabe04.simulation.sim_qr_detector import detect_simulated_station_qr_texts_bgr


class SimQrDetectorTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
