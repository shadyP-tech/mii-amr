import unittest

try:
    import cv2
    import numpy
except ImportError:
    cv2 = numpy = None

from scripts.aufgabe04.qr_scanning.isolated_qr_identity import decode_isolated_native_quad


@unittest.skipIf(cv2 is None, "OpenCV unavailable")
class IsolatedQrIdentityTest(unittest.TestCase):
    def setUp(self):
        self.frame = numpy.zeros((120, 120, 3), dtype=numpy.uint8)
        self.frame[50:91, 40:81] = (10, 20, 30)
        self.quad = ((40., 50.), (80., 50.), (80., 90.), (40., 90.))

    def _cv2_with_isolated_decoder(self, texts):
        test = self

        class Detector:
            def detectAndDecode(self, isolated):
                # The decoder must see only the chosen symbol, plus a new
                # white quiet border, instead of the surrounding scene.
                test.assertEqual(isolated.shape, (256, 256, 3))
                test.assertEqual(tuple(isolated[128, 128]), (10, 20, 30))
                test.assertEqual(tuple(isolated[0, 0]), (255, 255, 255))
                return texts, (numpy.array(((0, 0), (255, 0), (255, 255), (0, 255))),)

        class Cv2:
            wechat_qrcode_WeChatQRCode = Detector

            def __getattr__(self, name):
                return getattr(cv2, name)

        return Cv2()

    def test_isolated_identity_binds_native_geometry_not_wechat_crop_bounds(self):
        result = decode_isolated_native_quad(
            self.frame, numpy.array([self.quad]), self._cv2_with_isolated_decoder(("Start",)),
            image_shape=(60, 60, 3), scale=2., border_px=10,
        )
        self.assertEqual(result.text, "Start")
        self.assertEqual(result.detector, "opencv_quad_wechat_rectified")
        self.assertEqual(result.corners, ((15., 20.), (35., 20.), (35., 40.), (15., 40.)))

    def test_multiple_native_quads_or_isolated_payloads_are_rejected(self):
        for points, texts in (
            (numpy.array([self.quad, self.quad]), ("Start",)),
            (numpy.array([self.quad]), ("Start", "Start")),
            (numpy.array([self.quad]), ("",)),
        ):
            with self.subTest(texts=texts, shape=points.shape):
                self.assertIsNone(decode_isolated_native_quad(
                    self.frame, points, self._cv2_with_isolated_decoder(texts),
                    image_shape=self.frame.shape, scale=1., border_px=0,
                ))
