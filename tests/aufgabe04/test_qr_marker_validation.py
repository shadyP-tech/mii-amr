"""Persistent front evidence requires more than a tentative native QR quad."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional vision environment
    cv2 = np = None

from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_marker_validation import validate_qr_marker
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import QrQuadDetection
from scripts.aufgabe04.real_robot.observer.front_observation import front_observation_decision


FIXTURES = Path(__file__).parent / "fixtures" / "qr_marker_validation"


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class QrMarkerValidationTest(unittest.TestCase):
    def _recorded(self, name):
        sample = json.loads((FIXTURES / "manifest.json").read_text())["samples"][name]
        data = (FIXTURES / sample["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), sample["source_compressed_sha256"])
        raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame = rectify_bgr_frame(raw, SimpleNamespace(**sample["recorded_camera_info"]), cv2, np)
        x0, y0, x1, y1 = (sample["roi"][key] for key in ("x0", "y0", "x1", "y1"))
        quad = sample["quad"]
        detection = QrQuadDetection(
            tuple(ImagePoint(**point) for point in quad["corners"]),
            quad["scale"], quad["text"], quad["detector"],
        )
        return frame[y0:y1, x0:x1], detection

    def _synthetic(self, *, missing=(), rotation=0):
        # Finder patterns alone provide marker evidence, never decoded identity.
        modules = np.ones((21, 21), dtype=np.uint8) * 255
        for index, (x, y) in enumerate(((0, 0), (14, 0), (0, 14))):
            if index in missing:
                continue
            modules[y:y + 7, x:x + 7] = 0
            modules[y + 1:y + 6, x + 1:x + 6] = 255
            modules[y + 2:y + 5, x + 2:x + 5] = 0
        modules = np.rot90(modules, rotation)
        frame = np.pad(np.repeat(np.repeat(modules, 6, axis=0), 6, axis=1), 12, constant_values=255)
        return frame, QrQuadDetection(
            tuple(ImagePoint(u, v) for u, v in ((12, 12), (137, 12), (137, 137), (12, 137))),
            1.0,
        )

    def test_recorded_backside_false_quad_is_not_verified(self):
        frame, detection = self._recorded("backside_000022")
        self.assertEqual(detection.scale, 2.0)
        result = validate_qr_marker(cv2, frame, detection)
        self.assertFalse(result.verified)
        self.assertEqual(result.finder_count, 0)
        # Detection is retained as a conservative veto in this frame.
        self.assertEqual(len(detection.corners), 4)
        self.assertIsNone(detection.text)

    def test_recorded_front_finders_pass_without_identity_or_pose(self):
        frame, detection = self._recorded("front_000045")
        self.assertIsNone(detection.text)
        result = validate_qr_marker(cv2, frame, detection)
        self.assertTrue(result.verified)
        self.assertEqual(result.reason, "three_current_pixel_qr_finders")
        self.assertEqual(result.finder_count, 3)
        self.assertGreater(result.minimum_finder_score, 0.90)

    def test_decoded_identity_is_never_demoted_by_marker_geometry(self):
        frame, detection = self._recorded("backside_000022")
        result = validate_qr_marker(cv2, frame, replace(detection, text="QR_003"))
        self.assertTrue(result.verified)
        self.assertEqual(result.reason, "decoded_qr_identity")

    def test_missing_detection_has_no_marker_evidence(self):
        self.assertFalse(validate_qr_marker(cv2, None, None).verified)

    def test_rotated_finder_layouts_are_verified(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                frame, detection = self._synthetic(rotation=rotation)
                self.assertTrue(validate_qr_marker(cv2, frame, detection).verified)

    def test_partial_marker_cannot_set_persistent_front(self):
        for missing in ((0,), (1,), (2,), (0, 1), (0, 1, 2)):
            with self.subTest(missing=missing):
                frame, detection = self._synthetic(missing=missing)
                result = validate_qr_marker(cv2, frame, detection)
                self.assertFalse(result.verified)
                self.assertLess(result.finder_count, 3)

    def test_recorded_front_with_right_finder_removed_is_unverified(self):
        frame, detection = self._recorded("front_000045")
        frame = frame.copy()
        frame[:, 430:] = 255
        self.assertFalse(validate_qr_marker(cv2, frame, detection).verified)

    def test_recorded_false_quad_cannot_persist_into_next_backside_frame(self):
        frame, detection = self._recorded("backside_000022")
        marker = validate_qr_marker(cv2, frame, detection)
        current = front_observation_decision(
            qr_texts=(), qr_marker_detected=True,
            qr_marker_verified=marker.verified,
            estimate_source="model_backside_current_frame",
            marker_seen_in_stationary_epoch=False,
        )
        self.assertTrue(current.withhold_backside_axis)
        self.assertFalse(current.marker_seen_in_stationary_epoch)
        following = front_observation_decision(
            qr_texts=(), qr_marker_detected=False, qr_marker_verified=False,
            estimate_source="model_backside_current_frame",
            marker_seen_in_stationary_epoch=current.marker_seen_in_stationary_epoch,
        )
        self.assertFalse(following.withhold_backside_axis)

    def test_recorded_front_marker_persists_even_without_decoded_identity(self):
        frame, detection = self._recorded("front_000045")
        marker = validate_qr_marker(cv2, frame, detection)
        current = front_observation_decision(
            qr_texts=(), qr_marker_detected=True,
            qr_marker_verified=marker.verified,
            estimate_source="model_current_frame_refined",
            marker_seen_in_stationary_epoch=False,
        )
        self.assertTrue(current.marker_observed_now)
        following = front_observation_decision(
            qr_texts=(), qr_marker_detected=False, qr_marker_verified=False,
            estimate_source="model_backside_current_frame",
            marker_seen_in_stationary_epoch=current.marker_seen_in_stationary_epoch,
        )
        self.assertTrue(following.withhold_backside_axis)

    def test_subresolution_or_clipped_marker_is_not_verified(self):
        frame, detection = self._synthetic()
        small = cv2.resize(frame, (12, 12), interpolation=cv2.INTER_AREA)
        proposal = replace(detection, corners=tuple(
            ImagePoint(p.u_px * 12 / frame.shape[1], p.v_px * 12 / frame.shape[0])
            for p in detection.corners
        ))
        self.assertFalse(validate_qr_marker(cv2, small, proposal).verified)
        self.assertFalse(validate_qr_marker(cv2, frame[:, :100], detection).verified)

    def test_uniform_patch_does_not_pass_as_finders(self):
        frame, detection = self._synthetic()
        for intensity in (0, 100, 255):
            with self.subTest(intensity=intensity):
                self.assertFalse(validate_qr_marker(cv2, np.full_like(frame, intensity), detection).verified)

    def test_malformed_and_degenerate_quadrilaterals_fail_closed(self):
        frame, detection = self._synthetic()
        cases = (
            (),
            detection.corners[:3],
            (ImagePoint(float("nan"), 20),) + detection.corners[1:],
            (ImagePoint(float("inf"), 20),) + detection.corners[1:],
            (ImagePoint(-1, 20),) + detection.corners[1:],
            (ImagePoint(200, 20),) + detection.corners[1:],
            (ImagePoint(20, 20),) * 4,
            tuple(ImagePoint(u, v) for u, v in ((20, 20), (21, 20), (21, 21), (20, 21))),
            (detection.corners[0], detection.corners[2], detection.corners[1], detection.corners[3]),
        )
        for corners in cases:
            with self.subTest(corners=corners):
                result = validate_qr_marker(cv2, frame, replace(detection, corners=corners))
                self.assertFalse(result.verified)
                self.assertEqual(result.reason, "invalid_marker_quadrilateral")

    def test_empty_and_invalid_images_fail_closed(self):
        _, detection = self._synthetic()
        for frame in (None, np.zeros((0, 0), dtype=np.uint8), np.zeros((10,), dtype=np.uint8)):
            with self.subTest(frame=frame):
                result = validate_qr_marker(cv2, frame, detection)
                self.assertFalse(result.verified)
                self.assertEqual(result.reason, "invalid_marker_image")

    def test_noise_and_plain_rectangles_are_not_marker_evidence(self):
        frame, detection = self._synthetic()
        noise = np.random.default_rng(20260910).integers(0, 256, frame.shape, dtype=np.uint8)
        rectangle = np.full_like(frame, 255)
        cv2.rectangle(rectangle, (12, 12), (137, 137), 0, 6)
        cv2.putText(rectangle, "STAND", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        for image in (noise, rectangle):
            self.assertFalse(validate_qr_marker(cv2, image, detection).verified)

    def test_perspective_and_blur_keep_current_finder_evidence(self):
        frame, detection = self._synthetic()
        destination = np.float32(((25, 20), (165, 35), (154, 176), (16, 157)))
        transform = cv2.getPerspectiveTransform(
            np.float32([(p.u_px, p.v_px) for p in detection.corners]), destination,
        )
        distorted = cv2.warpPerspective(frame, transform, (190, 190), borderValue=255)
        distorted = cv2.GaussianBlur(distorted, (3, 3), 0.6)
        proposal = replace(detection, corners=tuple(ImagePoint(float(u), float(v)) for u, v in destination))
        self.assertTrue(validate_qr_marker(cv2, distorted, proposal).verified)


if __name__ == "__main__":
    unittest.main()
