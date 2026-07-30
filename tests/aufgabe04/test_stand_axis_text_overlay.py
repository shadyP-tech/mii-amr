from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import cv2 as real_cv2
    import numpy
except ImportError:
    real_cv2 = None
    numpy = None

from scripts.aufgabe04.perception.debug.stand_axis_viewer import (
    _roi_label_origin,
)
from scripts.aufgabe04.perception.debug.text_overlay import OverlayTextCursor
from scripts.aufgabe04.perception.stand_axis_handoff.models import (
    AxisHandoffDecision,
    CameraAxisEstimate,
    LidarAxisEstimate,
)
from scripts.aufgabe04.perception.stand_axis_handoff.overlay import (
    annotate_axis_handoff,
)
from scripts.aufgabe04.simulation.sim_head_roi import HeadRoi


class _FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        self.text_calls = []

    @staticmethod
    def getTextSize(text, _font_face, _font_scale, _thickness):
        return (max(1, len(text) * 7), 11), 3

    def putText(
        self,
        _frame,
        text,
        origin,
        _font_face,
        _font_scale,
        _color,
        _thickness,
    ):
        self.text_calls.append((text, origin))


class StandAxisTextOverlayTest(unittest.TestCase):
    def test_cursor_wraps_and_stacks_measured_rows_without_overlap(self):
        cv2 = _FakeCv2()
        frame = SimpleNamespace(shape=(180, 220, 3))
        cursor = OverlayTextCursor(
            left_px=12,
            top_px=8,
            right_margin_px=48,
            row_gap_px=4,
        )

        bounds = cursor.draw(
            cv2,
            frame,
            "camera axis rotation has a deliberately long diagnostic value",
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.52,
            color=(0, 255, 0),
            thickness=2,
        )

        self.assertGreater(len(bounds), 1)
        for previous, current in zip(bounds, bounds[1:]):
            self.assertGreaterEqual(current.top_px, previous.bottom_px + 4)
        self.assertTrue(all(bound.right_px <= 172 for bound in bounds))

    def test_handoff_rows_continue_below_existing_viewer_rows(self):
        cv2 = _FakeCv2()
        frame = SimpleNamespace(shape=(600, 800, 3))
        cursor = OverlayTextCursor()
        cursor.draw(
            cv2,
            frame,
            "camera axis status",
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.62,
            color=(0, 255, 0),
            thickness=2,
        )
        viewer_bottom = cursor.bottom_px
        decision = AxisHandoffDecision(
            status="camera_collecting",
            accepted=False,
            reason="camera_consensus_pending",
            lidar=LidarAxisEstimate(
                usable=True,
                reason="axis_estimated",
                angle_rad=0.2,
                sample_count=18,
                linearity=0.91,
            ),
            camera=CameraAxisEstimate(
                usable=False,
                reason="camera_consensus_pending",
                sample_count=1,
            ),
        )

        returned = annotate_axis_handoff(
            cv2,
            frame,
            decision,
            text_cursor=cursor,
        )

        handoff_origins = [
            origin
            for text, origin in cv2.text_calls
            if text.startswith(("handoff=", "lidar=", "camera=", "axis_delta="))
        ]
        self.assertIs(returned, cursor)
        self.assertEqual(len(handoff_origins), 4)
        self.assertGreaterEqual(handoff_origins[0][1] - 11, viewer_bottom)
        self.assertEqual(
            [origin[1] for origin in handoff_origins],
            sorted(origin[1] for origin in handoff_origins),
        )

    def test_roi_label_moves_below_roi_when_status_panel_occupies_top(self):
        cv2 = _FakeCv2()
        frame = SimpleNamespace(shape=(600, 800, 3))
        roi = HeadRoi(
            x0=90,
            y0=105,
            x1=620,
            y1=390,
            source="candidate_search",
            expected_head_px=80.0,
        )

        first = _roi_label_origin(
            cv2,
            frame,
            roi,
            "candidate search ROI",
            font_scale=0.42,
            thickness=1,
            reserved_top_px=180,
            label_slot=0,
        )
        second = _roi_label_origin(
            cv2,
            frame,
            roi,
            "target ROI camera=0.0deg scan=0.0deg depth=1.0m",
            font_scale=0.42,
            thickness=1,
            reserved_top_px=180,
            label_slot=1,
        )

        self.assertGreater(first[1], roi.y1)
        self.assertGreater(second[1], first[1])

    @unittest.skipIf(
        real_cv2 is None or numpy is None,
        "OpenCV and numpy are required for rendering",
    )
    def test_real_opencv_status_stack_fits_resized_viewer_frame(self):
        frame = numpy.zeros((300, 440, 3), dtype=numpy.uint8)
        cursor = OverlayTextCursor()
        for line in (
            "camera axis rot proxy=+0.180 camera_yaw=12.3deg",
            "L=82.0px R=79.0px ratio=1.038",
            "closer=left stand_side=qr_code_side med_ratio=1.020 med_proxy=+0.010",
        ):
            cursor.draw(
                real_cv2,
                frame,
                line,
                font_face=real_cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.52,
                color=(0, 255, 0),
                thickness=2,
            )
        decision = AxisHandoffDecision(
            status="camera_collecting",
            accepted=False,
            reason="camera_consensus_pending",
            lidar=LidarAxisEstimate(
                usable=True,
                reason="axis_estimated",
                angle_rad=0.2,
                sample_count=18,
                linearity=0.91,
            ),
            camera=CameraAxisEstimate(
                usable=False,
                reason="camera_consensus_pending",
                sample_count=1,
            ),
        )

        annotate_axis_handoff(
            real_cv2,
            frame,
            decision,
            text_cursor=cursor,
        )

        self.assertLess(cursor.bottom_px, frame.shape[0] - 30)
        self.assertGreater(int(numpy.count_nonzero(frame)), 0)


if __name__ == "__main__":
    unittest.main()
