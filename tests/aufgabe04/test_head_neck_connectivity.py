"""Rounded raw neck junctions stay distinct from missing or detached borders."""

import hashlib
import json
from pathlib import Path
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.head_model_neck import measure_head_neck_junction
from scripts.aufgabe04.perception.stand_axis.head_neck_connectivity import trace_raw_neck_junction
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/aufgabe04/fixtures/head_neck_junction_20260911T142702Z"


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class HeadNeckConnectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(
            ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
        cls.fixture = json.loads((FIXTURE / "inputs.json").read_text())

    def recorded(self):
        data = (FIXTURE / "raw_edges.png").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), self.fixture["raw_edges_sha256"])
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(list(raw.shape), self.fixture["raw_edges_shape"])
        corners = tuple(ImagePoint(**p) for p in self.fixture["refined_corners"])
        return raw, corners

    def trace(self, raw, **overrides):
        arguments = dict(bottom_edge_px=((40., 130.), (140., 130.)),
                         rail_columns_px=(83, 97), core_start_row_px=134,
                         core_run_length_px=12, min_rail_gap_px=7,
                         max_rail_gap_px=34, max_start_gap_px=0)
        arguments.update(overrides)
        return trace_raw_neck_junction(raw, **arguments)

    def rounded(self):
        raw = np.zeros((210, 210), np.uint8)
        raw[134:161, 83] = raw[134:161, 97] = 255
        for y, left, right in ((131, 81, 99), (132, 82, 98), (133, 83, 98)):
            raw[y, left] = raw[y, right] = 255
        return raw

    def test_saved_raw_junction_preserves_core_and_measures_connected_start(self):
        raw, corners = self.recorded()
        original = raw.copy()
        junction = measure_head_neck_junction(raw, corners, self.profile)
        self.assertTrue(junction.accepted, junction.reason)
        self.assertEqual(junction.start_gap_px, 0)
        self.assertEqual(junction.core_start_gap_px, 3)
        self.assertEqual(junction.max_start_gap_px, 2)
        self.assertEqual(junction.run_start_row_px, 124)
        self.assertEqual(junction.required_run_px, 13)
        self.assertEqual(junction.rail_columns_px, (54, 71))
        self.assertEqual(junction.raw_continuation.start_gaps_px, (0, 0))
        self.assertEqual(junction.raw_continuation.paths_px, (
            ((54, 124), (54, 123), (53, 122), (52, 121)),
            ((71, 124), (72, 123), (72, 122), (73, 121)),
        ))
        for path in junction.raw_continuation.paths_px:
            self.assertTrue(all(raw[y, x] > 0 for x, y in path))
        proposal = tuple(ImagePoint(**p) for p in self.fixture["proposal_corners"])
        estimate, debug, _pose = fit_current_measured_head(
            cv2, raw, model_profile=self.profile,
            camera=RectifiedCameraMatrix(**self.fixture["camera"]), proposal_corners=proposal)
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(debug.head_model_quality.pose_model, "measured_head_only")
        # A broad numerical replay bound, not a ground-truth hardware angle.
        self.assertAlmostEqual(estimate.yaw_deg, 26.2, delta=1.0)
        np.testing.assert_array_equal(raw, original)

    def test_same_saved_pixels_do_not_authenticate_inner_paper(self):
        raw, corners = self.recorded()
        cx, cy = (sum(getattr(p, name) for p in corners) / 4 for name in ("u_px", "v_px"))
        paper = tuple(ImagePoint(cx + (p.u_px - cx) * 71 / 78,
                                 cy + (p.v_px - cy) * 71 / 78) for p in corners)
        junction = measure_head_neck_junction(raw, paper, self.profile)
        self.assertFalse(junction.accepted)
        self.assertEqual(junction.reason, "head_neck_junction_gap_too_large")
        self.assertEqual(junction.start_gap_px, junction.core_start_gap_px)
        self.assertGreater(junction.start_gap_px, junction.max_start_gap_px)

    def test_missing_row_never_becomes_a_connected_junction(self):
        raw, corners = self.recorded()
        raw[123] = 0
        original = raw.copy()
        junction = measure_head_neck_junction(raw, corners, self.profile)
        self.assertFalse(junction.accepted)
        self.assertEqual(junction.reason, "head_neck_junction_gap_too_large")
        self.assertEqual(junction.start_gap_px, 3)
        self.assertEqual(junction.core_start_gap_px, 3)
        self.assertEqual(junction.run_start_row_px, 124)
        self.assertFalse(junction.raw_continuation.accepted)
        np.testing.assert_array_equal(raw, original)

    def test_rounded_synthetic_paths_use_only_neighboring_raw_pixels(self):
        raw = self.rounded()
        original = raw.copy()
        continuation = self.trace(raw)
        self.assertTrue(continuation.accepted, continuation.reason)
        self.assertEqual(continuation.start_gaps_px, (0, 0))
        for path in continuation.paths_px:
            anchor = path[0][0]
            self.assertTrue(all(raw[y, x] > 0 and abs(x - anchor) <= 2 for x, y in path))
            self.assertTrue(all(y1 - y2 == 1 and abs(x1 - x2) <= 1
                                for (x1, y1), (x2, y2) in zip(path, path[1:])))
        np.testing.assert_array_equal(raw, original)

    def test_sloping_bottom_has_separate_current_pixel_junction_rows(self):
        raw = np.zeros((210, 210), np.uint8)
        raw[131:161, 83] = raw[135:161, 97] = 255
        for x, y in ((81, 129), (82, 130), (99, 132), (98, 133), (98, 134)):
            raw[y, x] = 255
        result = self.trace(raw, bottom_edge_px=((40., 120.), (140., 140.)), core_start_row_px=135)
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.start_gaps_px, (0, 0))
        self.assertEqual(tuple(path[-1][1] for path in result.paths_px), (129, 132))

    def test_unbounded_depth_lateral_shift_and_missing_step_are_rejected(self):
        for mode in ("depth", "shift", "step", "one_missing", "nan_pixel"):
            with self.subTest(mode=mode):
                raw = self.rounded().astype(float)
                start = 134
                if mode == "depth":
                    raw[:] = 0
                    start = 138
                    raw[131:161, 83] = raw[131:161, 97] = 255
                elif mode == "shift":
                    raw[131:134] = 0
                    for y, left, right in ((131, 80, 100), (132, 81, 99), (133, 82, 98)):
                        raw[y, left] = raw[y, right] = 255
                elif mode == "step":
                    raw[132, 82] = 0
                    raw[132, 83] = 255  # Two columns away from the row above.
                elif mode == "one_missing":
                    raw[132, 82] = 0
                else:
                    raw[132, 82] = float("nan")
                result = self.trace(raw, core_start_row_px=start)
                self.assertFalse(result.accepted)
                self.assertGreater(max(result.start_gaps_px), 0)

    def test_crossing_or_out_of_width_paths_cannot_authenticate_pair(self):
        for mode in ("crossing", "wide"):
            with self.subTest(mode=mode):
                raw = np.zeros((210, 210), np.uint8)
                anchors = (83, 86) if mode == "crossing" else (83, 97)
                raw[134:161, anchors[0]] = raw[134:161, anchors[1]] = 255
                points = (((85, 131), (84, 131), (85, 132), (84, 132), (84, 133), (85, 133))
                          if mode == "crossing" else
                          ((81, 131), (99, 131), (82, 132), (98, 132), (83, 133), (98, 133)))
                for x, y in points:
                    raw[y, x] = 255
                result = self.trace(raw, rail_columns_px=anchors, min_rail_gap_px=3,
                                    max_rail_gap_px=anchors[1] - anchors[0])
                self.assertFalse(result.accepted)

    def test_invalid_clipped_vertical_and_nonfinite_bottoms_fail_closed(self):
        raw = self.rounded()
        for edge in (
            ((40., float("nan")), (140., 130.)),
            ((40., 130.), (float("inf"), 130.)),
            ((83., 100.), (83., 140.)),
            ((83., 100.), (83.001, 140.)),
            ((40., 20.), (140., 190.)),
            ((-1., 130.), (140., 130.)),
            ((40., 130.), (210., 130.)),
            ((40., -1.), (140., 130.)),
        ):
            with self.subTest(bottom=edge):
                result = self.trace(raw, bottom_edge_px=edge)
                self.assertFalse(result.accepted)
                self.assertEqual(result.reason, "raw_neck_continuation_input_invalid")
        for overrides in (dict(core_start_row_px=-1), dict(core_start_row_px=200),
                          dict(rail_columns_px=(-1, 97)), dict(rail_columns_px=(97, 83)),
                          dict(core_run_length_px=2), dict(max_start_gap_px=3)):
            with self.subTest(overrides=overrides):
                self.assertFalse(self.trace(raw, **overrides).accepted)

    def test_continuation_cannot_replace_missing_stable_core(self):
        raw = self.rounded()
        raw[138, 83] = 0
        result = self.trace(raw)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "raw_neck_continuation_core_unavailable")


if __name__ == "__main__":
    unittest.main()
