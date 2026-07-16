import ast
import sys
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover - exercised only when local deps are missing
    cv2 = None
    numpy = None


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.stand_axis_viewer import (  # noqa: E402
    HeadCandidateTemporalGate,
    WINDOW_EDGES,
    WINDOW_FACE_MASK,
    WINDOW_FRAME,
    WINDOW_MASK,
    WINDOW_RECTANGLE_MASK,
    _capture_head_display_snapshot,
    _detected_head_roi,
    _diagnostic_roi_image,
    _head_display_snapshot_for_selection,
    _initialize_display_windows,
    _resize_diagnostic_windows,
    _simulation_full_frame_edge_mode,
    _standalone_head_geometry_reason,
    _validate_runtime_args,
    build_parser,
)
from scripts.aufgabe04.perception.stand_axis_image import (  # noqa: E402
    ImagePoint,
    _SilhouetteFaceCandidate,
    _connected_border_mask_and_corners,
    _debug_rectangle_overlay_image,
    _edge_pixels_inside_polygon,
    _expanded_head_edge_roi,
    _face_quadrilateral_from_silhouette,
    _level_camera_endpoint_perspective_consistent,
    _plain_face_from_stem_cropped_edges,
    _quadrilateral_edge_support,
    _raw_side_evidence_and_corners,
    _scale_quadrilateral_about_center,
    _select_supported_head_corners,
    _stem_anchor_from_edges,
    _stem_anchor_candidates_from_edges,
    _stem_anchored_face_from_edges,
    _validated_refitted_head_corners,
    estimate_edge_on_axis_from_line,
    estimate_stand_axis_from_edges,
    estimate_stand_axis_from_corners,
    order_corners,
    quadrilateral_aspect_ratio,
    wide_row_band,
)
class StandAxisImageTest(unittest.TestCase):
    def make_corners(self, points):
        return tuple(ImagePoint(float(u), float(v)) for u, v in points)

    def make_equal_luminance_flicker_frame(self, head_color):
        """Approximate the two-stand Gazebo frame captured on 2026-07-15."""

        frame = numpy.full((480, 640, 3), (77, 77, 77), dtype=numpy.uint8)
        head = numpy.array(
            [[(303, 149), (337, 155), (337, 201), (303, 200)]],
            dtype=numpy.int32,
        )
        cv2.fillPoly(frame, head, head_color)
        cv2.rectangle(frame, (314, 199), (326, 281), head_color, thickness=cv2.FILLED)

        # QR-like achromatic texture is deliberately prominent. It must remain
        # measurement evidence without becoming the fitted head rectangle.
        cv2.rectangle(frame, (309, 164), (331, 190), (225, 225, 225), thickness=cv2.FILLED)
        cv2.rectangle(frame, (314, 169), (319, 176), (10, 10, 10), thickness=cv2.FILLED)
        cv2.rectangle(frame, (323, 179), (329, 187), (10, 10, 10), thickness=cv2.FILLED)
        cv2.line(frame, (0, 219), (639, 219), (68, 68, 68), thickness=2)

        # A second stand is close enough to pollute full-row envelopes but not
        # the target stem's local search corridor.
        other_color = (head_color[1], head_color[2], head_color[0])
        cv2.rectangle(frame, (246, 172), (261, 194), other_color, thickness=cv2.FILLED)
        cv2.rectangle(frame, (252, 194), (255, 254), other_color, thickness=cv2.FILLED)
        return frame

    @unittest.skipIf(numpy is None, "numpy is required for ROI tests")
    def test_simulation_diagnostic_mask_uses_same_target_roi_as_edges(self):
        class Roi:
            x0 = 20
            y0 = 10
            x1 = 60
            y1 = 50

        full_mask = numpy.zeros((80, 100), dtype=numpy.uint8)
        cropped = _diagnostic_roi_image(full_mask, Roi())

        self.assertEqual(cropped.shape, (40, 40))
        self.assertIs(_diagnostic_roi_image(full_mask, None), full_mask)

    def test_standalone_simulation_edges_use_full_frame_before_dynamic_roi(self):
        args = build_parser().parse_args(
            [
                "--sim-raw-image-topic",
                "/camera/image_raw",
                "--axis-source",
                "edges",
                "--lidar-bearing-source",
                "fixed",
            ]
        )

        self.assertTrue(_simulation_full_frame_edge_mode(args))
        _validate_runtime_args(args)

    def test_map_target_simulation_edges_keep_projected_target_roi(self):
        args = build_parser().parse_args(
            [
                "--sim-raw-image-topic",
                "/camera/image_raw",
                "--axis-source",
                "edges",
                "--lidar-bearing-source",
                "map-target",
            ]
        )

        self.assertFalse(_simulation_full_frame_edge_mode(args))

    def test_detected_head_roi_is_derived_from_full_frame_corners(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(90, 40), (170, 50), (165, 130), (85, 120)])
        )

        roi = _detected_head_roi(
            estimate,
            frame_width=640,
            frame_height=480,
            padding_scale=1.25,
        )

        self.assertIsNotNone(roi)
        self.assertEqual(roi.source, "edge_detected")
        self.assertLessEqual(roi.x0, 80)
        self.assertGreaterEqual(roi.x1, 175)
        self.assertLessEqual(roi.y0, 35)
        self.assertGreaterEqual(roi.y1, 135)

    def test_temporal_gate_rejects_one_frame_wall_candidate(self):
        gate = HeadCandidateTemporalGate(reacquire_frames=3)
        good = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
            ),
            yaw_deg=-20.0,
        )
        wall = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(0, 70), (430, 105), (250, 205), (145, 190)])
            ),
            yaw_deg=20.0,
        )

        self.assertEqual(gate.accept(good), (True, "accepted"))
        self.assertEqual(gate.accept(wall), (False, "temporal_head_outlier"))
        self.assertEqual(gate.accept(good), (True, "accepted"))

    def test_temporal_gate_reacquires_a_consistent_new_view(self):
        gate = HeadCandidateTemporalGate(reacquire_frames=3)
        initial = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(80, 80), (160, 80), (160, 160), (80, 160)])
            ),
            yaw_deg=0.0,
        )
        moved = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(300, 100), (390, 100), (390, 190), (300, 190)])
            ),
            yaw_deg=5.0,
        )

        self.assertEqual(gate.accept(initial), (True, "accepted"))
        self.assertEqual(gate.accept(moved), (False, "temporal_head_outlier"))
        self.assertEqual(gate.accept(moved), (False, "temporal_head_outlier"))
        self.assertEqual(gate.accept(moved), (True, "reacquired"))

    def test_temporal_gate_holds_last_valid_head_across_single_outlier(self):
        gate = HeadCandidateTemporalGate(reacquire_frames=3, hold_sec=0.35)
        good = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
            ),
            yaw_deg=-20.0,
        )
        wall = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(0, 70), (430, 105), (250, 205), (145, 190)])
            ),
            yaw_deg=20.0,
        )

        initial = gate.stabilize(good, now_sec=10.0)
        held = gate.stabilize(wall, now_sec=10.1)
        recovered = gate.stabilize(good, now_sec=10.2)

        self.assertTrue(initial.current_accepted)
        self.assertFalse(initial.held)
        self.assertFalse(held.current_accepted)
        self.assertTrue(held.held)
        self.assertTrue(held.estimate.usable)
        self.assertEqual(
            held.estimate.reason,
            "temporal_hold_after_temporal_head_outlier",
        )
        self.assertIsNotNone(
            _detected_head_roi(
                held.estimate,
                frame_width=640,
                frame_height=480,
                padding_scale=1.6,
            )
        )
        self.assertTrue(recovered.current_accepted)
        self.assertFalse(recovered.held)

    def test_temporal_hold_expires_instead_of_latching_stale_head(self):
        gate = HeadCandidateTemporalGate(reacquire_frames=3, hold_sec=0.35)
        good = estimate_stand_axis_from_corners(
            self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
        )
        missing = replace(
            good,
            usable=False,
            reason="silhouette_head_unavailable",
            mode="unavailable",
            corners=None,
        )

        gate.stabilize(good, now_sec=20.0)
        held = gate.stabilize(missing, now_sec=20.2)
        expired = gate.stabilize(missing, now_sec=20.36)

        self.assertTrue(held.held)
        self.assertIsNotNone(held.estimate)
        self.assertFalse(expired.held)
        self.assertIsNone(expired.estimate)
        self.assertEqual(expired.reason, "silhouette_head_unavailable")

    @unittest.skipIf(numpy is None, "numpy is required for display snapshot tests")
    def test_temporal_hold_reuses_one_atomic_accepted_display_frame(self):
        gate = HeadCandidateTemporalGate(reacquire_frames=3, hold_sec=0.35)
        good = estimate_stand_axis_from_corners(
            self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
        )
        missing = replace(
            good,
            usable=False,
            reason="silhouette_head_unavailable",
            mode="unavailable",
            corners=None,
        )
        accepted_selection = gate.stabilize(good, now_sec=30.0)
        accepted_pixels = numpy.full((3, 4), 17, dtype=numpy.uint8)
        accepted = _capture_head_display_snapshot(
            frame=accepted_pixels,
            mask=accepted_pixels + 1,
            edges=accepted_pixels + 2,
            face_mask=accepted_pixels + 3,
            rectangle_mask=accepted_pixels + 4,
            rectangle_overlay=accepted_pixels + 5,
            detected_head_roi=None,
            diagnostic_head_roi=None,
        )
        current_pixels = numpy.full((3, 4), 91, dtype=numpy.uint8)
        current = _capture_head_display_snapshot(
            frame=current_pixels,
            mask=current_pixels + 1,
            edges=current_pixels + 2,
            face_mask=current_pixels + 3,
            rectangle_mask=current_pixels + 4,
            rectangle_overlay=current_pixels + 5,
            detected_head_roi=None,
            diagnostic_head_roi=None,
        )
        held_selection = gate.stabilize(missing, now_sec=30.1)

        fresh_display = _head_display_snapshot_for_selection(
            accepted_selection,
            current=accepted,
            last_accepted=accepted,
        )
        held_display = _head_display_snapshot_for_selection(
            held_selection,
            current=current,
            last_accepted=accepted,
        )
        accepted_pixels[:, :] = 0

        self.assertIs(fresh_display, accepted)
        self.assertIs(held_display, accepted)
        self.assertTrue(numpy.all(held_display.frame == 17))
        self.assertTrue(numpy.all(held_display.mask == 18))
        self.assertTrue(numpy.all(held_display.edges == 19))
        self.assertTrue(numpy.all(held_display.face_mask == 20))
        self.assertTrue(numpy.all(held_display.rectangle_mask == 21))
        self.assertTrue(numpy.all(held_display.rectangle_overlay == 22))

    def test_temporal_reacquisition_requires_consecutive_usable_candidates(self):
        gate = HeadCandidateTemporalGate(reacquire_frames=3)
        initial = estimate_stand_axis_from_corners(
            self.make_corners([(80, 80), (160, 80), (160, 160), (80, 160)])
        )
        moved = estimate_stand_axis_from_corners(
            self.make_corners([(300, 100), (390, 100), (390, 190), (300, 190)])
        )
        missing = replace(
            initial,
            usable=False,
            reason="silhouette_head_unavailable",
            mode="unavailable",
            corners=None,
        )

        self.assertEqual(gate.accept(initial), (True, "accepted"))
        self.assertEqual(gate.accept(moved), (False, "temporal_head_outlier"))
        self.assertEqual(gate.accept(missing), (False, "silhouette_head_unavailable"))
        self.assertEqual(gate.accept(moved), (False, "temporal_head_outlier"))
        self.assertEqual(gate.accept(moved), (False, "temporal_head_outlier"))
        self.assertEqual(gate.accept(moved), (True, "reacquired"))

    def test_temporal_gate_never_reacquires_persistent_size_doubling(self):
        gate = HeadCandidateTemporalGate(reacquire_frames=3)
        initial = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
            ),
            yaw_deg=-20.0,
        )
        wall_sized = replace(
            estimate_stand_axis_from_corners(
                self.make_corners([(90, 20), (330, 25), (325, 245), (95, 240)])
            ),
            yaw_deg=-10.0,
        )

        self.assertEqual(gate.accept(initial), (True, "accepted"))
        for _ in range(5):
            self.assertEqual(
                gate.accept(wall_sized),
                (False, "temporal_head_outlier"),
            )

    def test_standalone_geometry_rejects_wall_sized_initial_candidate(self):
        good = estimate_stand_axis_from_corners(
            self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
        )
        wall_sized = estimate_stand_axis_from_corners(
            self.make_corners([(90, 20), (330, 25), (325, 245), (95, 240)])
        )

        self.assertIsNone(
            _standalone_head_geometry_reason(
                good,
                frame_width=640,
                frame_height=480,
            )
        )
        self.assertEqual(
            _standalone_head_geometry_reason(
                wall_sized,
                frame_width=640,
                frame_height=480,
            ),
            "head_candidate_too_large",
        )

    def test_refitted_head_corners_reject_connected_wall_expansion(self):
        rough = self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
        wall_corrupted = self.make_corners([(0, 70), (430, 105), (250, 205), (145, 190)])

        selected = _validated_refitted_head_corners(
            rough,
            wall_corrupted,
            image_shape=(480, 640),
            stem_center_x=200.0,
            stem_top_y=205.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNone(selected)

    def test_refitted_head_corners_accept_local_border_refinement(self):
        rough = self.make_corners([(150, 90), (250, 95), (248, 198), (152, 192)])
        refined = self.make_corners([(148, 88), (252, 94), (250, 200), (150, 194)])

        selected = _validated_refitted_head_corners(
            rough,
            refined,
            image_shape=(480, 640),
            stem_center_x=200.0,
            stem_top_y=205.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertEqual(selected, refined)

    def test_refitted_head_corners_accept_outer_board_from_inner_panel_seed(self):
        # Gazebo's 53.147 mm white panel is embedded in the 69.930 mm head.
        # At close range the panel can seed topology, but independently fitted
        # raw edges must be allowed to expand to the real board boundary.
        rough_panel = self.make_corners(
            [(294, 142), (346, 144), (345, 196), (293, 194)]
        )
        outer_board = self.make_corners(
            [(283, 133), (355, 133), (355, 205), (283, 205)]
        )

        selected = _validated_refitted_head_corners(
            rough_panel,
            outer_board,
            image_shape=(480, 640),
            stem_center_x=319.0,
            stem_top_y=205.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertEqual(selected, order_corners(outer_board))

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for edge support tests")
    def test_quadrilateral_edge_support_accepts_bottom_stem_notch(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        corners = self.make_corners([(55, 35), (165, 30), (168, 118), (58, 122)])
        cv2.line(cutout, (55, 35), (165, 30), 255, thickness=2)
        cv2.line(cutout, (55, 35), (58, 122), 255, thickness=2)
        cv2.line(cutout, (165, 30), (168, 118), 255, thickness=2)
        cv2.line(cutout, (58, 122), (102, 121), 255, thickness=2)
        cv2.line(cutout, (126, 120), (168, 118), 255, thickness=2)

        support = _quadrilateral_edge_support(cv2, cutout, corners)

        self.assertTrue(support.accepted)
        self.assertGreaterEqual(support.bottom_left, 0.45)
        self.assertGreaterEqual(support.bottom_right, 0.45)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for edge support tests")
    def test_quadrilateral_edge_support_rejects_constructed_bottom(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        corners = self.make_corners([(55, 35), (165, 30), (168, 118), (58, 122)])
        cv2.line(cutout, (55, 35), (165, 30), 255, thickness=2)
        cv2.line(cutout, (55, 35), (58, 122), 255, thickness=2)
        cv2.line(cutout, (165, 30), (168, 118), 255, thickness=2)

        support = _quadrilateral_edge_support(cv2, cutout, corners)
        selected, reason, selected_support = _select_supported_head_corners(
            cv2,
            cutout,
            corners,
            corners,
            image_shape=cutout.shape,
            stem_center_x=112.0,
            stem_top_y=122.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertFalse(support.accepted)
        self.assertLess(support.bottom_left, 0.45)
        self.assertLess(support.bottom_right, 0.45)
        self.assertIsNone(selected)
        self.assertEqual(reason, "head_rectangle_fit_unreliable")
        self.assertIsNotNone(selected_support)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for edge support tests")
    def test_independent_measurement_requires_raw_refitted_corners(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        rough = self.make_corners([(55, 35), (165, 30), (168, 118), (58, 122)])
        cv2.polylines(
            cutout,
            numpy.array(
                [[(55, 35), (165, 30), (168, 118), (58, 122)]],
                dtype=numpy.int32,
            ),
            True,
            255,
            thickness=2,
        )

        selected, reason, support = _select_supported_head_corners(
            cv2,
            cutout,
            rough,
            None,
            image_shape=cutout.shape,
            stem_center_x=112.0,
            stem_top_y=122.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
            allow_rough_fallback=False,
        )

        self.assertTrue(_quadrilateral_edge_support(cv2, cutout, rough).accepted)
        self.assertIsNone(selected)
        self.assertEqual(reason, "head_rectangle_fit_unreliable")
        self.assertIsNone(support)

    def test_order_corners_returns_expected_sequence(self):
        corners = order_corners(
            [
                ImagePoint(90, 80),
                ImagePoint(20, 20),
                ImagePoint(25, 85),
                ImagePoint(95, 25),
            ]
        )

        self.assertEqual([(point.u_px, point.v_px) for point in corners], [(20, 20), (95, 25), (90, 80), (25, 85)])

    def test_quadrilateral_aspect_ratio_uses_outer_edges(self):
        ratio = quadrilateral_aspect_ratio(
            self.make_corners([(40, 20), (100, 20), (100, 80), (40, 80)])
        )

        self.assertAlmostEqual(ratio, 1.0)

    def test_wide_row_band_selects_square_above_narrow_stem(self):
        row_widths = [4, 80, 82, 81, 79, 22, 20, 19]

        self.assertEqual(wide_row_band(row_widths, width_fraction=0.60), (1, 4))

    def test_scale_quadrilateral_about_center_uses_fixed_dimension_ratio(self):
        corners = _scale_quadrilateral_about_center(
            self.make_corners([(40, 30), (80, 30), (80, 70), (40, 70)]),
            1.5,
        )

        self.assertEqual(
            [(round(point.u_px), round(point.v_px)) for point in corners],
            [(30, 20), (90, 20), (90, 80), (30, 80)],
        )

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_face_mask_shows_selected_head_outline_only(self):
        silhouette = numpy.zeros((150, 160), dtype=numpy.uint8)
        cv2.rectangle(silhouette, (30, 20), (110, 100), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (62, 100), (78, 135), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (45, 42), (55, 52), 0, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (82, 65), (92, 75), 0, thickness=cv2.FILLED)

        candidate = _face_quadrilateral_from_silhouette(
            cv2,
            silhouette,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
            face_width_fraction=0.60,
        )

        self.assertIsNotNone(candidate)
        face_mask = candidate.face_mask
        self.assertEqual(face_mask.shape, silhouette.shape)
        self.assertEqual(int(face_mask[20, 30]), 255)
        self.assertEqual(int(face_mask[20, 110]), 255)
        self.assertEqual(int(face_mask[100, 30]), 255)
        self.assertEqual(int(face_mask[100, 110]), 255)
        self.assertEqual(int(face_mask[47, 50]), 0)
        self.assertEqual(int(face_mask[70, 87]), 0)
        self.assertEqual(int(face_mask[120, 70]), 0)
        self.assertEqual(int(face_mask[15, 70]), 0)
        self.assertEqual(candidate.corners, order_corners(candidate.corners))
        self.assertAlmostEqual(quadrilateral_aspect_ratio(candidate.corners), 1.0, delta=0.08)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_prefers_stand_shape_over_larger_plain_square(self):
        silhouette = numpy.zeros((170, 260), dtype=numpy.uint8)
        cv2.rectangle(silhouette, (140, 30), (235, 125), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (30, 25), (105, 100), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (60, 100), (75, 150), 255, thickness=cv2.FILLED)

        candidate = _face_quadrilateral_from_silhouette(
            cv2,
            silhouette,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
            face_width_fraction=0.60,
        )

        self.assertIsNotNone(candidate)
        x_values = [point.u_px for point in candidate.corners]
        self.assertLess(max(x_values), 120)
        self.assertGreater(min(x_values), 20)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_expands_inner_face_to_detached_outer_edge(self):
        edges = numpy.zeros((150, 180), dtype=numpy.uint8)
        cv2.rectangle(edges, (45, 25), (115, 95), 255, thickness=cv2.FILLED)
        cv2.line(edges, (135, 22), (137, 98), 255, thickness=2)
        cv2.line(edges, (78, 95), (78, 135), 255, thickness=3)

        candidate = _face_quadrilateral_from_silhouette(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
            face_width_fraction=0.60,
        )

        self.assertIsNotNone(candidate)
        self.assertGreater(max(point.u_px for point in candidate.corners), 130)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_stem_anchored_face_uses_outer_box_around_qr_clutter(self):
        edges = numpy.zeros((180, 220), dtype=numpy.uint8)
        cv2.rectangle(edges, (55, 25), (150, 120), 255, thickness=2)
        cv2.rectangle(edges, (70, 42), (132, 105), 255, thickness=2)
        cv2.line(edges, (85, 55), (128, 92), 255, thickness=2)
        cv2.line(edges, (95, 120), (95, 165), 255, thickness=2)
        cv2.line(edges, (110, 120), (110, 165), 255, thickness=2)

        candidate = _stem_anchored_face_from_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        x_values = [point.u_px for point in candidate.corners]
        y_values = [point.v_px for point in candidate.corners]
        self.assertLessEqual(min(x_values), 58)
        self.assertGreaterEqual(max(x_values), 147)
        self.assertLessEqual(min(y_values), 28)
        self.assertGreaterEqual(max(y_values), 117)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_stem_anchored_face_preserves_slanted_outer_silhouette(self):
        edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        outer = numpy.array([[(60, 30), (160, 20), (153, 122), (65, 115)]], dtype=numpy.int32)
        cv2.polylines(edges, outer, True, 255, thickness=2)
        cv2.rectangle(edges, (80, 45), (138, 104), 255, thickness=2)
        cv2.line(edges, (90, 58), (134, 92), 255, thickness=2)
        cv2.line(edges, (101, 118), (101, 172), 255, thickness=2)
        cv2.line(edges, (116, 118), (116, 172), 255, thickness=2)

        candidate = _stem_anchored_face_from_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        top_left, top_right, bottom_right, bottom_left = candidate.corners
        self.assertLess(top_left.u_px, bottom_left.u_px)
        self.assertGreater(top_right.u_px, bottom_right.u_px)
        self.assertLessEqual(top_left.u_px, 64)
        self.assertGreaterEqual(top_right.u_px, 156)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_stem_anchor_prefers_lower_stem_over_longer_head_border(self):
        edges = numpy.zeros((120, 160), dtype=numpy.uint8)
        cv2.rectangle(edges, (25, 12), (125, 66), 255, thickness=2)
        cv2.line(edges, (70, 65), (70, 112), 255, thickness=2)
        cv2.line(edges, (80, 65), (80, 112), 255, thickness=2)

        anchor = _stem_anchor_from_edges(cv2, edges, min_edge_height_px=12.0)

        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor[0], 75.0, delta=4.0)
        self.assertGreater(anchor[1], 55.0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_full_frame_stem_anchor_accepts_short_visible_gazebo_stem(self):
        edges = numpy.zeros((480, 640), dtype=numpy.uint8)
        cv2.rectangle(edges, (208, 98), (278, 202), 255, thickness=2)
        cv2.line(edges, (232, 224), (232, 286), 255, thickness=2)
        cv2.line(edges, (249, 224), (249, 286), 255, thickness=2)
        cv2.line(edges, (0, 288), (639, 276), 255, thickness=2)

        anchor = _stem_anchor_from_edges(cv2, edges, min_edge_height_px=8.0)

        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor[0], 240.5, delta=5.0)
        self.assertLess(anchor[1], 250.0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_simulation_low_contrast_stem_produces_contour_edge_cutout(self):
        frame = numpy.full((160, 160, 3), (55, 55, 55), dtype=numpy.uint8)
        head = numpy.array([[(32, 28), (123, 35), (123, 120), (34, 118)]], dtype=numpy.int32)
        cv2.fillPoly(frame, head, (0, 165, 51))
        cv2.rectangle(frame, (73, 120), (86, 159), (0, 165, 51), thickness=cv2.FILLED)
        cv2.rectangle(frame, (48, 48), (108, 104), (235, 235, 235), thickness=cv2.FILLED)
        cv2.line(frame, (16, 8), (145, 44), (95, 95, 95), thickness=2)

        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            canny_low=20,
            canny_high=60,
            close_kernel=5,
            min_area_px=900.0,
            min_face_area_fraction=0.0,
            min_edge_height_px=12.0,
            silhouette_only=True,
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.source, "edge_plain_face_stem_anchor")
        self.assertIsNotNone(artifacts.face_mask)
        self.assertEqual(artifacts.edges.shape, artifacts.face_mask.shape)
        # The unrelated diagonal wall edge must not leak into the head cutout.
        self.assertEqual(int(artifacts.face_mask[8, 16]), 0)
        self.assertGreater(cv2.countNonZero(artifacts.face_mask[25:125, 25:130]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_channel_union_recovers_equal_luminance_head_and_ignores_qr_rectangle(self):
        frame = self.make_equal_luminance_flicker_frame((40, 131, 0))
        options = dict(
            blur_kernel=5,
            canny_low=20,
            canny_high=60,
            dilate_iterations=0,
            close_kernel=3,
            close_iterations=1,
            min_face_area_fraction=0.0,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            silhouette_only=True,
        )

        gray_estimate, _gray_artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            **options,
        )
        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="channel_union",
            **options,
        )

        self.assertFalse(gray_estimate.usable)
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertIsNotNone(artifacts.raw_edges)
        self.assertIsNotNone(artifacts.face_mask)
        self.assertGreater(
            cv2.countNonZero(
                cv2.bitwise_and(artifacts.edges, cv2.bitwise_not(artifacts.raw_edges))
            ),
            0,
        )
        self.assertEqual(
            cv2.countNonZero(
                cv2.bitwise_and(
                    artifacts.face_mask,
                    cv2.bitwise_not(artifacts.raw_edges),
                )
            ),
            0,
        )
        expected = self.make_corners(
            [(303, 149), (337, 155), (337, 201), (303, 200)]
        )
        ordered = order_corners(estimate.corners)
        for actual, wanted in zip(ordered, expected):
            self.assertAlmostEqual(actual.u_px, wanted.u_px, delta=2.0)
            self.assertAlmostEqual(actual.v_px, wanted.v_px, delta=2.0)
        self.assertAlmostEqual(ordered[0].u_px, ordered[3].u_px, places=6)
        self.assertAlmostEqual(ordered[1].u_px, ordered[2].u_px, places=6)
        # The fitted lower edge follows the physical head near y=200, not the
        # QR texture's lower horizontal edge at y=190.
        self.assertGreater(min(point.v_px for point in estimate.corners[2:]), 198.0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_channel_union_is_invariant_to_head_color_channel_permutation(self):
        options = dict(
            edge_preprocess="channel_union",
            blur_kernel=5,
            canny_low=20,
            canny_high=60,
            dilate_iterations=0,
            close_kernel=3,
            close_iterations=1,
            min_face_area_fraction=0.0,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            silhouette_only=True,
        )
        estimates = []
        raw_masks = []
        for color in ((40, 131, 0), (131, 0, 40), (0, 40, 131)):
            estimate, artifacts = estimate_stand_axis_from_edges(
                cv2,
                self.make_equal_luminance_flicker_frame(color),
                **options,
            )
            self.assertTrue(estimate.usable, estimate.reason)
            estimates.append(order_corners(estimate.corners))
            raw_masks.append(artifacts.raw_edges)

        for raw_mask in raw_masks[1:]:
            self.assertTrue(numpy.array_equal(raw_masks[0], raw_mask))
        for corners in estimates[1:]:
            for baseline, permuted in zip(estimates[0], corners):
                self.assertAlmostEqual(baseline.u_px, permuted.u_px, places=6)
                self.assertAlmostEqual(baseline.v_px, permuted.v_px, places=6)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_gap_recovery_keeps_one_raw_measurement_domain(self):
        frame = numpy.full((120, 160, 3), 80, dtype=numpy.uint8)
        face_mask = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
        corners = self.make_corners([(45, 20), (115, 20), (115, 80), (45, 80)])
        reliable = _SilhouetteFaceCandidate(corners=corners, face_mask=face_mask)

        with patch(
            "scripts.aufgabe04.perception.stand_axis_image._plain_face_from_stem_cropped_edges",
            side_effect=[None, reliable],
        ) as detector:
            estimate, artifacts = estimate_stand_axis_from_edges(
                cv2,
                frame,
                edge_preprocess="channel_union",
                canny_low=20,
                canny_high=60,
                dilate_iterations=0,
                close_kernel=3,
                close_iterations=1,
                min_face_area_fraction=0.0,
                silhouette_only=True,
            )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(detector.call_count, 2)
        first_measurement = detector.call_args_list[0].kwargs["measurement_edges"]
        second_measurement = detector.call_args_list[1].kwargs["measurement_edges"]
        self.assertIs(first_measurement, second_measurement)
        self.assertIs(artifacts.raw_edges, first_measurement)
        self.assertIs(artifacts.edges, detector.call_args_list[1].args[1])

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_head_rejects_connected_wall_seam_branch_not_the_head(self):
        """A background seam may touch both head sides without becoming a corner.

        This reproduces the acquisition view from sim_hybrid_frontality_018:
        the upper/lower arena-wall boundary is visible on both sides of the
        foreground stand and is therefore connected to the head silhouette in
        Canny space.  The real head remains a complete four-sided shape with a
        long narrow stem below it.
        """

        frame = numpy.full((88, 88, 3), (80, 80, 80), dtype=numpy.uint8)
        frame[:30, :] = (190, 190, 190)
        head = numpy.array(
            [[(27, 10), (58, 18), (58, 64), (28, 64)]],
            dtype=numpy.int32,
        )
        cv2.fillPoly(frame, head, (0, 165, 51))
        cv2.rectangle(frame, (43, 64), (48, 87), (0, 165, 51), thickness=cv2.FILLED)
        # Internal label texture must remain irrelevant to silhouette yaw.
        cv2.rectangle(frame, (35, 25), (53, 48), (235, 235, 235), thickness=cv2.FILLED)

        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            canny_low=20,
            canny_high=60,
            close_kernel=3,
            min_area_px=293.0,
            min_face_area_fraction=0.0,
            min_edge_height_px=9.75,
            silhouette_only=True,
            stand_width_m=0.078,
            stand_distance_m=0.55,
            camera_fx_px=381.36246688,
            camera_fy_px=381.36246688,
            camera_cx_px=44.0,
            camera_cy_px=87.0,
        )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.source, "edge_plain_face_stem_anchor")
        self.assertEqual(estimate.mode, "face_visible")
        self.assertEqual(estimate.closer_side, "left")
        self.assertIsNotNone(estimate.yaw_deg)
        # The left image edge is closer, so public/ROS-positive yaw is left.
        self.assertGreater(estimate.yaw_deg, 35.0)
        self.assertLess(estimate.yaw_deg, 60.0)
        self.assertIsNotNone(estimate.corners)
        self.assertIsNotNone(artifacts.face_mask)
        self.assertIsNotNone(artifacts.rectangle_mask)
        ordered = order_corners(estimate.corners)
        self.assertLess(max(point.u_px for point in ordered), 64.0)
        self.assertGreater(min(point.u_px for point in ordered), 20.0)
        self.assertGreater(max(point.v_px for point in ordered), 60.0)
        # The seam remains visible in the full diagnostic edge image, but its
        # branches outside the foreground head must not enter the cutout.
        self.assertGreater(cv2.countNonZero(artifacts.edges[27:33, :]), 40)
        self.assertEqual(cv2.countNonZero(artifacts.face_mask[27:33, :20]), 0)
        self.assertEqual(cv2.countNonZero(artifacts.face_mask[27:33, 66:]), 0)
        blurred_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )
        raw_edges = cv2.Canny(blurred_gray, 20, 60)
        self.assertEqual(
            cv2.countNonZero(
                cv2.bitwise_and(
                    artifacts.face_mask,
                    cv2.bitwise_not(raw_edges),
                )
            ),
            0,
        )
        support = _quadrilateral_edge_support(cv2, artifacts.face_mask, estimate.corners)
        self.assertTrue(support.accepted)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_seam_fit_is_stable_for_half_pixel_hough_stem_center(self):
        frame = numpy.full((88, 88, 3), (80, 80, 80), dtype=numpy.uint8)
        frame[:30, :] = (190, 190, 190)
        cv2.fillPoly(
            frame,
            [numpy.array([(27, 10), (58, 18), (58, 64), (28, 64)], dtype=numpy.int32)],
            (0, 165, 51),
        )
        cv2.rectangle(frame, (43, 64), (48, 87), (0, 165, 51), thickness=cv2.FILLED)
        cv2.rectangle(frame, (35, 25), (53, 48), (235, 235, 235), thickness=cv2.FILLED)
        blurred_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )
        raw_edges = cv2.Canny(blurred_gray, 20, 60)
        localization_edges = cv2.dilate(
            raw_edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        localization_edges = cv2.morphologyEx(
            localization_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        candidates = []
        for stem_center_x in (44.0, 44.5):
            with patch(
                "scripts.aufgabe04.perception.stand_axis_image._stem_anchor_candidates_from_edges",
                return_value=[(stem_center_x, 66.0)],
            ):
                candidates.append(
                    _plain_face_from_stem_cropped_edges(
                        cv2,
                        localization_edges,
                        measurement_edges=raw_edges,
                        min_area_px=293.0,
                        min_edge_height_px=9.75,
                        min_aspect_ratio=0.45,
                        max_aspect_ratio=1.80,
                    )
                )

        integer_candidate, half_pixel_candidate = candidates
        self.assertIsNotNone(integer_candidate)
        self.assertIsNotNone(half_pixel_candidate)
        self.assertTrue(integer_candidate.rectangle_fit_reliable)
        self.assertTrue(half_pixel_candidate.rectangle_fit_reliable)
        for integer_corner, half_pixel_corner in zip(
            order_corners(integer_candidate.corners),
            order_corners(half_pixel_candidate.corners),
        ):
            self.assertAlmostEqual(integer_corner.u_px, half_pixel_corner.u_px, places=6)
            self.assertAlmostEqual(integer_corner.v_px, half_pixel_corner.v_px, places=6)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_ranked_stem_hypotheses_recover_when_seam_pair_outscores_stem(self):
        frame = numpy.full((88, 88, 3), (80, 80, 80), dtype=numpy.uint8)
        frame[:31, :] = (190, 190, 190)
        cv2.fillPoly(
            frame,
            [numpy.array([(25, 11), (56, 19), (56, 65), (26, 65)], dtype=numpy.int32)],
            (0, 165, 51),
        )
        cv2.rectangle(frame, (41, 65), (46, 87), (0, 165, 51), thickness=cv2.FILLED)
        cv2.rectangle(frame, (33, 26), (51, 49), (235, 235, 235), thickness=cv2.FILLED)

        blurred_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )
        raw_edges = cv2.Canny(blurred_gray, 20, 60)
        localization_edges = cv2.dilate(
            raw_edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        localization_edges = cv2.morphologyEx(
            localization_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        anchors = _stem_anchor_candidates_from_edges(
            cv2,
            localization_edges,
            min_edge_height_px=9.75,
        )
        self.assertLess(anchors[0][1], 40.0)
        self.assertTrue(any(top_y > 60.0 for _center_x, top_y in anchors[1:]))

        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            canny_low=20,
            canny_high=60,
            close_kernel=3,
            min_area_px=293.0,
            min_face_area_fraction=0.0,
            min_edge_height_px=9.75,
            silhouette_only=True,
            stand_width_m=0.078,
            stand_distance_m=0.55,
            camera_fx_px=381.36246688,
            camera_fy_px=381.36246688,
            camera_cx_px=44.0,
            camera_cy_px=87.0,
        )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.source, "edge_plain_face_stem_anchor")
        self.assertGreater(estimate.yaw_deg, 35.0)
        self.assertLess(estimate.yaw_deg, 60.0)
        support = _quadrilateral_edge_support(cv2, artifacts.face_mask, estimate.corners)
        self.assertTrue(support.accepted)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_wall_seam_cannot_synthesize_a_missing_head_bottom(self):
        raw_edges = numpy.zeros((87, 88), dtype=numpy.uint8)
        for start, end in (
            ((26, 12), (59, 19)),
            ((26, 12), (26, 63)),
            ((59, 19), (59, 63)),
            ((43, 63), (43, 86)),
            ((49, 63), (49, 86)),
            ((0, 29), (26, 29)),
            ((59, 29), (87, 29)),
        ):
            cv2.line(raw_edges, start, end, 255, thickness=2)
        localization_edges = cv2.dilate(
            raw_edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        localization_edges = cv2.morphologyEx(
            localization_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        candidate = _plain_face_from_stem_cropped_edges(
            cv2,
            localization_edges,
            measurement_edges=raw_edges,
            min_area_px=293.2,
            min_edge_height_px=9.7466,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        self.assertFalse(candidate.rectangle_fit_reliable)
        self.assertEqual(candidate.rectangle_fit_reason, "head_rectangle_fit_unreliable")
        support = _quadrilateral_edge_support(cv2, candidate.face_mask, candidate.corners)
        self.assertFalse(support.accepted)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_wall_seam_and_stem_without_head_stays_unavailable(self):
        raw_edges = numpy.zeros((87, 88), dtype=numpy.uint8)
        for start, end in (
            ((43, 63), (43, 86)),
            ((49, 63), (49, 86)),
            ((0, 29), (43, 29)),
            ((49, 29), (87, 29)),
        ):
            cv2.line(raw_edges, start, end, 255, thickness=2)
        localization_edges = cv2.dilate(
            raw_edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        localization_edges = cv2.morphologyEx(
            localization_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        candidate = _plain_face_from_stem_cropped_edges(
            cv2,
            localization_edges,
            measurement_edges=raw_edges,
            min_area_px=293.2,
            min_edge_height_px=9.7466,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNone(candidate)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_edge_exclusion_mask_removes_confirmed_wall_edges(self):
        frame = numpy.zeros((120, 160, 3), dtype=numpy.uint8)
        cv2.line(frame, (15, 70), (145, 70), (255, 255, 255), 2)
        exclusion = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
        exclusion[60:81, :] = 255

        _estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            blur_kernel=1,
            canny_low=20,
            canny_high=60,
            dilate_iterations=0,
            close_kernel=1,
            close_iterations=0,
            edge_exclusion_mask=exclusion,
        )

        self.assertEqual(int(numpy.count_nonzero(artifacts.edges[60:81, :])), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_standalone_simulation_pipeline_edges_full_frame_then_crops_head(self):
        frame = numpy.full((480, 640, 3), (55, 55, 55), dtype=numpy.uint8)
        cv2.line(frame, (0, 78), (639, 120), (90, 90, 90), thickness=3)
        cv2.line(frame, (0, 250), (639, 238), (90, 90, 90), thickness=3)
        head = numpy.array(
            [[(155, 86), (254, 96), (248, 198), (151, 188)]],
            dtype=numpy.int32,
        )
        cv2.fillPoly(frame, head, (0, 165, 51))
        cv2.rectangle(frame, (194, 190), (211, 300), (0, 165, 51), thickness=cv2.FILLED)
        cv2.rectangle(frame, (174, 115), (229, 171), (235, 235, 235), thickness=cv2.FILLED)

        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            canny_low=20,
            canny_high=60,
            min_face_area_fraction=0.0,
            silhouette_only=True,
        )
        roi = _detected_head_roi(
            estimate,
            frame_width=640,
            frame_height=480,
            padding_scale=1.4,
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(artifacts.edges.shape, (480, 640))
        self.assertIsNotNone(roi)
        self.assertLess(roi.x0, 155)
        self.assertGreater(roi.x1, 254)
        self.assertLess(roi.y0, 86)
        self.assertLess(roi.y1, 260)
        head_cutout = _diagnostic_roi_image(artifacts.face_mask, roi)
        rectangle_cutout = _diagnostic_roi_image(artifacts.rectangle_mask, roi)
        self.assertGreater(cv2.countNonZero(head_cutout), 0)
        self.assertGreater(cv2.countNonZero(rectangle_cutout), 0)
        # The diagonal arena wall may remain in the full edge image, but it
        # must not survive in the selected head cutout.
        self.assertEqual(cv2.countNonZero(artifacts.face_mask[:, 400:]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_standalone_pipeline_keeps_full_head_across_wall_mask_side_gap(self):
        frame = numpy.full((190, 220, 3), (55, 55, 55), dtype=numpy.uint8)
        actual = self.make_corners(
            [(60, 30), (110, 62), (110, 150), (60, 142)]
        )
        cv2.fillPoly(
            frame,
            numpy.array(
                [[(point.u_px, point.v_px) for point in actual]],
                dtype=numpy.int32,
            ),
            (0, 165, 51),
        )
        cv2.rectangle(frame, (76, 72), (101, 127), (235, 235, 235), cv2.FILLED)
        cv2.rectangle(frame, (79, 144), (91, 189), (0, 165, 51), cv2.FILLED)
        wall_exclusion = numpy.zeros(frame.shape[:2], dtype=numpy.uint8)
        wall_exclusion[100:125, 56:65] = 255

        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="channel_union",
            blur_kernel=5,
            canny_low=20,
            canny_high=60,
            dilate_iterations=0,
            close_kernel=3,
            close_iterations=1,
            min_face_area_fraction=0.0,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            silhouette_only=True,
            edge_exclusion_mask=wall_exclusion,
        )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertIsNotNone(artifacts.face_mask)
        for fitted, expected in zip(order_corners(estimate.corners), actual):
            self.assertAlmostEqual(fitted.u_px, expected.u_px, delta=2.0)
            self.assertAlmostEqual(fitted.v_px, expected.v_px, delta=3.0)
        self.assertEqual(
            cv2.countNonZero(artifacts.face_mask[103:122, 57:64]),
            0,
        )

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_color_agnostic_hybrid_localizes_connected_edges_but_fits_raw_edges(self):
        raw_edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        cv2.line(raw_edges, (64, 30), (161, 24), 255, thickness=2)
        cv2.line(raw_edges, (60, 35), (67, 116), 255, thickness=2)
        cv2.line(raw_edges, (165, 29), (159, 122), 255, thickness=2)
        cv2.line(raw_edges, (68, 120), (102, 122), 255, thickness=2)
        cv2.line(raw_edges, (125, 124), (158, 126), 255, thickness=2)
        cv2.line(raw_edges, (103, 122), (103, 175), 255, thickness=2)
        cv2.line(raw_edges, (124, 124), (124, 175), 255, thickness=2)
        cv2.rectangle(raw_edges, (86, 55), (140, 100), 255, thickness=2)
        cv2.line(raw_edges, (0, 5), (229, 8), 255, thickness=2)
        localization_edges = cv2.dilate(
            raw_edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        localization_edges = cv2.morphologyEx(
            localization_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        raw_topology = raw_edges.copy()
        localization_topology = localization_edges.copy()
        raw_topology[50:106, 81:146] = 0
        localization_topology[50:106, 81:146] = 0
        raw_components = cv2.connectedComponents(
            raw_topology[20:180, 45:180]
        )[0] - 1
        localization_components = cv2.connectedComponents(
            localization_topology[20:180, 45:180]
        )[0] - 1

        with patch(
            "scripts.aufgabe04.perception.stand_axis_image."
            "_connected_border_mask_and_corners",
            side_effect=AssertionError(
                "independent raw-edge measurement must not use a connected border"
            ),
        ):
            candidate = _plain_face_from_stem_cropped_edges(
                cv2,
                localization_edges,
                measurement_edges=raw_edges,
                min_area_px=500.0,
                min_edge_height_px=8.0,
                min_aspect_ratio=0.45,
                max_aspect_ratio=1.80,
            )

        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.rectangle_fit_reliable, candidate.rectangle_fit_reason)
        self.assertEqual(
            candidate.rectangle_fit_reason,
            "refitted_rectangle_edge_supported",
        )
        self.assertGreater(raw_components, localization_components)
        self.assertEqual(localization_components, 1)
        self.assertGreater(
            cv2.countNonZero(
                cv2.bitwise_and(
                    localization_edges,
                    cv2.bitwise_not(raw_edges),
                )
            ),
            0,
        )
        self.assertEqual(
            cv2.countNonZero(cv2.bitwise_and(candidate.face_mask, cv2.bitwise_not(raw_edges))),
            0,
        )
        self.assertGreater(cv2.countNonZero(candidate.face_mask[20:40, 55:170]), 0)
        self.assertGreater(cv2.countNonZero(candidate.face_mask[25:130, 55:75]), 0)
        self.assertGreater(cv2.countNonZero(candidate.face_mask[20:135, 150:175]), 0)
        self.assertGreater(cv2.countNonZero(candidate.face_mask[112:132, 60:170]), 0)
        self.assertEqual(cv2.countNonZero(candidate.face_mask[50:106, 81:146]), 0)
        self.assertEqual(cv2.countNonZero(candidate.face_mask[135:180, 95:132]), 0)
        self.assertEqual(cv2.countNonZero(candidate.face_mask[0:15, :]), 0)
        support = _quadrilateral_edge_support(cv2, candidate.face_mask, candidate.corners)
        self.assertTrue(support.accepted)
        expected_corners = self.make_corners(
            [(60, 30), (165, 24), (158, 126), (68, 120)]
        )
        for actual, expected in zip(order_corners(candidate.corners), expected_corners):
            self.assertAlmostEqual(actual.u_px, expected.u_px, delta=6.0)
            self.assertAlmostEqual(actual.v_px, expected.v_px, delta=6.0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_raw_side_refit_recovers_fragmented_outer_border_without_texture(self):
        raw_edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        expected = self.make_corners(
            [(60, 30), (165, 24), (158, 126), (68, 120)]
        )
        top_left, top_right, bottom_right, bottom_left = expected

        def draw_segment(start, end, interval_start, interval_end):
            start_xy = (
                round(start.u_px + (end.u_px - start.u_px) * interval_start),
                round(start.v_px + (end.v_px - start.v_px) * interval_start),
            )
            end_xy = (
                round(start.u_px + (end.u_px - start.u_px) * interval_end),
                round(start.v_px + (end.v_px - start.v_px) * interval_end),
            )
            cv2.line(raw_edges, start_xy, end_xy, 255, thickness=2)

        fragmented_sides = (
            (top_left, top_right, ((0.08, 0.32), (0.43, 0.68), (0.76, 0.92))),
            (top_right, bottom_right, ((0.08, 0.38), (0.52, 0.92))),
            (bottom_left, bottom_right, ((0.08, 0.35), (0.64, 0.92))),
            (top_left, bottom_left, ((0.08, 0.42), (0.55, 0.92))),
        )
        for start, end, intervals in fragmented_sides:
            for interval_start, interval_end in intervals:
                draw_segment(start, end, interval_start, interval_end)

        # Interior label/QR texture, an arena seam, and the stem are present in
        # raw Canny but must not become face-side evidence.
        cv2.rectangle(raw_edges, (86, 55), (140, 100), 255, thickness=2)
        cv2.line(raw_edges, (0, 10), (229, 10), 255, thickness=2)
        cv2.line(raw_edges, (103, 122), (103, 175), 255, thickness=2)
        cv2.line(raw_edges, (124, 124), (124, 175), 255, thickness=2)

        evidence_mask, refitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            expected,
        )

        self.assertIsNotNone(refitted)
        for actual, target in zip(order_corners(refitted), expected):
            self.assertAlmostEqual(actual.u_px, target.u_px, delta=5.0)
            self.assertAlmostEqual(actual.v_px, target.v_px, delta=3.0)
        refit_top_left, refit_top_right, refit_bottom_right, refit_bottom_left = (
            order_corners(refitted)
        )
        left_dx = refit_bottom_left.u_px - refit_top_left.u_px
        left_dy = refit_bottom_left.v_px - refit_top_left.v_px
        right_dx = refit_bottom_right.u_px - refit_top_right.u_px
        right_dy = refit_bottom_right.v_px - refit_top_right.v_px
        self.assertAlmostEqual(
            left_dx * right_dy - left_dy * right_dx,
            0.0,
            delta=0.05,
        )
        self.assertGreater(cv2.countNonZero(evidence_mask), 0)
        self.assertEqual(cv2.countNonZero(evidence_mask[50:106, 81:146]), 0)
        self.assertEqual(cv2.countNonZero(evidence_mask[:15, :]), 0)
        self.assertEqual(cv2.countNonZero(evidence_mask[135:180, 95:132]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_level_sim_camera_fits_outer_vertical_sides_not_inner_qr_edges(self):
        raw_edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        actual = self.make_corners(
            [(60, 30), (160, 45), (160, 145), (60, 130)]
        )
        rough = self.make_corners(
            [(64, 34), (156, 43), (156, 141), (64, 126)]
        )
        top_left, top_right, bottom_right, bottom_left = actual

        def draw_segment(start, end, interval_start, interval_end):
            start_xy = (
                round(start.u_px + (end.u_px - start.u_px) * interval_start),
                round(start.v_px + (end.v_px - start.v_px) * interval_start),
            )
            end_xy = (
                round(start.u_px + (end.u_px - start.u_px) * interval_end),
                round(start.v_px + (end.v_px - start.v_px) * interval_end),
            )
            cv2.line(raw_edges, start_xy, end_xy, 255, thickness=2)

        for start, end, intervals in (
            (top_left, top_right, ((0.08, 0.45), (0.55, 0.92))),
            (top_right, bottom_right, ((0.08, 0.48), (0.56, 0.92))),
            (bottom_left, bottom_right, ((0.08, 0.38), (0.62, 0.92))),
            (top_left, bottom_left, ((0.08, 0.46), (0.54, 0.92))),
        ):
            for interval_start, interval_end in intervals:
                draw_segment(start, end, interval_start, interval_end)

        # These inner verticals are deliberately just as close to the rough
        # proposal as the true silhouette sides. Outward selection must choose
        # x=60/x=160, not the QR frame at x=68/x=152.
        cv2.rectangle(raw_edges, (68, 55), (152, 119), 255, thickness=2)
        cv2.line(raw_edges, (104, 130), (104, 180), 255, thickness=2)
        cv2.line(raw_edges, (116, 132), (116, 180), 255, thickness=2)

        evidence_mask, refitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            rough,
            fixed_parallel_side_direction=(0.0, 1.0),
        )

        self.assertIsNotNone(refitted)
        fitted_top_left, fitted_top_right, fitted_bottom_right, fitted_bottom_left = (
            order_corners(refitted)
        )
        self.assertAlmostEqual(fitted_top_left.u_px, 60.0, delta=1.5)
        self.assertAlmostEqual(fitted_bottom_left.u_px, 60.0, delta=1.5)
        self.assertAlmostEqual(fitted_top_right.u_px, 160.0, delta=1.5)
        self.assertAlmostEqual(fitted_bottom_right.u_px, 160.0, delta=1.5)
        self.assertAlmostEqual(fitted_top_left.u_px, fitted_bottom_left.u_px, places=6)
        self.assertAlmostEqual(fitted_top_right.u_px, fitted_bottom_right.u_px, places=6)
        self.assertEqual(cv2.countNonZero(evidence_mask[58:116, 67:70]), 0)
        self.assertEqual(cv2.countNonZero(evidence_mask[58:116, 151:154]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_close_level_sim_camera_reaches_outer_sides_beyond_six_pixels(self):
        raw_edges = numpy.zeros((220, 240), dtype=numpy.uint8)
        actual = self.make_corners(
            [(40, 30), (180, 30), (180, 170), (40, 170)]
        )
        # At the closest validated simulation pose, the topology proposal is
        # the 53.147 mm white-panel boundary while the required rectangle is
        # the 69.930 mm board boundary. At 123 px head height this puts each
        # real outer side about 15 px beyond the historical 6 px band cap.
        rough_panel = self.make_corners(
            [(59, 49), (161, 49), (161, 151), (59, 151)]
        )
        cv2.rectangle(raw_edges, (40, 30), (180, 170), 255, thickness=2)
        cv2.rectangle(raw_edges, (57, 47), (163, 153), 255, thickness=2)
        cv2.line(raw_edges, (102, 170), (102, 215), 255, thickness=2)
        cv2.line(raw_edges, (118, 170), (118, 215), 255, thickness=2)

        evidence_mask, refitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            rough_panel,
            fixed_parallel_side_direction=(0.0, 1.0),
        )

        self.assertIsNotNone(refitted)
        for fitted, expected in zip(order_corners(refitted), actual):
            self.assertAlmostEqual(fitted.u_px, expected.u_px, delta=2.0)
            self.assertAlmostEqual(fitted.v_px, expected.v_px, delta=2.0)
        self.assertEqual(cv2.countNonZero(evidence_mask[58:143, 56:59]), 0)
        self.assertEqual(cv2.countNonZero(evidence_mask[58:143, 162:165]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_level_sim_camera_fits_outer_horizontal_sides_not_inner_frame_edges(self):
        raw_edges = numpy.zeros((180, 220), dtype=numpy.uint8)
        actual = self.make_corners(
            [(60, 30), (160, 30), (160, 130), (60, 130)]
        )
        rough = self.make_corners(
            [(64, 34), (156, 34), (156, 126), (64, 126)]
        )

        cv2.rectangle(raw_edges, (60, 30), (160, 130), 255, thickness=1)
        # Every inner frame side is closer to the rough proposal than the real
        # silhouette and remains inside the 3-6 px search band. Outward sampling
        # must still select y=30/y=130 instead of the inner y=37/y=123 edges.
        cv2.rectangle(raw_edges, (67, 37), (153, 123), 255, thickness=1)

        evidence_mask, refitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            rough,
            fixed_parallel_side_direction=(0.0, 1.0),
        )

        self.assertIsNotNone(refitted)
        fitted_top_left, fitted_top_right, fitted_bottom_right, fitted_bottom_left = (
            order_corners(refitted)
        )
        self.assertAlmostEqual(fitted_top_left.v_px, 30.0, delta=1.0)
        self.assertAlmostEqual(fitted_top_right.v_px, 30.0, delta=1.0)
        self.assertAlmostEqual(fitted_bottom_left.v_px, 130.0, delta=1.0)
        self.assertAlmostEqual(fitted_bottom_right.v_px, 130.0, delta=1.0)
        self.assertEqual(cv2.countNonZero(evidence_mask[36:39, 75:146]), 0)
        self.assertEqual(cv2.countNonZero(evidence_mask[122:125, 75:146]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_level_sim_camera_recovers_sloped_top_from_parallel_side_endpoints(self):
        raw_edges = numpy.zeros((190, 220), dtype=numpy.uint8)
        actual = self.make_corners(
            [(60, 30), (110, 62), (110, 150), (60, 142)]
        )
        cv2.polylines(
            raw_edges,
            numpy.array(
                [[(point.u_px, point.v_px) for point in actual]],
                dtype=numpy.int32,
            ),
            True,
            255,
            thickness=2,
        )
        cv2.rectangle(raw_edges, (70, 67), (102, 132), 255, thickness=2)

        # A row-envelope proposal localizes the right head but cannot express
        # its strongly sloped top. The two outer vertical sides still expose
        # the four real corner endpoints and must drive the final top/bottom
        # refit instead of rejecting the head at the 22-degree proposal gate.
        rough = self.make_corners(
            [(58, 42), (106, 42), (113, 145), (62, 145)]
        )
        evidence_mask, refitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            rough,
            fixed_parallel_side_direction=(0.0, 1.0),
        )

        self.assertIsNotNone(refitted)
        for fitted, expected in zip(order_corners(refitted), actual):
            self.assertAlmostEqual(fitted.u_px, expected.u_px, delta=2.0)
            self.assertAlmostEqual(fitted.v_px, expected.v_px, delta=3.0)
        self.assertEqual(cv2.countNonZero(evidence_mask[75:128, 73:100]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_level_sim_camera_bridges_wall_mask_gap_to_observed_side_corner(self):
        raw_edges = numpy.zeros((190, 220), dtype=numpy.uint8)
        actual = self.make_corners(
            [(60, 30), (110, 62), (110, 150), (60, 142)]
        )
        cv2.polylines(
            raw_edges,
            numpy.array(
                [[(point.u_px, point.v_px) for point in actual]],
                dtype=numpy.int32,
            ),
            True,
            255,
            thickness=2,
        )
        cv2.rectangle(raw_edges, (70, 67), (102, 132), 255, thickness=2)

        # The synchronized arena-wall mask can cross the near head side and
        # remove a short middle interval.  The real lower-left corner and the
        # bottom edge remain measured, so endpoint recovery must not truncate
        # the side at the upper fragment as the flickering live viewer did.
        raw_edges[100:125, 56:65] = 0
        rough = self.make_corners(
            [(58, 42), (106, 42), (113, 145), (62, 145)]
        )

        evidence_mask, refitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            rough,
            fixed_parallel_side_direction=(0.0, 1.0),
        )

        self.assertIsNotNone(refitted)
        for fitted, expected in zip(order_corners(refitted), actual):
            self.assertAlmostEqual(fitted.u_px, expected.u_px, delta=2.0)
            self.assertAlmostEqual(fitted.v_px, expected.v_px, delta=3.0)
        self.assertEqual(cv2.countNonZero(evidence_mask[103:122, 57:64]), 0)

    def test_level_camera_rejects_wall_seam_as_truncated_bottom_corner(self):
        self.assertTrue(
            _level_camera_endpoint_perspective_consistent(
                ImagePoint(297.0, 126.5),
                ImagePoint(297.0, 204.5),
                ImagePoint(339.0, 142.5),
                ImagePoint(339.0, 208.5),
            )
        )
        self.assertFalse(
            _level_camera_endpoint_perspective_consistent(
                ImagePoint(297.0, 127.5),
                ImagePoint(297.0, 180.5),
                ImagePoint(339.0, 142.5),
                ImagePoint(339.0, 209.5),
            )
        )

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_non_sim_camera_jointly_fits_one_tolerant_parallel_side_direction(self):
        raw_edges = numpy.zeros((180, 230), dtype=numpy.uint8)
        actual = numpy.array(
            [[(60, 30), (160, 22), (168, 122), (68, 130)]],
            dtype=numpy.int32,
        )
        rough = self.make_corners(
            [(62, 31), (158, 24), (166, 120), (70, 128)]
        )
        cv2.polylines(raw_edges, actual, True, 255, thickness=2)
        cv2.rectangle(raw_edges, (88, 52), (140, 104), 255, thickness=2)
        cv2.line(raw_edges, (108, 127), (108, 175), 255, thickness=2)
        cv2.line(raw_edges, (120, 126), (120, 175), 255, thickness=2)

        _evidence_mask, refitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            rough,
        )

        self.assertIsNotNone(refitted)
        top_left, top_right, bottom_right, bottom_left = order_corners(refitted)
        left_slope = (bottom_left.u_px - top_left.u_px) / (
            bottom_left.v_px - top_left.v_px
        )
        right_slope = (bottom_right.u_px - top_right.u_px) / (
            bottom_right.v_px - top_right.v_px
        )
        self.assertAlmostEqual(left_slope, right_slope, places=8)
        self.assertAlmostEqual(left_slope, 0.08, delta=0.02)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_color_agnostic_hybrid_rejects_missing_raw_head_boundary(self):
        localization_edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        head = numpy.array(
            [[(60, 30), (165, 24), (158, 126), (68, 120)]],
            dtype=numpy.int32,
        )
        cv2.polylines(localization_edges, head, True, 255, thickness=5)
        cv2.line(localization_edges, (103, 122), (103, 175), 255, thickness=5)
        cv2.line(localization_edges, (124, 124), (124, 175), 255, thickness=5)
        measurement_edges = numpy.zeros_like(localization_edges)
        cv2.line(measurement_edges, (64, 30), (161, 24), 255, thickness=2)
        cv2.line(measurement_edges, (60, 35), (67, 116), 255, thickness=2)
        cv2.line(measurement_edges, (165, 29), (159, 122), 255, thickness=2)
        cv2.rectangle(measurement_edges, (86, 55), (140, 100), 255, thickness=2)

        candidate = _plain_face_from_stem_cropped_edges(
            cv2,
            localization_edges,
            measurement_edges=measurement_edges,
            min_area_px=500.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        self.assertFalse(candidate.rectangle_fit_reliable)
        self.assertEqual(candidate.rectangle_fit_reason, "head_rectangle_fit_unreliable")
        support = _quadrilateral_edge_support(cv2, candidate.face_mask, candidate.corners)
        self.assertFalse(support.accepted)
        self.assertLess(support.bottom_left, 0.45)
        self.assertLess(support.bottom_right, 0.45)
        self.assertEqual(cv2.countNonZero(candidate.face_mask[112:132, 75:150]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_plain_face_measurement_edges_default_preserves_legacy_behavior(self):
        edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        cv2.polylines(
            edges,
            numpy.array([[(62, 35), (158, 28), (158, 122), (58, 118)]], dtype=numpy.int32),
            True,
            255,
            thickness=2,
        )
        cv2.line(edges, (100, 120), (100, 175), 255, thickness=2)
        cv2.line(edges, (115, 120), (115, 175), 255, thickness=2)
        options = dict(
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        implicit = _plain_face_from_stem_cropped_edges(cv2, edges, **options)
        explicit = _plain_face_from_stem_cropped_edges(
            cv2,
            edges,
            measurement_edges=None,
            **options,
        )

        self.assertIsNotNone(implicit)
        self.assertIsNotNone(explicit)
        self.assertEqual(implicit.corners, explicit.corners)
        self.assertEqual(implicit.rectangle_fit_reliable, explicit.rectangle_fit_reliable)
        self.assertTrue(numpy.array_equal(implicit.face_mask, explicit.face_mask))

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_plain_face_from_stem_ignores_internal_label(self):
        edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        cv2.polylines(
            edges,
            numpy.array([[(62, 35), (158, 28), (158, 122), (58, 118)]], dtype=numpy.int32),
            True,
            255,
            thickness=2,
        )
        cv2.rectangle(edges, (95, 65), (125, 75), 255, thickness=2)
        cv2.line(edges, (100, 120), (100, 175), 255, thickness=2)
        cv2.line(edges, (115, 120), (115, 175), 255, thickness=2)

        candidate = _plain_face_from_stem_cropped_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        x_values = [point.u_px for point in candidate.corners]
        y_values = [point.v_px for point in candidate.corners]
        self.assertLessEqual(min(x_values), 65)
        self.assertGreaterEqual(max(x_values), 155)
        self.assertLessEqual(min(y_values), 38)
        self.assertGreaterEqual(max(y_values), 116)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_plain_face_from_stem_uses_backside_outer_head_contour(self):
        edges = numpy.zeros((210, 260), dtype=numpy.uint8)
        outer = numpy.array([[(84, 36), (186, 50), (176, 146), (96, 132)]], dtype=numpy.int32)
        cv2.polylines(edges, outer, True, 255, thickness=2)
        cv2.rectangle(edges, (122, 82), (154, 94), 255, thickness=2)
        cv2.line(edges, (126, 137), (126, 196), 255, thickness=2)
        cv2.line(edges, (143, 140), (143, 196), 255, thickness=2)

        candidate = _plain_face_from_stem_cropped_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        top_left, top_right, bottom_right, bottom_left = candidate.corners
        self.assertLessEqual(top_left.u_px, 88)
        self.assertGreaterEqual(top_right.u_px, 182)
        self.assertLessEqual(min(point.u_px for point in candidate.corners), 88)
        self.assertGreaterEqual(max(point.u_px for point in candidate.corners), 182)
        self.assertLessEqual(min(point.v_px for point in candidate.corners), 38)
        self.assertGreaterEqual(max(point.v_px for point in candidate.corners), 140)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_border_uses_outer_points_not_internal_edges(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        outer = numpy.array([[(62, 30), (162, 24), (154, 124), (70, 118)]], dtype=numpy.int32)
        cv2.polylines(cutout, outer, True, 255, thickness=2)
        cv2.line(cutout, (82, 45), (145, 108), 255, thickness=2)
        cv2.rectangle(cutout, (95, 65), (126, 78), 255, thickness=2)

        face_mask, corners = _connected_border_mask_and_corners(
            cv2,
            cutout,
            cutout,
            fallback_corners=self.make_corners([(62, 30), (162, 24), (154, 124), (70, 118)]),
            min_edge_height_px=8.0,
        )

        top_left, top_right, bottom_right, bottom_left = corners
        self.assertLessEqual(top_left.u_px, 66)
        self.assertGreaterEqual(top_right.u_px, 158)
        self.assertLessEqual(bottom_left.u_px, 72)
        self.assertGreaterEqual(max(point.u_px for point in corners), 158)
        self.assertEqual(face_mask.shape, cutout.shape)
        self.assertEqual(int(face_mask[66, 96]), 255)
        self.assertEqual(int(face_mask[30, 64]), 255)
        self.assertEqual(int(face_mask[10, 64]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_border_debug_mask_keeps_visible_cuboid_edge(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        cv2.polylines(
            cutout,
            numpy.array([[(70, 42), (152, 36), (160, 124), (76, 126)]], dtype=numpy.int32),
            True,
            255,
            thickness=2,
        )
        cv2.line(cutout, (58, 52), (70, 42), 255, thickness=2)
        cv2.line(cutout, (58, 52), (62, 132), 255, thickness=2)
        cv2.line(cutout, (62, 132), (76, 126), 255, thickness=2)

        face_mask, corners = _connected_border_mask_and_corners(
            cv2,
            cutout,
            cutout,
            fallback_corners=self.make_corners([(70, 42), (152, 36), (160, 124), (76, 126)]),
            min_edge_height_px=8.0,
        )

        self.assertEqual(int(face_mask[52, 58]), 255)
        self.assertEqual(int(face_mask[42, 70]), 255)
        self.assertEqual(face_mask.shape, cutout.shape)
        self.assertEqual(len(corners), 4)
        self.assertLessEqual(min(point.u_px for point in corners), 60)
        self.assertGreaterEqual(max(point.u_px for point in corners), 158)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_border_fits_top_and_bottom_lines_around_stem_notch(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        cv2.line(cutout, (55, 35), (165, 30), 255, thickness=2)
        cv2.line(cutout, (55, 35), (58, 122), 255, thickness=2)
        cv2.line(cutout, (165, 30), (168, 118), 255, thickness=2)
        cv2.line(cutout, (58, 122), (102, 121), 255, thickness=2)
        cv2.line(cutout, (126, 120), (168, 118), 255, thickness=2)
        cv2.line(cutout, (102, 121), (102, 144), 255, thickness=2)
        cv2.line(cutout, (126, 120), (126, 144), 255, thickness=2)

        face_mask, corners = _connected_border_mask_and_corners(
            cv2,
            cutout,
            cutout,
            fallback_corners=self.make_corners([(55, 35), (165, 30), (168, 118), (58, 122)]),
            min_edge_height_px=8.0,
        )

        top_left, top_right, bottom_right, bottom_left = corners
        self.assertLess(top_right.v_px, bottom_right.v_px)
        self.assertLess(top_left.v_px, bottom_left.v_px)
        self.assertLessEqual(abs(bottom_left.v_px - 122), 4)
        self.assertLessEqual(abs(bottom_right.v_px - 118), 4)
        self.assertEqual(int(face_mask[144, 102]), 255)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_border_does_not_substitute_rough_corners_without_edges(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        rough = self.make_corners([(55, 35), (165, 30), (168, 118), (58, 122)])

        face_mask, corners = _connected_border_mask_and_corners(
            cv2,
            cutout,
            cutout,
            fallback_corners=rough,
            min_edge_height_px=8.0,
        )

        self.assertIs(face_mask, cutout)
        self.assertIsNone(corners)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_rectangle_overlay_shows_cutout_evidence_and_fitted_outline(self):
        face_mask = numpy.zeros((100, 120), dtype=numpy.uint8)
        corners = self.make_corners([(25, 20), (95, 20), (95, 75), (25, 75)])
        cv2.rectangle(face_mask, (45, 40), (55, 48), 255, thickness=1)

        overlay = _debug_rectangle_overlay_image(
            cv2,
            face_mask.shape,
            corners,
            face_mask,
        )

        self.assertTrue(numpy.any(overlay == 96))
        self.assertTrue(numpy.any(overlay == 255))
        self.assertEqual(int(overlay[40, 50]), 96)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_edge_estimator_returns_rectangle_debug_mask(self):
        frame = numpy.full((210, 260, 3), 255, dtype=numpy.uint8)
        cv2.rectangle(frame, (84, 36), (186, 146), (0, 0, 120), thickness=cv2.FILLED)
        cv2.rectangle(frame, (122, 82), (154, 94), (240, 240, 240), thickness=cv2.FILLED)
        cv2.rectangle(frame, (126, 137), (143, 196), (0, 0, 120), thickness=cv2.FILLED)

        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.source, "edge_plain_face_stem_anchor")
        self.assertIsNotNone(artifacts.face_mask)
        self.assertIsNotNone(artifacts.rectangle_mask)
        self.assertEqual(artifacts.rectangle_mask.shape, artifacts.edges.shape)
        self.assertGreater(cv2.countNonZero(artifacts.rectangle_mask), 0)
        self.assertIsNotNone(artifacts.rectangle_overlay)
        # A perfect fit can cover every thin raw Canny pixel with the brighter
        # rectangle outline; the separate face_mask retains the evidence.
        self.assertGreater(cv2.countNonZero(artifacts.face_mask), 0)
        self.assertTrue(numpy.any(artifacts.rectangle_overlay == 255))

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_only_uses_real_stem_anchored_head_cutout_without_qr_geometry(self):
        frame = numpy.full((210, 260, 3), 255, dtype=numpy.uint8)
        cv2.rectangle(frame, (84, 36), (186, 146), (0, 0, 120), thickness=cv2.FILLED)
        cv2.rectangle(frame, (110, 72), (146, 96), (240, 240, 240), thickness=cv2.FILLED)
        cv2.rectangle(frame, (126, 137), (143, 196), (0, 0, 120), thickness=cv2.FILLED)
        estimate, artifacts = estimate_stand_axis_from_edges(
            cv2, frame, edge_preprocess="gray", blur_kernel=1, min_area_px=250.0,
            silhouette_only=True,
        )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.source, "edge_plain_face_stem_anchor")
        self.assertIsNotNone(artifacts.face_mask)
        self.assertIsNotNone(artifacts.rectangle_mask)
        # Connected morphology localizes the head, but only untouched Canny
        # pixels near its outer boundary may support the fitted rectangle.
        raw_edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        self.assertEqual(
            cv2.countNonZero(cv2.bitwise_and(artifacts.face_mask, cv2.bitwise_not(raw_edges))),
            0,
        )
        self.assertGreater(
            cv2.countNonZero(
                cv2.bitwise_and(artifacts.edges, cv2.bitwise_not(raw_edges))
            ),
            0,
        )
        self.assertEqual(cv2.countNonZero(artifacts.face_mask[70:100, 105:151]), 0)
        self.assertEqual(cv2.countNonZero(artifacts.face_mask[165:205, 120:150]), 0)
        support = _quadrilateral_edge_support(cv2, artifacts.face_mask, estimate.corners)
        self.assertTrue(support.accepted)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_only_preserves_rejected_cutout_without_synthetic_fallback(self):
        frame = numpy.full((120, 160, 3), 255, dtype=numpy.uint8)
        face_mask = numpy.zeros((120, 160), dtype=numpy.uint8)
        corners = self.make_corners([(35, 20), (125, 20), (125, 85), (35, 85)])
        cv2.line(face_mask, (35, 20), (125, 20), 255, thickness=2)
        cv2.line(face_mask, (35, 20), (35, 85), 255, thickness=2)
        cv2.line(face_mask, (125, 20), (125, 85), 255, thickness=2)
        rejected = _SilhouetteFaceCandidate(
            corners=corners,
            face_mask=face_mask,
            rectangle_fit_reliable=False,
            rectangle_fit_reason="head_rectangle_fit_unreliable",
        )

        with patch(
            "scripts.aufgabe04.perception.stand_axis_image._plain_face_from_stem_cropped_edges",
            return_value=rejected,
        ), patch(
            "scripts.aufgabe04.perception.stand_axis_image._stem_anchored_face_from_edges"
        ) as synthetic_stem, patch(
            "scripts.aufgabe04.perception.stand_axis_image._face_quadrilateral_from_silhouette"
        ) as synthetic_silhouette:
            estimate, artifacts = estimate_stand_axis_from_edges(
                cv2,
                frame,
                edge_preprocess="gray",
                silhouette_only=True,
            )

        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "head_rectangle_fit_unreliable")
        self.assertEqual(estimate.corners, corners)
        self.assertIs(artifacts.face_mask, face_mask)
        self.assertIsNone(artifacts.rectangle_mask)
        self.assertIsNone(artifacts.rectangle_overlay)
        synthetic_stem.assert_not_called()
        synthetic_silhouette.assert_not_called()

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_edge_pixels_inside_polygon_uses_debug_margin_only(self):
        edges = numpy.zeros((100, 120), dtype=numpy.uint8)
        cv2.rectangle(edges, (30, 25), (90, 75), 255, thickness=1)
        cv2.line(edges, (27, 25), (27, 75), 255, thickness=1)
        corners = self.make_corners([(30, 25), (90, 25), (90, 75), (30, 75)])

        tight = _edge_pixels_inside_polygon(cv2, edges, corners, margin_px=0)
        expanded = _edge_pixels_inside_polygon(cv2, edges, corners, margin_px=4)

        self.assertEqual(int(tight[50, 27]), 0)
        self.assertEqual(int(expanded[50, 27]), 255)
        self.assertEqual(int(expanded[10, 27]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_expanded_head_roi_keeps_slanted_top_outside_rough_polygon(self):
        edges = numpy.zeros((140, 180), dtype=numpy.uint8)
        cv2.line(edges, (48, 18), (132, 34), 255, thickness=2)
        cv2.line(edges, (48, 18), (52, 105), 255, thickness=2)
        cv2.line(edges, (132, 34), (136, 112), 255, thickness=2)
        cv2.line(edges, (52, 105), (136, 112), 255, thickness=2)
        rough = self.make_corners([(54, 38), (128, 42), (132, 108), (56, 102)])

        edge_roi = _expanded_head_edge_roi(
            cv2,
            edges,
            rough,
            margin_px=10,
            stem_center_x=92.0,
            stem_top_y=108.0,
            min_edge_height_px=8.0,
        )

        self.assertEqual(int(edge_roi[18, 48]), 255)
        self.assertEqual(int(edge_roi[34, 132]), 255)
        self.assertEqual(int(edge_roi[10, 48]), 0)

    def test_front_facing_square_has_neutral_ratio(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(40, 25), (100, 25), (100, 85), (40, 85)])
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.mode, "face_visible")
        self.assertAlmostEqual(estimate.height_ratio, 1.0, delta=0.03)
        self.assertAlmostEqual(estimate.yaw_proxy, 0.0, delta=0.02)
        self.assertEqual(estimate.closer_side, "equal")

    def test_taller_left_edge_reports_left_as_closer(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(35, 20), (105, 35), (100, 80), (35, 95)])
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.mode, "face_visible")
        self.assertGreater(estimate.left_height_px, estimate.right_height_px)
        self.assertGreater(estimate.height_ratio, 1.0)
        self.assertGreater(estimate.yaw_proxy, 0.0)
        self.assertEqual(estimate.closer_side, "left")

    def test_edge_on_line_reports_side_on_without_ratio(self):
        estimate = estimate_edge_on_axis_from_line(
            ImagePoint(80, 20),
            ImagePoint(82, 92),
            min_edge_height_px=8.0,
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.mode, "edge_on")
        self.assertEqual(estimate.reason, "edge_on_approx_90_deg")
        self.assertIsNone(estimate.height_ratio)
        self.assertIsNone(estimate.yaw_proxy)
        self.assertIsNone(estimate.yaw_deg)
        self.assertEqual(estimate.closer_side, "side_on")
        self.assertIsNotNone(estimate.axis_line)

    def test_short_edge_on_line_is_not_usable(self):
        estimate = estimate_edge_on_axis_from_line(
            ImagePoint(80, 20),
            ImagePoint(80, 24),
            min_edge_height_px=8.0,
        )

        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.mode, "unavailable")
        self.assertEqual(estimate.reason, "edge_on_line_too_short")

    def test_optional_geometry_converts_ratio_to_yaw_degrees(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(35, 20), (105, 35), (100, 80), (35, 95)]),
            stand_width_m=1.0,
            stand_distance_m=0.25,
        )

        self.assertTrue(estimate.usable)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertGreater(estimate.yaw_deg, 0.0)

    def test_calibrated_geometry_uses_lidar_distance_and_known_face_size(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(40, 20), (79, 20), (79, 98), (40, 98)]),
            stand_width_m=0.078,
            stand_distance_m=1.0,
            camera_fx_px=1000.0,
        )

        self.assertTrue(estimate.usable)
        self.assertAlmostEqual(estimate.height_ratio, 1.0, delta=0.001)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertAlmostEqual(abs(estimate.yaw_deg), 60.0, delta=0.5)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for PnP tests")
    def test_square_pnp_uses_camera_intrinsics_with_public_yaw_handedness(self):
        face_size_m = 0.078
        object_points = numpy.array(
            [
                [-face_size_m / 2.0, -face_size_m / 2.0, 0.0],
                [face_size_m / 2.0, -face_size_m / 2.0, 0.0],
                [face_size_m / 2.0, face_size_m / 2.0, 0.0],
                [-face_size_m / 2.0, face_size_m / 2.0, 0.0],
            ],
            dtype=numpy.float64,
        )
        camera_matrix = numpy.array([[530.0, 0.0, 320.0], [0.0, 530.0, 240.0], [0.0, 0.0, 1.0]])
        for optical_yaw_deg in (-25.0, 25.0):
            with self.subTest(optical_yaw_deg=optical_yaw_deg):
                optical_yaw_rad = numpy.deg2rad(optical_yaw_deg)
                rvec, _jacobian = cv2.Rodrigues(
                    numpy.array(
                        [
                            [
                                numpy.cos(optical_yaw_rad),
                                0.0,
                                numpy.sin(optical_yaw_rad),
                            ],
                            [0.0, 1.0, 0.0],
                            [
                                -numpy.sin(optical_yaw_rad),
                                0.0,
                                numpy.cos(optical_yaw_rad),
                            ],
                        ],
                        dtype=numpy.float64,
                    )
                )
                tvec = numpy.array([[0.0], [0.0], [0.55]], dtype=numpy.float64)
                image_points, _jacobian = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    numpy.zeros((4, 1), dtype=numpy.float64),
                )
                corners = self.make_corners(
                    [
                        (float(point[0][0]), float(point[0][1]))
                        for point in image_points
                    ]
                )

                estimate = estimate_stand_axis_from_corners(
                    corners,
                    stand_width_m=face_size_m,
                    camera_fx_px=530.0,
                    camera_fy_px=530.0,
                    camera_cx_px=320.0,
                    camera_cy_px=240.0,
                    cv2=cv2,
                )

                self.assertTrue(estimate.usable)
                self.assertIsNotNone(estimate.yaw_deg)
                # OpenCV optical +x points image-right, whereas the public
                # stand-axis yaw convention is positive image-left / ROS CCW.
                self.assertAlmostEqual(
                    estimate.yaw_deg,
                    -optical_yaw_deg,
                    delta=1.0,
                )

    def test_stand_axis_viewer_requires_ros_compressed_image_topic(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

        args = parser.parse_args(["--compressed-image-topic", "/camera/image_raw/compressed"])
        self.assertEqual(args.compressed_image_topic, "/camera/image_raw/compressed")
        self.assertEqual(args.axis_source, "edges")
        self.assertEqual(args.diagnostic_window_size_px, 320)
        self.assertAlmostEqual(args.head_hold_sec, 0.35)

    def test_stand_axis_viewer_has_no_motion_arguments(self):
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--cmd-vel-topic", option_strings)
        self.assertNotIn("--run", option_strings)
        self.assertNotIn("--nav2-goal", option_strings)

    def test_stand_axis_viewer_applies_sim_lidar_extrinsic_to_wall_mask(self):
        viewer_path = (
            ROOT
            / "scripts"
            / "aufgabe04"
            / "perception"
            / "debug"
            / "stand_axis_viewer.py"
        )
        viewer_tree = ast.parse(viewer_path.read_text(encoding="utf-8"))
        wall_mask_calls = [
            node
            for node in ast.walk(viewer_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_confirmed_wall_exclusion_mask"
        ]

        self.assertEqual(len(wall_mask_calls), 1)
        keyword_values = {
            keyword.arg: keyword.value
            for keyword in wall_mask_calls[0].keywords
        }
        offset_value = keyword_values["lidar_forward_offset_m"]
        self.assertIsInstance(offset_value, ast.Attribute)
        self.assertEqual(offset_value.attr, "sim_lidar_forward_offset_m")
        self.assertIsInstance(offset_value.value, ast.Name)
        self.assertEqual(offset_value.value.id, "args")

    def test_stand_axis_viewer_exposes_edge_debug_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--axis-source",
                "edges",
                "--display-edges",
                "--canny-low",
                "40",
                "--canny-high",
                "120",
                "--edge-preprocess",
                "outer-border",
                "--hough-threshold",
                "18",
                "--hough-min-line-length-px",
                "10",
                "--hough-max-line-gap-px",
                "6",
                "--min-boundary-line-length-px",
                "42",
                "--face-width-fraction",
                "0.65",
                "--min-face-area-fraction",
                "0.35",
                "--display-face-mask",
            ]
        )

        self.assertEqual(args.axis_source, "edges")
        self.assertTrue(args.display_edges)
        self.assertTrue(args.display_face_mask)
        self.assertFalse(args.display_rectangle_mask)
        self.assertFalse(args.display_mask)
        self.assertEqual(args.canny_low, 40)
        self.assertEqual(args.canny_high, 120)
        self.assertEqual(args.edge_preprocess, "outer-border")
        self.assertEqual(args.hough_threshold, 18)
        self.assertEqual(args.hough_min_line_length_px, 10)
        self.assertEqual(args.hough_max_line_gap_px, 6)
        self.assertEqual(args.min_boundary_line_length_px, 42)
        self.assertAlmostEqual(args.face_width_fraction, 0.65)
        self.assertAlmostEqual(args.min_face_area_fraction, 0.35)

    def test_stand_axis_viewer_rectangle_window_is_independent(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--display-rectangle-mask",
            ]
        )

        self.assertTrue(args.display_rectangle_mask)
        self.assertFalse(args.display_face_mask)

    def test_stand_axis_viewer_preserves_native_roi_window_shape(self):
        class FakeCv2:
            WINDOW_AUTOSIZE = 1
            WINDOW_NORMAL = 2

            def __init__(self):
                self.named_windows = []
                self.resized_windows = []

            def namedWindow(self, name, mode):
                self.named_windows.append((name, mode))

            def resizeWindow(self, name, width, height):
                self.resized_windows.append((name, width, height))

        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--display-mask",
                "--display-edges",
                "--display-face-mask",
                "--display-rectangle-mask",
                "--diagnostic-window-size-px",
                "360",
            ]
        )
        fake_cv2 = FakeCv2()

        _initialize_display_windows(fake_cv2, args)

        self.assertEqual(
            fake_cv2.named_windows,
            [
                (WINDOW_FRAME, fake_cv2.WINDOW_AUTOSIZE),
                (WINDOW_MASK, fake_cv2.WINDOW_NORMAL),
                (WINDOW_EDGES, fake_cv2.WINDOW_NORMAL),
                (WINDOW_FACE_MASK, fake_cv2.WINDOW_AUTOSIZE),
                (WINDOW_RECTANGLE_MASK, fake_cv2.WINDOW_AUTOSIZE),
            ],
        )
        self.assertEqual(fake_cv2.resized_windows, [])

        size = _resize_diagnostic_windows(fake_cv2, args, (180, 300))

        self.assertEqual(size, (300, 180))
        self.assertEqual(
            fake_cv2.resized_windows,
            [
                (WINDOW_MASK, 300, 180),
                (WINDOW_EDGES, 300, 180),
            ],
        )

    def test_stand_axis_viewer_caps_large_roi_without_distortion(self):
        class FakeCv2:
            def __init__(self):
                self.resized_windows = []

            def resizeWindow(self, name, width, height):
                self.resized_windows.append((name, width, height))

        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--display-edges",
                "--display-face-mask",
                "--diagnostic-window-size-px",
                "320",
            ]
        )
        fake_cv2 = FakeCv2()

        size = _resize_diagnostic_windows(fake_cv2, args, (480, 640))

        self.assertEqual(size, (320, 240))
        self.assertEqual(
            fake_cv2.resized_windows,
            [
                (WINDOW_EDGES, 320, 240),
            ],
        )

    def test_stand_axis_viewer_exposes_nonblocking_qr_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--qr-decode-fps",
                "1.5",
                "--qr-result-ttl-sec",
                "0.75",
                "--front-face-to-qr-width-ratio",
                "1.2",
                "--stand-face-size-m",
                "0.078",
                "--camera-fx-px",
                "530.0",
                "--camera-fy-px",
                "531.0",
                "--camera-cx-px",
                "320.0",
                "--camera-cy-px",
                "240.0",
                "--camera-fx-is-full-resolution",
                "--stand-head-depth-m",
                "0.007",
                "--stand-head-bottom-height-m",
                "0.135",
                "--scan-topic",
                "/scan",
                "--use-lidar-distance",
                "--lidar-bearing-rad",
                "0.1",
                "--lidar-cone-deg",
                "7.5",
                "--max-scan-age-sec",
                "0.8",
                "--no-qr-decode",
            ]
        )

        self.assertAlmostEqual(args.qr_decode_fps, 1.5)
        self.assertAlmostEqual(args.qr_result_ttl_sec, 0.75)
        self.assertAlmostEqual(args.front_face_to_qr_width_ratio, 1.2)
        self.assertAlmostEqual(args.stand_face_size_m, 0.078)
        self.assertAlmostEqual(args.camera_fx_px, 530.0)
        self.assertAlmostEqual(args.camera_fy_px, 531.0)
        self.assertAlmostEqual(args.camera_cx_px, 320.0)
        self.assertAlmostEqual(args.camera_cy_px, 240.0)
        self.assertTrue(args.camera_fx_is_full_resolution)
        self.assertAlmostEqual(args.stand_head_depth_m, 0.007)
        self.assertAlmostEqual(args.stand_head_bottom_height_m, 0.135)
        self.assertEqual(args.scan_topic, "/scan")
        self.assertTrue(args.use_lidar_distance)
        self.assertAlmostEqual(args.lidar_bearing_rad, 0.1)
        self.assertAlmostEqual(args.lidar_cone_deg, 7.5)
        self.assertAlmostEqual(args.max_scan_age_sec, 0.8)
        self.assertAlmostEqual(args.sim_lidar_forward_offset_m, -0.032)
        self.assertTrue(args.no_qr_decode)


if __name__ == "__main__":
    unittest.main()
