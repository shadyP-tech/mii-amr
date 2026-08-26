from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from scripts.aufgabe04.perception import stand_axis_image
from scripts.aufgabe04.perception.stand_axis import (
    geometry,
    models,
    preprocessing,
    temporal_geometry,
)
from scripts.aufgabe04.perception.stand_axis import raw_support, stem_candidates
from scripts.aufgabe04.perception.stand_axis import (
    model_profile,
    model_pipeline,
    model_projection,
    model_refinement,
    pose_tracking,
    qr_pose_seed,
)


class StandAxisFacadeTest(unittest.TestCase):
    def test_extracted_symbols_keep_the_legacy_import_surface(self):
        expected_aliases = {
            "ImagePoint": models.ImagePoint,
            "StandAxisEdgeDebugArtifacts": models.StandAxisEdgeDebugArtifacts,
            "StandAxisImageEstimate": models.StandAxisImageEstimate,
            "_SilhouetteFaceCandidate": models._SilhouetteFaceCandidate,
            "estimate_edge_on_axis_from_line": (
                geometry.estimate_edge_on_axis_from_line
            ),
            "estimate_stand_axis_from_corners": (
                geometry.estimate_stand_axis_from_corners
            ),
            "order_corners": geometry.order_corners,
            "quadrilateral_aspect_ratio": geometry.quadrilateral_aspect_ratio,
            "wide_row_band": geometry.wide_row_band,
            "_debug_rectangle_overlay_image": (
                geometry._debug_rectangle_overlay_image
            ),
            "_largest_qr_quad": geometry._largest_qr_quad,
            "_scale_quadrilateral_about_center": (
                geometry._scale_quadrilateral_about_center
            ),
            "_canny_edges_from_frame": preprocessing._canny_edges_from_frame,
            "_edge_topology_hypotheses": (
                preprocessing._edge_topology_hypotheses
            ),
            "_topology_supported_measurement_edges": (
                preprocessing._topology_supported_measurement_edges
            ),
            "_quadrilateral_edge_support": (
                raw_support._quadrilateral_edge_support
            ),
            "_raw_side_evidence_and_corners": (
                raw_support._raw_side_evidence_and_corners
            ),
            "_select_supported_head_corners": (
                raw_support._select_supported_head_corners
            ),
            "_validated_refitted_head_corners": (
                raw_support._validated_refitted_head_corners
            ),
            "_connected_border_mask_and_corners": (
                stem_candidates._connected_border_mask_and_corners
            ),
            "_expanded_head_edge_roi": (
                stem_candidates._expanded_head_edge_roi
            ),
            "_stem_anchor_candidates_from_edges": (
                stem_candidates._stem_anchor_candidates_from_edges
            ),
            "_stem_owned_head_from_line_segments": (
                stem_candidates._stem_owned_head_from_line_segments
            ),
        }

        for name, extracted_symbol in expected_aliases.items():
            with self.subTest(name=name):
                self.assertIs(getattr(stand_axis_image, name), extracted_symbol)

    def test_all_preexisting_direct_imports_remain_available(self):
        direct_import_surface = {
            "ImagePoint",
            "StandAxisEdgeDebugArtifacts",
            "StandAxisImageEstimate",
            "_SilhouetteFaceCandidate",
            "_connected_border_mask_and_corners",
            "_debug_rectangle_overlay_image",
            "_edge_pixels_inside_polygon",
            "_expanded_head_edge_roi",
            "_face_quadrilateral_from_silhouette",
            "_level_camera_endpoint_perspective_consistent",
            "_largest_qr_quad",
            "_plain_face_from_stem_cropped_edges",
            "_quadrilateral_edge_support",
            "_raw_side_evidence_and_corners",
            "_scale_quadrilateral_about_center",
            "_select_supported_head_corners",
            "_stem_anchor_from_edges",
            "_stem_anchor_candidates_from_edges",
            "_stem_anchored_face_from_edges",
            "_stem_owned_head_from_line_segments",
            "_topology_supported_measurement_edges",
            "_validated_refitted_head_corners",
            "estimate_edge_on_axis_from_line",
            "estimate_stand_axis_from_corners",
            "estimate_stand_axis_from_edges",
            "estimate_stand_axis_from_mask",
            "order_corners",
            "quadrilateral_aspect_ratio",
            "wide_row_band",
        }

        missing = sorted(
            name
            for name in direct_import_surface
            if not hasattr(stand_axis_image, name)
        )
        self.assertEqual(missing, [])

    def test_metric_model_is_not_exported_through_legacy_detector_facade(self):
        self.assertFalse(
            hasattr(stand_axis_image, "estimate_stand_axis_from_metric_model")
        )

    def test_parallel_side_direction_is_keyword_only_with_legacy_default(self):
        parameter = inspect.signature(
            stand_axis_image.estimate_stand_axis_from_edges
        ).parameters["parallel_side_direction"]

        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameter.default, (0.0, 1.0))

    def test_extracted_modules_do_not_import_the_legacy_facade(self):
        for module in (
            geometry,
            models,
            preprocessing,
            raw_support,
            stem_candidates,
            temporal_geometry,
            model_profile,
            model_pipeline,
            model_projection,
            model_refinement,
            pose_tracking,
            qr_pose_seed,
        ):
            with self.subTest(module=module.__name__):
                source = Path(module.__file__).read_text(encoding="utf-8")
                self.assertNotIn("stand_axis_image", source)


if __name__ == "__main__":
    unittest.main()
