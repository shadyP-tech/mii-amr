import math
import unittest
import inspect

from scripts.aufgabe04.simulation.sim_head_roi import (
    project_target_to_camera,
    qr_corners_inside_roi,
    silhouette_close_kernel,
    silhouette_min_edge_height_px,
    stand_head_roi,
)
from scripts.aufgabe04.simulation.sim_synchronized_viewpoint_node import (
    SimSynchronizedViewpointNode,
)


class SimHeadRoiTest(unittest.TestCase):
    def test_close_kernel_is_capped_before_qr_texture_becomes_filled(self):
        self.assertEqual(silhouette_close_kernel(16.0), 3)
        self.assertEqual(silhouette_close_kernel(90.0), 5)
        self.assertEqual(silhouette_close_kernel(133.0), 7)
        self.assertEqual(silhouette_close_kernel(300.0), 7)

    def test_stem_edge_threshold_scales_but_is_capped_for_close_heads(self):
        self.assertAlmostEqual(silhouette_min_edge_height_px(33.0), 5.94)
        self.assertAlmostEqual(silhouette_min_edge_height_px(50.0), 9.0)
        self.assertAlmostEqual(silhouette_min_edge_height_px(99.2), 12.0)

    def test_failed_live_bundle_geometry_projects_the_complete_head(self):
        projection = project_target_to_camera(
            robot_x_m=-0.1880514912,
            robot_y_m=-0.1980467816,
            robot_z_m=0.008547,
            robot_yaw_rad=-2.3349926242,
            target_x_m=-0.395,
            target_y_m=-0.415,
            target_height_m=0.165035,
            camera_forward_offset_m=0.076,
            camera_lateral_offset_m=0.0,
            camera_height_m=0.093,
        )
        roi = stand_head_roi(
            frame_width=640,
            frame_height=480,
            bearing_rad=projection.bearing_rad,
            distance_m=0.3292365968,
            camera_fx_px=381.36246688,
            camera_fy_px=381.36246688,
            camera_cx_px=320.5,
            camera_cy_px=240.5,
            stand_face_size_m=0.078,
            camera_depth_m=projection.depth_m,
            target_height_delta_m=projection.height_delta_m,
        )

        self.assertAlmostEqual(projection.depth_m, 0.2238, delta=0.003)
        self.assertAlmostEqual(roi.expected_head_px, 133.0, delta=2.0)
        projected_center_y = (roi.y0 + roi.y1) / 2.0
        self.assertLess(projected_center_y, 150.0)
        # The failed crop began at row 141 and cut off the rendered head,
        # which occupied approximately rows 67..189.  The corrected crop
        # contains that complete interval with margin.
        self.assertLessEqual(roi.y0, 67)
        self.assertGreaterEqual(roi.y1, 189)

    def test_camera_projection_accounts_for_forward_sensor_baseline(self):
        projection = project_target_to_camera(
            robot_x_m=0.0,
            robot_y_m=0.0,
            robot_z_m=0.0,
            robot_yaw_rad=0.0,
            target_x_m=0.30,
            target_y_m=0.0,
            target_height_m=0.165,
            camera_forward_offset_m=0.076,
            camera_lateral_offset_m=0.0,
            camera_height_m=0.093,
        )

        self.assertAlmostEqual(projection.depth_m, 0.224, places=3)
        self.assertAlmostEqual(projection.bearing_rad, 0.0, places=6)
        self.assertAlmostEqual(projection.height_delta_m, 0.072, places=3)

    def test_camera_projection_maps_robot_right_to_increasing_image_columns(self):
        projection = project_target_to_camera(
            robot_x_m=-0.10326933698754208,
            robot_y_m=-0.1750773904064177,
            robot_z_m=0.00850170436169892,
            robot_yaw_rad=-2.3643049695857425,
            target_x_m=-0.395,
            target_y_m=-0.415,
            target_height_m=0.165035,
            camera_forward_offset_m=0.076,
            camera_lateral_offset_m=0.0,
            camera_height_m=0.093,
        )
        roi = stand_head_roi(
            frame_width=640,
            frame_height=480,
            bearing_rad=projection.bearing_rad,
            distance_m=0.39968204498291016,
            camera_fx_px=381.36246688,
            camera_fy_px=381.36246688,
            camera_cx_px=320.5,
            camera_cy_px=240.5,
            stand_face_size_m=0.078,
            camera_depth_m=projection.depth_m,
            target_height_delta_m=projection.height_delta_m,
            padding_scale=1.6,
        )

        # This is the geometry captured in sim_sync_control_strategy_005: the
        # stand was right of the optical axis at about image column 363.  The
        # old sign projected it to column 278 and clipped the head from the ROI.
        self.assertGreater(projection.bearing_rad, 0.0)
        self.assertAlmostEqual(math.degrees(projection.bearing_rad), 6.38, delta=0.1)
        self.assertAlmostEqual((roi.x0 + roi.x1) / 2.0, 363.0, delta=1.0)
        self.assertLessEqual(roi.x0, 318)
        self.assertGreaterEqual(roi.x1, 407)

    def test_synchronized_silhouette_pipeline_never_uses_qr_corners_for_roi(self):
        source = inspect.getsource(SimSynchronizedViewpointNode._process_latest)
        self.assertNotIn("qr_corners_px", source)
        self.assertIn("roi = lidar_seeded_roi", source)

    def test_lidar_bearing_projects_roi_horizontally(self):
        roi = stand_head_roi(
            frame_width=640, frame_height=480, bearing_rad=math.radians(10),
            distance_m=0.4, camera_fx_px=381.0, camera_cx_px=320.5,
            camera_cy_px=240.5, stand_face_size_m=0.078,
        )
        self.assertEqual(roi.source, "lidar_projected")
        self.assertGreater((roi.x0 + roi.x1) / 2.0, 320.5)
        self.assertAlmostEqual(roi.expected_head_px, 74.295, places=2)

    def test_qr_corners_seed_roi_without_using_them_as_axis(self):
        roi = stand_head_roi(
            frame_width=640, frame_height=480, bearing_rad=0.2,
            distance_m=0.4, camera_fx_px=381.0, camera_cx_px=320.5,
            camera_cy_px=240.5, stand_face_size_m=0.078,
            qr_corners_px=((390, 190), (430, 190), (430, 230), (390, 230)),
        )
        self.assertEqual(roi.source, "qr_seeded")
        self.assertAlmostEqual((roi.x0 + roi.x1) / 2.0, 410.0, delta=1.0)

    def test_roi_is_clipped_to_frame(self):
        roi = stand_head_roi(
            frame_width=100, frame_height=80, bearing_rad=0.4,
            distance_m=0.2, camera_fx_px=100.0, camera_cx_px=50.0,
            camera_cy_px=40.0, stand_face_size_m=0.078,
        )
        self.assertGreaterEqual(roi.x0, 0)
        self.assertLessEqual(roi.x1, 100)

    def test_target_behind_camera_has_no_roi(self):
        self.assertIsNone(
            stand_head_roi(
                frame_width=640, frame_height=480, bearing_rad=math.radians(128),
                distance_m=0.3, camera_fx_px=381.0, camera_cx_px=320.5,
                camera_cy_px=240.5, stand_face_size_m=0.078,
            )
        )

    def test_qr_must_be_entirely_inside_selected_roi(self):
        roi = stand_head_roi(
            frame_width=640, frame_height=480, bearing_rad=0.0,
            distance_m=0.4, camera_fx_px=381.0, camera_cx_px=320.5,
            camera_cy_px=240.5, stand_face_size_m=0.078,
        )
        self.assertTrue(qr_corners_inside_roi(((300, 210), (340, 210), (340, 250), (300, 250)), roi))
        self.assertFalse(qr_corners_inside_roi(((300, 210), (500, 210), (500, 250), (300, 250)), roi))


if __name__ == "__main__":
    unittest.main()
