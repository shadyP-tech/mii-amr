import math
import sys
import unittest
from pathlib import Path

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds  # noqa: E402
from scripts.aufgabe04.perception.sim_wall_edge_mask import (  # noqa: E402
    build_confirmed_wall_exclusion_mask,
    ray_distance_to_arena_wall,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan  # noqa: E402


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV/numpy unavailable")
class SimulationWallEdgeMaskTest(unittest.TestCase):
    def make_wall_scan(self, *, foreground_center=False, stamp=10.0):
        angle_min = math.radians(-80.0)
        increment = math.radians(1.0)
        ranges = []
        for index in range(161):
            bearing = angle_min + index * increment
            expected = ray_distance_to_arena_wall(
                origin_x_m=0.0,
                origin_y_m=0.0,
                ray_yaw_rad=bearing,
                arena=ArenaBounds(),
            )
            ranges.append(expected if expected is not None else float("inf"))
        if foreground_center:
            for index in range(77, 84):
                ranges[index] = 0.50
        return PlainLaserScan(
            ranges=tuple(ranges),
            angle_min=angle_min,
            angle_increment=increment,
            range_min=0.05,
            range_max=5.0,
            scan_frame_id="base_scan",
            scan_stamp_sec=stamp,
            receipt_sec=stamp,
        )

    def build(self, scan, *, foreground_support_mask=None):
        return build_confirmed_wall_exclusion_mask(
            cv2,
            numpy,
            scan=scan,
            image_stamp_sec=10.02,
            sync_tolerance_sec=0.08,
            robot_x_m=0.0,
            robot_y_m=0.0,
            robot_z_m=0.0,
            robot_yaw_rad=0.0,
            frame_width=640,
            frame_height=480,
            camera_fx_px=381.36246688,
            camera_fy_px=381.36246688,
            camera_cx_px=320.5,
            camera_cy_px=240.5,
            camera_forward_offset_m=0.076,
            camera_lateral_offset_m=0.0,
            camera_height_m=0.093,
            camera_yaw_offset_rad=0.0,
            foreground_support_mask=foreground_support_mask,
        )

    def test_ray_intersects_inner_wall_face(self):
        distance = ray_distance_to_arena_wall(
            origin_x_m=0.0,
            origin_y_m=0.0,
            ray_yaw_rad=0.0,
            arena=ArenaBounds(),
        )
        self.assertAlmostEqual(distance, 1.93, places=6)

    def test_confirmed_map_walls_create_an_image_mask(self):
        result = self.build(self.make_wall_scan())

        self.assertEqual(result.reason, "ok")
        self.assertGreater(result.confirmed_wall_samples, 20)
        self.assertGreater(int(numpy.count_nonzero(result.mask)), 1000)
        self.assertGreater(int(numpy.count_nonzero(result.mask[:, 300:340])), 0)

    def test_foreground_scan_cluster_protects_only_visual_stand_support(self):
        support = numpy.zeros((480, 640), dtype=numpy.uint8)
        support[185:225, 285:356] = 255
        result = self.build(
            self.make_wall_scan(foreground_center=True),
            foreground_support_mask=support,
        )

        self.assertEqual(result.reason, "ok")
        self.assertGreater(result.protected_foreground_samples, 0)
        self.assertEqual(int(numpy.count_nonzero(result.mask[190:220, 295:346])), 0)
        # The same columns outside the supported stand pixels must remain
        # masked; foreground evidence must never restore a whole image column.
        self.assertGreater(int(numpy.count_nonzero(result.mask[235:260, 295:346])), 0)
        self.assertGreater(int(numpy.count_nonzero(result.mask[:, :250])), 0)

    def test_foreground_corridor_keeps_dilated_support_at_beam_boundary(self):
        # At the live 0.45 m frontal pose, the projected foreground interval
        # ended one pixel inside the board and erased its right outer Canny
        # edge. A support dilation of seven pixels must protect its matching
        # radius on both sides of the measured beam interval.
        support = numpy.zeros((480, 640), dtype=numpy.uint8)
        cv2.line(support, (357, 180), (357, 230), 255, thickness=1)

        result = self.build(
            self.make_wall_scan(foreground_center=True),
            foreground_support_mask=support,
        )

        self.assertEqual(result.reason, "ok")
        self.assertEqual(int(result.mask[200, 357]), 0)
        # Protection remains local to measured visual support.
        self.assertGreater(int(result.mask[200, 362]), 0)

    def test_foreground_without_visual_support_does_not_restore_wall_columns(self):
        result = self.build(self.make_wall_scan(foreground_center=True))

        self.assertEqual(result.reason, "ok")
        self.assertGreater(result.protected_foreground_samples, 0)
        self.assertGreater(int(numpy.count_nonzero(result.mask[:, 300:340])), 0)

    def test_unsynchronized_scan_fails_open_without_mask(self):
        result = self.build(self.make_wall_scan(stamp=9.0))

        self.assertIsNone(result.mask)
        self.assertEqual(result.reason, "scan_image_unsynchronized")


if __name__ == "__main__":
    unittest.main()
