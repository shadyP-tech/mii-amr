import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import lidar_obstacle_debug_viz as debug_viz  # noqa: E402
import lidar_obstacle_map as overlay  # noqa: E402
import map_path_planner as planner  # noqa: E402


def metadata(resolution=0.1, origin=(0.0, 0.0, 0.0)):
    return planner.MapMetadata(
        yaml_path=Path("test.yaml"),
        image_path=Path("test.pgm"),
        resolution=resolution,
        origin=origin,
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.25,
        mode="trinary",
    )


def free_map(width=10, height=10, resolution=0.1):
    return planner.OccupancyMap(
        metadata=metadata(resolution=resolution),
        width=width,
        height=height,
        cells=[
            [planner.CELL_FREE for _ in range(width)]
            for _ in range(height)
        ],
    )


class LidarObstacleDebugVizTest(unittest.TestCase):
    def test_scan_point_layers_separate_roi_candidates(self):
        config = overlay.ObstacleOverlayConfig(
            forward_distance_m=0.55,
            forward_half_width_m=0.18,
            angle_window_deg=45.0,
            min_range_m=0.12,
            robot_footprint_radius_m=0.18,
        )
        points = [
            overlay.BaseFramePoint(0.45, 0.00),
            overlay.BaseFramePoint(0.45, 0.30),
            overlay.BaseFramePoint(-0.20, 0.00),
            overlay.BaseFramePoint(float("inf"), 0.00),
        ]

        layers = debug_viz.scan_point_layers_from_points(points, config)

        self.assertEqual(len(layers.raw_points), 3)
        self.assertEqual(layers.roi_points, (overlay.BaseFramePoint(0.45, 0.00),))
        self.assertEqual(layers.roi_rejected_count, 2)

    def test_classify_observation_cells_reports_current_map_filter_reasons(self):
        occ = free_map(width=5, height=5)
        occ.cells[1][1] = planner.CELL_OCCUPIED
        observations = [
            overlay.MapFrameObservation(0.25, 0.25),
            overlay.MapFrameObservation(0.15, 0.15),
            overlay.MapFrameObservation(0.25, 0.15),
            overlay.MapFrameObservation(99.0, 99.0),
            overlay.MapFrameObservation(float("nan"), 0.0),
        ]

        layers = debug_viz.classify_observation_cells(
            occ,
            observations,
            wall_band_cells={(2, 1)},
        )

        self.assertEqual(layers.total_observations, 5)
        self.assertEqual(layers.accepted_cells, frozenset({(2, 2)}))
        self.assertEqual(layers.rejected_static_cells, frozenset({(1, 1)}))
        self.assertEqual(layers.rejected_wall_band_cells, frozenset({(2, 1)}))
        self.assertEqual(layers.rejected_bounds, 1)
        self.assertEqual(layers.rejected_invalid_range, 1)

    def test_run_local_confirmation_uses_configured_hit_count(self):
        occ = free_map(width=8, height=8)
        run_map = overlay.RunLocalObstacleMap(
            occ,
            overlay.RunLocalMapConfig(
                min_hit_count=2,
                min_used_points=1,
                max_updates=10,
                inflation_radius_m=0.1,
            ),
        )
        observation = overlay.MapFrameObservation(0.25, 0.25)

        first = run_map.add_observations(overlay.ObservationBatch([observation]))
        second = run_map.add_observations(overlay.ObservationBatch([observation]))

        self.assertTrue(first.update_accepted)
        self.assertTrue(second.update_accepted)
        self.assertEqual(run_map.hit_counts[(2, 2)], 2)
        self.assertIn((2, 2), run_map.confirmed_raw_cells)
        self.assertGreater(len(run_map.inflated_obstacle_cells), 1)


if __name__ == "__main__":
    unittest.main()
