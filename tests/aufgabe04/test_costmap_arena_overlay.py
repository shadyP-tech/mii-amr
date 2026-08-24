import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds  # noqa: E402
from scripts.aufgabe04.navigation.planning.costmap import (  # noqa: E402
    CELL_SOURCE_ARENA_BOUNDARY,
    CELL_SOURCE_INFLATED,
    CELL_SOURCE_STATIC_OCCUPIED,
    CELL_SOURCE_UNKNOWN,
    Costmap,
)
from scripts.aufgabe04.navigation.planning.map_io import (  # noqa: E402
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.foundation.models import GridCell  # noqa: E402


def padded_grid() -> OccupancyGrid:
    rows = [[CELL_FREE] * 7 for _ in range(7)]
    rows[0][0] = CELL_OCCUPIED
    rows[6][6] = CELL_UNKNOWN
    return OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=1.0,
            origin=(-3.5, -3.5, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.20,
            mode="trinary",
        ),
        width=7,
        height=7,
        cells=tuple(tuple(row) for row in rows),
    )


def default_arena_padded_grid() -> OccupancyGrid:
    width = 100
    height = 60
    return OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=0.05,
            origin=(-2.5, -1.5, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.20,
            mode="trinary",
        ),
        width=width,
        height=height,
        cells=tuple(tuple([CELL_FREE] * width) for _ in range(height)),
    )


class CostmapArenaOverlayTest(unittest.TestCase):
    def setUp(self):
        self.base = Costmap.from_occupancy_grid(padded_grid())
        self.arena = ArenaBounds(length_m=3.0, width_m=3.0)

    def test_padded_free_cells_outside_arena_are_blocked_with_distinct_source(self):
        overlaid = self.base.with_arena_bounds(self.arena)

        outside_free = GridCell(1, 3)
        inside_free = GridCell(2, 3)
        self.assertFalse(self.base.is_blocked(outside_free))
        self.assertTrue(overlaid.is_blocked(outside_free))
        self.assertEqual(
            overlaid.cell_sources[outside_free],
            CELL_SOURCE_ARENA_BOUNDARY,
        )
        self.assertTrue(overlaid.is_traversable(inside_free))

    def test_overlay_preserves_static_occupied_and_unknown_provenance(self):
        overlaid = self.base.with_arena_bounds(self.arena)

        occupied = GridCell(0, 0)
        unknown = GridCell(6, 6)
        self.assertEqual(
            overlaid.cell_sources[occupied],
            CELL_SOURCE_STATIC_OCCUPIED,
        )
        self.assertEqual(overlaid.cell_sources[unknown], CELL_SOURCE_UNKNOWN)
        self.assertEqual(overlaid.cells, self.base.cells)

    def test_inflating_overlay_creates_interior_boundary_clearance(self):
        boundary_overlay = self.base.with_arena_bounds(self.arena)
        inflated = boundary_overlay.with_inflation(0.25)

        interior_edge = GridCell(2, 3)
        interior_center = GridCell(3, 3)
        self.assertTrue(boundary_overlay.is_traversable(interior_edge))
        self.assertTrue(inflated.is_blocked(interior_edge))
        self.assertEqual(
            inflated.cell_sources[interior_edge],
            CELL_SOURCE_INFLATED,
        )
        self.assertTrue(inflated.is_traversable(interior_center))

    def test_default_wall_and_inflation_block_live_near_wall_candidate(self):
        base = Costmap.from_occupancy_grid(default_arena_padded_grid())
        boundary_overlay = base.with_arena_bounds(ArenaBounds())
        inflated = boundary_overlay.with_inflation(0.23)

        near_positive_y_wall = base.world_to_grid(0.0, 0.743)
        arena_center = base.world_to_grid(0.0, 0.0)
        self.assertTrue(boundary_overlay.is_traversable(near_positive_y_wall))
        self.assertTrue(inflated.is_blocked(near_positive_y_wall))
        self.assertEqual(
            inflated.cell_sources[near_positive_y_wall],
            CELL_SOURCE_INFLATED,
        )
        self.assertTrue(inflated.is_traversable(arena_center))

    def test_invalid_or_nonfinite_arena_is_rejected(self):
        for arena in (
            ArenaBounds(length_m=0.0, width_m=3.0),
            ArenaBounds(length_m=float("inf"), width_m=3.0),
        ):
            with self.subTest(arena=arena):
                with self.assertRaises(ValueError):
                    self.base.with_arena_bounds(arena)


if __name__ == "__main__":
    unittest.main()
