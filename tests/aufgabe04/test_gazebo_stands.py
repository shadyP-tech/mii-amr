import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.simulation.generate_gazebo_world import (  # noqa: E402
    QR_SIZE,
    qr_matrix,
    world_sdf,
)


class GazeboStandsTest(unittest.TestCase):
    def test_station_qr_matrix_has_version_one_shape_and_finders(self):
        matrix = qr_matrix("A")
        self.assertEqual(len(matrix), QR_SIZE)
        self.assertTrue(all(len(row) == QR_SIZE for row in matrix))
        self.assertEqual(matrix[0][0:7], (1, 1, 1, 1, 1, 1, 1))
        self.assertEqual(matrix[0][14:21], (1, 1, 1, 1, 1, 1, 1))
        self.assertEqual(matrix[14][0:7], (1, 1, 1, 1, 1, 1, 1))
        self.assertEqual(matrix[3][3], 1)
        self.assertEqual(matrix[1][1], 0)

    def test_different_station_ids_change_qr_data(self):
        self.assertNotEqual(qr_matrix("A"), qr_matrix("B"))

    def test_world_contains_static_stands_and_qr_faces(self):
        world = world_sdf(
            [
                {"station_id": "A", "x_m": -0.4, "y_m": -0.4, "yaw_rad": 0.8},
                {"station_id": "B", "x_m": 0.4, "y_m": 0.4, "yaw_rad": -2.5},
            ]
        )
        self.assertIn('<world name="aufgabe04_stands">', world)
        self.assertIn('<model name="station_A">', world)
        self.assertIn('<model name="station_B">', world)
        self.assertEqual(world.count("<static>true</static>"), 4)
        self.assertIn('name="qr_white_panel"', world)
        self.assertIn('name="qr_00_00"', world)
        self.assertIn('name="head_board"', world)

    def test_generator_uses_layout_json_and_creates_parent(self):
        from scripts.aufgabe04.simulation.generate_gazebo_world import main

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            layout = directory / "layout.json"
            output = directory / "nested" / "world.world"
            layout.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stations": [
                            {"station_id": "A", "x_m": 0, "y_m": 0, "yaw_rad": 0}
                        ],
                    }
                )
            )
            self.assertEqual(main(["--layout", str(layout), "--output", str(output)]), 0)
            self.assertTrue(output.exists())
            self.assertIn("station_A", output.read_text())


if __name__ == "__main__":
    unittest.main()
