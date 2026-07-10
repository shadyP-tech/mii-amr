import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.stations.models import Station, StationPose  # noqa: E402
from scripts.aufgabe04.stations.station_layout_io import (  # noqa: E402
    load_station_layout_json,
    write_station_layout_csv,
    write_station_layout_json,
)


class StationLayoutIoTest(unittest.TestCase):
    def test_json_round_trip_normalizes_station_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layout.json"
            write_station_layout_json(
                path,
                [Station(" a ", StationPose(1.0, 2.0, 0.5), 0.3, 0.1)],
                {"seed": 42},
            )

            station_map = load_station_layout_json(path)
            payload = json.loads(path.read_text())
            text = path.read_text()

        self.assertEqual(list(station_map), ["A"])
        self.assertEqual(station_map["A"].pose, StationPose(1.0, 2.0, 0.5))
        self.assertEqual(payload["metadata"]["seed"], 42)
        self.assertTrue(text.endswith("\n"))

    def test_loader_rejects_duplicate_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stations": [
                            {
                                "station_id": "A",
                                "x_m": 0.0,
                                "y_m": 0.0,
                                "yaw_rad": 0.0,
                                "approach_offset_m": 0.3,
                                "keepout_radius_m": 0.1,
                            },
                            {
                                "station_id": " a ",
                                "x_m": 1.0,
                                "y_m": 0.0,
                                "yaw_rad": 0.0,
                                "approach_offset_m": 0.3,
                                "keepout_radius_m": 0.1,
                            },
                        ],
                    }
                )
            )
            missing = root / "missing.json"
            missing.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stations": [
                            {
                                "station_id": "A",
                                "x_m": 0.0,
                                "y_m": 0.0,
                                "yaw_rad": 0.0,
                                "approach_offset_m": 0.3,
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "duplicate station id"):
                load_station_layout_json(duplicate)
            with self.assertRaisesRegex(ValueError, "keepout_radius_m"):
                load_station_layout_json(missing)

    def test_csv_writer_emits_report_friendly_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layout.csv"
            write_station_layout_csv(path, [Station("B", StationPose(1.0, 2.0, 0.0), 0.3, 0.1)])

            text = path.read_text()

        self.assertIn("station_id,x_m,y_m,yaw_rad,approach_offset_m,keepout_radius_m", text)
        self.assertIn("B,1.0,2.0,0.0,0.3,0.1", text)


if __name__ == "__main__":
    unittest.main()
