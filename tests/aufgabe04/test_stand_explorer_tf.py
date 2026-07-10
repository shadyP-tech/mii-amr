import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception import stand_explorer_node  # noqa: E402


class StandExplorerTfTest(unittest.TestCase):
    def test_transform_age_accepts_timestamp_zero_static_transform(self):
        self.assertEqual(stand_explorer_node._transform_age_sec(100.0, 0.0), 0.0)

    def test_transform_age_clamps_future_timestamp_to_zero(self):
        self.assertEqual(stand_explorer_node._transform_age_sec(100.0, 100.2), 0.0)

    def test_transform_age_reports_dynamic_tf_staleness(self):
        self.assertAlmostEqual(stand_explorer_node._transform_age_sec(100.5, 100.1), 0.4)

    def test_lookup_uses_zero_time_for_latest_transform(self):
        marker = object()
        original_time = stand_explorer_node.Time
        try:
            stand_explorer_node.Time = lambda: marker
            self.assertIs(stand_explorer_node._latest_transform_time(), marker)
        finally:
            stand_explorer_node.Time = original_time


if __name__ == "__main__":
    unittest.main()
