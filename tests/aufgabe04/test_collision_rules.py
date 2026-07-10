import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.fleet.collision_rules import detect_close_robot_conflict  # noqa: E402
from scripts.aufgabe04.fleet.robot_status import RobotStatus  # noqa: E402


class CollisionRulesTest(unittest.TestCase):
    def test_detects_close_robot_conflict(self):
        conflict = detect_close_robot_conflict(
            RobotStatus("robot_1", x_m=0.0, y_m=0.0),
            RobotStatus("robot_2", x_m=0.1, y_m=0.0),
            min_separation_m=0.2,
        )

        self.assertIsNotNone(conflict)


if __name__ == "__main__":
    unittest.main()

