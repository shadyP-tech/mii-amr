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

    def test_missing_peer_position_fails_closed(self):
        conflict = detect_close_robot_conflict(
            RobotStatus("robot_1", x_m=0.0, y_m=0.0),
            RobotStatus("robot_2"),
            min_separation_m=0.2,
        )

        self.assertIsNotNone(conflict)
        self.assertTrue(conflict.fail_closed)
        self.assertIn("position", conflict.reason)

    def test_stale_peer_status_fails_closed(self):
        conflict = detect_close_robot_conflict(
            RobotStatus("robot_1", x_m=0.0, y_m=0.0, timestamp_sec=10.0),
            RobotStatus("robot_2", x_m=1.0, y_m=0.0, timestamp_sec=8.0),
            min_separation_m=0.1,
            now_sec=10.0,
            max_status_age_sec=0.5,
        )

        self.assertIsNotNone(conflict)
        self.assertTrue(conflict.fail_closed)
        self.assertIn("stale", conflict.reason)

    def test_swept_footprints_detect_future_crossing(self):
        conflict = detect_close_robot_conflict(
            RobotStatus(
                "robot_1",
                x_m=-1.0,
                y_m=0.0,
                timestamp_sec=1.0,
                velocity_x_mps=1.0,
                velocity_y_mps=0.0,
                footprint_radius_m=0.1,
            ),
            RobotStatus(
                "robot_2",
                x_m=0.0,
                y_m=-1.0,
                timestamp_sec=1.0,
                velocity_x_mps=0.0,
                velocity_y_mps=1.0,
                footprint_radius_m=0.1,
            ),
            min_separation_m=0.05,
            now_sec=1.0,
            max_status_age_sec=0.1,
            prediction_horizon_sec=2.0,
        )

        self.assertIsNotNone(conflict)
        self.assertAlmostEqual(conflict.closest_separation_m, 0.0)

    def test_loaded_footprints_expand_required_separation(self):
        conflict = detect_close_robot_conflict(
            RobotStatus(
                "robot_1",
                x_m=0.0,
                y_m=0.0,
                footprint_radius_m=0.1,
                loaded_footprint_radius_m=0.2,
                payload_loaded=True,
            ),
            RobotStatus(
                "robot_2",
                x_m=0.35,
                y_m=0.0,
                footprint_radius_m=0.1,
            ),
            min_separation_m=0.1,
        )

        self.assertIsNotNone(conflict)
        self.assertAlmostEqual(conflict.required_separation_m, 0.4)

    def test_loaded_peer_without_loaded_footprint_fails_closed(self):
        conflict = detect_close_robot_conflict(
            RobotStatus(
                "robot_1",
                x_m=0.0,
                y_m=0.0,
                footprint_radius_m=0.1,
                payload_loaded=True,
            ),
            RobotStatus("robot_2", x_m=1.0, y_m=0.0),
            min_separation_m=0.1,
        )

        self.assertIsNotNone(conflict)
        self.assertTrue(conflict.fail_closed)
        self.assertIn("loaded footprint", conflict.reason)

    def test_prediction_without_velocity_fails_closed(self):
        conflict = detect_close_robot_conflict(
            RobotStatus("robot_1", x_m=0.0, y_m=0.0),
            RobotStatus("robot_2", x_m=1.0, y_m=0.0),
            min_separation_m=0.1,
            prediction_horizon_sec=1.0,
        )

        self.assertIsNotNone(conflict)
        self.assertTrue(conflict.fail_closed)
        self.assertIn("velocity", conflict.reason)


if __name__ == "__main__":
    unittest.main()
