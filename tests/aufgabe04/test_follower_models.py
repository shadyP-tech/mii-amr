import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.follower_models import FollowerResult  # noqa: E402
from scripts.aufgabe04.navigation.simple_waypoint_follower import (  # noqa: E402
    FollowerResult as SimpleFollowerResult,
)


class FollowerModelsTest(unittest.TestCase):
    def test_simple_follower_uses_shared_result_model(self):
        self.assertIs(SimpleFollowerResult, FollowerResult)

    def test_result_model_is_frozen_and_adapter_neutral(self):
        result = FollowerResult("completed", "", 1.25, 0.5, True)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.distance_estimate_m, 0.5)
        with self.assertRaises(FrozenInstanceError):
            result.status = "stopped"


if __name__ == "__main__":
    unittest.main()
