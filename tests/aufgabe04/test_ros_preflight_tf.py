import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.localization import ros_preflight  # noqa: E402


class RosPreflightParameterTest(unittest.TestCase):
    def test_sim_time_parameter_is_provided_as_constructor_override(self):
        class FakeParameter:
            class Type:
                BOOL = "bool"

            def __init__(self, name, parameter_type, value):
                self.name = name
                self.parameter_type = parameter_type
                self.value = value

        original_parameter = ros_preflight.Parameter
        try:
            ros_preflight.Parameter = FakeParameter
            override = ros_preflight._node_parameter_overrides(True)[0]
            self.assertEqual(override.name, "use_sim_time")
            self.assertEqual(override.parameter_type, "bool")
            self.assertTrue(override.value)
        finally:
            ros_preflight.Parameter = original_parameter

    def test_preflight_result_persists_route_frame_pose(self):
        route_pose = {
            "frame_id": "map",
            "child_frame_id": "base_footprint",
            "x_m": -0.5,
            "y_m": -0.62,
            "yaw_rad": 1.7,
        }
        result = ros_preflight.RosPreflightResult(
            ok=True,
            failures=[],
            observations=[],
            runtime_config={},
            route_pose=route_pose,
        )

        self.assertEqual(result.to_json_dict()["route_pose"], route_pose)


if __name__ == "__main__":
    unittest.main()
