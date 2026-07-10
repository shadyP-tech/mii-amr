import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation import ros_preflight  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
