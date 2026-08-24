from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OrchestrationModuleBoundaryTest(unittest.TestCase):
    def test_station_segment_legacy_path_aliases_canonical_runtime(self):
        legacy = importlib.import_module(
            "scripts.aufgabe04.navigation.run_single_station_segment"
        )
        runtime = importlib.import_module(
            "scripts.aufgabe04.navigation.station_segment.runtime"
        )

        self.assertIs(legacy, runtime)

    def test_autonomous_runner_legacy_path_aliases_canonical_runtime(self):
        legacy = importlib.import_module(
            "scripts.aufgabe04.real_robot.run_autonomous_stand_exploration"
        )
        runtime = importlib.import_module(
            "scripts.aufgabe04.real_robot.autonomous_runner.runtime"
        )

        self.assertIs(legacy, runtime)

    def test_coverage_execution_legacy_path_aliases_public_execution_api(self):
        legacy = importlib.import_module(
            "scripts.aufgabe04.real_robot.autonomous_coverage_execution"
        )
        execution = importlib.import_module(
            "scripts.aufgabe04.real_robot.coverage_leg.execution"
        )

        self.assertIs(legacy, execution)

    def test_station_segment_responsibilities_have_dedicated_modules(self):
        package = ROOT / "scripts/aufgabe04/navigation/station_segment"
        expected_functions = {
            "cli.py": {"build_parser"},
            "argument_validation.py": {"prepare_runtime_arguments"},
            "execution_route_admission.py": {"admit_execution_route"},
            "route_bundle.py": {
                "_load_execution_route_leg",
                "_revalidate_authoritative_route_before_motion",
            },
            "localization_admission.py": {
                "_build_odom_execution_admission",
                "_admit_stationary_map_from_odom_window",
            },
            "reporting.py": {"_append_result", "_observation_log_rows"},
        }

        for filename, required in expected_functions.items():
            tree = ast.parse((package / filename).read_text())
            functions = {
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            self.assertTrue(required <= functions, filename)

    def test_coverage_recovery_phases_are_separate_mixins(self):
        package = ROOT / "scripts/aufgabe04/real_robot/coverage_leg"
        expected_classes = {
            "route_sealing.py": "RouteSealingMixin",
            "child_execution.py": "ChildExecutionMixin",
            "readiness_recovery.py": "ReadinessRecoveryMixin",
            "startup_recovery.py": "StartupRecoveryMixin",
            "runtime_recovery.py": "RuntimeRecoveryMixin",
        }

        for filename, class_name in expected_classes.items():
            tree = ast.parse((package / filename).read_text())
            classes = {
                node.name
                for node in tree.body
                if isinstance(node, ast.ClassDef)
            }
            self.assertIn(class_name, classes, filename)

    def test_new_orchestration_modules_do_not_publish_velocity(self):
        roots = (
            ROOT / "scripts/aufgabe04/navigation/station_segment",
            ROOT / "scripts/aufgabe04/real_robot/autonomous_runner",
            ROOT / "scripts/aufgabe04/real_robot/coverage_leg",
        )

        for package in roots:
            for path in package.glob("*.py"):
                source = path.read_text()
                self.assertNotIn("create_publisher", source, str(path))
                self.assertNotIn("cmd_vel_pub.publish", source, str(path))


if __name__ == "__main__":
    unittest.main()
