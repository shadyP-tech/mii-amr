from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OrchestrationModuleBoundaryTest(unittest.TestCase):
    def test_navigation_root_contains_only_metadata_and_compatibility_facade(self):
        package = ROOT / "scripts/aufgabe04/navigation"
        self.assertEqual(
            {path.name for path in package.glob("*.py")},
            {"__init__.py", "simple_waypoint_follower.py"},
        )

    def test_navigation_entrypoints_are_clustered(self):
        package = ROOT / "scripts/aufgabe04/navigation/entrypoints"
        expected = {
            "__init__.py",
            "compute_qr_facing_pose.py",
            "create_detected_station_confirmation.py",
            "generate_random_station_layout.py",
            "plan_arrival_catalog_route.py",
            "plan_detected_stand_exploration.py",
            "plan_first_detected_station.py",
            "plan_stand_coverage_survey.py",
            "plan_synchronized_viewpoint.py",
            "prepare_detected_stand_preapproach.py",
            "read_current_amcl_pose.py",
            "record_stand_candidate_decision.py",
            "record_stand_coverage_stop.py",
            "ros_preflight.py",
            "run_detected_stand_exploration_sim.py",
            "run_detected_stand_observe_plan.py",
            "run_single_station_segment.py",
            "run_station_route.py",
        }

        self.assertEqual(
            {path.name for path in package.glob("*.py")},
            expected,
        )
        self.assertTrue(
            (package / "run_first_detected_station_segment_with_bundle.sh").is_file()
        )

    def test_navigation_library_concerns_have_dedicated_packages(self):
        package = ROOT / "scripts/aufgabe04/navigation"
        expected = {
            "foundation",
            "planning",
            "control",
            "localization",
            "execution",
            "coverage",
            "approach",
            "missions",
            "entrypoints",
            "station_segment",
            "waypoint_follower",
        }

        for name in expected:
            self.assertTrue((package / name / "__init__.py").is_file(), name)

    def test_station_segment_entrypoint_aliases_canonical_runtime(self):
        entrypoint = importlib.import_module(
            "scripts.aufgabe04.navigation.entrypoints.run_single_station_segment"
        )
        runtime = importlib.import_module(
            "scripts.aufgabe04.navigation.station_segment.runtime"
        )

        self.assertIs(entrypoint, runtime)

    def test_waypoint_follower_legacy_path_aliases_canonical_runtime(self):
        legacy = importlib.import_module(
            "scripts.aufgabe04.navigation.simple_waypoint_follower"
        )
        runtime = importlib.import_module(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime"
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
            ROOT / "scripts/aufgabe04/navigation/foundation",
            ROOT / "scripts/aufgabe04/navigation/planning",
            ROOT / "scripts/aufgabe04/navigation/control",
            ROOT / "scripts/aufgabe04/navigation/localization",
            ROOT / "scripts/aufgabe04/navigation/execution",
            ROOT / "scripts/aufgabe04/navigation/coverage",
            ROOT / "scripts/aufgabe04/navigation/approach",
            ROOT / "scripts/aufgabe04/navigation/missions",
            ROOT / "scripts/aufgabe04/navigation/entrypoints",
            ROOT / "scripts/aufgabe04/navigation/station_segment",
            ROOT / "scripts/aufgabe04/real_robot/autonomous_runner",
            ROOT / "scripts/aufgabe04/real_robot/coverage_leg",
        )

        for package in roots:
            for path in package.glob("*.py"):
                source = path.read_text()
                self.assertNotIn("create_publisher", source, str(path))
                self.assertNotIn("cmd_vel_pub.publish", source, str(path))

    def test_camera_candidate_planning_modules_are_ros_free(self):
        package = ROOT / "scripts/aufgabe04/navigation/approach"
        paths = tuple(
            package / name
            for name in (
                "camera_axis_binding.py",
                "camera_candidate_selection.py",
                "candidate_preapproach_models.py",
                "candidate_preapproach_compute.py",
                "candidate_preapproach_materialization.py",
                "candidate_preapproach_planning.py",
                "candidate_preapproach_selection.py",
            )
        )
        forbidden_prefixes = (
            "rclpy",
            "geometry_msgs",
            "nav_msgs",
            "sensor_msgs",
            "scripts.aufgabe04.real_robot",
        )

        offenders = []
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules = (node.module or "",)
                else:
                    continue
                offenders.extend(
                    (path.name, module)
                    for module in modules
                    if module.startswith(forbidden_prefixes)
                )

        self.assertEqual(offenders, [])

    def test_candidate_preapproach_compatibility_facade_stays_small(self):
        facade = (
            ROOT
            / "scripts/aufgabe04/navigation/approach"
            / "candidate_preapproach_planning.py"
        )

        self.assertLessEqual(
            len(facade.read_text().splitlines()),
            160,
            "candidate_preapproach_planning.py must remain a compatibility facade",
        )


if __name__ == "__main__":
    unittest.main()
