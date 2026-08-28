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

    def test_autonomous_runner_entrypoint_aliases_canonical_runtime(self):
        entrypoint = importlib.import_module(
            "scripts.aufgabe04.real_robot.entrypoints."
            "run_autonomous_stand_exploration"
        )
        runtime = importlib.import_module(
            "scripts.aufgabe04.real_robot.autonomous_runner.runtime"
        )

        self.assertIs(entrypoint, runtime)

    def test_real_robot_root_contains_only_package_metadata(self):
        package = ROOT / "scripts/aufgabe04/real_robot"

        self.assertEqual(
            {path.name for path in package.glob("*.py")},
            {"__init__.py"},
        )

    def test_real_robot_imports_use_responsibility_packages(self):
        prefix = "scripts.aufgabe04.real_robot"
        allowed_packages = {
            "autonomous_runner",
            "candidate",
            "configuration",
            "coverage_leg",
            "entrypoints",
            "execution",
            "mission",
            "observer",
            "passive_survey",
            "readiness",
        }
        roots = (
            ROOT / "scripts/aufgabe04",
            ROOT / "tests/aufgabe04",
        )
        offenders = []

        for source_root in roots:
            for path in source_root.rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = (alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        modules = (node.module or "",)
                    else:
                        continue
                    for module in modules:
                        if module == prefix:
                            offenders.append((path.relative_to(ROOT), module))
                        elif module.startswith(prefix + "."):
                            package = module[len(prefix) + 1 :].split(".", 1)[0]
                            if package not in allowed_packages:
                                offenders.append((path.relative_to(ROOT), module))

        self.assertEqual(offenders, [])

    def test_real_robot_responsibilities_have_dedicated_packages(self):
        package = ROOT / "scripts/aufgabe04/real_robot"
        expected = {
            "configuration": {
                "capture_calibration.py",
                "create_profile.py",
                "geometry.py",
                "profile.py",
                "recommendation.py",
                "site_contract.py",
            },
            "observer": {
                "contract.py",
                "diagnostics.py",
                "evidence.py",
                "node.py",
                "process.py",
                "tf_retry.py",
            },
            "passive_survey": {"prepare.py", "finalize.py"},
            "candidate": {
                "approach.py",
                "observation_deferral.py",
                "recovery_failure.py",
                "runtime_recovery.py",
                "startup_recovery.py",
            },
            "readiness": {
                "candidate_planning_frame.py",
                "initial.py",
                "localization.py",
                "observation_tf_contract.py",
                "observation_tf_runtime.py",
                "post_observation.py",
                "preauthorization.py",
                "startup_reseal.py",
            },
            "mission": {
                "checkpoint_resume.py",
                "coverage.py",
                "exact_two_completion.py",
                "modes.py",
                "reporting.py",
                "session_manifest.py",
            },
            "execution": {
                "artifact_paths.py",
                "child_runner.py",
                "localization_recovery.py",
                "run_unloaded_segment.py",
            },
            "entrypoints": {
                "capture_camera_calibration.py",
                "create_hardware_profile.py",
                "finalize_passive_survey.py",
                "passive_viewpoint_node.py",
                "prepare_passive_survey.py",
                "run_autonomous_stand_exploration.py",
                "run_unloaded_segment.py",
            },
        }

        for name, modules in expected.items():
            responsibility_package = package / name
            self.assertTrue(
                (responsibility_package / "__init__.py").is_file(),
                name,
            )
            self.assertEqual(
                {
                    path.name
                    for path in responsibility_package.glob("*.py")
                    if path.name != "__init__.py"
                },
                modules,
                name,
            )

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
            ROOT / "scripts/aufgabe04/real_robot/configuration",
            ROOT / "scripts/aufgabe04/real_robot/observer",
            ROOT / "scripts/aufgabe04/real_robot/passive_survey",
            ROOT / "scripts/aufgabe04/real_robot/candidate",
            ROOT / "scripts/aufgabe04/real_robot/readiness",
            ROOT / "scripts/aufgabe04/real_robot/mission",
            ROOT / "scripts/aufgabe04/real_robot/execution",
            ROOT / "scripts/aufgabe04/real_robot/entrypoints",
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
