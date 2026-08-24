from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "scripts/aufgabe04/navigation/waypoint_follower"
RUNTIME_COMPONENT_ROOT = PACKAGE_ROOT / "runtime_components"
RUNTIME_MODULE = (
    "scripts.aufgabe04.navigation.waypoint_follower.runtime"
)


class WaypointFollowerModuleBoundaryTest(unittest.TestCase):
    def test_pure_package_import_does_not_load_ros_runtime(self):
        code = (
            "import sys; "
            "import scripts.aufgabe04.navigation.waypoint_follower.config; "
            f"assert {RUNTIME_MODULE!r} not in sys.modules"
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_legacy_entrypoint_aliases_the_runtime_module(self):
        legacy = importlib.import_module(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime"
        )
        runtime = importlib.import_module(RUNTIME_MODULE)

        self.assertIs(legacy, runtime)

    def test_legacy_entrypoint_preserves_supported_symbols(self):
        legacy = importlib.import_module(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime"
        )
        supported_symbols = (
            "CALLBACK_SERVICE_BACKGROUND_EXECUTOR",
            "FOLLOWER_EXECUTOR_NUM_THREADS",
            "FollowerConfig",
            "FollowerResult",
            "INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED",
            "PoseLookupResult",
            "STALE_TF_RECOVERY_MAX_CALLBACKS",
            "STALE_TF_RECOVERY_MAX_DURATION_SEC",
            "STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC",
            "STATIC_PHYSICAL_ROUTE_KINDS",
            "SimpleWaypointFollowerNode",
            "TF_LISTENER_NODE_NAME",
            "acquisition_goal_action",
            "certified_startup_join_action",
            "certified_startup_route_state",
            "certified_static_startup_decision",
            "compute_intermediate_terminal_heading_command",
            "controller_config_for_route_kind",
            "dynamic_join_envelope_failure",
            "dynamic_route_kind_transition_failure",
            "intermediate_terminal_heading_entry_tolerance_m",
            "intermediate_terminal_heading_hold_diagnostics",
            "reset_intermediate_terminal_heading_latch",
            "run_simple_waypoint_follower",
            "stuck_progress_details",
            "tf_lookup_failure_details",
            "viewpoint_sampling_target_timeout_failure",
            "viewpoint_sampling_timeout_failure",
        )

        missing = [
            name for name in supported_symbols if not hasattr(legacy, name)
        ]
        self.assertEqual(missing, [])

    def test_pure_modules_have_no_ros_imports(self):
        pure_modules = (
            "config.py",
            "pose_lookup.py",
            "route_admission.py",
            "route_phases.py",
            "startup.py",
            "terminal_heading.py",
        )
        forbidden_roots = {
            "geometry_msgs",
            "nav_msgs",
            "rclpy",
            "sensor_msgs",
            "std_srvs",
            "tf2_ros",
        }

        for filename in pure_modules:
            tree = ast.parse((PACKAGE_ROOT / filename).read_text())
            imported_roots = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported_roots.update(
                node.module.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertFalse(
                imported_roots & forbidden_roots,
                f"{filename} imports ROS runtime packages",
            )

    def test_runtime_remains_the_only_package_velocity_publisher(self):
        runtime_source = (PACKAGE_ROOT / "runtime.py").read_text()
        self.assertIn("self.cmd_vel_pub = self.create_publisher", runtime_source)
        self.assertIn("self.cmd_vel_pub.publish", runtime_source)

        for path in PACKAGE_ROOT.rglob("*.py"):
            if path.name == "runtime.py":
                continue
            source = path.read_text()
            self.assertNotIn("create_publisher", source, path.name)
            self.assertNotIn("cmd_vel_pub.publish", source, path.name)

    def test_runtime_components_are_split_by_operational_responsibility(self):
        expected_classes = {
            "amcl_recovery.py": "AmclRecoveryMixin",
            "blockage_recovery.py": "BlockageRecoveryRuntimeMixin",
            "callback_service.py": "CallbackServiceRuntimeMixin",
            "control_loop.py": "ControlLoopRuntimeMixin",
            "dynamic_routes.py": "DynamicRouteRuntimeMixin",
            "localization.py": "LocalizationRuntimeMixin",
            "localization_evidence.py": "LocalizationEvidenceMixin",
            "localization_sampling.py": "LocalizationSamplingMixin",
            "safety.py": "SafetyRuntimeMixin",
            "simulation_odom_recovery.py": "SimulationOdomRecoveryMixin",
        }

        for filename, class_name in expected_classes.items():
            path = RUNTIME_COMPONENT_ROOT / filename
            self.assertTrue(path.is_file(), filename)
            tree = ast.parse(path.read_text())
            classes = {
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            }
            self.assertIn(class_name, classes, filename)

    def test_runtime_keeps_node_lifecycle_and_motion_output(self):
        runtime_source = (PACKAGE_ROOT / "runtime.py").read_text()
        control_loop_source = (
            RUNTIME_COMPONENT_ROOT / "control_loop.py"
        ).read_text()

        self.assertIn("class SimpleWaypointFollowerNode(", runtime_source)
        self.assertIn("def _publish_velocity_command(", runtime_source)
        self.assertIn("def run_simple_waypoint_follower(", runtime_source)
        self.assertNotIn("def run(self) -> FollowerResult:", runtime_source)
        self.assertIn("def run(self) -> FollowerResult:", control_loop_source)
        self.assertNotIn("cmd_vel_pub.publish", control_loop_source)
        self.assertNotIn("def _refresh_dynamic_route(", runtime_source)
        self.assertNotIn("def _current_pose_lookup(", runtime_source)

    def test_live_runner_imports_owning_modules_not_entrypoint(self):
        entrypoint_source = (
            ROOT
            / "scripts/aufgabe04/navigation/entrypoints/run_single_station_segment.py"
        ).read_text()
        source = (
            ROOT
            / "scripts/aufgabe04/navigation/station_segment/runtime.py"
        ).read_text()

        self.assertNotIn(
            "navigation.entrypoints import",
            source,
        )
        self.assertIn("navigation.waypoint_follower.runtime import", source)
        self.assertIn("navigation.station_segment import runtime", entrypoint_source)


if __name__ == "__main__":
    unittest.main()
