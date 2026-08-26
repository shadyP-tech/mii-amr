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
            "AcquisitionGoalAction",
            "AcquisitionGoalDecision",
            "BlockageRecoveryAction",
            "CALLBACK_SERVICE_BACKGROUND_EXECUTOR",
            "FOLLOWER_EXECUTOR_NUM_THREADS",
            "FollowerConfig",
            "FollowerResult",
            "INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED",
            "PoseLookupResult",
            "RouteRefreshAction",
            "RouteCommandPhase",
            "STALE_TF_RECOVERY_MAX_CALLBACKS",
            "STALE_TF_RECOVERY_MAX_DURATION_SEC",
            "STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC",
            "STATIC_PHYSICAL_ROUTE_KINDS",
            "StartupJoinAction",
            "SimpleWaypointFollowerNode",
            "TF_LISTENER_NODE_NAME",
            "ViewpointSamplingDeadlineDecision",
            "acquisition_goal_action",
            "acquisition_goal_decision",
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
            "route_command_phase",
            "run_simple_waypoint_follower",
            "stuck_progress_details",
            "tf_lookup_failure_details",
            "viewpoint_sampling_deadline_decision",
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
            "directives.py",
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

    def test_control_decision_helpers_are_ros_free_and_side_effect_free(self):
        helper_modules = (
            "command_admission.py",
            "control_results.py",
            "recovery_dispatch.py",
            "route_step_resolution.py",
        )
        forbidden_import_roots = {
            "geometry_msgs",
            "nav_msgs",
            "rclpy",
            "sensor_msgs",
            "std_srvs",
            "tf2_ros",
        }
        forbidden_source_fragments = (
            "create_publisher",
            "cmd_vel_pub.publish",
            "publish_repeated_zero",
            "publish_zero",
            "_append_controller_trace",
            "time.sleep",
            "spin_once",
            "scripts.aufgabe03",
        )

        for filename in helper_modules:
            path = RUNTIME_COMPONENT_ROOT / filename
            source = path.read_text()
            tree = ast.parse(source)
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
                imported_roots & forbidden_import_roots,
                f"{filename} imports ROS runtime packages",
            )
            for fragment in forbidden_source_fragments:
                self.assertNotIn(fragment, source, filename)

    def test_runtime_components_are_split_by_operational_responsibility(self):
        expected_classes = {
            "amcl_recovery.py": "AmclRecoveryMixin",
            "blockage_recovery.py": "BlockageRecoveryRuntimeMixin",
            "callback_service.py": "CallbackServiceRuntimeMixin",
            "control_loop.py": "ControlLoopRuntimeMixin",
            "cycle_guard.py": "ControlCycleGuardRuntimeMixin",
            "dynamic_routes.py": "DynamicRouteRuntimeMixin",
            "localization.py": "LocalizationRuntimeMixin",
            "localization_evidence.py": "LocalizationEvidenceMixin",
            "localization_sampling.py": "LocalizationSamplingMixin",
            "motion_cycle_guard.py": "MotionCycleGuardRuntimeMixin",
            "route_step_resolution.py": "RouteStepResolutionRuntimeMixin",
            "route_cycle_guard.py": "RouteCycleGuardRuntimeMixin",
            "safety.py": "SafetyRuntimeMixin",
            "simulation_odom_recovery.py": "SimulationOdomRecoveryMixin",
            "step_cycle_guard.py": "StepCycleGuardRuntimeMixin",
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
        safety_source = (RUNTIME_COMPONENT_ROOT / "safety.py").read_text()
        route_step_source = (
            RUNTIME_COMPONENT_ROOT / "route_step_resolution.py"
        ).read_text()
        cycle_guard_source = (
            RUNTIME_COMPONENT_ROOT / "cycle_guard.py"
        ).read_text()
        route_cycle_guard_source = (
            RUNTIME_COMPONENT_ROOT / "route_cycle_guard.py"
        ).read_text()
        step_cycle_guard_source = (
            RUNTIME_COMPONENT_ROOT / "step_cycle_guard.py"
        ).read_text()
        motion_cycle_guard_source = (
            RUNTIME_COMPONENT_ROOT / "motion_cycle_guard.py"
        ).read_text()

        self.assertIn("class SimpleWaypointFollowerNode(", runtime_source)
        self.assertIn("def _publish_velocity_command(", runtime_source)
        self.assertIn("def run_simple_waypoint_follower(", runtime_source)
        self.assertNotIn("def run(self) -> FollowerResult:", runtime_source)
        self.assertIn("def run(self) -> FollowerResult:", control_loop_source)
        self.assertNotIn("return FollowerResult(", control_loop_source)
        self.assertNotIn("latest_stop_details = {", control_loop_source)
        self.assertIn("return control_result(", control_loop_source)
        self.assertIn("ros_shutdown_stop_details()", control_loop_source)
        self.assertIn("command_phase = route_command_phase(", route_step_source)
        self.assertIn(
            "sampling_deadline = viewpoint_sampling_deadline_decision(",
            route_cycle_guard_source,
        )
        self.assertIn(
            "goal_decision = acquisition_goal_decision(",
            step_cycle_guard_source,
        )
        control_loop_tree = ast.parse(control_loop_source)
        control_loop_class = next(
            node
            for node in control_loop_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ControlLoopRuntimeMixin"
        )
        run_node = next(
            node
            for node in control_loop_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        run_source = ast.get_source_segment(control_loop_source, run_node)
        step_cycle_guard_tree = ast.parse(step_cycle_guard_source)
        step_cycle_guard_class = next(
            node
            for node in step_cycle_guard_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "StepCycleGuardRuntimeMixin"
        )
        motion_cycle_guard_tree = ast.parse(motion_cycle_guard_source)
        motion_cycle_guard_class = next(
            node
            for node in motion_cycle_guard_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "MotionCycleGuardRuntimeMixin"
        )
        motion_cycle_guard_node = next(
            node
            for node in motion_cycle_guard_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_motion_cycle_guard_decision"
        )
        motion_cycle_guard_method_source = ast.get_source_segment(
            motion_cycle_guard_source,
            motion_cycle_guard_node,
        )
        lifecycle_node = next(
            node
            for node in step_cycle_guard_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_waypoint_lifecycle_decision"
        )
        lifecycle_source = ast.get_source_segment(
            step_cycle_guard_source,
            lifecycle_node,
        )
        prepare_node = next(
            node
            for node in control_loop_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_prepare_command_for_publication"
        )
        prepare_source = ast.get_source_segment(
            control_loop_source,
            prepare_node,
        )
        route_step_tree = ast.parse(route_step_source)
        route_step_class = next(
            node
            for node in route_step_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "RouteStepResolutionRuntimeMixin"
        )
        resolve_step_node = next(
            node
            for node in route_step_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_resolve_control_step"
        )
        resolve_step_source = ast.get_source_segment(
            route_step_source,
            resolve_step_node,
        )
        route_admission_node = next(
            node
            for node in step_cycle_guard_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_execution_route_admission_decision"
        )
        route_admission_source = ast.get_source_segment(
            step_cycle_guard_source,
            route_admission_node,
        )
        corner_evidence_node = next(
            node
            for node in step_cycle_guard_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_prepare_certified_corner_stop_evidence"
        )
        corner_evidence_source = ast.get_source_segment(
            step_cycle_guard_source,
            corner_evidence_node,
        )
        startup_admission_node = next(
            node
            for node in step_cycle_guard_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_startup_pose_admission_decision"
        )
        startup_admission_source = ast.get_source_segment(
            step_cycle_guard_source,
            startup_admission_node,
        )
        self.assertIn(
            "lifecycle = self._waypoint_lifecycle_decision(step, pose)",
            step_cycle_guard_source,
        )
        self.assertNotIn(
            "self._waypoint_lifecycle_decision(step, pose)",
            run_source,
        )
        self.assertNotIn("if step.reached_goal:", run_source)
        self.assertNotIn("waypoint_timeout_failure(", run_source)
        for effect in (
            "publish_zero",
            "publish_repeated_zero",
            "_hold_zero_control_period",
            "_append_controller_trace",
            "_publish_velocity_command",
        ):
            self.assertNotIn(effect, lifecycle_source)
        self.assertIn(
            "prepared_command = self._prepare_command_for_publication(",
            run_source,
        )
        self.assertLess(
            run_source.index(
                "motion_guard = self._motion_cycle_guard_decision("
            ),
            run_source.index(
                "prepared_command = self._prepare_command_for_publication("
            ),
        )
        self.assertNotIn("if not command_admission.finite:", run_source)
        self.assertNotIn("command_smoother.apply(", run_source)
        self.assertNotIn("command_shape_interval_sec(", run_source)
        for effect in (
            "publish_zero",
            "publish_repeated_zero",
            "_hold_zero_control_period",
            "_append_controller_trace",
            "_blockage_recovery_outcome",
            "_publish_velocity_command",
            "next_control_loop_timing",
            "time.sleep",
        ):
            self.assertNotIn(effect, prepare_source)
        self.assertIn("self._append_controller_trace(", run_source)
        self.assertIn("self._publish_velocity_command(", run_source)
        self.assertIn(
            "step_guard = self._step_cycle_guard_decision(",
            run_source,
        )
        self.assertNotIn(
            "step_resolution = self._resolve_control_step(pose)",
            run_source,
        )
        self.assertIn(
            "step_resolution = self._resolve_control_step(pose)",
            step_cycle_guard_source,
        )
        self.assertIn(
            "cycle_guard = self._control_cycle_guard_decision(",
            run_source,
        )
        self.assertIn(
            "route_guard = self._route_cycle_guard_decision(",
            run_source,
        )
        self.assertNotIn("self._refresh_dynamic_route(pose)", run_source)
        self.assertNotIn(
            "viewpoint_sampling_deadline_decision(",
            run_source,
        )
        self.assertIn(
            "route_refresh = self._refresh_dynamic_route(pose)",
            route_cycle_guard_source,
        )
        self.assertLess(
            route_cycle_guard_source.index(
                "route_refresh = self._refresh_dynamic_route(pose)"
            ),
            route_cycle_guard_source.index(
                "sampling_deadline = viewpoint_sampling_deadline_decision("
            ),
        )
        for forbidden_effect in (
            "_publish_velocity_command",
            "cmd_vel_pub.publish",
            "create_publisher",
            "_append_controller_trace",
            "time.sleep",
            "finish(",
        ):
            self.assertNotIn(forbidden_effect, route_cycle_guard_source)
        for operation in (
            "self._drain_runtime_callbacks()",
            "self._safety_failure()",
            "self._global_consistency_monitor_failure()",
            "self._current_pose_lookup_with_stale_recovery()",
        ):
            self.assertNotIn(operation, run_source)
            self.assertIn(operation, cycle_guard_source)
        self.assertLess(
            cycle_guard_source.index("self._drain_runtime_callbacks()"),
            cycle_guard_source.index("self._safety_failure()"),
        )
        self.assertLess(
            cycle_guard_source.index("self._safety_failure()"),
            cycle_guard_source.index(
                "self._global_consistency_monitor_failure()"
            ),
        )
        self.assertLess(
            cycle_guard_source.index(
                "self._global_consistency_monitor_failure()"
            ),
            cycle_guard_source.rindex(
                "self._current_pose_lookup_with_stale_recovery()"
            ),
        )
        for forbidden_effect in (
            "_publish_velocity_command",
            "cmd_vel_pub.publish",
            "create_publisher",
            "time.sleep",
            "finish(",
            "_refresh_dynamic_route",
        ):
            self.assertNotIn(forbidden_effect, cycle_guard_source)
        self.assertIn(
            "self._startup_pose_admission_decision(pose)",
            step_cycle_guard_source,
        )
        self.assertNotIn(
            "self._startup_pose_admission_decision(pose)",
            run_source,
        )
        self.assertNotIn("if self.target_index == 0:", run_source)
        self.assertNotIn("initial_pose_failure(", run_source)
        self.assertNotIn("certified_static_startup_decision(", run_source)
        self.assertIn("initial_pose_failure(", startup_admission_source)
        self.assertIn(
            "static_startup_decision_fn(",
            startup_admission_source,
        )
        self.assertIn(
            "static_startup_decision_fn=certified_static_startup_decision",
            control_loop_source,
        )
        for effect in (
            "publish_zero",
            "publish_repeated_zero",
            "_hold_zero_control_period",
            "_append_controller_trace",
            "_publish_velocity_command",
            "next_control_loop_timing",
            "time.sleep",
            "finish(",
        ):
            self.assertNotIn(effect, startup_admission_source)
        startup_effects_start = step_cycle_guard_source.index(
            "startup_admission = self._startup_pose_admission_decision(pose)"
        )
        self.assertLess(
            startup_effects_start,
            step_cycle_guard_source.index(
                "step_resolution = self._resolve_control_step(pose)"
            ),
        )
        startup_effects_source = step_cycle_guard_source[
            startup_effects_start:
        ]
        self.assertIn(
            "StartupPoseAdmissionAction.ZERO_HOLD",
            step_cycle_guard_source,
        )
        self.assertLess(
            startup_effects_source.index("self.publish_zero()"),
            startup_effects_source.index(
                "self._hold_zero_control_period(loop_period_sec)"
            ),
        )
        self.assertNotIn(
            "command_phase = route_command_phase(",
            run_source,
        )
        for operation in (
            "startup_join_action_fn(",
            "controller_config_for_route_kind(",
            "compute_join_anchor_command(",
            "compute_intermediate_terminal_heading_command(",
        ):
            self.assertNotIn(operation, run_source)
            self.assertIn(operation, resolve_step_source)
        for effect in (
            "publish_zero",
            "publish_repeated_zero",
            "_hold_zero_control_period",
            "_log_certified_corner_phase",
            "_execution_route_check",
            "_append_controller_trace",
            "_publish_velocity_command",
            "next_control_loop_timing",
            "time.sleep",
            "finish(",
        ):
            self.assertNotIn(effect, resolve_step_source)
        step_effects_start = step_cycle_guard_source.index(
            "step_resolution = self._resolve_control_step(pose)"
        )
        step_effects_source = step_cycle_guard_source[step_effects_start:]
        self.assertLess(
            step_effects_source.index("self.publish_zero()"),
            step_effects_source.index(
                "self._log_certified_corner_phase("
            ),
        )
        self.assertIn(
            "self._prepare_certified_corner_stop_evidence(",
            step_cycle_guard_source,
        )
        self.assertNotIn(
            "certified_corner_stop_details(",
            run_source,
        )
        self.assertIn(
            "stop_details = certified_corner_stop_details(",
            corner_evidence_source,
        )
        for effect in (
            "publish_zero",
            "publish_repeated_zero",
            "_hold_zero_control_period",
            "_log_certified_corner_phase",
            "_append_controller_trace",
            "_publish_velocity_command",
            "next_control_loop_timing",
            "time.sleep",
            "finish(",
            "latest_stop_details",
        ):
            self.assertNotIn(effect, corner_evidence_source)
        self.assertLess(
            step_effects_source.index("self.publish_zero()"),
            step_effects_source.index(
                "self._prepare_certified_corner_stop_evidence("
            ),
        )
        self.assertLess(
            step_effects_source.index(
                "self._prepare_certified_corner_stop_evidence("
            ),
            step_effects_source.index('event="certified_corner_stop"'),
        )
        self.assertIn(
            "self._execution_route_admission_decision(pose, step)",
            step_cycle_guard_source,
        )
        self.assertNotIn(
            "self._execution_route_admission_decision(pose, step)",
            run_source,
        )
        self.assertNotIn(
            "route_check = self._execution_route_check(pose, step)",
            run_source,
        )
        self.assertIn(
            "route_check = self._execution_route_check(pose, step)",
            route_admission_source,
        )
        self.assertIn(
            "stop_details=route_check.to_log_dict()",
            route_admission_source,
        )
        for effect in (
            "publish_zero",
            "publish_repeated_zero",
            "_hold_zero_control_period",
            "_append_controller_trace",
            "_publish_velocity_command",
            "next_control_loop_timing",
            "time.sleep",
            "finish(",
        ):
            self.assertNotIn(effect, route_admission_source)
        route_admission_effects_start = step_cycle_guard_source.index(
            "route_admission = self._execution_route_admission_decision("
        )
        route_admission_effects = step_cycle_guard_source[
            route_admission_effects_start:
        ]
        self.assertLess(
            route_admission_effects.index(
                "self.publish_repeated_zero()"
            ),
            route_admission_effects.index(
                'event="route_tube_stop"'
            ),
        )
        for forbidden_effect in (
            "_publish_velocity_command",
            "cmd_vel_pub.publish",
            "create_publisher",
            "self._drain_runtime_callbacks()",
            "self._refresh_dynamic_route(",
            "self._motion_command_admission_decision(",
            "self._progress_watchdog_decision(",
            "self._prepare_command_for_publication(",
            "next_control_loop_timing",
            "time.sleep",
            "finish(",
        ):
            self.assertNotIn(forbidden_effect, step_cycle_guard_source)
        cycle_guard_index = run_source.index(
            "cycle_guard = self._control_cycle_guard_decision("
        )
        route_guard_index = run_source.index(
            "route_guard = self._route_cycle_guard_decision("
        )
        step_guard_index = run_source.index(
            "step_guard = self._step_cycle_guard_decision("
        )
        motion_guard_index = run_source.index(
            "motion_guard = self._motion_cycle_guard_decision("
        )
        command_preparation_index = run_source.index(
            "prepared_command = self._prepare_command_for_publication("
        )
        control_trace_index = run_source.index('event="control_cycle"')
        motion_publish_index = run_source.index(
            "self._publish_velocity_command(shaped_command)"
        )
        timing_index = run_source.index("timing = next_control_loop_timing(")
        deadline_update_index = run_source.index(
            "self.control_loop_deadline_sec = timing.next_deadline_sec"
        )
        cadence_sleep_index = run_source.index("time.sleep(timing.sleep_sec)")
        final_zero_index = run_source.index(
            "finally:\n            self.publish_repeated_zero()"
        )
        self.assertLess(cycle_guard_index, route_guard_index)
        self.assertLess(route_guard_index, step_guard_index)
        self.assertLess(step_guard_index, motion_guard_index)
        self.assertLess(motion_guard_index, command_preparation_index)
        self.assertLess(command_preparation_index, control_trace_index)
        self.assertLess(control_trace_index, motion_publish_index)
        self.assertLess(motion_publish_index, timing_index)
        self.assertLess(timing_index, deadline_update_index)
        self.assertLess(deadline_update_index, cadence_sleep_index)
        self.assertLess(cadence_sleep_index, final_zero_index)
        safety_tree = ast.parse(safety_source)
        safety_class = next(
            node
            for node in safety_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SafetyRuntimeMixin"
        )
        motion_admission_node = next(
            node
            for node in safety_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_motion_command_admission_decision"
        )
        motion_admission_source = ast.get_source_segment(
            safety_source,
            motion_admission_node,
        )
        self.assertIn(
            "motion_admission = self._motion_command_admission_decision(",
            motion_cycle_guard_method_source,
        )
        self.assertNotIn(
            "self._motion_command_admission_decision(",
            run_source,
        )
        self.assertNotIn(
            "command_admission = command_admission_decision(",
            run_source,
        )
        self.assertNotIn("clearance_motion_floor_stop_details(", run_source)
        for effect in (
            "publish_zero",
            "publish_repeated_zero",
            "_hold_zero_control_period",
            "_append_controller_trace",
            "_blockage_recovery_outcome",
            "_publish_velocity_command",
        ):
            self.assertNotIn(effect, motion_admission_source)
        self.assertIn(
            "monotonic_fn=time.monotonic",
            control_loop_source,
        )
        for forbidden_effect in (
            "_publish_velocity_command",
            "cmd_vel_pub.publish",
            "create_publisher",
            "self._prepare_command_for_publication(",
            "self._drain_runtime_callbacks()",
            "self._refresh_dynamic_route(",
            "self._resolve_control_step(",
            "next_control_loop_timing",
            "time.sleep",
            "finish(",
        ):
            self.assertNotIn(
                forbidden_effect,
                motion_cycle_guard_method_source,
            )
        self.assertNotIn(
            'event="control_cycle"',
            motion_cycle_guard_method_source,
        )
        self.assertEqual(
            motion_cycle_guard_method_source.count(
                "self._append_controller_trace("
            ),
            1,
        )
        clearance_branch = motion_cycle_guard_method_source[
            motion_cycle_guard_method_source.index(
                "if motion_admission.stop_details is not None:"
            ):
        ]
        self.assertLess(
            clearance_branch.index("self.publish_repeated_zero()"),
            clearance_branch.index('event="motion_floor_zero_hold"'),
        )
        self.assertLess(
            clearance_branch.index('event="motion_floor_zero_hold"'),
            clearance_branch.index("self._blockage_recovery_outcome("),
        )
        stuck_branch = motion_cycle_guard_method_source[
            motion_cycle_guard_method_source.index(
                "if progress_decision.failure:"
            ):
        ]
        self.assertLess(
            stuck_branch.index("self.publish_repeated_zero()"),
            stuck_branch.index("self._blockage_recovery_outcome("),
        )
        self.assertLess(
            stuck_branch.index("self._blockage_recovery_outcome("),
            stuck_branch.index("self._hold_zero_control_period("),
        )
        self.assertIn(
            "progress_decision = self._progress_watchdog_decision(",
            motion_cycle_guard_method_source,
        )
        self.assertNotIn(
            "self._progress_watchdog_decision(",
            run_source,
        )
        self.assertNotIn(
            "progress_failure = self._progress_failure(",
            control_loop_source,
        )
        self.assertNotIn(
            "self.latest_stop_details = stuck_progress_details(",
            control_loop_source,
        )
        self.assertEqual(
            control_loop_source.count("self._blockage_recovery_outcome(")
            + cycle_guard_source.count("self._blockage_recovery_outcome(")
            + motion_cycle_guard_source.count(
                "self._blockage_recovery_outcome("
            ),
            3,
        )
        self.assertNotIn("blockage_recovery_eligible(", control_loop_source)
        self.assertNotIn(
            "blockage_recovery_disposition(",
            control_loop_source,
        )
        self.assertIn(
            "front_evidence = front_sector_recovery_evidence(",
            cycle_guard_source,
        )
        self.assertNotIn(
            'front_evidence.get("source") == "front_sector"',
            control_loop_source,
        )
        self.assertNotIn(
            "sampling_timeout = viewpoint_sampling_timeout_failure(",
            control_loop_source,
        )
        self.assertNotIn(
            "goal_action = acquisition_goal_action(",
            control_loop_source,
        )
        self.assertNotIn("if self.dynamic_join_pending:", control_loop_source)
        self.assertNotIn(
            "if self.start_egress_lock_index is not None:",
            control_loop_source,
        )
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
