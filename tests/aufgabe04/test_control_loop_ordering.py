from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.control.follower_safety import (
    OBSTACLE_TOO_CLOSE,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerConfig,
    ControllerStep,
    VelocityCommand,
)
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    StartupJoinAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    ExecutionRouteAdmissionStatus,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.command_admission import (
    command_admission_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    BlockageRecoveryAction,
    SimpleWaypointFollowerNode,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    ControlStepAction,
    ControlStepStopKind,
    RouteCommandPhase,
    WaypointLifecycleAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    CertifiedStaticStartupDecision,
    StartupPoseAdmissionAction,
    StartupPoseAdmissionDecision,
)


def _route_tube_stop_node(events: list[str]) -> SimpleWaypointFollowerNode:
    node = object.__new__(SimpleWaypointFollowerNode)
    pose = Pose2D(0.05, 0.04, 0.0)
    step = ControllerStep(
        command=VelocityCommand(0.03, 0.0),
        target_index=1,
        reached_goal=False,
        distance_to_target_m=0.15,
        pursuit_index=1,
        controlled_heading_error_rad=0.0,
    )
    node.waypoints = (Pose2D(0.0, 0.0), Pose2D(0.2, 0.0))
    node.follower_config = FollowerConfig(controller=ControllerConfig())
    node.current_route_kind = "stand_discovery_corridor"
    node.target_index = 1
    node.target_started_at = 0.0
    node.distance_estimate_m = 0.0
    node.motion_published = True
    node.last_pose = pose
    node.latest_stop_details = None
    node.waypoint_provider = None
    node.blockage_recovery_provider = None
    node.dynamic_join_pending = False
    node.dynamic_join_limit_m = None
    node.start_egress_lock_index = None
    node.start_egress_forward_alignment_index = None
    node.reverse_staging = False
    node.certified_static_start_pending = False
    node.certified_corner_latch = None
    node.intermediate_terminal_heading_latch = None
    node.axis_acquisition_hold_started_at = None
    node.viewpoint_sampling_started_at = None
    node.viewpoint_sampling_target_started_at = None
    node.latest_front_clearance_details = None
    node._wait_for_initial_runtime_inputs = lambda _started_at: ""
    node._drain_runtime_callbacks = lambda: None
    node._safety_failure = lambda: ""
    node._global_consistency_monitor_failure = lambda: ""
    node._current_pose_lookup_with_stale_recovery = lambda: SimpleNamespace(
        pose=pose
    )
    node._refresh_dynamic_route = lambda _pose: ""
    node._certified_corner_decision = lambda *_args: SimpleNamespace(
        failure="",
        step=step,
    )
    node._log_certified_corner_phase = lambda _step: None
    node._execution_route_check = lambda *_args: ExecutionRouteCheck(
        ok=False,
        reason="pose left certified route tube",
        pose_distance_to_segment_m=0.04,
        maximum_chord_distance_to_segment_m=0.04,
        active_segment_start_index=0,
        active_segment_end_index=1,
        target_index=1,
        pursuit_index=1,
        tracking_tube_radius_m=0.03,
    )
    node.publish_zero = lambda: events.append("zero")
    node.publish_repeated_zero = lambda: events.append("repeated_zero")
    node._append_controller_trace = lambda **_kwargs: events.append("trace") or ""
    return node


def _startup_route_check(*, ok: bool, target_index: int) -> ExecutionRouteCheck:
    return ExecutionRouteCheck(
        ok=ok,
        reason="" if ok else "pose left certified route tube",
        pose_distance_to_segment_m=0.0 if ok else 0.04,
        maximum_chord_distance_to_segment_m=0.0 if ok else 0.04,
        active_segment_start_index=0,
        active_segment_end_index=1,
        target_index=target_index,
        pursuit_index=target_index,
        tracking_tube_radius_m=0.03,
    )


class ControlLoopOrderingTest(unittest.TestCase):
    def test_startup_pose_admission_skips_after_initial_target(self):
        node = _route_tube_stop_node([])
        node.certified_static_start_pending = True

        decision = node._startup_pose_admission_decision(
            Pose2D(4.0, 4.0, 0.0)
        )

        self.assertIs(decision.action, StartupPoseAdmissionAction.PROCEED)
        self.assertEqual(decision.selected_target_index, 1)
        self.assertFalse(decision.static_start_consumed)
        self.assertTrue(node.certified_static_start_pending)

    def test_startup_pose_admission_initial_failure_preserves_static_gate(self):
        node = _route_tube_stop_node([])
        node.target_index = 0
        node.certified_static_start_pending = True

        decision = node._startup_pose_admission_decision(
            Pose2D(4.0, 4.0, 0.0)
        )

        self.assertIs(decision.action, StartupPoseAdmissionAction.STOP)
        self.assertIsNone(decision.selected_target_index)
        self.assertIn("initial pose", decision.stop_reason)
        self.assertIsNone(decision.stop_details)
        self.assertFalse(decision.static_start_consumed)
        self.assertTrue(node.certified_static_start_pending)

    def test_startup_pose_admission_accepts_first_vertex_without_effects(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node.target_index = 0
        node.certified_static_start_pending = True
        route_check = _startup_route_check(ok=True, target_index=0)

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.certified_static_startup_decision",
            return_value=CertifiedStaticStartupDecision(
                ok=True,
                target_index=0,
                route_check=route_check,
            ),
        ):
            decision = node._startup_pose_admission_decision(
                Pose2D(0.0, 0.0, 0.0)
            )

        self.assertIs(decision.action, StartupPoseAdmissionAction.PROCEED)
        self.assertEqual(decision.selected_target_index, 0)
        self.assertTrue(decision.static_start_consumed)
        self.assertFalse(node.certified_static_start_pending)
        self.assertEqual(events, [])

    def test_startup_pose_admission_selects_next_vertex_and_requests_hold(self):
        events: list[tuple[str, object]] = []
        node = _route_tube_stop_node([])
        node.target_index = 0
        node.certified_static_start_pending = True
        node._reset_progress_watchdog = lambda now: events.append(
            ("reset_progress", now)
        )
        route_check = _startup_route_check(ok=True, target_index=1)

        with (
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.certified_static_startup_decision",
                return_value=CertifiedStaticStartupDecision(
                    ok=True,
                    target_index=1,
                    route_check=route_check,
                ),
            ),
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.time.monotonic",
                side_effect=(4.0, 4.1),
            ),
        ):
            decision = node._startup_pose_admission_decision(
                Pose2D(0.05, 0.0, 0.0)
            )

        self.assertIs(decision.action, StartupPoseAdmissionAction.ZERO_HOLD)
        self.assertEqual(decision.selected_target_index, 1)
        self.assertTrue(decision.static_start_consumed)
        self.assertFalse(node.certified_static_start_pending)
        self.assertEqual(node.target_index, 1)
        self.assertEqual(node.target_started_at, 4.0)
        self.assertEqual(events, [("reset_progress", 4.1)])

    def test_startup_pose_admission_returns_certified_failure_evidence(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node.target_index = 0
        node.certified_static_start_pending = True
        route_check = _startup_route_check(ok=False, target_index=1)

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.certified_static_startup_decision",
            return_value=CertifiedStaticStartupDecision(
                ok=False,
                target_index=None,
                route_check=route_check,
            ),
        ):
            decision = node._startup_pose_admission_decision(
                Pose2D(0.0, 0.0, 0.0)
            )

        self.assertIs(decision.action, StartupPoseAdmissionAction.STOP)
        self.assertEqual(
            decision.stop_reason,
            "pose outside certified startup segment",
        )
        self.assertIsNone(decision.selected_target_index)
        self.assertEqual(
            decision.stop_details["certificate_reason"],
            "pose left certified route tube",
        )
        self.assertTrue(decision.static_start_consumed)
        self.assertFalse(node.certified_static_start_pending)
        self.assertEqual(events, [])

    def test_certified_corner_stop_evidence_includes_failure_route_check(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        failed_step = ControllerStep(
            command=VelocityCommand(0.0, 0.2),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.04,
            pursuit_index=1,
            controlled_heading_error_rad=0.3,
        )

        evidence = node._prepare_certified_corner_stop_evidence(
            Pose2D(0.05, 0.04, 0.0),
            failed_step,
            "certified corner hard tolerance exceeded",
        )

        self.assertIs(evidence.step, failed_step)
        self.assertEqual(
            evidence.stop_details["reason"],
            "certified corner hard tolerance exceeded",
        )
        self.assertEqual(evidence.stop_details["target_index"], 1)
        self.assertIsNotNone(evidence.route_check)
        self.assertEqual(events, [])

    def test_certified_corner_stop_evidence_preserves_route_check_error(self):
        node = _route_tube_stop_node([])
        node._execution_route_check = Mock(
            side_effect=ValueError("malformed route certificate")
        )
        failed_step = ControllerStep(
            command=VelocityCommand(0.0, 0.2),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.04,
            pursuit_index=1,
            controlled_heading_error_rad=0.3,
        )

        evidence = node._prepare_certified_corner_stop_evidence(
            Pose2D(0.05, 0.04, 0.0),
            failed_step,
            "certified corner hard tolerance exceeded",
        )

        self.assertIsNone(evidence.route_check)
        self.assertEqual(
            evidence.stop_details["route_check_error"],
            "malformed route certificate",
        )
        self.assertEqual(
            evidence.stop_details["route_check_error_type"],
            "ValueError",
        )
        self.assertEqual(
            evidence.stop_details["reason"],
            "certified corner hard tolerance exceeded",
        )

    def test_certified_corner_stop_evidence_skips_nonphysical_route_check(self):
        node = _route_tube_stop_node([])
        node.current_route_kind = "axis_acquisition"
        node._execution_route_check = Mock()
        failed_step = ControllerStep(
            command=VelocityCommand(0.0, 0.2),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.04,
            pursuit_index=1,
            controlled_heading_error_rad=0.3,
        )

        evidence = node._prepare_certified_corner_stop_evidence(
            Pose2D(0.05, 0.04, 0.0),
            failed_step,
            "certified corner hard tolerance exceeded",
        )

        self.assertIsNone(evidence.route_check)
        node._execution_route_check.assert_not_called()

    def test_execution_route_admission_skips_nonphysical_route(self):
        node = _route_tube_stop_node([])
        node.current_route_kind = "axis_acquisition"
        node._execution_route_check = Mock()
        step = ControllerStep(
            command=VelocityCommand(0.03, 0.0),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.15,
            pursuit_index=1,
            controlled_heading_error_rad=0.0,
        )

        admission = node._execution_route_admission_decision(
            Pose2D(0.05, 0.04, 0.0),
            step,
        )

        self.assertIs(
            admission.status,
            ExecutionRouteAdmissionStatus.SKIPPED,
        )
        self.assertIsNone(admission.route_check)
        self.assertIsNone(admission.stop_details)
        node._execution_route_check.assert_not_called()

    def test_execution_route_admission_skips_unbounded_dynamic_join(self):
        node = _route_tube_stop_node([])
        node.dynamic_join_pending = True
        node.dynamic_join_limit_m = 0.04
        node._execution_route_check = Mock()
        step = ControllerStep(
            command=VelocityCommand(0.03, 0.0),
            target_index=0,
            reached_goal=False,
            distance_to_target_m=0.15,
            pursuit_index=0,
            controlled_heading_error_rad=0.0,
        )

        admission = node._execution_route_admission_decision(
            Pose2D(0.05, 0.04, 0.0),
            step,
        )

        self.assertIs(
            admission.status,
            ExecutionRouteAdmissionStatus.SKIPPED,
        )
        node._execution_route_check.assert_not_called()

    def test_execution_route_admission_checks_bounded_dynamic_join(self):
        node = _route_tube_stop_node([])
        node.dynamic_join_pending = True
        node.dynamic_join_limit_m = 0.03
        route_check = ExecutionRouteCheck(
            ok=False,
            reason="pose left certified route tube",
            pose_distance_to_segment_m=0.04,
            maximum_chord_distance_to_segment_m=0.04,
            active_segment_start_index=0,
            active_segment_end_index=1,
            target_index=0,
            pursuit_index=0,
            tracking_tube_radius_m=0.03,
        )
        node._execution_route_check = Mock(return_value=route_check)
        step = ControllerStep(
            command=VelocityCommand(0.03, 0.0),
            target_index=0,
            reached_goal=False,
            distance_to_target_m=0.15,
            pursuit_index=0,
            controlled_heading_error_rad=0.0,
        )

        admission = node._execution_route_admission_decision(
            Pose2D(0.05, 0.04, 0.0),
            step,
        )

        self.assertIs(
            admission.status,
            ExecutionRouteAdmissionStatus.STOP,
        )
        self.assertIs(admission.route_check, route_check)
        self.assertEqual(
            admission.stop_details["reason"],
            "pose left certified route tube",
        )
        node._execution_route_check.assert_called_once()

    def test_execution_route_admission_returns_successful_check(self):
        node = _route_tube_stop_node([])
        route_check = ExecutionRouteCheck(
            ok=True,
            reason="",
            pose_distance_to_segment_m=0.0,
            maximum_chord_distance_to_segment_m=0.0,
            active_segment_start_index=0,
            active_segment_end_index=1,
            target_index=1,
            pursuit_index=1,
            tracking_tube_radius_m=0.03,
        )
        node._execution_route_check = Mock(return_value=route_check)
        step = ControllerStep(
            command=VelocityCommand(0.03, 0.0),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.15,
            pursuit_index=1,
            controlled_heading_error_rad=0.0,
        )

        admission = node._execution_route_admission_decision(
            Pose2D(0.05, 0.0, 0.0),
            step,
        )

        self.assertIs(
            admission.status,
            ExecutionRouteAdmissionStatus.ADMITTED,
        )
        self.assertIs(admission.route_check, route_check)
        self.assertIsNone(admission.stop_details)

    def test_control_step_resolution_dynamic_join_stop_has_no_effects(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node.dynamic_join_pending = True
        node.dynamic_join_limit_m = None

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.certified_startup_join_action",
            return_value=(
                StartupJoinAction.STOP,
                {"reason": "dynamic join envelope unavailable"},
            ),
        ):
            resolution = node._resolve_control_step(
                Pose2D(0.05, 0.04, 0.0)
            )

        self.assertIs(resolution.action, ControlStepAction.STOP)
        self.assertIs(
            resolution.command_phase,
            RouteCommandPhase.DYNAMIC_JOIN,
        )
        self.assertIs(
            resolution.stop_kind,
            ControlStepStopKind.DYNAMIC_JOIN,
        )
        self.assertEqual(
            resolution.stop_reason,
            "dynamic join envelope unavailable",
        )
        self.assertEqual(events, [])

    def test_control_step_resolution_dynamic_join_handoff_updates_state(self):
        effects: list[tuple[str, object]] = []
        node = _route_tube_stop_node([])
        node.dynamic_join_pending = True
        node.dynamic_join_limit_m = 0.03
        node.target_index = 1
        node._clear_intermediate_terminal_heading_latch = (
            lambda *, target_changed: effects.append(
                ("clear_terminal_heading", target_changed)
            )
        )
        node._reset_progress_watchdog = lambda now: effects.append(
            ("reset_progress", now)
        )

        with (
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.certified_startup_join_action",
                return_value=(StartupJoinAction.ZERO, None),
            ),
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.time.monotonic",
                side_effect=(4.0, 4.1),
            ),
        ):
            resolution = node._resolve_control_step(
                Pose2D(0.05, 0.04, 0.0)
            )

        self.assertIs(resolution.action, ControlStepAction.ZERO_HOLD)
        self.assertFalse(node.dynamic_join_pending)
        self.assertIsNone(node.dynamic_join_limit_m)
        self.assertEqual(node.target_index, 0)
        self.assertEqual(node.target_started_at, 4.0)
        self.assertEqual(
            effects,
            [
                ("clear_terminal_heading", True),
                ("reset_progress", 4.1),
            ],
        )

    def test_control_step_resolution_start_egress_requests_zero_hold(self):
        node = _route_tube_stop_node([])
        node.start_egress_lock_index = 1
        node._start_egress_command = Mock(return_value=None)

        resolution = node._resolve_control_step(
            Pose2D(0.05, 0.04, 0.0)
        )

        self.assertIs(resolution.action, ControlStepAction.ZERO_HOLD)
        self.assertIs(
            resolution.command_phase,
            RouteCommandPhase.START_EGRESS,
        )
        self.assertIsNone(resolution.step)
        node._start_egress_command.assert_called_once()

    def test_control_step_resolution_corner_stop_defers_all_effects(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        failed_step = ControllerStep(
            command=VelocityCommand(0.0, 0.2),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.04,
            pursuit_index=1,
            controlled_heading_error_rad=0.3,
        )
        node._certified_corner_decision = Mock(
            return_value=SimpleNamespace(
                failure="certified corner hard tolerance exceeded",
                step=failed_step,
            )
        )
        node._log_certified_corner_phase = Mock()
        node._execution_route_check = Mock()

        resolution = node._resolve_control_step(
            Pose2D(0.05, 0.04, 0.0)
        )

        self.assertIs(resolution.action, ControlStepAction.STOP)
        self.assertIs(
            resolution.stop_kind,
            ControlStepStopKind.CERTIFIED_CORNER,
        )
        self.assertIs(resolution.step, failed_step)
        self.assertIs(resolution.corner_step, failed_step)
        self.assertIsNone(resolution.stop_details)
        self.assertEqual(events, [])
        node._log_certified_corner_phase.assert_not_called()
        node._execution_route_check.assert_not_called()

    def test_command_preparation_rejects_nonfinite_before_smoothing(self):
        node = _route_tube_stop_node([])
        node.command_smoother = Mock()
        node.last_command_shape_at = 2.0
        command_admission = command_admission_decision(
            VelocityCommand(0.03, math.nan),
            front_clearance_scale=1.0,
            linear_motion_floor_mps=(
                node.follower_config.linear_motion_floor_mps
            ),
            physical_route=True,
        )

        prepared = node._prepare_command_for_publication(
            command_admission,
            now_monotonic=3.0,
            loop_period_sec=0.1,
        )

        self.assertIsNone(prepared.shaped_command)
        self.assertIsNone(prepared.shape_dt_sec)
        self.assertEqual(
            prepared.stop_details["fault_code"],
            "nonfinite_velocity_command",
        )
        node.command_smoother.apply.assert_not_called()
        self.assertEqual(node.last_command_shape_at, 2.0)

    def test_command_preparation_shapes_and_builds_trace_diagnostics(self):
        node = _route_tube_stop_node([])
        shaped_command = VelocityCommand(0.01, 0.05)
        node.command_smoother = Mock()
        node.command_smoother.apply.return_value = shaped_command
        node.last_command_shape_at = None
        command_admission = command_admission_decision(
            VelocityCommand(0.03, 0.2),
            front_clearance_scale=0.5,
            linear_motion_floor_mps=(
                node.follower_config.linear_motion_floor_mps
            ),
            physical_route=True,
        )

        prepared = node._prepare_command_for_publication(
            command_admission,
            now_monotonic=3.0,
            loop_period_sec=0.1,
        )

        self.assertIsNone(prepared.stop_details)
        self.assertIs(prepared.shaped_command, shaped_command)
        self.assertEqual(prepared.shape_dt_sec, 0.1)
        node.command_smoother.apply.assert_called_once_with(
            VelocityCommand(0.015, 0.2),
            dt_sec=0.1,
        )
        self.assertEqual(node.last_command_shape_at, 3.0)
        self.assertEqual(
            prepared.trace_diagnostics,
            {
                "driving_behavior": {
                    "command_smoothing_enabled": True,
                    "unshaped_effective_command": {
                        "linear_x_mps": 0.015,
                        "angular_z_radps": 0.2,
                    },
                    "shape_dt_sec": 0.1,
                }
            },
        )

    def test_waypoint_lifecycle_target_transition_resets_state_once(self):
        events: list[tuple[str, object]] = []
        node = _route_tube_stop_node([])
        node.target_index = 0
        node.certified_corner_latch = object()
        node._clear_intermediate_terminal_heading_latch = (
            lambda *, target_changed: events.append(
                ("clear_terminal_heading", target_changed)
            )
        )
        node._reset_progress_watchdog = lambda now: events.append(
            ("reset_progress", now)
        )
        step = ControllerStep(
            command=VelocityCommand(0.03, 0.0),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.15,
            pursuit_index=1,
            controlled_heading_error_rad=0.0,
        )

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.time.monotonic",
            side_effect=(10.0, 10.1, 10.2),
        ):
            decision = node._waypoint_lifecycle_decision(
                step,
                Pose2D(0.05, 0.04, 0.0),
            )

        self.assertIs(decision.action, WaypointLifecycleAction.PROCEED)
        self.assertEqual(node.target_index, 1)
        self.assertIsNone(node.certified_corner_latch)
        self.assertEqual(node.target_started_at, 10.0)
        self.assertEqual(
            events,
            [
                ("clear_terminal_heading", True),
                ("reset_progress", 10.1),
            ],
        )

    def test_waypoint_lifecycle_hold_updates_clock_without_motion_effects(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node.current_route_kind = "axis_acquisition"
        node.waypoint_provider = object()
        reached_goal_step = ControllerStep(
            command=VelocityCommand(0.0, 0.0),
            target_index=1,
            reached_goal=True,
            distance_to_target_m=0.0,
            pursuit_index=1,
            controlled_heading_error_rad=0.0,
        )

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.time.monotonic",
            return_value=10.0,
        ):
            decision = node._waypoint_lifecycle_decision(
                reached_goal_step,
                Pose2D(0.2, 0.0, 0.0),
            )

        self.assertIs(decision.action, WaypointLifecycleAction.HOLD)
        self.assertEqual(node.axis_acquisition_hold_started_at, 10.0)
        self.assertEqual(events, [])

    def test_waypoint_lifecycle_timeout_returns_pose_bound_evidence(self):
        node = _route_tube_stop_node([])
        node.follower_config = FollowerConfig(
            controller=ControllerConfig(),
            waypoint_timeout_sec=0.5,
        )
        node.target_started_at = 1.0
        node.axis_acquisition_target_revision = 3
        node.viewpoint_sampling_target_revision = 7
        pose = Pose2D(0.05, 0.04, 0.25)
        step = ControllerStep(
            command=VelocityCommand(0.03, 0.0),
            target_index=1,
            reached_goal=False,
            distance_to_target_m=0.15,
            pursuit_index=1,
            controlled_heading_error_rad=0.0,
        )

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_loop.time.monotonic",
            return_value=1.6,
        ):
            decision = node._waypoint_lifecycle_decision(step, pose)

        self.assertIs(decision.action, WaypointLifecycleAction.STOP)
        self.assertEqual(decision.stop_reason, "waypoint timeout")
        self.assertEqual(decision.evaluated_at, 1.6)
        self.assertEqual(decision.stop_details["target_index"], 1)
        self.assertEqual(decision.stop_details["robot_pose"]["yaw_rad"], 0.25)
        self.assertEqual(
            decision.stop_details["axis_acquisition_target_revision"],
            3,
        )
        self.assertEqual(
            decision.stop_details["viewpoint_sampling_target_revision"],
            7,
        )

    def test_ros_shutdown_returns_stopped_result_after_final_zero(self):
        events: list[str] = []
        node = object.__new__(SimpleWaypointFollowerNode)
        node.waypoints = (Pose2D(0.0, 0.0), Pose2D(0.2, 0.0))
        node.current_route_kind = "stand_discovery_corridor"
        node.follower_config = FollowerConfig(controller=ControllerConfig())
        node.distance_estimate_m = 0.12
        node.motion_published = True
        node._wait_for_initial_runtime_inputs = lambda _started_at: ""
        node.publish_repeated_zero = lambda: events.append("repeated_zero")

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: False),
        ):
            result = node.run()
        events.append("return")

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "ROS shutdown")
        self.assertEqual(result.distance_estimate_m, 0.12)
        self.assertTrue(result.motion_published)
        self.assertEqual(
            result.stop_details,
            {
                "reason": "ROS shutdown",
                "source": "rclpy",
                "phase": "control_loop",
                "fail_closed": True,
            },
        )
        self.assertEqual(events, ["repeated_zero", "repeated_zero", "return"])

    def test_run_applies_full_zero_cycle_for_startup_target_handoff(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node._startup_pose_admission_decision = Mock(
            return_value=StartupPoseAdmissionDecision(
                StartupPoseAdmissionAction.ZERO_HOLD,
                selected_target_index=1,
                static_start_consumed=True,
            )
        )
        node._hold_zero_control_period = (
            lambda _period: events.append("hold")
        )
        ok_values = iter((True, False))

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: next(ok_values)),
        ):
            result = node.run()

        self.assertEqual(result.stop_reason, "ROS shutdown")
        self.assertEqual(
            events,
            ["repeated_zero", "zero", "hold", "repeated_zero"],
        )
        node._startup_pose_admission_decision.assert_called_once()

    def test_route_tube_stop_zeroes_before_trace_and_return(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run()
        events.append("return")

        self.assertEqual(result.status, "stopped")
        self.assertEqual(
            result.stop_reason,
            "pose left certified route tube",
        )
        # The first repeated zero is startup admission.  The route-tube stop
        # must add another one before trace I/O, with final cleanup after it.
        stop_zero_index = events.index("repeated_zero", 1)
        self.assertLess(stop_zero_index, events.index("trace"))
        self.assertLess(stop_zero_index, events.index("return"))

    def test_sampling_deadline_still_zeroes_and_returns_evidence(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node.current_route_kind = "viewpoint_sampling"
        node.follower_config = FollowerConfig(
            controller=ControllerConfig(),
            viewpoint_sampling_timeout_sec=0.5,
            viewpoint_sampling_target_timeout_sec=1.0,
        )

        with (
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
                SimpleNamespace(ok=lambda: True),
            ),
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
                side_effect=(0.0, 0.1, 0.5),
            ),
        ):
            result = node.run()

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "viewpoint sampling timeout")
        self.assertEqual(result.stop_details["phase_elapsed_sec"], 0.5)
        self.assertEqual(result.stop_details["target_elapsed_sec"], 0.5)
        self.assertEqual(
            events,
            ["repeated_zero", "repeated_zero", "repeated_zero"],
        )

    def test_acquisition_timeout_reuses_clock_and_stops(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node.current_route_kind = "axis_acquisition"
        node.waypoint_provider = lambda _pose: None
        node.axis_acquisition_hold_started_at = 0.0
        node.follower_config = FollowerConfig(
            controller=ControllerConfig(),
            axis_acquisition_wait_timeout_sec=0.5,
        )
        reached_goal_step = ControllerStep(
            command=VelocityCommand(0.0, 0.0),
            target_index=1,
            reached_goal=True,
            distance_to_target_m=0.0,
            pursuit_index=1,
            controlled_heading_error_rad=0.0,
        )
        node._certified_corner_decision = lambda *_args: SimpleNamespace(
            failure="",
            step=reached_goal_step,
        )

        with (
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
                SimpleNamespace(ok=lambda: True),
            ),
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
                side_effect=(0.0, 0.1, 0.2, 0.5, 0.6),
            ),
        ):
            result = node.run()

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "axis acquisition timeout")
        self.assertEqual(result.stop_details["hold_elapsed_sec"], 0.5)
        self.assertEqual(node.axis_acquisition_hold_started_at, 0.0)
        self.assertEqual(
            events,
            ["repeated_zero", "repeated_zero", "repeated_zero"],
        )

    def test_clearance_recovery_keeps_zero_trace_replan_hold_order(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node._execution_route_check = lambda *_args: ExecutionRouteCheck(
            ok=True,
            reason="",
            pose_distance_to_segment_m=0.0,
            maximum_chord_distance_to_segment_m=0.0,
            active_segment_start_index=0,
            active_segment_end_index=1,
            target_index=1,
            pursuit_index=1,
            tracking_tube_radius_m=0.03,
        )
        node._motion_clearance_linear_scale = lambda _linear_x_mps: 0.1
        node.latest_front_clearance_details = {"source": "front_sector"}
        node.blockage_recovery_provider = object()
        node._attempt_blockage_recovery = (
            lambda *_args: events.append("recovery")
            or BlockageRecoveryAction.CLEARED
        )
        node._hold_zero_control_period = (
            lambda _period: events.append("hold")
        )
        ok_values = iter((True, False))

        with (
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
                SimpleNamespace(ok=lambda: next(ok_values)),
            ),
            patch(
                "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
                side_effect=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            ),
        ):
            result = node.run()

        self.assertEqual(result.stop_reason, "ROS shutdown")
        stop_zero_index = events.index("repeated_zero", 1)
        self.assertLess(stop_zero_index, events.index("trace"))
        self.assertLess(events.index("trace"), events.index("recovery"))
        self.assertLess(events.index("recovery"), events.index("zero"))
        self.assertLess(events.index("zero"), events.index("hold"))

    def test_obstacle_clear_keeps_zero_pose_recovery_hold_order(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        pose = Pose2D(0.05, 0.04, 0.0)
        node.latest_stop_details = {
            "front_clearance": {"source": "front_sector"}
        }
        node.blockage_recovery_provider = object()
        node._safety_failure = lambda: OBSTACLE_TOO_CLOSE
        node._current_pose_lookup_with_stale_recovery = (
            lambda: events.append("pose_lookup")
            or SimpleNamespace(pose=pose)
        )
        node._attempt_blockage_recovery = (
            lambda *_args: events.append("recovery")
            or BlockageRecoveryAction.CLEARED
        )
        node._hold_zero_control_period = (
            lambda _period: events.append("hold")
        )
        ok_values = iter((True, False))

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: next(ok_values)),
        ):
            result = node.run()

        self.assertEqual(result.stop_reason, "ROS shutdown")
        self.assertEqual(
            events,
            [
                "repeated_zero",
                "repeated_zero",
                "pose_lookup",
                "recovery",
                "zero",
                "hold",
                "repeated_zero",
            ],
        )

    def test_obstacle_recovery_without_pose_stops_with_original_reason(self):
        events: list[str] = []
        node = _route_tube_stop_node(events)
        node.latest_stop_details = {
            "front_clearance": {"source": "front_sector"}
        }
        node.blockage_recovery_provider = object()
        node._safety_failure = lambda: OBSTACLE_TOO_CLOSE
        node._current_pose_lookup_with_stale_recovery = (
            lambda: events.append("pose_lookup")
            or SimpleNamespace(pose=None)
        )
        node._attempt_blockage_recovery = lambda *_args: self.fail(
            "recovery must not run without a fresh pose"
        )

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ):
            result = node.run()

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, OBSTACLE_TOO_CLOSE)
        self.assertEqual(
            events,
            [
                "repeated_zero",
                "repeated_zero",
                "pose_lookup",
                "repeated_zero",
                "repeated_zero",
            ],
        )


if __name__ == "__main__":
    unittest.main()
