from __future__ import annotations

import unittest

from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import (
    acquisition_goal_stop_details,
    certified_corner_stop_details,
    certified_static_start_stop_details,
    clearance_motion_floor_stop_details,
    control_result,
    initial_runtime_input_stop_details,
    intermediate_terminal_heading_stop_details,
    noop_result,
    nonfinite_velocity_stop_details,
    ros_shutdown_stop_details,
    viewpoint_sampling_timeout_stop_details,
    waypoint_timeout_stop_details,
    with_controller_trace_failure,
    with_route_check_error,
)


class ControlResultsTest(unittest.TestCase):
    def test_noop_and_shutdown_results_have_explicit_contracts(self):
        noop = noop_result("fewer than two waypoints")
        shutdown = ros_shutdown_stop_details()

        self.assertEqual(noop.status, "noop")
        self.assertEqual(noop.duration_sec, 0.0)
        self.assertFalse(noop.motion_published)
        self.assertEqual(shutdown["reason"], "ROS shutdown")
        self.assertEqual(shutdown["source"], "rclpy")
        self.assertTrue(shutdown["fail_closed"])

    def test_control_result_uses_explicit_runtime_snapshot(self):
        details = {"reason": "blocked", "fail_closed": True}

        result = control_result(
            "stopped",
            "blocked",
            started_at=10.0,
            now_monotonic=12.5,
            distance_estimate_m=0.4,
            motion_published=True,
            stop_details=details,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.duration_sec, 2.5)
        self.assertEqual(result.distance_estimate_m, 0.4)
        self.assertTrue(result.motion_published)
        self.assertIs(result.stop_details, details)

    def test_startup_details_preserve_conflicting_upstream_markers(self):
        upstream = {
            "reason": "localization stop",
            "execution_phase": "upstream_phase",
            "phase": "upstream_source",
        }

        details = initial_runtime_input_stop_details(
            upstream,
            reason="fallback reason",
            motion_published=False,
        )

        self.assertEqual(details["reason"], "localization stop")
        self.assertEqual(details["execution_phase"], "upstream_phase")
        self.assertEqual(details["phase"], "upstream_source")
        self.assertFalse(details["motion_published"])
        self.assertIsNot(details, upstream)

    def test_startup_details_supply_fail_closed_fallback(self):
        details = initial_runtime_input_stop_details(
            None,
            reason="scan unavailable",
            motion_published=False,
        )

        self.assertEqual(details["reason"], "scan unavailable")
        self.assertEqual(details["source"], "initial_runtime_input_wait")
        self.assertEqual(details["execution_phase"], "before_motion")
        self.assertTrue(details["fail_closed"])

    def test_trace_failure_keeps_primary_stop_contract(self):
        details = with_controller_trace_failure(
            {"reason": "route departure", "fail_closed": True},
            "trace write failed",
            fail_closed=True,
        )

        self.assertEqual(details["reason"], "route departure")
        self.assertEqual(details["controller_trace_error"], "trace write failed")
        self.assertEqual(
            details["controller_trace_fault_code"],
            "controller_trace_write_failed",
        )
        self.assertTrue(details["fail_closed"])

    def test_waypoint_timeout_contract(self):
        details = waypoint_timeout_stop_details(
            reason="waypoint timeout",
            route_kind="stand_discovery_corridor",
            elapsed_sec=45.1,
            timeout_sec=45.0,
            target_index=2,
            pursuit_index=3,
            distance_to_target_m=0.11,
            progress_mode="path_tracking",
            axis_acquisition_target_revision=None,
            viewpoint_sampling_target_revision="revision-a",
            robot_x_m=1.0,
            robot_y_m=-0.5,
            robot_yaw_rad=0.2,
        )

        self.assertEqual(details["target_index"], 2)
        self.assertEqual(details["robot_pose"]["y_m"], -0.5)
        self.assertEqual(
            details["viewpoint_sampling_target_revision"],
            "revision-a",
        )
        self.assertTrue(details["fail_closed"])

    def test_clearance_and_nonfinite_contracts(self):
        clearance = clearance_motion_floor_stop_details(
            reason="clearance-limited motion floor",
            command_floor_details={"effective_class": "below_motion_floor"},
            front_clearance_scale=0.1,
            front_clearance_details={"source": "front_sector"},
            target_index=1,
            pursuit_index=1,
            distance_to_target_m=0.2,
            progress_mode="path_tracking",
        )
        nonfinite = nonfinite_velocity_stop_details(
            linear_x_mps=0.02,
            angular_z_radps=float("nan"),
        )

        self.assertEqual(clearance["source"], "linear_motion_floor")
        self.assertEqual(clearance["front_clearance"]["source"], "front_sector")
        self.assertTrue(clearance["fail_closed"])
        self.assertEqual(nonfinite["fault_code"], "nonfinite_velocity_command")
        self.assertTrue(nonfinite["fail_closed"])

    def test_sampling_start_and_goal_contracts(self):
        sampling = viewpoint_sampling_timeout_stop_details(
            reason="viewpoint_sampling_timeout",
            route_kind="viewpoint_sampling",
            phase_elapsed_sec=30.0,
            target_elapsed_sec=4.0,
            phase_timeout_sec=30.0,
            target_timeout_sec=10.0,
        )
        startup = certified_static_start_stop_details(
            {"pose_distance_to_segment_m": 0.04},
            certificate_reason="pose left certified route tube",
        )
        goal = acquisition_goal_stop_details(
            reason="axis_acquisition_timeout",
            route_kind="axis_acquisition",
            hold_elapsed_sec=12.0,
            timeout_sec=12.0,
        )

        self.assertEqual(sampling["target_elapsed_sec"], 4.0)
        self.assertEqual(startup["startup_target_candidates"], [0, 1])
        self.assertEqual(
            startup["certificate_reason"],
            "pose left certified route tube",
        )
        self.assertEqual(goal["hold_elapsed_sec"], 12.0)
        self.assertTrue(goal["fail_closed"])

    def test_corner_terminal_heading_and_route_error_contracts(self):
        corner = certified_corner_stop_details(
            reason="certified corner hard tolerance exceeded",
            route_kind="stand_discovery_corridor",
            target_index=2,
            pursuit_index=2,
            distance_to_vertex_m=0.031,
            release_tolerance_m=0.01,
            hold_tolerance_m=0.025,
            tracking_tube_radius_m=0.03,
            reacquire_attempts=2,
            max_reacquire_attempts=2,
        )
        terminal = intermediate_terminal_heading_stop_details(
            reason="terminal_heading_hold_exceeded",
            route_kind="viewpoint_sampling",
            target_index=3,
            distance_to_target_m=0.04,
            entry_tolerance_m=0.03,
            hold_tolerance_m=0.05,
            distance_comparison_epsilon_m=1.0e-9,
            hold_diagnostics={"observed_distance_m": 0.051},
        )
        route_error = with_route_check_error(
            corner,
            ValueError("malformed route check"),
        )

        self.assertEqual(corner["reacquire_attempts"], 2)
        self.assertEqual(
            terminal["effective_hold_limit_m"],
            0.05 + 1.0e-9,
        )
        self.assertEqual(terminal["observed_distance_m"], 0.051)
        self.assertEqual(route_error["reason"], corner["reason"])
        self.assertEqual(route_error["route_check_error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
