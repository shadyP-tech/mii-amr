"""Offline checks for faster, scoped, permit-bound return-to-Start travel."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.aufgabe04.navigation.control.driving_behavior import (
    CommandSmoother,
    CommandSmoothingConfig,
    controller_config_for_route_kind,
)
from scripts.aufgabe04.navigation.control.return_to_start_speed_policy import (
    RETURN_TO_START_ANGULAR_RADPS,
    RETURN_TO_START_LINEAR_MPS,
    controller_for_return_to_start_phase,
    return_to_start_speed_policy_evidence,
    return_to_start_speed_policy_failures,
    validate_return_to_start_speed_evidence,
)
from scripts.aufgabe04.navigation.control.safety_checks import validate_speed_limits
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerConfig,
    compute_certified_corner_transition,
    compute_waypoint_command,
    compute_join_anchor_command,
    compute_start_egress_vertex_command,
    VelocityCommand,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.station_segment.cli import build_parser
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.command_admission import command_admission_decision
from scripts.aufgabe04.navigation.waypoint_follower.startup import certified_static_startup_decision
from scripts.aufgabe04.real_robot.execution.child_runner import build_child_runner_command


class ReturnToStartSpeedPolicyTest(unittest.TestCase):
    def child_arguments(self, *, kind=MissionLegKind.RETURN_TO_START, dry=True):
        profile = SimpleNamespace(
            robot_id="turtlebot1", namespace="", scan_topic="scan",
            odom_topic="odom", cmd_vel_topic="cmd_vel", amcl_topic="amcl_pose",
            map_frame="map", odom_frame="odom", base_frame="base_footprint",
            localization_source="amcl", max_linear_speed_mps=0.055,
            max_angular_speed_radps=0.18, robot_radius_m=0.105,
            use_sim_time=False,
        )
        arguments = dict(
            profile=profile, route_csv=Path("route.csv"),
            diagnostics_json=Path("diagnostics.json"), certificate_json=Path("certificate.json"),
            run_id="return_start", session_root=Path("session"), dry_run=dry,
            candidate_snapshot=Path("snapshot.json"),
            uncertainty_map_yaml=Path("map.yaml"), localization_branch_proof_id="branch",
            odom_execution_certificate_json=Path("odom.json"),
            uncertainty_budget_json=Path("budget.json"),
            mission_leg_evidence_kind=kind, mission_leg_evidence_index=3,
            mission_leg_evidence_target_id="start_candidate",
        )
        if not dry:
            arguments.update(
                mission_leg_kind=kind, mission_leg_index=3,
                mission_leg_target_id="start_candidate", mission_leg_semantic_map_id="arena",
                mission_leg_motion_authorization_json=Path("master.json"),
                mission_leg_motion_permit_json=Path("permit.json"),
                mission_leg_dry_preflight_json=Path("dry_preflight.json"),
                mission_leg_dry_odom_certificate_json=Path("dry_odom.json"),
                mission_leg_dry_uncertainty_budget_json=Path("dry_budget.json"),
                mission_session_id="session",
            )
        return arguments

    def parsed_child(self, *, kind=MissionLegKind.RETURN_TO_START, dry=True):
        command = build_child_runner_command(**self.child_arguments(kind=kind, dry=dry))
        return build_parser().parse_args(command[2:])

    def test_dry_and_live_use_same_faster_envelope(self):
        dry = self.parsed_child()
        live = self.parsed_child(dry=False)
        self.assertEqual(dry.max_linear_mps, 0.15)
        self.assertEqual(dry.max_angular_radps, 0.60)
        self.assertEqual(dry.max_scan_age_sec, 0.25)
        self.assertEqual(dry.uncertainty_braking_latency_distance_m, 0.075)
        self.assertEqual(return_to_start_speed_policy_evidence(dry),
                         return_to_start_speed_policy_evidence(live))
        for args in (dry, live):
            self.assertEqual(return_to_start_speed_policy_failures(
                args, route_kind="admitted_candidate_pose", simulation_only=False,
            ), [])

    def test_existing_legs_retain_profile_caps_and_default_sensor_budget(self):
        for kind in (MissionLegKind.COVERAGE, MissionLegKind.CANDIDATE_PREAPPROACH,
                     MissionLegKind.OPPOSITE_FACE):
            with self.subTest(kind=kind):
                args = self.parsed_child(kind=kind)
                self.assertEqual((args.max_linear_mps, args.max_angular_radps), (0.055, 0.18))
                self.assertEqual(args.max_scan_age_sec, 1.0)
                self.assertEqual(args.uncertainty_braking_latency_distance_m, 0.015)
                self.assertEqual(return_to_start_speed_policy_evidence(args), {})
        self.assertFalse(validate_speed_limits(0.15, 0.60).ok)
        self.assertTrue(validate_speed_limits(0.055, 0.18).ok)

    def test_fast_profile_rejects_simulation(self):
        arguments = self.child_arguments()
        arguments["profile"].use_sim_time = True
        with self.assertRaisesRegex(ValueError, "physical profile"):
            build_child_runner_command(**arguments)

    def test_wrong_route_or_simulation_cannot_raise_ceiling(self):
        args = self.parsed_child()
        for route_kind, simulation_only in (("detected_stand_preapproach", False),
                                             ("admitted_candidate_pose", True)):
            with self.subTest(route_kind=route_kind, simulation_only=simulation_only):
                self.assertTrue(return_to_start_speed_policy_failures(
                    args, route_kind=route_kind, simulation_only=simulation_only,
                ))

    def test_fast_mode_requires_live_permit_and_unchanged_safety_parameters(self):
        mutations = {
            "mission_leg_kind": "coverage", "mission_leg_motion_permit_json": None,
            "allow_sim_time": True, "execution_pose_frame": "map", "operator_note": "loaded",
            "max_linear_mps": 0.22, "max_angular_radps": 1.0,
            "max_scan_age_sec": 1.0, "max_odom_age_sec": 1.0, "max_tf_age_sec": 1.0,
            "min_obstacle_distance_m": 0.10, "front_obstacle_slow_distance_m": 0.20,
            "uncertainty_braking_latency_distance_m": 0.015,
            "disable_command_smoothing": True,
        }
        for name, value in mutations.items():
            with self.subTest(name=name):
                args = self.parsed_child(dry=False)
                setattr(args, name, value)
                self.assertTrue(return_to_start_speed_policy_failures(
                    args, route_kind="admitted_candidate_pose", simulation_only=False,
                ))

    def test_speed_evidence_is_bound_to_the_exact_hashed_dry_file(self):
        args = self.parsed_child(dry=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dry.json"
            payload = {"ok": True, "travel_speed_policy": return_to_start_speed_policy_evidence(args)}
            path.write_text(json.dumps(payload))
            permit = SimpleNamespace(dry_preflight_path=str(path),
                                     mission_leg_kind=MissionLegKind.RETURN_TO_START,
                                     dry_preflight_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            validate_return_to_start_speed_evidence(args, permit)
            args.motion_speed_policy = "exploration"
            with self.assertRaisesRegex(ValueError, "dry/live speed policy mismatch"):
                validate_return_to_start_speed_evidence(args, permit)
            args.motion_speed_policy = "unloaded_return_to_start"
            args.max_linear_mps = 0.20
            with self.assertRaisesRegex(ValueError, "dry/live speed policy mismatch"):
                validate_return_to_start_speed_evidence(args, permit)
            path.write_text(json.dumps({"ok": True}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_return_to_start_speed_evidence(args, permit)
            permit.dry_preflight_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "dry/live speed policy mismatch"):
                validate_return_to_start_speed_evidence(args, permit)

    def controller(self, pose, route):
        config = controller_config_for_route_kind(
            ControllerConfig(max_linear_mps=RETURN_TO_START_LINEAR_MPS,
                             max_angular_radps=RETURN_TO_START_ANGULAR_RADPS),
            "admitted_candidate_pose", physical_waypoint_tolerance_m=0.02,
            physical_goal_tolerance_m=0.03,
        )
        return controller_for_return_to_start_phase(
            config, route_kind="admitted_candidate_pose", pose=pose, waypoints=route,
        )

    def test_open_straight_segment_reaches_new_cruise_cap(self):
        route = [Pose2D(0, 0, math.nan), Pose2D(1, 0, math.nan), Pose2D(2, 0, 0)]
        pose = Pose2D(0.5, 0, 0)
        command = compute_waypoint_command(pose, route, 1, self.controller(pose, route)).command
        self.assertEqual(command.linear_x_mps, 0.15)
        # A collinear intermediate vertex does not force precise travel.
        self.assertEqual(self.controller(Pose2D(0.95, 0, 0), route).max_linear_mps, 0.15)

    def test_terminal_approach_and_heading_keep_precise_caps(self):
        route = [Pose2D(0, 0, math.nan), Pose2D(2, 0, math.pi / 2)]
        pose = Pose2D(1.85, 0, 0)
        command = compute_waypoint_command(pose, route, 1, self.controller(pose, route)).command
        self.assertLessEqual(command.linear_x_mps, 0.055)
        at_goal = Pose2D(2, 0, 0)
        step = compute_waypoint_command(at_goal, route, 1, self.controller(at_goal, route))
        self.assertEqual(step.command.linear_x_mps, 0.0)
        self.assertEqual(step.command.angular_z_radps, 0.18)
        self.assertEqual(step.progress_mode, "terminal_heading")

    def test_material_corner_approach_and_turn_keep_precise_caps(self):
        route = [Pose2D(0, 0, math.nan), Pose2D(1, 0, math.nan), Pose2D(1, 1, 0)]
        pose = Pose2D(0.85, 0, 0)
        approach = compute_certified_corner_transition(pose, route, 1, self.controller(pose, route))
        self.assertLessEqual(approach.step.command.linear_x_mps, 0.055)
        at_corner = Pose2D(1, 0, 0)
        turn = compute_certified_corner_transition(at_corner, route, 1, self.controller(at_corner, route))
        self.assertEqual(turn.step.command.linear_x_mps, 0.0)
        self.assertLessEqual(abs(turn.step.command.angular_z_radps), 0.18)

    def test_ten_hz_route_finishes_faster_without_cutting_the_corner(self):
        route = [Pose2D(0, 0, math.nan), Pose2D(1, 0, math.nan), Pose2D(1, 1, math.pi / 2)]

        def simulate(fast):
            pose = Pose2D(0, 0, 0)
            index = 1
            latch = None
            smoother = CommandSmoother(CommandSmoothingConfig())
            peak_speed = 0.0
            for cycle in range(3000):
                config = self.controller(pose, route)
                if not fast:
                    config = replace(config, max_linear_mps=0.055, max_angular_radps=0.18)
                corner = compute_certified_corner_transition(pose, route, index, config, latch)
                self.assertFalse(corner.failure)
                latch = corner.latch
                step = corner.step or compute_waypoint_command(pose, route, index, config)
                if step.reached_goal:
                    self.assertLessEqual(math.hypot(pose.x_m - 1, pose.y_m - 1), 0.03)
                    return cycle * 0.1, peak_speed
                index = step.target_index
                command = smoother.apply(step.command, dt_sec=0.1)
                peak_speed = max(peak_speed, command.linear_x_mps)
                pose = Pose2D(
                    pose.x_m + command.linear_x_mps * math.cos(pose.yaw_rad) * 0.1,
                    pose.y_m + command.linear_x_mps * math.sin(pose.yaw_rad) * 0.1,
                    pose.yaw_rad + command.angular_z_radps * 0.1,
                )
                # Both legs are axis-aligned; the corner turn must not cut
                # across the inside of their certified 3 cm tracking tube.
                route_distance = min(abs(pose.y_m), abs(pose.x_m - 1))
                self.assertLessEqual(route_distance, 0.03)
            self.fail("bounded closed-loop route did not reach the admitted pose")

        fast_time, peak_speed = simulate(True)
        original_time, _ = simulate(False)
        self.assertAlmostEqual(peak_speed, 0.15)
        self.assertLess(fast_time, original_time * 0.80)

    def test_stationary_turn_never_translates_even_under_pose_drift_or_stale_smoothing(self):
        route = [Pose2D(0, 0, math.nan), Pose2D(0, 0, math.pi / 2)]
        for pose in (Pose2D(0, 0, 0), Pose2D(.02, 0, 0), Pose2D(.04, .02, 0),
                     Pose2D(-.04, 0, 0)):
            with self.subTest(pose=pose):
                config = self.controller(pose, route)
                self.assertEqual(config.max_linear_mps, 0.)
                startup = certified_static_startup_decision(pose, route, tracking_tube_radius_m=.03)
                self.assertEqual(startup.ok, math.hypot(pose.x_m, pose.y_m) <= .03)
                steps = [compute_waypoint_command(pose, route, 1, config),
                         compute_join_anchor_command(pose, route[0], config, join_tolerance_m=.01),
                         compute_start_egress_vertex_command(pose, route, 1, config)]
                for step in steps:
                    if step is None:
                        continue
                    admission = command_admission_decision(
                        step.command, front_clearance_scale=.5,
                        linear_motion_floor_mps=.01, physical_route=True,
                    )
                    smoother = CommandSmoother(CommandSmoothingConfig())
                    smoother.apply(VelocityCommand(.15, .6), dt_sec=10.)
                    shaped = smoother.apply(admission.effective_command, dt_sec=.1)
                    self.assertEqual(shaped.linear_x_mps, 0.)
                    self.assertLessEqual(abs(shaped.angular_z_radps), .18)

    def test_stationary_turn_completes_at_final_heading(self):
        route = [Pose2D(0, 0, math.nan), Pose2D(0, 0, math.pi / 2)]
        pose = Pose2D(0, 0, 0)
        for _ in range(200):
            step = compute_waypoint_command(pose, route, 1, self.controller(pose, route))
            self.assertEqual(step.command.linear_x_mps, 0.)
            if step.reached_goal:
                self.assertLessEqual(abs(pose.yaw_rad - math.pi / 2), .25)
                break
            pose = replace(pose, yaw_rad=pose.yaw_rad + .1 * step.command.angular_z_radps)
        else:
            self.fail("stationary final-heading alignment did not finish")


if __name__ == "__main__":
    unittest.main()
