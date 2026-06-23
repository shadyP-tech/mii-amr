from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "aufgabe03"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from waypoint_following.models import Pose2D, Waypoint  # noqa: E402
from waypoint_following import post_replan_recovery as recovery  # noqa: E402
from waypoint_following.scan_safety import evaluate_scan_safety  # noqa: E402


class FakeScan:
    def __init__(self, points, *, far_range=2.0):
        self.angle_min = -math.pi
        self.angle_increment = math.radians(1.0)
        self.range_min = 0.05
        self.range_max = 5.0
        self.ranges = [far_range] * 361
        for x, y in points:
            angle = math.atan2(y, x)
            distance = math.hypot(x, y)
            index = int(round((angle - self.angle_min) / self.angle_increment))
            if 0 <= index < len(self.ranges):
                self.ranges[index] = distance


class FakeRouteState:
    def __init__(self, points):
        self._waypoints = [
            Waypoint(index, float(x), float(y))
            for index, (x, y) in enumerate(points)
        ]

    def remaining_tracking_points(self):
        return self._waypoints

    def remaining(self):
        return self._waypoints


class FakeNode:
    def __init__(
        self,
        points,
        route_points,
        *,
        clearance_mode=recovery.POST_REPLAN_CLEARANCE_ROUTE_AWARE,
        steering_mode=recovery.POST_REPLAN_ESCAPE_STEERING_AUTO,
        preview_distance_m=0.0,
        clear_scan_samples=1,
    ):
        self.args = SimpleNamespace(
            scan_half_angle_deg=35.0,
            hard_stop_range_m=0.16,
            min_scan_range_m=0.40,
            rotation_stop_range_m=0.18,
            enable_lidar_map_replan=True,
            post_replan_recovery=recovery.DEFAULT_POST_REPLAN_RECOVERY,
            post_replan_clearance_mode=clearance_mode,
            post_replan_escape_steering_mode=steering_mode,
            post_replan_route_clearance_preview_distance_m=preview_distance_m,
            post_replan_escape_distance_m=0.12,
            post_replan_clear_scan_samples=clear_scan_samples,
            post_replan_timeout_sec=4.0,
            post_replan_escape_linear_speed_mps=0.02,
            post_replan_align_heading_error_deg=12.0,
            pure_pursuit_max_track_angular_speed_radps=0.09,
            robot_footprint_radius_m=0.10,
            run_local_map_clearance_margin_m=0.04,
            control_rate_hz=20.0,
            verbose=False,
        )
        self.last_scan = FakeScan(points)
        self.last_scan_received_sec = time.time()
        self.route_state = FakeRouteState(route_points)
        self.active_route_generation_id = 0
        self.post_replan_recovery = None
        self.max_post_replan_recovery_clear_count = 0
        self.last_post_replan_recovery_phase = ""
        self.last_post_replan_recovery_log_sec = None
        self.last_post_replan_recovery_status = ""
        self.last_post_replan_route_clearance_reason = ""
        self.last_post_replan_route_corridor_min_distance_m = None
        self.last_post_replan_route_corridor_blocked_count = 0
        self.last_post_replan_route_clear_side_obstacle_count = 0
        self.last_post_replan_route_corridor_preview_distance_m = 0.0
        self.last_post_replan_route_corridor_nearest_blocked_segment_index = None
        self.last_post_replan_route_corridor_nearest_blocked_progress_m = None
        self.last_post_replan_route_corridor_nearest_blocked_penetration_m = None
        self.last_post_replan_route_corridor_nearest_blocked_x_m = None
        self.last_post_replan_route_corridor_nearest_blocked_y_m = None
        self.last_post_replan_route_corridor_nearest_blocked_range_m = None
        self.last_post_replan_route_corridor_nearest_blocked_angle_deg = None
        self.last_post_replan_escape_route_blocked_streak = 0
        self.last_post_replan_escape_route_blocked_tolerated_count = 0
        self.last_post_replan_escape_route_block_decision = ""
        self.post_replan_route_block_repair_count = 0
        self.post_replan_route_block_repair_status = ""
        self.post_replan_route_block_repair_signature = ""
        self.post_replan_route_block_repair_extra_update_used = False
        self.post_replan_route_block_repair_failure_reason = ""
        self.last_post_replan_recovery_escape_steering_mode_resolved = ""
        self.last_post_replan_recovery_escape_odom_distance_m = None
        self.last_post_replan_recovery_escape_map_distance_m = None
        self.last_post_replan_recovery_escape_odom_stamp_delta_sec = None
        self.last_post_replan_recovery_escape_progress_source = ""
        self.last_post_replan_recovery_escape_no_motion_reason = ""
        self.odom_pose = None
        self.published = []
        self.motion_samples = []

    def evaluate_current_scan_safety(self, mode):
        return evaluate_scan_safety(
            self.last_scan.ranges,
            self.last_scan.angle_min,
            self.last_scan.angle_increment,
            self.last_scan.range_min,
            self.last_scan.range_max,
            mode,
            self.args.scan_half_angle_deg,
            self.args.hard_stop_range_m,
            self.args.min_scan_range_m,
            self.args.rotation_stop_range_m,
        )

    def current_scan_identity(self):
        return (1.0, self.last_scan_received_sec)

    def publish_velocity(self, linear_x, angular_z):
        self.published.append((linear_x, angular_z))

    def wait_one_control_cycle(self):
        return None

    def maybe_log_post_replan_recovery(self, *args, **kwargs):
        return None

    def lookup_odom_pose(self):
        return self.odom_pose

    def record_motion_sample(self, yaw_error_deg, linear_x, angular_z, sample_seconds):
        self.motion_samples.append((yaw_error_deg, linear_x, angular_z, sample_seconds))


def forward_clearance(node, pose=None, route_state=None):
    pose = pose or Pose2D(0.0, 0.0, 0.0)
    return recovery.post_replan_forward_clearance_safety(
        node,
        pose,
        route_state or node.route_state,
    )


SIDE_CONE_CLUSTER = [(0.30, 0.0), (0.30, 0.01)]


class RouteAwareClearanceTests(unittest.TestCase):
    def test_auto_preview_distance_has_floor_and_cap(self):
        self.assertAlmostEqual(
            recovery.post_replan_route_clearance_preview_distance_m(
                SimpleNamespace(
                    post_replan_route_clearance_preview_distance_m=0.0,
                    min_scan_range_m=0.10,
                    post_replan_escape_distance_m=0.12,
                )
            ),
            0.25,
        )
        self.assertAlmostEqual(
            recovery.post_replan_route_clearance_preview_distance_m(
                SimpleNamespace(
                    post_replan_route_clearance_preview_distance_m=0.0,
                    min_scan_range_m=0.70,
                    post_replan_escape_distance_m=0.12,
                )
            ),
            0.60,
        )
        self.assertAlmostEqual(
            recovery.post_replan_route_clearance_preview_distance_m(
                SimpleNamespace(
                    post_replan_route_clearance_preview_distance_m=0.42,
                    min_scan_range_m=0.10,
                    post_replan_escape_distance_m=0.12,
                )
            ),
            0.42,
        )

    def test_cone_obstacle_outside_route_corridor_passes(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )

        safety = forward_clearance(node)

        self.assertTrue(safety.safe)
        self.assertEqual(safety.reason, "route_clear_side_obstacle")
        self.assertEqual(node.last_post_replan_route_corridor_blocked_count, 0)
        self.assertGreater(node.last_post_replan_route_clear_side_obstacle_count, 0)

    def test_obstacle_outside_cone_inside_route_corridor_blocks(self):
        node = FakeNode(
            points=[(0.18, 0.31)],
            route_points=[(0.0, 0.0), (0.20, 0.35)],
        )

        safety = forward_clearance(node)

        self.assertFalse(safety.safe)
        self.assertEqual(safety.reason, "route_corridor_blocked")
        self.assertGreater(node.last_post_replan_route_corridor_blocked_count, 0)

    def test_obstacle_inside_route_corridor_blocks(self):
        node = FakeNode(
            points=[(0.30, 0.0)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )

        safety = forward_clearance(node)

        self.assertFalse(safety.safe)
        self.assertEqual(safety.reason, "route_corridor_blocked")

    def test_two_segment_preview_blocks_on_second_segment(self):
        node = FakeNode(
            points=[(0.25, 0.20)],
            route_points=[(0.0, 0.0), (0.25, 0.0), (0.25, 0.30)],
            preview_distance_m=0.55,
        )

        safety = forward_clearance(node)

        self.assertFalse(safety.safe)
        self.assertEqual(safety.reason, "route_corridor_blocked")

    def test_forward_sector_hard_stop_remains_unsafe(self):
        node = FakeNode(
            points=[(0.12, 0.0)],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )

        safety = forward_clearance(node)

        self.assertFalse(safety.safe)
        self.assertEqual(safety.reason, "hard_stop")

    def test_missing_route_geometry_falls_back_to_cone_behavior(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0)],
        )

        safety = forward_clearance(node)

        self.assertFalse(safety.safe)
        self.assertEqual(safety.reason, "soft_stop")
        self.assertEqual(
            node.last_post_replan_route_clearance_reason,
            "route_clearance_unavailable",
        )

    def test_missing_pose_data_falls_back_to_cone_behavior(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )

        safety = recovery.post_replan_forward_clearance_safety(
            node,
            None,
            node.route_state,
            cone_safety=node.evaluate_current_scan_safety("forward"),
        )

        self.assertFalse(safety.safe)
        self.assertEqual(safety.reason, "soft_stop")
        self.assertEqual(
            node.last_post_replan_route_clearance_reason,
            "route_clearance_unavailable",
        )

    def test_cone_mode_preserves_current_behavior(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0), (0.40, 0.25)],
            clearance_mode=recovery.POST_REPLAN_CLEARANCE_CONE,
        )

        safety = forward_clearance(node)

        self.assertFalse(safety.safe)
        self.assertEqual(safety.reason, "soft_stop")


class FakeBlockedByScan(RuntimeError):
    def __init__(self, scan_safety):
        super().__init__(scan_safety.reason)
        self.scan_safety = scan_safety


def make_escape_recovery(*, now=None, start_pose=None, start_odom_pose=None):
    now = time.time() if now is None else now
    return recovery.PostReplanRecoveryState(
        route_generation_id=0,
        activation_pose=start_pose or Pose2D(0.0, 0.0, 0.0),
        activation_time_sec=now - 1.0,
        activation_scan_stamp_sec=0.0,
        activation_scan_received_sec=now - 1.0,
        route_heading_deg=30.0,
        phase=recovery.POST_REPLAN_RECOVERY_ESCAPE,
        escape_start_pose=start_pose or Pose2D(0.0, 0.0, 0.0),
        escape_start_odom_pose=start_odom_pose,
        escape_start_time_sec=now - 1.0,
        escape_straight_until_progress_active=True,
    )


def controller_step(*, angular_z=0.0, mode="tracking"):
    return SimpleNamespace(
        reached=False,
        mode=mode,
        command=SimpleNamespace(angular_z=angular_z),
        yaw_error_deg=0.0,
    )


class PostReplanRecoveryStateTests(unittest.TestCase):
    def test_auto_escape_steering_mode_resolves_from_clearance_mode(self):
        route_aware = FakeNode(
            points=[],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )
        cone = FakeNode(
            points=[],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
            clearance_mode=recovery.POST_REPLAN_CLEARANCE_CONE,
        )

        self.assertEqual(
            recovery.resolve_post_replan_escape_steering_mode(route_aware.args),
            recovery.POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT,
        )
        self.assertEqual(
            recovery.resolve_post_replan_escape_steering_mode(cone.args),
            recovery.POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS,
        )

    def test_wait_clear_counts_route_clear_side_obstacle_as_clear(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )
        node.post_replan_recovery = recovery.PostReplanRecoveryState(
            route_generation_id=0,
            activation_pose=Pose2D(0.0, 0.0, 0.0),
            activation_time_sec=time.time(),
            activation_scan_stamp_sec=0.0,
            activation_scan_received_sec=node.last_scan_received_sec - 1.0,
            route_heading_deg=30.0,
            phase=recovery.POST_REPLAN_RECOVERY_WAIT_CLEAR,
        )

        handled = recovery.handle_post_replan_recovery(
            node,
            None,
            Pose2D(0.0, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertEqual(
            node.post_replan_recovery.phase,
            recovery.POST_REPLAN_RECOVERY_ESCAPE,
        )

    def test_escape_continues_when_route_corridor_is_clear(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )
        node.post_replan_recovery = recovery.PostReplanRecoveryState(
            route_generation_id=0,
            activation_pose=Pose2D(0.0, 0.0, 0.0),
            activation_time_sec=time.time(),
            activation_scan_stamp_sec=0.0,
            activation_scan_received_sec=node.last_scan_received_sec - 1.0,
            route_heading_deg=30.0,
            phase=recovery.POST_REPLAN_RECOVERY_ESCAPE,
            escape_start_pose=Pose2D(0.0, 0.0, 0.0),
            escape_start_time_sec=time.time(),
            escape_straight_until_progress_active=True,
        )
        step = SimpleNamespace(reached=False)

        handled = recovery.handle_post_replan_recovery(
            node,
            step,
            Pose2D(0.0, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertIn((node.args.post_replan_escape_linear_speed_mps, 0.0), node.published)

    def test_wait_clear_marginal_route_block_does_not_enter_escape(self):
        node = FakeNode(
            points=[(0.30, 0.131)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
            clear_scan_samples=1,
        )
        node.post_replan_recovery = recovery.PostReplanRecoveryState(
            route_generation_id=0,
            activation_pose=Pose2D(0.0, 0.0, 0.0),
            activation_time_sec=time.time(),
            activation_scan_stamp_sec=0.0,
            activation_scan_received_sec=node.last_scan_received_sec - 1.0,
            route_heading_deg=0.0,
            phase=recovery.POST_REPLAN_RECOVERY_WAIT_CLEAR,
        )

        handled = recovery.handle_post_replan_recovery(
            node,
            controller_step(),
            Pose2D(0.0, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertEqual(
            node.post_replan_recovery.phase,
            recovery.POST_REPLAN_RECOVERY_WAIT_CLEAR,
        )
        self.assertEqual(node.post_replan_recovery.clear_scan_count, 0)
        self.assertEqual(
            node.last_post_replan_route_clearance_reason,
            "route_corridor_blocked",
        )
        self.assertEqual(
            node.post_replan_recovery.escape_route_block_decision,
            "wait_clear_transient_blocked_tolerated",
        )
        self.assertTrue(all(linear == 0.0 for linear, _angular in node.published))

    def test_wait_clear_persistent_marginal_route_block_requests_repair(self):
        now = time.time()
        node = FakeNode(
            points=[(0.30, 0.131)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
            clear_scan_samples=1,
        )
        node.post_replan_recovery = recovery.PostReplanRecoveryState(
            route_generation_id=0,
            activation_pose=Pose2D(0.0, 0.0, 0.0),
            activation_time_sec=now,
            activation_scan_stamp_sec=0.0,
            activation_scan_received_sec=node.last_scan_received_sec - 1.0,
            route_heading_deg=0.0,
            phase=recovery.POST_REPLAN_RECOVERY_WAIT_CLEAR,
        )

        self.assertTrue(
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                now,
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )
        )
        node.last_scan = FakeScan([(0.30, 0.131)])
        node.last_scan_received_sec += 0.1

        with self.assertRaises(recovery.PostReplanRouteBlockedRepairNeeded) as ctx:
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                now + 0.1,
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertEqual(
            node.last_post_replan_escape_route_block_decision,
            "persistent_blocked_scan",
        )
        self.assertIsNotNone(ctx.exception.scan_identity)
        self.assertIsNotNone(ctx.exception.blockage_signature)
        self.assertIsNone(node.post_replan_recovery)

    def test_wait_clear_non_marginal_route_block_requests_repair(self):
        node = FakeNode(
            points=[(0.30, 0.10)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
            clear_scan_samples=1,
        )
        node.post_replan_recovery = recovery.PostReplanRecoveryState(
            route_generation_id=0,
            activation_pose=Pose2D(0.0, 0.0, 0.0),
            activation_time_sec=time.time(),
            activation_scan_stamp_sec=0.0,
            activation_scan_received_sec=node.last_scan_received_sec - 1.0,
            route_heading_deg=0.0,
            phase=recovery.POST_REPLAN_RECOVERY_WAIT_CLEAR,
        )

        with self.assertRaises(recovery.PostReplanRouteBlockedRepairNeeded) as ctx:
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                time.time(),
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertTrue(all(linear == 0.0 for linear, _angular in node.published))
        self.assertIsNotNone(ctx.exception.scan_identity)
        self.assertIsNotNone(ctx.exception.blockage_signature)
        self.assertEqual(
            node.last_post_replan_recovery_status,
            "route_block_repair_needed",
        )

    def test_post_replan_notes_include_route_block_repair_metadata(self):
        node = FakeNode(
            points=[(0.30, 0.10)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        forward_clearance(node)
        node.post_replan_route_block_repair_count = 1
        node.post_replan_route_block_repair_status = "failed"
        node.post_replan_route_block_repair_signature = "signature"
        node.post_replan_route_block_repair_extra_update_used = True
        node.post_replan_route_block_repair_failure_reason = (
            "post_replan_route_corridor_persistently_blocked"
        )

        notes = recovery.notes_with_post_replan_recovery_metadata(
            "notes",
            node.args,
            node,
        )

        self.assertIn("route_corridor_nearest_blocked_base_x_m=", notes)
        self.assertIn("route_corridor_nearest_blocked_penetration_m=", notes)
        self.assertIn("post_replan_route_block_repair_count=1", notes)
        self.assertIn("post_replan_route_block_repair_status=failed", notes)
        self.assertIn("post_replan_route_block_repair_signature=signature", notes)
        self.assertIn("post_replan_route_block_repair_extra_update_used=True", notes)
        self.assertIn(
            "post_replan_route_block_repair_failure_reason="
            "post_replan_route_corridor_persistently_blocked",
            notes,
        )

    def test_escape_tolerates_first_fresh_marginal_single_route_block(self):
        node = FakeNode(
            points=[(0.30, 0.131)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        node.post_replan_recovery = make_escape_recovery()

        handled = recovery.handle_post_replan_recovery(
            node,
            controller_step(),
            Pose2D(0.0, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertIn((node.args.post_replan_escape_linear_speed_mps, 0.0), node.published)
        self.assertEqual(node.post_replan_recovery.escape_route_blocked_streak, 1)
        self.assertEqual(
            node.post_replan_recovery.escape_route_block_decision,
            "transient_blocked_tolerated",
        )
        self.assertEqual(
            node.post_replan_recovery.escape_route_blocked_tolerated_count,
            1,
        )

    def test_escape_persistent_marginal_route_block_fails_on_second_fresh_scan(self):
        now = time.time()
        node = FakeNode(
            points=[(0.30, 0.131)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        node.post_replan_recovery = make_escape_recovery(now=now)

        self.assertTrue(
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                now,
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )
        )
        node.last_scan = FakeScan([(0.30, 0.131)])
        node.last_scan_received_sec += 0.1

        with self.assertRaisesRegex(RuntimeError, "post_replan_escape_blocked"):
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                now + 0.1,
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertEqual(
            node.last_post_replan_escape_route_block_decision,
            "persistent_blocked_scan",
        )

    def test_escape_same_scan_identity_does_not_increment_route_block_streak(self):
        now = time.time()
        node = FakeNode(
            points=[(0.30, 0.131)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        node.post_replan_recovery = make_escape_recovery(now=now)

        recovery.handle_post_replan_recovery(
            node,
            controller_step(),
            Pose2D(0.0, 0.0, 0.0),
            now,
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )
        recovery.handle_post_replan_recovery(
            node,
            controller_step(),
            Pose2D(0.0, 0.0, 0.0),
            now + 0.1,
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertEqual(node.post_replan_recovery.escape_route_blocked_streak, 1)
        self.assertEqual(
            node.post_replan_recovery.escape_route_block_decision,
            "transient_blocked_tolerated",
        )

    def test_escape_route_block_count_threshold_fails_immediately(self):
        node = FakeNode(
            points=[(0.30, 0.131), (0.30, -0.131)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        node.post_replan_recovery = make_escape_recovery()

        with self.assertRaisesRegex(RuntimeError, "post_replan_escape_blocked"):
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                time.time(),
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertEqual(
            node.last_post_replan_escape_route_block_decision,
            "blocked_count_threshold",
        )

    def test_escape_deep_single_route_block_fails_immediately(self):
        node = FakeNode(
            points=[(0.30, 0.10)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        node.post_replan_recovery = make_escape_recovery()

        with self.assertRaisesRegex(RuntimeError, "post_replan_escape_blocked"):
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                time.time(),
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertEqual(
            node.last_post_replan_escape_route_block_decision,
            "deep_single_block",
        )

    def test_escape_clear_scan_resets_marginal_route_block_streak(self):
        now = time.time()
        node = FakeNode(
            points=[(0.30, 0.131)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        node.post_replan_recovery = make_escape_recovery(now=now)

        recovery.handle_post_replan_recovery(
            node,
            controller_step(),
            Pose2D(0.0, 0.0, 0.0),
            now,
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )
        node.last_scan = FakeScan([])
        node.last_scan_received_sec += 0.1
        recovery.handle_post_replan_recovery(
            node,
            controller_step(),
            Pose2D(0.0, 0.0, 0.0),
            now + 0.1,
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertEqual(node.post_replan_recovery.escape_route_blocked_streak, 0)
        self.assertEqual(
            node.post_replan_recovery.escape_route_block_decision,
            "clear",
        )

    def test_cone_mode_soft_stop_escape_still_fails(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0), (0.40, 0.25)],
            clearance_mode=recovery.POST_REPLAN_CLEARANCE_CONE,
        )
        node.post_replan_recovery = make_escape_recovery()

        with self.assertRaisesRegex(RuntimeError, "post_replan_escape_blocked"):
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                time.time(),
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

    def test_escape_hard_stop_still_fails_immediately(self):
        node = FakeNode(
            points=[(0.12, 0.0)],
            route_points=[(0.0, 0.0), (0.50, 0.0)],
        )
        node.post_replan_recovery = make_escape_recovery()

        with self.assertRaises(FakeBlockedByScan):
            recovery.handle_post_replan_recovery(
                node,
                controller_step(),
                Pose2D(0.0, 0.0, 0.0),
                time.time(),
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertEqual(
            node.last_post_replan_escape_route_block_decision,
            "hard_stop",
        )

    def test_route_aware_early_escape_uses_capped_controller_angular_hint(self):
        node = FakeNode(
            points=SIDE_CONE_CLUSTER,
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )
        node.post_replan_recovery = make_escape_recovery()

        handled = recovery.handle_post_replan_recovery(
            node,
            controller_step(angular_z=0.12),
            Pose2D(0.0, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertIn(
            (
                node.args.post_replan_escape_linear_speed_mps,
                recovery.POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS,
            ),
            node.published,
        )
        self.assertEqual(
            node.post_replan_recovery.last_escape_angular_hint_source,
            "controller",
        )
        self.assertFalse(
            node.post_replan_recovery.escape_straight_until_progress_active,
        )

    def test_cone_mode_preserves_straight_until_progress_escape(self):
        node = FakeNode(
            points=[],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
            clearance_mode=recovery.POST_REPLAN_CLEARANCE_CONE,
        )
        node.post_replan_recovery = make_escape_recovery()

        handled = recovery.handle_post_replan_recovery(
            node,
            controller_step(angular_z=0.05),
            Pose2D(0.0, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertIn((node.args.post_replan_escape_linear_speed_mps, 0.0), node.published)
        self.assertEqual(
            node.post_replan_recovery.last_escape_angular_hint_source,
            "straight_until_progress",
        )

    def test_straight_escape_ignores_blocked_controller_after_initial_progress(self):
        node = FakeNode(
            points=[],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
            steering_mode=recovery.POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS,
        )
        node.post_replan_recovery = make_escape_recovery()
        node.post_replan_recovery.escape_start_direct_odom_pose = Pose2D(
            0.0,
            0.0,
            0.0,
            stamp_sec=10.0,
            frame_id="base_link",
        )
        node.last_odom_pose = Pose2D(
            0.017,
            0.0,
            0.0,
            stamp_sec=10.5,
            frame_id="base_link",
        )
        node.last_odom_received_sec = time.time()
        node.last_odom_frame_id = "odom"
        node.last_odom_child_frame_id = "base_link"

        handled = recovery.handle_post_replan_recovery(
            node,
            controller_step(mode="blocked"),
            Pose2D(0.017, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertIn((node.args.post_replan_escape_linear_speed_mps, 0.0), node.published)
        self.assertIsNotNone(node.post_replan_recovery)
        self.assertFalse(
            node.post_replan_recovery.escape_straight_until_progress_active,
        )
        self.assertEqual(
            node.post_replan_recovery.last_escape_angular_hint_source,
            "straight_escape",
        )

    def test_route_hint_controller_blocked_and_off_route_abort(self):
        for mode, expected in (
            ("blocked", "post_replan_escape_controller_blocked"),
            ("off_route", "post_replan_escape_off_route"),
        ):
            with self.subTest(mode=mode):
                node = FakeNode(
                    points=[],
                    route_points=[(0.0, 0.0), (0.40, 0.25)],
                )
                node.post_replan_recovery = make_escape_recovery()

                with self.assertRaisesRegex(RuntimeError, expected):
                    recovery.handle_post_replan_recovery(
                        node,
                        controller_step(mode=mode),
                        Pose2D(0.0, 0.0, 0.0),
                        time.time(),
                        route_state=node.route_state,
                        blocked_error_type=FakeBlockedByScan,
                    )

    def test_route_hint_missing_step_records_unavailable_fallback(self):
        node = FakeNode(
            points=[],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )
        node.post_replan_recovery = make_escape_recovery()

        handled = recovery.handle_post_replan_recovery(
            node,
            None,
            Pose2D(0.0, 0.0, 0.0),
            time.time(),
            route_state=node.route_state,
            blocked_error_type=FakeBlockedByScan,
        )

        self.assertTrue(handled)
        self.assertIn((node.args.post_replan_escape_linear_speed_mps, 0.0), node.published)
        self.assertEqual(
            node.post_replan_recovery.last_escape_angular_hint_source,
            "route_hint_unavailable",
        )

    def test_no_motion_watchdog_still_fires_for_static_odom(self):
        now = time.time()
        node = FakeNode(
            points=[],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )
        node.odom_pose = Pose2D(0.0, 0.0, 0.0, stamp_sec=5.0)
        node.post_replan_recovery = make_escape_recovery(
            now=now,
            start_odom_pose=Pose2D(0.0, 0.0, 0.0, stamp_sec=1.0),
        )
        node.post_replan_recovery.first_escape_command_time_sec = now - 4.0
        node.post_replan_recovery.last_progress_time_sec = now - 3.1

        with self.assertRaisesRegex(RuntimeError, "post_replan_escape_no_motion"):
            recovery.handle_post_replan_recovery(
                node,
                controller_step(angular_z=0.02),
                Pose2D(0.0, 0.0, 0.0),
                now,
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertEqual(
            node.last_post_replan_recovery_escape_no_motion_reason,
            "cmd_vel_no_odom_motion",
        )

    def test_static_odom_with_moving_map_is_diagnostic_only(self):
        now = time.time()
        node = FakeNode(
            points=[],
            route_points=[(0.0, 0.0), (0.40, 0.25)],
        )
        node.odom_pose = Pose2D(0.0, 0.0, 0.0, stamp_sec=1.0)
        node.post_replan_recovery = make_escape_recovery(
            now=now,
            start_pose=Pose2D(0.0, 0.0, 0.0),
            start_odom_pose=Pose2D(0.0, 0.0, 0.0, stamp_sec=1.0),
        )
        node.post_replan_recovery.first_escape_command_time_sec = now - 4.0
        node.post_replan_recovery.last_progress_time_sec = now - 3.1

        with self.assertRaisesRegex(RuntimeError, "post_replan_escape_no_motion"):
            recovery.handle_post_replan_recovery(
                node,
                controller_step(angular_z=0.02),
                Pose2D(0.004, 0.0, 0.0),
                now,
                route_state=node.route_state,
                blocked_error_type=FakeBlockedByScan,
            )

        self.assertEqual(
            node.last_post_replan_recovery_escape_no_motion_reason,
            "odom_static_map_moved",
        )
        self.assertEqual(
            node.last_post_replan_recovery_best_escape_distance_m,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
