from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FollowLoopContext:
    blocked_by_scan_error_type: type[Exception]
    build_command_smoother: Callable[..., Any]
    build_path_controller: Callable[..., Any]
    build_sparse_tracking_validation: Callable[..., Any]
    compat_follower_type: type
    format_optional_m: Callable[[Any], str]
    guard_block_signature: Callable[..., Any]
    post_replan_recovery_escape: str
    post_replan_recovery_should_preempt_controller: Callable[..., bool]
    publish_rviz_obstacles_if_available: Callable[..., Any]
    publish_rviz_route_if_available: Callable[..., Any]
    rclpy: Any
    replan_trigger_known_corridor: str
    replan_trigger_lookahead_guard: str
    replan_trigger_scan_blockage: str
    reset_command_smoother: Callable[..., Any]
    reset_route_projection_controller: Callable[..., Any]
    route_state_type: type
    smoothed_step_command: Callable[..., Any]
    waypoint_timeout_error_type: type[Exception]


def _setdefault_attr(node, name, value):
    if not hasattr(node, name):
        setattr(node, name, value)


def ensure_follow_runtime_state(node, build_command_smoother):
    if not hasattr(node, "command_smoother"):
        node.command_smoother = build_command_smoother(node.args)
    _setdefault_attr(node, "last_smoothed_command_time_sec", None)
    _setdefault_attr(node, "active_route_generation_id", 0)
    _setdefault_attr(node, "post_replan_recovery", None)
    _setdefault_attr(node, "post_replan_recovery_activations", 0)
    _setdefault_attr(node, "last_post_replan_recovery_status", "")
    _setdefault_attr(node, "last_post_replan_recovery_phase", "")
    _setdefault_attr(node, "last_post_replan_recovery_clear_count", 0)
    _setdefault_attr(node, "max_post_replan_recovery_clear_count", 0)
    _setdefault_attr(node, "last_post_replan_recovery_escape_distance_m", 0.0)
    _setdefault_attr(node, "last_post_replan_recovery_best_escape_distance_m", 0.0)
    _setdefault_attr(node, "last_post_replan_recovery_escape_distance_source", "")
    _setdefault_attr(node, "last_post_replan_recovery_escape_no_motion_elapsed_sec", None)
    _setdefault_attr(node, "last_post_replan_recovery_escape_straight_active", False)
    _setdefault_attr(node, "last_post_replan_recovery_escape_elapsed_sec", None)
    _setdefault_attr(node, "last_post_replan_recovery_escape_timeout_sec", None)
    _setdefault_attr(node, "last_post_replan_recovery_heading_error_deg", None)
    _setdefault_attr(node, "last_post_replan_recovery_alignment_heading_deg", None)
    _setdefault_attr(node, "last_post_replan_recovery_alignment_heading_source", "")
    _setdefault_attr(node, "last_post_replan_recovery_alignment_segment_index", None)
    _setdefault_attr(node, "last_post_replan_recovery_alignment_segment_ratio", None)
    _setdefault_attr(node, "last_post_replan_recovery_escape_command_linear_mps", 0.0)
    _setdefault_attr(node, "last_post_replan_recovery_escape_command_angular_radps", 0.0)
    _setdefault_attr(node, "last_post_replan_recovery_escape_angular_hint_source", "")
    _setdefault_attr(node, "last_post_replan_recovery_escape_steering_mode_resolved", "")
    _setdefault_attr(node, "last_post_replan_recovery_escape_odom_distance_m", None)
    _setdefault_attr(node, "last_post_replan_recovery_escape_map_distance_m", None)
    _setdefault_attr(node, "last_post_replan_recovery_escape_odom_stamp_delta_sec", None)
    _setdefault_attr(node, "last_post_replan_recovery_escape_progress_source", "")
    _setdefault_attr(node, "last_post_replan_recovery_escape_no_motion_reason", "")
    _setdefault_attr(node, "last_post_replan_clearance_search_attempted", False)
    _setdefault_attr(node, "last_post_replan_clearance_search_direction", 0.0)
    _setdefault_attr(node, "last_post_replan_clearance_search_yaw_delta_deg", 0.0)
    _setdefault_attr(node, "last_post_replan_clearance_search_baseline_p05_m", None)
    _setdefault_attr(node, "last_post_replan_clearance_search_best_p05_m", None)
    _setdefault_attr(node, "last_post_replan_clearance_search_baseline_min_m", None)
    _setdefault_attr(node, "last_post_replan_clearance_search_best_min_m", None)
    _setdefault_attr(node, "last_post_replan_clearance_search_result", "")
    _setdefault_attr(node, "last_post_replan_clearance_search_direction_source", "")
    _setdefault_attr(node, "last_post_replan_activation_min_target_distance_m", 0.0)
    _setdefault_attr(node, "last_post_replan_activation_pruned_sparse_count", 0)
    _setdefault_attr(node, "last_post_replan_activation_pruned_dense_count", 0)
    _setdefault_attr(node, "last_post_replan_activation_projection_progress_m", None)
    _setdefault_attr(node, "last_post_replan_activation_first_target_distance_m", None)
    _setdefault_attr(node, "last_post_replan_activation_status", "")
    _setdefault_attr(node, "last_post_replan_recovery_log_sec", None)


def follow_waypoints(
    node,
    waypoints,
    tracking_points=None,
    tracking_validation=None,
    *,
    context: FollowLoopContext,
):
    self = node
    BlockedByScanError = context.blocked_by_scan_error_type
    POST_REPLAN_RECOVERY_ESCAPE = context.post_replan_recovery_escape
    REPLAN_TRIGGER_KNOWN_CORRIDOR = context.replan_trigger_known_corridor
    REPLAN_TRIGGER_LOOKAHEAD_GUARD = context.replan_trigger_lookahead_guard
    REPLAN_TRIGGER_SCAN_BLOCKAGE = context.replan_trigger_scan_blockage
    RouteState = context.route_state_type
    WaypointFollower = context.compat_follower_type
    WaypointTimeoutError = context.waypoint_timeout_error_type
    build_command_smoother = context.build_command_smoother
    build_path_controller = context.build_path_controller
    build_sparse_tracking_validation = context.build_sparse_tracking_validation
    format_optional_m = context.format_optional_m
    guard_block_signature = context.guard_block_signature
    post_replan_recovery_should_preempt_controller = (
        context.post_replan_recovery_should_preempt_controller
    )
    publish_rviz_obstacles_if_available = context.publish_rviz_obstacles_if_available
    publish_rviz_route_if_available = context.publish_rviz_route_if_available
    rclpy = context.rclpy
    reset_command_smoother = context.reset_command_smoother
    reset_route_projection_controller = context.reset_route_projection_controller
    smoothed_step_command = context.smoothed_step_command
    reached_count = 0
    start_pose, _frame, amcl_health = self.check_health_or_recover()
    final_pose = start_pose
    last_scan_safety = None
    self.start_pose = start_pose
    self.final_pose = final_pose
    self.last_amcl_health = amcl_health
    ensure_follow_runtime_state(self, build_command_smoother)
    reset_command_smoother(self)

    waypoints = list(waypoints)
    continuous_tracking = self.args.controller == "pure-pursuit"
    if not continuous_tracking:
        tracking_points = None
        tracking_validation = build_sparse_tracking_validation(
            source="ignored_stop_go",
            point_count=0,
            status="ignored",
        )
    tracking_source = (
        tracking_validation.source
        if tracking_validation is not None
        else ("csv" if tracking_points is not None else "waypoints")
    )
    route_state = RouteState(
        waypoints,
        tracking_points=tracking_points,
        tracking_source=tracking_source,
        tracking_validation=tracking_validation,
    )
    controller = build_path_controller(
        self.args,
        lookahead_guard=getattr(self, "lookahead_guard", None),
    )
    self._current_path_controller = controller
    publish_rviz_route_if_available(self, waypoints, current_pose=start_pose)
    publish_rviz_obstacles_if_available(self)
    if self.args.enable_lidar_map_replan:
        waypoints = WaypointFollower._compat_method(
            self,
            "initialize_run_local_route",
            start_pose,
            waypoints,
        )
        last_replan_tracking_points = getattr(
            self,
            "last_replan_tracking_points",
            None,
        )
        replacement_tracking_points = (
            last_replan_tracking_points
            if last_replan_tracking_points is not None
            else tracking_points
        )
        replacement_tracking_source = (
            getattr(self, "last_replan_tracking_source", "waypoints")
            if last_replan_tracking_points is not None
            else tracking_source
        )
        replacement_tracking_validation = (
            getattr(self, "last_replan_tracking_validation", None)
            if last_replan_tracking_points is not None
            else tracking_validation
        )
        route_state.replace_route(
            waypoints,
            tracking_points=replacement_tracking_points,
            tracking_source=replacement_tracking_source,
            tracking_validation=replacement_tracking_validation,
        )
        self.active_route_generation_id += 1
        WaypointFollower.reset_post_replan_recovery(self, "route_replaced")
        reset_command_smoother(self)
        controller = build_path_controller(
            self.args,
            lookahead_guard=getattr(self, "lookahead_guard", None),
        )
        self._current_path_controller = controller
        publish_rviz_route_if_available(self, waypoints, current_pose=start_pose)
        if self.args.lidar_replan_artifact_only:
            self.stop_repeatedly()
            return {
                "reached_count": reached_count,
                "start_pose": start_pose,
                "final_pose": final_pose,
                "scan_safety": last_scan_safety,
                "amcl_health": amcl_health,
                "base_frame_used": self.base_frame_used,
                "status": "replan_artifact_only_complete",
            }

    while not route_state.complete:
        waypoint = route_state.current_waypoint()
        publish_rviz_route_if_available(
            self,
            route_state.remaining(),
            current_pose=final_pose,
            current_waypoint_index=0,
        )
        self.get_logger().info(
            f"[{route_state.current_waypoint_index + 1}/{len(route_state.waypoints)}] "
            f"target waypoint {waypoint.index}: "
            f"x={waypoint.x:.3f}, y={waypoint.y:.3f}"
        )
        waypoint_start = time.time()
        reached_current = False
        replanned_current = False

        while rclpy.ok():
            pose, _frame, amcl_health = self.check_health_or_recover()
            final_pose = pose
            self.final_pose = final_pose
            self.last_amcl_health = amcl_health
            recovery = getattr(self, "post_replan_recovery", None)
            if (
                post_replan_recovery_should_preempt_controller(
                    recovery,
                    self.args,
                )
                and WaypointFollower.handle_post_replan_recovery(
                    self,
                    None,
                    pose,
                    time.time(),
                    route_state,
                )
            ):
                continue
            step = controller.compute(pose, route_state)
            self.last_lookahead_guard_result = step.guard_result
            if hasattr(self, "record_route_projection_result"):
                self.record_route_projection_result(step)
                self.maybe_log_route_projection_result(step, time.time())
            if hasattr(self, "maybe_log_velocity_scheduler_result"):
                self.maybe_log_velocity_scheduler_result(
                    step.velocity_schedule_result,
                    time.time(),
                )
            if (
                self.args.verbose
                and step.guard_result is not None
                and step.guard_result.status != "clear"
            ):
                self.get_logger().info(
                    "Pure-pursuit lookahead guard result: "
                    f"mode={self.args.pure_pursuit_lookahead_guard}, "
                    f"status={step.guard_result.status}, "
                    "selected_distance_m="
                    f"{format_optional_m(step.guard_result.selected_target_distance_m)}, "
                    f"blocked_cells={step.guard_result.blocked_cell_count}"
                )

            recovery = getattr(self, "post_replan_recovery", None)
            if (
                recovery is not None
                and recovery.phase == POST_REPLAN_RECOVERY_ESCAPE
                and WaypointFollower.handle_post_replan_recovery(
                    self,
                    step,
                    pose,
                    time.time(),
                    route_state,
                )
            ):
                continue

            if step.reached:
                if continuous_tracking:
                    route_state.mark_complete()
                    reached_count = len(route_state.waypoints)
                else:
                    reached_count += 1
                self.reached_count = reached_count
                if self.args.enable_lidar_map_replan:
                    WaypointFollower._compat_method(
                        self,
                        "prune_run_local_obstacles_after_progress",
                        pose,
                        route_state.waypoints[
                            route_state.current_waypoint_index + 1:
                        ],
                    )
                self.last_known_corridor_repair_signature = None
                self.suppressed_known_corridor_signature = None
                self.last_scan_block_budget_repair_signature = None
                self.last_lookahead_guard_block_signature = None
                WaypointFollower.reset_post_replan_recovery(self, "reached")
                reset_command_smoother(self)
                reset_route_projection_controller(controller)
                self.stop_repeatedly()
                self.spin_for(self.args.settle_sec)
                if not continuous_tracking:
                    route_state.advance()
                reached_current = True
                break

            if continuous_tracking:
                before_index = route_state.current_waypoint_index
                if route_state.advance_if_reached(
                    pose,
                    self.args.waypoint_tolerance_m,
                    self.args.pure_pursuit_goal_tolerance_m,
                ):
                    reached_count = max(
                        reached_count,
                        route_state.current_waypoint_index,
                    )
                    self.reached_count = reached_count
                if route_state.current_waypoint_index != before_index:
                    waypoint_start = time.time()

            if time.time() - waypoint_start > self.args.max_waypoint_time_sec:
                raise WaypointTimeoutError(waypoint)

            if step.mode == "off_route":
                WaypointFollower.reset_post_replan_recovery(self, "off_route")
                reset_command_smoother(self)
                reset_route_projection_controller(controller)
                self.stop_repeatedly()
                raise RuntimeError("pure_pursuit_off_tracking_route")

            if step.mode == "blocked":
                if self.args.verbose and step.guard_result is not None:
                    self.get_logger().warn(
                        "Pure-pursuit lookahead guard blocked motion: "
                        f"status={step.guard_result.status}, "
                        f"blocked_cells={step.guard_result.blocked_cell_count}"
                    )
                reset_command_smoother(self)
                reset_route_projection_controller(controller)
                WaypointFollower.reset_post_replan_recovery(
                    self,
                    "lookahead_blocked",
                )
                self.stop_repeatedly()
                last_scan_safety = self.check_scan_or_raise("forward")
                self.last_scan_safety = last_scan_safety
                if not self.args.enable_lidar_map_replan:
                    raise RuntimeError("pure_pursuit_lookahead_blocked")
                guard_signature = guard_block_signature(
                    pose,
                    route_state.remaining_tracking_points(),
                )
                if (
                    guard_signature
                    == getattr(
                        self,
                        "last_lookahead_guard_block_signature",
                        None,
                    )
                ):
                    raise RuntimeError(
                        "pure_pursuit_lookahead_blocked_after_unchanged_replan"
                    )
                self.last_lookahead_guard_block_signature = guard_signature
                remaining = route_state.remaining()
                replanned = WaypointFollower._compat_method(
                    self,
                    "replan_after_blockage",
                    pose,
                    remaining,
                    trigger=REPLAN_TRIGGER_LOOKAHEAD_GUARD,
                )
                publish_rviz_route_if_available(
                    self,
                    replanned,
                    current_pose=pose,
                    current_waypoint_index=0,
                )
                if self.args.lidar_replan_artifact_only:
                    self.stop_repeatedly()
                    return {
                        "reached_count": reached_count,
                        "start_pose": start_pose,
                        "final_pose": final_pose,
                        "scan_safety": last_scan_safety,
                        "amcl_health": amcl_health,
                        "base_frame_used": self.base_frame_used,
                        "status": "replan_artifact_only_complete",
                    }
                waypoints = self.prune_replanned_waypoints_for_progress(
                    replanned,
                    pose,
                )
                route_state.replace_route(
                    waypoints,
                    tracking_points=getattr(
                        self,
                        "last_replan_tracking_points",
                        None,
                    ),
                    tracking_source=getattr(
                        self,
                        "last_replan_tracking_source",
                        "waypoints",
                    ),
                    tracking_validation=getattr(
                        self,
                        "last_replan_tracking_validation",
                        None,
                    ),
                )
                self.active_route_generation_id += 1
                WaypointFollower.reset_post_replan_recovery(self, "route_replaced")
                reset_command_smoother(self)
                controller = build_path_controller(
                    self.args,
                    lookahead_guard=getattr(self, "lookahead_guard", None),
                )
                self._current_path_controller = controller
                replanned_current = True
                break

            if WaypointFollower.handle_post_replan_recovery(
                self,
                step,
                pose,
                time.time(),
                route_state,
            ):
                continue

            try:
                last_scan_safety = self.check_scan_or_raise(step.mode)
                self.last_scan_safety = last_scan_safety
                self.last_scan_block_budget_repair_signature = None
            except BlockedByScanError as exc:
                reset_command_smoother(self)
                reset_route_projection_controller(controller)
                if self.args.enable_lidar_map_replan:
                    remaining = route_state.remaining()
                    replanned = WaypointFollower._compat_method(
                        self,
                        "replan_after_blockage",
                        pose,
                        remaining,
                        trigger=REPLAN_TRIGGER_SCAN_BLOCKAGE,
                    )
                    publish_rviz_route_if_available(
                        self,
                        replanned,
                        current_pose=pose,
                        current_waypoint_index=0,
                    )
                    if self.args.lidar_replan_artifact_only:
                        self.stop_repeatedly()
                        return {
                            "reached_count": reached_count,
                            "start_pose": start_pose,
                            "final_pose": final_pose,
                            "scan_safety": exc.scan_safety,
                            "amcl_health": amcl_health,
                            "base_frame_used": self.base_frame_used,
                            "status": "replan_artifact_only_complete",
                        }
                    activation_route = (
                        WaypointFollower.prepare_run_local_route_activation(
                            self,
                            replanned,
                            pose,
                            route_state.final_goal(),
                            REPLAN_TRIGGER_SCAN_BLOCKAGE,
                        )
                    )
                    if activation_route.goal_reached:
                        route_state.mark_complete()
                        reached_count = len(route_state.waypoints)
                        self.reached_count = reached_count
                        WaypointFollower.reset_post_replan_recovery(
                            self,
                            "goal_reached_after_replan_activation",
                        )
                        reset_command_smoother(self)
                        self.stop_repeatedly()
                        reached_current = True
                        break
                    if not activation_route.waypoints:
                        raise RuntimeError("post_replan_no_meaningful_target")
                    waypoints = activation_route.waypoints
                    route_state.replace_route(
                        waypoints,
                        tracking_points=activation_route.tracking_points,
                        tracking_source=activation_route.tracking_source,
                        tracking_validation=activation_route.tracking_validation,
                    )
                    self.active_route_generation_id += 1
                    reset_command_smoother(self)
                    controller = build_path_controller(
                        self.args,
                        lookahead_guard=getattr(self, "lookahead_guard", None),
                    )
                    self._current_path_controller = controller
                    WaypointFollower.activate_post_replan_recovery(
                        self,
                        pose,
                        route_state,
                    )
                    replanned_current = True
                    break
                raise BlockedByScanError(exc.scan_safety, waypoint) from exc
            if self.args.enable_lidar_map_replan and self.run_local_map is not None:
                remaining = route_state.remaining()
                WaypointFollower._compat_method(
                    self,
                    "prune_run_local_obstacles_after_progress",
                    pose,
                    remaining,
                )
                blocked_cells = WaypointFollower._compat_method(
                    self,
                    "corridor_blocked_cells",
                    pose,
                    remaining,
                )
                if blocked_cells and not WaypointFollower._compat_method(
                    self,
                    "suppress_repeated_known_corridor_repair",
                    remaining,
                ):
                    publish_rviz_obstacles_if_available(self, blocked_cells)
                    reset_route_projection_controller(controller)
                    self.stop_repeatedly()
                    replanned = WaypointFollower._compat_method(
                        self,
                        "replan_after_blockage",
                        pose,
                        remaining,
                        trigger=REPLAN_TRIGGER_KNOWN_CORRIDOR,
                    )
                    publish_rviz_route_if_available(
                        self,
                        replanned,
                        current_pose=pose,
                        current_waypoint_index=0,
                    )
                    if self.args.lidar_replan_artifact_only:
                        self.stop_repeatedly()
                        return {
                            "reached_count": reached_count,
                            "start_pose": start_pose,
                            "final_pose": final_pose,
                            "scan_safety": last_scan_safety,
                            "amcl_health": amcl_health,
                            "base_frame_used": self.base_frame_used,
                            "status": "replan_artifact_only_complete",
                        }
                    activation_route = (
                        WaypointFollower.prepare_run_local_route_activation(
                            self,
                            replanned,
                            pose,
                            route_state.final_goal(),
                            REPLAN_TRIGGER_KNOWN_CORRIDOR,
                        )
                    )
                    if activation_route.goal_reached:
                        route_state.mark_complete()
                        reached_count = len(route_state.waypoints)
                        self.reached_count = reached_count
                        WaypointFollower.reset_post_replan_recovery(
                            self,
                            "goal_reached_after_known_corridor_activation",
                        )
                        reset_command_smoother(self)
                        self.stop_repeatedly()
                        reached_current = True
                        break
                    if not activation_route.waypoints:
                        raise RuntimeError("known_corridor_no_meaningful_target")
                    waypoints = activation_route.waypoints
                    route_state.replace_route(
                        waypoints,
                        tracking_points=activation_route.tracking_points,
                        tracking_source=activation_route.tracking_source,
                        tracking_validation=activation_route.tracking_validation,
                    )
                    self.active_route_generation_id += 1
                    WaypointFollower.reset_post_replan_recovery(
                        self,
                        "route_replaced",
                    )
                    reset_command_smoother(self)
                    controller = build_path_controller(
                        self.args,
                        lookahead_guard=getattr(self, "lookahead_guard", None),
                    )
                    self._current_path_controller = controller
                    self.remember_known_corridor_repair(waypoints)
                    replanned_current = True
                    break
            command = smoothed_step_command(self, step, time.time())
            self.record_motion_sample(
                step.yaw_error_deg,
                command.linear_x,
                command.angular_z,
                1.0 / self.args.control_rate_hz,
            )
            self.publish_velocity(command.linear_x, command.angular_z)
            rclpy.spin_once(self, timeout_sec=1.0 / self.args.control_rate_hz)
            time.sleep(1.0 / self.args.control_rate_hz)

        if replanned_current:
            continue
        if reached_current:
            continue
        raise RuntimeError("ROS shutdown while following waypoints")

    return {
        "reached_count": reached_count,
        "start_pose": start_pose,
        "final_pose": final_pose,
        "scan_safety": last_scan_safety,
        "amcl_health": amcl_health,
        "base_frame_used": self.base_frame_used,
    }
