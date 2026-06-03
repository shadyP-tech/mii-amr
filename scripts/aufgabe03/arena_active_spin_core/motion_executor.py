from __future__ import annotations

import math

from .curve_following import (
    active_explore_curve_execution_record,
    active_explore_curve_path,
    pure_pursuit_curve_command,
    select_curve_lookahead_target,
)
from .diagnostics import update_safety_minima
from .math_utils import distance_2d, shortest_angle_delta_rad
from .models import (
    DEFAULT_STOP_COUNT,
    DEFAULT_STOP_HZ,
    ActiveExploreMotionError,
    CenterRepositionStep,
)
from .scan_safety import (
    dynamic_lateral_heading_from_scan,
    evaluate_clearance,
    evaluate_reposition_clearance,
)


def stop_repeatedly(
    publisher,
    twist_factory,
    sleep_fn,
    count=DEFAULT_STOP_COUNT,
    hz=DEFAULT_STOP_HZ,
):
    delay = 1.0 / hz
    for _ in range(count):
        publisher.publish(twist_factory())
        sleep_fn(delay)


class ActiveSpinMotionExecutor:
    def __init__(self, session):
        self.session = session

    def run_spin(self, publisher):
        session = self.session
        session.wait_for_fresh_inputs()
        session.cmd_vel_publisher_check()
        session.print_operator_prompt()
        session.refresh_fresh_inputs_after_prompt()

        previous_yaw = session.latest_odom_yaw_rad
        if previous_yaw is None:
            raise RuntimeError("fresh_odom_unavailable")
        session.collecting = True
        accumulated = 0.0
        target = 2.0 * math.pi - math.radians(
            session.config.spin_complete_tolerance_deg
        )
        period = 1.0 / session.config.control_rate_hz
        start = session.now()
        last_progress_time = start
        last_progress_yaw = 0.0

        while session.rclpy.ok():
            if session.now() - start > session.config.max_spin_sec:
                session.diagnostics["spin"]["timeout"] = True
                raise RuntimeError("arena_active_spin_timeout")
            session.publish_spin_command(publisher)
            session.rclpy.spin_once(session.node, timeout_sec=period)
            now = session.now()
            scan_age = session.fresh_scan_age_sec()
            odom_age = session.fresh_odom_age_sec()
            if scan_age is None or scan_age > session.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_spin")
            if odom_age is None or odom_age > session.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_spin")

            clearance = evaluate_clearance(session.latest_scan, session.config)
            update_safety_minima(session.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")

            current_yaw = session.latest_odom_yaw_rad
            delta = shortest_angle_delta_rad(previous_yaw, current_yaw)
            accumulated += delta
            previous_yaw = current_yaw
            session.diagnostics["spin"]["accumulated_rad"] = accumulated
            session.diagnostics["spin"]["duration_sec"] = now - start
            if abs(accumulated) >= target:
                return accumulated, now - start

            if now - last_progress_time >= session.config.progress_check_sec:
                progress_rate = abs(accumulated - last_progress_yaw) / (
                    now - last_progress_time
                )
                if progress_rate < session.config.min_angular_progress_rad_s:
                    raise RuntimeError("insufficient_angular_progress")
                last_progress_time = now
                last_progress_yaw = accumulated

        raise RuntimeError("ros_shutdown_during_arena_active_spin")

    def turn_to_heading(self, publisher, target_yaw_rad):
        session = self.session
        tolerance = math.radians(session.config.center_reposition_heading_tolerance_deg)
        deadline = session.now() + max(
            8.0,
            math.pi
            / max(0.01, abs(session.config.center_reposition_angular_speed_rad_s))
            + 3.0,
        )
        period = 1.0 / session.config.control_rate_hz
        while session.rclpy.ok() and session.now() <= deadline:
            session.rclpy.spin_once(session.node, timeout_sec=period)
            scan_age = session.fresh_scan_age_sec()
            odom_age = session.fresh_odom_age_sec()
            if scan_age is None or scan_age > session.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_turn")
            if odom_age is None or odom_age > session.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_turn")
            clearance = evaluate_clearance(session.latest_scan, session.config)
            update_safety_minima(session.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(
                    f"reposition_turn_clearance_failed:{clearance.reason}"
                )
            delta = shortest_angle_delta_rad(
                session.latest_odom_yaw_rad,
                target_yaw_rad,
            )
            if abs(delta) <= tolerance:
                return
            session.publish_turn_command(publisher, target_yaw_rad)
        raise RuntimeError("center_reposition_turn_timeout")

    def drive_forward(self, publisher, distance_m):
        session = self.session
        if session.latest_odom_pose is None:
            raise RuntimeError("fresh_odom_unavailable_before_reposition_drive")
        start_x = session.latest_odom_pose.x
        start_y = session.latest_odom_pose.y
        deadline = session.now() + max(
            8.0,
            distance_m
            / max(0.01, abs(session.config.center_reposition_linear_speed_mps))
            + 3.0,
        )
        period = 1.0 / session.config.control_rate_hz
        while session.rclpy.ok() and session.now() <= deadline:
            session.rclpy.spin_once(session.node, timeout_sec=period)
            scan_age = session.fresh_scan_age_sec()
            odom_age = session.fresh_odom_age_sec()
            if scan_age is None or scan_age > session.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_drive")
            if odom_age is None or odom_age > session.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_drive")
            clearance = evaluate_reposition_clearance(session.latest_scan, session.config)
            update_safety_minima(session.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(
                    f"reposition_drive_clearance_failed:{clearance.reason}"
                )
            dx = session.latest_odom_pose.x - start_x
            dy = session.latest_odom_pose.y - start_y
            driven = math.hypot(dx, dy)
            if driven >= distance_m:
                return driven
            session.publish_drive_command(publisher)
        raise RuntimeError("center_reposition_drive_timeout")

    def execute_center_reposition(self, publisher, action):
        session = self.session
        steps = list(action.steps)
        if (
            not steps
            and action.odom_heading_rad is not None
            and action.planned_distance_m is not None
        ):
            steps = [
                CenterRepositionStep(
                    kind="legacy",
                    reason=action.reason,
                    planned_distance_m=action.planned_distance_m,
                    local_heading_rad=action.local_heading_rad,
                    odom_heading_rad=action.odom_heading_rad,
                )
            ]
        if not action.ok or not steps:
            raise RuntimeError(action.reason)
        session.wait_for_fresh_inputs()
        session.print_reposition_prompt(action)
        session.refresh_fresh_inputs_after_prompt()
        clearance = evaluate_reposition_clearance(session.latest_scan, session.config)
        update_safety_minima(session.diagnostics, clearance)
        if not clearance.ok:
            raise RuntimeError(
                f"reposition_precheck_clearance_failed:{clearance.reason}"
            )

        start = session.now()
        total_driven = 0.0
        step_records = []
        for index, step in enumerate(steps):
            if index > 0:
                session.wait_for_fresh_inputs()
            step_start = session.now()
            step_record = step.to_dict()
            target_heading = step.odom_heading_rad
            if step.dynamic_heading:
                if session.latest_odom_yaw_rad is None:
                    raise RuntimeError(
                        "fresh_odom_unavailable_before_dynamic_lateral_turn"
                    )
                dynamic_heading = dynamic_lateral_heading_from_scan(
                    session.latest_scan,
                    session.latest_odom_yaw_rad,
                )
                target_heading = dynamic_heading["odom_heading_rad"]
                step_record["dynamic_heading_result"] = dynamic_heading
                step_record["odom_heading_rad"] = target_heading
            session.turn_to_heading(publisher, target_heading)
            stop_repeatedly(publisher, session.twist_factory, session.sleep_fn)
            session.wait_for_fresh_inputs()
            driven = session.drive_forward(publisher, step.planned_distance_m)
            stop_repeatedly(publisher, session.twist_factory, session.sleep_fn)
            total_driven += driven
            step_records.append(
                {
                    **step_record,
                    "driven_distance_m": driven,
                    "duration_sec": session.now() - step_start,
                }
            )
        record = action.to_dict()
        record["steps"] = step_records
        record["driven_distance_m"] = total_driven
        record["duration_sec"] = session.now() - start
        return record

    def execute_active_explore_cmd_vel(self, publisher, candidate, distance_limit_m=None):
        session = self.session
        previous_collecting = session.collecting_explore_map
        session.collecting_explore_map = True
        try:
            session.wait_for_fresh_inputs()
            move_limit = session.config.active_explore_max_single_move_m
            if distance_limit_m is not None:
                move_limit = min(move_limit, max(0.0, distance_limit_m))
            path_points = active_explore_curve_path(
                candidate,
                session.latest_odom_pose,
                move_limit,
            )
            session.print_active_explore_prompt(candidate, path_points)
            session.refresh_fresh_inputs_after_prompt()

            start = session.now()
            deadline = session.now() + max(
                8.0,
                move_limit
                / max(0.01, abs(session.config.active_explore_curve_linear_speed_mps))
                + 5.0,
            )
            period = 1.0 / session.config.control_rate_hz
            final_target = path_points[-1]
            candidate_goal = (
                candidate.path_world[-1]
                if candidate.path_world
                else (
                    candidate.simplified_path_world[-1]
                    if candidate.simplified_path_world
                    else (candidate.target_x, candidate.target_y)
                )
            )
            path_truncated = (
                distance_2d(final_target, candidate_goal)
                > session.config.active_explore_curve_goal_tolerance_m
            )
            previous_point = (
                float(session.latest_odom_pose.x),
                float(session.latest_odom_pose.y),
            )
            total_driven = 0.0
            curve_samples = []

            while session.rclpy.ok() and session.now() <= deadline:
                session.rclpy.spin_once(session.node, timeout_sec=period)
                scan_age = session.fresh_scan_age_sec()
                odom_age = session.fresh_odom_age_sec()
                if scan_age is None or scan_age > session.config.max_odom_scan_age_sec:
                    raise RuntimeError("stale_scan_during_active_explore_curve")
                if odom_age is None or odom_age > session.config.max_odom_scan_age_sec:
                    raise RuntimeError("stale_odom_during_active_explore_curve")
                if session.latest_odom_pose is None:
                    raise RuntimeError(
                        "fresh_odom_unavailable_during_active_explore_curve"
                    )

                current_point = (
                    float(session.latest_odom_pose.x),
                    float(session.latest_odom_pose.y),
                )
                delta = distance_2d(previous_point, current_point)
                if math.isfinite(delta):
                    total_driven += delta
                previous_point = current_point

                clearance = evaluate_reposition_clearance(
                    session.latest_scan,
                    session.config,
                )
                update_safety_minima(session.diagnostics, clearance)
                if not clearance.ok:
                    stop_repeatedly(publisher, session.twist_factory, session.sleep_fn)
                    if total_driven >= session.config.active_explore_min_progress_before_spin_m:
                        final_target_distance_m = distance_2d(
                            current_point,
                            final_target,
                        )
                        return active_explore_curve_execution_record(
                            candidate,
                            path_points,
                            curve_samples,
                            total_driven,
                            session.now() - start,
                            "clearance_stop_after_progress",
                            clearance_failure_reason=clearance.reason,
                            target_x=float(final_target[0]),
                            target_y=float(final_target[1]),
                            final_target_distance_m=final_target_distance_m,
                            goal_reached=(
                                final_target_distance_m
                                <= session.config.active_explore_curve_goal_tolerance_m
                            ),
                            path_truncated=path_truncated,
                        )
                    raise RuntimeError(
                        f"active_explore_curve_clearance_failed:{clearance.reason}"
                    )

                final_target_distance_m = distance_2d(current_point, final_target)
                if (
                    total_driven >= move_limit
                    or final_target_distance_m
                    <= session.config.active_explore_curve_goal_tolerance_m
                ):
                    stop_repeatedly(publisher, session.twist_factory, session.sleep_fn)
                    return active_explore_curve_execution_record(
                        candidate,
                        path_points,
                        curve_samples,
                        total_driven,
                        session.now() - start,
                        "completed",
                        target_x=float(final_target[0]),
                        target_y=float(final_target[1]),
                        final_target_distance_m=final_target_distance_m,
                        goal_reached=(
                            final_target_distance_m
                            <= session.config.active_explore_curve_goal_tolerance_m
                        ),
                        path_truncated=path_truncated,
                    )

                target = select_curve_lookahead_target(
                    path_points,
                    current_point,
                    session.config.active_explore_curve_lookahead_m,
                )
                linear_x, angular_z, alpha = pure_pursuit_curve_command(
                    session.latest_odom_pose,
                    target,
                    session.config.active_explore_curve_lookahead_m,
                    session.config.active_explore_curve_linear_speed_mps,
                    session.config.active_explore_curve_max_angular_rad_s,
                )
                remaining = max(0.0, move_limit - total_driven)
                linear_x = min(linear_x, remaining / max(period, 1e-6))
                curve_samples.append(
                    {
                        "odom_x": float(session.latest_odom_pose.x),
                        "odom_y": float(session.latest_odom_pose.y),
                        "odom_yaw_rad": math.radians(
                            float(session.latest_odom_pose.yaw_deg)
                        ),
                        "target_x": float(target[0]),
                        "target_y": float(target[1]),
                        "alpha_rad": alpha,
                        "linear_x_mps": linear_x,
                        "angular_z_rad_s": angular_z,
                        "front_clearance_m": clearance.front_min_m,
                        "left_clearance_m": clearance.left_min_m,
                        "right_clearance_m": clearance.right_min_m,
                    }
                )
                session.publish_curve_command(publisher, linear_x, angular_z)

            timeout_sec = deadline - start
            current_point = (
                float(session.latest_odom_pose.x),
                float(session.latest_odom_pose.y),
            )
            final_target_distance_m = distance_2d(current_point, final_target)
            record = active_explore_curve_execution_record(
                candidate,
                path_points,
                curve_samples,
                total_driven,
                session.now() - start,
                "timeout_stop_after_progress",
                timeout_sec=timeout_sec,
                target_x=float(final_target[0]),
                target_y=float(final_target[1]),
                final_target_distance_m=final_target_distance_m,
                goal_reached=(
                    final_target_distance_m
                    <= session.config.active_explore_curve_goal_tolerance_m
                ),
                path_truncated=path_truncated,
            )
            if total_driven >= session.config.active_explore_min_progress_before_spin_m:
                stop_repeatedly(publisher, session.twist_factory, session.sleep_fn)
                return record
            record["executed"] = False
            record["stop_reason"] = "active_explore_curve_timeout_before_progress"
            raise ActiveExploreMotionError(
                "active_explore_curve_timeout_before_progress",
                record,
            )
        except Exception:
            stop_repeatedly(publisher, session.twist_factory, session.sleep_fn)
            raise
        finally:
            session.collecting_explore_map = previous_collecting
