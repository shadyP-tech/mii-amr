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
)
from .scan_safety import (
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
            session.publish_spin_command(publisher)

        raise RuntimeError("ros_shutdown_during_arena_active_spin")


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
