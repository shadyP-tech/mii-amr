from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Callable

from arena_geometry_localizer import analyze_scan_samples
from arena_active_explore import (
    ActiveExplorePlan,
    build_local_grid_from_scan_samples,
    build_observed_local_grid_from_scan_samples,
    geometry_is_recoverable,
    plan_active_explore_recovery,
)

from .active_explore_policy import ActiveExplorePolicy
from .curve_following import (
    active_explore_curve_execution_record,
    active_explore_curve_path,
    pure_pursuit_curve_command,
    select_curve_lookahead_target,
)
from .diagnostics import (
    active_explore_config_from_arena_config,
    effective_recovery_mode,
    initial_diagnostics,
    spin_diagnostics_template,
    update_safety_minima,
    write_diagnostics_json,
)
from .explore_mission import (
    EXPLORE_ACTION_CONFIRM_SHADOW_MAP,
    EXPLORE_ACTION_DRIVE_CANDIDATE,
    EXPLORE_ACTION_FAIL,
    EXPLORE_ACTION_RUN_LOCALIZATION_SPIN,
    EXPLORE_PHASE_FAILED,
    EXPLORE_PHASE_LOCALIZATION_SPIN,
    EXPLORE_PHASE_SHADOW_MAPPING,
    ExploreMissionController,
    ExploreMissionMotionResult,
    shadow_map_status,
)
from .math_utils import distance_2d, shortest_angle_delta_rad
from .models import (
    ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE,
    ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN,
    DEFAULT_STOP_COUNT,
    DEFAULT_STOP_HZ,
    ActiveExploreMotionError,
    ArenaActiveSpinConfig,
    ArenaActiveSpinResult,
    CenterRepositionAction,
    CenterRepositionStep,
)
from .reposition import choose_center_reposition_action, choose_heater_approach_reposition_action
from .scan_safety import (
    dynamic_lateral_heading_from_scan,
    evaluate_clearance,
    evaluate_reposition_clearance,
    min_valid_scan_range,
    odom_pose_from_msg,
    pose_prior_from_localizer_result,
    scan_sample_from_msg,
)
from .temporary_map import (
    filter_scan_samples_with_temporary_obstacle_map,
    temporary_grid_localizer_obstacle_mask,
    valid_scan_range_count,
)


def stop_repeatedly(
    publisher,
    twist_factory: Callable[[], object],
    sleep_fn: Callable[[float], None] = time.sleep,
    count=DEFAULT_STOP_COUNT,
    hz=DEFAULT_STOP_HZ,
):
    delay = 1.0 / hz
    for _ in range(count):
        publisher.publish(twist_factory())
        sleep_fn(delay)


class ArenaActiveSpinSession:
    def __init__(
        self,
        node,
        config: ArenaActiveSpinConfig,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input,
        time_fn=time.time,
        sleep_fn=time.sleep,
        analyze_fn=analyze_scan_samples,
        temporary_map_callback=None,
        active_explore_plan_callback=None,
    ):
        self.node = node
        self.config = config
        self.rclpy = rclpy_module
        self.twist_factory = twist_factory
        self.input_fn = input_fn
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.analyze_fn = analyze_fn
        self.temporary_map_callback = temporary_map_callback
        self.active_explore_plan_callback = active_explore_plan_callback
        self.last_temporary_map_publish_sec = None
        self.latest_scan = None
        self.latest_scan_received_sec = None
        self.latest_odom_pose = None
        self.latest_odom_yaw_rad = None
        self.latest_odom_received_sec = None
        self.collecting = False
        self.collecting_explore_map = False
        self.samples = []
        self.explore_samples = []
        self.active_explore_startup_spin_samples = ()
        self.active_explore_final_spin_memory_samples = None
        self.rejected_samples = 0
        self.diagnostics = initial_diagnostics(config)
        self.active_explore_policy = ActiveExplorePolicy(self.diagnostics)
        self.active_explore_mission = ExploreMissionController(
            config,
            self.diagnostics,
            self.active_explore_policy,
        )
        self.scan_subscription = node.create_subscription(
            scan_msg_type,
            config.scan_topic,
            self.scan_callback,
            qos_profile,
        )
        self.odom_subscription = node.create_subscription(
            odom_msg_type,
            config.odom_topic,
            self.odom_callback,
            10,
        )

    def now(self):
        return self.time_fn()

    def scan_callback(self, msg):
        received_sec = self.now()
        self.latest_scan = msg
        self.latest_scan_received_sec = received_sec
        collecting_localizer = self.collecting
        collecting_explore = (
            effective_recovery_mode(self.config) == "active_explore"
            and self.config.active_explore_use_accumulated_map
            and (self.collecting or self.collecting_explore_map)
        )
        if not collecting_localizer and not collecting_explore:
            return
        if self.latest_odom_pose is None or self.latest_odom_received_sec is None:
            if collecting_localizer:
                self.rejected_samples += 1
            return
        if received_sec - self.latest_odom_received_sec > self.config.max_odom_scan_age_sec:
            if collecting_localizer:
                self.rejected_samples += 1
            return
        sample = scan_sample_from_msg(msg, self.latest_odom_pose)
        if collecting_localizer:
            self.samples.append(sample)
        if collecting_explore:
            self.append_explore_sample(sample)

    def append_explore_sample(self, sample):
        self.explore_samples.append(sample)
        max_samples = max(1, int(self.config.active_explore_map_max_samples))
        if len(self.explore_samples) > max_samples:
            del self.explore_samples[: len(self.explore_samples) - max_samples]
        self.diagnostics["active_explore"]["temporary_map"]["scan_samples_stored"] = (
            len(self.explore_samples)
        )
        self.publish_temporary_map_if_ready()

    def update_temporary_map_diagnostics(self, planning_grid, display_grid=None):
        display_counts = None
        if display_grid is not None:
            display_counts = display_grid.to_dict()
        self.diagnostics["active_explore"]["temporary_map"] = {
            "frame": "odom",
            "source": "accumulated_spin_and_recovery_scans",
            "scan_samples_stored": len(self.explore_samples),
            "display_grid": display_counts,
            "planning_grid": planning_grid.to_dict(),
            "grid": planning_grid.to_dict(),
        }

    def publish_temporary_map_if_ready(self, force=False, grid=None, display_grid=None):
        if self.temporary_map_callback is None:
            return
        if (
            effective_recovery_mode(self.config) != "active_explore"
            or not self.config.active_explore_use_accumulated_map
            or not self.explore_samples
            or self.latest_odom_pose is None
        ):
            return
        now = self.now()
        period_sec = self.config.active_explore_temporary_map_publish_period_sec
        if (
            not force
            and self.last_temporary_map_publish_sec is not None
            and now - self.last_temporary_map_publish_sec < period_sec
        ):
            return
        active_config = active_explore_config_from_arena_config(self.config)
        if grid is None:
            grid = build_local_grid_from_scan_samples(
                self.explore_samples,
                self.latest_odom_pose,
                active_config,
            )
        if display_grid is None:
            display_grid = build_observed_local_grid_from_scan_samples(
                self.explore_samples,
                self.latest_odom_pose,
                active_config,
            )
        self.update_temporary_map_diagnostics(grid, display_grid=display_grid)
        self.last_temporary_map_publish_sec = now
        try:
            self.temporary_map_callback(display_grid, grid)
        except Exception as exc:
            self.diagnostics["active_explore"]["temporary_map"]["publish_error"] = str(exc)

    def publish_active_explore_plan_if_ready(self, plan, move_limit_m):
        if self.active_explore_plan_callback is None or self.latest_odom_pose is None:
            return
        try:
            self.active_explore_plan_callback(
                plan,
                self.latest_odom_pose,
                move_limit_m,
            )
        except Exception as exc:
            self.diagnostics["active_explore"]["path_viz_publish_error"] = str(exc)

    def active_explore_spin_safety(self):
        self.wait_for_fresh_inputs()
        clearance = evaluate_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        full_min_range = min_valid_scan_range(self.latest_scan)
        required = self.config.min_front_clearance_m
        if full_min_range is None:
            ok = False
            reason = "spin_clearance_missing"
        elif full_min_range < required:
            ok = False
            reason = "spin_full_clearance_below_front_limit"
        else:
            ok = True
            reason = "ok"
        return {
            "ok": ok,
            "reason": reason,
            "full_min_range_m": full_min_range,
            "required_min_range_m": required,
            "sector_clearance": asdict(clearance),
        }

    def print_active_explore_spin_skip(self, spin_safety):
        print("\nArena-active post-motion spin skipped")
        print(f"  reason: {spin_safety['reason']}")
        print(f"  full min range: {spin_safety['full_min_range_m']}")
        print(f"  required min range: {spin_safety['required_min_range_m']}")
        print("  expected action: replan toward active-explore frontier without rotating")

    def print_active_explore_phase_spin_skip(self, reason):
        print("\nArena-active post-motion spin skipped")
        print(f"  reason: {reason}")
        print("  expected action: keep exploring obstacle shadow without rotating")

    def odom_callback(self, msg):
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_yaw_rad = math.radians(self.latest_odom_pose.yaw_deg)
        self.latest_odom_received_sec = self.now()

    def fresh_scan_age_sec(self):
        if self.latest_scan_received_sec is None:
            return None
        return self.now() - self.latest_scan_received_sec

    def fresh_odom_age_sec(self):
        if self.latest_odom_received_sec is None:
            return None
        return self.now() - self.latest_odom_received_sec

    def wait_for_fresh_inputs(self):
        deadline = self.now() + min(5.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable")

    def refresh_fresh_inputs_after_prompt(self):
        deadline = self.now() + min(2.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable_after_prompt")

    def cmd_vel_publisher_check(self):
        count = None
        if hasattr(self.node, "count_publishers"):
            count = self.node.count_publishers(self.config.cmd_vel_topic)
        unexpected = None if count is None else max(0, int(count) - 1)
        self.diagnostics["cmd_vel_publishers"] = {
            "count": count,
            "unexpected_count": unexpected,
            "allowed": self.config.allow_extra_cmd_vel_publishers,
        }
        if (
            unexpected is not None
            and unexpected > 0
            and not self.config.allow_extra_cmd_vel_publishers
        ):
            raise RuntimeError("unexpected_cmd_vel_publishers")

    def print_operator_prompt(self):
        scan_age = self.fresh_scan_age_sec()
        odom_age = self.fresh_odom_age_sec()
        clearance = evaluate_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        print("\nArena-active spin-only startup")
        print(f"  angular speed: {self.config.angular_speed_rad_s:.3f} rad/s")
        print(f"  direction: {self.config.spin_direction}")
        print(f"  max spin time: {self.config.max_spin_sec:.1f} s")
        print(f"  front clearance: {clearance.front_min_m}")
        print(f"  left clearance: {clearance.left_min_m}")
        print(f"  right clearance: {clearance.right_min_m}")
        print(f"  rear clearance: {clearance.rear_min_m}")
        print(f"  latest scan age: {scan_age}")
        print(f"  latest odom age: {odom_age}")
        print(f"  cmd_vel publisher check: {self.diagnostics['cmd_vel_publishers']}")
        print("  expected action: rotate in place 360 degrees")
        if not clearance.ok:
            raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start arena-active spin, or Ctrl+C to abort: ")

    def publish_spin_command(self, publisher):
        command = self.twist_factory()
        sign = 1.0 if self.config.spin_direction == "ccw" else -1.0
        command.angular.z = sign * abs(self.config.angular_speed_rad_s)
        publisher.publish(command)

    def run_spin(self, publisher):
        self.wait_for_fresh_inputs()
        self.cmd_vel_publisher_check()
        self.print_operator_prompt()
        self.refresh_fresh_inputs_after_prompt()

        previous_yaw = self.latest_odom_yaw_rad
        if previous_yaw is None:
            raise RuntimeError("fresh_odom_unavailable")
        self.collecting = True
        accumulated = 0.0
        target = 2.0 * math.pi - math.radians(self.config.spin_complete_tolerance_deg)
        period = 1.0 / self.config.control_rate_hz
        start = self.now()
        last_progress_time = start
        last_progress_yaw = 0.0

        while self.rclpy.ok():
            if self.now() - start > self.config.max_spin_sec:
                self.diagnostics["spin"]["timeout"] = True
                raise RuntimeError("arena_active_spin_timeout")
            self.publish_spin_command(publisher)
            self.rclpy.spin_once(self.node, timeout_sec=period)
            now = self.now()
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_spin")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_spin")

            clearance = evaluate_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")

            current_yaw = self.latest_odom_yaw_rad
            delta = shortest_angle_delta_rad(previous_yaw, current_yaw)
            accumulated += delta
            previous_yaw = current_yaw
            self.diagnostics["spin"]["accumulated_rad"] = accumulated
            self.diagnostics["spin"]["duration_sec"] = now - start
            if abs(accumulated) >= target:
                return accumulated, now - start

            if now - last_progress_time >= self.config.progress_check_sec:
                progress_rate = abs(accumulated - last_progress_yaw) / (now - last_progress_time)
                if progress_rate < self.config.min_angular_progress_rad_s:
                    raise RuntimeError("insufficient_angular_progress")
                last_progress_time = now
                last_progress_yaw = accumulated

        raise RuntimeError("ros_shutdown_during_arena_active_spin")

    def reset_spin_collection(self, attempt_index):
        self.collecting = False
        self.samples = []
        self.rejected_samples = 0
        self.diagnostics["spin"] = {
            **spin_diagnostics_template(),
            "attempt_index": attempt_index,
        }

    def run_spin_attempt(self, publisher, attempt_index):
        self.reset_spin_collection(attempt_index)
        self.run_spin(publisher)
        self.collecting = False
        stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
        self.sleep_fn(self.config.stop_settle_sec)
        if (
            attempt_index == 0
            and effective_recovery_mode(self.config) == "active_explore"
        ):
            self.active_explore_startup_spin_samples = tuple(self.samples)
        self.diagnostics["spin_attempts"].append(
            {
                **self.diagnostics["spin"],
                "scan_samples_collected": len(self.samples),
                "scan_samples_used": len(self.samples),
                "rejected_scan_samples": self.rejected_samples,
            }
        )

    def active_explore_localizer_filter_reason_disabled(self):
        if effective_recovery_mode(self.config) != "active_explore":
            return "not_active_explore"
        if not self.config.active_explore_use_accumulated_map:
            return "accumulated_map_disabled"
        attempt_index = self.diagnostics.get("spin", {}).get("attempt_index")
        if attempt_index == 0:
            return "first_spin"
        if self.active_explore_phase != ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN:
            return "not_final_active_explore_localization_spin"
        if not self.shadow_explore_complete:
            return "shadow_explore_not_complete"
        return None

    def active_explore_localizer_filter_grid(self, grid_samples=None):
        samples = self.explore_samples if grid_samples is None else grid_samples
        if not samples:
            return None, "no_temporary_map_samples"
        if self.latest_odom_pose is None:
            return None, "missing_latest_odom_pose"
        grid = build_local_grid_from_scan_samples(
            samples,
            self.latest_odom_pose,
            active_explore_config_from_arena_config(self.config),
        )
        self.update_temporary_map_diagnostics(grid)
        return grid, "ok"

    def active_explore_localizer_memory_samples(self):
        if self.active_explore_final_spin_memory_samples is not None:
            return list(self.active_explore_final_spin_memory_samples)
        return list(self.explore_samples)

    def dedupe_samples_by_identity(self, sample_groups):
        deduped = []
        seen = set()
        for samples in sample_groups:
            for sample in samples:
                key = id(sample)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(sample)
        return deduped

    def stride_valid_range_count_for_sample(self, sample):
        stride = max(1, int(self.config.range_stride))
        count = 0
        for index, value in enumerate(sample.ranges):
            if index % stride != 0:
                continue
            if value is None or not math.isfinite(value):
                continue
            if value < sample.range_min or value > sample.range_max:
                continue
            count += 1
        return count

    def select_samples_for_point_budget(self, samples, point_budget):
        samples = list(samples)
        if not samples or point_budget <= 0:
            return []
        point_counts = [
            self.stride_valid_range_count_for_sample(sample)
            for sample in samples
        ]
        valid_indices = [
            index
            for index, point_count in enumerate(point_counts)
            if point_count > 0
        ]
        if not valid_indices:
            return []
        total_points = sum(point_counts[index] for index in valid_indices)
        if total_points <= point_budget:
            return [samples[index] for index in valid_indices]
        average_points = total_points / len(valid_indices)
        target_count = max(1, int(point_budget / max(1.0, average_points)))
        target_count = min(target_count, len(valid_indices))
        if target_count == 1:
            selected_indices = [valid_indices[len(valid_indices) // 2]]
        else:
            selected_indices = []
            max_position = len(valid_indices) - 1
            for selection_index in range(target_count):
                position = round(selection_index * max_position / (target_count - 1))
                selected_indices.append(valid_indices[position])
            selected_indices = sorted(set(selected_indices))

        selected = []
        used_points = 0
        for index in selected_indices:
            point_count = point_counts[index]
            if selected and used_points + point_count > point_budget:
                continue
            selected.append(samples[index])
            used_points += point_count
        if not selected:
            selected.append(samples[selected_indices[0]])
        return selected

    def pose_bin_for_mapping_sample(self, sample):
        pose = sample.odom_pose
        if pose is None:
            return None
        try:
            x = float(pose.x)
            y = float(pose.y)
            yaw_deg = float(pose.yaw_deg)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw_deg)):
            return None
        yaw_wrapped_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
        return (
            math.floor(x / 0.15),
            math.floor(y / 0.15),
            math.floor(yaw_wrapped_deg / 20.0),
        )

    def active_explore_mapping_memory_candidates(self, memory_samples, excluded_ids):
        candidates = []
        pose_bins = set()
        for sample in memory_samples:
            if id(sample) in excluded_ids:
                continue
            pose_bin = self.pose_bin_for_mapping_sample(sample)
            if pose_bin is None or pose_bin in pose_bins:
                continue
            pose_bins.add(pose_bin)
            candidates.append(sample)
        return candidates, len(pose_bins)

    def active_explore_localizer_point_budgets(self, final_samples, startup_samples):
        max_points = max(1, int(self.config.max_points or 1))
        final_budget = int(round(max_points * 0.40))
        startup_budget = int(round(max_points * 0.40))
        mapping_budget = max(0, max_points - final_budget - startup_budget)
        if not startup_samples:
            final_budget += startup_budget
            startup_budget = 0
        if not final_samples:
            startup_budget += final_budget
            final_budget = 0
        if not final_samples and not startup_samples:
            mapping_budget = max_points
        return final_budget, startup_budget, mapping_budget

    def balanced_active_explore_localizer_samples(self, memory_samples):
        memory_samples = list(memory_samples)
        final_samples = list(self.samples)
        startup_samples = list(self.active_explore_startup_spin_samples)
        raw_combined_samples = self.dedupe_samples_by_identity(
            (memory_samples, final_samples)
        )
        final_budget, startup_budget, mapping_budget = (
            self.active_explore_localizer_point_budgets(
                final_samples,
                startup_samples,
            )
        )

        selected_final = self.select_samples_for_point_budget(
            final_samples,
            final_budget,
        )
        selected_startup = self.select_samples_for_point_budget(
            startup_samples,
            startup_budget,
        )
        excluded_ids = {id(sample) for sample in final_samples}
        excluded_ids.update(id(sample) for sample in startup_samples)
        mapping_candidates, pose_bin_count = (
            self.active_explore_mapping_memory_candidates(
                memory_samples,
                excluded_ids,
            )
        )
        selected_mapping = self.select_samples_for_point_budget(
            mapping_candidates,
            mapping_budget,
        )

        balanced_samples = []
        seen = set()
        selected_counts = {
            "final_spin": 0,
            "startup_spin": 0,
            "mapping_memory": 0,
        }
        for group_name, samples in (
            ("final_spin", selected_final),
            ("startup_spin", selected_startup),
            ("mapping_memory", selected_mapping),
        ):
            for sample in samples:
                key = id(sample)
                if key in seen:
                    continue
                seen.add(key)
                balanced_samples.append(sample)
                selected_counts[group_name] += 1

        diagnostics = {
            "raw_combined_sample_count": len(raw_combined_samples),
            "startup_spin_sample_count": len(startup_samples),
            "selected_final_spin_sample_count": selected_counts["final_spin"],
            "selected_startup_spin_sample_count": selected_counts["startup_spin"],
            "selected_mapping_memory_sample_count": selected_counts["mapping_memory"],
            "balanced_sample_count": len(balanced_samples),
            "localizer_sample_order": [
                group_name
                for group_name in (
                    "final_spin",
                    "startup_spin",
                    "mapping_memory",
                )
                if selected_counts[group_name] > 0
            ],
            "final_spin_point_budget": final_budget,
            "startup_spin_point_budget": startup_budget,
            "mapping_memory_point_budget": mapping_budget,
            "mapping_memory_candidate_count": len(mapping_candidates),
            "mapping_memory_pose_bin_count": pose_bin_count,
        }
        return balanced_samples, diagnostics

    def active_explore_filtered_localizer_samples(self):
        diagnostics = {
            "enabled": False,
            "reason": "",
            "input_sample_count": len(self.samples),
            "output_sample_count": len(self.samples),
            "memory_sample_count": 0,
            "final_spin_sample_count": len(self.samples),
            "startup_spin_sample_count": len(self.active_explore_startup_spin_samples),
            "raw_combined_sample_count": len(self.samples),
            "combined_sample_count": len(self.samples),
            "selected_final_spin_sample_count": len(self.samples),
            "selected_startup_spin_sample_count": 0,
            "selected_mapping_memory_sample_count": 0,
            "balanced_sample_count": len(self.samples),
            "localizer_sample_order": ["raw_current_spin"] if self.samples else [],
            "final_spin_point_budget": 0,
            "startup_spin_point_budget": 0,
            "mapping_memory_point_budget": 0,
            "mapping_memory_candidate_count": 0,
            "mapping_memory_pose_bin_count": 0,
            "used_accumulated_memory": False,
            "valid_ranges_before": valid_scan_range_count(self.samples),
            "valid_ranges_after": valid_scan_range_count(self.samples),
            "filtered_range_count": 0,
            "obstacle_mask_cell_count": 0,
            "protected_wall_cell_count": 0,
            "temporary_grid_cell_counts": None,
            "final_spin_attempt_index": self.diagnostics.get("spin", {}).get(
                "attempt_index"
            ),
        }
        disabled_reason = self.active_explore_localizer_filter_reason_disabled()
        if disabled_reason is not None:
            diagnostics["reason"] = disabled_reason
            self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return self.samples

        memory_samples = self.active_explore_localizer_memory_samples()
        localizer_samples, balance_diagnostics = (
            self.balanced_active_explore_localizer_samples(memory_samples)
        )
        valid_ranges_before = valid_scan_range_count(localizer_samples)
        diagnostics["memory_sample_count"] = len(memory_samples)
        diagnostics.update(balance_diagnostics)
        diagnostics["combined_sample_count"] = len(localizer_samples)
        diagnostics["input_sample_count"] = len(localizer_samples)
        diagnostics["output_sample_count"] = len(localizer_samples)
        diagnostics["used_accumulated_memory"] = (
            balance_diagnostics["selected_startup_spin_sample_count"] > 0
            or balance_diagnostics["selected_mapping_memory_sample_count"] > 0
        )
        diagnostics["valid_ranges_before"] = valid_ranges_before
        diagnostics["valid_ranges_after"] = valid_ranges_before

        grid, grid_reason = self.active_explore_localizer_filter_grid(memory_samples)
        if grid is None:
            diagnostics["reason"] = grid_reason
            self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return localizer_samples

        diagnostics["temporary_grid_cell_counts"] = grid.to_dict()["cell_counts"]
        obstacle_mask, protected_wall_cells, mask_diagnostics = (
            temporary_grid_localizer_obstacle_mask(grid)
        )
        diagnostics.update(mask_diagnostics)
        diagnostics["obstacle_mask_cell_count"] = len(obstacle_mask)
        diagnostics["protected_wall_cell_count"] = len(protected_wall_cells)
        if not obstacle_mask:
            diagnostics["reason"] = "no_temporary_obstacle_mask"
            self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return localizer_samples

        filtered_samples, filtered_range_count = (
            filter_scan_samples_with_temporary_obstacle_map(
                localizer_samples,
                grid,
                obstacle_mask,
            )
        )
        filtered_valid_ranges_after = valid_scan_range_count(filtered_samples)
        if valid_ranges_before > 0 and filtered_valid_ranges_after <= 0:
            diagnostics["reason"] = "obstacle_filter_removed_all_ranges"
            diagnostics["filtered_range_count"] = filtered_range_count
            diagnostics["filtered_valid_ranges_after"] = filtered_valid_ranges_after
            self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return localizer_samples

        diagnostics["enabled"] = True
        diagnostics["reason"] = "filtered_temporary_obstacles"
        diagnostics["output_sample_count"] = len(filtered_samples)
        diagnostics["filtered_range_count"] = filtered_range_count
        diagnostics["valid_ranges_after"] = filtered_valid_ranges_after
        self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
        return filtered_samples

    def analyze_result(self):
        localizer_samples = self.active_explore_filtered_localizer_samples()
        if len(localizer_samples) < self.config.min_scan_samples:
            raise RuntimeError(
                "insufficient_scan_samples:"
                f"{len(localizer_samples)}<{self.config.min_scan_samples}"
            )
        self.diagnostics["samples"]["scan_samples_used"] = len(localizer_samples)
        result = self.analyze_fn(
            localizer_samples,
            self.config.arena_config,
            range_stride=self.config.range_stride,
            max_points=self.config.max_points,
        )
        self.diagnostics["localizer_result"] = result.to_dict()
        return result

    def pose_prior_from_result_or_raise(self, result):
        if not result.success:
            raise RuntimeError(f"arena_localizer_failed:{result.failure_reason}")
        pose_prior = pose_prior_from_localizer_result(result)
        if pose_prior is None:
            raise RuntimeError("arena_localizer_missing_pose_prior")
        return pose_prior

    def first_sample_origin_yaw_rad(self):
        for sample in self.samples:
            if sample.odom_pose is not None:
                return math.radians(sample.odom_pose.yaw_deg)
        return 0.0

    def publish_turn_command(self, publisher, target_yaw_rad):
        if self.latest_odom_yaw_rad is None:
            raise RuntimeError("fresh_odom_unavailable_during_reposition_turn")
        command = self.twist_factory()
        delta = shortest_angle_delta_rad(self.latest_odom_yaw_rad, target_yaw_rad)
        command.angular.z = (
            1.0 if delta >= 0.0 else -1.0
        ) * abs(self.config.center_reposition_angular_speed_rad_s)
        publisher.publish(command)

    def turn_to_heading(self, publisher, target_yaw_rad):
        tolerance = math.radians(self.config.center_reposition_heading_tolerance_deg)
        deadline = self.now() + max(
            8.0,
            math.pi / max(0.01, abs(self.config.center_reposition_angular_speed_rad_s))
            + 3.0,
        )
        period = 1.0 / self.config.control_rate_hz
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=period)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_turn")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_turn")
            clearance = evaluate_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"reposition_turn_clearance_failed:{clearance.reason}")
            delta = shortest_angle_delta_rad(self.latest_odom_yaw_rad, target_yaw_rad)
            if abs(delta) <= tolerance:
                return
            self.publish_turn_command(publisher, target_yaw_rad)
        raise RuntimeError("center_reposition_turn_timeout")

    def publish_drive_command(self, publisher):
        command = self.twist_factory()
        command.linear.x = abs(self.config.center_reposition_linear_speed_mps)
        publisher.publish(command)

    def drive_forward(self, publisher, distance_m):
        if self.latest_odom_pose is None:
            raise RuntimeError("fresh_odom_unavailable_before_reposition_drive")
        start_x = self.latest_odom_pose.x
        start_y = self.latest_odom_pose.y
        deadline = self.now() + max(
            8.0,
            distance_m / max(0.01, abs(self.config.center_reposition_linear_speed_mps))
            + 3.0,
        )
        period = 1.0 / self.config.control_rate_hz
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=period)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_drive")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_drive")
            clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"reposition_drive_clearance_failed:{clearance.reason}")
            dx = self.latest_odom_pose.x - start_x
            dy = self.latest_odom_pose.y - start_y
            if math.hypot(dx, dy) >= distance_m:
                return math.hypot(dx, dy)
            self.publish_drive_command(publisher)
        raise RuntimeError("center_reposition_drive_timeout")

    def print_reposition_prompt(self, action: CenterRepositionAction):
        print("\nArena-active reposition recovery")
        print(f"  nearest short wall: {action.nearest_axis_side}")
        print(f"  away direction: {action.away_axis_side}")
        print(f"  nearest range: {action.nearest_short_wall_range_m}")
        print(f"  target nearest range: {action.target_nearest_short_wall_range_m}")
        print(f"  suspected heater wall: {action.suspected_heater_axis_side}")
        print(f"  suspected heater range: {action.suspected_heater_range_m}")
        print(f"  target heater range: {action.heater_approach_target_range_m}")
        print(f"  heater scores: {action.heater_scores}")
        print(f"  heater delta: {action.heater_profile_delta}")
        print(f"  lateral offset: {action.lateral_offset_m}")
        print(f"  target lateral offset: {action.lateral_target_offset_m}")
        steps = list(action.steps)
        if not steps and action.odom_heading_rad is not None and action.planned_distance_m is not None:
            steps = [
                CenterRepositionStep(
                    kind="legacy",
                    reason=action.reason,
                    planned_distance_m=action.planned_distance_m,
                    local_heading_rad=action.local_heading_rad,
                    odom_heading_rad=action.odom_heading_rad,
                )
            ]
        for index, step in enumerate(steps, start=1):
            heading_text = f"{math.degrees(step.odom_heading_rad):.1f} deg"
            if step.dynamic_heading:
                heading_text = f"dynamic ({step.dynamic_heading_source}), initial estimate {heading_text}"
            print(
                f"  step {index} {step.kind}: "
                f"distance={step.planned_distance_m:.3f} m, "
                f"target odom heading={heading_text}"
            )
        if action.lateral_step_skipped:
            print(f"  lateral step: skipped ({action.lateral_skip_reason})")
        print("  expected action: turn, drive, optionally turn sideways, drive, then spin again")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start center reposition, or Ctrl+C to abort: ")

    def execute_center_reposition(self, publisher, action: CenterRepositionAction):
        steps = list(action.steps)
        if not steps and action.odom_heading_rad is not None and action.planned_distance_m is not None:
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
        self.wait_for_fresh_inputs()
        self.print_reposition_prompt(action)
        self.refresh_fresh_inputs_after_prompt()
        clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        if not clearance.ok:
            raise RuntimeError(f"reposition_precheck_clearance_failed:{clearance.reason}")

        start = self.now()
        total_driven = 0.0
        step_records = []
        for index, step in enumerate(steps):
            if index > 0:
                self.wait_for_fresh_inputs()
            step_start = self.now()
            step_record = step.to_dict()
            target_heading = step.odom_heading_rad
            if step.dynamic_heading:
                if self.latest_odom_yaw_rad is None:
                    raise RuntimeError("fresh_odom_unavailable_before_dynamic_lateral_turn")
                dynamic_heading = dynamic_lateral_heading_from_scan(
                    self.latest_scan,
                    self.latest_odom_yaw_rad,
                )
                target_heading = dynamic_heading["odom_heading_rad"]
                step_record["dynamic_heading_result"] = dynamic_heading
                step_record["odom_heading_rad"] = target_heading
            self.turn_to_heading(publisher, target_heading)
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.wait_for_fresh_inputs()
            driven = self.drive_forward(publisher, step.planned_distance_m)
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            total_driven += driven
            step_records.append(
                {
                    **step_record,
                    "driven_distance_m": driven,
                    "duration_sec": self.now() - step_start,
                }
            )
        record = action.to_dict()
        record["steps"] = step_records
        record["driven_distance_m"] = total_driven
        record["duration_sec"] = self.now() - start
        return record

    def plan_active_explore_recovery(self, result):
        active_config = active_explore_config_from_arena_config(self.config)
        geometry_ok, reason = geometry_is_recoverable(result, active_config)
        if not geometry_ok:
            return ActiveExplorePlan(False, reason, None, (), None)
        self.wait_for_fresh_inputs()
        origin_yaw = self.first_sample_origin_yaw_rad()
        grid = None
        if (
            self.config.active_explore_use_accumulated_map
            and self.explore_samples
            and self.latest_odom_pose is not None
        ):
            grid = build_local_grid_from_scan_samples(
                self.explore_samples,
                self.latest_odom_pose,
                active_config,
            )
            display_grid = build_observed_local_grid_from_scan_samples(
                self.explore_samples,
                self.latest_odom_pose,
                active_config,
            )
            self.update_temporary_map_diagnostics(grid, display_grid=display_grid)
            self.publish_temporary_map_if_ready(
                force=True,
                grid=grid,
                display_grid=display_grid,
            )
        return plan_active_explore_recovery(
            result,
            self.latest_scan,
            self.latest_odom_pose,
            active_config,
            origin_yaw_rad=origin_yaw,
            grid=grid,
        )

    @property
    def active_explore_frontier_goal(self):
        return self.active_explore_policy.frontier_goal

    @active_explore_frontier_goal.setter
    def active_explore_frontier_goal(self, value):
        self.active_explore_policy.frontier_goal = value
        self.diagnostics["active_explore"]["persistent_frontier_goal"] = (
            self.active_explore_policy.frontier_goal_diagnostics()
        )

    @property
    def active_explore_phase(self):
        return self.active_explore_policy.phase

    @active_explore_phase.setter
    def active_explore_phase(self, value):
        self.active_explore_policy.set_phase(value)
        self.active_explore_mission.sync_from_policy()

    @property
    def shadow_frontier_empty_replans(self):
        return self.active_explore_policy.shadow_frontier_empty_replans

    @shadow_frontier_empty_replans.setter
    def shadow_frontier_empty_replans(self, value):
        self.active_explore_policy.shadow_frontier_empty_replans = value
        self.active_explore_mission.shadow_confirmation_count = int(value)
        self.active_explore_policy.update_phase_diagnostics()

    @property
    def shadow_explore_complete(self):
        return self.active_explore_policy.shadow_explore_complete

    @shadow_explore_complete.setter
    def shadow_explore_complete(self, value):
        self.active_explore_policy.shadow_explore_complete = value
        self.active_explore_mission.sync_from_policy()
        self.active_explore_policy.update_phase_diagnostics()

    def active_explore_frontier_goal_diagnostics(self):
        return self.active_explore_policy.frontier_goal_diagnostics()

    def clear_active_explore_frontier_goal(self, reason):
        return self.active_explore_policy.clear_frontier_goal(reason)

    def store_active_explore_frontier_goal(self, candidate, attempt_index):
        return self.active_explore_policy.store_frontier_goal(candidate, attempt_index)

    def update_active_explore_frontier_progress(self, driven_distance_m):
        return self.active_explore_policy.update_frontier_progress(driven_distance_m)

    def set_active_explore_phase(self, phase):
        return self.active_explore_policy.set_phase(phase)

    def update_active_explore_phase_diagnostics(self):
        return self.active_explore_policy.update_phase_diagnostics()

    def latest_odom_point(self):
        if self.latest_odom_pose is None:
            return None
        return [float(self.latest_odom_pose.x), float(self.latest_odom_pose.y)]

    def apply_active_explore_phase_selection(self, plan, attempt_index):
        return self.active_explore_policy.select_for_phase(
            plan,
            attempt_index,
            current_pose_point=self.latest_odom_point(),
        )

    def print_active_explore_prompt(self, candidate, path_points):
        print("\nArena-active active-explore recovery")
        print(f"  executor: {self.config.recovery_executor}")
        print(f"  selected candidate: {candidate.kind}")
        print(f"  score: {candidate.score}")
        print(f"  score components: {candidate.score_components}")
        print(f"  path length: {candidate.path_length_m}")
        print(
            "  curve follower: "
            f"lookahead={self.config.active_explore_curve_lookahead_m:.3f} m, "
            f"linear={self.config.active_explore_curve_linear_speed_mps:.3f} m/s, "
            f"max angular={self.config.active_explore_curve_max_angular_rad_s:.3f} rad/s"
        )
        print(f"  curve path points: {len(path_points)}")
        if candidate.kind == "obstacle_shadow_frontier":
            print(
                "  expected action: follow short odom-frame curve, "
                "update temporary map, then replan without forced spin"
            )
        elif self.active_explore_phase == ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE:
            print(
                "  expected action: follow localization-friendly curve, "
                "stop, then spin if safe"
            )
        else:
            print("  expected action: follow short odom-frame curve and stop")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start active-explore recovery, or Ctrl+C to abort: ")

    def publish_curve_command(self, publisher, linear_x, angular_z):
        command = self.twist_factory()
        command.linear.x = float(linear_x)
        command.angular.z = float(angular_z)
        publisher.publish(command)

    def execute_active_explore_cmd_vel(self, publisher, candidate, distance_limit_m=None):
        previous_collecting = self.collecting_explore_map
        self.collecting_explore_map = True
        try:
            self.wait_for_fresh_inputs()
            move_limit = self.config.active_explore_max_single_move_m
            if distance_limit_m is not None:
                move_limit = min(move_limit, max(0.0, distance_limit_m))
            path_points = active_explore_curve_path(
                candidate,
                self.latest_odom_pose,
                move_limit,
            )
            self.print_active_explore_prompt(candidate, path_points)
            self.refresh_fresh_inputs_after_prompt()

            start = self.now()
            deadline = self.now() + max(
                8.0,
                move_limit
                / max(0.01, abs(self.config.active_explore_curve_linear_speed_mps))
                + 5.0,
            )
            period = 1.0 / self.config.control_rate_hz
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
                > self.config.active_explore_curve_goal_tolerance_m
            )
            previous_point = (
                float(self.latest_odom_pose.x),
                float(self.latest_odom_pose.y),
            )
            total_driven = 0.0
            curve_samples = []

            while self.rclpy.ok() and self.now() <= deadline:
                self.rclpy.spin_once(self.node, timeout_sec=period)
                scan_age = self.fresh_scan_age_sec()
                odom_age = self.fresh_odom_age_sec()
                if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                    raise RuntimeError("stale_scan_during_active_explore_curve")
                if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                    raise RuntimeError("stale_odom_during_active_explore_curve")
                if self.latest_odom_pose is None:
                    raise RuntimeError("fresh_odom_unavailable_during_active_explore_curve")

                current_point = (
                    float(self.latest_odom_pose.x),
                    float(self.latest_odom_pose.y),
                )
                delta = distance_2d(previous_point, current_point)
                if math.isfinite(delta):
                    total_driven += delta
                previous_point = current_point

                clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
                update_safety_minima(self.diagnostics, clearance)
                if not clearance.ok:
                    stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
                    if total_driven >= self.config.active_explore_min_progress_before_spin_m:
                        final_target_distance_m = distance_2d(current_point, final_target)
                        return active_explore_curve_execution_record(
                            candidate,
                            path_points,
                            curve_samples,
                            total_driven,
                            self.now() - start,
                            "clearance_stop_after_progress",
                            clearance_failure_reason=clearance.reason,
                            target_x=float(final_target[0]),
                            target_y=float(final_target[1]),
                            final_target_distance_m=final_target_distance_m,
                            goal_reached=(
                                final_target_distance_m
                                <= self.config.active_explore_curve_goal_tolerance_m
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
                    <= self.config.active_explore_curve_goal_tolerance_m
                ):
                    stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
                    return active_explore_curve_execution_record(
                        candidate,
                        path_points,
                        curve_samples,
                        total_driven,
                        self.now() - start,
                        "completed",
                        target_x=float(final_target[0]),
                        target_y=float(final_target[1]),
                        final_target_distance_m=final_target_distance_m,
                        goal_reached=(
                            final_target_distance_m
                            <= self.config.active_explore_curve_goal_tolerance_m
                        ),
                        path_truncated=path_truncated,
                    )

                target = select_curve_lookahead_target(
                    path_points,
                    current_point,
                    self.config.active_explore_curve_lookahead_m,
                )
                linear_x, angular_z, alpha = pure_pursuit_curve_command(
                    self.latest_odom_pose,
                    target,
                    self.config.active_explore_curve_lookahead_m,
                    self.config.active_explore_curve_linear_speed_mps,
                    self.config.active_explore_curve_max_angular_rad_s,
                )
                remaining = max(0.0, move_limit - total_driven)
                linear_x = min(linear_x, remaining / max(period, 1e-6))
                curve_samples.append(
                    {
                        "odom_x": float(self.latest_odom_pose.x),
                        "odom_y": float(self.latest_odom_pose.y),
                        "odom_yaw_rad": math.radians(float(self.latest_odom_pose.yaw_deg)),
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
                self.publish_curve_command(publisher, linear_x, angular_z)

            timeout_sec = deadline - start
            current_point = (
                float(self.latest_odom_pose.x),
                float(self.latest_odom_pose.y),
            )
            final_target_distance_m = distance_2d(current_point, final_target)
            record = active_explore_curve_execution_record(
                candidate,
                path_points,
                curve_samples,
                total_driven,
                self.now() - start,
                "timeout_stop_after_progress",
                timeout_sec=timeout_sec,
                target_x=float(final_target[0]),
                target_y=float(final_target[1]),
                final_target_distance_m=final_target_distance_m,
                goal_reached=(
                    final_target_distance_m
                    <= self.config.active_explore_curve_goal_tolerance_m
                ),
                path_truncated=path_truncated,
            )
            if total_driven >= self.config.active_explore_min_progress_before_spin_m:
                stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
                return record
            record["executed"] = False
            record["stop_reason"] = "active_explore_curve_timeout_before_progress"
            raise ActiveExploreMotionError(
                "active_explore_curve_timeout_before_progress",
                record,
            )
        except Exception:
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            raise
        finally:
            self.collecting_explore_map = previous_collecting

    def run_active_explore_localization_spin(self, publisher, attempt_record):
        spin_safety = self.active_explore_spin_safety()
        attempt_record["post_motion_spin_safety"] = spin_safety
        if not spin_safety["ok"]:
            decision = {
                "action": "skip",
                "reason": spin_safety["reason"],
                "active_explore_phase": self.active_explore_phase,
                "shadow_explore_complete": self.shadow_explore_complete,
                "spin_safety": spin_safety,
            }
            attempt_record["post_motion_spin_decision"] = decision
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            attempt_record["post_recovery_spin_skipped"] = True
            attempt_record["post_recovery_spin_skip_reason"] = spin_safety["reason"]
            self.active_explore_mission.record_spin({"success": False})
            self.print_active_explore_spin_skip(spin_safety)
            return None

        mission_decision = attempt_record.get("mission_decision", {})
        spin_reason = "localization_pose_reached"
        if mission_decision.get("action") == EXPLORE_ACTION_RUN_LOCALIZATION_SPIN:
            spin_reason = mission_decision.get("reason") or spin_reason
        decision = {
            "action": "spin",
            "reason": spin_reason,
            "active_explore_phase": ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN,
            "shadow_explore_complete": self.shadow_explore_complete,
            "spin_safety": spin_safety,
        }
        attempt_record["post_motion_spin_decision"] = decision
        attempt_record["post_recovery_spin_skipped"] = False
        self.active_explore_policy.set_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN)
        previous_memory_samples = self.active_explore_final_spin_memory_samples
        self.active_explore_final_spin_memory_samples = tuple(self.explore_samples)
        try:
            self.run_spin_attempt(
                publisher,
                attempt_index=len(self.diagnostics["spin_attempts"]),
            )
            result = self.analyze_result()
        finally:
            self.active_explore_final_spin_memory_samples = previous_memory_samples
        attempt_record["post_recovery_spin_result"] = result.to_dict()
        attempt_record["post_recovery_success"] = result.success
        attempt_record["post_recovery_failure_reason"] = result.failure_reason
        attempt_record["post_recovery_classifier_reason"] = (
            result.short_wall_classification.reason
        )
        self.active_explore_mission.record_spin(result)
        return result

    def run_legacy_recovery(self, publisher, result):
        reposition_attempts = 0
        while (
            not result.success
            and reposition_attempts < self.config.center_reposition_max_attempts
        ):
            origin_yaw = self.first_sample_origin_yaw_rad()
            action = choose_center_reposition_action(result, self.config, origin_yaw)
            attempt_record = {
                "attempt_index": len(self.diagnostics["reposition"]["attempts"]),
                "stage": "center",
                "previous_failure_reason": result.failure_reason,
                "previous_classifier_reason": result.short_wall_classification.reason,
                "action": action.to_dict(),
            }
            self.diagnostics["reposition"]["attempts"].append(attempt_record)
            if not action.ok:
                break
            self.diagnostics["fallback_used"] = True
            motion_record = self.execute_center_reposition(publisher, action)
            attempt_record["motion"] = motion_record
            reposition_attempts += 1
            self.run_spin_attempt(publisher, attempt_index=reposition_attempts)
            result = self.analyze_result()
            attempt_record["post_reposition_success"] = result.success
            attempt_record["post_reposition_failure_reason"] = result.failure_reason
            attempt_record["post_reposition_classifier_reason"] = (
                result.short_wall_classification.reason
            )

        heater_approach_attempts = 0
        while (
            not result.success
            and self.config.center_reposition_enable_heater_approach
            and heater_approach_attempts
            < self.config.center_reposition_heater_approach_max_attempts
        ):
            origin_yaw = self.first_sample_origin_yaw_rad()
            action = choose_heater_approach_reposition_action(
                result,
                self.config,
                origin_yaw,
            )
            attempt_record = {
                "attempt_index": len(self.diagnostics["reposition"]["attempts"]),
                "stage": "heater_approach",
                "previous_failure_reason": result.failure_reason,
                "previous_classifier_reason": result.short_wall_classification.reason,
                "action": action.to_dict(),
            }
            self.diagnostics["reposition"]["attempts"].append(attempt_record)
            if not action.ok:
                break
            self.diagnostics["fallback_used"] = True
            motion_record = self.execute_center_reposition(publisher, action)
            attempt_record["motion"] = motion_record
            heater_approach_attempts += 1
            self.run_spin_attempt(
                publisher,
                attempt_index=len(self.diagnostics["spin_attempts"]),
            )
            result = self.analyze_result()
            attempt_record["post_reposition_success"] = result.success
            attempt_record["post_reposition_failure_reason"] = result.failure_reason
            attempt_record["post_reposition_classifier_reason"] = (
                result.short_wall_classification.reason
            )
        return result

    def run_active_explore_recovery(self, publisher, result):
        if self.config.recovery_executor not in {"dry_run", "cmd_vel"}:
            raise RuntimeError(f"active_explore_executor_unknown:{self.config.recovery_executor}")

        self.active_explore_mission.sync_from_policy()
        total_distance = float(self.diagnostics["active_explore"].get("total_distance_m", 0.0))
        while not result.success:
            attempt_index = len(self.diagnostics["active_explore"]["attempts"])
            plan = self.plan_active_explore_recovery(result)
            map_status = shadow_map_status(plan.grid, plan)
            self.diagnostics["active_explore"]["shadow_map_status"] = map_status
            decision = self.active_explore_mission.next_decision(
                result,
                plan,
                map_status,
                current_pose_point=self.latest_odom_point(),
            )
            effective_plan = decision.plan
            selection_diagnostics = decision.diagnostics
            plan_dict = plan.to_dict()
            rejected_unknown = [
                candidate
                for candidate in effective_plan.candidates
                if candidate.rejection_reason == "goal_unknown"
            ]
            attempt_record = {
                "attempt_index": attempt_index,
                "stage": "active_explore",
                "executor": self.config.recovery_executor,
                "previous_failure_reason": result.failure_reason,
                "previous_classifier_reason": result.short_wall_classification.reason,
                "plan": effective_plan.to_dict(),
                "raw_plan": plan_dict,
                "mission_decision": decision.to_dict(),
                **selection_diagnostics,
                "local_grid_stats": (
                    None
                    if effective_plan.grid is None
                    else effective_plan.grid.to_dict()["cell_counts"]
                ),
                "shadow_map_status": map_status,
                "rejected_unknown_space_candidates": len(rejected_unknown),
                "execution": {
                    "executed": False,
                    "stop_reason": "not_started",
                    "driven_distance_m": 0.0,
                },
            }
            self.diagnostics["active_explore"]["attempts"].append(attempt_record)
            preview_limit = min(
                self.config.active_explore_max_single_move_m,
                max(0.0, self.config.active_explore_max_total_distance_m - total_distance),
            )
            self.publish_active_explore_plan_if_ready(effective_plan, preview_limit)

            if decision.action == EXPLORE_ACTION_CONFIRM_SHADOW_MAP:
                attempt_record["execution"]["stop_reason"] = decision.reason
                continue

            if decision.action == EXPLORE_ACTION_FAIL:
                attempt_record["execution"]["stop_reason"] = decision.reason
                break

            if decision.action == EXPLORE_ACTION_RUN_LOCALIZATION_SPIN:
                spin_result = self.run_active_explore_localization_spin(
                    publisher,
                    attempt_record,
                )
                if spin_result is not None:
                    result = spin_result
                if self.active_explore_mission.phase == EXPLORE_PHASE_LOCALIZATION_SPIN:
                    continue
                if self.active_explore_mission.phase == EXPLORE_PHASE_FAILED:
                    break
                continue

            if decision.action != EXPLORE_ACTION_DRIVE_CANDIDATE:
                attempt_record["execution"]["stop_reason"] = decision.reason
                break

            if self.config.recovery_executor == "dry_run":
                attempt_record["execution"] = {
                    "executor": "dry_run",
                    "executed": False,
                    "stop_reason": "dry_run",
                    "driven_distance_m": 0.0,
                }
                break

            remaining_distance = (
                self.config.active_explore_max_total_distance_m - total_distance
            )
            if remaining_distance <= 0.0:
                attempt_record["execution"]["stop_reason"] = (
                    "active_explore_total_distance_exhausted"
                )
                break
            self.diagnostics["fallback_used"] = True
            try:
                motion_record = self.execute_active_explore_cmd_vel(
                    publisher,
                    decision.selected,
                    distance_limit_m=remaining_distance,
                )
            except ActiveExploreMotionError as exc:
                motion_record = exc.record
                attempt_record["execution"] = motion_record
                total_distance += float(motion_record.get("driven_distance_m", 0.0))
                self.diagnostics["active_explore"]["total_distance_m"] = total_distance
                self.update_active_explore_frontier_progress(
                    motion_record.get("driven_distance_m", 0.0)
                )
                self.clear_active_explore_frontier_goal(exc.reason)
                raise
            except Exception:
                self.clear_active_explore_frontier_goal("active_explore_motion_failed")
                raise
            total_distance += float(motion_record.get("driven_distance_m", 0.0))
            self.update_active_explore_frontier_progress(
                motion_record.get("driven_distance_m", 0.0)
            )
            if total_distance > self.config.active_explore_max_total_distance_m + 1e-6:
                motion_record["stop_reason"] = "active_explore_total_distance_exceeded"
                attempt_record["execution"] = motion_record
                self.diagnostics["active_explore"]["total_distance_m"] = total_distance
                self.clear_active_explore_frontier_goal(
                    "active_explore_total_distance_exceeded"
                )
                raise RuntimeError("active_explore_total_distance_exceeded")

            attempt_record["execution"] = motion_record
            self.diagnostics["active_explore"]["total_distance_m"] = total_distance
            motion_result = ExploreMissionMotionResult.from_execution_record(motion_record)
            self.active_explore_mission.record_motion(decision, motion_result)
            if self.active_explore_mission.phase == EXPLORE_PHASE_SHADOW_MAPPING:
                decision = {
                    "action": "skip",
                    "reason": "shadow_exploration_not_complete",
                    "active_explore_phase": self.active_explore_phase,
                    "shadow_explore_complete": self.shadow_explore_complete,
                    "shadow_frontier_status": self.diagnostics["active_explore"].get(
                        "shadow_frontier_status"
                    ),
                }
                attempt_record["post_motion_spin_decision"] = decision
                attempt_record["post_recovery_spin_skipped"] = True
                attempt_record["post_recovery_spin_skip_reason"] = decision["reason"]
                self.print_active_explore_phase_spin_skip(decision["reason"])
                continue

            spin_result = self.run_active_explore_localization_spin(
                publisher,
                attempt_record,
            )
            if spin_result is not None:
                result = spin_result
        return result

    def finish_failure(self, reason, exception=None):
        self.diagnostics["success"] = False
        self.diagnostics["failure_reason"] = reason
        if exception is not None:
            self.diagnostics["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(False, reason, None, self.diagnostics, path)

    def finish_success(self, pose_prior):
        self.diagnostics["success"] = True
        self.diagnostics["failure_reason"] = ""
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        self.diagnostics["initialpose"] = {
            "published": False,
            "reason": "dry_run" if self.config.dry_run else "pending_runner_publication",
        }
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(True, None, pose_prior, self.diagnostics, path)

    def run(self, publisher):
        try:
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.run_spin_attempt(publisher, attempt_index=0)
            result = self.analyze_result()
            recovery_mode = effective_recovery_mode(self.config)
            if not result.success and recovery_mode == "legacy":
                result = self.run_legacy_recovery(publisher, result)
            elif not result.success and recovery_mode == "active_explore":
                result = self.run_active_explore_recovery(publisher, result)
            pose_prior = self.pose_prior_from_result_or_raise(result)
            return self.finish_success(pose_prior)
        except KeyboardInterrupt:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure("keyboard_interrupt")
        except Exception as exc:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure(str(exc), exception=exc)


def run_arena_active_spin(
    node,
    publisher,
    config: ArenaActiveSpinConfig,
    rclpy_module,
    twist_factory,
    scan_msg_type,
    odom_msg_type,
    qos_profile,
    input_fn=input,
    time_fn=time.time,
    sleep_fn=time.sleep,
    analyze_fn=analyze_scan_samples,
    temporary_map_callback=None,
    active_explore_plan_callback=None,
):
    session = ArenaActiveSpinSession(
        node,
        config,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input_fn,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        analyze_fn=analyze_fn,
        temporary_map_callback=temporary_map_callback,
        active_explore_plan_callback=active_explore_plan_callback,
    )
    return session.run(publisher)
