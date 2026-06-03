from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Callable

from arena_geometry_localization import ScanPointCache, analyze_scan_samples
from arena_active_explore import (
    ActiveExplorePlan,
    geometry_is_recoverable,
    plan_active_explore_recovery,
)

from .active_explore_recovery import ActiveExploreRecoveryRunner
from .active_explore_policy import ActiveExplorePolicy
from .diagnostics import (
    active_explore_config_from_arena_config,
    effective_recovery_mode,
    initial_diagnostics,
    spin_diagnostics_template,
    update_safety_minima,
    write_diagnostics_json,
)
from .explore_mission import (
    EXPLORE_ACTION_RUN_LOCALIZATION_SPIN,
    ExploreMissionController,
)
from .localizer_filter import ActiveExploreLocalizerFilter
from .math_utils import shortest_angle_delta_rad
from .models import (
    ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE,
    ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN,
    DEFAULT_STOP_COUNT,
    DEFAULT_STOP_HZ,
    ArenaActiveSpinConfig,
    ArenaActiveSpinResult,
    CenterRepositionAction,
    CenterRepositionStep,
)
from .motion_executor import ActiveSpinMotionExecutor
from .reposition import choose_center_reposition_action, choose_heater_approach_reposition_action
from .scan_safety import (
    evaluate_clearance,
    min_valid_scan_range,
    odom_pose_from_msg,
    pose_prior_from_localizer_result,
    scan_sample_from_msg,
)
from .temporary_map_manager import TemporaryMapManager


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
        self.latest_scan = None
        self.latest_scan_received_sec = None
        self.latest_odom_pose = None
        self.latest_odom_yaw_rad = None
        self.latest_odom_received_sec = None
        self.collecting = False
        self.collecting_explore_map = False
        self.samples = []
        self.active_explore_startup_spin_samples = ()
        self.active_explore_final_spin_memory_samples = None
        self.rejected_samples = 0
        self.diagnostics = initial_diagnostics(config)
        self._temporary_map_manager = TemporaryMapManager(self)
        self._localizer_filter = ActiveExploreLocalizerFilter(self)
        self.localizer_point_cache = ScanPointCache()
        self._motion_executor = ActiveSpinMotionExecutor(self)
        self._active_explore_recovery = ActiveExploreRecoveryRunner(self)
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

    @property
    def explore_samples(self):
        return self._temporary_map_manager.samples

    @explore_samples.setter
    def explore_samples(self, value):
        self._temporary_map_manager.set_samples(value)

    @property
    def last_temporary_map_publish_sec(self):
        return self._temporary_map_manager.last_publish_sec

    @last_temporary_map_publish_sec.setter
    def last_temporary_map_publish_sec(self, value):
        self._temporary_map_manager.last_publish_sec = value

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
        self._temporary_map_manager.append_sample(sample)
        self.publish_temporary_map_if_ready()

    def update_temporary_map_diagnostics(self, planning_grid, display_grid=None):
        self._temporary_map_manager.update_diagnostics(
            planning_grid,
            display_grid=display_grid,
        )

    def publish_temporary_map_if_ready(self, force=False, grid=None, display_grid=None):
        self._temporary_map_manager.publish_if_ready(
            force=force,
            grid=grid,
            display_grid=display_grid,
        )

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
        if not self.config.verbose:
            print(f"Arena-active: post-motion spin skipped ({spin_safety['reason']})")
            return
        print("\nArena-active post-motion spin skipped")
        print(f"  reason: {spin_safety['reason']}")
        print(f"  full min range: {spin_safety['full_min_range_m']}")
        print(f"  required min range: {spin_safety['required_min_range_m']}")
        print("  expected action: replan toward active-explore frontier without rotating")

    def print_active_explore_phase_spin_skip(self, reason):
        if not self.config.verbose:
            print(f"Arena-active: post-motion spin skipped ({reason})")
            return
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
        if not self.config.verbose:
            print(
                "Arena-active: rotating 360 deg "
                f"(front clearance={clearance.front_min_m})"
            )
            if not clearance.ok:
                raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")
            if self.config.require_operator_confirmation:
                self.input_fn(
                    "Press Enter to start arena-active spin, or Ctrl+C to abort: "
                )
            return
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
        return self._motion_executor.run_spin(publisher)

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
        return self._localizer_filter.reason_disabled()

    def active_explore_localizer_filter_grid(self, grid_samples=None):
        samples = self.explore_samples if grid_samples is None else grid_samples
        if not samples:
            return None, "no_temporary_map_samples"
        if self.latest_odom_pose is None:
            return None, "missing_latest_odom_pose"
        grid = self._temporary_map_manager.planning_grid(
            samples,
            robot_pose=self.latest_odom_pose,
            active_config=active_explore_config_from_arena_config(self.config),
        )
        self.update_temporary_map_diagnostics(grid)
        return grid, "ok"

    def active_explore_localizer_memory_samples(self):
        return self._localizer_filter.memory_samples()

    def dedupe_samples_by_identity(self, sample_groups):
        return self._localizer_filter.dedupe_samples_by_identity(sample_groups)

    def stride_valid_range_count_for_sample(self, sample):
        return self._localizer_filter.stride_valid_range_count_for_sample(sample)

    def select_samples_for_point_budget(self, samples, point_budget):
        return self._localizer_filter.select_samples_for_point_budget(
            samples,
            point_budget,
        )

    def pose_bin_for_mapping_sample(self, sample):
        return self._localizer_filter.pose_bin_for_mapping_sample(sample)

    def active_explore_mapping_memory_candidates(self, memory_samples, excluded_ids):
        return self._localizer_filter.mapping_memory_candidates(
            memory_samples,
            excluded_ids,
        )

    def active_explore_localizer_point_budgets(self, final_samples, startup_samples):
        return self._localizer_filter.point_budgets(final_samples, startup_samples)

    def balanced_active_explore_localizer_samples(self, memory_samples):
        return self._localizer_filter.balanced_samples(memory_samples)

    def active_explore_filtered_localizer_samples(self):
        return self._localizer_filter.filtered_samples()

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
            point_cache=self.localizer_point_cache,
            sample_point_limits=self._localizer_filter.last_sample_point_limits,
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
        return self._motion_executor.turn_to_heading(publisher, target_yaw_rad)

    def publish_drive_command(self, publisher):
        command = self.twist_factory()
        command.linear.x = abs(self.config.center_reposition_linear_speed_mps)
        publisher.publish(command)

    def drive_forward(self, publisher, distance_m):
        return self._motion_executor.drive_forward(publisher, distance_m)

    def print_reposition_prompt(self, action: CenterRepositionAction):
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
        if not self.config.verbose:
            total_distance = sum(step.planned_distance_m for step in steps)
            print(
                "Arena-active reposition: "
                f"{len(steps)} step(s), total planned distance={total_distance:.3f} m"
            )
            if self.config.require_operator_confirmation:
                self.input_fn("Press Enter to start center reposition, or Ctrl+C to abort: ")
            return
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
        return self._motion_executor.execute_center_reposition(publisher, action)

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
            grid = self._temporary_map_manager.planning_grid(
                active_config=active_config,
            )
            display_grid = self._temporary_map_manager.display_grid(
                active_config=active_config,
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
        if not self.config.verbose:
            path_length = (
                f"{candidate.path_length_m:.3f} m"
                if candidate.path_length_m is not None
                else "unknown"
            )
            print(
                "Arena-active explore: "
                f"candidate={candidate.kind}, path={path_length}, "
                f"curve_points={len(path_points)}"
            )
            if self.config.require_operator_confirmation:
                self.input_fn("Press Enter to start active-explore recovery, or Ctrl+C to abort: ")
            return
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
        return self._motion_executor.execute_active_explore_cmd_vel(
            publisher,
            candidate,
            distance_limit_m=distance_limit_m,
        )

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
        return self._active_explore_recovery.run(publisher, result)

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
