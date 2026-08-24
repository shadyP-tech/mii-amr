"""Configuration contract for the Aufgabe 04 waypoint follower."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from scripts.aufgabe04.navigation.control.driving_behavior import CommandSmoothingConfig
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    DEFAULT_LINEAR_MOTION_FLOOR_MPS,
    PersistentObstacleConfig,
)
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    CertifiedCornerControlConfig,
    ControllerConfig,
    StartEgressControlConfig,
)


@dataclass(frozen=True)
class FollowerConfig:
    controller: ControllerConfig
    min_obstacle_distance_m: float = 0.20
    omnidirectional_hard_stop_distance_m: float = 0.12
    front_obstacle_slow_distance_m: float = 0.38
    front_obstacle_sector_rad: float = math.radians(35.0)
    max_scan_age_sec: float = 1.0
    max_odom_age_sec: float = 1.0
    max_tf_age_sec: float = 1.0
    max_future_timestamp_sec: float = 0.25
    runtime_nomotion_update_service: str = "request_nomotion_update"
    runtime_nomotion_update_timeout_sec: float = 2.0
    amcl_edge_future_tolerance_sec: float = 1.1
    allow_simulation_odom_after_stale_tf: bool = False
    initial_sensor_wait_sec: float = 2.0
    waypoint_timeout_sec: float = 45.0
    stuck_timeout_sec: float = 8.0
    stuck_progress_epsilon_m: float = 0.03
    stuck_heading_progress_epsilon_rad: float = 0.10
    linear_motion_floor_mps: float = DEFAULT_LINEAR_MOTION_FLOOR_MPS
    blockage_confirmation_timeout_sec: float = 1.2
    persistent_obstacle_config: PersistentObstacleConfig | None = None
    initial_distance_limit_m: float = 0.35
    control_rate_hz: float = 10.0
    allowed_cmd_vel_publishers: Sequence[str] = ()
    dynamic_route_refresh_sec: float = 0.0
    dynamic_join_tolerance_m: float = 0.01
    start_egress_waypoint_tolerance_m: float = 0.02
    start_egress_alignment_tolerance_rad: float = 0.10
    start_egress_max_linear_mps: float = 0.03
    initial_start_egress_waypoint_index: int | None = None
    initial_start_join_clearance_m: float | None = None
    initial_route_kind: str = ""
    axis_acquisition_wait_timeout_sec: float = 12.0
    viewpoint_sampling_timeout_sec: float = 30.0
    viewpoint_sampling_target_timeout_sec: float = 30.0
    viewpoint_sampling_goal_tolerance_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M
    )
    viewpoint_sampling_terminal_heading_hold_tolerance_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
    )
    viewpoint_sampling_target_distance_m: float = (
        DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M
    )
    viewpoint_sampling_terminal_heading_target_envelope_radius_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    )
    viewpoint_sampling_heading_tolerance_rad: float = math.radians(5.0)
    physical_waypoint_tolerance_m: float = 0.02
    physical_goal_tolerance_m: float = 0.03
    certified_route_tube_radius_m: float = 0.03
    certified_route_chord_sample_spacing_m: float = 0.01
    certified_corner_turn_threshold_rad: float = 0.20
    certified_corner_release_tolerance_m: float = 0.01
    certified_corner_hold_tolerance_m: float = 0.025
    certified_corner_alignment_tolerance_rad: float = 0.10
    certified_corner_max_reacquire_attempts: int = 2
    command_smoothing: CommandSmoothingConfig = field(
        default_factory=CommandSmoothingConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.command_smoothing, CommandSmoothingConfig):
            raise ValueError(
                "command_smoothing must be a CommandSmoothingConfig"
            )
        if (
            not math.isfinite(self.linear_motion_floor_mps)
            or self.linear_motion_floor_mps <= 0.0
        ):
            raise ValueError("linear_motion_floor_mps must be finite and positive")
        if (
            not math.isfinite(self.control_rate_hz)
            or self.control_rate_hz < 1.0
        ):
            raise ValueError("control_rate_hz must be finite and at least 1 Hz")
        if (
            self.command_smoothing.enabled
            and self.command_smoothing.max_linear_accel_mps2
            / self.control_rate_hz
            + 1.0e-12
            < self.linear_motion_floor_mps
        ):
            raise ValueError(
                "linear command smoothing must reach the motion floor within "
                "one control period"
            )
        if (
            not math.isfinite(self.blockage_confirmation_timeout_sec)
            or self.blockage_confirmation_timeout_sec <= 0.0
        ):
            raise ValueError(
                "blockage_confirmation_timeout_sec must be finite and positive"
            )
        obstacle_config = self.persistent_obstacle_config
        if obstacle_config is None:
            obstacle_config = PersistentObstacleConfig(
                min_front_range_m=self.omnidirectional_hard_stop_distance_m,
                max_front_range_m=self.front_obstacle_slow_distance_m,
                front_sector_half_width_rad=self.front_obstacle_sector_rad,
            )
            object.__setattr__(
                self,
                "persistent_obstacle_config",
                obstacle_config,
            )
        elif not isinstance(obstacle_config, PersistentObstacleConfig):
            raise ValueError(
                "persistent_obstacle_config must be a PersistentObstacleConfig"
            )
        if (
            obstacle_config.min_front_range_m
            < self.omnidirectional_hard_stop_distance_m
            or obstacle_config.max_front_range_m
            > self.front_obstacle_slow_distance_m
            or obstacle_config.front_sector_half_width_rad
            > self.front_obstacle_sector_rad
        ):
            raise ValueError(
                "persistent obstacle bounds must remain inside the follower's "
                "hard-stop, slow-distance, and front-sector bounds"
            )
        if (
            self.blockage_confirmation_timeout_sec
            < obstacle_config.max_sample_window_sec
        ):
            raise ValueError(
                "blockage confirmation timeout must cover the sample window"
            )
        if (
            not math.isfinite(self.omnidirectional_hard_stop_distance_m)
            or self.omnidirectional_hard_stop_distance_m <= 0.0
            or self.omnidirectional_hard_stop_distance_m
            >= self.min_obstacle_distance_m
        ):
            raise ValueError(
                "omnidirectional_hard_stop_distance_m must be finite, positive, "
                "and smaller than min_obstacle_distance_m"
            )
        if not isinstance(
            self.allow_simulation_odom_after_stale_tf,
            bool,
        ):
            raise ValueError(
                "allow_simulation_odom_after_stale_tf must be boolean"
            )
        if (
            not math.isfinite(self.dynamic_join_tolerance_m)
            or self.dynamic_join_tolerance_m <= 0.0
        ):
            raise ValueError("dynamic_join_tolerance_m must be finite and positive")
        if (
            not math.isfinite(self.start_egress_waypoint_tolerance_m)
            or self.start_egress_waypoint_tolerance_m <= 0.0
        ):
            raise ValueError(
                "start_egress_waypoint_tolerance_m must be finite and positive"
            )
        # Validate the paired egress controls through their pure value object
        # so the ROS wrapper and offline controller tests share one contract.
        StartEgressControlConfig(
            alignment_tolerance_rad=self.start_egress_alignment_tolerance_rad,
            max_linear_mps=self.start_egress_max_linear_mps,
        )
        if self.initial_start_egress_waypoint_index is not None and (
            not isinstance(self.initial_start_egress_waypoint_index, int)
            or isinstance(self.initial_start_egress_waypoint_index, bool)
            or self.initial_start_egress_waypoint_index <= 0
        ):
            raise ValueError(
                "initial_start_egress_waypoint_index must be a positive integer"
            )
        if self.initial_start_join_clearance_m is not None and (
            not math.isfinite(self.initial_start_join_clearance_m)
            or self.initial_start_join_clearance_m <= 0.0
        ):
            raise ValueError(
                "initial_start_join_clearance_m must be finite and positive"
            )
        if (self.initial_start_egress_waypoint_index is None) != (
            self.initial_start_join_clearance_m is None
        ):
            raise ValueError(
                "initial start-egress lock and start-join clearance must be paired"
            )
        if (
            not math.isfinite(self.stuck_heading_progress_epsilon_rad)
            or self.stuck_heading_progress_epsilon_rad <= 0.0
        ):
            raise ValueError(
                "stuck_heading_progress_epsilon_rad must be finite and positive"
            )
        if (
            not math.isfinite(self.axis_acquisition_wait_timeout_sec)
            or self.axis_acquisition_wait_timeout_sec <= 0.0
        ):
            raise ValueError(
                "axis_acquisition_wait_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_timeout_sec)
            or self.viewpoint_sampling_timeout_sec <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_target_timeout_sec)
            or self.viewpoint_sampling_target_timeout_sec <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_target_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_goal_tolerance_m)
            or self.viewpoint_sampling_goal_tolerance_m <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_goal_tolerance_m must be finite and positive"
            )
        if (
            not math.isfinite(
                self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            )
            or self.viewpoint_sampling_terminal_heading_hold_tolerance_m <= 0.0
            or self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            > INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
            or self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            < min(
                self.viewpoint_sampling_goal_tolerance_m,
                INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
            )
        ):
            raise ValueError(
                "viewpoint_sampling_terminal_heading_hold_tolerance_m must be "
                "finite, no smaller than the effective sampling tolerance, "
                "and no greater than 0.020"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_target_distance_m)
            or self.viewpoint_sampling_target_distance_m
            <= self.viewpoint_sampling_terminal_heading_hold_tolerance_m
        ):
            raise ValueError(
                "viewpoint_sampling_target_distance_m must be finite and greater "
                "than the terminal-heading radial hold tolerance"
            )
        if (
            not math.isfinite(
                self.viewpoint_sampling_terminal_heading_target_envelope_radius_m
            )
            or self.viewpoint_sampling_terminal_heading_target_envelope_radius_m
            < self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            or self.viewpoint_sampling_terminal_heading_target_envelope_radius_m
            > INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
        ):
            raise ValueError(
                "viewpoint_sampling_terminal_heading_target_envelope_radius_m "
                "must be finite, no smaller than the radial hold tolerance, "
                "and no greater than 0.030"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_heading_tolerance_rad)
            or self.viewpoint_sampling_heading_tolerance_rad <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_heading_tolerance_rad must be finite and positive"
            )
        if (
            not math.isfinite(self.physical_waypoint_tolerance_m)
            or self.physical_waypoint_tolerance_m <= 0.0
        ):
            raise ValueError(
                "physical_waypoint_tolerance_m must be finite and positive"
            )
        if (
            not math.isfinite(self.physical_goal_tolerance_m)
            or self.physical_goal_tolerance_m <= 0.0
        ):
            raise ValueError("physical_goal_tolerance_m must be finite and positive")
        if (
            not math.isfinite(self.max_future_timestamp_sec)
            or self.max_future_timestamp_sec < 0.0
        ):
            raise ValueError("max_future_timestamp_sec must be finite and non-negative")
        runtime_service = str(self.runtime_nomotion_update_service).strip()
        if not runtime_service:
            raise ValueError("runtime_nomotion_update_service must not be empty")
        object.__setattr__(
            self,
            "runtime_nomotion_update_service",
            runtime_service,
        )
        if (
            not math.isfinite(self.runtime_nomotion_update_timeout_sec)
            or self.runtime_nomotion_update_timeout_sec <= 0.0
            or self.runtime_nomotion_update_timeout_sec > 2.0
        ):
            raise ValueError(
                "runtime_nomotion_update_timeout_sec must be finite, positive, "
                "and no greater than 2.0"
            )
        if (
            not math.isfinite(self.amcl_edge_future_tolerance_sec)
            or self.amcl_edge_future_tolerance_sec < 0.0
        ):
            raise ValueError(
                "amcl_edge_future_tolerance_sec must be finite and non-negative"
            )
        if (
            not math.isfinite(self.certified_route_tube_radius_m)
            or self.certified_route_tube_radius_m <= 0.0
        ):
            raise ValueError(
                "certified_route_tube_radius_m must be finite and positive"
            )
        if (
            not math.isfinite(self.certified_route_chord_sample_spacing_m)
            or self.certified_route_chord_sample_spacing_m <= 0.0
        ):
            raise ValueError(
                "certified_route_chord_sample_spacing_m must be finite and positive"
            )
        corner_config = CertifiedCornerControlConfig(
            turn_threshold_rad=self.certified_corner_turn_threshold_rad,
            release_tolerance_m=self.certified_corner_release_tolerance_m,
            hold_tolerance_m=self.certified_corner_hold_tolerance_m,
            hard_tolerance_m=self.certified_route_tube_radius_m,
            alignment_tolerance_rad=(
                self.certified_corner_alignment_tolerance_rad
            ),
            max_reacquire_attempts=(
                self.certified_corner_max_reacquire_attempts
            ),
        )
        if (
            corner_config.release_tolerance_m
            > self.physical_waypoint_tolerance_m
        ):
            raise ValueError(
                "certified corner release tolerance must not exceed the "
                "physical waypoint tolerance"
            )
        if corner_config.hold_tolerance_m >= self.certified_route_tube_radius_m:
            raise ValueError(
                "certified corner hold tolerance must remain strictly inside "
                "the certified route tube"
            )
        if self.physical_goal_tolerance_m > self.certified_route_tube_radius_m:
            raise ValueError(
                "physical_goal_tolerance_m must not exceed the certified route tube"
            )
        if (
            self.physical_waypoint_tolerance_m
            > self.certified_route_tube_radius_m
        ):
            raise ValueError(
                "physical_waypoint_tolerance_m must not exceed "
                "the certified route tube"
            )
