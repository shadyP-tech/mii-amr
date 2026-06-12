from __future__ import annotations

import math
from dataclasses import dataclass

from .math_utils import clamp, normalize_angle_rad
from .models import TwistCommand


SPEED_PROFILE_FIXED = "fixed"
SPEED_PROFILE_CURVATURE_AWARE = "curvature-aware"
SPEED_PROFILE_MODES = (
    SPEED_PROFILE_CURVATURE_AWARE,
    SPEED_PROFILE_FIXED,
)
SCHEDULER_STATUS_DEADBAND = "deadband"
SCHEDULER_STATUS_ANGULAR_RAMP = "angular_ramp"
SCHEDULER_STATUS_CURVATURE_BLEND = "curvature_blend"
SCHEDULER_STATUS_CURVATURE_LIMITED = "curvature_limited"
SCHEDULER_STATUS_ROTATE = "rotate"


@dataclass(frozen=True)
class PurePursuitGeometry:
    alpha_rad: float
    curvature_1pm: float
    lookahead_m: float
    target_point: tuple[float, float]


@dataclass(frozen=True)
class VelocityScheduleResult:
    command: TwistCommand
    mode: str
    status: str
    raw_linear_x: float
    scheduled_linear_x: float
    angular_z: float
    curvature_1pm: float
    alpha_deg: float
    rotate_fallback: bool
    lateral_error_m: float
    angular_scale: float
    speed_limit_blend: float
    raw_angular_z: float
    scheduled_angular_z: float


@dataclass(frozen=True)
class PurePursuitVelocityConfig:
    linear_speed_mps: float
    min_linear_speed_mps: float
    max_angular_speed_radps: float
    yaw_gain: float
    max_lateral_accel_mps2: float
    turn_speed_margin: float
    rotate_start_heading_error_deg: float
    rotate_stop_heading_error_deg: float
    min_curvature_linear_speed_mps: float
    heading_deadband_deg: float
    lateral_deadband_m: float
    curvature_limit_start_heading_error_deg: float
    curvature_limit_full_heading_error_deg: float


def pure_pursuit_geometry(current_pose, target_point, lookahead_m):
    dx = float(target_point[0]) - float(current_pose.x)
    dy = float(target_point[1]) - float(current_pose.y)
    target_heading = math.atan2(dy, dx)
    yaw = math.radians(float(current_pose.yaw_deg))
    alpha = normalize_angle_rad(target_heading - yaw)
    effective_lookahead = max(0.01, float(lookahead_m))
    curvature = 2.0 * math.sin(alpha) / effective_lookahead
    return PurePursuitGeometry(
        alpha_rad=alpha,
        curvature_1pm=curvature,
        lookahead_m=effective_lookahead,
        target_point=(float(target_point[0]), float(target_point[1])),
    )


def smoothstep(value):
    x = clamp(float(value), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class PurePursuitVelocityScheduler:
    def __init__(self, config: PurePursuitVelocityConfig):
        self.config = config
        self.mode = "forward"

    @classmethod
    def from_args(cls, args):
        return cls(
            PurePursuitVelocityConfig(
                linear_speed_mps=getattr(args, "linear_speed", 0.04),
                min_linear_speed_mps=getattr(args, "min_linear_speed", 0.012),
                max_angular_speed_radps=getattr(
                    args,
                    "pure_pursuit_max_track_angular_speed_radps",
                    getattr(args, "max_angular_speed", 0.09),
                ),
                yaw_gain=getattr(args, "yaw_gain", 0.35),
                max_lateral_accel_mps2=getattr(
                    args,
                    "pure_pursuit_max_lateral_accel_mps2",
                    0.04,
                ),
                turn_speed_margin=getattr(args, "pure_pursuit_turn_speed_margin", 0.85),
                rotate_start_heading_error_deg=(
                    getattr(
                        args,
                        "pure_pursuit_rotate_start_heading_error_deg",
                        75.0,
                    )
                ),
                rotate_stop_heading_error_deg=(
                    getattr(
                        args,
                        "pure_pursuit_rotate_stop_heading_error_deg",
                        35.0,
                    )
                ),
                min_curvature_linear_speed_mps=(
                    getattr(
                        args,
                        "pure_pursuit_min_curvature_linear_speed_mps",
                        getattr(args, "min_linear_speed", 0.012),
                    )
                ),
                heading_deadband_deg=getattr(
                    args,
                    "pure_pursuit_heading_deadband_deg",
                    4.0,
                ),
                lateral_deadband_m=getattr(
                    args,
                    "pure_pursuit_lateral_deadband_m",
                    0.03,
                ),
                curvature_limit_start_heading_error_deg=getattr(
                    args,
                    "pure_pursuit_curvature_limit_start_heading_error_deg",
                    12.0,
                ),
                curvature_limit_full_heading_error_deg=getattr(
                    args,
                    "pure_pursuit_curvature_limit_full_heading_error_deg",
                    30.0,
                ),
            )
        )

    def reset(self):
        self.mode = "forward"

    def schedule(self, geometry, allow_rotate=True, linear_speed_cap_mps=None):
        alpha_deg = math.degrees(geometry.alpha_rad)
        raw_linear = abs(
            self.config.linear_speed_mps
            if linear_speed_cap_mps is None
            else float(linear_speed_cap_mps)
        )
        curvature = float(geometry.curvature_1pm)
        lateral_error_m = geometry.lookahead_m * math.sin(geometry.alpha_rad)
        raw_angular_z = raw_linear * curvature
        if allow_rotate and self._should_rotate(alpha_deg):
            self.mode = "rotate"
            angular_z = clamp(
                math.radians(alpha_deg) * self.config.yaw_gain,
                -abs(self.config.max_angular_speed_radps),
                abs(self.config.max_angular_speed_radps),
            )
            return VelocityScheduleResult(
                command=TwistCommand(0.0, angular_z),
                mode="rotate",
                status=SCHEDULER_STATUS_ROTATE,
                raw_linear_x=raw_linear,
                scheduled_linear_x=0.0,
                angular_z=angular_z,
                curvature_1pm=curvature,
                alpha_deg=alpha_deg,
                rotate_fallback=True,
                lateral_error_m=lateral_error_m,
                angular_scale=0.0,
                speed_limit_blend=1.0,
                raw_angular_z=raw_angular_z,
                scheduled_angular_z=angular_z,
            )

        self.mode = "forward"
        abs_curvature = abs(curvature)
        abs_alpha_deg = abs(alpha_deg)
        deadbanded = (
            abs_alpha_deg <= self.config.heading_deadband_deg
            and abs(lateral_error_m) <= self.config.lateral_deadband_m
        )
        if deadbanded:
            return VelocityScheduleResult(
                command=TwistCommand(raw_linear, 0.0),
                mode="forward",
                status=SCHEDULER_STATUS_DEADBAND,
                raw_linear_x=raw_linear,
                scheduled_linear_x=raw_linear,
                angular_z=0.0,
                curvature_1pm=curvature,
                alpha_deg=alpha_deg,
                rotate_fallback=False,
                lateral_error_m=lateral_error_m,
                angular_scale=0.0,
                speed_limit_blend=0.0,
                raw_angular_z=raw_angular_z,
                scheduled_angular_z=0.0,
            )

        limit_start = self.config.curvature_limit_start_heading_error_deg
        limit_full = self.config.curvature_limit_full_heading_error_deg
        angular_scale = smoothstep(
            (abs_alpha_deg - self.config.heading_deadband_deg)
            / max(1e-9, limit_start - self.config.heading_deadband_deg)
        )
        speed_limit_blend = smoothstep(
            (abs_alpha_deg - limit_start)
            / max(1e-9, limit_full - limit_start)
        )

        feasible_linear = self._feasible_curvature_speed(raw_linear, abs_curvature)
        scheduled_linear = (
            raw_linear * (1.0 - speed_limit_blend)
            + feasible_linear * speed_limit_blend
        )

        angular_z = clamp(
            scheduled_linear * curvature * angular_scale,
            -abs(self.config.max_angular_speed_radps),
            abs(self.config.max_angular_speed_radps),
        )
        if abs_alpha_deg < limit_start:
            status = SCHEDULER_STATUS_ANGULAR_RAMP
        elif abs_alpha_deg < limit_full:
            status = SCHEDULER_STATUS_CURVATURE_BLEND
        else:
            status = SCHEDULER_STATUS_CURVATURE_LIMITED
        return VelocityScheduleResult(
            command=TwistCommand(scheduled_linear, angular_z),
            mode="forward",
            status=status,
            raw_linear_x=raw_linear,
            scheduled_linear_x=scheduled_linear,
            angular_z=angular_z,
            curvature_1pm=curvature,
            alpha_deg=alpha_deg,
            rotate_fallback=False,
            lateral_error_m=lateral_error_m,
            angular_scale=angular_scale,
            speed_limit_blend=speed_limit_blend,
            raw_angular_z=raw_angular_z,
            scheduled_angular_z=angular_z,
        )

    def _feasible_curvature_speed(self, raw_linear, abs_curvature):
        if abs_curvature <= 1e-9:
            return raw_linear
        angular_limit = (
            abs(self.config.max_angular_speed_radps)
            * self.config.turn_speed_margin
            / abs_curvature
        )
        lateral_limit = math.sqrt(self.config.max_lateral_accel_mps2 / abs_curvature)
        return min(raw_linear, angular_limit, lateral_limit)

    def _should_rotate(self, alpha_deg):
        abs_error = abs(alpha_deg)
        if self.mode == "rotate":
            return abs_error > self.config.rotate_stop_heading_error_deg
        return abs_error > self.config.rotate_start_heading_error_deg
