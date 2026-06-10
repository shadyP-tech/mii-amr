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
    raw_linear_x: float
    scheduled_linear_x: float
    angular_z: float
    curvature_1pm: float
    alpha_deg: float
    rotate_fallback: bool


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
                max_angular_speed_radps=getattr(args, "max_angular_speed", 0.09),
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
            )
        )

    def reset(self):
        self.mode = "forward"

    def schedule(self, geometry):
        alpha_deg = math.degrees(geometry.alpha_rad)
        if self._should_rotate(alpha_deg):
            self.mode = "rotate"
            angular_z = clamp(
                math.radians(alpha_deg) * self.config.yaw_gain,
                -abs(self.config.max_angular_speed_radps),
                abs(self.config.max_angular_speed_radps),
            )
            return VelocityScheduleResult(
                command=TwistCommand(0.0, angular_z),
                mode="rotate",
                raw_linear_x=abs(self.config.linear_speed_mps),
                scheduled_linear_x=0.0,
                angular_z=angular_z,
                curvature_1pm=geometry.curvature_1pm,
                alpha_deg=alpha_deg,
                rotate_fallback=True,
            )

        self.mode = "forward"
        raw_linear = abs(self.config.linear_speed_mps)
        curvature = float(geometry.curvature_1pm)
        abs_curvature = abs(curvature)
        if abs_curvature <= 1e-9:
            scheduled_linear = raw_linear
        else:
            angular_limit = (
                abs(self.config.max_angular_speed_radps)
                * self.config.turn_speed_margin
                / abs_curvature
            )
            lateral_limit = math.sqrt(
                self.config.max_lateral_accel_mps2 / abs_curvature
            )
            scheduled_linear = min(raw_linear, angular_limit, lateral_limit)
            minimum = min(
                self.config.min_curvature_linear_speed_mps,
                angular_limit,
                lateral_limit,
                raw_linear,
            )
            if scheduled_linear > 0.0:
                scheduled_linear = max(scheduled_linear, minimum)

        angular_z = clamp(
            scheduled_linear * curvature,
            -abs(self.config.max_angular_speed_radps),
            abs(self.config.max_angular_speed_radps),
        )
        return VelocityScheduleResult(
            command=TwistCommand(scheduled_linear, angular_z),
            mode="forward",
            raw_linear_x=raw_linear,
            scheduled_linear_x=scheduled_linear,
            angular_z=angular_z,
            curvature_1pm=curvature,
            alpha_deg=alpha_deg,
            rotate_fallback=False,
        )

    def _should_rotate(self, alpha_deg):
        abs_error = abs(alpha_deg)
        if self.mode == "rotate":
            return abs_error > self.config.rotate_stop_heading_error_deg
        return abs_error > self.config.rotate_start_heading_error_deg
