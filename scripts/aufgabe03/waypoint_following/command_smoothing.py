from __future__ import annotations

import math
from dataclasses import dataclass

from .math_utils import clamp
from .models import TwistCommand


COMMAND_SMOOTHING_OFF = "off"
COMMAND_SMOOTHING_RATE_LIMIT = "rate-limit"
COMMAND_SMOOTHING_MODES = (
    COMMAND_SMOOTHING_OFF,
    COMMAND_SMOOTHING_RATE_LIMIT,
)


@dataclass(frozen=True)
class CommandSmoothingConfig:
    max_linear_accel_mps2: float
    max_linear_decel_mps2: float
    max_angular_accel_radps2: float
    max_angular_decel_radps2: float
    final_decel_distance_m: float
    min_smoothed_linear_speed_mps: float


class CommandSmoother:
    def __init__(self, config: CommandSmoothingConfig):
        self.config = config
        self.previous_command: TwistCommand | None = None

    def reset(self):
        self.previous_command = None

    def apply(
        self,
        raw_command,
        dt_sec,
        distance_to_goal_m,
        goal_tolerance_m,
    ) -> TwistCommand:
        raw = TwistCommand(
            max(0.0, float(raw_command.linear_x)),
            float(raw_command.angular_z),
        )
        if raw.linear_x == 0.0 and raw.angular_z == 0.0:
            self.reset()
            return TwistCommand(0.0, 0.0)

        previous = self.previous_command or TwistCommand(0.0, 0.0)
        dt = 0.0 if dt_sec is None else float(dt_sec)
        if not math.isfinite(dt):
            dt = 0.0

        target_linear = self._target_linear(raw.linear_x, distance_to_goal_m, goal_tolerance_m)
        if dt <= 0.0:
            command = TwistCommand(
                min(max(0.0, previous.linear_x), target_linear),
                self._zero_dt_angular(previous.angular_z, raw.angular_z),
            )
            self.previous_command = command
            return command

        linear_x = self._rate_limit_linear(previous.linear_x, target_linear, dt)
        angular_z = self._rate_limit_angular(previous.angular_z, raw.angular_z, dt)
        command = TwistCommand(linear_x, angular_z)
        self.previous_command = command
        return command

    def _target_linear(self, raw_linear, distance_to_goal_m, goal_tolerance_m):
        target = max(0.0, float(raw_linear))
        distance = 0.0 if distance_to_goal_m is None else float(distance_to_goal_m)
        tolerance = 0.0 if goal_tolerance_m is None else float(goal_tolerance_m)
        if math.isfinite(distance) and self.config.final_decel_distance_m > 0.0:
            scale = clamp(
                max(0.0, distance) / self.config.final_decel_distance_m,
                0.0,
                1.0,
            )
            target *= scale
        minimum = self.config.min_smoothed_linear_speed_mps
        if (
            raw_linear > 0.0
            and distance > tolerance
            and target >= minimum
        ):
            target = max(target, minimum)
        return min(max(0.0, target), raw_linear)

    def _rate_limit_linear(self, previous, target, dt):
        previous = max(0.0, float(previous))
        target = max(0.0, float(target))
        if target >= previous:
            max_delta = self.config.max_linear_accel_mps2 * dt
        else:
            max_delta = self.config.max_linear_decel_mps2 * dt
        return max(0.0, previous + clamp(target - previous, -max_delta, max_delta))

    def _rate_limit_angular(self, previous, target, dt):
        previous = float(previous)
        target = float(target)
        if target == previous:
            return target
        if self._angular_decelerating(previous, target):
            max_delta = self.config.max_angular_decel_radps2 * dt
        else:
            max_delta = self.config.max_angular_accel_radps2 * dt
        return previous + clamp(target - previous, -max_delta, max_delta)

    @staticmethod
    def _angular_decelerating(previous, target):
        if previous == 0.0:
            return False
        if target == 0.0:
            return True
        if previous * target < 0.0:
            return True
        return abs(target) < abs(previous)

    @staticmethod
    def _zero_dt_angular(previous, target):
        previous = float(previous)
        target = float(target)
        if previous == 0.0:
            return 0.0
        if target == 0.0:
            return previous
        if previous * target < 0.0:
            return previous
        if abs(previous) > abs(target):
            return target
        return previous
