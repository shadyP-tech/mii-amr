"""Pure validation helpers for final runtime velocity commands."""

from __future__ import annotations

import math


def finite_velocity_command(
    linear_x_mps: object,
    angular_z_radps: object,
) -> bool:
    try:
        values = (float(linear_x_mps), float(angular_z_radps))
    except (TypeError, ValueError, OverflowError):
        return False
    return all(math.isfinite(value) for value in values)
