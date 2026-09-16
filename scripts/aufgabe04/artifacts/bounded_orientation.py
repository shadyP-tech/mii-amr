"""Motion-neutral interval contract for current-pixel head orientation.

An interval preserves competing 3-D fits; its midpoint is only a route target,
not a claim that the physical angle is known exactly. A single collision-checked
route may use it only when its actual endpoint works for the entire interval.
Stand keepouts are circular, so their route clearance is rotation-invariant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math


BOUNDED_ORIENTATION_POLICY = "current_head_noise_expanded_interval"
MAXIMUM_ORIENTATION_HALF_WIDTH_RAD = math.radians(15.0)
MAXIMUM_QR_VIEW_OBLIQUITY_RAD = math.radians(20.0)
TERMINAL_POSITION_RESERVE_M = 0.03
_KEYS = {"policy", "center_rad", "half_width_rad", "sample_count"}


class BoundedOrientationViewUnavailableError(ValueError):
    """Valid bounded evidence needs a different view, not relaxed gates."""


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"bounded orientation {name} must be finite")
    return float(value)


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _axis(value):
    return (value + math.pi / 2.0) % math.pi - math.pi / 2.0


@dataclass(frozen=True)
class BoundedOrientation:
    policy: str
    center_rad: float
    half_width_rad: float
    sample_count: int

    def payload(self):
        return asdict(self)

    def rotated(self, yaw_delta_rad):
        return BoundedOrientation(self.policy, _axis(self.center_rad + _number(yaw_delta_rad, "rotation")),
                                  self.half_width_rad, self.sample_count)


def validated_bounded_orientation(payload, *, expected_axis_rad=None, expected_sample_count=None):
    if not isinstance(payload, Mapping) or set(payload) != _KEYS:
        raise ValueError("bounded orientation has unexpected fields")
    if payload["policy"] != BOUNDED_ORIENTATION_POLICY:
        raise ValueError("bounded orientation policy is unsupported")
    center = _number(payload["center_rad"], "center_rad")
    half_width = _number(payload["half_width_rad"], "half_width_rad")
    if not 0.0 <= half_width <= MAXIMUM_ORIENTATION_HALF_WIDTH_RAD:
        raise ValueError("bounded orientation half width exceeds 15 degrees")
    count = payload["sample_count"]
    if type(count) is not int or not 7 <= count <= 32:
        raise ValueError("bounded orientation requires 7 to 32 current samples")
    if expected_axis_rad is not None and abs(_axis(center - _number(expected_axis_rad, "expected axis"))) > 1e-9:
        raise ValueError("bounded orientation center differs from selected axis")
    if expected_sample_count is not None and count != expected_sample_count:
        raise ValueError("bounded orientation sample count differs from receipt")
    return BoundedOrientation(BOUNDED_ORIENTATION_POLICY, _axis(center), half_width, count)


def validate_opposite_orientation(bounded, *, selected_normal_rad, robot_side_rad, robot_side_uncertainty_rad=0.0):
    """Every plausible face must remain at least 120 degrees from the robot."""
    normal = _number(selected_normal_rad, "selected normal")
    robot_side = _number(robot_side_rad, "observing robot side")
    reserve = _number(robot_side_uncertainty_rad, "observing side uncertainty")
    if not 0 <= reserve < math.pi / 2:
        raise BoundedOrientationViewUnavailableError("bounded orientation observing side is unresolved")
    if abs(_angle(normal - robot_side - math.pi)) + bounded.half_width_rad + reserve > math.pi / 3.0 + 1e-12:
        raise BoundedOrientationViewUnavailableError("bounded orientation does not resolve the opposite face for every angle")


def validate_bounded_endpoint(payload, *, selected_normal_rad, stand_x_m, stand_y_m, stand_uncertainty_m,
                              target_x_m, target_y_m, expected_sample_count=None,
                              observing_robot_x_m=None, observing_robot_y_m=None):
    """Analytically bound incidence for one endpoint, including arrival error.

    This certifies viewing geometry only. The caller must also validate the
    same route against its ordinary circular stand and static obstacle limits.
    """
    normal = _number(selected_normal_rad, "selected normal")
    bounded = validated_bounded_orientation(payload, expected_axis_rad=normal - math.pi / 2.0,
                                            expected_sample_count=expected_sample_count)
    dx = _number(target_x_m, "target x") - _number(stand_x_m, "stand x")
    dy = _number(target_y_m, "target y") - _number(stand_y_m, "stand y")
    distance = math.hypot(dx, dy)
    stand_uncertainty = _number(stand_uncertainty_m, "stand uncertainty")
    if stand_uncertainty < 0:
        raise ValueError("bounded orientation stand uncertainty must be nonnegative")
    total_reserve = TERMINAL_POSITION_RESERVE_M + stand_uncertainty
    if distance <= total_reserve:
        raise BoundedOrientationViewUnavailableError("bounded orientation endpoint has insufficient stand range")
    radial_error = abs(_angle(math.atan2(dy, dx) - normal))
    position_reserve = math.asin(total_reserve / distance)
    worst = radial_error + bounded.half_width_rad + position_reserve
    if worst > MAXIMUM_QR_VIEW_OBLIQUITY_RAD + 1e-12:
        raise BoundedOrientationViewUnavailableError("bounded orientation endpoint exceeds QR viewing obliquity for plausible angles")
    if (observing_robot_x_m is None) != (observing_robot_y_m is None):
        raise ValueError("bounded orientation observing robot requires both coordinates")
    if observing_robot_x_m is not None:
        source_dx = _number(observing_robot_x_m, "observing robot x") - stand_x_m
        source_dy = _number(observing_robot_y_m, "observing robot y") - stand_y_m
        source_distance = math.hypot(source_dx, source_dy)
        if source_distance <= stand_uncertainty:
            raise BoundedOrientationViewUnavailableError("bounded orientation observing side is unresolved")
        validate_opposite_orientation(
            bounded, selected_normal_rad=normal, robot_side_rad=math.atan2(source_dy, source_dx),
            robot_side_uncertainty_rad=math.asin(stand_uncertainty / source_distance),
        )
    return {"policy": BOUNDED_ORIENTATION_POLICY, "bounded_orientation": bounded.payload(),
            "all_plausible_angles_supported": True, "actual_endpoint_checked": True,
            "worst_case_view_obliquity_rad": worst,
            "maximum_view_obliquity_rad": MAXIMUM_QR_VIEW_OBLIQUITY_RAD,
            "terminal_position_reserve_m": TERMINAL_POSITION_RESERVE_M,
            "stand_center_uncertainty_m": stand_uncertainty,
            "total_position_reserve_m": total_reserve,
            "opposite_side_with_center_uncertainty_checked": observing_robot_x_m is not None,
            "clearance_basis": "same_route_rotation_invariant_circular_stand_keepout",
            "motion_authorized": False}


def endpoint_evidence_matches(recorded, recomputed):
    """CSV serialization may round coordinates, never the viewing policy."""
    if recomputed is None:
        return recorded is None
    if not isinstance(recorded, Mapping) or set(recorded) != set(recomputed):
        return False
    return all(
        (isinstance(recorded[key], (int, float)) and not isinstance(recorded[key], bool)
         and math.isfinite(recorded[key])
         and math.isclose(recorded[key], value, rel_tol=0.0, abs_tol=1e-7))
        if key == "worst_case_view_obliquity_rad" else recorded[key] == value
        for key, value in recomputed.items()
    )
