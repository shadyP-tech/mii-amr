"""One bounded outward framing search while retaining a promising bearing."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.real_robot.observer.camera_framing import validate_camera_framing_hint


MINIMUM_DISTANCE_CHANGE_M = 0.10
MINIMUM_ACHIEVED_DISTANCE_CHANGE_M = 0.08
MAXIMUM_RECOVERY_BEARING_ERROR_RAD = math.radians(5.0)
MAX_DISTANCE_RECOVERY_PROPOSALS = 3


@dataclass(frozen=True)
class CameraDistanceRecovery:
    standoffs_m: tuple[float, ...]
    minimum_range_m: float
    maximum_range_m: float
    source_hint: dict

    def to_dict(self) -> dict:
        return {
            "standoffs_m": list(self.standoffs_m),
            "minimum_range_m": self.minimum_range_m,
            "maximum_range_m": self.maximum_range_m,
            "source_hint": dict(self.source_hint),
            "motion_authorized": False,
        }


def select_camera_distance_recovery(
    hint: object, *, candidate_uid: str, current_range_m: float,
    preferred_range_m: float, maximum_allowed_range_m: float,
) -> CameraDistanceRecovery | None:
    """Use explicit unresolved front geometry, never distance alone.

    The controller limits this search to once per candidate. Each proposal
    uses the shared route budget; the resulting observation uses a normal
    camera slot. A closer fallback is not offered if all outward poses fail.
    """
    evidence = validate_camera_framing_hint(hint)
    if evidence is None or evidence["target_key"] != candidate_uid:
        return None
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (current_range_m, preferred_range_m, maximum_allowed_range_m)):
        return None
    # The admitted stationary view must still describe the recorded range.
    if abs(current_range_m - evidence["range_m"]) > MINIMUM_DISTANCE_CHANGE_M:
        return None
    minimum = max(current_range_m + MINIMUM_DISTANCE_CHANGE_M, evidence["minimum_range_m"])
    maximum = min(preferred_range_m, maximum_allowed_range_m, evidence["maximum_range_m"])
    if minimum > maximum + 1.0e-9:
        return None
    offsets = [maximum]
    while len(offsets) < MAX_DISTANCE_RECOVERY_PROPOSALS and offsets[-1] - minimum > 1.0e-9:
        offsets.append(max(minimum, offsets[-1] - MINIMUM_DISTANCE_CHANGE_M))
    return CameraDistanceRecovery(tuple(offsets), evidence["minimum_range_m"],
                                  min(maximum_allowed_range_m, evidence["maximum_range_m"]), evidence)


def distance_recovery_goal_is_useful(
    *, start_range_m: float, goal_range_m: float, requested_normal_rad: float,
    achieved_normal_rad: float, minimum_range_m: float, maximum_range_m: float,
) -> bool:
    values = (start_range_m, goal_range_m, requested_normal_rad, achieved_normal_rad,
              minimum_range_m, maximum_range_m)
    return (all(type(v) in (int, float) and math.isfinite(v) for v in values)
            and goal_range_m >= start_range_m + MINIMUM_ACHIEVED_DISTANCE_CHANGE_M
            and minimum_range_m <= goal_range_m <= maximum_range_m
            and abs(math.remainder(achieved_normal_rad - requested_normal_rad, 2 * math.pi))
            <= MAXIMUM_RECOVERY_BEARING_ERROR_RAD)
