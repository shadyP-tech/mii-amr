"""ROS-free stopped-pose geometry admission for camera candidates.

The camera handoff may use this module after navigation has stopped and a
fresh robot pose is available.  It checks only whether the target is inside a
configured range envelope and close enough to the robot's optical-axis
heading.  It neither proves stationarity nor authorizes corrective motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scripts.aufgabe04.navigation.foundation.models import Pose2D


CANDIDATE_ARRIVAL_ADMISSION_SCHEMA_VERSION = 1
DEFAULT_MAX_BEARING_ERROR_RAD = math.radians(3.0)

ARRIVAL_GEOMETRY_ADMITTED = "arrival_geometry_admitted"
ARRIVAL_GEOMETRY_REJECTED = "arrival_geometry_rejected"

REASON_RANGE_BELOW_MINIMUM = "range_below_minimum"
REASON_RANGE_ABOVE_MAXIMUM = "range_above_maximum"
REASON_BEARING_ERROR_ABOVE_MAXIMUM = "bearing_error_above_maximum"

ERROR_INVALID_CONFIGURATION = "invalid_arrival_admission_configuration"
ERROR_INVALID_ROBOT_POSE = "invalid_arrival_robot_pose"
ERROR_INVALID_TARGET = "invalid_arrival_target"


class CandidateArrivalAdmissionError(ValueError):
    """Stable caller-error type for malformed admission inputs."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class CandidateArrivalAdmissionConfig:
    """Inclusive range and bearing limits for one stopped camera check."""

    min_range_m: float
    max_range_m: float
    max_bearing_error_rad: float = DEFAULT_MAX_BEARING_ERROR_RAD

    def __post_init__(self) -> None:
        minimum = _finite_numeric(
            self.min_range_m,
            "min_range_m",
            code=ERROR_INVALID_CONFIGURATION,
        )
        maximum = _finite_numeric(
            self.max_range_m,
            "max_range_m",
            code=ERROR_INVALID_CONFIGURATION,
        )
        bearing_limit = _finite_numeric(
            self.max_bearing_error_rad,
            "max_bearing_error_rad",
            code=ERROR_INVALID_CONFIGURATION,
        )
        if minimum < 0.0:
            raise CandidateArrivalAdmissionError(
                ERROR_INVALID_CONFIGURATION,
                "min_range_m must be non-negative",
            )
        if maximum < minimum:
            raise CandidateArrivalAdmissionError(
                ERROR_INVALID_CONFIGURATION,
                "max_range_m must be greater than or equal to min_range_m",
            )
        if not 0.0 <= bearing_limit <= math.pi:
            raise CandidateArrivalAdmissionError(
                ERROR_INVALID_CONFIGURATION,
                "max_bearing_error_rad must be between zero and pi",
            )
        object.__setattr__(self, "min_range_m", minimum)
        object.__setattr__(self, "max_range_m", maximum)
        object.__setattr__(self, "max_bearing_error_rad", bearing_limit)

    def to_evidence_dict(self) -> dict[str, float]:
        return {
            "min_range_m": self.min_range_m,
            "max_range_m": self.max_range_m,
            "max_bearing_error_rad": self.max_bearing_error_rad,
            "max_bearing_error_deg": math.degrees(
                self.max_bearing_error_rad
            ),
        }


@dataclass(frozen=True)
class CandidateArrivalAdmissionDecision:
    """Complete deterministic evidence for one stopped-pose geometry check."""

    accepted: bool
    decision: str
    reasons: tuple[str, ...]
    robot_pose: Pose2D
    target_x_m: float
    target_y_m: float
    target_bearing_rad: float
    range_m: float
    signed_bearing_error_rad: float
    absolute_bearing_error_rad: float
    range_above_minimum: bool
    range_below_maximum: bool
    bearing_within_limit: bool
    config: CandidateArrivalAdmissionConfig
    schema_version: int = field(
        default=CANDIDATE_ARRIVAL_ADMISSION_SCHEMA_VERSION,
        init=False,
    )
    motion_authorized: bool = field(default=False, init=False)

    @property
    def fail_closed(self) -> bool:
        return not self.accepted

    def to_evidence_dict(self) -> dict[str, object]:
        """Return stable JSON-ready evidence without motion authority."""

        return {
            "schema_version": self.schema_version,
            "admission_kind": "candidate_stopped_arrival_geometry",
            "accepted": self.accepted,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "fail_closed": self.fail_closed,
            "motion_authorized": self.motion_authorized,
            "scope": {
                "requires_stopped_pose": True,
                "proves_stationarity": False,
                "authorizes_corrective_motion": False,
            },
            "robot_pose": {
                "x_m": self.robot_pose.x_m,
                "y_m": self.robot_pose.y_m,
                "yaw_rad": self.robot_pose.yaw_rad,
            },
            "target": {
                "x_m": self.target_x_m,
                "y_m": self.target_y_m,
                "bearing_rad": self.target_bearing_rad,
            },
            "measurements": {
                "range_m": self.range_m,
                "signed_bearing_error_rad": self.signed_bearing_error_rad,
                "absolute_bearing_error_rad": (
                    self.absolute_bearing_error_rad
                ),
            },
            "thresholds": self.config.to_evidence_dict(),
            "checks": {
                "range_above_minimum": self.range_above_minimum,
                "range_below_maximum": self.range_below_maximum,
                "bearing_within_limit": self.bearing_within_limit,
            },
            "threshold_semantics": "inclusive",
        }


def evaluate_candidate_arrival_admission(
    robot_pose: Pose2D,
    *,
    target_x_m: float,
    target_y_m: float,
    config: CandidateArrivalAdmissionConfig,
) -> CandidateArrivalAdmissionDecision:
    """Evaluate target range and bearing from a finite stopped robot pose.

    The returned decision is evidence only.  Even an accepted decision has
    ``motion_authorized=False`` and cannot be used as velocity authority.
    Malformed inputs raise :class:`CandidateArrivalAdmissionError`; ordinary
    threshold misses return a rejected decision with stable reason codes.
    """

    if not isinstance(config, CandidateArrivalAdmissionConfig):
        raise CandidateArrivalAdmissionError(
            ERROR_INVALID_CONFIGURATION,
            "config must be CandidateArrivalAdmissionConfig",
        )
    if not isinstance(robot_pose, Pose2D):
        raise CandidateArrivalAdmissionError(
            ERROR_INVALID_ROBOT_POSE,
            "robot_pose must be Pose2D",
        )
    robot_x_m = _finite_numeric(
        robot_pose.x_m,
        "robot_pose.x_m",
        code=ERROR_INVALID_ROBOT_POSE,
    )
    robot_y_m = _finite_numeric(
        robot_pose.y_m,
        "robot_pose.y_m",
        code=ERROR_INVALID_ROBOT_POSE,
    )
    robot_yaw_rad = _finite_numeric(
        robot_pose.yaw_rad,
        "robot_pose.yaw_rad",
        code=ERROR_INVALID_ROBOT_POSE,
    )
    target_x = _finite_numeric(
        target_x_m,
        "target_x_m",
        code=ERROR_INVALID_TARGET,
    )
    target_y = _finite_numeric(
        target_y_m,
        "target_y_m",
        code=ERROR_INVALID_TARGET,
    )

    validated_pose = Pose2D(robot_x_m, robot_y_m, robot_yaw_rad)
    delta_x_m = target_x - robot_x_m
    delta_y_m = target_y - robot_y_m
    range_m = math.hypot(delta_x_m, delta_y_m)
    target_bearing_rad = math.atan2(delta_y_m, delta_x_m)
    signed_bearing_error_rad = _normalize_angle(
        target_bearing_rad - robot_yaw_rad
    )
    absolute_bearing_error_rad = abs(signed_bearing_error_rad)

    range_above_minimum = range_m >= config.min_range_m
    range_below_maximum = range_m <= config.max_range_m
    bearing_within_limit = (
        absolute_bearing_error_rad <= config.max_bearing_error_rad
    )

    reasons: list[str] = []
    if not range_above_minimum:
        reasons.append(REASON_RANGE_BELOW_MINIMUM)
    if not range_below_maximum:
        reasons.append(REASON_RANGE_ABOVE_MAXIMUM)
    if not bearing_within_limit:
        reasons.append(REASON_BEARING_ERROR_ABOVE_MAXIMUM)
    accepted = not reasons

    return CandidateArrivalAdmissionDecision(
        accepted=accepted,
        decision=(
            ARRIVAL_GEOMETRY_ADMITTED
            if accepted
            else ARRIVAL_GEOMETRY_REJECTED
        ),
        reasons=tuple(reasons),
        robot_pose=validated_pose,
        target_x_m=target_x,
        target_y_m=target_y,
        target_bearing_rad=target_bearing_rad,
        range_m=range_m,
        signed_bearing_error_rad=signed_bearing_error_rad,
        absolute_bearing_error_rad=absolute_bearing_error_rad,
        range_above_minimum=range_above_minimum,
        range_below_maximum=range_below_maximum,
        bearing_within_limit=bearing_within_limit,
        config=config,
    )


def _finite_numeric(value: object, name: str, *, code: str) -> float:
    if isinstance(value, bool):
        raise CandidateArrivalAdmissionError(code, f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateArrivalAdmissionError(
            code,
            f"{name} must be numeric",
        ) from exc
    if not math.isfinite(result):
        raise CandidateArrivalAdmissionError(code, f"{name} must be finite")
    return result


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
