"""Shared pure arrival/hold contract for viewpoint-sampling targets.

The follower and the simulation observer consume the same geometry here.  The
contract deliberately distinguishes a strict arrival entry from the wider
terminal-heading hold envelope: entry proves that the commanded target was
actually reached, while the hold envelope tolerates small in-place-yaw drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from scripts.aufgabe04.navigation.foundation.models import Pose2D


VIEWPOINT_SAMPLING_CONTRACT_NAME = "viewpoint_sampling_arrival_hold"
VIEWPOINT_SAMPLING_CONTRACT_VERSION = 1

INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M = 0.018
INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M = 0.020
INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M = 0.030
INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M = 1.0e-5
DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M = 0.33
DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M = 0.017


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _json_safe_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class ViewpointSamplingHoldConfig:
    """Validated geometry shared by strict arrival and terminal-heading hold."""

    entry_tolerance_m: float = INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M
    hold_tolerance_m: float = INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
    target_envelope_radius_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    )
    target_distance_m: float = DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M
    distance_comparison_epsilon_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
    )

    def __post_init__(self) -> None:
        _finite_positive(self.entry_tolerance_m, "entry_tolerance_m")
        _finite_positive(self.hold_tolerance_m, "hold_tolerance_m")
        _finite_positive(
            self.target_envelope_radius_m,
            "target_envelope_radius_m",
        )
        _finite_positive(self.target_distance_m, "target_distance_m")
        if (
            not math.isfinite(self.distance_comparison_epsilon_m)
            or self.distance_comparison_epsilon_m < 0.0
        ):
            raise ValueError(
                "distance_comparison_epsilon_m must be finite and non-negative"
            )
        if (
            self.entry_tolerance_m
            > INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M
        ):
            raise ValueError("entry_tolerance_m must be no greater than 0.018")
        if self.hold_tolerance_m < self.entry_tolerance_m:
            raise ValueError(
                "hold_tolerance_m must be no smaller than entry_tolerance_m"
            )
        if (
            self.hold_tolerance_m
            > INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
        ):
            raise ValueError("hold_tolerance_m must be no greater than 0.020")
        if self.target_envelope_radius_m < self.hold_tolerance_m:
            raise ValueError(
                "target_envelope_radius_m must be no smaller than "
                "hold_tolerance_m"
            )
        if (
            self.target_envelope_radius_m
            > INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
        ):
            raise ValueError(
                "target_envelope_radius_m must be no greater than 0.030"
            )
        if self.target_distance_m <= self.hold_tolerance_m:
            raise ValueError(
                "target_distance_m must be greater than hold_tolerance_m"
            )
        if (
            self.distance_comparison_epsilon_m
            > INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
        ):
            raise ValueError(
                "distance_comparison_epsilon_m must be no greater than 1e-5"
            )


DEFAULT_VIEWPOINT_SAMPLING_HOLD_CONFIG = ViewpointSamplingHoldConfig()


@dataclass(frozen=True)
class ViewpointSamplingHoldMetrics:
    """Exact target-envelope and inferred-stand annulus measurements."""

    target_yaw_rad: float
    heading_is_finite: bool
    target_envelope_distance_m: float
    target_envelope_radius_m: float
    target_envelope_within_limit: bool
    nominal_target_distance_m: float
    inferred_stand_center_x_m: float
    inferred_stand_center_y_m: float
    inferred_stand_distance_m: float
    annulus_min_m: float
    annulus_max_m: float
    inferred_stand_distance_within_annulus: bool
    distance_comparison_epsilon_m: float
    within_hold: bool
    hold_model: str = (
        "target_envelope_and_inferred_stand_distance_annulus"
    )
    distance_unit: str = "m"
    target_yaw_unit: str = "rad"

    def to_diagnostics_dict(self) -> dict[str, object]:
        """Return the follower's existing diagnostic shape, including NaNs."""

        return {
            "hold_model": self.hold_model,
            "distance_unit": self.distance_unit,
            "target_yaw_unit": self.target_yaw_unit,
            "target_yaw_rad": self.target_yaw_rad,
            "heading_is_finite": self.heading_is_finite,
            "target_envelope_distance_m": self.target_envelope_distance_m,
            "target_envelope_radius_m": self.target_envelope_radius_m,
            "target_envelope_within_limit": (
                self.target_envelope_within_limit
            ),
            "nominal_target_distance_m": self.nominal_target_distance_m,
            "inferred_stand_center_x_m": self.inferred_stand_center_x_m,
            "inferred_stand_center_y_m": self.inferred_stand_center_y_m,
            "inferred_stand_distance_m": self.inferred_stand_distance_m,
            "annulus_min_m": self.annulus_min_m,
            "annulus_max_m": self.annulus_max_m,
            "inferred_stand_distance_within_annulus": (
                self.inferred_stand_distance_within_annulus
            ),
            "distance_comparison_epsilon_m": (
                self.distance_comparison_epsilon_m
            ),
            "within_hold": self.within_hold,
        }

    def to_status_dict(self) -> dict[str, object]:
        """Return JSON-safe evidence without non-standard NaN/Infinity values."""

        status = self.to_diagnostics_dict()
        for key, value in tuple(status.items()):
            if isinstance(value, float):
                status[key] = _json_safe_float(value)
        return status


def viewpoint_sampling_hold_metrics(
    pose: Pose2D,
    target_pose: Pose2D,
    *,
    config: ViewpointSamplingHoldConfig = (
        DEFAULT_VIEWPOINT_SAMPLING_HOLD_CONFIG
    ),
) -> ViewpointSamplingHoldMetrics:
    """Measure the shared hold predicates, failing closed on nonfinite poses."""

    target_envelope_distance_m = math.hypot(
        pose.x_m - target_pose.x_m,
        pose.y_m - target_pose.y_m,
    )
    heading_is_finite = math.isfinite(pose.yaw_rad) and math.isfinite(
        target_pose.yaw_rad
    )
    target_envelope_within_limit = (
        math.isfinite(target_envelope_distance_m)
        and target_envelope_distance_m <= config.target_envelope_radius_m
    )

    inferred_stand_center_x_m = math.nan
    inferred_stand_center_y_m = math.nan
    if all(
        math.isfinite(value)
        for value in (
            target_pose.x_m,
            target_pose.y_m,
            target_pose.yaw_rad,
            config.target_distance_m,
        )
    ):
        inferred_stand_center_x_m = (
            target_pose.x_m
            + config.target_distance_m * math.cos(target_pose.yaw_rad)
        )
        inferred_stand_center_y_m = (
            target_pose.y_m
            + config.target_distance_m * math.sin(target_pose.yaw_rad)
        )
    inferred_stand_distance_m = math.hypot(
        pose.x_m - inferred_stand_center_x_m,
        pose.y_m - inferred_stand_center_y_m,
    )
    annulus_min_m = config.target_distance_m - config.hold_tolerance_m
    annulus_max_m = config.target_distance_m + config.hold_tolerance_m
    inferred_stand_distance_within_annulus = (
        math.isfinite(inferred_stand_distance_m)
        and math.isfinite(annulus_min_m)
        and math.isfinite(annulus_max_m)
        and inferred_stand_distance_m
        >= annulus_min_m - config.distance_comparison_epsilon_m
        and inferred_stand_distance_m
        <= annulus_max_m + config.distance_comparison_epsilon_m
    )
    within_hold = (
        heading_is_finite
        and target_envelope_within_limit
        and inferred_stand_distance_within_annulus
    )
    return ViewpointSamplingHoldMetrics(
        target_yaw_rad=target_pose.yaw_rad,
        heading_is_finite=heading_is_finite,
        target_envelope_distance_m=target_envelope_distance_m,
        target_envelope_radius_m=config.target_envelope_radius_m,
        target_envelope_within_limit=target_envelope_within_limit,
        nominal_target_distance_m=config.target_distance_m,
        inferred_stand_center_x_m=inferred_stand_center_x_m,
        inferred_stand_center_y_m=inferred_stand_center_y_m,
        inferred_stand_distance_m=inferred_stand_distance_m,
        annulus_min_m=annulus_min_m,
        annulus_max_m=annulus_max_m,
        inferred_stand_distance_within_annulus=(
            inferred_stand_distance_within_annulus
        ),
        distance_comparison_epsilon_m=(
            config.distance_comparison_epsilon_m
        ),
        within_hold=within_hold,
    )


@dataclass(frozen=True)
class ViewpointSamplingMaterialTarget:
    """Identity key whose material changes reset strict-arrival evidence."""

    pose: Pose2D
    face_id: str
    target_revision: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.face_id, str) or not self.face_id.strip():
            raise ValueError("face_id must be a non-empty string")
        if self.target_revision is not None and (
            type(self.target_revision) is not int
            or self.target_revision < 0
        ):
            raise ValueError("target_revision must be a non-negative integer or None")
        if not all(
            math.isfinite(value)
            for value in (
                self.pose.x_m,
                self.pose.y_m,
                self.pose.yaw_rad,
            )
        ):
            raise ValueError("material target pose must be finite")

    def to_status_dict(self) -> dict[str, object]:
        return {
            "pose": {
                "x_m": self.pose.x_m,
                "y_m": self.pose.y_m,
                "yaw_rad": self.pose.yaw_rad,
            },
            "face_id": self.face_id,
            "target_revision": self.target_revision,
        }


@dataclass(frozen=True)
class ViewpointSamplingArrivalEvidence:
    """One JSON-serializable observation of the two-state arrival latch."""

    state: str
    armed: bool
    strict_ever_armed: bool
    hold_valid: bool
    strict_entry_within_limit: bool
    strict_entry_tolerance_m: float
    transition_reason: str
    reset_reason: str | None
    disarm_reason: str | None
    material_target: ViewpointSamplingMaterialTarget
    metrics: ViewpointSamplingHoldMetrics

    @property
    def arrived(self) -> bool:
        return self.armed

    def to_status_dict(self) -> dict[str, object]:
        return {
            "contract_name": VIEWPOINT_SAMPLING_CONTRACT_NAME,
            "contract_version": VIEWPOINT_SAMPLING_CONTRACT_VERSION,
            "state": self.state,
            "armed": self.armed,
            "arrived": self.arrived,
            "strict_ever_armed": self.strict_ever_armed,
            "hold_valid": self.hold_valid,
            "strict_entry_within_limit": self.strict_entry_within_limit,
            "strict_entry_tolerance_m": self.strict_entry_tolerance_m,
            "transition_reason": self.transition_reason,
            "reset_reason": self.reset_reason,
            "disarm_reason": self.disarm_reason,
            "material_target": self.material_target.to_status_dict(),
            "metrics": self.metrics.to_status_dict(),
        }


@dataclass
class ViewpointSamplingArrivalLatch:
    """ROS-free strict-entry/hold hysteresis for one material target."""

    strict_entry_tolerance_m: float = (
        DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M
    )
    hold_config: ViewpointSamplingHoldConfig = field(
        default_factory=ViewpointSamplingHoldConfig
    )
    _material_target: ViewpointSamplingMaterialTarget | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _armed: bool = field(default=False, init=False, repr=False)
    _strict_ever_armed: bool = field(default=False, init=False, repr=False)
    _reset_reason: str = field(
        default="not_initialized",
        init=False,
        repr=False,
    )
    _disarm_reason: str | None = field(default=None, init=False, repr=False)
    _last_evidence: ViewpointSamplingArrivalEvidence | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _finite_positive(
            self.strict_entry_tolerance_m,
            "strict_entry_tolerance_m",
        )
        if self.strict_entry_tolerance_m > self.hold_config.entry_tolerance_m:
            raise ValueError(
                "strict_entry_tolerance_m must be no greater than "
                "hold_config.entry_tolerance_m"
            )

    @property
    def arrived(self) -> bool:
        return self._armed

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def strict_ever_armed(self) -> bool:
        return self._strict_ever_armed

    @property
    def material_target(self) -> ViewpointSamplingMaterialTarget | None:
        return self._material_target

    def reset(self, reason: str = "explicit_reset") -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reset reason must be a non-empty string")
        self._material_target = None
        self._armed = False
        self._strict_ever_armed = False
        self._reset_reason = reason
        self._disarm_reason = None
        self._last_evidence = None

    @staticmethod
    def _target_change_reason(
        previous: ViewpointSamplingMaterialTarget | None,
        current: ViewpointSamplingMaterialTarget,
    ) -> str:
        if previous is None:
            return "material_target_initialized"
        changed: list[str] = []
        if previous.pose != current.pose:
            changed.append("pose")
        if previous.face_id != current.face_id:
            changed.append("face")
        if previous.target_revision != current.target_revision:
            changed.append("revision")
        return "material_target_" + "_".join(changed) + "_changed"

    def update(
        self,
        *,
        pose: Pose2D,
        target: ViewpointSamplingMaterialTarget,
    ) -> ViewpointSamplingArrivalEvidence:
        metrics = viewpoint_sampling_hold_metrics(
            pose,
            target.pose,
            config=self.hold_config,
        )
        if target != self._material_target:
            reset_reason = self._target_change_reason(
                self._material_target,
                target,
            )
            transition_reason = (
                "target_initialized_unarmed"
                if self._material_target is None
                else "material_target_reset_unarmed"
            )
            self._material_target = target
            self._armed = False
            self._strict_ever_armed = False
            self._reset_reason = reset_reason
            self._disarm_reason = None
            evidence = ViewpointSamplingArrivalEvidence(
                state="unarmed",
                armed=False,
                strict_ever_armed=False,
                hold_valid=metrics.within_hold,
                strict_entry_within_limit=False,
                strict_entry_tolerance_m=self.strict_entry_tolerance_m,
                transition_reason=transition_reason,
                reset_reason=reset_reason,
                disarm_reason=None,
                material_target=target,
                metrics=metrics,
            )
            self._last_evidence = evidence
            return evidence

        strict_entry_within_limit = (
            metrics.within_hold
            and math.isfinite(metrics.target_envelope_distance_m)
            and metrics.target_envelope_distance_m
            <= self.strict_entry_tolerance_m
        )
        if self._armed:
            if metrics.within_hold:
                transition_reason = "armed_hold_valid"
            else:
                self._armed = False
                transition_reason = "hold_invalid_disarmed"
                if not metrics.heading_is_finite:
                    self._disarm_reason = "nonfinite_heading"
                elif not metrics.target_envelope_within_limit:
                    self._disarm_reason = "target_envelope_exceeded"
                else:
                    self._disarm_reason = "inferred_stand_annulus_exceeded"
        elif strict_entry_within_limit:
            self._armed = True
            self._strict_ever_armed = True
            self._disarm_reason = None
            transition_reason = "strict_entry_armed"
        else:
            transition_reason = "awaiting_strict_entry"

        evidence = ViewpointSamplingArrivalEvidence(
            state="armed" if self._armed else "unarmed",
            armed=self._armed,
            strict_ever_armed=self._strict_ever_armed,
            hold_valid=metrics.within_hold,
            strict_entry_within_limit=strict_entry_within_limit,
            strict_entry_tolerance_m=self.strict_entry_tolerance_m,
            transition_reason=transition_reason,
            reset_reason=self._reset_reason,
            disarm_reason=self._disarm_reason,
            material_target=target,
            metrics=metrics,
        )
        self._last_evidence = evidence
        return evidence

    def to_status_dict(self) -> dict[str, Any]:
        if self._last_evidence is not None:
            return self._last_evidence.to_status_dict()
        return {
            "contract_name": VIEWPOINT_SAMPLING_CONTRACT_NAME,
            "contract_version": VIEWPOINT_SAMPLING_CONTRACT_VERSION,
            "state": "unarmed",
            "armed": False,
            "arrived": False,
            "strict_ever_armed": self._strict_ever_armed,
            "hold_valid": False,
            "strict_entry_within_limit": False,
            "strict_entry_tolerance_m": self.strict_entry_tolerance_m,
            "transition_reason": "reset",
            "reset_reason": self._reset_reason,
            "disarm_reason": None,
            "material_target": None,
            "metrics": None,
        }


__all__ = [
    "DEFAULT_VIEWPOINT_SAMPLING_HOLD_CONFIG",
    "DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M",
    "DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M",
    "INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M",
    "INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M",
    "INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M",
    "INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M",
    "VIEWPOINT_SAMPLING_CONTRACT_NAME",
    "VIEWPOINT_SAMPLING_CONTRACT_VERSION",
    "ViewpointSamplingArrivalEvidence",
    "ViewpointSamplingArrivalLatch",
    "ViewpointSamplingHoldConfig",
    "ViewpointSamplingHoldMetrics",
    "ViewpointSamplingMaterialTarget",
    "viewpoint_sampling_hold_metrics",
]
