"""Pure boundary between map planning and odom-frame route execution.

The global planner continues to own immutable routes in ``map``.  A stopped
admission step freezes one ``map_from_odom`` transform, after which this module
can convert map poses and adopted map-route revisions into the ``odom`` frame.
The controller must not use a later localization correction to steer that
already-authorized route.

Continuity threshold semantics are deliberately exact and fail closed:

* translation and absolute normalized relative yaw are accepted when they are
  less than or equal to their configured limits;
* a missing or malformed live transform is rejected; and
* any value above either limit requires a zero-command cycle and resealing.

This module is ROS-free.  It does not call a planner, publish velocity, or
mutate a route source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    PlanarTransform2D,
    map_pose_to_odom,
    normalize_yaw,
    odom_pose_to_map,
    pose_route_sha256,
    transform_map_route_to_odom,
)


CONTINUITY_EVIDENCE_SCHEMA_VERSION = 1
CONTINUITY_ACCEPTED = "continue_odom_execution"
CONTINUITY_RESEAL = "force_zero_reseal"


@dataclass(frozen=True)
class OdomExecutionContext:
    """Immutable identity and transform limits for one execution authority."""

    map_frame: str
    odom_frame: str
    base_frame: str
    frozen_map_from_odom: PlanarTransform2D
    certificate_sha256: str
    max_map_from_odom_translation_drift_m: float
    max_map_from_odom_yaw_drift_rad: float

    def __post_init__(self) -> None:
        frames = (
            _frame_id(self.map_frame, "map_frame"),
            _frame_id(self.odom_frame, "odom_frame"),
            _frame_id(self.base_frame, "base_frame"),
        )
        if len(set(frames)) != len(frames):
            raise ValueError("map_frame, odom_frame, and base_frame must be distinct")
        _sha256(self.certificate_sha256, "certificate_sha256")
        transform = _validated_transform(self.frozen_map_from_odom)
        translation_limit = _finite_nonnegative(
            self.max_map_from_odom_translation_drift_m,
            "max_map_from_odom_translation_drift_m",
        )
        yaw_limit = _finite_nonnegative(
            self.max_map_from_odom_yaw_drift_rad,
            "max_map_from_odom_yaw_drift_rad",
        )
        if yaw_limit > math.pi:
            raise ValueError("max_map_from_odom_yaw_drift_rad must be <= pi")

        object.__setattr__(self, "frozen_map_from_odom", transform)
        object.__setattr__(
            self,
            "max_map_from_odom_translation_drift_m",
            translation_limit,
        )
        object.__setattr__(
            self,
            "max_map_from_odom_yaw_drift_rad",
            yaw_limit,
        )

    @property
    def odom_execution_certificate_sha256(self) -> str:
        """Descriptive alias used in emitted route and continuity evidence."""

        return self.certificate_sha256

    def map_pose_to_odom(self, pose_map: Pose2D) -> Pose2D:
        """Convert one planner pose using only the frozen transform."""

        return map_pose_to_odom(pose_map, self.frozen_map_from_odom)

    def odom_pose_to_map(self, pose_odom: Pose2D) -> Pose2D:
        """Convert one execution pose using only the frozen transform."""

        return odom_pose_to_map(pose_odom, self.frozen_map_from_odom)


@dataclass(frozen=True)
class MapOdomContinuityResult:
    """One fail-closed comparison of live and frozen ``map_from_odom``."""

    accepted: bool
    reason: str
    decision: str
    certificate_sha256: str
    map_frame: str
    odom_frame: str
    base_frame: str
    frozen_map_from_odom: PlanarTransform2D
    live_map_from_odom: PlanarTransform2D | None
    relative_translation_x_m: float | None
    relative_translation_y_m: float | None
    translation_drift_m: float | None
    relative_yaw_rad: float | None
    absolute_yaw_drift_rad: float | None
    max_translation_drift_m: float
    max_yaw_drift_rad: float
    validation_error: str | None = None

    @property
    def requires_zero_reseal(self) -> bool:
        return not self.accepted

    @property
    def requires_zero_cycle(self) -> bool:
        return not self.accepted

    def to_evidence(self) -> dict[str, Any]:
        """Return JSON-ready evidence without exposing mutable internal state."""

        return {
            "schema_version": CONTINUITY_EVIDENCE_SCHEMA_VERSION,
            "accepted": self.accepted,
            "decision": self.decision,
            "reason": self.reason,
            "fail_closed": not self.accepted,
            "requires_zero_cycle": self.requires_zero_cycle,
            "requires_reseal": self.requires_zero_reseal,
            "threshold_semantics": "accept_if_observed_less_than_or_equal_to_limit",
            "certificate_sha256": self.certificate_sha256,
            "map_frame": self.map_frame,
            "odom_frame": self.odom_frame,
            "base_frame": self.base_frame,
            "frozen_map_from_odom": _transform_evidence(
                self.frozen_map_from_odom
            ),
            "live_map_from_odom": (
                None
                if self.live_map_from_odom is None
                else _transform_evidence(self.live_map_from_odom)
            ),
            "relative_translation_x_m": self.relative_translation_x_m,
            "relative_translation_y_m": self.relative_translation_y_m,
            "translation_drift_m": self.translation_drift_m,
            "relative_yaw_rad": self.relative_yaw_rad,
            "absolute_yaw_drift_rad": self.absolute_yaw_drift_rad,
            "max_translation_drift_m": self.max_translation_drift_m,
            "max_yaw_drift_rad": self.max_yaw_drift_rad,
            "validation_error": self.validation_error,
        }


def evaluate_map_odom_continuity(
    context: OdomExecutionContext,
    live_map_from_odom: object | None,
) -> MapOdomContinuityResult:
    """Compare a live localization correction with the frozen transform.

    Equality at either limit is accepted.  Missing or malformed live evidence
    returns a rejected result instead of raising so a motion-side caller can
    deterministically publish zero and terminate the current authorization.
    A malformed execution context remains a programming/configuration error
    and raises :class:`ValueError`.
    """

    context = _validated_context(context)
    if live_map_from_odom is None:
        return _unavailable_continuity_result(
            context,
            reason="map_from_odom_missing",
            validation_error="live map_from_odom is missing",
        )
    try:
        live = _validated_transform(live_map_from_odom)
    except (TypeError, ValueError) as exc:
        return _unavailable_continuity_result(
            context,
            reason="map_from_odom_malformed",
            validation_error=str(exc),
        )

    frozen = context.frozen_map_from_odom
    map_delta_x = live.x_m - frozen.x_m
    map_delta_y = live.y_m - frozen.y_m
    cosine = math.cos(frozen.yaw_rad)
    sine = math.sin(frozen.yaw_rad)
    # Express the relative translation in the frozen odom basis.  Its norm is
    # rotation-invariant, while the components make transform direction clear
    # in persisted evidence.
    relative_x = _canonical_zero(cosine * map_delta_x + sine * map_delta_y)
    relative_y = _canonical_zero(-sine * map_delta_x + cosine * map_delta_y)
    translation_drift = math.hypot(relative_x, relative_y)
    relative_yaw = normalize_yaw(live.yaw_rad - frozen.yaw_rad)
    absolute_yaw_drift = abs(relative_yaw)
    translation_ok = (
        translation_drift
        <= context.max_map_from_odom_translation_drift_m
    )
    yaw_ok = absolute_yaw_drift <= context.max_map_from_odom_yaw_drift_rad

    if translation_ok and yaw_ok:
        accepted = True
        reason = "map_from_odom_continuous"
        decision = CONTINUITY_ACCEPTED
    else:
        accepted = False
        decision = CONTINUITY_RESEAL
        if not translation_ok and not yaw_ok:
            reason = "map_from_odom_translation_and_yaw_drift"
        elif not translation_ok:
            reason = "map_from_odom_translation_drift"
        else:
            reason = "map_from_odom_yaw_drift"

    return MapOdomContinuityResult(
        accepted=accepted,
        reason=reason,
        decision=decision,
        certificate_sha256=context.certificate_sha256,
        map_frame=context.map_frame,
        odom_frame=context.odom_frame,
        base_frame=context.base_frame,
        frozen_map_from_odom=frozen,
        live_map_from_odom=live,
        relative_translation_x_m=relative_x,
        relative_translation_y_m=relative_y,
        translation_drift_m=translation_drift,
        relative_yaw_rad=relative_yaw,
        absolute_yaw_drift_rad=absolute_yaw_drift,
        max_translation_drift_m=(
            context.max_map_from_odom_translation_drift_m
        ),
        max_yaw_drift_rad=context.max_map_from_odom_yaw_drift_rad,
    )


def adapt_map_route_update_to_odom(
    update: RouteUpdate,
    context: OdomExecutionContext,
) -> RouteUpdate:
    """Transform an ADOPT update and leave every other update untouched.

    The planner's map-route hash is retained as
    ``event_fields['source_map_route_sha256']``.  ``route_hash`` becomes the
    canonical hash of the transformed odom pose route.  Adaptation always
    requires a complete zero-command handoff cycle.
    """

    if not isinstance(update, RouteUpdate):
        raise ValueError("update must be a RouteUpdate")
    if update.kind is not RouteUpdateKind.ADOPT:
        return update

    context = _validated_context(context)
    source_map_sha256 = _sha256(update.route_hash, "update.route_hash")
    odom_waypoints = transform_map_route_to_odom(
        update.waypoints,
        context.frozen_map_from_odom,
    )
    odom_route_sha256 = pose_route_sha256(odom_waypoints)
    map_pose_route_sha256 = pose_route_sha256(update.waypoints)
    event_fields = dict(_event_fields(update.event_fields))
    additions: dict[str, object] = {
        "source_map_route_sha256": source_map_sha256,
        "source_map_pose_route_sha256": map_pose_route_sha256,
        "transformed_odom_route_sha256": odom_route_sha256,
        "odom_execution_certificate_sha256": context.certificate_sha256,
        "source_route_frame": context.map_frame,
        "execution_route_frame": context.odom_frame,
    }
    for name, value in additions.items():
        if name in event_fields and event_fields[name] != value:
            raise ValueError(
                f"update.event_fields[{name!r}] conflicts with odom adaptation"
            )
        event_fields[name] = value
    if (
        "requires_zero_cycle" in event_fields
        and event_fields["requires_zero_cycle"] is not True
    ):
        raise ValueError(
            "update.event_fields['requires_zero_cycle'] must be true for ADOPT"
        )

    return RouteUpdate(
        kind=update.kind,
        waypoints=odom_waypoints,
        target_index=update.target_index,
        reason=update.reason,
        route_revision=update.route_revision,
        target_revision=update.target_revision,
        route_hash=odom_route_sha256,
        requires_zero_cycle=True,
        event_name=update.event_name,
        event_fields=event_fields,
    )


def _unavailable_continuity_result(
    context: OdomExecutionContext,
    *,
    reason: str,
    validation_error: str,
) -> MapOdomContinuityResult:
    return MapOdomContinuityResult(
        accepted=False,
        reason=reason,
        decision=CONTINUITY_RESEAL,
        certificate_sha256=context.certificate_sha256,
        map_frame=context.map_frame,
        odom_frame=context.odom_frame,
        base_frame=context.base_frame,
        frozen_map_from_odom=context.frozen_map_from_odom,
        live_map_from_odom=None,
        relative_translation_x_m=None,
        relative_translation_y_m=None,
        translation_drift_m=None,
        relative_yaw_rad=None,
        absolute_yaw_drift_rad=None,
        max_translation_drift_m=(
            context.max_map_from_odom_translation_drift_m
        ),
        max_yaw_drift_rad=context.max_map_from_odom_yaw_drift_rad,
        validation_error=validation_error,
    )


def _validated_context(context: OdomExecutionContext) -> OdomExecutionContext:
    if not isinstance(context, OdomExecutionContext):
        raise ValueError("context must be an OdomExecutionContext")
    # Reconstruct to reject an object whose frozen fields were deliberately
    # bypassed or corrupted after construction.
    return OdomExecutionContext(
        map_frame=context.map_frame,
        odom_frame=context.odom_frame,
        base_frame=context.base_frame,
        frozen_map_from_odom=context.frozen_map_from_odom,
        certificate_sha256=context.certificate_sha256,
        max_map_from_odom_translation_drift_m=(
            context.max_map_from_odom_translation_drift_m
        ),
        max_map_from_odom_yaw_drift_rad=(
            context.max_map_from_odom_yaw_drift_rad
        ),
    )


def _validated_transform(value: object) -> PlanarTransform2D:
    if not isinstance(value, PlanarTransform2D):
        raise ValueError("map_from_odom must be a PlanarTransform2D")
    return PlanarTransform2D(value.x_m, value.y_m, value.yaw_rad)


def _transform_evidence(transform: PlanarTransform2D) -> dict[str, float]:
    return {
        "x_m": transform.x_m,
        "y_m": transform.y_m,
        "yaw_rad": transform.yaw_rad,
    }


def _event_fields(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("update.event_fields must be a mapping")
    return value


def _frame_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty frame id")
    if value != value.strip() or value.startswith("/") or any(
        character.isspace() for character in value
    ):
        raise ValueError(
            f"{name} must be an unprefixed frame id without whitespace"
        )
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return _canonical_zero(result)


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value
