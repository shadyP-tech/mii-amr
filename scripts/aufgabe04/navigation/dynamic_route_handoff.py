"""Pure, fail-closed hand-off of immutable simulation route revisions."""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.route_revision_store import (
    LoadedRouteRevision,
    RouteRevisionError,
    file_sha256,
    read_route_revision,
)
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg, poses_from_waypoints


class RouteUpdateKind(str, enum.Enum):
    ADOPT = "adopt"
    UNCHANGED = "unchanged"
    REJECT = "reject"
    STOP = "stop"
    COMPLETE = "complete"


@dataclass(frozen=True)
class RouteUpdate:
    """One polling result for a motion-side route consumer."""

    kind: RouteUpdateKind
    waypoints: Tuple[Pose2D, ...] = ()
    target_index: int | None = None
    reason: str = ""
    route_revision: int | None = None
    target_revision: int | None = None
    route_hash: str | None = None
    requires_zero_cycle: bool = False
    event_name: str | None = None
    event_fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StartEgressCertificate:
    required: bool
    waypoint_index: int | None = None
    minimum_route_clearance_m: float | None = None


def _finite_nonnegative(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


def _point_to_segment_distance_m(
    point: Pose2D,
    start: Pose2D,
    end: Pose2D,
) -> float:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-20:
        return math.hypot(point.x_m - start.x_m, point.y_m - start.y_m)
    fraction = max(
        0.0,
        min(
            1.0,
            (
                (point.x_m - start.x_m) * dx
                + (point.y_m - start.y_m) * dy
            )
            / denominator,
        ),
    )
    return math.hypot(
        point.x_m - (start.x_m + fraction * dx),
        point.y_m - (start.y_m + fraction * dy),
    )


def validate_start_egress_certificate(
    safety: Mapping[str, Any],
    waypoints: Tuple[Pose2D, ...],
    planned_start: Pose2D,
) -> StartEgressCertificate:
    """Validate the planner evidence required for one-cell raster egress."""

    raw_required = safety.get("known_stand_start_cell_exempted", False)
    if not isinstance(raw_required, bool):
        raise ValueError("known_stand_start_cell_exempted must be boolean")
    if not raw_required:
        return StartEgressCertificate(False)

    if len(waypoints) < 2:
        raise ValueError("start-cell exemption route lacks waypoint 1")
    if math.hypot(
        waypoints[1].x_m - waypoints[0].x_m,
        waypoints[1].y_m - waypoints[0].y_m,
    ) <= 1.0e-9:
        raise ValueError("start-cell exemption waypoint 1 is not an egress vertex")

    start_cell = safety.get("known_stand_start_cell")
    if not isinstance(start_cell, Mapping):
        raise ValueError("known_stand_start_cell is missing")
    for coordinate in ("x", "y"):
        value = start_cell.get(coordinate)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"known_stand_start_cell.{coordinate} must be integer")

    rasterized_count = safety.get("known_stand_keepout_rasterized_cell_count")
    blocked_count = safety.get("known_stand_keepout_cell_count")
    if (
        not isinstance(rasterized_count, int)
        or isinstance(rasterized_count, bool)
        or rasterized_count <= 0
    ):
        raise ValueError("known stand rasterized cell count must be positive")
    if (
        not isinstance(blocked_count, int)
        or isinstance(blocked_count, bool)
        or blocked_count < 0
        or blocked_count != rasterized_count - 1
    ):
        raise ValueError("start-cell exemption must remove exactly one raster cell")

    keepouts = safety.get("known_stand_keepouts")
    evidence = safety.get("known_stand_keepout_clearances")
    if not isinstance(keepouts, (list, tuple)) or not keepouts:
        raise ValueError("start-cell exemption has no known stand keepouts")
    if not isinstance(evidence, (list, tuple)) or len(evidence) != len(keepouts):
        raise ValueError("continuous keepout-clearance evidence is incomplete")

    route_minimum = math.inf
    for index, (keepout, clearance) in enumerate(zip(keepouts, evidence)):
        if not isinstance(keepout, Mapping) or not isinstance(clearance, Mapping):
            raise ValueError(f"known stand clearance {index} must be an object")
        try:
            x_m = float(keepout["x_m"])
            y_m = float(keepout["y_m"])
            radius_m = float(keepout["radius_m"])
            evidence_x_m = float(clearance["x_m"])
            evidence_y_m = float(clearance["y_m"])
            evidence_radius_m = float(clearance["radius_m"])
            reported_minimum_m = float(clearance["minimum_route_clearance_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"known stand clearance {index} is malformed: {exc}"
            ) from exc
        values = (
            x_m,
            y_m,
            radius_m,
            evidence_x_m,
            evidence_y_m,
            evidence_radius_m,
            reported_minimum_m,
        )
        if not all(math.isfinite(value) for value in values) or radius_m <= 0.0:
            raise ValueError(f"known stand clearance {index} is not finite")
        if not (
            math.isclose(x_m, evidence_x_m, abs_tol=1.0e-9)
            and math.isclose(y_m, evidence_y_m, abs_tol=1.0e-9)
            and math.isclose(radius_m, evidence_radius_m, abs_tol=1.0e-9)
        ):
            raise ValueError(f"known stand clearance {index} identity mismatch")
        if reported_minimum_m <= radius_m + 1.0e-10:
            raise ValueError(f"known stand clearance {index} is not outside its disk")

        center = Pose2D(x_m, y_m)
        exact_start_clearance_m = math.hypot(
            planned_start.x_m - x_m,
            planned_start.y_m - y_m,
        )
        if exact_start_clearance_m <= radius_m + 1.0e-10:
            raise ValueError(f"route start lies inside known stand keepout {index}")
        measured_minimum_m = min(
            _point_to_segment_distance_m(center, segment_start, segment_end)
            for segment_start, segment_end in zip(waypoints, waypoints[1:])
        )
        if measured_minimum_m <= radius_m + 1.0e-10:
            raise ValueError(f"adopted route crosses known stand keepout {index}")
        if not math.isclose(
            measured_minimum_m,
            reported_minimum_m,
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        ):
            raise ValueError(
                f"known stand clearance {index} does not match adopted route"
            )
        route_minimum = min(route_minimum, measured_minimum_m)

    return StartEgressCertificate(
        required=True,
        waypoint_index=1,
        minimum_route_clearance_m=route_minimum,
    )


def forward_splice_waypoints(
    current_pose: Pose2D,
    waypoints: Tuple[Pose2D, ...],
    *,
    certified_radius_m: float,
) -> tuple[Tuple[Pose2D, ...], float]:
    """Join a replacement route without commanding a return to its old start.

    The live pose and the synthesized forward point are both constrained to
    the planner-certified free disk around waypoint zero.  Because a disk is
    convex, their connecting segment is collision-free; from the splice point
    onward the original checked route is preserved.
    """

    if len(waypoints) < 2:
        raise ValueError("forward splice requires at least two waypoints")
    radius = _finite_nonnegative(certified_radius_m, "certified_radius_m")
    if radius <= 0.0:
        raise ValueError("certified_radius_m must be positive")
    start, following = waypoints[0], waypoints[1]
    segment_length = math.hypot(
        following.x_m - start.x_m,
        following.y_m - start.y_m,
    )
    runtime_anchor = Pose2D(current_pose.x_m, current_pose.y_m, float("nan"))
    if segment_length <= 1.0e-9:
        return (runtime_anchor, *waypoints[1:]), 0.0

    forward_distance = min(segment_length, 0.8 * radius)
    fraction = forward_distance / segment_length
    if fraction >= 1.0 - 1.0e-9:
        return (runtime_anchor, *waypoints[1:]), segment_length
    constrained_heading = (
        start.yaw_rad
        if math.isfinite(start.yaw_rad)
        and math.isfinite(following.yaw_rad)
        and abs(start.yaw_rad - following.yaw_rad) <= 1.0e-6
        else float("nan")
    )
    splice = Pose2D(
        start.x_m + fraction * (following.x_m - start.x_m),
        start.y_m + fraction * (following.y_m - start.y_m),
        constrained_heading,
    )
    return (runtime_anchor, splice, *waypoints[1:]), forward_distance


class DynamicRouteSource:
    """Stateful reader that turns committed manifests into safe route updates.

    ``poll`` is side-effect free outside this object's memory and performs no
    ROS operations.  An adopted suffix starts with the selected join anchor;
    consumers should begin pursuing waypoint index 1 after publishing one full
    zero-command control cycle.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        stream_id: str,
        leg_index: int = 0,
        expected_writer_id: str | None = None,
        allow_writer_takeover: bool = False,
        max_manifest_age_sec: float | None = None,
        max_observation_age_sec: float | None = None,
        max_join_distance_m: float = 0.35,
        max_forward_window: int = 24,
        terminal_route_lock_distance_m: float = 0.0,
        forward_splice_min_offset_m: float = 0.01,
        thinning_min_spacing_m: float = 0.0,
        require_contiguous_revision: bool = False,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        if leg_index < 0:
            raise ValueError("leg_index must be non-negative")
        if max_forward_window < 0:
            raise ValueError("max_forward_window must be non-negative")
        self.manifest_path = Path(manifest_path)
        self.stream_id = str(stream_id)
        self.leg_index = int(leg_index)
        self.expected_writer_id = expected_writer_id
        self.allow_writer_takeover = bool(allow_writer_takeover)
        self.max_manifest_age_sec = (
            None
            if max_manifest_age_sec is None
            else _finite_nonnegative(max_manifest_age_sec, "max_manifest_age_sec")
        )
        self.max_observation_age_sec = (
            None
            if max_observation_age_sec is None
            else _finite_nonnegative(max_observation_age_sec, "max_observation_age_sec")
        )
        self.max_join_distance_m = _finite_nonnegative(
            max_join_distance_m, "max_join_distance_m"
        )
        self.max_forward_window = int(max_forward_window)
        self.terminal_route_lock_distance_m = _finite_nonnegative(
            terminal_route_lock_distance_m, "terminal_route_lock_distance_m"
        )
        self.forward_splice_min_offset_m = _finite_nonnegative(
            forward_splice_min_offset_m, "forward_splice_min_offset_m"
        )
        self.thinning_min_spacing_m = _finite_nonnegative(
            thinning_min_spacing_m, "thinning_min_spacing_m"
        )
        self.require_contiguous_revision = bool(require_contiguous_revision)
        self.now_fn = now_fn

        self.last_seen_revision: int | None = None
        self.last_seen_manifest_hash: str | None = None
        self.last_adopted_revision: int | None = None
        self.last_route_hash: str | None = None
        self.last_target_revision: int | None = None
        self.last_writer_id: str | None = None
        self.last_writer_generation: int | None = None
        self.last_adopted_target: Pose2D | None = None
        self._last_event_signature: tuple[str, str, int | None] | None = None

    def _event_name_once(
        self,
        *,
        kind: RouteUpdateKind,
        reason: str,
        revision: int | None,
        event_name: str,
    ) -> str | None:
        signature = (kind.value, reason, revision)
        if signature == self._last_event_signature:
            return None
        self._last_event_signature = signature
        return event_name

    def _base_fields(self, loaded: LoadedRouteRevision | None) -> dict[str, Any]:
        fields: dict[str, Any] = {"stream_id": self.stream_id}
        if loaded is None:
            return fields
        manifest = loaded.manifest
        fields.update(
            {
                "route_revision": loaded.route_revision,
                "target_revision": loaded.target_revision,
                "route_sha256": loaded.route_hash,
                "manifest_sha256": loaded.manifest_sha256,
                "writer_id": loaded.writer_id,
                "writer_generation": loaded.writer_generation,
                "published_unix_sec": manifest.get("published_unix_sec"),
                "observation_unix_sec": manifest.get("observation_unix_sec"),
                "source_robot_pose": manifest.get("source_robot_pose", {}),
                "target": manifest.get("target", {}),
                "evidence": manifest.get("evidence", {}),
                "previous_route_length_m": manifest.get("previous_route_length_m"),
                "new_route_length_m": manifest.get("new_route_length_m"),
                "safety_diagnostics": manifest.get("safety_diagnostics", {}),
            }
        )
        return fields

    def _fault(
        self,
        reason: str,
        *,
        code: str,
        loaded: LoadedRouteRevision | None = None,
        stop: bool = True,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> RouteUpdate:
        kind = RouteUpdateKind.STOP if stop else RouteUpdateKind.REJECT
        revision = loaded.route_revision if loaded is not None else self.last_seen_revision
        fields = self._base_fields(loaded)
        fields.update({"reason": reason, "fault_code": code, "fail_closed": stop})
        if extra_fields is not None:
            fields.update(dict(extra_fields))
        return RouteUpdate(
            kind=kind,
            reason=reason,
            route_revision=revision,
            target_revision=(loaded.target_revision if loaded is not None else None),
            route_hash=(loaded.route_hash if loaded is not None else None),
            requires_zero_cycle=stop,
            event_name=self._event_name_once(
                kind=kind,
                reason=f"{code}:{reason}",
                revision=revision,
                event_name="dynamic_route_stopped" if stop else "dynamic_route_rejected",
            ),
            event_fields=fields,
        )

    def _validate_writer_transition(self, loaded: LoadedRouteRevision) -> None:
        if self.last_writer_id is None:
            return
        if loaded.writer_id == self.last_writer_id:
            if loaded.writer_generation != self.last_writer_generation:
                raise RouteRevisionError(
                    "writer_generation_changed",
                    "writer generation changed without a writer takeover",
                )
            return
        if not self.allow_writer_takeover:
            raise RouteRevisionError("wrong_writer", "route writer changed during an active stream")
        record = loaded.manifest.get("writer_takeover")
        if not isinstance(record, Mapping):
            raise RouteRevisionError("invalid_takeover", "writer change lacks takeover evidence")
        if (
            record.get("previous_writer_id") != self.last_writer_id
            or record.get("previous_writer_generation") != self.last_writer_generation
            or loaded.writer_generation != int(self.last_writer_generation or 0) + 1
        ):
            raise RouteRevisionError(
                "invalid_takeover", "writer takeover does not continue the observed ownership chain"
            )

    def _remember_seen(self, loaded: LoadedRouteRevision) -> None:
        self.last_seen_revision = loaded.route_revision
        self.last_seen_manifest_hash = loaded.manifest_sha256
        self.last_writer_id = loaded.writer_id
        self.last_writer_generation = loaded.writer_generation

    def poll(self, current_pose: Pose2D, now_unix_sec: float | None = None) -> RouteUpdate:
        """Return ADOPT, UNCHANGED, COMPLETE, REJECT, or fail-closed STOP."""

        if not all(
            math.isfinite(value)
            for value in (current_pose.x_m, current_pose.y_m, current_pose.yaw_rad)
        ):
            return self._fault(
                "current robot pose is non-finite",
                code="invalid_current_pose",
                stop=True,
            )
        now = self.now_fn() if now_unix_sec is None else now_unix_sec
        terminal_route_locked = (
            self.terminal_route_lock_distance_m > 0.0
            and self.last_adopted_target is not None
            and math.hypot(
                current_pose.x_m - self.last_adopted_target.x_m,
                current_pose.y_m - self.last_adopted_target.y_m,
            ) <= self.terminal_route_lock_distance_m
        )
        try:
            loaded = read_route_revision(
                self.manifest_path,
                expected_stream_id=self.stream_id,
                expected_writer_id=self.expected_writer_id,
                last_route_revision=self.last_seen_revision,
                last_manifest_sha256=self.last_seen_manifest_hash,
                require_contiguous_revision=self.require_contiguous_revision,
                # Inside the installed, collision-checked terminal corridor,
                # route/perception age must not force a return to its entrance.
                # Revisions are still polled and validated, so a changed camera
                # target is adopted immediately. Runtime LiDAR safety remains
                # enforced by the follower.
                max_manifest_age_sec=(
                    None if terminal_route_locked else self.max_manifest_age_sec
                ),
                max_observation_age_sec=(
                    None if terminal_route_locked else self.max_observation_age_sec
                ),
                now_unix_sec=now,
            )
            self._validate_writer_transition(loaded)
        except RouteRevisionError as exc:
            return self._fault(str(exc), code=exc.code, stop=True)
        except OSError as exc:
            return self._fault(str(exc), code="manifest_io", stop=True)

        if self.last_target_revision is not None and loaded.target_revision < self.last_target_revision:
            return self._fault(
                "target revision rolled back",
                code="target_revision_rollback",
                loaded=loaded,
                stop=True,
            )

        if loaded.status == "withdrawn":
            self._remember_seen(loaded)
            self.last_target_revision = loaded.target_revision
            reason = loaded.reason
            fields = self._base_fields(loaded)
            fields.update({"reason": reason, "fail_closed": True})
            return RouteUpdate(
                kind=RouteUpdateKind.STOP,
                reason=reason,
                route_revision=loaded.route_revision,
                target_revision=loaded.target_revision,
                requires_zero_cycle=True,
                event_name=self._event_name_once(
                    kind=RouteUpdateKind.STOP,
                    reason=f"withdrawn:{reason}",
                    revision=loaded.route_revision,
                    event_name="dynamic_route_withdrawn",
                ),
                event_fields=fields,
            )

        if loaded.status == "survey_complete":
            self._remember_seen(loaded)
            self.last_target_revision = loaded.target_revision
            reason = loaded.reason
            fields = self._base_fields(loaded)
            fields.update(
                {
                    "reason": reason,
                    "completion": loaded.manifest.get("completion", {}),
                    "fail_closed": False,
                }
            )
            return RouteUpdate(
                kind=RouteUpdateKind.COMPLETE,
                reason=reason,
                route_revision=loaded.route_revision,
                target_revision=loaded.target_revision,
                route_hash=loaded.route_hash,
                requires_zero_cycle=True,
                event_name=self._event_name_once(
                    kind=RouteUpdateKind.COMPLETE,
                    reason=f"survey_complete:{reason}",
                    revision=loaded.route_revision,
                    event_name="dynamic_survey_completed",
                ),
                event_fields=fields,
            )

        if loaded.duplicate:
            return RouteUpdate(
                kind=RouteUpdateKind.UNCHANGED,
                reason="route revision unchanged",
                route_revision=loaded.route_revision,
                target_revision=loaded.target_revision,
                route_hash=loaded.route_hash,
                event_fields=self._base_fields(loaded),
            )

        same_installed_geometry = (
            self.last_route_hash is not None
            and loaded.route_hash == self.last_route_hash
            and self.last_target_revision is not None
            and loaded.target_revision == self.last_target_revision
        )

        # Ownership/revision metadata is now trusted.  Remember it even when a
        # geometrically unsafe join is rejected, so the same rejection is not
        # emitted indefinitely and a subsequent revision can supersede it.
        self._remember_seen(loaded)
        self.last_target_revision = loaded.target_revision

        if loaded.route_path is None or loaded.route_hash is None:
            return self._fault(
                "active revision has no validated route artifact",
                code="artifact_unavailable",
                loaded=loaded,
                stop=True,
            )
        try:
            selected = load_route_leg(
                loaded.route_path,
                self.leg_index,
                require_motion=True,
                thinning_min_spacing_m=self.thinning_min_spacing_m,
            )
            # Defend against replacement between manifest validation and CSV
            # parsing.  Immutable revisions must retain the committed digest.
            if file_sha256(loaded.route_path) != loaded.route_hash:
                raise RouteRevisionError(
                    "artifact_hash_mismatch", "route changed while it was being loaded"
                )
        except RouteRevisionError as exc:
            return self._fault(str(exc), code=exc.code, loaded=loaded, stop=True)
        except (OSError, ValueError) as exc:
            return self._fault(
                f"route CSV validation failed: {exc}",
                code="route_csv_invalid",
                loaded=loaded,
                stop=True,
            )

        full_waypoints = poses_from_waypoints(selected.executable_waypoints)
        if len(full_waypoints) < 2:
            return self._fault(
                "route has no forward waypoint after its anchor",
                code="route_too_short",
                loaded=loaded,
                stop=True,
            )
        source_payload = loaded.manifest.get("source_robot_pose")
        try:
            if not isinstance(source_payload, Mapping):
                raise ValueError("source_robot_pose is missing")
            planned_start = Pose2D(
                float(source_payload["x_m"]),
                float(source_payload["y_m"]),
                float(source_payload["yaw_rad"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._fault(
                f"route source pose is invalid: {exc}",
                code="invalid_join_certificate",
                loaded=loaded,
                stop=True,
            )
        if not all(
            math.isfinite(value)
            for value in (planned_start.x_m, planned_start.y_m, planned_start.yaw_rad)
        ):
            return self._fault(
                "route source pose is non-finite",
                code="invalid_join_certificate",
                loaded=loaded,
                stop=True,
            )
        route_anchor = full_waypoints[0]
        anchor_error = math.hypot(
            route_anchor.x_m - planned_start.x_m,
            route_anchor.y_m - planned_start.y_m,
        )
        if anchor_error > 1.0e-6:
            return self._fault(
                "route anchor does not match its certified source pose",
                code="invalid_join_certificate",
                loaded=loaded,
                stop=True,
            )
        safety = loaded.manifest.get("safety_diagnostics")
        try:
            if not isinstance(safety, Mapping):
                raise ValueError("safety_diagnostics is missing")
            certified_clearance = float(safety["start_join_clearance_m"])
        except (KeyError, TypeError, ValueError) as exc:
            return self._fault(
                f"start join clearance is invalid: {exc}",
                code="invalid_join_certificate",
                loaded=loaded,
                stop=True,
            )
        if not math.isfinite(certified_clearance) or certified_clearance <= 0.0:
            return self._fault(
                "start join clearance must be finite and positive",
                code="invalid_join_certificate",
                loaded=loaded,
                stop=True,
            )
        try:
            egress_certificate = validate_start_egress_certificate(
                safety,
                tuple(full_waypoints),
                planned_start,
            )
        except ValueError as exc:
            return self._fault(
                f"start egress certificate is invalid: {exc}",
                code="invalid_egress_certificate",
                loaded=loaded,
                stop=True,
            )
        if same_installed_geometry:
            self.last_adopted_revision = loaded.route_revision
            fields = self._base_fields(loaded)
            fields.update(
                {
                    "heartbeat": True,
                    "installed_route_unchanged": True,
                    "adoption_robot_pose": {
                        "x_m": current_pose.x_m,
                        "y_m": current_pose.y_m,
                        "yaw_rad": current_pose.yaw_rad,
                    },
                }
            )
            return RouteUpdate(
                kind=RouteUpdateKind.UNCHANGED,
                reason="fresh heartbeat for installed route geometry",
                route_revision=loaded.route_revision,
                target_revision=loaded.target_revision,
                route_hash=loaded.route_hash,
                event_fields=fields,
            )
        join_index = 0
        join_distance = math.hypot(
            route_anchor.x_m - current_pose.x_m,
            route_anchor.y_m - current_pose.y_m,
        )
        effective_join_limit = min(self.max_join_distance_m, certified_clearance)
        if join_distance > effective_join_limit:
            return self._fault(
                f"certified route-start anchor is {join_distance:.3f}m away "
                f"(limit {effective_join_limit:.3f}m)",
                code="unsafe_route_join",
                loaded=loaded,
                stop=True,
                extra_fields={
                    "join_index": join_index,
                    "join_distance_m": join_distance,
                    "max_join_distance_m": self.max_join_distance_m,
                    "certified_start_join_clearance_m": certified_clearance,
                    "effective_join_limit_m": effective_join_limit,
                    "adoption_robot_pose": {
                        "x_m": current_pose.x_m,
                        "y_m": current_pose.y_m,
                        "yaw_rad": current_pose.yaw_rad,
                    },
                },
            )

        splice_forward_distance = 0.0
        if (
            not egress_certificate.required
            and join_distance > self.forward_splice_min_offset_m
        ):
            suffix, splice_forward_distance = forward_splice_waypoints(
                current_pose,
                tuple(full_waypoints),
                certified_radius_m=effective_join_limit,
            )
        else:
            suffix = tuple(full_waypoints)
        fields = self._base_fields(loaded)
        fields.update(
            {
                "route_kind": selected.route_kind,
                "join_index": join_index,
                "join_distance_m": join_distance,
                "max_join_distance_m": self.max_join_distance_m,
                "certified_start_join_clearance_m": certified_clearance,
                "effective_join_limit_m": effective_join_limit,
                "max_forward_window": self.max_forward_window,
                "full_waypoint_count": len(full_waypoints),
                "adopted_waypoint_count": len(suffix),
                "adopted_target_index": 0,
                "forward_splice": (
                    not egress_certificate.required
                    and join_distance > self.forward_splice_min_offset_m
                ),
                "splice_forward_distance_m": splice_forward_distance,
                "start_egress_vertex_lock": egress_certificate.required,
                "start_egress_waypoint_index": (
                    egress_certificate.waypoint_index
                ),
                "start_egress_continuous_clearance_validated": (
                    egress_certificate.required
                ),
                "start_egress_minimum_route_clearance_m": (
                    egress_certificate.minimum_route_clearance_m
                ),
                "adoption_robot_pose": {
                    "x_m": current_pose.x_m,
                    "y_m": current_pose.y_m,
                    "yaw_rad": current_pose.yaw_rad,
                },
                "requires_zero_cycle": True,
                "route_path": str(loaded.route_path),
            }
        )
        self.last_adopted_revision = loaded.route_revision
        self.last_route_hash = loaded.route_hash
        self.last_adopted_target = full_waypoints[-1]
        self._last_event_signature = None
        return RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=suffix,
            target_index=0,
            reason="new monotonic route revision joined through certified forward splice",
            route_revision=loaded.route_revision,
            target_revision=loaded.target_revision,
            route_hash=loaded.route_hash,
            requires_zero_cycle=True,
            event_name="dynamic_route_adopted",
            event_fields=fields,
        )


# Names retained for integration sites that describe the component by role.
DynamicRouteHandoff = DynamicRouteSource
RouteSource = DynamicRouteSource
