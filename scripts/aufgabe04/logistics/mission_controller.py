"""Pure, fail-closed single-robot mission controller for Aufgabe 04.

This module performs no ROS, HTTP, filesystem, or motion I/O.  It turns a
strict QR observation plus a freshly validated server task into immutable
navigation dispatches.  The server order is copied once and cannot be changed
by retry/replan handling.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from scripts.aufgabe04.logistics.models import MissionState
from scripts.aufgabe04.logistics.server_validation.models import (
    ValidatedServerTask,
    server_order_sha256,
)
from scripts.aufgabe04.qr_scanning.events import (
    QRObservationEvent,
    QRValidationPolicy,
    StationIdentityRegistry,
    ValidatedQRObservation,
    validate_qr_observation,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MissionControlError(ValueError):
    """A rejected transition.  The controller never advances on this error."""


@dataclass(frozen=True)
class MissionControllerPolicy:
    qr_validation: QRValidationPolicy = QRValidationPolicy()
    max_status_age_sec: float = 30.0
    max_plan_age_sec: float = 300.0
    max_task_validation_age_sec: float = 5.0
    max_server_future_skew_sec: float = 0.1
    max_attempts_per_station: int = 3
    max_confirmation_rejections_per_station: int = 3

    def __post_init__(self) -> None:
        for field_name in (
            "max_status_age_sec",
            "max_plan_age_sec",
            "max_task_validation_age_sec",
            "max_server_future_skew_sec",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        if self.max_status_age_sec <= 0.0 or self.max_plan_age_sec <= 0.0:
            raise ValueError("server status and plan age limits must be positive")
        if self.max_task_validation_age_sec <= 0.0:
            raise ValueError("max_task_validation_age_sec must be positive")
        if isinstance(self.max_attempts_per_station, bool) or self.max_attempts_per_station <= 0:
            raise ValueError("max_attempts_per_station must be positive")
        if (
            isinstance(self.max_confirmation_rejections_per_station, bool)
            or self.max_confirmation_rejections_per_station <= 0
        ):
            raise ValueError("max_confirmation_rejections_per_station must be positive")


@dataclass(frozen=True)
class MissionDispatch:
    dispatch_id: str
    robot_id: str
    mission_id: str
    station_id: str
    station_index: int
    attempt_number: int
    issued_at_sec: float
    ordered_station_ids: Tuple[str, ...]
    server_order_sha256: str


@dataclass(frozen=True)
class MissionControllerSnapshot:
    state: MissionState
    robot_id: str
    mission_id: str
    ordered_station_ids: Tuple[str, ...]
    current_index: int
    current_attempt: int
    confirmation_rejections: int
    active_dispatch: Optional[MissionDispatch]
    accepted_event_ids: Tuple[str, ...]
    accepted_sample_ids: Tuple[str, ...]
    failure_reason: str

    @property
    def current_station_id(self) -> str:
        if self.current_index >= len(self.ordered_station_ids):
            return ""
        return self.ordered_station_ids[self.current_index]


class MissionController:
    """In-memory state machine that emits navigation-neutral dispatches."""

    def __init__(
        self,
        *,
        registry: StationIdentityRegistry,
        robot_id: str,
        policy: MissionControllerPolicy = MissionControllerPolicy(),
    ):
        if not robot_id.strip():
            raise ValueError("robot_id must not be empty")
        self._registry = registry
        self._robot_id = robot_id.strip()
        self._policy = policy
        self._state = MissionState.IDLE
        self._mission_id = ""
        self._ordered_station_ids: Tuple[str, ...] = ()
        self._server_order_sha256 = ""
        self._current_index = 0
        self._current_attempt = 0
        self._confirmation_rejections = 0
        self._active_dispatch: Optional[MissionDispatch] = None
        self._accepted_event_ids = set()
        self._accepted_sample_ids = set()
        self._dispatch_sequence = 0
        self._failure_reason = ""

    @property
    def snapshot(self) -> MissionControllerSnapshot:
        return MissionControllerSnapshot(
            state=self._state,
            robot_id=self._robot_id,
            mission_id=self._mission_id,
            ordered_station_ids=self._ordered_station_ids,
            current_index=self._current_index,
            current_attempt=self._current_attempt,
            confirmation_rejections=self._confirmation_rejections,
            active_dispatch=self._active_dispatch,
            accepted_event_ids=tuple(sorted(self._accepted_event_ids)),
            accepted_sample_ids=tuple(sorted(self._accepted_sample_ids)),
            failure_reason=self._failure_reason,
        )

    def begin(
        self,
        *,
        initial_qr: QRObservationEvent,
        server_task: ValidatedServerTask,
        now_sec: float,
    ) -> MissionDispatch:
        """Accept one fresh task and dispatch its first target exactly as ordered."""

        if self._state != MissionState.IDLE:
            raise MissionControlError("mission controller has already been started")
        now = self._validated_now(now_sec)
        validated_qr = self._validate_qr(initial_qr, now)
        canonical_order = self._validate_server_task(server_task, validated_qr, now)

        self._consume_qr(initial_qr)
        self._mission_id = server_task.mission_id
        self._ordered_station_ids = canonical_order
        self._server_order_sha256 = server_task.order_sha256
        self._current_index = 0
        self._current_attempt = 1
        self._confirmation_rejections = 0
        self._state = MissionState.NAVIGATING
        return self._issue_dispatch(now)

    def retry_current(
        self,
        *,
        dispatch_id: str,
        reason: str,
        now_sec: float,
    ) -> MissionDispatch:
        """Issue another dispatch for the same indexed station and immutable order."""

        self._require_active_dispatch(dispatch_id)
        if not reason.strip():
            raise MissionControlError("retry reason must not be empty")
        now = self._validated_now(now_sec)
        if self._current_attempt >= self._policy.max_attempts_per_station:
            self._fail(
                "navigation retry budget exhausted for "
                f"{self._ordered_station_ids[self._current_index]}: {reason.strip()}"
            )
            raise MissionControlError(self._failure_reason)
        self._current_attempt += 1
        return self._issue_dispatch(now)

    def confirm_arrival(
        self,
        *,
        dispatch_id: str,
        arrival_qr: QRObservationEvent,
        now_sec: float,
    ) -> Optional[MissionDispatch]:
        """Advance only when a new post-dispatch QR confirms the expected station."""

        active_dispatch = self._require_active_dispatch(dispatch_id)
        now = self._validated_now(now_sec)
        try:
            validated_qr = self._validate_qr(arrival_qr, now)
        except ValueError as exc:
            raise MissionControlError(str(exc)) from exc

        # Once its integrity/freshness is valid, an event is consumed even when
        # its station is wrong.  It can never be replayed after a later retry.
        self._consume_qr(arrival_qr)
        if (
            arrival_qr.observed_at_sec + self._policy.qr_validation.max_future_skew_sec
            < active_dispatch.issued_at_sec
        ):
            self._reject_confirmation("arrival QR predates the active dispatch")
        expected_station = self._ordered_station_ids[self._current_index]
        if validated_qr.identity.station_id != expected_station:
            self._reject_confirmation(
                "arrival QR confirms the wrong station: "
                f"expected {expected_station}, observed {validated_qr.identity.station_id}"
            )

        self._current_index += 1
        self._current_attempt = 0
        self._confirmation_rejections = 0
        self._active_dispatch = None
        if self._current_index >= len(self._ordered_station_ids):
            self._state = MissionState.COMPLETED
            return None

        self._current_attempt = 1
        return self._issue_dispatch(now)

    def abort(self, reason: str) -> None:
        if self._state in (MissionState.COMPLETED, MissionState.FAILED):
            raise MissionControlError(f"cannot abort mission in state {self._state.value}")
        if not reason.strip():
            raise MissionControlError("abort reason must not be empty")
        self._fail(reason.strip())

    def _validate_qr(self, event: QRObservationEvent, now_sec: float) -> ValidatedQRObservation:
        validated = validate_qr_observation(
            event,
            registry=self._registry,
            now_sec=now_sec,
            policy=self._policy.qr_validation,
            expected_robot_id=self._robot_id,
            seen_event_ids=self._accepted_event_ids,
        )
        repeated_samples = sorted(
            set(event.consensus.sample_ids) & self._accepted_sample_ids
        )
        if repeated_samples:
            raise ValueError(
                "replayed QR consensus sample_id(s): "
                + ", ".join(repeated_samples)
            )
        return validated

    def _consume_qr(self, event: QRObservationEvent) -> None:
        """Consume both the event envelope and every contributing camera frame."""

        self._accepted_event_ids.add(event.event_id)
        self._accepted_sample_ids.update(event.consensus.sample_ids)

    def _validate_server_task(
        self,
        task: ValidatedServerTask,
        qr: ValidatedQRObservation,
        now_sec: float,
    ) -> Tuple[str, ...]:
        if task.robot_id != self._robot_id:
            raise MissionControlError("server task robot_id does not match the controller")
        if not task.mission_id.strip():
            raise MissionControlError("server task mission_id must not be empty")
        if not task.ordered_station_ids:
            raise MissionControlError("server task has no validated ordered_station_ids")
        timestamps = {
            "status_observed_at_sec": task.status_observed_at_sec,
            "plan_generated_at_sec": task.plan_generated_at_sec,
            "validated_at_sec": task.validated_at_sec,
        }
        normalized_timestamps = {}
        for field_name, value in timestamps.items():
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MissionControlError(f"server task is missing {field_name}")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise MissionControlError(f"server task {field_name} is invalid")
            if value > now_sec + self._policy.max_server_future_skew_sec:
                raise MissionControlError(f"server task {field_name} is in the future")
            normalized_timestamps[field_name] = value
        if now_sec - normalized_timestamps["status_observed_at_sec"] > self._policy.max_status_age_sec:
            raise MissionControlError("server task status is stale")
        if now_sec - normalized_timestamps["plan_generated_at_sec"] > self._policy.max_plan_age_sec:
            raise MissionControlError("server task plan is stale")
        if now_sec - normalized_timestamps["validated_at_sec"] > self._policy.max_task_validation_age_sec:
            raise MissionControlError("server task validation is stale")
        if (
            normalized_timestamps["validated_at_sec"] + self._policy.max_server_future_skew_sec
            < normalized_timestamps["status_observed_at_sec"]
            or normalized_timestamps["validated_at_sec"] + self._policy.max_server_future_skew_sec
            < normalized_timestamps["plan_generated_at_sec"]
        ):
            raise MissionControlError("server task was validated before its source snapshots")
        if not _SHA256_RE.fullmatch(task.source_plan_sha256):
            raise MissionControlError("server task source_plan_sha256 is missing or invalid")
        expected_order_digest = server_order_sha256(
            robot_id=task.robot_id,
            mission_id=task.mission_id,
            target_station=task.target_station,
            plan_step_index=task.plan_step_index,
            ordered_station_ids=task.ordered_station_ids,
            plan_generated_at_sec=normalized_timestamps["plan_generated_at_sec"],
        )
        if task.order_sha256 != expected_order_digest:
            raise MissionControlError("server task order_sha256 does not match its ordered stations")

        try:
            current_identity = self._registry.resolve(task.resolved_current_station)
            canonical_order = self._registry.canonical_station_order(task.ordered_station_ids)
            canonical_target = self._registry.resolve(task.target_station).station_id
        except ValueError as exc:
            raise MissionControlError(str(exc)) from exc
        if qr.identity.station_id != current_identity.station_id:
            raise MissionControlError(
                "initial QR does not identify the server-resolved current station: "
                f"expected {current_identity.station_id}, observed {qr.identity.station_id}"
            )
        if canonical_order[0] != canonical_target:
            raise MissionControlError("server target is not the first station in the remaining order")
        return canonical_order

    def _issue_dispatch(self, now_sec: float) -> MissionDispatch:
        self._dispatch_sequence += 1
        station_id = self._ordered_station_ids[self._current_index]
        dispatch_id = (
            f"{self._mission_id}:{self._current_index}:{self._current_attempt}:"
            f"{self._dispatch_sequence}:{self._server_order_sha256[:12]}"
        )
        dispatch = MissionDispatch(
            dispatch_id=dispatch_id,
            robot_id=self._robot_id,
            mission_id=self._mission_id,
            station_id=station_id,
            station_index=self._current_index,
            attempt_number=self._current_attempt,
            issued_at_sec=now_sec,
            ordered_station_ids=self._ordered_station_ids,
            server_order_sha256=self._server_order_sha256,
        )
        self._active_dispatch = dispatch
        return dispatch

    def _reject_confirmation(self, reason: str) -> None:
        self._confirmation_rejections += 1
        if self._confirmation_rejections >= self._policy.max_confirmation_rejections_per_station:
            self._fail(f"arrival confirmation budget exhausted: {reason}")
            raise MissionControlError(self._failure_reason)
        raise MissionControlError(reason)

    def _require_navigating(self) -> None:
        if self._state != MissionState.NAVIGATING or self._active_dispatch is None:
            raise MissionControlError(f"mission is not awaiting arrival confirmation: {self._state.value}")

    def _require_active_dispatch(self, dispatch_id: str) -> MissionDispatch:
        self._require_navigating()
        active_dispatch = self._active_dispatch
        if active_dispatch is None:
            raise MissionControlError("mission has no active navigation dispatch")
        if dispatch_id != active_dispatch.dispatch_id:
            raise MissionControlError(
                "dispatch_id does not match the active navigation dispatch: "
                f"expected {active_dispatch.dispatch_id}, received {dispatch_id}"
            )
        return active_dispatch

    def _fail(self, reason: str) -> None:
        self._state = MissionState.FAILED
        self._failure_reason = reason
        self._active_dispatch = None

    @staticmethod
    def _validated_now(now_sec: float) -> float:
        if isinstance(now_sec, bool) or not isinstance(now_sec, (int, float)):
            raise MissionControlError("now_sec must be numeric")
        now = float(now_sec)
        if not math.isfinite(now) or now < 0.0:
            raise MissionControlError("now_sec must be finite and non-negative")
        return now
