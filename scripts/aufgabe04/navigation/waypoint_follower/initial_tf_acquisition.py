"""Bounded acquisition of an execution TF edge in a new, stopped listener.

The ordinary sensor deadline stays unchanged. One additional acquisition phase
is available only for a never-acquired execution edge, with fresh sensors and
a demonstrably serviced TF executor. This is neither runtime TF recovery nor
localization resealing; it cannot authorize a command or bypass admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Mapping


DEFAULT_INITIAL_TF_ACQUISITION_WAIT_SEC = 3.0
TF_EXECUTOR_HEARTBEAT_PERIOD_SEC = 0.05
TF_EXECUTOR_HEARTBEAT_MAX_AGE_SEC = 0.5
_COLD_EXCEPTION_TYPES = frozenset({"LookupException", "ConnectivityException"})


class TfExecutorHeartbeat:
    """Evidence that the isolated executor services its own timer callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._last_callback_sec: float | None = None

    def tick(self) -> None:
        with self._lock:
            self._count += 1
            self._last_callback_sec = time.monotonic()

    def snapshot(self, *, thread_alive: bool) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            age = None if self._last_callback_sec is None else now - self._last_callback_sec
            count = self._count
        ready = (
            thread_alive is True and count > 0 and age is not None
            and 0.0 <= age <= TF_EXECUTOR_HEARTBEAT_MAX_AGE_SEC
        )
        return {
            "ready": ready,
            "thread_alive": thread_alive,
            "heartbeat_count": count,
            "heartbeat_age_sec": age,
            "heartbeat_max_age_sec": TF_EXECUTOR_HEARTBEAT_MAX_AGE_SEC,
            "tf_delivery_proven": False,
        }


@dataclass
class InitialTfAcquisition:
    """ROS-free state and evidence for one initial runtime-input wait."""

    started_at: float
    sensor_wait_sec: float
    acquisition_wait_sec: float
    phase: str = field(default="initial_sensor_wait", init=False)
    edges: dict[str, dict[str, object]] = field(default_factory=dict, init=False)
    extension_used: bool = field(default=False, init=False)
    executor_health: dict[str, object] = field(default_factory=dict, init=False)
    denial_reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if type(self.started_at) not in (int, float) or not math.isfinite(self.started_at):
            raise ValueError("startup acquisition start must be finite")
        if (type(self.sensor_wait_sec) not in (int, float)
                or not math.isfinite(self.sensor_wait_sec) or self.sensor_wait_sec <= 0):
            raise ValueError("initial sensor wait must be finite and positive")
        if (type(self.acquisition_wait_sec) not in (int, float)
                or not math.isfinite(self.acquisition_wait_sec)
                or not 0 <= self.acquisition_wait_sec <= DEFAULT_INITIAL_TF_ACQUISITION_WAIT_SEC):
            raise ValueError("initial TF acquisition wait must be between 0 and 3 seconds")

    def record_edge(
        self, role: str, *, target_frame: str, source_frame: str,
        ready: bool, details: Mapping[str, object] | None,
    ) -> None:
        edge = self.edges.setdefault(role, {
            "target_frame": target_frame, "source_frame": source_frame,
            "attempt_count": 0, "successful_sample_count": 0,
            "non_acquisition_failure_seen": False, "recent_failures": [],
        })
        edge["attempt_count"] += 1
        if ready:
            edge["successful_sample_count"] += 1
            return
        failure = dict(details or {})
        cold = (
            is_cold_execution_tf_failure(failure)
            and failure["target_frame"] == target_frame
            and failure["source_frame"] == source_frame
        )
        edge["non_acquisition_failure_seen"] |= not cold
        edge["recent_failures"].append({
            "attempt": edge["attempt_count"],
            "reason": failure.get("reason", "missing_failure_details"),
            "exception_type": failure.get("exception_type"),
            "monitor_warning": failure.get("monitor_warning"),
        })
        del edge["recent_failures"][:-8]

    def can_continue(
        self, *, now: float, motion_published: bool, sensors_fresh: bool,
        failure_details: Mapping[str, object], executor_health: Mapping[str, object],
    ) -> bool:
        self.executor_health = dict(executor_health)
        if motion_published is not False:
            self.denial_reason = "motion_already_published"
            return False
        deadline = self.started_at + self.sensor_wait_sec
        if self.phase == "initial_sensor_wait" and now < deadline:
            return True
        edge = self.edges.get("execution_pose", {})
        if self.acquisition_wait_sec <= 0:
            self.denial_reason = "cold_tf_acquisition_disabled"
        elif not sensors_fresh:
            self.denial_reason = "sensor_inputs_not_fresh"
        elif executor_health.get("ready") is not True:
            self.denial_reason = "tf_executor_not_ready"
        elif edge.get("successful_sample_count", 0):
            self.denial_reason = "execution_edge_already_acquired"
        elif edge.get("non_acquisition_failure_seen", False):
            self.denial_reason = "execution_edge_has_non_acquisition_failure"
        elif not is_cold_execution_tf_failure(failure_details):
            self.denial_reason = "failure_not_initial_execution_tf_acquisition"
        elif (failure_details.get("target_frame") != edge.get("target_frame")
                or failure_details.get("source_frame") != edge.get("source_frame")):
            self.denial_reason = "failure_not_execution_tf_edge"
        elif now >= deadline + self.acquisition_wait_sec:
            self.denial_reason = "cold_tf_acquisition_deadline_exhausted"
        else:
            self.phase = "cold_tf_acquisition"
            self.extension_used = True
            return True
        return False

    def acquisition_deadline_exhausted(self, now: float) -> bool:
        """A callback wait cannot admit a sample after the extra phase expires."""

        if self.extension_used and now >= (
            self.started_at + self.sensor_wait_sec + self.acquisition_wait_sec
        ):
            self.denial_reason = "cold_tf_acquisition_deadline_exhausted"
            return True
        return False

    def to_evidence(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "initial_sensor_wait_sec": self.sensor_wait_sec,
            "initial_tf_acquisition_wait_sec": self.acquisition_wait_sec,
            "maximum_startup_wait_sec": self.sensor_wait_sec + self.acquisition_wait_sec,
            "extension_used": self.extension_used,
            "denial_reason": self.denial_reason,
            "executor_health": dict(self.executor_health),
            "edges": {role: {**edge, "recent_failures": list(edge["recent_failures"])}
                      for role, edge in self.edges.items()},
            "motion_authorized": False,
            "fresh_sensor_and_localization_admission_required": True,
        }


def is_cold_execution_tf_failure(details: Mapping[str, object]) -> bool:
    return (
        details.get("source") == "tf_lookup"
        and details.get("reason") == "lookup_exception"
        and details.get("exception_type") in _COLD_EXCEPTION_TYPES
        and isinstance(details.get("target_frame"), str)
        and bool(details["target_frame"].strip("/"))
        and isinstance(details.get("source_frame"), str)
        and bool(details["source_frame"].strip("/"))
    )
