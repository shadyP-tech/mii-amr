"""Bounded acquisition of an execution TF edge in a new, stopped listener.

The ordinary sensor deadline stays unchanged. One additional acquisition phase
is available only for never-acquired required TF edges, with fresh sensors and
a demonstrably serviced TF executor. Within an already-entered cold phase, a
structurally valid first stale global sample may wait for a fresh replacement
under the same deadline. This is neither runtime TF recovery nor localization
resealing; it cannot authorize a command or bypass admission.
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
    required_edges: tuple[str, ...] = ("execution_pose",)
    phase: str = field(default="initial_sensor_wait", init=False)
    edges: dict[str, dict[str, object]] = field(default_factory=dict, init=False)
    extension_used: bool = field(default=False, init=False)
    executor_health: dict[str, object] = field(default_factory=dict, init=False)
    denial_reason: str = field(default="", init=False)
    failed_edge_role: str = field(default="", init=False)
    sensor_inputs_fresh: bool = field(default=False, init=False)
    admission_failure_seen: bool = field(default=False, init=False)
    elapsed_sec: float = field(default=0.0, init=False)
    deadline_exhausted: bool = field(default=False, init=False)
    execution_context: dict[str, object] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if (not self.required_edges or len(set(self.required_edges)) != len(self.required_edges)
                or any(role not in ("execution_pose", "global_consistency") for role in self.required_edges)):
            raise ValueError("initial acquisition requires distinct configured TF roles")
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
            "non_acquisition_failure_count": 0, "waitable_stale_sample_count": 0,
        })
        edge["attempt_count"] += 1
        edge["current_ready"] = ready
        edge["last_sample"] = dict(details or {})
        if ready:
            edge["successful_sample_count"] += 1
            return
        failure = dict(details or {})
        cold = (
            is_cold_execution_tf_failure(failure)
            and failure["target_frame"] == target_frame
            and failure["source_frame"] == source_frame
        )
        waitable_stale = self._waitable_first_stale_global_sample(role, edge, failure)
        edge["non_acquisition_failure_seen"] |= not cold
        edge["non_acquisition_failure_count"] += not cold
        edge["waitable_stale_sample_count"] += waitable_stale
        edge["recent_failures"].append({
            "attempt": edge["attempt_count"],
            "reason": failure.get("reason", "missing_failure_details"),
            "exception_type": failure.get("exception_type"),
            "monitor_warning": failure.get("monitor_warning"),
            "age_sec": failure.get("age_sec"), "stamp_sec": failure.get("stamp_sec"),
            "structural_validation_passed": failure.get("structural_validation_passed"),
            "waitable_first_stale_global_sample": waitable_stale,
        })
        del edge["recent_failures"][:-8]

    def can_continue(
        self, *, now: float, motion_published: bool, sensors_fresh: bool,
        failure_details: Mapping[str, object], executor_health: Mapping[str, object],
    ) -> bool:
        self.executor_health = dict(executor_health)
        self.sensor_inputs_fresh = sensors_fresh
        self.elapsed_sec = max(0.0, now - self.started_at)
        if motion_published is not False:
            self.denial_reason = "motion_already_published"
            return False
        deadline = self.started_at + self.sensor_wait_sec
        if self.phase == "initial_sensor_wait" and now < deadline:
            return True
        failed_edges = [
            (role, edge) for role, edge in self.edges.items()
            if edge.get("current_ready") is not True
        ]
        matched_roles = [
            role for role, edge in failed_edges
            if failure_details.get("target_frame") == edge["target_frame"]
            and failure_details.get("source_frame") == edge["source_frame"]
        ]
        waitable_stale = (
            matched_roles == ["global_consistency"]
            and self._waitable_first_stale_global_sample(
                "global_consistency", self.edges["global_consistency"], failure_details,
            )
        )
        if self.acquisition_wait_sec <= 0:
            self.denial_reason = "cold_tf_acquisition_disabled"
        elif not sensors_fresh:
            self.denial_reason = "sensor_inputs_not_fresh"
        elif executor_health.get("ready") is not True:
            self.denial_reason = "tf_executor_not_ready"
        elif self.admission_failure_seen:
            self.denial_reason = "continuity_admission_failed"
        elif any(edge.get("successful_sample_count", 0) for _, edge in failed_edges):
            self.denial_reason = "required_tf_edge_already_acquired"
        elif any(
            edge.get("non_acquisition_failure_seen", False)
            and not (role == "global_consistency" and waitable_stale)
            for role, edge in self.edges.items()
        ):
            self.denial_reason = "required_tf_edge_has_non_acquisition_failure"
        elif not is_cold_execution_tf_failure(failure_details) and not waitable_stale:
            self.denial_reason = "failure_not_initial_tf_acquisition"
        elif len(matched_roles) != 1 or matched_roles[0] not in self.required_edges:
            self.denial_reason = "failure_not_required_tf_edge"
        elif now >= deadline + self.acquisition_wait_sec:
            self.denial_reason = "cold_tf_acquisition_deadline_exhausted"
            self.deadline_exhausted = True
        else:
            self.failed_edge_role = matched_roles[0]
            self.phase = "cold_tf_acquisition"
            self.extension_used = True
            return True
        return False

    def _waitable_first_stale_global_sample(self, role, edge, sample) -> bool:
        """Classify a stopped wait candidate; retain its non-cold history.

        Sensor/executor health, zero motion, admission and the absolute deadline
        are checked by ``can_continue``. A later missing sample cannot use this
        exception, nor can the stale history become cold-only reseal evidence.
        """

        context = self.execution_context
        execution = self.edges.get("execution_pose", {})
        execution_sample = execution.get("last_sample", {})
        if (
            role != "global_consistency" or self.phase != "cold_tf_acquisition"
            or not self.extension_used or not isinstance(context, Mapping)
            or set(self.required_edges) != {"execution_pose", "global_consistency"}
            or edge["successful_sample_count"] != 0
            or edge["non_acquisition_failure_count"] != edge["waitable_stale_sample_count"]
            or execution.get("current_ready") is not True
            or execution.get("successful_sample_count", 0) <= 0
            or execution.get("non_acquisition_failure_seen") is not False
            or execution.get("target_frame") != context.get("odom_frame")
            or execution.get("source_frame") != context.get("base_frame")
            or edge["target_frame"] != context.get("map_frame")
            or edge["source_frame"] != context.get("odom_frame")
        ):
            return False
        for current, target, source in (
            (sample, context.get("map_frame"), context.get("odom_frame")),
            (execution_sample, context.get("odom_frame"), context.get("base_frame")),
        ):
            if (current.get("source") != "tf_lookup"
                    or current.get("target_frame") != target or current.get("source_frame") != source
                    or not all(type(current.get(key)) in (int, float) and math.isfinite(current[key])
                               for key in ("stamp_sec", "age_sec", "max_age_sec", "max_future_sec"))
                    or current["stamp_sec"] < 0 or current["max_age_sec"] <= 0
                    or current["max_future_sec"] < 0):
                return False
        return (
            sample.get("reason") == "stale_transform"
            and sample.get("structural_validation_passed") is True
            and sample.get("available") is False and sample.get("validation_passed") is False
            and sample["age_sec"] > sample["max_age_sec"]
            and execution_sample.get("reason") == "fresh_transform"
            and execution_sample.get("available") is True
            and execution_sample.get("validation_passed") is True
            and -execution_sample["max_future_sec"] <= execution_sample["age_sec"] <= execution_sample["max_age_sec"]
        )

    def acquisition_deadline_exhausted(self, now: float) -> bool:
        """A callback wait cannot admit a sample after the extra phase expires."""

        self.elapsed_sec = max(0.0, now - self.started_at)
        if now >= (
            self.started_at + self.sensor_wait_sec + self.acquisition_wait_sec
        ):
            self.denial_reason = "cold_tf_acquisition_deadline_exhausted"
            self.deadline_exhausted = True
            return True
        return False

    def to_evidence(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "phase": self.phase,
            "initial_sensor_wait_sec": self.sensor_wait_sec,
            "initial_tf_acquisition_wait_sec": self.acquisition_wait_sec,
            "maximum_startup_wait_sec": self.sensor_wait_sec + self.acquisition_wait_sec,
            "extension_used": self.extension_used,
            "denial_reason": self.denial_reason,
            "required_edges": list(self.required_edges),
            "failed_edge_role": self.failed_edge_role,
            "sensor_inputs_fresh": self.sensor_inputs_fresh,
            "admission_failure_seen": self.admission_failure_seen,
            "elapsed_sec": self.elapsed_sec,
            "deadline_exhausted": self.deadline_exhausted,
            "execution_context": self.execution_context,
            "executor_health": dict(self.executor_health),
            "edges": {role: {**edge, "recent_failures": list(edge["recent_failures"]),
                             "last_sample": dict(edge["last_sample"])}
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
