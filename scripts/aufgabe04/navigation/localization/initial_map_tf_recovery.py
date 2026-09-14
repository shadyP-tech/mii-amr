"""Validate stopped, certificate-bound initial global-TF recovery evidence.

This admits only a new localization/route preparation attempt. It does not
authorize motion or turn an established transform failure into startup work.
"""

from __future__ import annotations

import math
from typing import Mapping

from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    OdomExecutionContext,
    validate_map_odom_continuity_evidence,
)
from scripts.aufgabe04.navigation.localization.map_odom_drift_reference import RouteDriftAnchor


def initial_map_tf_recovery_error(
    details: Mapping[str, object], *, context: OdomExecutionContext | None = None,
) -> str:
    """Return a diagnostic rejection reason, or empty for complete evidence."""

    if details.get("continuity") is not None:
        return "unexpected_continuity_for_initial_map_tf"
    return _initial_tf_report_error(details, drift=False, execution_context=context)


def initial_tf_drift_report_error(
    details: Mapping[str, object], *, context: OdomExecutionContext | None = None,
) -> str:
    """Require a current-schema drift report to match its sampled context."""

    return _initial_tf_report_error(details, drift=True, execution_context=context)


def _initial_tf_report_error(
    details: Mapping[str, object], *, drift: bool,
    execution_context: OdomExecutionContext | None,
) -> str:
    expected = {
        "execution_phase": "before_motion", "phase": "initial_runtime_input_wait",
        "motion_published": False, "fail_closed": True,
    }
    if not drift:
        expected.update(source="tf_lookup", reason="lookup_exception", available=False, validation_passed=False)
    if any(not _exact(details.get(k), v) for k, v in expected.items()):
        return "invalid_initial_map_tf_stop"
    state = details.get("initial_tf_acquisition")
    if not isinstance(state, Mapping):
        return "invalid_initial_tf_acquisition"
    required = {
        "schema_version": 2, "failed_edge_role": "global_consistency",
        "sensor_inputs_fresh": True, "admission_failure_seen": drift,
        "motion_authorized": False,
        "fresh_sensor_and_localization_admission_required": True,
    }
    if any(not _exact(state.get(k), v) for k, v in required.items()):
        return "invalid_initial_map_tf_acquisition_state"
    if (state.get("phase") not in ("initial_sensor_wait", "cold_tf_acquisition")
            or type(state.get("extension_used")) is not bool
            or type(state.get("deadline_exhausted")) is not bool):
        return "invalid_initial_tf_acquisition_phase"
    if not drift and (state.get("deadline_exhausted") is not True
                      or state.get("denial_reason") != "cold_tf_acquisition_deadline_exhausted"):
        return "initial_map_tf_deadline_not_exhausted"
    if drift and state.get("denial_reason") not in (
        "continuity_admission_failed", "cold_tf_acquisition_deadline_exhausted",
    ):
        return "invalid_initial_tf_drift_admission"
    if _budget_error(state, exhausted=not drift):
        return "invalid_initial_map_tf_budget"
    health = state.get("executor_health")
    if (not isinstance(health, Mapping) or health.get("ready") is not True
            or health.get("thread_alive") is not True
            or not _positive_count(health.get("heartbeat_count"))
            or not _finite(health.get("heartbeat_age_sec"))
            or not _finite(health.get("heartbeat_max_age_sec"))
            or not 0 <= health["heartbeat_age_sec"] <= health["heartbeat_max_age_sec"] <= 0.5):
        return "initial_map_tf_executor_not_ready"
    identity = state.get("execution_context")
    if not isinstance(identity, Mapping):
        return "initial_map_tf_execution_context_missing"
    fields = {
        "map_frame", "odom_frame", "base_frame", "certificate_sha256",
        "frozen_map_from_odom", "max_map_from_odom_translation_drift_m",
        "max_map_from_odom_yaw_drift_rad",
    }
    if set(identity) not in (fields, fields | {"drift_reference"}):
        return "invalid_initial_map_tf_execution_context"
    try:
        transform = identity["frozen_map_from_odom"]
        if not isinstance(transform, Mapping):
            return "invalid_initial_map_tf_execution_context"
        context = OdomExecutionContext(
            map_frame=identity["map_frame"], odom_frame=identity["odom_frame"],
            base_frame=identity["base_frame"], certificate_sha256=identity["certificate_sha256"],
            frozen_map_from_odom=PlanarTransform2D(**transform),
            max_map_from_odom_translation_drift_m=identity["max_map_from_odom_translation_drift_m"],
            max_map_from_odom_yaw_drift_rad=identity["max_map_from_odom_yaw_drift_rad"],
            drift_reference=(
                RouteDriftAnchor.from_evidence(identity["drift_reference"])
                if "drift_reference" in identity else None
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return "invalid_initial_map_tf_execution_context"
    if execution_context is not None and context != execution_context:
        return "initial_map_tf_execution_context_mismatch"
    if drift:
        continuity = details.get("continuity")
        if not isinstance(continuity, Mapping):
            return "initial_tf_drift_continuity_missing"
        for name in ("certificate_sha256", "map_frame", "odom_frame", "base_frame"):
            if continuity.get(name) != getattr(context, name):
                return "initial_tf_drift_context_mismatch"
        try:
            frozen = PlanarTransform2D(**continuity["frozen_map_from_odom"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return "initial_tf_drift_context_mismatch"
        if (frozen != context.frozen_map_from_odom
                or continuity.get("max_translation_drift_m") != context.max_map_from_odom_translation_drift_m
                or continuity.get("max_yaw_drift_rad") != context.max_map_from_odom_yaw_drift_rad):
            return "initial_tf_drift_context_mismatch"
        if (context.drift_reference is not None or "drift_reference" in continuity
                or continuity.get("schema_version") != 1):
            try:
                validate_map_odom_continuity_evidence(continuity, context=context)
            except (KeyError, TypeError, ValueError, OverflowError):
                return "initial_tf_drift_context_mismatch"
    edges = state.get("edges")
    required_edges = state.get("required_edges")
    if (not isinstance(edges, Mapping)
            or not isinstance(required_edges, (list, tuple))
            or len(required_edges) != 2
            or not all(isinstance(role, str) for role in required_edges)
            or set(required_edges) != {"execution_pose", "global_consistency"}
            or set(edges) != set(required_edges)):
        return "invalid_initial_map_tf_required_edges"
    for role, target, source, ready in (
        ("execution_pose", context.odom_frame, context.base_frame, True),
        ("global_consistency", context.map_frame, context.odom_frame, drift),
    ):
        edge = edges[role]
        if (not isinstance(edge, Mapping)
                or edge.get("target_frame") != target or edge.get("source_frame") != source
                or edge.get("current_ready") is not ready
                or edge.get("non_acquisition_failure_seen") is not False
                or not _positive_count(edge.get("attempt_count"))):
            return "invalid_initial_map_tf_edge_history"
        counters = ("non_acquisition_failure_count", "waitable_stale_sample_count")
        # Older schema-2 evidence has neither counter. New evidence must not
        # conceal a stale-input history behind a contradictory false flag.
        if any(name in edge for name in counters) and any(
            type(edge.get(name)) is not int or edge[name] != 0 for name in counters
        ):
            return "invalid_initial_map_tf_acquisition_counters"
        successes = edge.get("successful_sample_count")
        if (type(successes) is not int
                or not 0 <= successes <= edge["attempt_count"]
                or (successes > 0) is not ready):
            return "invalid_initial_map_tf_edge_acquisition"
        sample = edge.get("last_sample")
        if (not isinstance(sample, Mapping)
                or sample.get("target_frame") != target or sample.get("source_frame") != source
                or sample.get("available") is not ready
                or sample.get("validation_passed") is not ready):
            return "invalid_initial_map_tf_last_sample"
        if ready:
            if not _fresh_sample(sample):
                return "initial_map_tf_execution_sample_not_fresh"
        elif not _cold_sample(sample) or any(
            details.get(key) != sample.get(key)
            for key in (
                "source", "reason", "exception_type", "exception", "target_frame", "source_frame",
                "stamp_sec", "age_sec", "max_age_sec", "max_future_sec",
            )
        ):
            return "initial_map_tf_failure_not_exact_cold_edge"
    return ""


def _cold_sample(sample: Mapping[str, object]) -> bool:
    return (
        sample.get("source") == "tf_lookup"
        and sample.get("reason") == "lookup_exception"
        and sample.get("exception_type") in ("LookupException", "ConnectivityException")
        and sample.get("stamp_sec") is None and sample.get("age_sec") is None
        and _finite(sample.get("max_age_sec")) and sample["max_age_sec"] > 0
        and _finite(sample.get("max_future_sec")) and sample["max_future_sec"] >= 0
    )


def _fresh_sample(sample: Mapping[str, object]) -> bool:
    return (
        sample.get("source") == "tf_lookup" and sample.get("reason") == "fresh_transform"
        and all(_finite(sample.get(k)) for k in ("stamp_sec", "age_sec", "max_age_sec", "max_future_sec"))
        and sample["stamp_sec"] >= 0 and sample["max_age_sec"] > 0 and sample["max_future_sec"] >= 0
        and -sample["max_future_sec"] <= sample["age_sec"] <= sample["max_age_sec"]
    )


def _budget_error(state: Mapping[str, object], *, exhausted: bool) -> bool:
    names = ("initial_sensor_wait_sec", "initial_tf_acquisition_wait_sec", "maximum_startup_wait_sec", "elapsed_sec")
    if not all(_finite(state.get(k)) for k in names):
        return True
    sensor, extra, maximum, elapsed = (state[k] for k in names)
    return not (sensor > 0 and 0 <= extra <= 3.0
                and math.isclose(maximum, sensor + extra, rel_tol=0, abs_tol=1e-9)
                and elapsed >= 0 and (not exhausted or extra > 0 and elapsed >= maximum))


def _positive_count(value: object) -> bool:
    return type(value) is int and value > 0


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _exact(value: object, expected: object) -> bool:
    return type(value) is type(expected) and value == expected
