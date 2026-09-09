"""Stopped sensor/TF startup orchestration, separate from motion recovery."""

from __future__ import annotations

import time

from scripts.aufgabe04.navigation.control.waypoint_controller import VelocityCommand
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import RuntimeBindingProxy
from scripts.aufgabe04.navigation.waypoint_follower.initial_tf_acquisition import InitialTfAcquisition
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.tf_sampling import (
    refresh_tf_sample_age,
    refreshed_tf_sample_details,
)

rclpy = RuntimeBindingProxy("rclpy", None)


def wait_for_initial_runtime_inputs(node, started_at: float) -> str:
    """Acquire all required edges in this listener under one zero-motion budget."""

    config = node.follower_config
    context = getattr(node, "odom_execution_context", None)
    state = InitialTfAcquisition(
        started_at, config.initial_sensor_wait_sec,
        getattr(config, "initial_tf_acquisition_wait_sec", 0.0),
        required_edges=("execution_pose", "global_consistency") if context is not None else ("execution_pose",),
    )
    if context is not None:
        frozen = context.frozen_map_from_odom
        state.execution_context = {
            "certificate_sha256": context.certificate_sha256,
            "map_frame": context.map_frame, "odom_frame": context.odom_frame,
            "base_frame": context.base_frame,
            "frozen_map_from_odom": {"x_m": frozen.x_m, "y_m": frozen.y_m, "yaw_rad": frozen.yaw_rad},
            "max_map_from_odom_translation_drift_m": context.max_map_from_odom_translation_drift_m,
            "max_map_from_odom_yaw_drift_rad": context.max_map_from_odom_yaw_drift_rad,
        }
    motion_failure = _motion_contract_failure(node, state)
    if motion_failure:
        return motion_failure
    node.publish_zero()
    executor_probe = getattr(node, "initial_tf_executor_health_probe", None)
    last_failure = "initial TF acquisition deadline exhausted"
    while rclpy.ok():
        node._service_or_wait_for_callbacks(0.05)
        motion_failure = _motion_contract_failure(node, state)
        if motion_failure:
            return motion_failure
        health = {} if executor_probe is None else executor_probe()
        state.executor_health = dict(health)
        sensor_failure = _sensor_failure(node)
        state.sensor_inputs_fresh = not sensor_failure
        if state.acquisition_deadline_exhausted(time.monotonic()):
            age_failure = "" if sensor_failure else _recheck_ready_edge_ages(node, state)
            return _finish_wait_failure(node, state, sensor_failure or age_failure or last_failure)
        failure = sensor_failure or _sample_initial_edges(node, state)
        if not sensor_failure:
            # Failed TF lookups can also block. Extra acquisition eligibility
            # requires current sensor/executor evidence, not the earlier probe.
            motion_failure = _motion_contract_failure(node, state)
            if motion_failure:
                return motion_failure
            sensor_failure = _sensor_failure(node)
            state.sensor_inputs_fresh = not sensor_failure
            health = {} if executor_probe is None else executor_probe()
            state.executor_health = dict(health)
            failure = sensor_failure or failure
            if not sensor_failure and failure:
                # A stale first global sample may consume only the remaining
                # cold-acquisition budget. Its already-acquired execution edge
                # must still be fresh after the failed lookup and live probes.
                failure = _recheck_ready_edge_ages(node, state) or failure
        if not failure:
            # A lookup can block. Recheck the live admission inputs after both
            # samples/continuity checks and before reporting startup ready.
            motion_failure = _motion_contract_failure(node, state)
            if motion_failure:
                return motion_failure
            failure = _sensor_failure(node)
            state.sensor_inputs_fresh = not failure
            health = {} if executor_probe is None else executor_probe()
            state.executor_health = dict(health)
            if state.acquisition_deadline_exhausted(time.monotonic()):
                if not failure:
                    failure = "initial TF acquisition deadline exhausted"
                    node.latest_stop_details = {
                        "reason": failure, "source": "initial_tf_acquisition", "fail_closed": True,
                    }
                return _finish_wait_failure(node, state, failure)
            if not failure and executor_probe is not None and health.get("ready") is not True:
                failure = "TF listener executor not ready"
                node.latest_stop_details = {
                    "reason": failure, "source": "tf_executor_readiness", "fail_closed": True,
                }
            if not failure:
                failure = _recheck_ready_edge_ages(node, state)
            if not failure:
                node.latest_initial_tf_acquisition = state.to_evidence()
                trace_failure = _trace(node, "initial_runtime_input_ready", state)
                if trace_failure:
                    return trace_failure
                node.latest_stop_details = None
                return ""
        last_failure = failure
        prior_phase = state.phase
        keep_waiting = state.can_continue(
            now=time.monotonic(), motion_published=getattr(node, "motion_published", False),
            sensors_fresh=state.sensor_inputs_fresh,
            failure_details=dict(node.latest_stop_details or {}), executor_health=health,
        )
        node.latest_initial_tf_acquisition = state.to_evidence()
        if not keep_waiting:
            return _finish_wait_failure(node, state, last_failure)
        if state.phase != prior_phase:
            trace_failure = _trace(node, "initial_tf_acquisition_started", state)
            if trace_failure:
                return trace_failure
        node.publish_zero()
    return "ROS shutdown"


def _sensor_failure(node) -> str:
    config = node.follower_config
    scan_failure = node._freshness_failure(
        "scan", node.latest_scan, node.latest_scan_receipt, config.max_scan_age_sec,
    )
    return scan_failure or node._freshness_failure(
        "odom", node.latest_odom, node.latest_odom_receipt, config.max_odom_age_sec,
    )


def _sample_initial_edges(node, state: InitialTfAcquisition) -> str:
    context = getattr(node, "odom_execution_context", None)
    runtime = getattr(node, "runtime_config", None)
    target = context.odom_frame if context is not None else runtime.map_frame
    base = context.base_frame if context is not None else runtime.base_frame
    samples = [("execution_pose", target, base, node._current_pose_lookup())]
    if context is not None:
        samples.append(("global_consistency", context.map_frame, context.odom_frame, node._map_from_odom_lookup()))
    samples = [(role, target, source, refresh_tf_sample_age(node, lookup))
               for role, target, source, lookup in samples]
    for role, target, source, lookup in samples:
        details = dict(lookup.details or {})
        if lookup.stamp_sec is not None:
            details["stamp_sec"] = lookup.stamp_sec
        state.record_edge(role, target_frame=target, source_frame=source,
                          ready=lookup.pose is not None, details=details)
    if context is not None and samples[1][3].pose is not None:
        # This is the exact map edge sampled above, not a second lookup that
        # could hide a missing/invalid input or compare a different transform.
        failure = node._global_consistency_monitor_failure(map_lookup=samples[1][3])
        if failure:
            state.admission_failure_seen = True
            state.failed_edge_role = "global_consistency"
            return failure
    failed = [(role, target, source, lookup) for role, target, source, lookup in samples if lookup.pose is None]
    if failed:
        # Surface established-edge loss/invalid input ahead of another edge's
        # cold absence; the policy independently checks every edge's history.
        failed.sort(key=lambda row: not (
            state.edges[row[0]]["successful_sample_count"]
            or state.edges[row[0]]["non_acquisition_failure_seen"]
        ))
        role, target, source, lookup = failed[0]
        state.failed_edge_role = role
        failure = f"TF transform unavailable: {target} <- {source}"
        node.latest_stop_details = {
            **state.edges[role]["last_sample"], "stop_reason": failure, "fail_closed": True,
        }
        return failure
    state.failed_edge_role = ""
    return ""


def _recheck_ready_edge_ages(node, state: InitialTfAcquisition) -> str:
    for role, edge in state.edges.items():
        if edge.get("current_ready") is not True:
            continue
        sample = edge["last_sample"]
        updated = refreshed_tf_sample_details(node, sample, sample.get("stamp_sec"))
        edge["last_sample"] = updated
        if updated["available"] is not True:
            state.record_edge(role, target_frame=edge["target_frame"], source_frame=edge["source_frame"],
                              ready=False, details=updated)
            state.failed_edge_role = role
            node.latest_stop_details = {**updated, "fail_closed": True}
            return updated["stop_reason"]
    return ""


def _motion_contract_failure(node, state: InitialTfAcquisition) -> str:
    if getattr(node, "motion_published", False) is False:
        return ""
    state.denial_reason = "motion_already_published"
    failure = "initial runtime input acquisition requested after motion"
    node.publish_zero()
    node.latest_stop_details = {
        "source": "initial_runtime_input_wait", "reason": failure,
        "execution_phase": "after_motion", "motion_published": True,
        "fail_closed": True,
    }
    return _finish_wait_failure(node, state, failure)


def _finish_wait_failure(node, state: InitialTfAcquisition, failure: str) -> str:
    node.latest_initial_tf_acquisition = state.to_evidence()
    node.latest_stop_details = {
        "reason": failure, "source": "initial_runtime_input_wait",
        **dict(node.latest_stop_details or {}),
        "initial_tf_acquisition": state.to_evidence(),
        "fail_closed": True,
    }
    return failure


def _trace(node, event: str, state: InitialTfAcquisition) -> str:
    if getattr(node, "controller_trace_writer", None) is None:
        return ""
    failure = node._append_controller_trace(
        event=event, reason=state.phase, fail_closed=False,
        effective_command=VelocityCommand(0.0, 0.0),
        diagnostics={"initial_tf_acquisition": state.to_evidence()},
    )
    if failure:
        node.latest_stop_details = {
            **dict(node.latest_stop_details or {}),
            "reason": failure, "source": "controller_trace",
            "initial_tf_acquisition": state.to_evidence(), "fail_closed": True,
        }
    return failure
