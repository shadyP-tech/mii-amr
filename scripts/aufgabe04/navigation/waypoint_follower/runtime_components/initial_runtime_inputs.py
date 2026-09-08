"""Stopped sensor/TF startup orchestration, separate from motion recovery."""

from __future__ import annotations

import time

from scripts.aufgabe04.navigation.control.waypoint_controller import VelocityCommand
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import RuntimeBindingProxy
from scripts.aufgabe04.navigation.waypoint_follower.initial_tf_acquisition import InitialTfAcquisition

rclpy = RuntimeBindingProxy("rclpy", None)


def wait_for_initial_runtime_inputs(node, started_at: float) -> str:
    """Use one listener continuously; publish only zero while acquiring inputs."""

    config = node.follower_config
    state = InitialTfAcquisition(
        started_at, config.initial_sensor_wait_sec,
        getattr(config, "initial_tf_acquisition_wait_sec", 0.0),
    )
    motion_failure = _motion_contract_failure(node, state)
    if motion_failure:
        return motion_failure

    context = getattr(node, "odom_execution_context", None)
    runtime = getattr(node, "runtime_config", None)
    target = context.odom_frame if context is not None else runtime.map_frame
    base = context.base_frame if context is not None else runtime.base_frame
    executor_probe = getattr(node, "initial_tf_executor_health_probe", None)
    last_failure = "missing scan"
    while rclpy.ok():
        node._service_or_wait_for_callbacks(0.05)
        motion_failure = _motion_contract_failure(node, state)
        if motion_failure:
            return motion_failure
        health = {} if executor_probe is None else executor_probe()
        state.executor_health = dict(health)
        if state.extension_used and state.acquisition_deadline_exhausted(time.monotonic()):
            return _finish_wait_failure(node, state, last_failure)
        scan_failure = node._freshness_failure(
            "scan", node.latest_scan, node.latest_scan_receipt, config.max_scan_age_sec,
        )
        odom_failure = ""
        if scan_failure:
            last_failure = scan_failure
        else:
            odom_failure = node._freshness_failure(
                "odom", node.latest_odom, node.latest_odom_receipt, config.max_odom_age_sec,
            )
            if odom_failure:
                last_failure = odom_failure
            else:
                lookup = node._current_pose_lookup()
                state.record_edge(
                    "execution_pose", target_frame=target, source_frame=base,
                    ready=lookup.pose is not None, details=lookup.details,
                )
                if lookup.pose is None:
                    last_failure = f"TF transform unavailable: {target} <- {base}"
                    node.latest_stop_details = {
                        **dict(lookup.details or {}), "stop_reason": last_failure,
                    }
                else:
                    # Acquiring odom<-base does not replace the map<-odom
                    # continuity check or any later certified-start gate.
                    localization_failure = node._global_consistency_monitor_failure()
                    if context is not None:
                        state.record_edge(
                            "global_consistency", target_frame=context.map_frame,
                            source_frame=context.odom_frame,
                            ready=not localization_failure,
                            details=node.latest_stop_details if localization_failure else None,
                        )
                    if localization_failure:
                        last_failure = localization_failure
                    elif executor_probe is not None and health.get("ready") is not True:
                        last_failure = "TF listener executor not ready"
                        node.latest_stop_details = {
                            "reason": last_failure, "source": "tf_executor_readiness",
                            "fail_closed": True,
                        }
                    else:
                        # TF calls may block briefly. A lookup begun within
                        # the extra phase must not admit motion after it ends.
                        if state.extension_used and state.acquisition_deadline_exhausted(time.monotonic()):
                            failure = "initial TF acquisition deadline exhausted"
                            node.latest_stop_details = {
                                "reason": failure, "source": "initial_tf_acquisition",
                                "fail_closed": True,
                            }
                            return _finish_wait_failure(node, state, failure)
                        node.latest_initial_tf_acquisition = state.to_evidence()
                        trace_failure = _trace(node, "initial_runtime_input_ready", state)
                        if trace_failure:
                            return trace_failure
                        node.latest_stop_details = None
                        return ""
        prior_phase = state.phase
        keep_waiting = state.can_continue(
            now=time.monotonic(), motion_published=getattr(node, "motion_published", False),
            sensors_fresh=not scan_failure and not odom_failure,
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
        **dict(node.latest_stop_details or {}),
        "initial_tf_acquisition": state.to_evidence(),
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
