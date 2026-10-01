"""One zero-held AMCL refresh for a new listener missing its first map edge.

This helper never admits a transform or grants motion. It only requests an
AMCL publication inside the caller's existing absolute startup deadline.
Every TF sample and frozen-frame continuity check stays in the startup loop.
"""

from __future__ import annotations

import math
import time

from scripts.aufgabe04.navigation.localization.tf_stale_recovery_policy import (
    StationarityLimits, evaluate_stationarity,
)
from scripts.aufgabe04.navigation.waypoint_follower.initial_tf_acquisition import (
    is_cold_execution_tf_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import RuntimeBindingProxy

Empty = RuntimeBindingProxy("Empty", None)


class InitialAmclRefresh:
    """Nonblocking service lifecycle; poll only under current startup gates."""

    def __init__(self):
        self.active = False
        self.future = None
        self.last_odom = None
        self.anchor_odom = None
        self.request_deadline = None
        self.response_map_attempt = None
        self.evidence = {
            "service_requested": False, "service_completed": False,
            "request_count": 0, "zero_held": True, "motion_authorized": False,
            "requires_fresh_tf_and_continuity": True,
        }

    def _eligible(self, node, state):
        runtime = getattr(node, "runtime_config", None)
        context = getattr(node, "odom_execution_context", None)
        global_edge = state.edges.get("global_consistency", {})
        execution = state.edges.get("execution_pose", {})
        return (
            state.phase == "cold_tf_acquisition" and context is not None
            and bool(context.certificate_sha256)
            and getattr(runtime, "localization_source", "") == "amcl"
            and getattr(runtime, "use_sim_time", True) is False
            and getattr(node, "motion_published", False) is False
            and state.sensor_inputs_fresh and state.executor_health.get("ready") is True
            and (not state.sensor_acquisition.executor_health
                 or state.sensor_acquisition.executor_health.get("ready") is True)
            and not state.admission_failure_seen
            and execution.get("current_ready") is True
            and not execution.get("non_acquisition_failure_seen", False)
            and global_edge.get("successful_sample_count") == 0
            and not global_edge.get("non_acquisition_failure_seen", False)
            and global_edge.get("target_frame") == context.map_frame
            and global_edge.get("source_frame") == context.odom_frame
            and is_cold_execution_tf_failure(global_edge.get("last_sample", {}))
        )

    def poll(self, node, state, now):
        """Return (terminal failure, hold even if TF is ready).

        An ack must be followed by another normal map sample. In particular,
        a service result never substitutes for the newly created TF buffer.
        """
        absolute_deadline = state.started_at + state.sensor_wait_sec + state.acquisition_wait_sec
        if now >= absolute_deadline:
            return "", self.active
        if not self.active:
            if not self._eligible(node, state):
                return "", False
            self.active = True
            self.evidence.update(
                status="confirming_stationarity", started_monotonic_sec=now,
                absolute_deadline_monotonic_sec=absolute_deadline,
                service_name=getattr(node, "runtime_nomotion_update_service", ""),
            )
        # Existing startup policy retains precedence for sensor, TF, executor,
        # and continuity failures; never request through a failed live gate.
        if (not state.sensor_inputs_fresh or state.executor_health.get("ready") is not True
                or state.admission_failure_seen
                or state.edges.get("execution_pose", {}).get("current_ready") is not True
                or getattr(node, "motion_published", False) is not False):
            return "", True
        node.publish_zero()
        ownership_failure = node._cmd_vel_ownership_failure()
        if ownership_failure:
            return self._fail(node, "cmd_vel_ownership_failed", ownership_failure)
        sample = node._odom_stationarity_sample()
        if sample is None:
            return self._fail(node, "stationarity_unavailable")
        if self.last_odom is None:
            self.last_odom = self.anchor_odom = sample
            return "", True
        limits = StationarityLimits(max_sample_age_sec=min(.5, node.follower_config.max_odom_age_sec))
        decision = evaluate_stationarity(self.last_odom, sample, now_sec=node._ros_now_sec(), limits=limits)
        self.evidence["stationarity"] = decision.to_log_dict()
        if not decision.accepted:
            waitable = {"odom_callback_not_advanced", "odom_stamp_not_advanced", "odom_sample_separation_too_short"}
            if set(decision.reasons) <= waitable:
                return "", True
            return self._fail(node, "robot_not_stationary", decision.reason)
        anchor = self.anchor_odom
        yaw_delta = abs(math.atan2(math.sin(sample.yaw_rad-anchor.yaw_rad), math.cos(sample.yaw_rad-anchor.yaw_rad)))
        if math.hypot(sample.x_m-anchor.x_m, sample.y_m-anchor.y_m) > limits.max_translation_m or yaw_delta > limits.max_yaw_rad:
            return self._fail(node, "robot_not_stationary", "cumulative odometry pose changed")
        self.last_odom = sample
        now = time.monotonic()
        if now >= absolute_deadline:
            return "", True
        if self.future is None:
            # Recheck cold-only eligibility immediately before the one write.
            # If TF appeared naturally, there is no need to request anything.
            if not self._eligible(node, state):
                self.evidence["status"] = "not_needed_tf_arrived"
                return "", False
            client = getattr(node, "runtime_nomotion_update_client", None)
            if client is None:
                return self._fail(node, "service_unavailable")
            if not client.service_is_ready():
                self.evidence["status"] = "waiting_for_service"
                return "", True
            freshness_failure = node._post_stale_tf_recovery_freshness_failure()
            if freshness_failure:
                return self._fail(node, "pre_request_sensor_freshness_failed", freshness_failure)
            now = time.monotonic()
            if now >= absolute_deadline:
                return "", True
            self.request_deadline = min(absolute_deadline,
                now + node.follower_config.runtime_nomotion_update_timeout_sec)
            self.evidence.update(status="service_pending", service_requested=True,
                request_count=1, request_monotonic_sec=now,
                request_deadline_monotonic_sec=self.request_deadline,
                stationarity_before_request=decision.to_log_dict())
            try:
                self.future = client.call_async(Empty.Request())
                if self.future is None:
                    raise RuntimeError("AMCL service returned no future")
            except Exception as exc:
                return self._fail(node, "service_request_failed", str(exc))
            return "", True
        if self.response_map_attempt is None:
            if now >= self.request_deadline:
                return self._fail(node, "service_timeout")
            try:
                if not self.future.done():
                    return "", True
                error = self.future.exception()
                if error is not None:
                    raise error
                if self.future.result() is None:
                    raise RuntimeError("AMCL service returned no response")
            except Exception as exc:
                return self._fail(node, "service_failed", str(exc))
            self.response_map_attempt = state.edges.get("global_consistency", {}).get("attempt_count", 0)
            self.evidence.update(status="awaiting_fresh_global_tf", service_completed=True,
                response_monotonic_sec=now)
            return "", True
        edge = state.edges.get("global_consistency", {})
        if edge.get("current_ready") is True and edge.get("attempt_count", 0) > self.response_map_attempt:
            self.evidence.update(status="fresh_tf_and_continuity_observed",
                admitted_global_sample=dict(edge["last_sample"]))
            return "", False
        return "", True

    def close(self):
        """Release a pending local future without retrying or undoing AMCL."""
        if self.future is not None:
            try:
                if not self.future.done():
                    self.future.cancel()
                    self.evidence["pending_future_cancelled"] = True
            except Exception as exc:
                self.evidence["future_cleanup_error"] = f"{type(exc).__name__}: {exc}"

    def _fail(self, node, code, error=""):
        reason = f"initial AMCL refresh failed: {code}"
        self.evidence.update(status="failed", failure_code=code, error=error)
        node.latest_stop_details = {
            "reason": reason, "stop_reason": reason,
            "source": "initial_amcl_refresh", "fault_code": code,
            "initial_amcl_refresh": dict(self.evidence), "fail_closed": True,
        }
        return reason, True
