"""Callback-service orchestration for the waypoint-follower runtime."""

from __future__ import annotations

import math
import time

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None

from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import (
    RuntimeBindingProxy,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.constants import (
    CALLBACK_SERVICE_BACKGROUND_EXECUTOR,
)

rclpy = RuntimeBindingProxy("rclpy", rclpy)


class CallbackServiceRuntimeMixin:
    """Callback-drain behavior mixed into the sole follower node."""

    def _drain_runtime_callbacks(
        self,
        max_callbacks: int = 12,
        *,
        max_duration_sec: float | None = None,
        spin_timeout_sec: float = 0.0,
    ) -> dict[str, object]:
        """Service callbacks in caller mode or wait for the background executor.

        Production runs use continuously spinning follower and TF executors.
        Their ordinary control-loop drain is therefore an immediate no-op.  A
        bounded stale-TF recovery instead waits its full safety window while
        the follower executor services scan/odometry/clock and the isolated TF
        executor services TF subscriptions.  ``spin_count`` remains scoped to
        caller spins performed here; it never claims work done by either
        background executor.  The caller-spin branch remains only for ROS-free
        focused tests and direct node use outside
        :func:`run_simple_waypoint_follower`.
        """
        if (
            not isinstance(max_callbacks, int)
            or isinstance(max_callbacks, bool)
            or max_callbacks <= 0
        ):
            raise ValueError("max_callbacks must be a positive integer")
        if max_duration_sec is not None and (
            not math.isfinite(max_duration_sec) or max_duration_sec <= 0.0
        ):
            raise ValueError("max_duration_sec must be finite and positive")
        if not math.isfinite(spin_timeout_sec) or spin_timeout_sec < 0.0:
            raise ValueError("spin_timeout_sec must be finite and non-negative")

        started_at = time.monotonic()
        if self.callback_service_mode == CALLBACK_SERVICE_BACKGROUND_EXECUTOR:
            waited_for_background_callbacks = max_duration_sec is not None
            if waited_for_background_callbacks:
                time.sleep(max_duration_sec)
            elapsed_sec = time.monotonic() - started_at
            return {
                "callback_service_mode": self.callback_service_mode,
                "spin_count": 0,
                "elapsed_sec": elapsed_sec,
                "max_callbacks": max_callbacks,
                "max_duration_sec": max_duration_sec,
                "spin_timeout_sec": spin_timeout_sec,
                "deadline_reached": waited_for_background_callbacks,
                "background_wait_requested_sec": (
                    max_duration_sec if waited_for_background_callbacks else 0.0
                ),
            }

        spin_count = 0
        deadline_reached = False
        for _ in range(max_callbacks):
            elapsed_sec = time.monotonic() - started_at
            if (
                max_duration_sec is not None
                and elapsed_sec >= max_duration_sec
            ):
                deadline_reached = True
                break
            timeout_sec = spin_timeout_sec
            if max_duration_sec is not None:
                timeout_sec = min(
                    timeout_sec,
                    max(0.0, max_duration_sec - elapsed_sec),
                )
            rclpy.spin_once(self, timeout_sec=timeout_sec)
            spin_count += 1
        elapsed_sec = time.monotonic() - started_at
        return {
            "callback_service_mode": self.callback_service_mode,
            "spin_count": spin_count,
            "elapsed_sec": elapsed_sec,
            "max_callbacks": max_callbacks,
            "max_duration_sec": max_duration_sec,
            "spin_timeout_sec": spin_timeout_sec,
            "deadline_reached": deadline_reached,
        }
