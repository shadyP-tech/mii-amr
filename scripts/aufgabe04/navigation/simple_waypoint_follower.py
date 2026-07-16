"""Minimal ROS2 waypoint follower for one Aufgabe 04 route segment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.follower_safety import (
    NO_VALID_FRONT_SECTOR_SCAN_RANGES,
    OBSTACLE_TOO_CLOSE,
    cmd_vel_ownership_failure,
    front_sector_decision,
    initial_pose_failure,
    linear_scale_for_front_clearance,
    message_freshness_failure,
    obstacle_decision,
    stuck_progress_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.ros_runtime_config import ResolvedRuntimeConfig
from scripts.aufgabe04.navigation.waypoint_controller import (
    ControllerConfig,
    compute_join_anchor_command,
    compute_waypoint_command,
    reverse_staging_is_preferred,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Twist = None
    LaserScan = None
    Odometry = None
    Duration = None
    Node = object
    Parameter = None
    qos_profile_sensor_data = None
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


@dataclass(frozen=True)
class FollowerConfig:
    controller: ControllerConfig
    min_obstacle_distance_m: float = 0.20
    front_obstacle_slow_distance_m: float = 0.38
    front_obstacle_sector_rad: float = math.radians(35.0)
    max_scan_age_sec: float = 1.0
    max_odom_age_sec: float = 1.0
    max_tf_age_sec: float = 1.0
    initial_sensor_wait_sec: float = 2.0
    waypoint_timeout_sec: float = 45.0
    stuck_timeout_sec: float = 8.0
    stuck_progress_epsilon_m: float = 0.03
    initial_distance_limit_m: float = 0.35
    control_rate_hz: float = 10.0
    allowed_cmd_vel_publishers: Sequence[str] = ()
    dynamic_route_refresh_sec: float = 0.0
    dynamic_join_tolerance_m: float = 0.01
    initial_route_kind: str = ""
    axis_acquisition_wait_timeout_sec: float = 12.0
    viewpoint_sampling_timeout_sec: float = 30.0
    viewpoint_sampling_goal_tolerance_m: float = 0.01
    viewpoint_sampling_heading_tolerance_rad: float = math.radians(5.0)
    physical_goal_tolerance_m: float = 0.03

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.dynamic_join_tolerance_m)
            or self.dynamic_join_tolerance_m <= 0.0
        ):
            raise ValueError("dynamic_join_tolerance_m must be finite and positive")
        if (
            not math.isfinite(self.axis_acquisition_wait_timeout_sec)
            or self.axis_acquisition_wait_timeout_sec <= 0.0
        ):
            raise ValueError(
                "axis_acquisition_wait_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_timeout_sec)
            or self.viewpoint_sampling_timeout_sec <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_goal_tolerance_m)
            or self.viewpoint_sampling_goal_tolerance_m <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_goal_tolerance_m must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_heading_tolerance_rad)
            or self.viewpoint_sampling_heading_tolerance_rad <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_heading_tolerance_rad must be finite and positive"
            )
        if (
            not math.isfinite(self.physical_goal_tolerance_m)
            or self.physical_goal_tolerance_m <= 0.0
        ):
            raise ValueError("physical_goal_tolerance_m must be finite and positive")


INTERMEDIATE_ROUTE_KINDS = frozenset(
    {"axis_acquisition", "viewpoint_sampling"}
)
PHYSICAL_ROUTE_KINDS = frozenset(
    {"synchronized_face_approach", "synchronized_viewpoint"}
)
DYNAMIC_VIEWPOINT_ROUTE_KINDS = INTERMEDIATE_ROUTE_KINDS | PHYSICAL_ROUTE_KINDS


def controller_config_for_route_kind(
    config: ControllerConfig,
    route_kind: str,
    *,
    reverse_staging: bool = False,
    viewpoint_sampling_goal_tolerance_m: float | None = None,
    viewpoint_sampling_heading_tolerance_rad: float | None = None,
    physical_goal_tolerance_m: float | None = None,
) -> ControllerConfig:
    """Apply terminal-heading constraints only to physical face approaches.

    Acquisition and viewpoint-sampling waypoints may carry a finite terminal
    yaw, but enforcing that yaw throughout translation conflicts with pursuit
    of the geometric segment.  The normal final-position alignment still
    enforces the yaw once the sampling point has actually been reached.
    """

    if route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        return config
    physical = route_kind in PHYSICAL_ROUTE_KINDS
    goal_tolerance = config.goal_tolerance_m
    if (
        route_kind == "viewpoint_sampling"
        and viewpoint_sampling_goal_tolerance_m is not None
    ):
        goal_tolerance = min(goal_tolerance, viewpoint_sampling_goal_tolerance_m)
    if physical and physical_goal_tolerance_m is not None:
        goal_tolerance = min(goal_tolerance, physical_goal_tolerance_m)
    heading_tolerance = config.heading_tolerance_rad
    if (
        route_kind == "viewpoint_sampling"
        and viewpoint_sampling_heading_tolerance_rad is not None
    ):
        heading_tolerance = min(
            heading_tolerance, viewpoint_sampling_heading_tolerance_rad
        )
    return replace(
        config,
        goal_tolerance_m=goal_tolerance,
        heading_tolerance_rad=heading_tolerance,
        enforce_heading_corridor=physical,
        reverse_staging=physical and reverse_staging,
    )


def dynamic_route_kind_transition_failure(
    current_route_kind: str, next_route_kind: str
) -> str:
    """Validate monotonic acquisition -> sampling -> physical handoffs."""

    if not next_route_kind:
        return "missing dynamic route kind"
    if next_route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        return f"unknown dynamic route kind: {next_route_kind}"
    if current_route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        return f"unknown current dynamic route kind: {current_route_kind or '<missing>'}"
    if current_route_kind == next_route_kind:
        return ""
    if current_route_kind == "axis_acquisition" and next_route_kind in (
        {"viewpoint_sampling"} | PHYSICAL_ROUTE_KINDS
    ):
        return ""
    if current_route_kind == "viewpoint_sampling" and next_route_kind in PHYSICAL_ROUTE_KINDS:
        return ""
    if current_route_kind in PHYSICAL_ROUTE_KINDS and next_route_kind in PHYSICAL_ROUTE_KINDS:
        return ""
    return (
        "backward dynamic route phase transition: "
        f"{current_route_kind}->{next_route_kind}"
    )


def viewpoint_sampling_timeout_failure(
    *,
    route_kind: str,
    phase_started_at: float | None,
    now_monotonic: float,
    timeout_sec: float,
) -> str:
    if route_kind != "viewpoint_sampling":
        return ""
    if phase_started_at is None:
        return "viewpoint_sampling_clock_unavailable"
    if now_monotonic - phase_started_at >= timeout_sec:
        return "viewpoint_sampling_timeout"
    return ""


def acquisition_goal_action(
    *,
    route_kind: str,
    provider_available: bool,
    hold_elapsed_sec: float,
    timeout_sec: float,
) -> str:
    """Decide whether a geometrically reached route is mission-terminal."""

    if route_kind not in INTERMEDIATE_ROUTE_KINDS:
        return "complete"
    if not provider_available:
        return "missing_dynamic_route_provider"
    if route_kind == "axis_acquisition" and hold_elapsed_sec >= timeout_sec:
        return "axis_acquisition_timeout"
    return "hold_for_physical_face"


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _node_identity(endpoint) -> str:
    namespace = getattr(endpoint, "node_namespace", "") or ""
    name = getattr(endpoint, "node_name", "") or ""
    return _format_node_identity(namespace, name)


def _format_node_identity(namespace: str, name: str) -> str:
    if namespace in ("", "/"):
        return f"/{name}"
    return f"{namespace.rstrip('/')}/{name}"


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

@dataclass(frozen=True)
class PoseLookupResult:
    pose: Pose2D | None
    details: dict[str, object] | None = None


def tf_lookup_failure_details(
    *,
    reason: str,
    target_frame: str,
    source_frame: str,
    max_age_sec: float,
    age_sec: float | None = None,
    exception: BaseException | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "map-to-base transform unavailable",
        "source": "tf_lookup",
        "reason": reason,
        "target_frame": target_frame,
        "source_frame": source_frame,
        "max_age_sec": max_age_sec,
    }
    if age_sec is not None:
        payload["age_sec"] = age_sec
    if exception is not None:
        payload["exception_type"] = exception.__class__.__name__
        payload["exception"] = str(exception)
    return payload


def dynamic_join_envelope_failure(
    pose: Pose2D,
    anchor: Pose2D,
    effective_join_limit_m: float | None,
) -> dict[str, object] | None:
    """Fail closed if a live pose leaves or cannot define the certified join disk."""

    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)):
        return {
            "reason": "current robot pose is non-finite during dynamic-route join",
            "fault_code": "invalid_current_pose",
            "fail_closed": True,
        }
    if (
        effective_join_limit_m is None
        or not math.isfinite(effective_join_limit_m)
        or effective_join_limit_m <= 0.0
    ):
        return {
            "reason": "dynamic-route join envelope is invalid",
            "fault_code": "invalid_route_update",
            "fail_closed": True,
            "effective_join_limit_m": effective_join_limit_m,
        }
    join_distance = math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m)
    if not math.isfinite(join_distance):
        return {
            "reason": "dynamic-route join distance is non-finite",
            "fault_code": "invalid_current_pose",
            "fail_closed": True,
        }
    if join_distance > effective_join_limit_m:
        return {
            "reason": "robot left the certified dynamic-route join envelope",
            "fault_code": "join_envelope_exceeded",
            "fail_closed": True,
            "join_distance_m": join_distance,
            "effective_join_limit_m": effective_join_limit_m,
        }
    return None


def stuck_progress_details(
    *,
    target_index: int,
    distance_to_target_m: float,
    last_progress_distance_m: float,
    elapsed_without_progress_sec: float,
    max_without_progress_sec: float,
    progress_epsilon_m: float,
    commanded_linear_x_mps: float,
    commanded_angular_z_radps: float,
    front_clearance_scale: float,
    effective_linear_x_mps: float,
    front_clearance_details: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "stuck no progress",
        "source": "progress_monitor",
        "target_index": target_index,
        "distance_to_target_m": distance_to_target_m,
        "last_progress_distance_m": last_progress_distance_m,
        "elapsed_without_progress_sec": elapsed_without_progress_sec,
        "max_without_progress_sec": max_without_progress_sec,
        "progress_epsilon_m": progress_epsilon_m,
        "commanded_linear_x_mps": commanded_linear_x_mps,
        "commanded_angular_z_radps": commanded_angular_z_radps,
        "front_clearance_scale": front_clearance_scale,
        "effective_linear_x_mps": effective_linear_x_mps,
    }
    if front_clearance_details is not None:
        payload["front_clearance"] = front_clearance_details
    return payload


class SimpleWaypointFollowerNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(
        self,
        runtime_config: ResolvedRuntimeConfig,
        waypoints: Sequence[Pose2D],
        follower_config: FollowerConfig,
        waypoint_provider: Callable[[Pose2D], RouteUpdate | None] | None = None,
        route_update_callback: Callable[[RouteUpdate], None] | None = None,
    ) -> None:
        super().__init__("aufgabe04_simple_waypoint_follower")
        self.runtime_config = runtime_config
        self.waypoints = tuple(waypoints)
        self.follower_config = follower_config
        self.waypoint_provider = waypoint_provider
        self.route_update_callback = route_update_callback
        self.last_route_refresh_at = 0.0
        self.initial_route_refresh_pending = waypoint_provider is not None
        self.dynamic_join_pending = False
        self.dynamic_join_limit_m: float | None = None
        self.current_route_kind = follower_config.initial_route_kind
        self.reverse_staging = False
        self.axis_acquisition_hold_started_at: float | None = None
        self.viewpoint_sampling_started_at: float | None = None
        self.target_index = 0
        self.latest_scan = None
        self.latest_scan_receipt = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.motion_published = False
        self.distance_estimate_m = 0.0
        self.last_pose = None
        self.last_progress_distance_m = math.inf
        self.last_progress_at = time.monotonic()
        self.target_started_at = time.monotonic()
        self.latest_stop_details = None
        self.latest_front_clearance_details = None
        self._configure_sim_time(runtime_config.use_sim_time)

        self.cmd_vel_pub = self.create_publisher(Twist, runtime_config.cmd_vel_topic, 10)
        self.create_subscription(
            LaserScan,
            runtime_config.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(Odometry, runtime_config.odom_topic, self._odom_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _configure_sim_time(self, use_sim_time: bool) -> None:
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", use_sim_time)
            return
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)])

    def _scan_callback(self, msg) -> None:
        self.latest_scan = msg
        # Transport freshness belongs to the process monotonic clock.  A new
        # simulation node can receive a scan before its first /clock callback;
        # recording that receipt with ROS time would store zero and then look
        # thousands of seconds stale as soon as simulated time activates.
        self.latest_scan_receipt = time.monotonic()

    def _odom_callback(self, msg) -> None:
        self.latest_odom = msg
        self.latest_odom_receipt = time.monotonic()

    def publish_zero(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def publish_repeated_zero(self, count: int = 5) -> None:
        for _ in range(count):
            self.publish_zero()
            rclpy.spin_once(self, timeout_sec=0.02)

    def _drain_runtime_callbacks(self, max_callbacks: int = 12) -> None:
        """Service all currently ready sensor/TF callbacks without blocking.

        A single ``spin_once`` per controller tick can permanently starve TF
        when scan and odometry publishers run faster than the controller.  A
        bounded drain keeps the control period finite while allowing the TF
        listener to consume the newest transform.
        """
        for _ in range(max_callbacks):
            rclpy.spin_once(self, timeout_sec=0.0)

    def run(self) -> FollowerResult:
        if len(self.waypoints) < 2:
            return FollowerResult("noop", "fewer than two waypoints", 0.0, 0.0, False)
        started_at = time.monotonic()
        if self.current_route_kind == "viewpoint_sampling":
            self.viewpoint_sampling_started_at = started_at
        self.publish_repeated_zero()
        startup_failure = self._wait_for_initial_runtime_inputs(started_at)
        if startup_failure:
            self.publish_repeated_zero()
            return FollowerResult(
                "stopped",
                startup_failure,
                time.monotonic() - started_at,
                self.distance_estimate_m,
                self.motion_published,
            )
        loop_sleep_sec = 1.0 / max(self.follower_config.control_rate_hz, 1.0)
        try:
            while rclpy.ok():
                self._drain_runtime_callbacks()
                safety_failure = self._safety_failure()
                if safety_failure:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        safety_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                pose_lookup = self._current_pose_lookup()
                pose = pose_lookup.pose
                if pose is None:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        "map-to-base transform unavailable",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        pose_lookup.details,
                    )
                route_refresh = self._refresh_dynamic_route(pose)
                if route_refresh == "stopped":
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        str((self.latest_stop_details or {}).get("reason", "dynamic route withdrawn")),
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                sampling_timeout = viewpoint_sampling_timeout_failure(
                    route_kind=self.current_route_kind,
                    phase_started_at=self.viewpoint_sampling_started_at,
                    now_monotonic=time.monotonic(),
                    timeout_sec=self.follower_config.viewpoint_sampling_timeout_sec,
                )
                if sampling_timeout:
                    self.latest_stop_details = {
                        "reason": sampling_timeout,
                        "route_kind": self.current_route_kind,
                        "phase_elapsed_sec": (
                            None
                            if self.viewpoint_sampling_started_at is None
                            else time.monotonic()
                            - self.viewpoint_sampling_started_at
                        ),
                        "timeout_sec": self.follower_config.viewpoint_sampling_timeout_sec,
                        "fail_closed": True,
                    }
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        sampling_timeout.replace("_", " "),
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                if route_refresh == "adopted":
                    # A verified handoff still gets one complete zero-command
                    # control period before the new route may command motion.
                    self.publish_zero()
                    time.sleep(loop_sleep_sec)
                    continue
                if self.last_pose is not None:
                    self.distance_estimate_m += math.hypot(
                        pose.x_m - self.last_pose.x_m,
                        pose.y_m - self.last_pose.y_m,
                    )
                self.last_pose = pose
                if self.target_index == 0:
                    initial_failure = initial_pose_failure(
                        pose,
                        self.waypoints[0],
                        self.follower_config.initial_distance_limit_m,
                    )
                    if initial_failure:
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            initial_failure,
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                        )
                if self.dynamic_join_pending:
                    join_failure = dynamic_join_envelope_failure(
                        pose,
                        self.waypoints[0],
                        self.dynamic_join_limit_m,
                    )
                    if join_failure is not None:
                        self.latest_stop_details = join_failure
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            str(join_failure["reason"]),
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                    join_distance = math.hypot(
                        pose.x_m - self.waypoints[0].x_m,
                        pose.y_m - self.waypoints[0].y_m,
                    )
                    if join_distance <= self.follower_config.dynamic_join_tolerance_m:
                        self.dynamic_join_pending = False
                        self.dynamic_join_limit_m = None
                        self.target_index = 0
                        self.target_started_at = time.monotonic()
                        self.last_progress_distance_m = math.inf
                        self.last_progress_at = time.monotonic()
                        self.publish_zero()
                        time.sleep(loop_sleep_sec)
                        continue
                    # During handoff, pursue only the collision-certified route
                    # start.  Normal progress advancement/lookahead would form
                    # an unchecked chord from the live pose to waypoint 1.
                    step = compute_join_anchor_command(
                        pose,
                        self.waypoints[0],
                        controller_config_for_route_kind(
                            self.follower_config.controller,
                            self.current_route_kind,
                            reverse_staging=self.reverse_staging,
                            viewpoint_sampling_goal_tolerance_m=(
                                self.follower_config.viewpoint_sampling_goal_tolerance_m
                            ),
                            viewpoint_sampling_heading_tolerance_rad=(
                                self.follower_config.viewpoint_sampling_heading_tolerance_rad
                            ),
                            physical_goal_tolerance_m=(
                                self.follower_config.physical_goal_tolerance_m
                            ),
                        ),
                        join_tolerance_m=self.follower_config.dynamic_join_tolerance_m,
                    )
                else:
                    step = compute_waypoint_command(
                        pose,
                        self.waypoints,
                        self.target_index,
                        controller_config_for_route_kind(
                            self.follower_config.controller,
                            self.current_route_kind,
                            reverse_staging=self.reverse_staging,
                            viewpoint_sampling_goal_tolerance_m=(
                                self.follower_config.viewpoint_sampling_goal_tolerance_m
                            ),
                            viewpoint_sampling_heading_tolerance_rad=(
                                self.follower_config.viewpoint_sampling_heading_tolerance_rad
                            ),
                            physical_goal_tolerance_m=(
                                self.follower_config.physical_goal_tolerance_m
                            ),
                        ),
                    )
                if step.target_index != self.target_index:
                    self.target_index = step.target_index
                    self.target_started_at = time.monotonic()
                    self.last_progress_distance_m = math.inf
                    self.last_progress_at = time.monotonic()
                if step.reached_goal:
                    now_monotonic = time.monotonic()
                    if self.axis_acquisition_hold_started_at is None:
                        self.axis_acquisition_hold_started_at = now_monotonic
                    hold_elapsed = (
                        now_monotonic - self.axis_acquisition_hold_started_at
                    )
                    goal_action = acquisition_goal_action(
                        route_kind=self.current_route_kind,
                        provider_available=self.waypoint_provider is not None,
                        hold_elapsed_sec=hold_elapsed,
                        timeout_sec=(
                            self.follower_config.axis_acquisition_wait_timeout_sec
                        ),
                    )
                    if goal_action == "hold_for_physical_face":
                        # Remain stationary but keep spinning sensor callbacks
                        # and polling the manifest. A physical-face revision
                        # will be adopted at the top of a subsequent cycle.
                        self.publish_zero()
                        time.sleep(loop_sleep_sec)
                        continue
                    if goal_action != "complete":
                        self.latest_stop_details = {
                            "reason": goal_action,
                            "route_kind": self.current_route_kind,
                            "hold_elapsed_sec": hold_elapsed,
                            "timeout_sec": (
                                self.follower_config.axis_acquisition_wait_timeout_sec
                            ),
                            "fail_closed": True,
                        }
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            goal_action.replace("_", " "),
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "completed",
                        "",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                timeout_failure = waypoint_timeout_failure(
                    time.monotonic() - self.target_started_at,
                    self.follower_config.waypoint_timeout_sec,
                )
                if timeout_failure:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        timeout_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                now_monotonic = time.monotonic()
                front_clearance_scale = self._motion_clearance_linear_scale(
                    step.command.linear_x_mps
                )
                effective_linear_x_mps = step.command.linear_x_mps * front_clearance_scale
                progress_failure = self._progress_failure(
                    step.distance_to_target_m,
                    now_monotonic,
                    abs(step.command.linear_x_mps) > 0.0,
                )
                if progress_failure:
                    self.latest_stop_details = stuck_progress_details(
                        target_index=self.target_index,
                        distance_to_target_m=step.distance_to_target_m,
                        last_progress_distance_m=self.last_progress_distance_m,
                        elapsed_without_progress_sec=now_monotonic - self.last_progress_at,
                        max_without_progress_sec=self.follower_config.stuck_timeout_sec,
                        progress_epsilon_m=self.follower_config.stuck_progress_epsilon_m,
                        commanded_linear_x_mps=step.command.linear_x_mps,
                        commanded_angular_z_radps=step.command.angular_z_radps,
                        front_clearance_scale=front_clearance_scale,
                        effective_linear_x_mps=effective_linear_x_mps,
                        front_clearance_details=self.latest_front_clearance_details,
                    )
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        progress_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                twist = Twist()
                twist.linear.x = effective_linear_x_mps
                twist.angular.z = step.command.angular_z_radps
                self.cmd_vel_pub.publish(twist)
                self.motion_published = self.motion_published or abs(twist.linear.x) > 0.0 or abs(twist.angular.z) > 0.0
                time.sleep(loop_sleep_sec)
        finally:
            self.publish_repeated_zero()

    def _refresh_dynamic_route(self, pose: Pose2D) -> str:
        if self.waypoint_provider is None:
            return ""
        now = time.monotonic()
        initial_refresh = self.initial_route_refresh_pending
        if (
            not initial_refresh
            and self.follower_config.dynamic_route_refresh_sec <= 0.0
        ):
            return ""
        if (
            not initial_refresh
            and now - self.last_route_refresh_at
            < self.follower_config.dynamic_route_refresh_sec
        ):
            return ""
        self.initial_route_refresh_pending = False
        self.last_route_refresh_at = now
        try:
            update = self.waypoint_provider(pose)
        except Exception as exc:
            self.latest_stop_details = {
                "reason": f"dynamic route provider failed: {exc}",
                "fault_code": "route_provider_exception",
                "fail_closed": True,
            }
            return "stopped"
        if update is None:
            return ""
        if update.kind is RouteUpdateKind.UNCHANGED:
            return ""
        if update.kind is RouteUpdateKind.REJECT:
            self.publish_zero()
            if not self._emit_route_update(update):
                return "stopped"
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "dynamic route update rejected",
                "fail_closed": True,
            }
            return "stopped"
        if update.kind is RouteUpdateKind.STOP:
            # Zero first: semantic logging is synchronous and must never leave
            # the previous nonzero Twist active if it blocks or raises.
            self.publish_zero()
            if not self._emit_route_update(update):
                return "stopped"
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "dynamic route withdrawn",
            }
            return "stopped"
        replacement = tuple(update.waypoints)
        if update.kind is not RouteUpdateKind.ADOPT or len(replacement) < 2:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption contained fewer than two waypoints",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        if update.target_index is None or not 0 <= update.target_index < len(replacement):
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption contained an invalid target index",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        try:
            join_limit = float(update.event_fields["effective_join_limit_m"])
        except (KeyError, TypeError, ValueError) as exc:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": f"dynamic route adoption lacks a valid join envelope: {exc}",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        if not math.isfinite(join_limit) or join_limit <= 0.0:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption join envelope is not positive and finite",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        next_route_kind = str(update.event_fields.get("route_kind", ""))
        phase_failure = dynamic_route_kind_transition_failure(
            self.current_route_kind, next_route_kind
        )
        if phase_failure:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": phase_failure,
                "fault_code": "invalid_route_phase",
                "current_route_kind": self.current_route_kind,
                "next_route_kind": next_route_kind,
                "fail_closed": True,
            }
            return "stopped"
        previous_route_kind = self.current_route_kind
        self.publish_zero()
        self.waypoints = replacement
        self.current_route_kind = next_route_kind
        self.reverse_staging = (
            next_route_kind in PHYSICAL_ROUTE_KINDS
            and reverse_staging_is_preferred(pose, replacement)
        )
        if next_route_kind in PHYSICAL_ROUTE_KINDS:
            update = replace(
                update,
                event_fields={
                    **dict(update.event_fields),
                    "staging_motion": (
                        "reverse" if self.reverse_staging else "forward"
                    ),
                    "physical_goal_tolerance_m": (
                        self.follower_config.physical_goal_tolerance_m
                    ),
                },
            )
        if next_route_kind != previous_route_kind:
            self.axis_acquisition_hold_started_at = None
            self.viewpoint_sampling_started_at = (
                now if next_route_kind == "viewpoint_sampling" else None
            )
        self.target_index = update.target_index
        self.target_started_at = now
        self.last_progress_distance_m = math.inf
        self.last_progress_at = now
        self.last_pose = pose
        self.dynamic_join_pending = True
        self.dynamic_join_limit_m = join_limit
        if not self._emit_route_update(update):
            return "stopped"
        return "adopted"

    def _emit_route_update(self, update: RouteUpdate) -> bool:
        if update.event_name is None or self.route_update_callback is None:
            return True
        try:
            self.route_update_callback(update)
        except Exception as exc:
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": f"dynamic route event callback failed: {exc}",
                "fault_code": "route_event_callback_exception",
                "fail_closed": True,
            }
            return False
        return True

    def _wait_for_initial_runtime_inputs(self, started_at: float) -> str:
        deadline = started_at + self.follower_config.initial_sensor_wait_sec
        last_failure = "missing scan"
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            scan_failure = self._freshness_failure(
                "scan",
                self.latest_scan,
                self.latest_scan_receipt,
                self.follower_config.max_scan_age_sec,
            )
            if scan_failure:
                last_failure = scan_failure
            else:
                odom_failure = self._freshness_failure(
                    "odom",
                    self.latest_odom,
                    self.latest_odom_receipt,
                    self.follower_config.max_odom_age_sec,
                )
                if odom_failure:
                    last_failure = odom_failure
                else:
                    pose_lookup = self._current_pose_lookup()
                    if pose_lookup.pose is None:
                        self.latest_stop_details = pose_lookup.details
                        last_failure = "map-to-base transform unavailable"
                    else:
                        return ""
            if time.monotonic() >= deadline:
                return last_failure
            self.publish_zero()
        return "ROS shutdown"

    def _safety_failure(self) -> str:
        self.latest_stop_details = None
        scan_failure = self._freshness_failure("scan", self.latest_scan, self.latest_scan_receipt, self.follower_config.max_scan_age_sec)
        if scan_failure:
            return scan_failure
        odom_failure = self._freshness_failure("odom", self.latest_odom, self.latest_odom_receipt, self.follower_config.max_odom_age_sec)
        if odom_failure:
            return odom_failure
        obstacle_failure = self._obstacle_failure()
        if obstacle_failure:
            return obstacle_failure
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            return ownership_failure
        return ""

    def _freshness_failure(self, name: str, msg, receipt, max_age_sec: float) -> str:
        if msg is None or receipt is None:
            failure = message_freshness_failure(
                name,
                has_message=False,
                receipt_age_sec=None,
                header_age_sec=None,
                max_age_sec=max_age_sec,
            )
            self.latest_stop_details = {
                "reason": failure,
                "source": "message_freshness",
                "sensor": name,
                "has_message": False,
                "receipt_age_sec": None,
                "header_age_sec": None,
                "max_age_sec": max_age_sec,
                "fail_closed": True,
            }
            return failure
        now = self.get_clock().now()
        receipt_age = time.monotonic() - float(receipt)
        header_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1_000_000_000.0
        failure = message_freshness_failure(
            name,
            has_message=True,
            receipt_age_sec=receipt_age,
            header_age_sec=header_age,
            max_age_sec=max_age_sec,
        )
        if failure:
            self.latest_stop_details = {
                "reason": failure,
                "source": "message_freshness",
                "sensor": name,
                "has_message": True,
                "receipt_age_sec": receipt_age,
                "header_age_sec": header_age,
                "max_age_sec": max_age_sec,
                "receipt_stale": receipt_age > max_age_sec,
                "header_stale": header_age > max_age_sec,
                "fail_closed": True,
            }
        return failure

    def _scan_range_min(self) -> float | None:
        return float(getattr(self.latest_scan, "range_min")) if hasattr(self.latest_scan, "range_min") else None

    def _scan_range_max(self) -> float | None:
        return float(getattr(self.latest_scan, "range_max")) if hasattr(self.latest_scan, "range_max") else None

    def _obstacle_failure(self) -> str:
        decision = obstacle_decision(
            getattr(self.latest_scan, "ranges", None),
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
        )
        if decision.stop_reason:
            self.latest_stop_details = decision.to_log_dict()
        return decision.stop_reason

    def _motion_clearance_linear_scale(self, linear_x_mps: float) -> float:
        if self.latest_scan is None:
            return 1.0
        if abs(linear_x_mps) <= 1.0e-12:
            self.latest_front_clearance_details = None
            return 1.0
        reversing = linear_x_mps < 0.0
        decision = front_sector_decision(
            getattr(self.latest_scan, "ranges", None),
            float(getattr(self.latest_scan, "angle_min", 0.0)),
            float(getattr(self.latest_scan, "angle_increment", 0.0)),
            math.pi if reversing else 0.0,
            self.follower_config.front_obstacle_sector_rad,
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
            source="rear_sector" if reversing else "front_sector",
        )
        self.latest_front_clearance_details = decision.to_log_dict()
        if decision.stop_reason in (NO_VALID_FRONT_SECTOR_SCAN_RANGES, OBSTACLE_TOO_CLOSE):
            return 0.0
        return linear_scale_for_front_clearance(
            decision.nearest_valid_range_m,
            self.follower_config.min_obstacle_distance_m,
            self.follower_config.front_obstacle_slow_distance_m,
        )

    def _progress_failure(
        self,
        distance_to_target_m: float,
        now_monotonic: float,
        forward_motion_commanded: bool,
    ) -> str:
        if distance_to_target_m + self.follower_config.stuck_progress_epsilon_m < self.last_progress_distance_m:
            self.last_progress_distance_m = distance_to_target_m
            self.last_progress_at = now_monotonic
            return ""
        return stuck_progress_failure(
            now_monotonic - self.last_progress_at,
            self.follower_config.stuck_timeout_sec,
            forward_motion_commanded,
        )

    def _cmd_vel_ownership_failure(self) -> str:
        publishers = self.get_publishers_info_by_topic(self.runtime_config.cmd_vel_topic)
        publisher_identities = sorted({_node_identity(publisher) for publisher in publishers})
        self_identity = _format_node_identity(self.get_namespace(), self.get_name())
        return cmd_vel_ownership_failure(
            publisher_identities,
            self_identity,
            self.follower_config.allowed_cmd_vel_publishers,
        )

    def _current_pose_lookup(self) -> PoseLookupResult:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.runtime_config.map_frame,
                self.runtime_config.base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            return PoseLookupResult(
                None,
                tf_lookup_failure_details(
                    reason="lookup_exception",
                    target_frame=self.runtime_config.map_frame,
                    source_frame=self.runtime_config.base_frame,
                    max_age_sec=self.follower_config.max_tf_age_sec,
                    exception=exc,
                ),
            )
        age = (self.get_clock().now() - Time.from_msg(transform.header.stamp)).nanoseconds / 1_000_000_000.0
        if age > self.follower_config.max_tf_age_sec:
            return PoseLookupResult(
                None,
                tf_lookup_failure_details(
                    reason="stale_transform",
                    target_frame=self.runtime_config.map_frame,
                    source_frame=self.runtime_config.base_frame,
                    max_age_sec=self.follower_config.max_tf_age_sec,
                    age_sec=age,
                ),
            )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return PoseLookupResult(Pose2D(translation.x, translation.y, _yaw_from_quaternion(rotation)))

    def _current_pose(self) -> Pose2D | None:
        return self._current_pose_lookup().pose


def run_simple_waypoint_follower(
    runtime_config: ResolvedRuntimeConfig,
    waypoints: Sequence[Pose2D],
    follower_config: FollowerConfig,
    waypoint_provider: Callable[[Pose2D], RouteUpdate | None] | None = None,
    route_update_callback: Callable[[RouteUpdate], None] | None = None,
) -> FollowerResult:
    _require_ros()
    rclpy.init(args=None)
    node = SimpleWaypointFollowerNode(
        runtime_config,
        waypoints,
        follower_config,
        waypoint_provider,
        route_update_callback,
    )
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
