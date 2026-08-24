"""Minimal ROS2 waypoint follower for one Aufgabe 04 route segment."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scripts.aufgabe04.navigation.controller_trace import (
    ControllerTraceRecord,
    ControllerTraceWriter,
)
from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
)
from scripts.aufgabe04.navigation.driving_behavior import (
    CommandSmoother,
    STATIC_PHYSICAL_ROUTE_KINDS,
    STATIC_STARTUP_SEGMENT_JOIN_ROUTE_KINDS,
    controller_config_for_route_kind,
)
from scripts.aufgabe04.navigation.execution_route_certificate import (
    ExecutionRouteCheck,
    check_execution_route_tube,
)
from scripts.aufgabe04.navigation.odom_route_adapter import (
    OdomExecutionContext,
)
from scripts.aufgabe04.navigation.waypoint_follower.pose_lookup import (
    SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE,
    PoseLookupResult,
    _yaw_from_quaternion,
    tf_lookup_failure_details,
)
from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.ros_runtime_config import (
    ResolvedRuntimeConfig,
    resolve_topic,
)
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
)
from scripts.aufgabe04.navigation.waypoint_controller import (
    CertifiedCornerControlConfig,
    CertifiedCornerTransitionDecision,
    CertifiedCornerTransitionLatch,
    ControllerConfig,
    ControllerStep,
    StartEgressControlConfig,
    VelocityCommand,
    compute_certified_corner_transition,
    compute_reverse_egress_forward_alignment_command,
    compute_start_egress_vertex_command,
)
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    certified_startup_join_action,
    dynamic_join_envelope_failure,
    stuck_progress_details,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    acquisition_goal_action,
    dynamic_route_kind_transition_failure,
    viewpoint_sampling_target_timeout_failure,
    viewpoint_sampling_timeout_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    CertifiedStaticStartupDecision,
    CertifiedStartupRouteState,
    certified_startup_route_state,
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading import (
    INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
    IntermediateTerminalHeadingDecision,
    IntermediateTerminalHeadingLatch,
    compute_intermediate_terminal_heading_command,
    intermediate_terminal_heading_entry_tolerance_m,
    intermediate_terminal_heading_hold_diagnostics,
    reset_intermediate_terminal_heading_latch,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components import (
    BlockageRecoveryRuntimeMixin,
    CallbackServiceRuntimeMixin,
    ControlLoopRuntimeMixin,
    DynamicRouteRuntimeMixin,
    LocalizationRuntimeMixin,
    SafetyRuntimeMixin,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.constants import (
    CALLBACK_SERVICE_BACKGROUND_EXECUTOR,
    CALLBACK_SERVICE_CALLER_SPIN,
    FOLLOWER_EXECUTOR_NUM_THREADS,
    STALE_TF_RECOVERY_MAX_CALLBACKS,
    STALE_TF_RECOVERY_MAX_DURATION_SEC,
    STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC,
    TF_LISTENER_NODE_NAME,
)


try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Twist = None
    LaserScan = None
    Odometry = None
    Empty = None
    Duration = None
    MultiThreadedExecutor = None
    SingleThreadedExecutor = None
    Node = object
    Parameter = None
    qos_profile_sensor_data = None
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _create_dedicated_tf_listener(runtime_config: ResolvedRuntimeConfig):
    """Create an isolated owner for only the TF listener subscriptions."""

    listener_node = Node(
        TF_LISTENER_NODE_NAME,
        namespace=runtime_config.namespace,
    )
    tf_buffer = Buffer(node=listener_node)
    tf_listener = TransformListener(
        tf_buffer,
        listener_node,
        spin_thread=False,
    )
    return listener_node, tf_buffer, tf_listener


class SimpleWaypointFollowerNode(
    ControlLoopRuntimeMixin,
    CallbackServiceRuntimeMixin,
    BlockageRecoveryRuntimeMixin,
    DynamicRouteRuntimeMixin,
    SafetyRuntimeMixin,
    LocalizationRuntimeMixin,
    Node,
):  # pragma: no cover - requires ROS runtime.
    def __init__(
        self,
        runtime_config: ResolvedRuntimeConfig,
        waypoints: Sequence[Pose2D],
        follower_config: FollowerConfig,
        waypoint_provider: Callable[[Pose2D], RouteUpdate | None] | None = None,
        route_update_callback: Callable[[RouteUpdate], None] | None = None,
        blockage_recovery_provider: (
            Callable[
                [Pose2D, str, Mapping[str, object]],
                RouteUpdate | None,
            ]
            | None
        ) = None,
        *,
        controller_trace_path: Path | None = None,
        odom_execution_context: OdomExecutionContext | None = None,
        tf_buffer=None,
    ) -> None:
        # Topic resolution alone does not namespace the ROS node identity.
        # Create the publisher in the resolved namespace so the identity bound
        # into the execution certificate is the identity DDS actually reports.
        super().__init__(
            "aufgabe04_simple_waypoint_follower",
            namespace=runtime_config.namespace,
        )
        self.runtime_config = runtime_config
        self.waypoints = tuple(waypoints)
        self.follower_config = follower_config
        self.waypoint_provider = waypoint_provider
        self.route_update_callback = route_update_callback
        self.blockage_recovery_provider = blockage_recovery_provider
        if odom_execution_context is not None:
            if (
                odom_execution_context.map_frame != runtime_config.map_frame
                or odom_execution_context.odom_frame
                != runtime_config.odom_frame
                or odom_execution_context.base_frame
                != runtime_config.base_frame
            ):
                raise ValueError(
                    "odom execution context frames differ from runtime frames"
                )
        self.odom_execution_context = odom_execution_context
        self.controller_trace_writer = (
            None
            if controller_trace_path is None
            else ControllerTraceWriter(Path(controller_trace_path))
        )
        self.controller_route_revision = 0
        self.queued_route_update: RouteUpdate | None = None
        self.last_route_refresh_at = 0.0
        self.initial_route_refresh_pending = waypoint_provider is not None
        startup_state = certified_startup_route_state(
            follower_config,
            len(self.waypoints),
        )
        self.dynamic_join_pending = startup_state.join_pending
        self.dynamic_join_limit_m = startup_state.join_limit_m
        self.start_egress_lock_index = startup_state.egress_lock_index
        self.start_egress_reverse = False
        self.start_egress_reverse_until_index: int | None = None
        self.start_egress_forward_alignment_index: int | None = None
        self.current_route_kind = follower_config.initial_route_kind
        self.certified_static_start_pending = (
            self.current_route_kind in STATIC_STARTUP_SEGMENT_JOIN_ROUTE_KINDS
            and not startup_state.join_pending
        )
        self.intermediate_terminal_heading_latch: (
            IntermediateTerminalHeadingLatch | None
        ) = None
        self.certified_corner_latch: CertifiedCornerTransitionLatch | None = None
        self._last_certified_corner_phase: tuple[int, str] | None = None
        self.command_smoother = CommandSmoother(follower_config.command_smoothing)
        self.last_command_shape_at: float | None = None
        self.control_loop_deadline_sec: float | None = None
        self.reverse_staging = False
        self.axis_acquisition_hold_started_at: float | None = None
        self.axis_acquisition_target_revision: int | None = None
        self.viewpoint_sampling_started_at: float | None = None
        self.viewpoint_sampling_target_started_at: float | None = None
        self.viewpoint_sampling_target_revision: int | None = None
        self.target_index = 0
        self.latest_scan = None
        self.latest_scan_receipt = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.latest_odom_callback_count = 0
        self.motion_published = False
        self.distance_estimate_m = 0.0
        self.last_pose = None
        self.last_progress_distance_m = math.inf
        self.last_progress_heading_error_rad = math.inf
        self.last_progress_target_index: int | None = None
        self.last_progress_pursuit_index: int | None = None
        self.last_progress_mode: str | None = None
        self.progress_heading_modes_seen: set[str] = set()
        self.progress_heading_error_by_mode: dict[str, float] = {}
        self.last_progress_at = time.monotonic()
        self.target_started_at = time.monotonic()
        self.latest_stop_details = None
        self.latest_front_clearance_details = None
        self._simulation_odom_fallback_active = False
        self._simulation_odom_fallback_episode = 0
        self._background_callback_service_enabled = False
        self._configure_sim_time(runtime_config.use_sim_time)

        # Service names follow the same ROS graph-name resolution rules as
        # topics.  A relative default therefore follows the robot namespace,
        # while an explicitly absolute operator value remains global.
        self.runtime_nomotion_update_service = resolve_topic(
            follower_config.runtime_nomotion_update_service,
            runtime_config.namespace,
        )
        self.runtime_nomotion_update_client = None
        if (
            runtime_config.localization_source == "amcl"
            and not runtime_config.use_sim_time
        ):
            self.runtime_nomotion_update_client = self.create_client(
                Empty,
                self.runtime_nomotion_update_service,
            )

        self.cmd_vel_pub = self.create_publisher(Twist, runtime_config.cmd_vel_topic, 10)
        self.create_subscription(
            LaserScan,
            runtime_config.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(Odometry, runtime_config.odom_topic, self._odom_callback, 10)
        if tf_buffer is None:
            # Preserve direct-node construction for focused ROS use.  The
            # production runner always injects the buffer serviced by its
            # isolated listener node/executor below.
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.tf_buffer = tf_buffer
            self.tf_listener = None

    def _configure_sim_time(self, use_sim_time: bool) -> None:
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", use_sim_time)
            return
        self.set_parameters(
            [Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)]
        )

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
        self.latest_odom_callback_count = (
            getattr(self, "latest_odom_callback_count", 0) + 1
        )

    def publish_zero(self) -> None:
        self.command_smoother.reset()
        self.last_command_shape_at = None
        self.cmd_vel_pub.publish(Twist())

    def _publish_velocity_command(self, command: VelocityCommand) -> None:
        """Publish one already-validated command from the sole motion edge."""

        twist = Twist()
        twist.linear.x = command.linear_x_mps
        twist.angular.z = command.angular_z_radps
        self.cmd_vel_pub.publish(twist)
        self.motion_published = self.motion_published or (
            abs(twist.linear.x) > 0.0 or abs(twist.angular.z) > 0.0
        )

    def _hold_zero_control_period(self, period_sec: float) -> None:
        """Preserve a full zero handoff and restart the normal deadline cadence."""

        time.sleep(period_sec)
        self.control_loop_deadline_sec = time.monotonic() + period_sec

    @property
    def callback_service_mode(self) -> str:
        if getattr(self, "_background_callback_service_enabled", False):
            return CALLBACK_SERVICE_BACKGROUND_EXECUTOR
        return CALLBACK_SERVICE_CALLER_SPIN

    def enable_background_callback_service(self) -> None:
        """Mark this node as owned by the continuously spinning executor."""

        self._background_callback_service_enabled = True

    def disable_background_callback_service(self) -> None:
        """Return callback ownership to the caller after executor teardown."""

        self._background_callback_service_enabled = False

    def _service_or_wait_for_callbacks(self, timeout_sec: float) -> None:
        """Wait for background callbacks, or service them in legacy caller mode."""

        if self.callback_service_mode == CALLBACK_SERVICE_BACKGROUND_EXECUTOR:
            time.sleep(timeout_sec)
            return
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def publish_repeated_zero(self, count: int = 5) -> None:
        for _ in range(count):
            self.publish_zero()
            self._service_or_wait_for_callbacks(0.02)

    def _latest_odom_pose(self) -> Pose2D | None:
        """Return a finite unit-quaternion odom pose for stationarity checks."""

        try:
            raw_pose = self.latest_odom.pose.pose
            position = raw_pose.position
            orientation = raw_pose.orientation
            quaternion = (
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            x_m = float(position.x)
            y_m = float(position.y)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in (x_m, y_m, *quaternion)):
            return None
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE:
            return None
        yaw_rad = _yaw_from_quaternion(orientation)
        if not math.isfinite(yaw_rad):
            return None
        return Pose2D(x_m, y_m, yaw_rad)
    def _egress_trace_phase(self) -> str:
        if getattr(self, "dynamic_join_pending", False):
            return "dynamic_join"
        if getattr(self, "start_egress_reverse", False):
            return "straight_reverse"
        if getattr(self, "start_egress_forward_alignment_index", None) is not None:
            return "reverse_to_forward_alignment"
        if getattr(self, "start_egress_lock_index", None) is not None:
            return "start_egress"
        return ""

    def _append_controller_trace(
        self,
        *,
        event: str,
        pose: Pose2D | None = None,
        step: ControllerStep | None = None,
        route_check: ExecutionRouteCheck | None = None,
        nominal_command: VelocityCommand | None = None,
        effective_command: VelocityCommand | None = None,
        reason: str = "",
        fail_closed: bool = False,
        front_cluster_summary: Mapping[str, object] | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> str:
        """Append one evidence record or return a fail-closed write error."""

        writer = getattr(self, "controller_trace_writer", None)
        if writer is None:
            return ""
        try:
            writer.append(
                ControllerTraceRecord(
                    timestamp_sec=time.monotonic(),
                    event=event,
                    reason=reason,
                    fail_closed=fail_closed,
                    route_revision=getattr(
                        self,
                        "controller_route_revision",
                        0,
                    ),
                    route_kind=getattr(self, "current_route_kind", ""),
                    target_index=(
                        getattr(self, "target_index", None)
                        if step is None
                        else step.target_index
                    ),
                    pursuit_index=(None if step is None else step.pursuit_index),
                    progress_mode=("" if step is None else step.progress_mode),
                    egress_phase=self._egress_trace_phase(),
                    map_pose=pose,
                    odom_pose=self._latest_odom_pose(),
                    active_segment_start_index=(
                        None
                        if route_check is None
                        else route_check.active_segment_start_index
                    ),
                    active_segment_end_index=(
                        None
                        if route_check is None
                        else route_check.active_segment_end_index
                    ),
                    distance_to_target_m=(
                        None if step is None else step.distance_to_target_m
                    ),
                    pose_distance_to_segment_m=(
                        None
                        if route_check is None
                        else route_check.pose_distance_to_segment_m
                    ),
                    maximum_chord_distance_to_segment_m=(
                        None
                        if route_check is None
                        else route_check.maximum_chord_distance_to_segment_m
                    ),
                    tracking_tube_radius_m=(
                        None
                        if route_check is None
                        else route_check.tracking_tube_radius_m
                    ),
                    nominal_command=nominal_command,
                    effective_command=effective_command,
                    front_clearance=getattr(
                        self,
                        "latest_front_clearance_details",
                        None,
                    ),
                    front_cluster_summary=front_cluster_summary,
                    diagnostics=diagnostics,
                )
            )
        except Exception as exc:
            failure = f"controller trace write failed: {exc}"
            self.latest_stop_details = {
                **dict(getattr(self, "latest_stop_details", None) or {}),
                "reason": failure,
                "fault_code": "controller_trace_write_failed",
                "controller_trace_path": str(writer.path),
                "fail_closed": True,
            }
            return failure
        return ""


    def _start_egress_command(
        self,
        pose: Pose2D,
        controller_config: ControllerConfig,
    ):
        """Return a locked command, or release the lock at its exact vertex."""

        waypoint_index = self.start_egress_lock_index
        if waypoint_index is None:
            raise ValueError("start egress command requested without an active lock")
        step = compute_start_egress_vertex_command(
            pose,
            self.waypoints,
            waypoint_index,
            controller_config,
            reach_tolerance_m=(
                self.follower_config.start_egress_waypoint_tolerance_m
            ),
            egress_config=StartEgressControlConfig(
                alignment_tolerance_rad=(
                    self.follower_config.start_egress_alignment_tolerance_rad
                ),
                max_linear_mps=self.follower_config.start_egress_max_linear_mps,
            ),
            reverse=getattr(self, "start_egress_reverse", False),
        )
        if step is not None:
            return step
        reverse_until_index = self.start_egress_reverse_until_index
        if (
            self.start_egress_reverse
            and reverse_until_index is not None
            and waypoint_index < reverse_until_index
        ):
            # Consume a zero-command cycle at every certified reverse vertex.
            # The next tick locks to the immediately following vertex, so the
            # route checker switches to that exact segment without lookahead.
            self.start_egress_lock_index = waypoint_index + 1
            if waypoint_index != self.target_index:
                self._clear_intermediate_terminal_heading_latch(
                    target_changed=True,
                )
            self.target_index = waypoint_index
            self.target_started_at = time.monotonic()
            self._reset_progress_watchdog(time.monotonic())
            return None
        was_reverse = self.start_egress_reverse
        self.start_egress_lock_index = None
        self.start_egress_reverse = False
        self.start_egress_reverse_until_index = None
        if waypoint_index != self.target_index:
            self._clear_intermediate_terminal_heading_latch(
                target_changed=True,
            )
        self.target_index = waypoint_index
        self.target_started_at = time.monotonic()
        self._reset_progress_watchdog(time.monotonic())
        if not was_reverse:
            self.start_egress_forward_alignment_index = None
        return None

    def _reverse_egress_forward_alignment_command(
        self,
        pose: Pose2D,
        controller_config: ControllerConfig,
    ) -> ControllerStep:
        """Rotate onto the certified outgoing segment after reverse escape."""

        waypoint_index = self.start_egress_forward_alignment_index
        if waypoint_index is None:
            raise ValueError(
                "reverse-egress forward alignment requested without an index"
            )
        step = compute_reverse_egress_forward_alignment_command(
            pose,
            self.waypoints,
            waypoint_index,
            controller_config,
            alignment_tolerance_rad=(
                self.follower_config.start_egress_alignment_tolerance_rad
            ),
        )
        if step.progress_mode == "reverse_egress_forward_handoff":
            # The caller still runs the outgoing-segment route-tube check for
            # this zero-command handoff cycle before ordinary tracking resumes.
            self.start_egress_forward_alignment_index = None
        return step

    def _certified_corner_decision(
        self,
        pose: Pose2D,
        controller_config: ControllerConfig,
    ) -> CertifiedCornerTransitionDecision:
        """Apply the physical discovery-route sharp-vertex contract."""

        if self.current_route_kind != "stand_discovery_corridor":
            self.certified_corner_latch = None
            return CertifiedCornerTransitionDecision(None, None)
        decision = compute_certified_corner_transition(
            pose,
            self.waypoints,
            self.target_index,
            controller_config,
            self.certified_corner_latch,
            CertifiedCornerControlConfig(
                turn_threshold_rad=(
                    self.follower_config.certified_corner_turn_threshold_rad
                ),
                release_tolerance_m=(
                    self.follower_config.certified_corner_release_tolerance_m
                ),
                hold_tolerance_m=(
                    self.follower_config.certified_corner_hold_tolerance_m
                ),
                hard_tolerance_m=(
                    self.follower_config.certified_route_tube_radius_m
                ),
                alignment_tolerance_rad=(
                    self.follower_config
                    .certified_corner_alignment_tolerance_rad
                ),
                max_reacquire_attempts=(
                    self.follower_config
                    .certified_corner_max_reacquire_attempts
                ),
            ),
        )
        self.certified_corner_latch = decision.latch
        return decision

    def _execution_route_check(
        self,
        pose: Pose2D,
        step: ControllerStep,
    ) -> ExecutionRouteCheck:
        """Check the live pose and pursuit chord against the active segment."""

        return check_execution_route_tube(
            pose,
            self.waypoints,
            target_index=step.target_index,
            pursuit_index=step.pursuit_index,
            tracking_tube_radius_m=(
                self.follower_config.certified_route_tube_radius_m
            ),
            chord_sample_spacing_m=(
                self.follower_config.certified_route_chord_sample_spacing_m
            ),
        )

    def _log_certified_corner_phase(
        self,
        step: ControllerStep | None,
    ) -> None:
        if step is None or not step.progress_mode.startswith("certified_corner_"):
            self._last_certified_corner_phase = None
            return
        phase = (step.target_index, step.progress_mode)
        if phase == self._last_certified_corner_phase:
            return
        self._last_certified_corner_phase = phase
        try:
            logger = self.get_logger()
        except Exception:
            return
        reacquire_attempts = (
            0
            if self.certified_corner_latch is None
            else self.certified_corner_latch.reacquire_attempts
        )
        logger.info(
            "certified sharp-corner transition: "
            f"phase={step.progress_mode} target_index={step.target_index} "
            f"distance_m={step.distance_to_target_m:.6f} "
            f"heading_error_rad={step.controlled_heading_error_rad:.6f} "
            f"linear_mps={step.command.linear_x_mps:.6f} "
            f"angular_radps={step.command.angular_z_radps:.6f} "
            f"reacquire_attempts={reacquire_attempts}"
        )

    def _reset_progress_watchdog(self, now_monotonic: float) -> None:
        self.last_progress_distance_m = math.inf
        self.last_progress_heading_error_rad = math.inf
        self.last_progress_target_index = None
        self.last_progress_pursuit_index = None
        self.last_progress_mode = None
        self.progress_heading_modes_seen.clear()
        self.progress_heading_error_by_mode.clear()
        self.last_progress_at = now_monotonic

    def _clear_intermediate_terminal_heading_latch(
        self,
        *,
        material_route_revision: bool = False,
        target_changed: bool = False,
    ) -> None:
        self.intermediate_terminal_heading_latch = (
            reset_intermediate_terminal_heading_latch(
                getattr(self, "intermediate_terminal_heading_latch", None),
                material_route_revision=material_route_revision,
                target_changed=target_changed,
            )
        )






























def run_simple_waypoint_follower(
    runtime_config: ResolvedRuntimeConfig,
    waypoints: Sequence[Pose2D],
    follower_config: FollowerConfig,
    waypoint_provider: Callable[[Pose2D], RouteUpdate | None] | None = None,
    route_update_callback: Callable[[RouteUpdate], None] | None = None,
    blockage_recovery_provider: (
        Callable[
            [Pose2D, str, Mapping[str, object]],
            RouteUpdate | None,
        ]
        | None
    ) = None,
    controller_trace_path: Path | None = None,
    odom_execution_context: OdomExecutionContext | None = None,
) -> FollowerResult:
    _require_ros()
    rclpy.init(args=None)
    node = None
    listener_node = None
    tf_listener = None
    follower_executor = None
    tf_executor = None
    follower_executor_thread = None
    tf_executor_thread = None
    node_added_to_follower_executor = False
    listener_node_added_to_tf_executor = False
    try:
        listener_node, tf_buffer, tf_listener = _create_dedicated_tf_listener(
            runtime_config
        )
        node_kwargs = {"tf_buffer": tf_buffer}
        if controller_trace_path is not None:
            node_kwargs["controller_trace_path"] = controller_trace_path
        if odom_execution_context is not None:
            node_kwargs["odom_execution_context"] = odom_execution_context
        node = SimpleWaypointFollowerNode(
            runtime_config,
            waypoints,
            follower_config,
            waypoint_provider,
            route_update_callback,
            blockage_recovery_provider,
            **node_kwargs,
        )
        follower_executor = MultiThreadedExecutor(
            num_threads=FOLLOWER_EXECUTOR_NUM_THREADS,
        )
        tf_executor = SingleThreadedExecutor()
        node_added_to_follower_executor = follower_executor.add_node(node)
        if not node_added_to_follower_executor:
            raise RuntimeError(
                "failed to add waypoint follower to background executor"
            )
        listener_node_added_to_tf_executor = tf_executor.add_node(
            listener_node
        )
        if not listener_node_added_to_tf_executor:
            raise RuntimeError(
                "failed to add TF listener to isolated background executor"
            )
        node.enable_background_callback_service()
        tf_executor_thread = threading.Thread(
            target=tf_executor.spin,
            name="aufgabe04-follower-tf-listener",
            daemon=False,
        )
        follower_executor_thread = threading.Thread(
            target=follower_executor.spin,
            name="aufgabe04-follower-callbacks",
            daemon=False,
        )
        # Give TF its independent callback service before the follower starts
        # its sensor wait/control loop.
        tf_executor_thread.start()
        follower_executor_thread.start()
        return node.run()
    finally:
        if follower_executor is not None:
            follower_executor.shutdown()
        if tf_executor is not None:
            tf_executor.shutdown()
        if (
            follower_executor_thread is not None
            and follower_executor_thread.ident is not None
        ):
            follower_executor_thread.join()
        if (
            tf_executor_thread is not None
            and tf_executor_thread.ident is not None
        ):
            tf_executor_thread.join()
        if (
            follower_executor is not None
            and node is not None
            and node_added_to_follower_executor
        ):
            follower_executor.remove_node(node)
        if (
            tf_executor is not None
            and listener_node is not None
            and listener_node_added_to_tf_executor
        ):
            tf_executor.remove_node(listener_node)
        if node is not None:
            node.disable_background_callback_service()
            node.destroy_node()
        if tf_listener is not None:
            tf_listener.unregister()
        if listener_node is not None:
            listener_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
