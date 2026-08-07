"""ROS2 preflight observations for Aufgabe 04 station-segment runs.

This module observes ROS graph, topic, and TF state. It never publishes motion.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from scripts.aufgabe04.navigation.localization_ownership import (
    LocalizationOwnershipEvidence,
    evaluate_localization_ownership,
)
from scripts.aufgabe04.navigation.localization_preflight_evidence import (
    build_dynamic_map_to_odom_freshness,
    build_localization_ownership_observation_data,
    find_external_tf_owner_candidates,
)
from scripts.aufgabe04.navigation.ros_runtime_config import (
    ResolvedRuntimeConfig,
    RuntimeConfig,
    resolve_topic,
    resolve_runtime_config,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from action_msgs.msg import GoalStatusArray
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Empty
    from tf2_msgs.msg import TFMessage
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    GoalStatusArray = None
    LaserScan = None
    Empty = None
    Odometry = None
    PoseWithCovarianceStamped = None
    TFMessage = None
    Duration = None
    Node = object
    Parameter = None
    qos_profile_sensor_data = None
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


ACTIVE_GOAL_STATUS = {1, 2, 3, 4}


def _node_identity(endpoint) -> str:
    namespace = getattr(endpoint, "node_namespace", "") or ""
    name = getattr(endpoint, "node_name", "") or ""
    return _node_identity_from_names(namespace, name)


def _node_identity_from_names(namespace: str, name: str) -> str:
    if namespace in ("", "/"):
        return f"/{name}"
    return f"{namespace.rstrip('/')}/{name}"


@dataclass(frozen=True)
class RosObservation:
    name: str
    ok: bool
    detail: str
    data: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StationaryAmclPoseSample:
    x_m: float
    y_m: float
    yaw_rad: float
    covariance: Tuple[float, ...] = ()


def _angular_distance_rad(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _maximum_position_std_m(sample: StationaryAmclPoseSample) -> float | None:
    if len(sample.covariance) < 8:
        return None
    covariance_xx = float(sample.covariance[0])
    covariance_xy = float(sample.covariance[1])
    covariance_yy = float(sample.covariance[7])
    if not all(
        math.isfinite(value)
        for value in (covariance_xx, covariance_xy, covariance_yy)
    ):
        return None
    if covariance_xx < 0.0 or covariance_yy < 0.0:
        return None
    discriminant = max(
        0.0,
        (covariance_xx - covariance_yy) ** 2 + 4.0 * covariance_xy**2,
    )
    maximum_eigenvalue = 0.5 * (
        covariance_xx + covariance_yy + math.sqrt(discriminant)
    )
    if maximum_eigenvalue < 0.0:
        return None
    return math.sqrt(maximum_eigenvalue)


def _yaw_std_rad(sample: StationaryAmclPoseSample) -> float | None:
    if len(sample.covariance) < 36:
        return None
    covariance_yaw = float(sample.covariance[35])
    if not math.isfinite(covariance_yaw) or covariance_yaw < 0.0:
        return None
    return math.sqrt(covariance_yaw)


def evaluate_stationary_amcl_stability(
    samples: Sequence[StationaryAmclPoseSample],
    *,
    required_sample_count: int,
    max_position_spread_m: float,
    max_yaw_spread_rad: float,
    max_position_std_m: float = 0.015,
    max_yaw_std_rad: float = 0.03,
) -> RosObservation:
    """Evaluate no-motion AMCL means and uncertainty before authorization."""

    if (
        not isinstance(required_sample_count, int)
        or isinstance(required_sample_count, bool)
        or required_sample_count < 2
    ):
        raise ValueError("required_sample_count must be an integer >= 2")
    for name, value in {
        "max_position_spread_m": max_position_spread_m,
        "max_yaw_spread_rad": max_yaw_spread_rad,
        "max_position_std_m": max_position_std_m,
        "max_yaw_std_rad": max_yaw_std_rad,
    }.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    position_spread_m = 0.0
    yaw_spread_rad = 0.0
    for first in samples:
        for second in samples:
            position_spread_m = max(
                position_spread_m,
                math.hypot(first.x_m - second.x_m, first.y_m - second.y_m),
            )
            yaw_spread_rad = max(
                yaw_spread_rad,
                _angular_distance_rad(first.yaw_rad, second.yaw_rad),
            )
    position_stds = [
        value
        for value in (_maximum_position_std_m(sample) for sample in samples)
        if value is not None
    ]
    yaw_stds = [
        value
        for value in (_yaw_std_rad(sample) for sample in samples)
        if value is not None
    ]
    enough_samples = len(samples) >= required_sample_count
    position_ok = position_spread_m <= max_position_spread_m
    yaw_ok = yaw_spread_rad <= max_yaw_spread_rad
    position_covariance_complete = len(position_stds) == len(samples)
    yaw_covariance_complete = len(yaw_stds) == len(samples)
    maximum_position_std_m = max(position_stds) if position_stds else None
    maximum_yaw_std_rad = max(yaw_stds) if yaw_stds else None
    position_std_ok = (
        position_covariance_complete
        and maximum_position_std_m is not None
        and maximum_position_std_m <= max_position_std_m
    )
    yaw_std_ok = (
        yaw_covariance_complete
        and maximum_yaw_std_rad is not None
        and maximum_yaw_std_rad <= max_yaw_std_rad
    )
    ok = (
        enough_samples
        and position_ok
        and yaw_ok
        and position_std_ok
        and yaw_std_ok
    )
    position_std_detail = (
        "missing"
        if maximum_position_std_m is None
        else f"{maximum_position_std_m:.4f}m"
    )
    yaw_std_detail = (
        "missing"
        if maximum_yaw_std_rad is None
        else f"{maximum_yaw_std_rad:.4f}rad"
    )
    detail = (
        f"samples={len(samples)}/{required_sample_count} "
        f"position_spread={position_spread_m:.4f}m "
        f"yaw_spread={yaw_spread_rad:.4f}rad "
        f"position_std={position_std_detail}/{max_position_std_m:.4f}m "
        f"yaw_std={yaw_std_detail}/{max_yaw_std_rad:.4f}rad"
    )
    return RosObservation(
        "stationary AMCL stability",
        ok,
        detail,
        {
            "sample_count": len(samples),
            "required_sample_count": required_sample_count,
            "maximum_position_spread_m": position_spread_m,
            "maximum_yaw_spread_rad": yaw_spread_rad,
            "max_allowed_position_spread_m": max_position_spread_m,
            "max_allowed_yaw_spread_rad": max_yaw_spread_rad,
            "maximum_reported_position_std_m": maximum_position_std_m,
            "maximum_reported_yaw_std_rad": maximum_yaw_std_rad,
            "max_allowed_position_std_m": max_position_std_m,
            "max_allowed_yaw_std_rad": max_yaw_std_rad,
            "position_covariance_complete": position_covariance_complete,
            "yaw_covariance_complete": yaw_covariance_complete,
        },
    )


def evaluate_latest_stationary_amcl_window(
    samples: Sequence[StationaryAmclPoseSample],
    *,
    required_sample_count: int,
    max_position_spread_m: float,
    max_yaw_spread_rad: float,
    max_position_std_m: float = 0.015,
    max_yaw_std_rad: float = 0.03,
) -> RosObservation:
    """Evaluate only the newest complete stationary convergence window.

    AMCL starts with deliberately broad covariance after an ``/initialpose``
    update.  Those settling samples must remain observable, but they must not
    permanently poison a later consecutive window that satisfies the same
    strict admission limits.  An incomplete window still fails closed.
    """

    if (
        not isinstance(required_sample_count, int)
        or isinstance(required_sample_count, bool)
        or required_sample_count < 2
    ):
        raise ValueError("required_sample_count must be an integer >= 2")
    window_start_index = max(0, len(samples) - required_sample_count)
    window = samples[window_start_index:]
    observation = evaluate_stationary_amcl_stability(
        window,
        required_sample_count=required_sample_count,
        max_position_spread_m=max_position_spread_m,
        max_yaw_spread_rad=max_yaw_spread_rad,
        max_position_std_m=max_position_std_m,
        max_yaw_std_rad=max_yaw_std_rad,
    )
    return RosObservation(
        observation.name,
        observation.ok,
        observation.detail,
        {
            **observation.data,
            "total_sample_count": len(samples),
            "window_start_index": window_start_index,
        },
    )


@dataclass(frozen=True)
class RosPreflightResult:
    ok: bool
    failures: List[str]
    observations: List[RosObservation]
    runtime_config: Dict[str, object]
    route_pose: Dict[str, object] | None = None
    odom_pose: Dict[str, object] | None = None
    map_from_odom: Dict[str, object] | None = None
    stationary_amcl_samples: List[Dict[str, object]] = field(
        default_factory=list
    )

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "failures": self.failures,
            "observations": [asdict(observation) for observation in self.observations],
            "runtime_config": self.runtime_config,
            "route_pose": self.route_pose,
            "odom_pose": self.odom_pose,
            "map_from_odom": self.map_from_odom,
            "stationary_amcl_samples": self.stationary_amcl_samples,
        }


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _node_parameter_overrides(use_sim_time: bool):
    """Provide the ROS clock parameter without redeclaring it in the node."""

    if Parameter is None:
        _require_ros()
    return [Parameter("use_sim_time", Parameter.Type.BOOL, bool(use_sim_time))]


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _topic_type_names(node: Node, topic: str) -> List[str]:
    return [item[0] for item in node.get_topic_names_and_types() if item[0] == topic for _ in item[1]]


def _topic_types(node: Node, topic: str) -> List[str]:
    for name, types in node.get_topic_names_and_types():
        if name == topic:
            return list(types)
    return []


def _frame_id(frame_id: str) -> str:
    return frame_id.strip("/")


class RosPreflightNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(
        self,
        config: ResolvedRuntimeConfig,
        *,
        max_scan_age_sec: float,
        max_odom_age_sec: float,
        max_tf_age_sec: float,
        max_amcl_age_sec: float,
        max_future_timestamp_sec: float,
        max_localization_tf_future_sec: float,
        observation_window_sec: float,
        allow_idle_nav2: bool,
        allowed_cmd_vel_publishers: Sequence[str],
        require_real_time: bool,
        request_nomotion_update: bool,
        nomotion_update_service: str,
        nomotion_update_timeout_sec: float,
        stationary_amcl_sample_count: int,
        stationary_amcl_sample_interval_sec: float,
        max_stationary_amcl_position_spread_m: float,
        max_stationary_amcl_yaw_spread_rad: float,
        max_stationary_amcl_position_std_m: float,
        max_stationary_amcl_yaw_std_rad: float,
        execution_pose_owner: str,
        global_consistency_monitor: str,
        frozen_map_transform_certified: bool,
    ) -> None:
        super().__init__(
            "aufgabe04_ros_preflight",
            parameter_overrides=_node_parameter_overrides(config.use_sim_time),
        )
        self.config = config
        self.max_scan_age_sec = max_scan_age_sec
        self.max_odom_age_sec = max_odom_age_sec
        self.max_tf_age_sec = max_tf_age_sec
        self.max_amcl_age_sec = max_amcl_age_sec
        self.max_future_timestamp_sec = max_future_timestamp_sec
        self.max_localization_tf_future_sec = max_localization_tf_future_sec
        self.observation_window_sec = observation_window_sec
        self.allow_idle_nav2 = allow_idle_nav2
        self.allowed_cmd_vel_publishers = tuple(allowed_cmd_vel_publishers)
        self.require_real_time = require_real_time
        self.nomotion_update_timeout_sec = nomotion_update_timeout_sec
        self.stationary_amcl_sample_count = stationary_amcl_sample_count
        self.stationary_amcl_sample_interval_sec = (
            stationary_amcl_sample_interval_sec
        )
        self.max_stationary_amcl_position_spread_m = (
            max_stationary_amcl_position_spread_m
        )
        self.max_stationary_amcl_yaw_spread_rad = (
            max_stationary_amcl_yaw_spread_rad
        )
        self.max_stationary_amcl_position_std_m = (
            max_stationary_amcl_position_std_m
        )
        self.max_stationary_amcl_yaw_std_rad = (
            max_stationary_amcl_yaw_std_rad
        )
        self.execution_pose_owner = str(execution_pose_owner).strip()
        self.global_consistency_monitor = str(
            global_consistency_monitor
        ).strip()
        self.frozen_map_transform_certified = bool(
            frozen_map_transform_certified
        )
        self.stationary_amcl_samples: List[StationaryAmclPoseSample] = []
        self.latest_scan = None
        self.latest_scan_receipt = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.latest_amcl = None
        self.latest_amcl_receipt = None
        self.latest_nav2_status = None
        self.latest_dynamic_map_to_odom = None
        self.latest_dynamic_map_to_odom_receipt = None
        self.nomotion_client = (
            self.create_client(Empty, nomotion_update_service)
            if request_nomotion_update and config.localization_source == "amcl"
            else None
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.dynamic_tf_topics = self._dynamic_tf_topic_candidates()
        for topic in self.dynamic_tf_topics:
            self.create_subscription(TFMessage, topic, self._dynamic_tf_callback, 10)
        self.create_subscription(
            LaserScan,
            config.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(Odometry, config.odom_topic, self._odom_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            config.amcl_topic,
            self._amcl_callback,
            10,
        )
        self.nav2_status_topics = self._find_nav2_status_topics()
        for topic in self.nav2_status_topics:
            self.create_subscription(GoalStatusArray, topic, self._nav2_status_callback, 10)

    def _scan_callback(self, msg) -> None:
        self.latest_scan = msg
        self.latest_scan_receipt = self.get_clock().now()

    def _odom_callback(self, msg) -> None:
        self.latest_odom = msg
        self.latest_odom_receipt = self.get_clock().now()

    def _amcl_callback(self, msg) -> None:
        self.latest_amcl = msg
        self.latest_amcl_receipt = self.get_clock().now()

    def _nav2_status_callback(self, msg) -> None:
        self.latest_nav2_status = msg

    def _dynamic_tf_callback(self, msg) -> None:
        for transform in msg.transforms:
            if self._is_configured_map_to_odom(transform):
                self.latest_dynamic_map_to_odom = transform
                self.latest_dynamic_map_to_odom_receipt = self.get_clock().now()

    def _is_configured_map_to_odom(self, transform) -> bool:
        return (
            _frame_id(transform.header.frame_id) == _frame_id(self.config.map_frame)
            and _frame_id(transform.child_frame_id) == _frame_id(self.config.odom_frame)
        )

    def _dynamic_tf_topic_candidates(self) -> List[str]:
        return sorted({"/tf", resolve_topic("tf", self.config.namespace)})

    def _find_nav2_status_topics(self) -> List[str]:
        topics = [
            "/navigate_to_pose/_action/status",
            f"{self.config.namespace}/navigate_to_pose/_action/status"
            if self.config.namespace
            else "/navigate_to_pose/_action/status",
        ]
        for name, types in self.get_topic_names_and_types():
            if name.endswith("/_action/status") and any(
                type_name == "action_msgs/msg/GoalStatusArray" for type_name in types
            ):
                topics.append(name)
        return sorted(set(topics))

    def collect(self) -> RosPreflightResult:
        deadline = time.monotonic() + self.observation_window_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        observations: List[RosObservation] = []
        failures: List[str] = []
        if self.nomotion_client is not None:
            observations.append(self._refresh_stationary_amcl())
            stability = self._observe_stationary_amcl_stability()
            observations.append(stability)
            if not stability.ok:
                failures.append(
                    f"stationary AMCL stability: {stability.detail}"
                )

        self._observe_topic(
            observations,
            failures,
            self.config.scan_topic,
            "sensor_msgs/msg/LaserScan",
        )
        self._observe_topic(observations, failures, self.config.odom_topic, "nav_msgs/msg/Odometry")
        if self.config.localization_source == "amcl":
            self._observe_topic(
                observations,
                failures,
                self.config.amcl_topic,
                "geometry_msgs/msg/PoseWithCovarianceStamped",
            )
        self._observe_fresh_message(
            observations,
            failures,
            "scan freshness",
            self.latest_scan,
            self.latest_scan_receipt,
            self.max_scan_age_sec,
        )
        self._observe_fresh_message(
            observations,
            failures,
            "odom freshness",
            self.latest_odom,
            self.latest_odom_receipt,
            self.max_odom_age_sec,
        )
        if self.config.localization_source == "amcl":
            self._observe_fresh_message(
                observations,
                failures,
                "amcl freshness",
                self.latest_amcl,
                self.latest_amcl_receipt,
                self.max_amcl_age_sec,
            )
        map_to_base_ok, map_to_base_data = self._observe_tf(
            observations,
            failures,
            self.config.map_frame,
            self.config.base_frame,
            self.max_tf_age_sec,
        )
        odom_to_base_ok, odom_to_base_data = self._observe_tf(
            observations,
            failures,
            self.config.odom_frame,
            self.config.base_frame,
            self.max_tf_age_sec,
        )
        map_to_odom_ok = False
        map_to_odom_transform_data: Dict[str, object] = {}
        if self.execution_pose_owner == "odom":
            map_to_odom_ok, map_to_odom_transform_data = self._observe_tf(
                observations,
                failures,
                self.config.map_frame,
                self.config.odom_frame,
                self.max_tf_age_sec,
                max_future_sec=self.max_localization_tf_future_sec,
            )
        self._observe_localization_ownership(
            observations,
            failures,
            route_transform_fresh=map_to_base_ok,
            route_transform_data=map_to_base_data,
            odom_to_base_fresh=odom_to_base_ok,
            odom_to_base_data=odom_to_base_data,
        )
        self._observe_use_sim_time(observations, failures)
        self._observe_cmd_vel_ownership(observations, failures)
        route_pose = None
        if map_to_base_ok and all(
            key in map_to_base_data for key in ("x_m", "y_m", "yaw_rad")
        ):
            route_pose = {
                "frame_id": self.config.map_frame,
                "child_frame_id": self.config.base_frame,
                "x_m": map_to_base_data["x_m"],
                "y_m": map_to_base_data["y_m"],
                "yaw_rad": map_to_base_data["yaw_rad"],
            }
        odom_pose = None
        if odom_to_base_ok and all(
            key in odom_to_base_data for key in ("x_m", "y_m", "yaw_rad")
        ):
            odom_pose = {
                "frame_id": self.config.odom_frame,
                "child_frame_id": self.config.base_frame,
                "x_m": odom_to_base_data["x_m"],
                "y_m": odom_to_base_data["y_m"],
                "yaw_rad": odom_to_base_data["yaw_rad"],
            }
        map_from_odom = (
            dict(map_to_odom_transform_data) if map_to_odom_ok else None
        )
        return RosPreflightResult(
            ok=not failures,
            failures=failures,
            observations=observations,
            runtime_config=self.config.as_log_dict(),
            route_pose=route_pose,
            odom_pose=odom_pose,
            map_from_odom=map_from_odom,
            stationary_amcl_samples=[
                {
                    "x_m": sample.x_m,
                    "y_m": sample.y_m,
                    "yaw_rad": sample.yaw_rad,
                    "covariance": list(sample.covariance),
                }
                for sample in self.stationary_amcl_samples
            ],
        )

    def _refresh_stationary_amcl(self) -> RosObservation:
        """Request AMCL only after this node's subscriptions and TF listener exist."""

        fresh, _data = self._message_freshness(
            self.latest_amcl,
            self.latest_amcl_receipt,
            self.max_amcl_age_sec,
        )
        if fresh and self._route_transform_available():
            return RosObservation(
                "stationary AMCL refresh",
                True,
                "fresh AMCL already observed",
                {"service_requested": False},
            )
        deadline = time.monotonic() + self.nomotion_update_timeout_sec
        future = None
        while rclpy.ok() and time.monotonic() < deadline:
            if future is None and self.nomotion_client.service_is_ready():
                future = self.nomotion_client.call_async(Empty.Request())
            rclpy.spin_once(self, timeout_sec=0.05)
            fresh, data = self._message_freshness(
                self.latest_amcl,
                self.latest_amcl_receipt,
                self.max_amcl_age_sec,
            )
            if fresh and self._route_transform_available():
                return RosObservation(
                    "stationary AMCL refresh",
                    True,
                    "fresh AMCL observed after stationary refresh",
                    {
                        "service_requested": future is not None,
                        **data,
                    },
                )
            if future is not None and future.done() and future.exception() is not None:
                return RosObservation(
                    "stationary AMCL refresh",
                    False,
                    f"service failed: {future.exception()}",
                    {"service_requested": True},
                )
        return RosObservation(
            "stationary AMCL refresh",
            False,
            "timed out waiting for fresh AMCL",
            {
                "service_requested": future is not None,
                "timeout_sec": self.nomotion_update_timeout_sec,
            },
        )

    def _route_transform_available(self) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                self.config.map_frame,
                self.config.base_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except TransformException:
            return False
        return True

    def _stationary_amcl_sample(self) -> StationaryAmclPoseSample | None:
        msg = self.latest_amcl
        if msg is None:
            return None
        pose = msg.pose.pose
        orientation = pose.orientation
        yaw_rad = math.atan2(
            2.0
            * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        return StationaryAmclPoseSample(
            x_m=float(pose.position.x),
            y_m=float(pose.position.y),
            yaw_rad=yaw_rad,
            covariance=tuple(float(value) for value in msg.pose.covariance),
        )

    def _observe_stationary_amcl_stability(self) -> RosObservation:
        deadline = time.monotonic() + self.nomotion_update_timeout_sec
        samples: List[StationaryAmclPoseSample] = []
        service_failures: List[str] = []
        service_request_count = 0
        observation = evaluate_latest_stationary_amcl_window(
            samples,
            required_sample_count=self.stationary_amcl_sample_count,
            max_position_spread_m=(
                self.max_stationary_amcl_position_spread_m
            ),
            max_yaw_spread_rad=self.max_stationary_amcl_yaw_spread_rad,
            max_position_std_m=self.max_stationary_amcl_position_std_m,
            max_yaw_std_rad=self.max_stationary_amcl_yaw_std_rad,
        )
        while (
            rclpy.ok()
            and not observation.ok
            and time.monotonic() < deadline
        ):
            while (
                rclpy.ok()
                and not self.nomotion_client.service_is_ready()
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(self, timeout_sec=0.05)
            if not self.nomotion_client.service_is_ready():
                break
            baseline_receipt_ns = (
                None
                if self.latest_amcl_receipt is None
                else self.latest_amcl_receipt.nanoseconds
            )
            future = self.nomotion_client.call_async(Empty.Request())
            service_request_count += 1
            sample = None
            sample_deadline = min(
                deadline,
                time.monotonic()
                + max(1.0, 2.0 * self.stationary_amcl_sample_interval_sec),
            )
            while rclpy.ok() and time.monotonic() < sample_deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if future.done() and future.exception() is not None:
                    service_failures.append(str(future.exception()))
                    break
                receipt_ns = (
                    None
                    if self.latest_amcl_receipt is None
                    else self.latest_amcl_receipt.nanoseconds
                )
                fresh, _data = self._message_freshness(
                    self.latest_amcl,
                    self.latest_amcl_receipt,
                    self.max_amcl_age_sec,
                )
                if (
                    receipt_ns is not None
                    and receipt_ns != baseline_receipt_ns
                    and fresh
                    and self._route_transform_available()
                ):
                    sample = self._stationary_amcl_sample()
                    break
            if sample is None:
                service_failures.append(
                    "no fresh AMCL publication followed no-motion request"
                )
            else:
                samples.append(sample)
                observation = evaluate_latest_stationary_amcl_window(
                    samples,
                    required_sample_count=self.stationary_amcl_sample_count,
                    max_position_spread_m=(
                        self.max_stationary_amcl_position_spread_m
                    ),
                    max_yaw_spread_rad=(
                        self.max_stationary_amcl_yaw_spread_rad
                    ),
                    max_position_std_m=(
                        self.max_stationary_amcl_position_std_m
                    ),
                    max_yaw_std_rad=(
                        self.max_stationary_amcl_yaw_std_rad
                    ),
                )
            interval_deadline = min(
                deadline,
                time.monotonic() + self.stationary_amcl_sample_interval_sec,
            )
            while rclpy.ok() and time.monotonic() < interval_deadline:
                rclpy.spin_once(self, timeout_sec=0.05)

        self.stationary_amcl_samples = list(
            samples[-self.stationary_amcl_sample_count :]
        )
        return RosObservation(
            observation.name,
            observation.ok,
            observation.detail,
            {
                **observation.data,
                "service_request_count": service_request_count,
                "service_failures": service_failures,
                "timeout_sec": self.nomotion_update_timeout_sec,
                "sample_interval_sec": (
                    self.stationary_amcl_sample_interval_sec
                ),
            },
        )

    def _observe_topic(
        self,
        observations: List[RosObservation],
        failures: List[str],
        topic: str,
        expected_type: str,
    ) -> None:
        types = _topic_types(self, topic)
        ok = expected_type in types
        detail = f"types={types or 'none'}"
        observations.append(RosObservation(topic, ok, detail, {"expected_type": expected_type}))
        if not ok:
            failures.append(f"{topic} missing expected type {expected_type}")

    def _observe_fresh_message(
        self,
        observations: List[RosObservation],
        failures: List[str],
        name: str,
        msg,
        receipt,
        max_age_sec: float,
    ) -> None:
        if msg is None or receipt is None:
            observations.append(RosObservation(name, False, "no message received"))
            failures.append(f"{name}: no message received")
            return
        now = self.get_clock().now()
        receipt_age = (now - receipt).nanoseconds / 1_000_000_000.0
        header_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1_000_000_000.0
        ok = (
            -self.max_future_timestamp_sec <= receipt_age <= max_age_sec
            and -self.max_future_timestamp_sec <= header_age <= max_age_sec
        )
        observations.append(
            RosObservation(
                name,
                ok,
                f"receipt_age={receipt_age:.3f}s header_age={header_age:.3f}s",
                {
                    "receipt_age_sec": receipt_age,
                    "header_age_sec": header_age,
                    "max_future_sec": self.max_future_timestamp_sec,
                },
            )
        )
        if not ok:
            failures.append(
                f"{name}: "
                + (
                    "future-dated message"
                    if receipt_age < -self.max_future_timestamp_sec
                    or header_age < -self.max_future_timestamp_sec
                    else "stale message"
                )
            )

    def _message_freshness(self, msg, receipt, max_age_sec: float) -> Tuple[bool, Dict[str, object]]:
        if msg is None or receipt is None:
            return False, {"received": False}
        now = self.get_clock().now()
        receipt_age = (now - receipt).nanoseconds / 1_000_000_000.0
        header_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1_000_000_000.0
        ok = (
            -self.max_future_timestamp_sec <= receipt_age <= max_age_sec
            and -self.max_future_timestamp_sec <= header_age <= max_age_sec
        )
        return ok, {
            "received": True,
            "receipt_age_sec": receipt_age,
            "header_age_sec": header_age,
            "max_future_sec": self.max_future_timestamp_sec,
        }

    def _observe_tf(
        self,
        observations: List[RosObservation],
        failures: List[str],
        target_frame: str,
        source_frame: str,
        max_age_sec: float,
        *,
        max_future_sec: float | None = None,
    ) -> Tuple[bool, Dict[str, object]]:
        name = f"tf {target_frame}->{source_frame}"
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            data = {"available": False, "error": str(exc)}
            observations.append(RosObservation(name, False, str(exc), data))
            failures.append(f"{name}: unavailable")
            return False, data
        capture_time = self.get_clock().now()
        transform_stamp = Time.from_msg(transform.header.stamp)
        age = (capture_time - transform_stamp).nanoseconds / 1_000_000_000.0
        accepted_future_sec = (
            self.max_future_timestamp_sec
            if max_future_sec is None
            else max_future_sec
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        observed_target_frame = _frame_id(
            str(getattr(transform.header, "frame_id", ""))
        )
        observed_source_frame = _frame_id(
            str(getattr(transform, "child_frame_id", ""))
        )
        frame_identity_ok = (
            observed_target_frame == _frame_id(target_frame)
            and observed_source_frame == _frame_id(source_frame)
        )
        quaternion = tuple(
            float(value)
            for value in (rotation.x, rotation.y, rotation.z, rotation.w)
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        quaternion_ok = (
            all(math.isfinite(value) for value in quaternion)
            and abs(quaternion_norm - 1.0) <= 1.0e-3
        )
        yaw_rad = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        data = {
            "available": True,
            "age_sec": age,
            "max_future_sec": accepted_future_sec,
            "target_frame": target_frame,
            "source_frame": source_frame,
            "observed_target_frame": observed_target_frame,
            "observed_source_frame": observed_source_frame,
            "stamp_sec": transform_stamp.nanoseconds / 1_000_000_000.0,
            "capture_time_sec": capture_time.nanoseconds / 1_000_000_000.0,
            "x_m": float(translation.x),
            "y_m": float(translation.y),
            "yaw_rad": yaw_rad,
            "quaternion": {
                "x": quaternion[0],
                "y": quaternion[1],
                "z": quaternion[2],
                "w": quaternion[3],
                "norm": quaternion_norm,
            },
        }
        pose_values_ok = all(
            math.isfinite(value)
            for value in (
                data["x_m"],
                data["y_m"],
                data["yaw_rad"],
            )
        )
        ok = (
            -accepted_future_sec <= age <= max_age_sec
            and frame_identity_ok
            and quaternion_ok
            and pose_values_ok
        )
        observations.append(RosObservation(name, ok, f"age={age:.3f}s", data))
        if not ok:
            if not frame_identity_ok:
                failure = "transform frame identity mismatch"
            elif not quaternion_ok:
                failure = "invalid transform quaternion"
            elif not pose_values_ok:
                failure = "non-finite transform"
            elif age < -accepted_future_sec:
                failure = "future-dated transform"
            else:
                failure = "stale transform"
            failures.append(f"{name}: {failure}")
        return ok, data

    def _observe_localization_ownership(
        self,
        observations: List[RosObservation],
        failures: List[str],
        *,
        route_transform_fresh: bool,
        route_transform_data: Dict[str, object],
        odom_to_base_fresh: bool,
        odom_to_base_data: Dict[str, object],
    ) -> None:
        amcl_fresh, amcl_data = self._message_freshness(
            self.latest_amcl,
            self.latest_amcl_receipt,
            self.max_amcl_age_sec,
        )
        map_to_odom_fresh, map_to_odom_data = self._dynamic_map_to_odom_freshness()
        owner_candidates = self._external_tf_owner_candidates()
        evidence = LocalizationOwnershipEvidence(
            localization_source=self.config.localization_source,
            amcl_fresh=amcl_fresh,
            map_to_odom_dynamic_fresh=map_to_odom_fresh,
            route_transform_fresh=route_transform_fresh,
            odom_to_base_fresh=odom_to_base_fresh,
            route_uses_odom_frame=(
                self.execution_pose_owner == "odom"
                or self.config.map_frame == self.config.odom_frame
            ),
            external_tf_owner_candidates=owner_candidates,
            execution_pose_owner=self.execution_pose_owner,
            global_consistency_monitor=self.global_consistency_monitor,
            frozen_map_transform_certified=(
                self.frozen_map_transform_certified
            ),
        )
        decision = evaluate_localization_ownership(evidence)
        data = build_localization_ownership_observation_data(
            decision_data=decision.data,
            map_frame=self.config.map_frame,
            odom_frame=self.config.odom_frame,
            base_frame=self.config.base_frame,
            amcl_topic=self.config.amcl_topic,
            dynamic_tf_topics=self.dynamic_tf_topics,
            amcl_data=amcl_data,
            map_to_odom_dynamic_data=map_to_odom_data,
            route_transform_data=route_transform_data,
            odom_to_base_data=odom_to_base_data,
        )
        observations.append(
            RosObservation(
                "localization transform ownership",
                decision.ok,
                "ok" if decision.ok else decision.failure,
                data,
            )
        )
        if not decision.ok:
            failures.append(decision.failure)

    def _dynamic_map_to_odom_freshness(self) -> Tuple[bool, Dict[str, object]]:
        if self.latest_dynamic_map_to_odom is None or self.latest_dynamic_map_to_odom_receipt is None:
            return build_dynamic_map_to_odom_freshness(
                has_dynamic_transform=False,
                receipt_age_sec=None,
                header_age_sec=None,
                max_age_sec=self.max_tf_age_sec,
            )
        now = self.get_clock().now()
        receipt_age = (now - self.latest_dynamic_map_to_odom_receipt).nanoseconds / 1_000_000_000.0
        header_age = (
            now - Time.from_msg(self.latest_dynamic_map_to_odom.header.stamp)
        ).nanoseconds / 1_000_000_000.0
        return build_dynamic_map_to_odom_freshness(
            has_dynamic_transform=True,
            receipt_age_sec=receipt_age,
            header_age_sec=header_age,
            max_age_sec=self.max_tf_age_sec,
            max_future_sec=self.max_localization_tf_future_sec,
        )

    def _external_tf_owner_candidates(self) -> List[str]:
        node_items = []
        try:
            node_items = self.get_node_names_and_namespaces()
        except AttributeError:
            node_items = []
        topic_names = [topic for topic, _types in self.get_topic_names_and_types()]
        try:
            service_items = self.get_service_names_and_types()
        except AttributeError:
            service_items = []
        service_names = [service for service, _types in service_items]
        return find_external_tf_owner_candidates(
            resolved_namespace=self.config.namespace,
            node_items=node_items,
            topic_names=topic_names,
            service_names=service_names,
        )

    def _observe_use_sim_time(self, observations: List[RosObservation], failures: List[str]) -> None:
        value = bool(self.get_parameter("use_sim_time").value)
        ok = (not self.require_real_time) or value is False
        observations.append(RosObservation("use_sim_time", ok, str(value), {"value": value}))
        if not ok:
            failures.append("use_sim_time must be false for real robot runs")

    def _observe_cmd_vel_ownership(
        self,
        observations: List[RosObservation],
        failures: List[str],
    ) -> None:
        publishers = self.get_publishers_info_by_topic(self.config.cmd_vel_topic)
        publisher_identities = sorted({_node_identity(publisher) for publisher in publishers})
        active_nav2 = self._has_active_nav2_goal()
        nav2_status_observed = self.latest_nav2_status is not None
        allowed = set(self.allowed_cmd_vel_publishers)
        unknown_publishers = [
            identity
            for identity in publisher_identities
            if identity not in allowed
        ]
        ok = not active_nav2 and not unknown_publishers
        observations.append(
            RosObservation(
                "cmd_vel ownership",
                ok,
                (
                    f"publishers={publisher_identities} active_nav2={active_nav2} "
                    f"nav2_status_observed={nav2_status_observed}"
                ),
                {
                    "cmd_vel_topic": self.config.cmd_vel_topic,
                    "publishers": publisher_identities,
                    "active_nav2_goal": active_nav2,
                    "nav2_status_observed": nav2_status_observed,
                    "allowed_publishers": sorted(allowed),
                },
            )
        )
        if active_nav2:
            failures.append("active Nav2 goal/controller detected")
        if unknown_publishers:
            failures.append(f"unapproved cmd_vel publishers: {', '.join(unknown_publishers)}")

    def _has_active_nav2_goal(self) -> bool:
        if self.latest_nav2_status is None:
            return False
        return any(status.status in ACTIVE_GOAL_STATUS for status in self.latest_nav2_status.status_list)


def run_ros_preflight(
    config: ResolvedRuntimeConfig,
    *,
    max_scan_age_sec: float = 1.0,
    max_odom_age_sec: float = 1.0,
    max_tf_age_sec: float = 1.0,
    max_amcl_age_sec: float = 2.0,
    max_future_timestamp_sec: float = 0.25,
    max_localization_tf_future_sec: float | None = None,
    observation_window_sec: float = 2.0,
    allow_idle_nav2: bool = False,
    allowed_cmd_vel_publishers: Sequence[str] = (),
    require_real_time: bool = True,
    request_nomotion_update: bool = False,
    nomotion_update_service: str = "/request_nomotion_update",
    nomotion_update_timeout_sec: float = 15.0,
    stationary_amcl_sample_count: int = 5,
    stationary_amcl_sample_interval_sec: float = 0.5,
    max_stationary_amcl_position_spread_m: float = 0.015,
    max_stationary_amcl_yaw_spread_rad: float = 0.03,
    max_stationary_amcl_position_std_m: float = 0.015,
    max_stationary_amcl_yaw_std_rad: float = 0.03,
    execution_pose_owner: str = "",
    global_consistency_monitor: str = "",
    frozen_map_transform_certified: bool = False,
) -> RosPreflightResult:
    if (
        not math.isfinite(max_future_timestamp_sec)
        or max_future_timestamp_sec < 0.0
    ):
        raise ValueError("max_future_timestamp_sec must be finite and non-negative")
    localization_tf_future_sec = (
        max_future_timestamp_sec
        if max_localization_tf_future_sec is None
        else max_localization_tf_future_sec
    )
    if (
        not math.isfinite(localization_tf_future_sec)
        or localization_tf_future_sec < 0.0
    ):
        raise ValueError(
            "max_localization_tf_future_sec must be finite and non-negative"
        )
    if (
        not math.isfinite(nomotion_update_timeout_sec)
        or nomotion_update_timeout_sec <= 0.0
    ):
        raise ValueError("nomotion_update_timeout_sec must be finite and positive")
    if (
        not isinstance(stationary_amcl_sample_count, int)
        or isinstance(stationary_amcl_sample_count, bool)
        or stationary_amcl_sample_count < 2
    ):
        raise ValueError("stationary_amcl_sample_count must be an integer >= 2")
    for name, value in {
        "stationary_amcl_sample_interval_sec": (
            stationary_amcl_sample_interval_sec
        ),
        "max_stationary_amcl_position_spread_m": (
            max_stationary_amcl_position_spread_m
        ),
        "max_stationary_amcl_yaw_spread_rad": (
            max_stationary_amcl_yaw_spread_rad
        ),
        "max_stationary_amcl_position_std_m": (
            max_stationary_amcl_position_std_m
        ),
        "max_stationary_amcl_yaw_std_rad": (
            max_stationary_amcl_yaw_std_rad
        ),
    }.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    _require_ros()
    rclpy.init(args=None)
    node = RosPreflightNode(
        config,
        max_scan_age_sec=max_scan_age_sec,
        max_odom_age_sec=max_odom_age_sec,
        max_tf_age_sec=max_tf_age_sec,
        max_amcl_age_sec=max_amcl_age_sec,
        max_future_timestamp_sec=max_future_timestamp_sec,
        max_localization_tf_future_sec=localization_tf_future_sec,
        observation_window_sec=observation_window_sec,
        allow_idle_nav2=allow_idle_nav2,
        allowed_cmd_vel_publishers=allowed_cmd_vel_publishers,
        require_real_time=require_real_time,
        request_nomotion_update=request_nomotion_update,
        nomotion_update_service=nomotion_update_service,
        nomotion_update_timeout_sec=nomotion_update_timeout_sec,
        stationary_amcl_sample_count=stationary_amcl_sample_count,
        stationary_amcl_sample_interval_sec=(
            stationary_amcl_sample_interval_sec
        ),
        max_stationary_amcl_position_spread_m=(
            max_stationary_amcl_position_spread_m
        ),
        max_stationary_amcl_yaw_spread_rad=(
            max_stationary_amcl_yaw_spread_rad
        ),
        max_stationary_amcl_position_std_m=(
            max_stationary_amcl_position_std_m
        ),
        max_stationary_amcl_yaw_std_rad=max_stationary_amcl_yaw_std_rad,
        execution_pose_owner=execution_pose_owner,
        global_consistency_monitor=global_consistency_monitor,
        frozen_map_transform_certified=frozen_map_transform_certified,
    )
    try:
        return node.collect()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--localization-source", default="amcl", choices=["amcl", "tf"])
    parser.add_argument("--allow-sim-time", action="store_true")
    parser.add_argument("--max-future-timestamp-sec", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=[],
        help="Namespace-qualified node identity allowed in preflight, e.g. /robot1/controller_server",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.max_future_timestamp_sec)
        or args.max_future_timestamp_sec < 0.0
    ):
        parser.error("--max-future-timestamp-sec must be non-negative")
    runtime_config = resolve_runtime_config(
        RuntimeConfig(
            namespace=args.namespace,
            scan_topic=args.scan_topic,
            odom_topic=args.odom_topic,
            cmd_vel_topic=args.cmd_vel_topic,
            amcl_topic=args.amcl_topic,
            map_frame=args.map_frame,
            odom_frame=args.odom_frame,
            base_frame=args.base_frame,
            localization_source=args.localization_source,
            use_sim_time=args.allow_sim_time,
        )
    )
    try:
        result = run_ros_preflight(
            runtime_config,
            max_future_timestamp_sec=args.max_future_timestamp_sec,
            allowed_cmd_vel_publishers=args.allowed_cmd_vel_publisher,
            require_real_time=not args.allow_sim_time,
        )
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    text = json.dumps(result.to_json_dict(), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
