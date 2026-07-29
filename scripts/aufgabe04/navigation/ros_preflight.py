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
class RosPreflightResult:
    ok: bool
    failures: List[str]
    observations: List[RosObservation]
    runtime_config: Dict[str, object]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "failures": self.failures,
            "observations": [asdict(observation) for observation in self.observations],
            "runtime_config": self.runtime_config,
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
        return RosPreflightResult(
            ok=not failures,
            failures=failures,
            observations=observations,
            runtime_config=self.config.as_log_dict(),
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
        age = (self.get_clock().now() - Time.from_msg(transform.header.stamp)).nanoseconds / 1_000_000_000.0
        ok = -self.max_future_timestamp_sec <= age <= max_age_sec
        data = {
            "available": True,
            "age_sec": age,
            "max_future_sec": self.max_future_timestamp_sec,
        }
        observations.append(RosObservation(name, ok, f"age={age:.3f}s", data))
        if not ok:
            failures.append(
                f"{name}: "
                + (
                    "future-dated transform"
                    if age < -self.max_future_timestamp_sec
                    else "stale transform"
                )
            )
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
            route_uses_odom_frame=self.config.map_frame == self.config.odom_frame,
            external_tf_owner_candidates=owner_candidates,
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
