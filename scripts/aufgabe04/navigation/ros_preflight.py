"""ROS2 preflight observations for Aufgabe 04 station-segment runs.

This module observes ROS graph, topic, and TF state. It never publishes motion.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from scripts.aufgabe04.navigation.ros_runtime_config import (
    ResolvedRuntimeConfig,
    RuntimeConfig,
    resolve_runtime_config,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from action_msgs.msg import GoalStatusArray
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    GoalStatusArray = None
    LaserScan = None
    Odometry = None
    PoseWithCovarianceStamped = None
    Duration = None
    Node = object
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


ACTIVE_GOAL_STATUS = {1, 2, 3, 4}


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


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _topic_type_names(node: Node, topic: str) -> List[str]:
    return [item[0] for item in node.get_topic_names_and_types() if item[0] == topic for _ in item[1]]


def _topic_types(node: Node, topic: str) -> List[str]:
    for name, types in node.get_topic_names_and_types():
        if name == topic:
            return list(types)
    return []


class RosPreflightNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(
        self,
        config: ResolvedRuntimeConfig,
        *,
        max_scan_age_sec: float,
        max_odom_age_sec: float,
        max_tf_age_sec: float,
        max_amcl_age_sec: float,
        observation_window_sec: float,
        allow_idle_nav2: bool,
        allowed_cmd_vel_publishers: Sequence[str],
        require_real_time: bool,
    ) -> None:
        super().__init__("aufgabe04_ros_preflight")
        self.config = config
        self.max_scan_age_sec = max_scan_age_sec
        self.max_odom_age_sec = max_odom_age_sec
        self.max_tf_age_sec = max_tf_age_sec
        self.max_amcl_age_sec = max_amcl_age_sec
        self.observation_window_sec = observation_window_sec
        self.allow_idle_nav2 = allow_idle_nav2
        self.allowed_cmd_vel_publishers = tuple(allowed_cmd_vel_publishers)
        self.require_real_time = require_real_time
        self.declare_parameter("use_sim_time", False)
        self.latest_scan = None
        self.latest_scan_receipt = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.latest_amcl = None
        self.latest_amcl_receipt = None
        self.latest_nav2_status = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, config.scan_topic, self._scan_callback, 10)
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
        self._observe_tf(
            observations,
            failures,
            self.config.map_frame,
            self.config.base_frame,
            self.max_tf_age_sec,
        )
        self._observe_tf(
            observations,
            failures,
            self.config.odom_frame,
            self.config.base_frame,
            self.max_tf_age_sec,
        )
        self._observe_use_sim_time(observations, failures)
        self._observe_cmd_vel_ownership(observations, failures)
        return RosPreflightResult(
            ok=not failures,
            failures=failures,
            observations=observations,
            runtime_config=self.config.as_log_dict(),
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
        ok = receipt_age <= max_age_sec and header_age <= max_age_sec
        observations.append(
            RosObservation(
                name,
                ok,
                f"receipt_age={receipt_age:.3f}s header_age={header_age:.3f}s",
                {"receipt_age_sec": receipt_age, "header_age_sec": header_age},
            )
        )
        if not ok:
            failures.append(f"{name}: stale message")

    def _observe_tf(
        self,
        observations: List[RosObservation],
        failures: List[str],
        target_frame: str,
        source_frame: str,
        max_age_sec: float,
    ) -> None:
        name = f"tf {target_frame}->{source_frame}"
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            observations.append(RosObservation(name, False, str(exc)))
            failures.append(f"{name}: unavailable")
            return
        age = (self.get_clock().now() - Time.from_msg(transform.header.stamp)).nanoseconds / 1_000_000_000.0
        ok = age <= max_age_sec
        observations.append(RosObservation(name, ok, f"age={age:.3f}s", {"age_sec": age}))
        if not ok:
            failures.append(f"{name}: stale transform")

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
        publisher_names = sorted({publisher.node_name for publisher in publishers})
        active_nav2 = self._has_active_nav2_goal()
        nav2_publishers = [name for name in publisher_names if "nav" in name.lower()]
        nav2_status_observed = self.latest_nav2_status is not None
        allowed = set(self.allowed_cmd_vel_publishers)
        unknown_publishers = [
            name
            for name in publisher_names
            if name not in allowed and not (self.allow_idle_nav2 and "nav" in name.lower())
        ]
        ambiguous_nav2 = bool(nav2_publishers) and self.allow_idle_nav2 and not nav2_status_observed
        ok = not active_nav2 and not unknown_publishers and not ambiguous_nav2
        observations.append(
            RosObservation(
                "cmd_vel ownership",
                ok,
                (
                    f"publishers={publisher_names} active_nav2={active_nav2} "
                    f"nav2_status_observed={nav2_status_observed}"
                ),
                {
                    "cmd_vel_topic": self.config.cmd_vel_topic,
                    "publishers": publisher_names,
                    "active_nav2_goal": active_nav2,
                    "nav2_status_observed": nav2_status_observed,
                    "allowed_publishers": sorted(allowed),
                },
            )
        )
        if active_nav2:
            failures.append("active Nav2 goal/controller detected")
        if ambiguous_nav2:
            failures.append("Nav2 cmd_vel publisher present but no NavigateToPose status observed")
        if unknown_publishers:
            failures.append(f"unapproved cmd_vel publishers: {', '.join(unknown_publishers)}")
        if publishers and not ok:
            return
        if publisher_names and not self.allow_idle_nav2 and not allowed:
            failures.append("cmd_vel publishers present and no allowlist configured")

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
    observation_window_sec: float = 2.0,
    allow_idle_nav2: bool = True,
    allowed_cmd_vel_publishers: Sequence[str] = (),
    require_real_time: bool = True,
) -> RosPreflightResult:
    _require_ros()
    rclpy.init(args=None)
    node = RosPreflightNode(
        config,
        max_scan_age_sec=max_scan_age_sec,
        max_odom_age_sec=max_odom_age_sec,
        max_tf_age_sec=max_tf_age_sec,
        max_amcl_age_sec=max_amcl_age_sec,
        observation_window_sec=observation_window_sec,
        allow_idle_nav2=allow_idle_nav2,
        allowed_cmd_vel_publishers=allowed_cmd_vel_publishers,
        require_real_time=require_real_time,
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
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--allowed-cmd-vel-publisher", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
