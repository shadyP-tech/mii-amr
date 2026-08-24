"""Runtime topic and frame normalization for Aufgabe 04 ROS tools."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict


def _clean_namespace(namespace: str) -> str:
    namespace = namespace.strip()
    if not namespace or namespace == "/":
        return ""
    return "/" + namespace.strip("/")


def resolve_topic(topic: str, namespace: str = "") -> str:
    if not topic:
        raise ValueError("topic name must not be empty")
    if topic.startswith("/"):
        return topic
    cleaned_namespace = _clean_namespace(namespace)
    if cleaned_namespace:
        return f"{cleaned_namespace}/{topic.strip('/')}"
    return f"/{topic.strip('/')}"


@dataclass(frozen=True)
class RuntimeConfig:
    namespace: str = ""
    scan_topic: str = "scan"
    odom_topic: str = "odom"
    cmd_vel_topic: str = "cmd_vel"
    amcl_topic: str = "amcl_pose"
    map_frame: str = "map"
    odom_frame: str = "odom"
    base_frame: str = "base_footprint"
    localization_source: str = "amcl"
    use_sim_time: bool = False
    ros_domain_id: str = ""


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    configured: RuntimeConfig
    namespace: str
    scan_topic: str
    odom_topic: str
    cmd_vel_topic: str
    amcl_topic: str
    map_frame: str
    odom_frame: str
    base_frame: str
    localization_source: str
    use_sim_time: bool
    ros_domain_id: str

    def as_log_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["configured"] = asdict(self.configured)
        return payload


def resolve_runtime_config(config: RuntimeConfig) -> ResolvedRuntimeConfig:
    namespace = _clean_namespace(config.namespace)
    ros_domain_id = config.ros_domain_id or os.environ.get("ROS_DOMAIN_ID", "")
    return ResolvedRuntimeConfig(
        configured=config,
        namespace=namespace,
        scan_topic=resolve_topic(config.scan_topic, namespace),
        odom_topic=resolve_topic(config.odom_topic, namespace),
        cmd_vel_topic=resolve_topic(config.cmd_vel_topic, namespace),
        amcl_topic=resolve_topic(config.amcl_topic, namespace),
        map_frame=config.map_frame,
        odom_frame=config.odom_frame,
        base_frame=config.base_frame,
        localization_source=config.localization_source,
        use_sim_time=config.use_sim_time,
        ros_domain_id=ros_domain_id,
    )
