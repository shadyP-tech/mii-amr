"""ROS-free localization evidence normalization for Aufgabe 04 preflight."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple


NodeNameNamespace = Tuple[str, str]


def find_external_tf_owner_candidates(
    *,
    resolved_namespace: str,
    node_items: Sequence[NodeNameNamespace],
    topic_names: Sequence[str],
    service_names: Sequence[str],
) -> list[str]:
    """Return namespace-scoped graph evidence for active external TF owners."""
    candidates = set()
    for name, node_namespace in node_items:
        if _is_namespace_scoped(node_namespace, resolved_namespace) and "slam_toolbox" in name:
            candidates.add(_node_identity_from_names(node_namespace, name))
    for topic in topic_names:
        if _name_in_namespace(topic, resolved_namespace) and "slam_toolbox" in topic:
            candidates.add(topic)
    for service in service_names:
        if _name_in_namespace(service, resolved_namespace) and "slam_toolbox" in service:
            candidates.add(service)
    return sorted(candidates)


def build_dynamic_map_to_odom_freshness(
    *,
    has_dynamic_transform: bool,
    receipt_age_sec: float | None,
    header_age_sec: float | None,
    max_age_sec: float,
    max_future_sec: float = 0.25,
) -> tuple[bool, Dict[str, object]]:
    """Evaluate dynamic /tf map->odom evidence from primitive age values."""
    if not has_dynamic_transform or receipt_age_sec is None or header_age_sec is None:
        return False, {"available": False, "dynamic": False}
    timestamps_valid = math.isfinite(receipt_age_sec) and math.isfinite(
        header_age_sec
    )
    ok = (
        timestamps_valid
        and -max_future_sec <= receipt_age_sec <= max_age_sec
        and -max_future_sec <= header_age_sec <= max_age_sec
    )
    return ok, {
        "available": True,
        "dynamic": True,
        "receipt_age_sec": receipt_age_sec,
        "header_age_sec": header_age_sec,
        "max_future_sec": max_future_sec,
        "future_dated": (
            timestamps_valid
            and (
                receipt_age_sec < -max_future_sec
                or header_age_sec < -max_future_sec
            )
        ),
    }


def build_localization_ownership_observation_data(
    *,
    decision_data: Dict[str, object],
    map_frame: str,
    odom_frame: str,
    base_frame: str,
    amcl_topic: str,
    dynamic_tf_topics: Sequence[str],
    amcl_data: Dict[str, object],
    map_to_odom_dynamic_data: Dict[str, object],
    route_transform_data: Dict[str, object],
    odom_to_base_data: Dict[str, object],
) -> Dict[str, object]:
    """Merge ownership decision data with localization preflight observations."""
    data = dict(decision_data)
    data.update(
        {
            "map_frame": map_frame,
            "odom_frame": odom_frame,
            "base_frame": base_frame,
            "amcl_topic": amcl_topic,
            "dynamic_tf_topics": list(dynamic_tf_topics),
            "amcl": amcl_data,
            "map_to_odom_dynamic": map_to_odom_dynamic_data,
            "route_transform": route_transform_data,
            "odom_to_base": odom_to_base_data,
        }
    )
    return data


def _node_identity_from_names(namespace: str, name: str) -> str:
    if namespace in ("", "/"):
        return f"/{name}"
    return f"{namespace.rstrip('/')}/{name}"


def _is_namespace_scoped(node_namespace: str, resolved_namespace: str) -> bool:
    expected = resolved_namespace or "/"
    actual = node_namespace or "/"
    return actual.rstrip("/") == expected.rstrip("/")


def _name_in_namespace(name: str, resolved_namespace: str) -> bool:
    if not resolved_namespace:
        first_segment = name.strip("/").split("/", 1)[0]
        return "slam_toolbox" in first_segment
    return name == resolved_namespace or name.startswith(f"{resolved_namespace}/")
