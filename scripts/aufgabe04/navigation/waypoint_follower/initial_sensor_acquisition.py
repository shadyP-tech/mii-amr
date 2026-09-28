"""First-delivery eligibility and bounded diagnostics for stopped startup."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
import threading


@dataclass
class InitialSensorAcquisition:
    sensors: dict[str, dict[str, object]] = field(default_factory=dict)
    executor_health: dict[str, object] = field(default_factory=dict)
    graph: dict[str, object] = field(default_factory=dict)

    def record(self, name, *, has_message, failure, details):
        sensor = self.sensors.setdefault(name, {
            "ever_received": False, "non_acquisition_failure_seen": False,
        })
        missing = (
            not has_message and bool(failure)
            and details.get("source") == "message_freshness"
            and details.get("sensor") == name and details.get("has_message") is False
        )
        sensor["non_acquisition_failure_seen"] |= bool(failure) and (
            not missing or sensor["ever_received"]
        )
        sensor["ever_received"] |= has_message
        sensor.update(current_fresh=not bool(failure), missing_first_message=missing,
                      last_failure=dict(details) if failure else {})

    def waiting_only_for_first_delivery(self) -> bool:
        return (
            set(self.sensors) == {"scan", "odom"}
            and any(s["missing_first_message"] for s in self.sensors.values())
            and all(
                not s["non_acquisition_failure_seen"]
                and (s["current_fresh"] or (
                    s["missing_first_message"] and not s["ever_received"]
                )) for s in self.sensors.values()
            )
        )

    def to_evidence(self):
        return {
            "sensors": {name: dict(value) for name, value in self.sensors.items()},
            "executor_health": dict(self.executor_health),
            "publisher_graph": dict(self.graph),
        }


class SensorReceipts:
    """Two counters only; no message storage or ROS graph work in callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sensors = {name: {"count": 0, "first_receipt_sec": None,
                                "last_receipt_sec": None} for name in ("scan", "odom")}

    def record(self, name, receipt):
        with self._lock:
            sensor = self._sensors[name]
            if sensor["count"] == 0:
                sensor["first_receipt_sec"] = receipt
            sensor["count"] += 1
            sensor["last_receipt_sec"] = receipt

    def snapshot(self):
        with self._lock:
            return {name: dict(value) for name, value in self._sensors.items()}


def publisher_diagnostics(node):
    """Diagnostic only, queried at phase transitions/failure, never per frame.

    A matched publisher is not proof of delivery or permission to move. Bound
    saved endpoint evidence and retain graph-query errors without hiding the
    original startup failure.
    """
    result = {}
    for name, subscription in getattr(node, "initial_sensor_subscriptions", {}).items():
        entry = result[name] = {}
        try:
            entry["topic"] = subscription.topic_name
            entry["requested_qos"] = _qos(subscription.qos_profile)
            entry["matched_publisher_count"] = subscription.get_publisher_count()
            publishers = node.get_publishers_info_by_topic(subscription.topic_name)
            entry["discovered_publisher_count"] = len(publishers)
            entry["publishers"] = [
                {"node_name": p.node_name, "node_namespace": p.node_namespace,
                 "offered_qos": _qos(p.qos_profile)} for p in islice(publishers, 8)
            ]
        except Exception as exc:
            entry["diagnostic_error"] = f"{type(exc).__name__}: {exc}"[:512]
    return result


def _qos(profile):
    return {name: int(getattr(profile, name))
            for name in ("reliability", "durability", "history", "depth")}
