"""Pure Aufgabe 04 navigation safety gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class PreflightStatus:
    ok: bool
    failures: List[str]


def validate_required_topics(available_topics: Iterable[str], required_topics: Iterable[str]) -> PreflightStatus:
    available = set(available_topics)
    missing = [topic for topic in required_topics if topic not in available]
    return PreflightStatus(ok=not missing, failures=[f"missing topic: {topic}" for topic in missing])


def validate_route_diagnostics_json(
    path: Path,
    leg_index: int,
    *,
    csv_point_count: int,
    require_motion: bool = True,
) -> PreflightStatus:
    failures: List[str] = []
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return PreflightStatus(ok=False, failures=[f"invalid diagnostics JSON: {exc}"])

    legs = payload.get("legs")
    if not isinstance(legs, list):
        return PreflightStatus(ok=False, failures=["diagnostics JSON missing legs list"])
    if leg_index < 0 or leg_index >= len(legs):
        return PreflightStatus(ok=False, failures=[f"diagnostics missing leg_index {leg_index}"])

    leg = legs[leg_index]
    if not isinstance(leg, dict):
        return PreflightStatus(ok=False, failures=[f"diagnostics leg {leg_index} must be an object"])
    diagnostics = leg.get("diagnostics")
    if not isinstance(diagnostics, dict):
        failures.append(f"diagnostics leg {leg_index} missing diagnostics object")
    elif diagnostics.get("status") != "ok":
        failures.append(f"diagnostics leg {leg_index} status is not ok")
    if leg.get("failure") is not None:
        failures.append(f"diagnostics leg {leg_index} has failure")

    route_point_count = leg.get("route_point_count")
    if route_point_count != csv_point_count:
        failures.append(
            f"diagnostics leg {leg_index} route_point_count {route_point_count} "
            f"does not match CSV count {csv_point_count}"
        )
    route_length = leg.get("route_length_m")
    if not isinstance(route_length, (int, float)) or not math.isfinite(route_length):
        failures.append(f"diagnostics leg {leg_index} route_length_m must be finite")
    elif require_motion and route_length <= 0.0:
        failures.append(f"diagnostics leg {leg_index} route_length_m must be positive for motion")

    return PreflightStatus(ok=not failures, failures=failures)


def validate_speed_limits(
    max_linear_mps: float,
    max_angular_radps: float,
    *,
    min_linear_mps: float = 0.0,
    max_allowed_linear_mps: float = 0.06,
    min_angular_radps: float = 0.0,
    max_allowed_angular_radps: float = 0.20,
) -> PreflightStatus:
    failures: List[str] = []
    values = {
        "max_linear_mps": max_linear_mps,
        "max_angular_radps": max_angular_radps,
    }
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{name} must be finite")
    if not failures:
        if max_linear_mps <= min_linear_mps:
            failures.append("max_linear_mps must be positive")
        if max_linear_mps > max_allowed_linear_mps:
            failures.append(f"max_linear_mps exceeds {max_allowed_linear_mps:.3f} m/s")
        if max_angular_radps <= min_angular_radps:
            failures.append("max_angular_radps must be positive")
        if max_angular_radps > max_allowed_angular_radps:
            failures.append(f"max_angular_radps exceeds {max_allowed_angular_radps:.3f} rad/s")
    return PreflightStatus(ok=not failures, failures=failures)
