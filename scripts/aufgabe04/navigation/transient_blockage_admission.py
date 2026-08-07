"""ROS-free orchestration for a stopped transient-blockage evidence window.

The caller owns sensors, callback service, and the zero-velocity publisher.
This module owns only the bounded sampling state machine and delegates every
individual sample's scan/TF/odom validation back to the caller.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.transient_blockage_policy import (
    PersistentObstacleConfig,
    StationaryFrontSectorSample,
    confirm_persistent_obstacle,
    confirm_stationary_clearance,
)


@dataclass(frozen=True)
class StationaryBlockageAdmission:
    """Stopped multi-scan admission result for one transient replan."""

    status: str
    pose: Pose2D | None
    front_clearance: dict[str, object] | None
    evidence: dict[str, object]


def collect_stationary_blockage_admission(
    *,
    config: PersistentObstacleConfig,
    timeout_sec: float,
    clearance_threshold_m: float,
    initial_scan_receipt: float | None,
    runtime_ok: Callable[[], bool],
    publish_zero: Callable[[], None],
    service_callbacks: Callable[[float], None],
    current_scan_receipt: Callable[[], object],
    capture_sample: Callable[
        [],
        tuple[StationaryFrontSectorSample | None, Mapping[str, object]],
    ],
    monotonic: Callable[[], float] | None = None,
) -> StationaryBlockageAdmission:
    """Collect post-stop scans until obstacle, clearance, or timeout wins."""

    if not isinstance(config, PersistentObstacleConfig):
        raise ValueError("config must be a PersistentObstacleConfig")
    if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
        raise ValueError("timeout_sec must be finite and positive")
    if not math.isfinite(clearance_threshold_m) or clearance_threshold_m <= 0.0:
        raise ValueError("clearance_threshold_m must be finite and positive")
    clock = time.monotonic if monotonic is None else monotonic
    started_at = clock()
    normalized_initial_receipt = _finite_or_none(initial_scan_receipt)
    last_scan_receipt = normalized_initial_receipt
    samples: list[StationaryFrontSectorSample] = []
    last_sample_failure: dict[str, object] = {}
    last_front_details: dict[str, object] | None = None
    obstacle = confirm_persistent_obstacle((), now_sec=started_at, config=config)
    clearance = confirm_stationary_clearance(
        (),
        now_sec=started_at,
        clearance_threshold_m=clearance_threshold_m,
        config=config,
    )

    while runtime_ok() and clock() - started_at <= timeout_sec:
        publish_zero()
        service_callbacks(min(0.04, config.min_sample_separation_sec / 2.0))
        receipt = _finite_or_none(current_scan_receipt())
        if (
            receipt is None
            or (
                last_scan_receipt is not None
                and receipt <= last_scan_receipt
            )
        ):
            continue
        last_scan_receipt = receipt
        sample, sample_details = capture_sample()
        if sample is None:
            last_sample_failure = dict(sample_details)
            continue
        samples.append(sample)
        last_front_details = dict(sample_details)
        now_sec = clock()
        obstacle = confirm_persistent_obstacle(
            samples,
            now_sec=now_sec,
            config=config,
        )
        clearance = confirm_stationary_clearance(
            samples,
            now_sec=now_sec,
            clearance_threshold_m=clearance_threshold_m,
            config=config,
        )
        common_evidence = {
            "zero_hold_duration_sec": now_sec - started_at,
            "stationary_obstacle_confirmation": obstacle.to_log_dict(),
            "stationary_clearance_confirmation": clearance.to_log_dict(),
        }
        if obstacle.confirmed:
            front_clearance = {
                **dict(last_front_details or {}),
                "nearest_valid_range_m": obstacle.median_front_range_m,
                "nearest_valid_bearing_rad": obstacle.median_front_bearing_rad,
                "source": "front_sector",
                "stationary_confirmation_status": (
                    "persistent_obstacle_confirmed"
                ),
            }
            return StationaryBlockageAdmission(
                "confirmed",
                sample.map_pose,
                front_clearance,
                {
                    "status": "persistent_obstacle_confirmed",
                    **common_evidence,
                },
            )
        if clearance.confirmed:
            return StationaryBlockageAdmission(
                "cleared",
                sample.map_pose,
                dict(last_front_details or {}),
                {
                    "status": "stationary_front_clearance_confirmed",
                    **common_evidence,
                },
            )

    now_sec = clock()
    return StationaryBlockageAdmission(
        "failed",
        samples[-1].map_pose if samples else None,
        dict(last_front_details or {}) if last_front_details else None,
        {
            "status": "stationary_blockage_unconfirmed",
            "zero_hold_duration_sec": now_sec - started_at,
            "confirmation_timeout_sec": timeout_sec,
            "initial_scan_receipt": normalized_initial_receipt,
            "last_scan_receipt": last_scan_receipt,
            "stationary_obstacle_confirmation": obstacle.to_log_dict(),
            "stationary_clearance_confirmation": clearance.to_log_dict(),
            "last_sample_failure": last_sample_failure,
        },
    )


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None
