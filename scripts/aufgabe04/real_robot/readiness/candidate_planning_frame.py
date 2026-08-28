"""Fail-closed admission of the camera candidate-planning frame.

The candidate route is planned in ``map`` while the frozen LiDAR candidates
must be reprojected through a current ``map <- odom`` transform.  This module
keeps that special evidence contract out of the autonomous runner: it declares
the extra no-motion preflight requirement and validates the returned paired
window plus final fresh transform before constructing the planning frame.
"""

from __future__ import annotations

import math
from typing import Mapping

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosPreflightRequirements,
    RosPreflightResult,
)


CANDIDATE_PLANNING_FRAME_PREFLIGHT_REQUIREMENTS = RosPreflightRequirements(
    require_stationary_map_from_odom_pairing=True,
)

# Candidate routes are certified with a 0.03 m tracking tube, and the stopped
# localization gate uses a conservative 0.03 rad yaw bound.  A final TF lookup
# taken while the robot is stopped must remain inside those bounds relative to
# the last transform paired with the admitted AMCL window.  Keep the readiness
# thresholds local: importing route-planning constants here would couple the
# no-motion evidence boundary back to motion-planning modules.
MAX_STOPPED_MAP_FROM_ODOM_TRANSLATION_DELTA_M = 0.03
MAX_STOPPED_MAP_FROM_ODOM_YAW_DELTA_RAD = 0.03


def _frame_id(value: object) -> str:
    return str(value).strip().strip("/")


def _require_successful_observation(
    preflight: RosPreflightResult,
    *,
    name: str,
):
    matches = [
        observation
        for observation in preflight.observations
        if observation.name == name
    ]
    if len(matches) != 1 or not matches[0].ok:
        raise RuntimeError(
            f"candidate planning frame lacks successful {name} evidence"
        )
    return matches[0]


def _validate_paired_transform_window(
    preflight: RosPreflightResult,
    *,
    map_frame: str,
    odom_frame: str,
) -> Mapping[str, object]:
    evidence = preflight.preflight_requirements
    if not isinstance(evidence, Mapping) or (
        evidence.get("stationary_map_from_odom_pairing_requested") is not True
        or evidence.get("stationary_map_from_odom_pairing_required") is not True
    ):
        raise RuntimeError(
            "candidate planning frame preflight did not explicitly require "
            "stationary map<-odom pairing"
        )

    observation = _require_successful_observation(
        preflight,
        name="stationary map<-odom transform samples",
    )
    required_count = observation.data.get("required_pair_count")
    if (
        not isinstance(required_count, int)
        or isinstance(required_count, bool)
        or required_count < 2
    ):
        raise RuntimeError(
            "candidate planning frame pairing evidence has an invalid "
            "required sample count"
        )
    samples = preflight.stationary_map_from_odom_samples
    if len(samples) != required_count:
        raise RuntimeError(
            "candidate planning frame pairing evidence is incomplete"
        )

    previous_stamp = -1
    previous_receipt = -1
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise RuntimeError(
                "candidate planning frame pairing sample is malformed"
            )
        if (
            sample.get("source") != "direct_dynamic_tf"
            or _frame_id(sample.get("target_frame")) != map_frame
            or _frame_id(sample.get("source_frame")) != odom_frame
            or sample.get("amcl_sample_index") != index
        ):
            raise RuntimeError(
                "candidate planning frame pairing sample provenance is invalid"
            )
        stamp = sample.get("stamp_nanoseconds")
        receipt = sample.get("receipt_time_nanoseconds")
        if (
            not isinstance(stamp, int)
            or isinstance(stamp, bool)
            or not isinstance(receipt, int)
            or isinstance(receipt, bool)
            or stamp <= previous_stamp
            or receipt <= previous_receipt
        ):
            raise RuntimeError(
                "candidate planning frame pairing sample order is invalid"
            )
        previous_stamp = stamp
        previous_receipt = receipt
    return samples[-1]


def _finite_planar_pose(
    value: Mapping[str, object],
    *,
    name: str,
) -> tuple[float, float, float]:
    if any(
        isinstance(value.get(key), bool)
        for key in ("x_m", "y_m", "yaw_rad")
    ):
        raise RuntimeError(
            f"candidate planning frame {name} map<-odom pose is invalid"
        )
    try:
        x_m = float(value["x_m"])
        y_m = float(value["y_m"])
        yaw_rad = float(value["yaw_rad"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"candidate planning frame {name} map<-odom pose is invalid: "
            f"{exc}"
        ) from exc
    if not all(math.isfinite(component) for component in (x_m, y_m, yaw_rad)):
        raise RuntimeError(
            f"candidate planning frame {name} map<-odom pose contains "
            "non-finite values"
        )
    return x_m, y_m, yaw_rad


def _transform_stamp_seconds(
    value: Mapping[str, object],
    *,
    name: str,
) -> float:
    stamp_nanoseconds = value.get("stamp_nanoseconds")
    if stamp_nanoseconds is not None:
        if (
            not isinstance(stamp_nanoseconds, int)
            or isinstance(stamp_nanoseconds, bool)
            or stamp_nanoseconds < 0
        ):
            raise RuntimeError(
                f"candidate planning frame {name} map<-odom timestamp is "
                "invalid"
            )
        return stamp_nanoseconds / 1_000_000_000.0

    stamp_sec = value.get("stamp_sec")
    if isinstance(stamp_sec, bool):
        raise RuntimeError(
            f"candidate planning frame {name} map<-odom timestamp is invalid"
        )
    try:
        parsed_stamp_sec = float(stamp_sec)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"candidate planning frame {name} map<-odom timestamp is missing "
            "or invalid"
        ) from exc
    if not math.isfinite(parsed_stamp_sec) or parsed_stamp_sec < 0.0:
        raise RuntimeError(
            f"candidate planning frame {name} map<-odom timestamp is invalid"
        )
    return parsed_stamp_sec


def _shortest_yaw_delta_rad(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def _validate_final_transform_consistency(
    map_from_odom: Mapping[str, object],
    *,
    last_paired_sample: Mapping[str, object],
) -> tuple[float, float, float]:
    """Bind the fresh final TF lookup to the accepted stopped window."""

    final_pose = _finite_planar_pose(map_from_odom, name="final")
    paired_pose = _finite_planar_pose(
        last_paired_sample,
        name="last paired",
    )
    final_stamp_sec = _transform_stamp_seconds(
        map_from_odom,
        name="final",
    )
    paired_stamp_sec = _transform_stamp_seconds(
        last_paired_sample,
        name="last paired",
    )
    if final_stamp_sec < paired_stamp_sec:
        raise RuntimeError(
            "candidate planning frame final map<-odom transform is older "
            "than the last paired stopped sample"
        )

    translation_delta_m = math.hypot(
        final_pose[0] - paired_pose[0],
        final_pose[1] - paired_pose[1],
    )
    if translation_delta_m > MAX_STOPPED_MAP_FROM_ODOM_TRANSLATION_DELTA_M:
        raise RuntimeError(
            "candidate planning frame stopped map<-odom translation delta "
            f"{translation_delta_m:.6f} m exceeds "
            f"{MAX_STOPPED_MAP_FROM_ODOM_TRANSLATION_DELTA_M:.3f} m"
        )

    yaw_delta_rad = _shortest_yaw_delta_rad(final_pose[2], paired_pose[2])
    if yaw_delta_rad > MAX_STOPPED_MAP_FROM_ODOM_YAW_DELTA_RAD:
        raise RuntimeError(
            "candidate planning frame stopped map<-odom yaw delta "
            f"{yaw_delta_rad:.6f} rad exceeds "
            f"{MAX_STOPPED_MAP_FROM_ODOM_YAW_DELTA_RAD:.3f} rad"
        )
    return final_pose


def build_candidate_planning_frame(
    preflight: RosPreflightResult,
    *,
    current_pose: Pose2D,
    map_frame: str,
    odom_frame: str,
) -> CandidatePlanningFrame:
    """Validate stopped transform evidence and bind the planning frame."""

    expected_map_frame = _frame_id(map_frame)
    expected_odom_frame = _frame_id(odom_frame)
    if not preflight.ok:
        raise RuntimeError("candidate planning frame preflight was not admitted")
    last_paired_sample = _validate_paired_transform_window(
        preflight,
        map_frame=expected_map_frame,
        odom_frame=expected_odom_frame,
    )
    direct_tf_observation = _require_successful_observation(
        preflight,
        name=f"tf {map_frame}->{odom_frame}",
    )

    map_from_odom = preflight.map_from_odom
    if map_from_odom is None:
        raise RuntimeError(
            "candidate planning frame preflight returned no map<-odom transform"
        )
    if (
        _frame_id(map_from_odom.get("target_frame")) != expected_map_frame
        or _frame_id(map_from_odom.get("source_frame")) != expected_odom_frame
    ):
        raise RuntimeError(
            "candidate planning frame map<-odom frame identity mismatch"
        )
    if (
        _frame_id(direct_tf_observation.data.get("target_frame"))
        != expected_map_frame
        or _frame_id(direct_tf_observation.data.get("source_frame"))
        != expected_odom_frame
    ):
        raise RuntimeError(
            "candidate planning frame direct TF evidence identity mismatch"
        )
    transform_pose = _validate_final_transform_consistency(
        map_from_odom,
        last_paired_sample=last_paired_sample,
    )
    transform = PlanarTransform2D(*transform_pose)

    return CandidatePlanningFrame(
        current_pose=current_pose,
        map_from_odom=transform,
        map_frame=expected_map_frame,
        odom_frame=expected_odom_frame,
    )


__all__ = [
    "CANDIDATE_PLANNING_FRAME_PREFLIGHT_REQUIREMENTS",
    "MAX_STOPPED_MAP_FROM_ODOM_TRANSLATION_DELTA_M",
    "MAX_STOPPED_MAP_FROM_ODOM_YAW_DELTA_RAD",
    "build_candidate_planning_frame",
]
