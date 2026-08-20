#!/usr/bin/env python3
"""Survey every simulated stand and record arrivals without visiting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds  # noqa: E402
from scripts.aufgabe04.navigation.dynamic_approach_planner import (  # noqa: E402
    DynamicApproachConfig,
)
from scripts.aufgabe04.navigation.route_revision_store import (  # noqa: E402
    RouteRevisionError,
    read_route_revision,
)
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (  # noqa: E402
    DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M,
    DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
    VIEWPOINT_SAMPLING_CONTRACT_NAME,
    VIEWPOINT_SAMPLING_CONTRACT_VERSION,
    ViewpointSamplingHoldConfig,
)
from scripts.aufgabe04.navigation.map_io import (  # noqa: E402
    load_occupancy_grid_with_bundle,
    write_frozen_map_bundle,
)
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.axis_acquisition_feedback import (  # noqa: E402
    AXIS_ACQUISITION_FEEDBACK_CONTRACT,
    AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION,
)
from scripts.aufgabe04.navigation.plan_detected_stand_exploration import (  # noqa: E402
    read_current_tf_pose,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (  # noqa: E402
    arrival_pose_catalog_sha256,
    freeze_arrival_pose_catalog,
    load_arrival_pose_catalog,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import (  # noqa: E402
    CatalogProvenance,
)
from scripts.aufgabe04.stations.candidate_snapshot import (  # noqa: E402
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (  # noqa: E402
    load_station_identity_registry,
    station_identity_registry_sha256,
)
from scripts.aufgabe04.artifacts import (  # noqa: E402
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    SurveyManifest,
    artifact_reference,
    write_survey_manifest,
)
from scripts.aufgabe04.artifacts.content_store import (  # noqa: E402
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.simulation.sim_viewpoint_optimization import (  # noqa: E402
    DEFAULT_TANGENTIAL_CORRECTION_GAIN,
    ViewpointConfig,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DEFAULT_KNOWN_STAND_KEEPOUT_RADIUS_M = 0.26
DEFAULT_STAND_RADIUS_M = 0.06
DEFAULT_STAND_UNCERTAINTY_M = 0.02
DEFAULT_TARGET_DISTANCE_M = DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M
DEFAULT_OBSERVER_MAX_CENTER_ERROR_DEG = 12.0
DEFAULT_OBSERVER_MAX_TANGENTIAL_STEP_DEG = 20.0
DEFAULT_PLANNER_TARGET_YAW_THRESHOLD_DEG = 4.0
SURVEY_PLANNER_TRACKING_MARGIN_M = 0.03
_ENVELOPE_EPSILON_M = (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
    + 1.0e-12
)
_FOLLOWER_SOURCE_RELATIVE_PATHS = (
    "scripts/aufgabe04/navigation/simple_waypoint_follower.py",
    "scripts/aufgabe04/navigation/waypoint_follower/__init__.py",
    "scripts/aufgabe04/navigation/waypoint_follower/config.py",
    "scripts/aufgabe04/navigation/waypoint_follower/pose_lookup.py",
    "scripts/aufgabe04/navigation/waypoint_follower/route_admission.py",
    "scripts/aufgabe04/navigation/waypoint_follower/route_phases.py",
    "scripts/aufgabe04/navigation/waypoint_follower/runtime.py",
    "scripts/aufgabe04/navigation/waypoint_follower/startup.py",
    "scripts/aufgabe04/navigation/waypoint_follower/terminal_heading.py",
)


@dataclass(frozen=True)
class SurveyCandidate:
    candidate_uid: str
    stand_id: str
    x_m: float
    y_m: float
    radius_m: float = DEFAULT_STAND_RADIUS_M
    uncertainty_m: float = DEFAULT_STAND_UNCERTAINTY_M
    keepout_radius_m: float | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _follower_source_sha256_by_path() -> dict[str, str]:
    """Hash every source file that implements the modular follower boundary."""

    return {
        relative_path: _file_sha256(ROOT / relative_path)
        for relative_path in _FOLLOWER_SOURCE_RELATIVE_PATHS
    }


def _arena_bounds_from_args(args) -> ArenaBounds:
    return ArenaBounds(
        length_m=args.arena_length_m,
        width_m=args.arena_width_m,
        center_x_m=args.arena_center_x_m,
        center_y_m=args.arena_center_y_m,
        yaw_deg=args.arena_yaw_deg,
        margin_m=args.arena_margin_m,
    )


def _survey_config_payload(args) -> dict[str, object]:
    return {
        "axis_sample_count": args.axis_sample_count,
        "known_stand_keepout_radius_m": args.known_stand_keepout_radius_m,
        "target_distance_m": args.target_distance_m,
        "dynamic_route_refresh_sec": args.dynamic_route_refresh_sec,
        "allow_simulation_odom_after_stale_tf": (
            args.allow_simulation_odom_after_stale_tf
        ),
        "initial_start_pose": {
            "x_m": args.initial_start_x,
            "y_m": args.initial_start_y,
            "yaw_rad": args.initial_start_yaw,
        },
        "refresh_start_from_tf": args.refresh_start_from_tf,
        "start_tf_timeout_sec": args.start_tf_timeout_sec,
        "start_tf_lookup_timeout_sec": args.start_tf_lookup_timeout_sec,
        "startup_timeout_sec": args.startup_timeout_sec,
        "candidate_timeout_sec": args.candidate_timeout_sec,
        "preflight_observation_window_sec": (
            args.preflight_observation_window_sec
        ),
        "initial_sensor_wait_sec": args.initial_sensor_wait_sec,
        "waypoint_timeout_sec": args.waypoint_timeout_sec,
        "lidar_clearance_margin_m": args.lidar_clearance_margin_m,
        "arena_bounds": _arena_bounds_from_args(args).to_metadata(),
        "viewpoint_sampling_timeout_sec": (
            args.viewpoint_sampling_timeout_sec
        ),
        "viewpoint_sampling_target_timeout_sec": (
            args.viewpoint_sampling_target_timeout_sec
        ),
        "viewpoint_sampling_goal_tolerance_m": (
            args.viewpoint_sampling_goal_tolerance_m
        ),
        "sampling_arrival_tolerance_m": args.sampling_arrival_tolerance_m,
        "viewpoint_sampling_contract_name": VIEWPOINT_SAMPLING_CONTRACT_NAME,
        "viewpoint_sampling_contract_version": (
            VIEWPOINT_SAMPLING_CONTRACT_VERSION
        ),
        "viewpoint_sampling_contract_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/navigation/"
            "viewpoint_sampling_contract.py"
        ),
        "tangential_correction_gain": args.tangential_correction_gain,
        "observer_max_center_error_deg": (
            args.observer_max_center_error_deg
        ),
        "observer_max_tangential_step_deg": (
            args.observer_max_tangential_step_deg
        ),
        "planner_target_yaw_threshold_deg": (
            args.planner_target_yaw_threshold_deg
        ),
        "viewpoint_sampling_terminal_heading_hold_tolerance_m": (
            args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        "viewpoint_sampling_terminal_heading_target_envelope_radius_m": (
            args
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m
        ),
        "viewpoint_sampling_terminal_heading_minimum_stand_distance_m": (
            args.target_distance_m
            - args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        "viewpoint_sampling_terminal_heading_maximum_stand_distance_m": (
            args.target_distance_m
            + args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        "intermediate_terminal_heading_distance_comparison_epsilon_m": (
            INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
        ),
        "viewpoint_sampling_heading_tolerance_rad": (
            args.viewpoint_sampling_heading_tolerance_rad
        ),
        "axis_acquisition_search_step_deg": (
            args.axis_acquisition_search_step_deg
        ),
        "axis_acquisition_search_max_targets": (
            args.axis_acquisition_search_max_targets
        ),
        "axis_acquisition_arrival_tolerance_m": (
            args.axis_acquisition_arrival_tolerance_m
        ),
        "axis_acquisition_feedback_contract": (
            AXIS_ACQUISITION_FEEDBACK_CONTRACT
        ),
        "axis_acquisition_feedback_schema_version": (
            AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION
        ),
        "axis_acquisition_feedback_max_age_sec": (
            args.axis_acquisition_feedback_max_age_sec
        ),
        # The concrete filename is a mutable per-candidate runtime detail. Only
        # its observer/planner-only semantics belong in the frozen config.
        "axis_acquisition_feedback_scope": (
            "per_candidate_observer_planner_sidecar"
        ),
        "axis_acquisition_feedback_is_motion_input": False,
        "lidar_stand_range_tolerance_m": args.lidar_stand_range_tolerance_m,
        "map_frame": args.map_frame,
        "odom_frame": args.odom_frame,
        "base_frame": args.base_frame,
        "scan_frame": args.scan_frame,
        "camera_frame": args.camera_frame,
        "image_topic": args.image_topic,
        "scan_topic": args.scan_topic,
        "odom_topic": args.odom_topic,
        "cmd_vel_topic": args.cmd_vel_topic,
        "allowed_cmd_vel_publishers": sorted(
            set(args.allowed_cmd_vel_publisher)
        ),
        "coordinator_source_sha256": _file_sha256(Path(__file__)),
        "observer_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/simulation/"
            "sim_synchronized_viewpoint_node.py"
        ),
        "viewpoint_optimizer_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/simulation/"
            "sim_viewpoint_optimization.py"
        ),
        "planner_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/navigation/"
            "plan_synchronized_viewpoint.py"
        ),
        "runner_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/navigation/"
            "run_single_station_segment.py"
        ),
        "follower_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/navigation/"
            "simple_waypoint_follower.py"
        ),
        "follower_module_source_sha256": (
            _follower_source_sha256_by_path()
        ),
        "waypoint_controller_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/navigation/"
            "waypoint_controller.py"
        ),
    }


def _validate_heading_contract(args) -> None:
    values = {
        "observer_max_center_error_deg": (
            args.observer_max_center_error_deg
        ),
        "observer_max_tangential_step_deg": (
            args.observer_max_tangential_step_deg
        ),
        "planner_target_yaw_threshold_deg": (
            args.planner_target_yaw_threshold_deg
        ),
        "viewpoint_sampling_heading_tolerance_rad": (
            args.viewpoint_sampling_heading_tolerance_rad
        ),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.observer_max_tangential_step_deg > 45.0:
        raise ValueError(
            "observer_max_tangential_step_deg must be no greater than 45"
        )

    follower_heading_tolerance_deg = math.degrees(
        args.viewpoint_sampling_heading_tolerance_rad
    )
    if not (
        args.observer_max_center_error_deg
        > follower_heading_tolerance_deg
    ):
        raise ValueError(
            "observer_max_center_error_deg must be strictly greater than "
            "the follower sampling heading tolerance in degrees "
            f"({args.observer_max_center_error_deg:.6f} > "
            f"{follower_heading_tolerance_deg:.6f})"
        )
    maximum_planner_threshold_deg = (
        args.observer_max_center_error_deg
        - follower_heading_tolerance_deg
    )
    if not (
        args.planner_target_yaw_threshold_deg
        < maximum_planner_threshold_deg
    ):
        raise ValueError(
            "planner_target_yaw_threshold_deg must be strictly less than "
            "observer_max_center_error_deg minus the follower sampling "
            "heading tolerance in degrees "
            f"({args.planner_target_yaw_threshold_deg:.6f} < "
            f"{args.observer_max_center_error_deg:.6f} - "
            f"{follower_heading_tolerance_deg:.6f})"
        )


def _validate_target_distance(
    args,
    candidates: tuple[SurveyCandidate, ...],
) -> None:
    target_distance_m = args.target_distance_m
    if not math.isfinite(target_distance_m) or target_distance_m <= 0.0:
        raise ValueError("target_distance_m must be finite and positive")
    goal_tolerance_m = args.viewpoint_sampling_goal_tolerance_m
    if not math.isfinite(goal_tolerance_m) or goal_tolerance_m <= 0.0:
        raise ValueError(
            "viewpoint_sampling_goal_tolerance_m must be finite and positive"
        )
    hold_tolerance_m = (
        args.viewpoint_sampling_terminal_heading_hold_tolerance_m
    )
    effective_entry_tolerance_m = min(
        goal_tolerance_m,
        INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    )
    sampling_arrival_tolerance_m = args.sampling_arrival_tolerance_m
    if (
        not math.isfinite(sampling_arrival_tolerance_m)
        or sampling_arrival_tolerance_m <= 0.0
    ):
        raise ValueError(
            "sampling_arrival_tolerance_m must be finite and positive"
        )
    if sampling_arrival_tolerance_m > effective_entry_tolerance_m:
        raise ValueError(
            "sampling_arrival_tolerance_m must be no greater than the "
            "effective follower entry tolerance "
            f"{effective_entry_tolerance_m:.6f} m"
        )
    if (
        not math.isfinite(hold_tolerance_m)
        or hold_tolerance_m <= 0.0
        or hold_tolerance_m
        > INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
        or hold_tolerance_m < effective_entry_tolerance_m
    ):
        raise ValueError(
            "viewpoint_sampling_terminal_heading_hold_tolerance_m must be "
            "finite, no smaller than the effective entry tolerance, and no "
            "greater than 0.020"
        )
    target_envelope_radius_m = (
        args.viewpoint_sampling_terminal_heading_target_envelope_radius_m
    )
    if (
        not math.isfinite(target_envelope_radius_m)
        or target_envelope_radius_m < hold_tolerance_m
        or target_envelope_radius_m
        > INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    ):
        raise ValueError(
            "viewpoint_sampling_terminal_heading_target_envelope_radius_m "
            "must be no smaller than the radial hold tolerance and no "
            "greater than 0.030"
        )
    ViewpointSamplingHoldConfig(
        entry_tolerance_m=effective_entry_tolerance_m,
        hold_tolerance_m=hold_tolerance_m,
        target_distance_m=target_distance_m,
        target_envelope_radius_m=target_envelope_radius_m,
    )
    # The latch may complete anywhere in its bounded hold disk, so the
    # cross-stage planner/observer contract must validate that full disk—not
    # only the tighter capture threshold.
    effective_hold_limit_m = (
        hold_tolerance_m
        + INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
    )
    envelope_min_m = target_distance_m - effective_hold_limit_m
    envelope_max_m = target_distance_m + effective_hold_limit_m
    if (
        envelope_min_m + _ENVELOPE_EPSILON_M
        < ViewpointConfig.min_distance_m
    ):
        raise ValueError(
            f"target envelope lower bound {envelope_min_m:.6f} m is below "
            "the simulation observer minimum distance "
            f"{ViewpointConfig.min_distance_m:.6f} m"
        )
    if (
        envelope_max_m
        > ViewpointConfig.max_distance_m + _ENVELOPE_EPSILON_M
    ):
        raise ValueError(
            f"target envelope upper bound {envelope_max_m:.6f} m exceeds "
            "the simulation observer maximum distance "
            f"{ViewpointConfig.max_distance_m:.6f} m"
        )

    for candidate in candidates:
        # These are the same values used by the survey planner command:
        # candidate geometry arrives through the observer recommendation,
        # while plan_synchronized_viewpoint uses a 0.03 m tracking margin.
        config = DynamicApproachConfig(
            stand_radius_m=candidate.radius_m,
            stand_position_uncertainty_m=candidate.uncertainty_m,
            tracking_margin_m=SURVEY_PLANNER_TRACKING_MARGIN_M,
            standoff_distance_m=target_distance_m,
            lidar_stop_distance_m=args.min_obstacle_distance_m,
            lidar_clearance_margin_m=args.lidar_clearance_margin_m,
        )
        if (
            envelope_min_m + _ENVELOPE_EPSILON_M
            < config.minimum_lidar_standoff_m
        ):
            raise ValueError(
                f"target envelope lower bound {envelope_min_m:.6f} m is "
                f"incompatible with candidate {candidate.candidate_uid}: "
                "DynamicApproachConfig.minimum_lidar_standoff_m="
                f"{config.minimum_lidar_standoff_m:.6f} m"
            )


def _calibration_profile_payload(args) -> dict[str, object]:
    return {
        "profile_kind": "simulation_camera_lidar_defaults",
        "image_topic": args.image_topic,
        "scan_topic": args.scan_topic,
        "camera_frame": args.camera_frame,
        "scan_frame": args.scan_frame,
        "observer_source_sha256": _file_sha256(
            ROOT
            / "scripts/aufgabe04/simulation/"
            "sim_synchronized_viewpoint_node.py"
        ),
    }


def _catalog_provenance(
    args,
    *,
    map_sha256: str,
    world_sha256: str,
    map_bundle_sha256: str = "",
    candidate_snapshot_sha256: str = "",
    station_identity_registry_sha256: str = "",
    survey_config_sha256: str = "",
    calibration_profile_sha256: str = "",
    survey_input_binding_sha256: str = "",
) -> CatalogProvenance:
    """Bind every coordinator catalog read to this exact simulation run."""

    return CatalogProvenance(
        planning_frame=args.map_frame,
        map_yaml_sha256=map_sha256,
        world_id=args.world.stem,
        world_sha256=world_sha256,
        session_id=args.session_id,
        environment="simulation",
        map_bundle_sha256=map_bundle_sha256,
        candidate_snapshot_sha256=candidate_snapshot_sha256,
        station_identity_registry_sha256=station_identity_registry_sha256,
        survey_config_sha256=survey_config_sha256,
        calibration_profile_sha256=calibration_profile_sha256,
        survey_input_binding_sha256=survey_input_binding_sha256,
    )


def _survey_stream_id(session_id: str, candidate_uid: str) -> str:
    """Return a bounded stream identity unique to one survey session/candidate."""

    identity = f"{session_id}\0{candidate_uid}".encode("utf-8")
    return f"survey-{hashlib.sha256(identity).hexdigest()[:32]}"


def _load_candidates(path: Path) -> tuple[SurveyCandidate, ...]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("candidates JSON must contain a candidates list")
    candidates = []
    seen = set()
    for index, item in enumerate(payload["candidates"]):
        if not isinstance(item, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        candidate = SurveyCandidate(
            candidate_uid=str(item["candidate_uid"]).strip(),
            stand_id=str(item.get("stand_id", item["candidate_uid"])).strip(),
            x_m=float(item["x_m"]),
            y_m=float(item["y_m"]),
            radius_m=float(item.get("radius_m", DEFAULT_STAND_RADIUS_M)),
            uncertainty_m=float(
                item.get("uncertainty_m", DEFAULT_STAND_UNCERTAINTY_M)
            ),
            keepout_radius_m=(
                None
                if item.get("keepout_radius_m") is None
                else float(item["keepout_radius_m"])
            ),
        )
        if not _SAFE_ID.fullmatch(candidate.candidate_uid):
            raise ValueError(
                "candidate_uid values must be safe identifiers containing only "
                "letters, digits, '.', '_', or '-'"
            )
        if not _SAFE_ID.fullmatch(candidate.stand_id):
            raise ValueError(f"candidates[{index}].stand_id is not a safe identifier")
        if candidate.candidate_uid in seen:
            raise ValueError("candidate_uid values must be unique")
        if not math.isfinite(candidate.x_m) or not math.isfinite(candidate.y_m):
            raise ValueError(f"candidates[{index}] coordinates must be finite")
        if not math.isfinite(candidate.radius_m) or candidate.radius_m <= 0.0:
            raise ValueError(f"candidates[{index}].radius_m must be positive")
        if (
            not math.isfinite(candidate.uncertainty_m)
            or candidate.uncertainty_m < 0.0
        ):
            raise ValueError(
                f"candidates[{index}].uncertainty_m must be non-negative"
            )
        if candidate.keepout_radius_m is not None and (
            not math.isfinite(candidate.keepout_radius_m)
            or candidate.keepout_radius_m <= 0.0
        ):
            raise ValueError(
                f"candidates[{index}].keepout_radius_m must be finite and positive"
            )
        seen.add(candidate.candidate_uid)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("at least one candidate is required")
    return tuple(candidates)


def _load_snapshot_candidates(snapshot, registry) -> tuple[SurveyCandidate, ...]:
    candidates = []
    for candidate in snapshot.candidates:
        identity = registry.for_candidate(candidate.candidate_uid)
        if identity is None:
            raise ValueError(
                f"candidate {candidate.candidate_uid} has no station identity"
            )
        candidates.append(
            SurveyCandidate(
                candidate_uid=candidate.candidate_uid,
                # The simulation observer compares decoded QR data with this
                # canonical registry value. Candidate UIDs are never treated
                # as QR/server station IDs implicitly.
                stand_id=identity.qr_id,
                x_m=candidate.geometry.x_m,
                y_m=candidate.geometry.y_m,
                radius_m=candidate.geometry.radius_m,
                uncertainty_m=candidate.geometry.uncertainty_m,
                keepout_radius_m=candidate.geometry.keepout_radius_m,
            )
        )
    return tuple(candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    candidate_source = parser.add_mutually_exclusive_group(required=True)
    candidate_source.add_argument("--candidates-json", type=Path)
    candidate_source.add_argument("--candidate-snapshot", type=Path)
    parser.add_argument("--station-identity-registry", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-candidate-json",
        action="store_true",
        help=(
            "Explicit compatibility escape hatch for unsealed candidate JSON; "
            "not suitable as migration evidence."
        ),
    )
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--catalog-id", default="sim_arrival_survey")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument(
        "--arena-center-x-m",
        type=float,
        default=ArenaBounds.center_x_m,
    )
    parser.add_argument(
        "--arena-center-y-m",
        type=float,
        default=ArenaBounds.center_y_m,
    )
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    parser.add_argument("--map-bundle-json", type=Path, default=None)
    parser.add_argument("--survey-manifest", type=Path, default=None)
    parser.add_argument(
        "--survey-input-binding",
        type=Path,
        default=None,
        help=(
            "Stable immutable session-input binding; defaults beside --catalog "
            "and prevents cross-snapshot resume."
        ),
    )
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--scan-frame", default="base_scan")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--initial-start-x", type=float, default=0.0)
    parser.add_argument("--initial-start-y", type=float, default=0.0)
    parser.add_argument("--initial-start-yaw", type=float, default=0.0)
    parser.add_argument(
        "--refresh-start-from-tf",
        action="store_true",
        help=(
            "Resolve map-frame -> base-frame TF immediately before planning "
            "each candidate instead of reusing the nominal initial start."
        ),
    )
    parser.add_argument("--start-tf-timeout-sec", type=float, default=3.0)
    parser.add_argument(
        "--start-tf-lookup-timeout-sec",
        type=float,
        default=0.2,
    )
    parser.add_argument("--startup-timeout-sec", type=float, default=20.0)
    parser.add_argument("--candidate-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--preflight-observation-window-sec",
        type=float,
        default=6.0,
        help="DDS/TF discovery window for each newly started simulation runner.",
    )
    parser.add_argument(
        "--initial-sensor-wait-sec",
        type=float,
        default=6.0,
        help="Follower startup window for scan, odometry, and TF discovery.",
    )
    parser.add_argument(
        "--waypoint-timeout-sec",
        type=float,
        default=60.0,
        help=(
            "Survey-only timeout for one continuously progressing route "
            "target. The standalone follower keeps its stricter default."
        ),
    )
    parser.add_argument("--dynamic-route-refresh-sec", type=float, default=0.10)
    parser.add_argument(
        "--allow-simulation-odom-after-stale-tf",
        action="store_true",
        default=False,
        help=(
            "Explicitly allow the simulation survey runner to use exact-frame "
            "odometry only after its existing zero plus bounded stale-TF "
            "retry also fails; this is not real-robot migration evidence."
        ),
    )
    parser.add_argument(
        "--min-obstacle-distance-m",
        type=float,
        default=0.20,
        help=(
            "Single live LaserScan stop threshold shared by the survey planner "
            "and route executor."
        ),
    )
    parser.add_argument(
        "--lidar-clearance-margin-m",
        type=float,
        default=0.02,
        help=(
            "Additional static-map clearance beyond the live LaserScan stop "
            "distance. This keeps discretized survey routes from terminating "
            "on the executor's exact stop threshold."
        ),
    )
    parser.add_argument(
        "--target-distance-m",
        type=float,
        default=DEFAULT_TARGET_DISTANCE_M,
        help=(
            "Final observer target distance; must satisfy the survey planner's "
            "LiDAR standoff and the observer's admissible distance band."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-timeout-sec",
        type=float,
        default=180.0,
        help="Total bounded time for the complete viewpoint-sampling phase.",
    )
    parser.add_argument(
        "--viewpoint-sampling-target-timeout-sec",
        type=float,
        default=60.0,
        help="Maximum convergence time for each material sampling target.",
    )
    parser.add_argument(
        "--viewpoint-sampling-goal-tolerance-m",
        type=float,
        default=DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M,
        help=(
            "Simulation survey terminal-position entry tolerance. It is "
            "strictly inside the 0.02 m terminal-heading hold envelope so the "
            "complete acceptance envelope remains inside the observer and "
            "planner bounds."
        ),
    )
    parser.add_argument(
        "--sampling-arrival-tolerance-m",
        type=float,
        default=DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M,
        help=(
            "Observer distance gate for advancing a latched sampling target. "
            "It must be no greater than the follower's effective terminal "
            "entry tolerance."
        ),
    )
    parser.add_argument(
        "--tangential-correction-gain",
        type=float,
        default=DEFAULT_TANGENTIAL_CORRECTION_GAIN,
        help=(
            "Simulation observer gain applied to each camera-derived "
            "tangential correction before its maximum-step clamp."
        ),
    )
    parser.add_argument(
        "--observer-max-center-error-deg",
        "--max-center-error-deg",
        dest="observer_max_center_error_deg",
        type=float,
        default=DEFAULT_OBSERVER_MAX_CENTER_ERROR_DEG,
        help=(
            "Observer image-center acceptance gate. It is frozen into the "
            "survey configuration and forwarded explicitly."
        ),
    )
    parser.add_argument(
        "--observer-max-tangential-step-deg",
        "--max-tangential-step-deg",
        dest="observer_max_tangential_step_deg",
        type=float,
        default=DEFAULT_OBSERVER_MAX_TANGENTIAL_STEP_DEG,
        help=(
            "Maximum observer-side center-error correction for one revised "
            "sampling target."
        ),
    )
    parser.add_argument(
        "--planner-target-yaw-threshold-deg",
        "--target-yaw-threshold-deg",
        dest="planner_target_yaw_threshold_deg",
        type=float,
        default=DEFAULT_PLANNER_TARGET_YAW_THRESHOLD_DEG,
        help=(
            "Material sampling-target yaw threshold forwarded to the "
            "synchronized planner."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
        type=float,
        default=INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
        help=(
            "Maximum simulation survey radius retained after terminal-yaw "
            "capture; exceeding it stops rather than resuming translation."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
        type=float,
        default=INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
        help=(
            "Maximum simulation pose-to-target drift retained during "
            "zero-linear terminal yaw; radial stand distance remains bounded "
            "by the separate hold tolerance."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-heading-tolerance-rad",
        type=float,
        default=math.radians(5.0),
    )
    parser.add_argument(
        "--axis-acquisition-search-step-deg",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--axis-acquisition-search-max-targets",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--axis-acquisition-arrival-tolerance-m",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--axis-acquisition-feedback-max-age-sec",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--lidar-stand-range-tolerance-m",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--axis-sample-count",
        type=int,
        default=7,
        help=(
            "Minimum distinct dynamic silhouette samples required before an "
            "axis may be committed and recorded in the catalog."
        ),
    )
    parser.add_argument(
        "--known-stand-keepout-radius-m",
        type=float,
        default=DEFAULT_KNOWN_STAND_KEEPOUT_RADIUS_M,
        help=(
            "Default total robot-center exclusion radius passed for every known "
            "stand; a candidate keepout_radius_m field overrides it."
        ),
    )
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=["/behavior_server", "/velocity_smoother"],
    )
    return parser


def _observer_command(args, candidate: SurveyCandidate, output: Path, stream_id: str):
    return [
        sys.executable,
        "scripts/aufgabe04/simulation/sim_synchronized_viewpoint_node.py",
        "--image-topic", args.image_topic,
        "--scan-topic", args.scan_topic,
        "--odom-topic", args.odom_topic,
        "--stand-x", str(candidate.x_m),
        "--stand-y", str(candidate.y_m),
        "--stand-id", candidate.stand_id,
        "--stand-radius-m", str(candidate.radius_m),
        "--stand-uncertainty-m", str(candidate.uncertainty_m),
        "--target-distance-m", str(args.target_distance_m),
        "--sampling-arrival-tolerance-m",
        str(args.sampling_arrival_tolerance_m),
        "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
        str(args.viewpoint_sampling_terminal_heading_hold_tolerance_m),
        "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
        str(
            args
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m
        ),
        "--tangential-correction-gain",
        str(args.tangential_correction_gain),
        "--max-center-error-deg",
        str(args.observer_max_center_error_deg),
        "--max-tangential-step-deg",
        str(args.observer_max_tangential_step_deg),
        "--lidar-stand-range-tolerance-m",
        str(args.lidar_stand_range_tolerance_m),
        "--stream-id", stream_id,
        "--map-frame", args.map_frame,
        "--base-frame", args.base_frame,
        "--scan-frame", args.scan_frame,
        "--camera-frame", args.camera_frame,
        "--status-json", str(output / "observer_status.json"),
        "--recommended-pose-json", str(output / "recommendation.json"),
        "--observation-json", str(output / "camera_observation.json"),
        "--debug-dir", str(output / "perception_debug"),
        "--dynamic-min-axis-samples", str(args.axis_sample_count),
        "--axis-acquisition-search-step-deg",
        str(args.axis_acquisition_search_step_deg),
        "--axis-acquisition-search-max-targets",
        str(args.axis_acquisition_search_max_targets),
        "--axis-acquisition-arrival-tolerance-m",
        str(args.axis_acquisition_arrival_tolerance_m),
        "--axis-acquisition-feedback-max-age-sec",
        str(args.axis_acquisition_feedback_max_age_sec),
        "--axis-acquisition-feedback-json",
        str(output / "axis_acquisition_feedback.json"),
    ]


def _planner_command(
    args,
    candidate: SurveyCandidate,
    candidates: tuple[SurveyCandidate, ...],
    output: Path,
    stream_id: str,
    world_sha256: str,
    start_pose: Pose2D | None = None,
):
    start_pose = start_pose or Pose2D(
        args.initial_start_x,
        args.initial_start_y,
        args.initial_start_yaw,
    )
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/plan_synchronized_viewpoint.py",
        "--map", str(args.map),
        # Keep signed scientific-notation values attached to their options.
        # argparse otherwise interprets values such as ``-5.0e-06`` as a new
        # option and the synchronized planner exits before publishing a route.
        f"--start-x={start_pose.x_m}",
        f"--start-y={start_pose.y_m}",
        f"--start-yaw={start_pose.yaw_rad}",
        "--recommended-pose-json", str(output / "recommendation.json"),
        "--route-csv", str(output / "survey_route.csv"),
        "--diagnostics-json", str(output / "survey_route_diagnostics.json"),
        "--route-manifest", str(output / "survey_route.manifest.json"),
        "--stream-id", stream_id,
        "--writer-id", f"planner-{stream_id}",
        "--workflow-mode", "survey-only",
        "--arrival-pose-catalog", str(args.catalog),
        "--catalog-id", args.catalog_id,
        "--candidate-uid", candidate.candidate_uid,
        "--world-id", args.world.stem,
        "--world-sha256", world_sha256,
        "--session-id", args.session_id,
        "--map-frame", args.map_frame,
        "--semantic-map-id", args.semantic_map_id or args.map.stem,
        "--arena-length-m", str(args.arena_length_m),
        "--arena-width-m", str(args.arena_width_m),
        "--arena-center-x-m", str(args.arena_center_x_m),
        "--arena-center-y-m", str(args.arena_center_y_m),
        "--arena-yaw-deg", str(args.arena_yaw_deg),
        "--arena-margin-m", str(args.arena_margin_m),
        "--expected-map-bundle-sha256", getattr(args, "map_bundle_sha256", ""),
        "--candidate-snapshot-sha256", getattr(
            args, "candidate_snapshot_sha256", ""
        ),
        "--station-identity-registry-sha256", getattr(
            args, "station_identity_registry_sha256", ""
        ),
        "--survey-config-sha256", getattr(args, "survey_config_sha256", ""),
        "--calibration-profile-sha256", getattr(
            args, "calibration_profile_sha256", ""
        ),
        "--survey-input-binding-sha256", getattr(
            args, "survey_input_binding_sha256", ""
        ),
        "--axis-sample-count", str(args.axis_sample_count),
        "--lidar-stop-distance-m", str(args.min_obstacle_distance_m),
        "--lidar-clearance-margin-m", str(args.lidar_clearance_margin_m),
        "--target-yaw-threshold-deg",
        str(args.planner_target_yaw_threshold_deg),
        "--axis-acquisition-arrival-tolerance-m",
        str(args.axis_acquisition_arrival_tolerance_m),
        "--axis-acquisition-search-max-targets",
        str(args.axis_acquisition_search_max_targets),
        "--axis-acquisition-feedback-max-age-sec",
        str(args.axis_acquisition_feedback_max_age_sec),
        "--axis-acquisition-feedback-json",
        str(output / "axis_acquisition_feedback.json"),
        "--watch",
    ]
    for item in candidates:
        command.extend(["--expected-candidate-uid", item.candidate_uid])
        if item.candidate_uid == candidate.candidate_uid:
            # The current stand has target-specific body keepout and LiDAR
            # standoff validation inside plan_axis_acquisition / fixed arrival
            # validation.  Applying the larger non-target transit disk here
            # would rasterize valid 0.30 m viewpoint samples as obstacles.
            continue
        command.extend(
            [
                "--known-stand-keepout",
                str(item.x_m),
                str(item.y_m),
                str(
                    max(
                        args.known_stand_keepout_radius_m,
                        0.0
                        if item.keepout_radius_m is None
                        else item.keepout_radius_m,
                    )
                ),
            ]
        )
    return command


def _runner_command(args, candidate: SurveyCandidate, output: Path):
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/run_single_station_segment.py",
        "--leg-index", "0",
        "--route-csv", str(output / "survey_route.csv"),
        "--diagnostics-json", str(output / "survey_route_diagnostics.json"),
        "--route-manifest", str(output / "survey_route.manifest.json"),
        "--run-id", f"survey_{args.session_id}_{candidate.candidate_uid}",
        "--scan-topic", args.scan_topic,
        "--odom-topic", args.odom_topic,
        "--cmd-vel-topic", args.cmd_vel_topic,
        "--localization-source", "tf",
        "--map-frame", args.map_frame,
        "--odom-frame", args.odom_frame,
        "--base-frame", args.base_frame,
        "--allow-sim-time",
        "--preflight-observation-window-sec",
        str(args.preflight_observation_window_sec),
        "--initial-sensor-wait-sec",
        str(args.initial_sensor_wait_sec),
        "--waypoint-timeout-sec",
        str(args.waypoint_timeout_sec),
        "--viewpoint-sampling-timeout-sec",
        str(args.viewpoint_sampling_timeout_sec),
        "--viewpoint-sampling-target-timeout-sec",
        str(args.viewpoint_sampling_target_timeout_sec),
        "--viewpoint-sampling-goal-tolerance-m",
        str(args.viewpoint_sampling_goal_tolerance_m),
        "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
        str(args.viewpoint_sampling_terminal_heading_hold_tolerance_m),
        "--viewpoint-sampling-target-distance-m",
        str(args.target_distance_m),
        "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
        str(
            args
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m
        ),
        "--viewpoint-sampling-heading-tolerance-rad",
        str(args.viewpoint_sampling_heading_tolerance_rad),
        "--dynamic-route-refresh-sec", str(args.dynamic_route_refresh_sec),
        "--min-obstacle-distance-m", str(args.min_obstacle_distance_m),
        "--operator-note", f"survey-only candidate {candidate.candidate_uid}",
    ]
    if args.allow_simulation_odom_after_stale_tf:
        command.append("--allow-simulation-odom-after-stale-tf")
    for publisher in args.allowed_cmd_vel_publisher:
        command.extend(["--allowed-cmd-vel-publisher", publisher])
    return command


def _candidate_start_pose(args) -> Pose2D:
    """Resolve the live route start before every independently surveyed stand."""

    if not args.refresh_start_from_tf:
        return Pose2D(
            args.initial_start_x,
            args.initial_start_y,
            args.initial_start_yaw,
        )
    return read_current_tf_pose(
        target_frame=args.map_frame,
        source_frame=args.base_frame,
        timeout_sec=args.start_tf_timeout_sec,
        lookup_timeout_sec=args.start_tf_lookup_timeout_sec,
        use_sim_time=True,
    )


def _wait_for_route(
    manifest: Path,
    stream_id: str,
    timeout_sec: float,
    *,
    not_before_unix_sec: float | None = None,
) -> str:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            loaded = read_route_revision(
                manifest,
                expected_stream_id=stream_id,
                verify_artifacts=True,
            )
        except (OSError, RouteRevisionError):
            time.sleep(0.10)
            continue
        if (
            not_before_unix_sec is not None
            and float(loaded.manifest["published_unix_sec"]) < not_before_unix_sec
        ):
            # The output directory is intentionally resumable.  Never hand an
            # old active manifest to a newly launched follower, even when the
            # same session/candidate stream is retried.
            time.sleep(0.10)
            continue
        if loaded.status == "active":
            return "active"
        if loaded.status == "survey_complete":
            return "survey_complete"
        time.sleep(0.10)
    raise TimeoutError("timed out waiting for an active survey route")


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _survey_completion_available(
    *,
    catalog_path: Path,
    provenance: CatalogProvenance,
    manifest_path: Path,
    stream_id: str,
    candidate_uid: str,
) -> bool:
    """Return true only when both durable hand-offs prove survey success."""

    try:
        catalog = load_arrival_pose_catalog(
            catalog_path,
            required_provenance=provenance,
        )
        terminal = read_route_revision(
            manifest_path,
            expected_stream_id=stream_id,
            verify_artifacts=False,
        )
    except (OSError, ValueError, RouteRevisionError):
        return False
    if catalog.record_for(candidate_uid) is None or terminal.status != "survey_complete":
        return False
    completion = terminal.manifest.get("completion")
    if not isinstance(completion, dict):
        return False
    return (
        completion.get("candidate_uid") == candidate_uid
        and completion.get("catalog_sha256")
        == arrival_pose_catalog_sha256(catalog)
    )


def _survey_one(
    args,
    candidate: SurveyCandidate,
    candidates: tuple[SurveyCandidate, ...],
    world_sha256: str,
    provenance: CatalogProvenance,
) -> None:
    output = args.output_dir / candidate.candidate_uid
    output.mkdir(parents=True, exist_ok=True)
    stream_id = _survey_stream_id(args.session_id, candidate.candidate_uid)
    observer_log = (output / "observer.log").open("w")
    planner_log = (output / "planner.log").open("w")
    observer = planner = runner = None
    try:
        start_pose = _candidate_start_pose(args)
        launched_unix_sec = time.time()
        observer = subprocess.Popen(
            _observer_command(args, candidate, output, stream_id),
            stdout=observer_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        planner = subprocess.Popen(
            _planner_command(
                args,
                candidate,
                candidates,
                output,
                stream_id,
                world_sha256,
                start_pose,
            ),
            stdout=planner_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        state = _wait_for_route(
            output / "survey_route.manifest.json",
            stream_id,
            args.startup_timeout_sec,
            not_before_unix_sec=launched_unix_sec,
        )
        if state == "active":
            runner = subprocess.Popen(
                _runner_command(args, candidate, output),
            )
            returncode = runner.wait(timeout=args.candidate_timeout_sec)
            if returncode != 0 and not _survey_completion_available(
                catalog_path=args.catalog,
                provenance=provenance,
                manifest_path=output / "survey_route.manifest.json",
                stream_id=stream_id,
                candidate_uid=candidate.candidate_uid,
            ):
                raise RuntimeError(
                    f"survey follower failed for {candidate.candidate_uid}: "
                    f"exit {returncode}"
                )
        catalog = load_arrival_pose_catalog(
            args.catalog,
            required_provenance=provenance,
        )
        if catalog.record_for(candidate.candidate_uid) is None:
            raise RuntimeError(
                f"survey ended without catalog record for {candidate.candidate_uid}"
            )
    finally:
        # Stop the motion process first while its ROS context is still alive,
        # allowing the follower's finally block to publish repeated zero Twist.
        _terminate(runner)
        _terminate(planner)
        _terminate(observer)
        observer_log.close()
        planner_log.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_heading_contract(args)
        arena_bounds = _arena_bounds_from_args(args)
        arena_bounds.validate()
        _grid, map_bundle = load_occupancy_grid_with_bundle(
            args.map,
            semantic_map_id=args.semantic_map_id or args.map.stem,
            planning_frame=args.map_frame,
        )
        args.map_bundle_sha256 = map_bundle.bundle_sha256
        args.map_bundle_json = args.map_bundle_json or args.output_dir / (
            f"map_bundle_{map_bundle.bundle_sha256[:16]}.json"
        )
        if args.candidate_snapshot is not None:
            if args.station_identity_registry is None:
                raise ValueError(
                    "--candidate-snapshot requires --station-identity-registry"
                )
            snapshot = load_candidate_snapshot(
                args.candidate_snapshot,
                required_map_bundle_sha256=map_bundle.bundle_sha256,
            )
            registry = load_station_identity_registry(
                args.station_identity_registry,
                candidate_snapshot=snapshot,
            )
            candidates = _load_snapshot_candidates(snapshot, registry)
            args.candidate_snapshot_sha256 = candidate_snapshot_sha256(snapshot)
            identity_registry_sha256 = station_identity_registry_sha256(registry)
        else:
            if not args.allow_legacy_candidate_json:
                raise ValueError(
                    "legacy --candidates-json is unsealed; use --candidate-snapshot "
                    "and --station-identity-registry, or explicitly pass "
                    "--allow-legacy-candidate-json"
                )
            candidates = _load_candidates(args.candidates_json)
            args.candidate_snapshot_sha256 = _file_sha256(args.candidates_json)
            identity_registry_sha256 = ""
        for name, value in (
            ("catalog_id", args.catalog_id),
            ("session_id", args.session_id),
            ("world_id", args.world.stem),
        ):
            if not _SAFE_ID.fullmatch(value):
                raise ValueError(f"{name} is not a safe identifier: {value!r}")
        positive_values = {
            "startup_timeout_sec": args.startup_timeout_sec,
            "candidate_timeout_sec": args.candidate_timeout_sec,
            "preflight_observation_window_sec": (
                args.preflight_observation_window_sec
            ),
            "initial_sensor_wait_sec": args.initial_sensor_wait_sec,
            "waypoint_timeout_sec": args.waypoint_timeout_sec,
            "min_obstacle_distance_m": args.min_obstacle_distance_m,
            "start_tf_timeout_sec": args.start_tf_timeout_sec,
            "start_tf_lookup_timeout_sec": args.start_tf_lookup_timeout_sec,
            "viewpoint_sampling_timeout_sec": (
                args.viewpoint_sampling_timeout_sec
            ),
            "viewpoint_sampling_target_timeout_sec": (
                args.viewpoint_sampling_target_timeout_sec
            ),
            "viewpoint_sampling_goal_tolerance_m": (
                args.viewpoint_sampling_goal_tolerance_m
            ),
            "sampling_arrival_tolerance_m": (
                args.sampling_arrival_tolerance_m
            ),
            "tangential_correction_gain": args.tangential_correction_gain,
            "viewpoint_sampling_terminal_heading_hold_tolerance_m": (
                args.viewpoint_sampling_terminal_heading_hold_tolerance_m
            ),
            "viewpoint_sampling_terminal_heading_target_envelope_radius_m": (
                args
                .viewpoint_sampling_terminal_heading_target_envelope_radius_m
            ),
            "viewpoint_sampling_heading_tolerance_rad": (
                args.viewpoint_sampling_heading_tolerance_rad
            ),
            "axis_acquisition_search_step_deg": (
                args.axis_acquisition_search_step_deg
            ),
            "axis_acquisition_arrival_tolerance_m": (
                args.axis_acquisition_arrival_tolerance_m
            ),
            "axis_acquisition_feedback_max_age_sec": (
                args.axis_acquisition_feedback_max_age_sec
            ),
            "lidar_stand_range_tolerance_m": (
                args.lidar_stand_range_tolerance_m
            ),
            "known_stand_keepout_radius_m": args.known_stand_keepout_radius_m,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if args.tangential_correction_gain > 1.0:
            raise ValueError(
                "tangential_correction_gain must be no greater than 1.0"
            )
        if (
            not math.isfinite(args.lidar_clearance_margin_m)
            or args.lidar_clearance_margin_m < 0.0
        ):
            raise ValueError(
                "lidar_clearance_margin_m must be finite and non-negative"
            )
        _validate_target_distance(args, candidates)
        if args.axis_sample_count < 7:
            raise ValueError("axis_sample_count must be at least 7")
        if args.axis_acquisition_search_max_targets < 1:
            raise ValueError(
                "axis_acquisition_search_max_targets must be positive"
            )
        if args.axis_acquisition_search_step_deg > 90.0:
            raise ValueError(
                "axis_acquisition_search_step_deg must not exceed 90"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_frozen_map_bundle(args.map_bundle_json, map_bundle)
        map_sha256 = map_bundle.yaml_sha256
        world_sha256 = _file_sha256(args.world)
        survey_config_payload = _survey_config_payload(args)
        survey_config_sha256 = payload_sha256(survey_config_payload)
        survey_config_path = args.output_dir / (
            f"survey_config_{survey_config_sha256}.json"
        )
        write_content_hashed_json(
            survey_config_path,
            survey_config_payload,
            hash_field="survey_config_sha256",
        )
        calibration_payload = _calibration_profile_payload(args)
        calibration_sha256 = payload_sha256(calibration_payload)
        calibration_path = args.output_dir / (
            f"calibration_profile_{calibration_sha256}.json"
        )
        write_content_hashed_json(
            calibration_path,
            calibration_payload,
            hash_field="calibration_profile_sha256",
        )
        survey_input_binding_path = (
            args.survey_input_binding
            or args.catalog.with_suffix(args.catalog.suffix + ".survey_inputs.json")
        )
        survey_input_binding_payload = {
            "catalog_id": args.catalog_id,
            "session_id": args.session_id,
            "planning_frame": args.map_frame,
            "map_bundle_sha256": map_bundle.bundle_sha256,
            "candidate_snapshot_sha256": args.candidate_snapshot_sha256,
            "station_identity_registry_sha256": identity_registry_sha256,
            "world_id": args.world.stem,
            "world_sha256": world_sha256,
            "survey_config_sha256": survey_config_sha256,
            "calibration_profile_sha256": calibration_sha256,
        }
        survey_input_binding_sha256 = write_content_hashed_json(
            survey_input_binding_path,
            survey_input_binding_payload,
            hash_field="survey_input_binding_sha256",
        )
        args.station_identity_registry_sha256 = identity_registry_sha256
        args.survey_config_sha256 = survey_config_sha256
        args.calibration_profile_sha256 = calibration_sha256
        args.survey_input_binding_sha256 = survey_input_binding_sha256
        provenance = _catalog_provenance(
            args,
            map_sha256=map_sha256,
            world_sha256=world_sha256,
            map_bundle_sha256=map_bundle.bundle_sha256,
            candidate_snapshot_sha256=args.candidate_snapshot_sha256,
            station_identity_registry_sha256=identity_registry_sha256,
            survey_config_sha256=survey_config_sha256,
            calibration_profile_sha256=calibration_sha256,
            survey_input_binding_sha256=survey_input_binding_sha256,
        )
        for candidate in candidates:
            if args.catalog.exists():
                catalog = load_arrival_pose_catalog(
                    args.catalog,
                    required_provenance=provenance,
                )
                record = catalog.record_for(candidate.candidate_uid)
                if record is not None:
                    if (
                        abs(record.stand.x_m - candidate.x_m) > 1.0e-6
                        or abs(record.stand.y_m - candidate.y_m) > 1.0e-6
                        or abs(record.stand.radius_m - candidate.radius_m) > 1.0e-9
                        or abs(
                            record.stand.uncertainty_m - candidate.uncertainty_m
                        ) > 1.0e-9
                        or record.stand_id != candidate.stand_id
                        or record.validation.validated_map_yaml_sha256
                        != map_bundle.yaml_sha256
                    ):
                        raise ValueError(
                            "resume candidate identity/geometry/map mismatch for "
                            f"{candidate.candidate_uid}"
                        )
                    print(f"Skipping already surveyed candidate {candidate.candidate_uid}")
                    continue
            print(f"Surveying {candidate.candidate_uid} at ({candidate.x_m}, {candidate.y_m})")
            _survey_one(
                args,
                candidate,
                candidates,
                world_sha256,
                provenance,
            )
        catalog = load_arrival_pose_catalog(
            args.catalog,
            required_provenance=provenance,
        )
        if not catalog.complete:
            unresolved = sorted(
                set(catalog.expected_candidate_uids)
                - set(catalog.resolved_candidate_uids)
            )
            raise RuntimeError(f"arrival survey incomplete: {unresolved}")
        if not catalog.frozen:
            catalog = freeze_arrival_pose_catalog(
                catalog,
                frozen_unix_sec=max(time.time(), catalog.updated_unix_sec),
            )
            write_arrival_pose_catalog(args.catalog, catalog)
        survey_manifest_path = None
        survey_manifest_sha256 = None
        if args.candidate_snapshot is not None:
            catalog_sha256 = arrival_pose_catalog_sha256(catalog)
            survey_manifest = SurveyManifest(
                schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
                manifest_id=(
                    f"survey_{args.session_id}_{catalog_sha256[:16]}"
                ),
                created_unix_sec=catalog.updated_unix_sec,
                session_id=args.session_id,
                environment="simulation",
                planning_frame=args.map_frame,
                map_bundle=artifact_reference(
                    "map_bundle",
                    map_bundle.semantic_map_id,
                    map_bundle.bundle_sha256,
                ),
                candidate_snapshot=artifact_reference(
                    "candidate_snapshot",
                    snapshot.snapshot_id,
                    args.candidate_snapshot_sha256,
                ),
                environment_descriptor=artifact_reference(
                    "simulation_world",
                    args.world.stem,
                    world_sha256,
                ),
                survey_config=artifact_reference(
                    "survey_config",
                    f"survey_config_{survey_config_sha256[:16]}",
                    survey_config_sha256,
                ),
                calibration_profile=artifact_reference(
                    "calibration_profile",
                    f"sim_calibration_{calibration_sha256[:16]}",
                    calibration_sha256,
                ),
                arrival_pose_catalog=artifact_reference(
                    "arrival_pose_catalog",
                    catalog.catalog_id,
                    catalog_sha256,
                ),
            )
            survey_manifest_path = args.survey_manifest or args.output_dir / (
                f"{survey_manifest.manifest_id}.json"
            )
            survey_manifest_sha256 = write_survey_manifest(
                survey_manifest_path,
                survey_manifest,
            )
        print(
            json.dumps(
                {
                    "ok": True,
                    "catalog": str(args.catalog),
                    "catalog_revision": catalog.revision,
                    "candidate_count": len(catalog.records),
                    "complete": catalog.complete,
                    "map_bundle_json": str(args.map_bundle_json),
                    "map_bundle_sha256": map_bundle.bundle_sha256,
                    "candidate_snapshot_sha256": args.candidate_snapshot_sha256,
                    "station_identity_registry_sha256": (
                        identity_registry_sha256
                    ),
                    "survey_input_binding": str(survey_input_binding_path),
                    "survey_input_binding_sha256": survey_input_binding_sha256,
                    "survey_config": str(survey_config_path),
                    "survey_config_sha256": survey_config_sha256,
                    "calibration_profile": str(calibration_path),
                    "calibration_profile_sha256": calibration_sha256,
                    "survey_manifest": (
                        "" if survey_manifest_path is None else str(survey_manifest_path)
                    ),
                    "survey_manifest_sha256": survey_manifest_sha256 or "",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
