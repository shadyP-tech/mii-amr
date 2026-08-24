"""Run one validated Aufgabe 04 station-route segment on a TurtleBot."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosPreflightResult,
    run_ros_preflight,
)
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import (
    RuntimeConfig,
    resolve_topic,
    resolve_runtime_config,
)
from scripts.aufgabe04.navigation.foundation.run_events import configure_event_logger, emit_event
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    DynamicRouteSource,
    RouteUpdate,
    RouteUpdateKind,
    validate_arena_boundary_evidence,
)
from scripts.aufgabe04.navigation.control.driving_behavior import (
    CATALOG_PHYSICAL_ROUTE_KINDS,
    CommandSmoothingConfig,
    DYNAMIC_VIEWPOINT_ROUTE_KINDS,
    HEADING_CORRIDOR_ROUTE_KINDS,
    PHYSICAL_ROUTE_KINDS,
    STATIC_PHYSICAL_ROUTE_KINDS,
    controller_config_for_route_kind,
)
from scripts.aufgabe04.navigation.foundation.content_hashed_evidence import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator import (
    CoverageReplanCoordinator,
)
from scripts.aufgabe04.navigation.execution.route_revision_store import (
    LoadedRouteRevision,
    RouteRevisionError,
    read_committed_revision,
    read_route_revision,
)
from scripts.aufgabe04.navigation.control.safety_checks import (
    catalog_start_egress_certificate,
    validate_catalog_route_binding_json,
    validate_route_diagnostics_json,
    validate_speed_limits,
)
from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import (
    DETECTED_STAND_PREAPPROACH_ROUTE_KIND,
    validate_detected_stand_preapproach_binding,
)
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
    validate_stand_discovery_route_binding,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    load_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.foundation.segment_run_logger import append_segment_run
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    execution_route_certificate_sha256,
    load_execution_route_certificate,
    validate_execution_route_identity,
)
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.execution.mission_leg_motion_consumption import (
    consume_mission_leg_motion_permit,
    default_mission_leg_motion_consumption_receipt_path,
    mission_leg_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind,
    MissionLegMotionPermit,
    mission_leg_motion_permit_sha256,
    validate_mission_leg_motion_permit_for_execution,
)
from scripts.aufgabe04.navigation.execution.mission_leg_identity_args import (
    build_mission_leg_event_fields,
    resolve_coverage_mission_leg_identity,
    resolve_explicit_mission_leg_evidence_identity,
    resolve_mission_leg_event_identity,
    resolve_startup_reseal_permit_identity,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    OdomExecutionCertificate,
    PlanarTransform2D,
    odom_execution_certificate_sha256,
    odom_pose_to_map,
    pose_route_sha256,
    transform_map_route_to_odom,
    validate_odom_execution_identity,
    write_odom_execution_certificate,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    OdomExecutionContext,
    adapt_map_route_update_to_odom,
    evaluate_map_odom_stationary_stability,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    evaluate_route_uncertainty_admission,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
    RuntimeLocalizationMotionPermit,
    runtime_localization_motion_permit_sha256,
    validate_runtime_localization_motion_permit_for_execution,
)
from scripts.aufgabe04.navigation.execution.runtime_motion_consumption import (
    consume_runtime_motion_permit,
    default_runtime_motion_consumption_receipt_path,
    runtime_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    StartupResealMotionPermit,
    startup_reseal_motion_permit_sha256,
    validate_startup_reseal_motion_permit_for_execution,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_consumption import (
    consume_startup_reseal_motion_permit,
    default_startup_reseal_motion_consumption_receipt_path,
    startup_reseal_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.execution.mission_execution_gate import (
    DiagnosticsSnapshot,
    MissionExecutionBinding,
    load_diagnostics_snapshot,
    validate_logistics_execution_bundle,
)
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
)
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    run_simple_waypoint_follower,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading import (
    intermediate_terminal_heading_entry_tolerance_m,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    DEFAULT_LINEAR_MOTION_FLOOR_MPS,
    PersistentObstacleConfig,
)
from scripts.aufgabe04.navigation.coverage.transient_overlay_resume_state import (
    TransientOverlayResumeState,
    transient_overlay_resume_state_sha256,
    validate_transient_overlay_resume_state_diagnostics_binding,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.planning.waypoint_csv import (
    SelectedRouteLeg,
    load_route_leg,
    poses_from_waypoints,
)


DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/station_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/station_route_diagnostics.json")


from .reporting import (
    _append_jsonl,
    _record_motion_authorization_rejection,
    _base_log_row,
    _append_result,
    _append_status_result,
    _observation_log_rows,
)


DEFAULT_RUN_LOG = Path("results/aufgabe04/station_segment_runs.csv")
DEFAULT_EVENT_LOG_DIR = Path("results/aufgabe04/run_events")
_CATALOG_ROUTE_INITIAL_DISTANCE_LIMIT_M = 0.15
LEGACY_SIMULATION_ROUTE_KIND = "legacy_simulation_waypoint"


from .route_bundle import (
    _execution_initial_distance_limit,
    _static_start_preflight_rejection,
    _simulation_odom_fallback_admission_failure,
    _load_execution_route_leg,
    _runtime_command_owner,
    _execution_certificate_failures,
    _resolved_map_execution_certificate,
    _authoritative_route_paths,
    _revalidate_authoritative_route_before_motion,
)








from .argument_validation import prepare_runtime_arguments
from .execution_route_admission import (
    admit_execution_route,
    _validated_coverage_replan_resume_state,
)

from .cli import (
    build_parser,
)








from .localization_admission import (
    _preflight_pose,
    _preflight_map_from_odom,
    _preflight_stationary_map_from_odom_window,
    _conservative_preflight_covariance,
    _angle_distance_rad,
    _covariance_bounded_continuity_limits,
    _admit_stationary_map_from_odom_window,
    _build_odom_execution_admission,
    _OdomRouteUncertaintyGate,
    _OdomBlockageRecoveryAdapter,
)




















def _physical_checklist(args, resolved) -> None:
    print("\nThis command will publish to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - clear the arena and station approach zones")
    print("  - keep an operator beside the robot")
    print("  - keep Ctrl+C ready in this terminal and physical stop available")
    print(f"  - keep a separate terminal ready to publish zero Twist to {resolved.cmd_vel_topic}")
    print("  - verify the resolved namespace, topics, and frames match this robot")
    print("  - verify no active Nav2 goal/controller or other follower is publishing velocity commands")
    print("  - verify scan, odom, TF, and configured localization data are fresh")
    print("  - verify exactly one AMCL or SLAM source owns the route localization transform")
    print("  - verify real-robot runtime nodes are not using simulated time")
    print(f"  - after RUN, wait up to {args.initial_sensor_wait_sec:.1f}s for follower scan/odom/TF before motion")
    print(f"Run ID: {args.run_id}")
    print(f"Resolved cmd_vel: {resolved.cmd_vel_topic}")


def _confirm_motion(args, resolved) -> bool:
    if args.allow_sim_time:
        print("Simulation run detected (--allow-sim-time); starting without a blocking RUN prompt.")
        return True
    _physical_checklist(args, resolved)
    response = input("Type RUN to start station-segment following: ").strip()
    return response == "RUN"


def _validated_runtime_localization_motion_permit(
    args,
    resolved,
    *,
    route_csv_path: Path,
    diagnostics_path: Path,
) -> RuntimeLocalizationMotionPermit | None:
    """Return the exact recovery permit or preserve normal interactive motion."""

    paths = (
        args.mission_motion_authorization_json,
        args.runtime_localization_motion_permit_json,
    )
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError(
            "mission motion authorization and runtime localization permit "
            "must be supplied together"
        )
    if args.dry_run:
        raise ValueError("runtime localization motion permit is live-run only")
    if args.allow_sim_time:
        raise ValueError(
            "runtime localization motion permit is physical-runtime only"
        )
    if args.execution_pose_frame != "odom":
        raise ValueError(
            "runtime localization motion permit requires odom execution"
        )
    if args.route_certificate_json is None:
        raise ValueError(
            "runtime localization motion permit requires a map route certificate"
        )
    if args.coverage_transient_replan_leg_index is None:
        raise ValueError(
            "runtime localization motion permit requires a coverage leg index"
        )
    target_viewpoint_id = str(
        args.coverage_transient_replan_target_viewpoint_id
    ).strip()
    semantic_map_id = str(
        args.coverage_transient_replan_semantic_map_id
    ).strip()
    session_id = str(args.mission_session_id).strip()
    if not target_viewpoint_id or not semantic_map_id or not session_id:
        raise ValueError(
            "runtime localization motion permit requires session, semantic map, "
            "and target identities"
        )
    return validate_runtime_localization_motion_permit_for_execution(
        args.runtime_localization_motion_permit_json,
        master_authorization_path=args.mission_motion_authorization_json,
        session_id=session_id,
        run_id=args.run_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        cmd_vel_topic=resolved.cmd_vel_topic,
        semantic_map_id=semantic_map_id,
        target_viewpoint_id=target_viewpoint_id,
        leg_index=args.coverage_transient_replan_leg_index,
        localization_branch_proof_id=args.localization_branch_proof_id,
        route_csv_path=route_csv_path,
        diagnostics_path=diagnostics_path,
        map_route_certificate_path=args.route_certificate_json,
    )


def _validated_startup_reseal_motion_permit(
    args,
    resolved,
    *,
    route_csv_path: Path,
    diagnostics_path: Path,
) -> StartupResealMotionPermit | None:
    """Return one exact startup-reseal permit or preserve normal prompting."""

    core_fields = (
        args.startup_reseal_motion_authorization_json,
        args.startup_reseal_motion_permit_json,
        str(args.startup_reseal_semantic_map_id).strip() or None,
    )
    generic_identity_fields = (
        args.startup_reseal_mission_leg_kind,
        args.startup_reseal_mission_leg_index,
        str(args.startup_reseal_target_id).strip() or None,
    )
    legacy_identity_present = bool(
        str(args.startup_reseal_target_viewpoint_id).strip()
    )
    if (
        all(value is None for value in core_fields)
        and all(value is None for value in generic_identity_fields)
        and not legacy_identity_present
    ):
        return None
    if any(value is None for value in core_fields):
        raise ValueError(
            "startup-reseal motion authorization arguments must be supplied together"
        )
    generic_identity_requested = any(
        value is not None for value in generic_identity_fields
    )
    if generic_identity_requested and any(
        value is None for value in generic_identity_fields
    ):
        raise ValueError(
            "generic startup-reseal identity arguments must be supplied together"
        )
    if any(
        value is not None
        for value in (
            args.mission_motion_authorization_json,
            args.runtime_localization_motion_permit_json,
            args.mission_leg_motion_authorization_json,
            args.mission_leg_motion_permit_json,
        )
    ):
        raise ValueError(
            "startup-reseal, routine-leg, and runtime-localization permits "
            "are mutually exclusive"
        )
    if args.dry_run:
        raise ValueError("startup-reseal motion permit is live-run only")
    if args.allow_sim_time:
        raise ValueError("startup-reseal motion permit is physical-runtime only")
    if args.execution_pose_frame != "odom":
        raise ValueError("startup-reseal motion permit requires odom execution")
    if args.route_certificate_json is None:
        raise ValueError(
            "startup-reseal motion permit requires a map route certificate"
        )
    if generic_identity_requested:
        mission_leg_kind = MissionLegKind(
            args.startup_reseal_mission_leg_kind
        )
        mission_leg_index = args.startup_reseal_mission_leg_index
        target_id = str(args.startup_reseal_target_id).strip()
        if mission_leg_index is None or mission_leg_index < 0:
            raise ValueError(
                "startup-reseal mission leg index must be non-negative"
            )
    else:
        if (
            args.coverage_transient_replan_leg_index is None
            or not legacy_identity_present
        ):
            raise ValueError(
                "legacy startup-reseal identity requires a coverage leg"
            )
        mission_leg_kind = MissionLegKind.COVERAGE
        mission_leg_index = args.coverage_transient_replan_leg_index
        target_id = str(args.startup_reseal_target_viewpoint_id).strip()
    if mission_leg_kind is MissionLegKind.COVERAGE:
        coverage_identity = resolve_coverage_mission_leg_identity(args)
        if coverage_identity is None:
            raise ValueError(
                "coverage startup-reseal permit requires coverage "
                "transient-replan identity"
            )
        if coverage_identity != (
            mission_leg_kind,
            mission_leg_index,
            target_id,
        ):
            raise ValueError(
                "startup-reseal and coverage transient-replan identities "
                "mismatch"
            )
    elif args.coverage_transient_replan_enabled:
        raise ValueError(
            "non-coverage startup-reseal permit cannot carry coverage "
            "transient replanning"
        )
    evidence_identity = resolve_explicit_mission_leg_evidence_identity(args)
    if evidence_identity is not None and evidence_identity != (
        mission_leg_kind,
        mission_leg_index,
        target_id,
    ):
        raise ValueError(
            "mission-leg evidence and startup-reseal identities mismatch"
        )
    session_id = str(args.mission_session_id).strip()
    if not session_id:
        raise ValueError("startup-reseal motion permit requires mission_session_id")
    return validate_startup_reseal_motion_permit_for_execution(
        args.startup_reseal_motion_permit_json,
        master_authorization_path=(
            args.startup_reseal_motion_authorization_json
        ),
        run_id=args.run_id,
        session_id=session_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        cmd_vel_topic=resolved.cmd_vel_topic,
        semantic_map_id=str(args.startup_reseal_semantic_map_id).strip(),
        target_viewpoint_id=target_id,
        leg_index=mission_leg_index,
        mission_leg_kind=mission_leg_kind,
        mission_leg_index=mission_leg_index,
        target_id=target_id,
        localization_branch_proof_id=args.localization_branch_proof_id,
        route_csv_path=route_csv_path,
        diagnostics_path=diagnostics_path,
        map_route_certificate_path=args.route_certificate_json,
    )




def _validated_mission_leg_motion_permit(
    args,
    resolved,
    *,
    route_csv_path: Path,
    diagnostics_path: Path,
) -> MissionLegMotionPermit | None:
    """Return one exact routine-leg permit or preserve interactive motion."""

    fields = (
        args.mission_leg_motion_authorization_json,
        args.mission_leg_motion_permit_json,
        args.mission_leg_kind,
        args.mission_leg_index,
        str(args.mission_leg_target_id).strip() or None,
        str(args.mission_leg_semantic_map_id).strip() or None,
        args.mission_leg_dry_preflight_json,
        args.mission_leg_dry_odom_certificate_json,
        args.mission_leg_dry_uncertainty_budget_json,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise ValueError(
            "mission-leg motion authorization arguments must be supplied together"
        )
    if (
        args.mission_motion_authorization_json is not None
        or args.runtime_localization_motion_permit_json is not None
    ):
        raise ValueError(
            "routine mission-leg and runtime-localization permits are "
            "mutually exclusive"
        )
    if args.dry_run:
        raise ValueError("mission-leg motion permit is live-run only")
    if args.allow_sim_time:
        raise ValueError("mission-leg motion permit is physical-runtime only")
    if args.execution_pose_frame != "odom":
        raise ValueError("mission-leg motion permit requires odom execution")
    if args.route_certificate_json is None:
        raise ValueError(
            "mission-leg motion permit requires a map route certificate"
        )
    session_id = str(args.mission_session_id).strip()
    if not session_id:
        raise ValueError("mission-leg motion permit requires mission_session_id")
    assert args.mission_leg_kind is not None
    assert args.mission_leg_index is not None
    return validate_mission_leg_motion_permit_for_execution(
        args.mission_leg_motion_permit_json,
        master_authorization_path=(
            args.mission_leg_motion_authorization_json
        ),
        session_id=session_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        cmd_vel_topic=resolved.cmd_vel_topic,
        semantic_map_id=str(args.mission_leg_semantic_map_id).strip(),
        localization_branch_proof_id=args.localization_branch_proof_id,
        run_id=args.run_id,
        mission_leg_kind=args.mission_leg_kind,
        mission_leg_index=args.mission_leg_index,
        target_id=str(args.mission_leg_target_id).strip(),
        route_csv_path=route_csv_path,
        diagnostics_path=diagnostics_path,
        map_route_certificate_path=args.route_certificate_json,
        dry_preflight_path=args.mission_leg_dry_preflight_json,
        dry_odom_certificate_path=(
            args.mission_leg_dry_odom_certificate_json
        ),
        dry_uncertainty_budget_path=(
            args.mission_leg_dry_uncertainty_budget_json
        ),
    )




def _prompt_for_initialpose(args, resolved) -> None:
    if not args.prompt_for_initialpose:
        return
    print("\nInitial-pose refresh required before ROS preflight.")
    print("AMCL often publishes only once after RViz 2D Pose Estimate.")
    print("The preflight subscriber must already be active, so do not click yet.")
    print(f"AMCL topic: {resolved.amcl_topic}")
    print(
        "Press Enter here, then immediately click 2D Pose Estimate in RViz "
        f"during the next {args.preflight_observation_window_sec:.1f}s."
    )
    input("Press Enter, then click 2D Pose Estimate immediately: ")














def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    odom_execution_enabled, event_logger = prepare_runtime_arguments(
        parser,
        args,
    )
    require_motion = not args.allow_noop
    runtime_config = RuntimeConfig(
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
    resolved = resolve_runtime_config(runtime_config)
    resolved_runtime_nomotion_update_service = resolve_topic(
        args.runtime_nomotion_update_service,
        resolved.namespace,
    )
    (
        route_csv_path,
        diagnostics_json_path,
        committed_route,
        leg,
        diagnostics_snapshot,
        route_purpose,
        coverage_replan_resume_state,
        allow_simulation_odom_after_stale_tf,
        catalog_egress_certificate,
        mission_execution_binding,
    ) = admit_execution_route(
        parser=parser,
        args=args,
        resolved=resolved,
        resolved_runtime_nomotion_update_service=(
            resolved_runtime_nomotion_update_service
        ),
        odom_execution_enabled=odom_execution_enabled,
        event_logger=event_logger,
        require_motion=require_motion,
    )
    print("Resolved runtime config:")
    print(json.dumps(resolved.as_log_dict(), indent=2, sort_keys=True))
    print(f"Semantic log: {args.semantic_log}")
    print(f"Results CSV: {args.results_csv}")
    print(
        "Route leg: "
        f"raw={len(leg.raw_waypoints)} executable={len(leg.executable_waypoints)} "
        f"length={leg.route_length_m:.3f}m"
    )
    if args.allow_noop and leg.route_length_m <= 0.0:
        result = FollowerResult("noop", "zero-length leg", 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=False, result=result)
        emit_event(
            event_logger,
            "dry_run_completed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            status=result.status,
            stop_reason=result.stop_reason,
            motion_published=result.motion_published,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        print("No-op leg logged; no motion was published.")
        return 0

    _prompt_for_initialpose(args, resolved)

    try:
        preflight = run_ros_preflight(
            resolved,
            max_scan_age_sec=args.max_scan_age_sec,
            max_odom_age_sec=args.max_odom_age_sec,
            max_tf_age_sec=args.max_tf_age_sec,
            max_amcl_age_sec=args.max_amcl_age_sec,
            max_future_timestamp_sec=args.max_future_timestamp_sec,
            max_localization_tf_future_sec=(
                args.max_localization_tf_future_sec
            ),
            observation_window_sec=args.preflight_observation_window_sec,
            allowed_cmd_vel_publishers=args.allowed_cmd_vel_publisher,
            require_real_time=not args.allow_sim_time,
            request_nomotion_update=(
                resolved.localization_source == "amcl"
                and not args.skip_nomotion_update_before_preflight
            ),
            nomotion_update_service=args.nomotion_update_service,
            nomotion_update_timeout_sec=args.nomotion_update_timeout_sec,
            stationary_amcl_sample_count=(
                args.stationary_amcl_sample_count
            ),
            stationary_amcl_sample_interval_sec=(
                args.stationary_amcl_sample_interval_sec
            ),
            max_stationary_amcl_position_spread_m=(
                args.max_stationary_amcl_position_spread_m
            ),
            max_stationary_amcl_yaw_spread_rad=(
                args.max_stationary_amcl_yaw_spread_rad
            ),
            max_stationary_amcl_position_std_m=(
                args.max_stationary_amcl_position_std_m
            ),
            max_stationary_amcl_yaw_std_rad=(
                args.max_stationary_amcl_yaw_std_rad
            ),
            execution_pose_owner=(
                "odom" if args.execution_pose_frame == "odom" else ""
            ),
            global_consistency_monitor=(
                "amcl" if args.execution_pose_frame == "odom" else ""
            ),
            # The runner constructs and validates the certificate immediately
            # from this stopped capture before dry-run success or RUN can be
            # reached. This flag selects the intended ownership contract; the
            # resulting artifact is still mandatory below.
            frozen_map_transform_certified=(
                args.execution_pose_frame == "odom"
            ),
        )
    except RuntimeError as exc:
        stop_reason = str(exc)
        emit_event(
            event_logger,
            "preflight_failed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            **build_mission_leg_event_fields(args),
            failures=[stop_reason],
            observations=[],
            runtime_config=resolved.as_log_dict(),
        )
        _append_status_result(
            args,
            resolved,
            leg,
            preflight_ok=False,
            status="preflight_unavailable",
            stop_reason=stop_reason,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="preflight_unavailable",
            stop_reason=stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        parser.exit(2, f"error: ROS preflight failed to run: {exc}\n")
    preflight_text = json.dumps(preflight.to_json_dict(), indent=2, sort_keys=True)
    if args.preflight_json is not None:
        args.preflight_json.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_json.write_text(preflight_text + "\n")
    print(preflight_text)
    if not preflight.ok:
        emit_event(
            event_logger,
            "preflight_failed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            failures=preflight.failures,
            observations=_observation_log_rows(preflight.observations),
            runtime_config=preflight.runtime_config,
        )
        result = FollowerResult("preflight_failed", "; ".join(preflight.failures), 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=False, result=result)
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            stop_reason=result.stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 1
    emit_event(
        event_logger,
        "preflight_passed",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        failures=[],
        observations=_observation_log_rows(preflight.observations),
        runtime_config=preflight.runtime_config,
    )
    startup_rejection = _static_start_preflight_rejection(
        preflight,
        leg,
        map_frame=resolved.map_frame,
        base_frame=resolved.base_frame,
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
    )
    if startup_rejection is not None:
        _append_result(
            args,
            resolved,
            leg,
            preflight_ok=True,
            result=startup_rejection,
        )
        emit_event(
            event_logger,
            "startup_route_rejected",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            **build_mission_leg_event_fields(args),
            status=startup_rejection.status,
            stop_reason=startup_rejection.stop_reason,
            motion_published=False,
            stop_details=startup_rejection.stop_details,
        )
        emit_event(
            event_logger,
            "safety_stop",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            **build_mission_leg_event_fields(args),
            status=startup_rejection.status,
            stop_reason=startup_rejection.stop_reason,
            motion_published=False,
            stop_details=startup_rejection.stop_details,
            duration_sec=0.0,
            distance_estimate_m=0.0,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=startup_rejection.status,
            stop_reason=startup_rejection.stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 1
    execution_waypoints = poses_from_waypoints(leg.executable_waypoints)
    odom_execution_context: OdomExecutionContext | None = None
    odom_execution_evidence: dict[str, object] = {}
    odom_replacement_route_gate: _OdomRouteUncertaintyGate | None = None
    if args.execution_pose_frame == "odom":
        try:
            (
                execution_waypoints,
                odom_execution_context,
                odom_execution_evidence,
                odom_replacement_route_gate,
            ) = _build_odom_execution_admission(
                args=args,
                resolved=resolved,
                leg=leg,
                preflight=preflight,
                diagnostics_snapshot=diagnostics_snapshot,
            )
        except (OSError, ValueError) as exc:
            stop_reason = f"odom execution admission failed: {exc}"
            stop_details = {
                "reason": stop_reason,
                "fault_code": "odom_execution_admission_failed",
                "execution_pose_owner": "odom",
                "global_consistency_monitor": "amcl",
                "motion_published": False,
                "fail_closed": True,
            }
            result = FollowerResult(
                "preflight_failed",
                stop_reason,
                0.0,
                0.0,
                False,
                stop_details,
            )
            _append_result(
                args,
                resolved,
                leg,
                preflight_ok=False,
                result=result,
            )
            emit_event(
                event_logger,
                "odom_execution_admission_failed",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status=result.status,
                stop_reason=stop_reason,
                motion_published=False,
                stop_details=stop_details,
            )
            emit_event(
                event_logger,
                "safety_stop",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status=result.status,
                stop_reason=stop_reason,
                motion_published=False,
                stop_details=stop_details,
                duration_sec=0.0,
                distance_estimate_m=0.0,
            )
            emit_event(
                event_logger,
                "run_finished",
                run_id=args.run_id,
                final_status=result.status,
                stop_reason=stop_reason,
                results_csv=str(args.results_csv),
                semantic_log_path=str(args.semantic_log),
                preflight_json_path=str(args.preflight_json or ""),
            )
            return 1
        emit_event(
            event_logger,
            "odom_execution_sealed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            **odom_execution_evidence,
        )
    if args.dry_run:
        result = FollowerResult("dry_run_ok", "", 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=True, result=result)
        emit_event(
            event_logger,
            "dry_run_completed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            **build_mission_leg_event_fields(args),
            status=result.status,
            motion_published=result.motion_published,
            results_csv=str(args.results_csv),
            execution_pose_frame=args.execution_pose_frame,
            odom_execution_evidence=odom_execution_evidence,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 0
    runtime_motion_permit = None
    mission_leg_motion_permit = None
    startup_reseal_motion_permit = None
    try:
        runtime_motion_permit = _validated_runtime_localization_motion_permit(
            args,
            resolved,
            route_csv_path=route_csv_path,
            diagnostics_path=diagnostics_json_path,
        )
        mission_leg_motion_permit = _validated_mission_leg_motion_permit(
            args,
            resolved,
            route_csv_path=route_csv_path,
            diagnostics_path=diagnostics_json_path,
        )
        startup_reseal_motion_permit = (
            _validated_startup_reseal_motion_permit(
                args,
                resolved,
                route_csv_path=route_csv_path,
                diagnostics_path=diagnostics_json_path,
            )
        )
    except ValueError as exc:
        return _record_motion_authorization_rejection(
            args=args,
            resolved=resolved,
            leg=leg,
            event_logger=event_logger,
            failure=exc,
        )

    if (
        runtime_motion_permit is None
        and mission_leg_motion_permit is None
        and startup_reseal_motion_permit is None
        and not _confirm_motion(args, resolved)
    ):
        result = FollowerResult("aborted", "operator did not type RUN", 0.0, 0.0, False)
        _append_result(args, resolved, leg, preflight_ok=True, result=result)
        emit_event(
            event_logger,
            "operator_aborted",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            status=result.status,
            stop_reason=result.stop_reason,
            motion_published=result.motion_published,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status=result.status,
            stop_reason=result.stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        return 1

    if committed_route is not None:
        try:
            _revalidate_authoritative_route_before_motion(args, committed_route)
        except (OSError, RouteRevisionError) as exc:
            stop_reason = f"authoritative route revalidation failed: {exc}"
            emit_event(
                event_logger,
                "route_manifest_rejected",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status="stopped",
                phase="immediately_before_motion",
                stop_reason=stop_reason,
                route_manifest=str(committed_route.manifest_path),
            )
            result = FollowerResult(
                "stopped",
                stop_reason,
                0.0,
                0.0,
                False,
                {
                    "fault_code": getattr(exc, "code", "route_revalidation_io"),
                    "fail_closed": True,
                },
            )
            _append_result(args, resolved, leg, preflight_ok=True, result=result)
            emit_event(
                event_logger,
                "safety_stop",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status=result.status,
                stop_reason=result.stop_reason,
                motion_published=False,
                stop_details=result.stop_details,
            )
            emit_event(
                event_logger,
                "run_finished",
                run_id=args.run_id,
                final_status=result.status,
                stop_reason=result.stop_reason,
                results_csv=str(args.results_csv),
                semantic_log_path=str(args.semantic_log),
                preflight_json_path=str(args.preflight_json or ""),
            )
            return 1

    if mission_execution_binding is not None:
        try:
            current_diagnostics_snapshot = load_diagnostics_snapshot(
                diagnostics_json_path
            )
            current_binding = validate_logistics_execution_bundle(
                route_leg=leg,
                diagnostics_path=diagnostics_json_path,
                route_certificate_path=args.route_certificate_json,
                mission_plan_path=args.mission_plan_manifest,
                survey_manifest_path=args.survey_manifest,
                route_bundle_path=args.route_bundle_json,
                planner_config_path=args.planner_config_json,
                runtime_map_bundle_path=args.runtime_map_bundle_json,
                runtime_environment_path=args.runtime_environment,
                candidate_snapshot_path=args.candidate_snapshot,
                station_identity_registry_path=args.station_identity_registry,
                arrival_pose_catalog_path=args.arrival_pose_catalog,
                task_snapshot_path=args.task_snapshot,
                robot_id=args.robot_id,
                runtime_planning_frame=resolved.map_frame,
                diagnostics_snapshot=current_diagnostics_snapshot,
            )
            if current_binding != mission_execution_binding:
                raise ValueError("mission execution artifacts changed before motion")
        except (OSError, ValueError) as exc:
            stop_reason = f"mission execution revalidation failed: {exc}"
            result = FollowerResult(
                "stopped",
                stop_reason,
                0.0,
                0.0,
                False,
                {"fault_code": "mission_revalidation_failed", "fail_closed": True},
            )
            _append_result(args, resolved, leg, preflight_ok=True, result=result)
            emit_event(
                event_logger,
                "mission_execution_rejected",
                run_id=args.run_id,
                leg_index=leg.leg_index,
                status="stopped",
                phase="immediately_before_motion",
                stop_reason=stop_reason,
            )
            emit_event(
                event_logger,
                "run_finished",
                run_id=args.run_id,
                final_status=result.status,
                stop_reason=result.stop_reason,
                results_csv=str(args.results_csv),
                semantic_log_path=str(args.semantic_log),
                preflight_json_path=str(args.preflight_json or ""),
            )
            return 1

    execution_initial_distance_limit_m = _execution_initial_distance_limit(
        args.initial_distance_limit_m,
        leg.route_kind,
    )
    static_start_join_clearance_m = (
        None
        if catalog_egress_certificate is None
        or not catalog_egress_certificate.required
        or catalog_egress_certificate.start_join_clearance_m is None
        else min(
            execution_initial_distance_limit_m,
            catalog_egress_certificate.start_join_clearance_m,
        )
    )
    follower_config = FollowerConfig(
        controller=ControllerConfig(
            max_linear_mps=args.max_linear_mps,
            max_angular_radps=args.max_angular_radps,
            goal_tolerance_m=args.goal_tolerance_m,
            heading_tolerance_rad=args.heading_tolerance_rad,
            lookahead_distance_m=args.lookahead_distance_m,
            slow_heading_error_rad=args.slow_heading_error_rad,
            stop_heading_error_rad=args.stop_heading_error_rad,
            min_linear_speed_scale=args.min_linear_speed_scale,
            max_progress_advance_m=args.max_progress_advance_m,
            enforce_heading_corridor=(
                leg.route_kind in HEADING_CORRIDOR_ROUTE_KINDS
            ),
            exact_vertex_pursuit=leg.route_kind in PHYSICAL_ROUTE_KINDS,
        ),
        command_smoothing=CommandSmoothingConfig(
            enabled=not args.disable_command_smoothing,
            max_linear_accel_mps2=args.max_linear_accel_mps2,
            max_angular_accel_radps2=args.max_angular_accel_radps2,
        ),
        min_obstacle_distance_m=args.min_obstacle_distance_m,
        omnidirectional_hard_stop_distance_m=(
            args.omnidirectional_hard_stop_distance_m
        ),
        front_obstacle_slow_distance_m=args.front_obstacle_slow_distance_m,
        front_obstacle_sector_rad=args.front_obstacle_sector_rad,
        max_scan_age_sec=args.max_scan_age_sec,
        max_odom_age_sec=args.max_odom_age_sec,
        max_tf_age_sec=args.max_tf_age_sec,
        max_future_timestamp_sec=args.max_future_timestamp_sec,
        amcl_edge_future_tolerance_sec=(
            args.max_localization_tf_future_sec
        ),
        runtime_nomotion_update_service=(
            args.runtime_nomotion_update_service
        ),
        runtime_nomotion_update_timeout_sec=(
            args.runtime_nomotion_update_timeout_sec
        ),
        allow_simulation_odom_after_stale_tf=(
            allow_simulation_odom_after_stale_tf
        ),
        initial_sensor_wait_sec=args.initial_sensor_wait_sec,
        waypoint_timeout_sec=args.waypoint_timeout_sec,
        stuck_timeout_sec=args.stuck_timeout_sec,
        stuck_progress_epsilon_m=args.stuck_progress_epsilon_m,
        stuck_heading_progress_epsilon_rad=(
            args.stuck_heading_progress_epsilon_rad
        ),
        linear_motion_floor_mps=args.linear_motion_floor_mps,
        blockage_confirmation_timeout_sec=(
            args.blockage_confirmation_timeout_sec
        ),
        persistent_obstacle_config=PersistentObstacleConfig(
            min_distinct_samples=args.blockage_confirmation_min_samples,
            min_front_range_m=args.omnidirectional_hard_stop_distance_m,
            max_front_range_m=args.front_obstacle_slow_distance_m,
            front_sector_half_width_rad=args.front_obstacle_sector_rad,
        ),
        initial_distance_limit_m=execution_initial_distance_limit_m,
        allowed_cmd_vel_publishers=tuple(args.allowed_cmd_vel_publisher),
        dynamic_route_refresh_sec=args.dynamic_route_refresh_sec,
        dynamic_join_tolerance_m=args.dynamic_route_join_tolerance_m,
        start_egress_waypoint_tolerance_m=(
            args.start_egress_waypoint_tolerance_m
        ),
        start_egress_alignment_tolerance_rad=(
            args.start_egress_alignment_tolerance_rad
        ),
        start_egress_max_linear_mps=args.start_egress_max_linear_mps,
        initial_start_egress_waypoint_index=(
            None
            if catalog_egress_certificate is None
            else catalog_egress_certificate.waypoint_index
        ),
        initial_start_join_clearance_m=static_start_join_clearance_m,
        initial_route_kind=leg.route_kind,
        axis_acquisition_wait_timeout_sec=args.axis_acquisition_wait_timeout_sec,
        viewpoint_sampling_timeout_sec=args.viewpoint_sampling_timeout_sec,
        viewpoint_sampling_target_timeout_sec=(
            args.viewpoint_sampling_target_timeout_sec
        ),
        viewpoint_sampling_goal_tolerance_m=(
            args.viewpoint_sampling_goal_tolerance_m
        ),
        viewpoint_sampling_terminal_heading_hold_tolerance_m=(
            args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        viewpoint_sampling_target_distance_m=(
            args.viewpoint_sampling_target_distance_m
        ),
        viewpoint_sampling_terminal_heading_target_envelope_radius_m=(
            args
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m
        ),
        viewpoint_sampling_heading_tolerance_rad=(
            args.viewpoint_sampling_heading_tolerance_rad
        ),
        physical_waypoint_tolerance_m=args.physical_waypoint_tolerance_m,
        physical_goal_tolerance_m=args.physical_goal_tolerance_m,
        certified_route_tube_radius_m=args.certified_route_tube_radius_m,
        certified_route_chord_sample_spacing_m=(
            args.certified_route_chord_sample_spacing_m
        ),
        certified_corner_max_reacquire_attempts=(
            args.certified_corner_max_reacquire_attempts
        ),
    )
    resolved_controller_config = controller_config_for_route_kind(
        follower_config.controller,
        leg.route_kind,
        viewpoint_sampling_goal_tolerance_m=(
            follower_config.viewpoint_sampling_goal_tolerance_m
        ),
        viewpoint_sampling_heading_tolerance_rad=(
            follower_config.viewpoint_sampling_heading_tolerance_rad
        ),
        physical_waypoint_tolerance_m=(
            follower_config.physical_waypoint_tolerance_m
        ),
        physical_goal_tolerance_m=follower_config.physical_goal_tolerance_m,
    )
    resolved_terminal_goal_tolerance_m = (
        resolved_controller_config.goal_tolerance_m
        if resolved_controller_config.terminal_goal_tolerance_m is None
        else resolved_controller_config.terminal_goal_tolerance_m
    )
    emit_event(
        event_logger,
        "controller_config_resolved",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        route_kind=leg.route_kind,
        max_linear_mps=follower_config.controller.max_linear_mps,
        max_angular_radps=follower_config.controller.max_angular_radps,
        min_obstacle_distance_m=follower_config.min_obstacle_distance_m,
        omnidirectional_hard_stop_distance_m=(
            follower_config.omnidirectional_hard_stop_distance_m
        ),
        coverage_transient_replan_enabled=(
            args.coverage_transient_replan_enabled
        ),
        coverage_transient_replan_max_count=(
            args.coverage_transient_replan_max_count
        ),
        coverage_transient_replan_resume_state_json=str(
            args.coverage_transient_replan_resume_state_json or ""
        ),
        coverage_transient_replan_initial_count=(
            0
            if coverage_replan_resume_state is None
            else coverage_replan_resume_state.completed_replan_count
        ),
        coverage_transient_replan_remaining_count=(
            args.coverage_transient_replan_max_count
            if coverage_replan_resume_state is None
            else coverage_replan_resume_state.remaining_replans
        ),
        linear_motion_floor_mps=follower_config.linear_motion_floor_mps,
        blockage_confirmation_timeout_sec=(
            follower_config.blockage_confirmation_timeout_sec
        ),
        blockage_confirmation_thresholds=(
            follower_config.persistent_obstacle_config.to_log_dict()
            if follower_config.persistent_obstacle_config is not None
            else {}
        ),
        controller_trace_jsonl=str(args.controller_trace_jsonl or ""),
        effective_goal_tolerance_m=resolved_terminal_goal_tolerance_m,
        effective_intermediate_goal_tolerance_m=(
            resolved_controller_config.goal_tolerance_m
        ),
        effective_terminal_goal_tolerance_m=(
            resolved_terminal_goal_tolerance_m
        ),
        intermediate_terminal_heading_entry_tolerance_m=(
            intermediate_terminal_heading_entry_tolerance_m(
                resolved_controller_config
            )
        ),
        intermediate_terminal_heading_hold_tolerance_m=(
            follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        intermediate_terminal_heading_distance_comparison_epsilon_m=(
            INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
        ),
        intermediate_terminal_heading_effective_hold_limit_m=(
            follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
            + INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
        ),
        intermediate_terminal_heading_target_distance_m=(
            follower_config.viewpoint_sampling_target_distance_m
        ),
        intermediate_terminal_heading_target_envelope_radius_m=(
            follower_config
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m
        ),
        intermediate_terminal_heading_minimum_stand_distance_m=(
            follower_config.viewpoint_sampling_target_distance_m
            - follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        intermediate_terminal_heading_maximum_stand_distance_m=(
            follower_config.viewpoint_sampling_target_distance_m
            + follower_config
            .viewpoint_sampling_terminal_heading_hold_tolerance_m
        ),
        heading_tolerance_rad=resolved_controller_config.heading_tolerance_rad,
        enforce_heading_corridor=(
            resolved_controller_config.enforce_heading_corridor
        ),
        slow_heading_error_rad=follower_config.controller.slow_heading_error_rad,
        stop_heading_error_rad=follower_config.controller.stop_heading_error_rad,
        exact_vertex_pursuit=resolved_controller_config.exact_vertex_pursuit,
        exact_vertex_alignment_enabled=(
            resolved_controller_config.exact_vertex_pursuit
        ),
        command_smoothing_enabled=follower_config.command_smoothing.enabled,
        max_linear_accel_mps2=(
            follower_config.command_smoothing.max_linear_accel_mps2
        ),
        max_angular_accel_radps2=(
            follower_config.command_smoothing.max_angular_accel_radps2
        ),
        start_egress_waypoint_index=(
            follower_config.initial_start_egress_waypoint_index
        ),
        start_egress_waypoint_tolerance_m=(
            follower_config.start_egress_waypoint_tolerance_m
        ),
        start_egress_alignment_tolerance_rad=(
            follower_config.start_egress_alignment_tolerance_rad
        ),
        start_egress_max_linear_mps=(
            follower_config.start_egress_max_linear_mps
        ),
        initial_start_join_clearance_m=(
            follower_config.initial_start_join_clearance_m
        ),
        certified_route_tube_radius_m=(
            follower_config.certified_route_tube_radius_m
        ),
        certified_route_chord_sample_spacing_m=(
            follower_config.certified_route_chord_sample_spacing_m
        ),
        certified_corner_transition_enabled=(
            leg.route_kind == "stand_discovery_corridor"
        ),
        certified_corner_turn_threshold_rad=(
            follower_config.certified_corner_turn_threshold_rad
        ),
        certified_corner_release_tolerance_m=(
            follower_config.certified_corner_release_tolerance_m
        ),
        certified_corner_hold_tolerance_m=(
            follower_config.certified_corner_hold_tolerance_m
        ),
        certified_corner_alignment_tolerance_rad=(
            follower_config.certified_corner_alignment_tolerance_rad
        ),
        certified_corner_max_reacquire_attempts=(
            follower_config.certified_corner_max_reacquire_attempts
        ),
        allow_simulation_odom_after_stale_tf=(
            follower_config.allow_simulation_odom_after_stale_tf
        ),
        amcl_edge_future_tolerance_sec=(
            follower_config.amcl_edge_future_tolerance_sec
        ),
        runtime_nomotion_update_service=(
            resolved_runtime_nomotion_update_service
        ),
        runtime_nomotion_update_service_configured=(
            follower_config.runtime_nomotion_update_service
        ),
        runtime_nomotion_update_timeout_sec=(
            follower_config.runtime_nomotion_update_timeout_sec
        ),
        route_purpose=route_purpose,
        route_simulation_only=leg.simulation_only,
    )
    waypoint_provider = None
    blockage_recovery_provider = None

    def route_update_callback(update):
        event_name = {
            "dynamic_route_adopted": "route_reloaded",
            "dynamic_route_withdrawn": "route_withdrawn",
            "dynamic_route_rejected": "route_reload_rejected",
            "dynamic_route_stopped": "route_reload_rejected",
            "dynamic_survey_completed": "survey_completed",
        }.get(update.event_name, update.event_name)
        if event_name is None:
            return
        emit_event(
            event_logger,
            event_name,
            run_id=args.run_id,
            leg_index=args.leg_index,
            **dict(update.event_fields),
        )
        if (
            event_name == "transient_navigation_blockage_replanned"
            and args.coverage_transient_replan_enabled
        ):
            # The coordinator prepares artifacts while zero is held, but this
            # callback runs only after the follower has atomically installed
            # the replacement.  Persist "replanned" here so the parent never
            # mistakes a merely prepared route for an adopted one.
            _append_jsonl(
                Path(args.coverage_transient_replan_session_root)
                / "adaptive_replans.jsonl",
                {
                    "schema_version": 1,
                    "event": event_name,
                    "timestamp": time.time(),
                    "run_id": args.run_id,
                    "leg_index": args.coverage_transient_replan_leg_index,
                    **dict(update.event_fields),
                },
            )

    if args.coverage_transient_replan_enabled:
        if coverage_replan_resume_state is not None:
            try:
                live_resume_state = _validated_coverage_replan_resume_state(
                    args,
                    diagnostics_path=diagnostics_json_path,
                )
                if live_resume_state != coverage_replan_resume_state:
                    raise ValueError(
                        "transient overlay resume state changed before motion"
                    )
            except (OSError, ValueError) as exc:
                return _record_motion_authorization_rejection(
                    args=args,
                    resolved=resolved,
                    leg=leg,
                    event_logger=event_logger,
                    failure=(
                        "transient overlay resume-state revalidation failed: "
                        f"{exc}"
                    ),
                )
        blockage_recovery_provider = CoverageReplanCoordinator(
            survey_root=args.coverage_transient_replan_survey_root,
            session_root=args.coverage_transient_replan_session_root,
            map_yaml=args.coverage_transient_replan_map,
            semantic_map_id=args.coverage_transient_replan_semantic_map_id,
            target_viewpoint_id=(
                args.coverage_transient_replan_target_viewpoint_id
            ),
            run_id=args.run_id,
            coverage_leg_index=args.coverage_transient_replan_leg_index,
            route_leg_index=leg.leg_index,
            command_owner=_runtime_command_owner(resolved.namespace),
            robot_radius_m=args.coverage_transient_replan_robot_radius_m,
            max_replans=args.coverage_transient_replan_max_count,
            replan_count=(
                0
                if coverage_replan_resume_state is None
                else coverage_replan_resume_state.completed_replan_count
            ),
            overlay_path=(
                None
                if coverage_replan_resume_state is None
                else Path(
                    coverage_replan_resume_state.transient_obstacle_overlay_path
                )
            ),
            adopted_route_hashes=(
                set()
                if coverage_replan_resume_state is None
                else set(
                    coverage_replan_resume_state.adopted_route_sha256s
                )
                | {leg.source_sha256}
            ),
            tracking_tube_radius_m=args.certified_route_tube_radius_m,
            forward_translation_heading_limit_rad=(
                follower_config.controller.stop_heading_error_rad
            ),
            reverse_connector_alignment_tolerance_rad=(
                follower_config.start_egress_alignment_tolerance_rad
            ),
        )
        if odom_execution_context is not None:
            assert odom_replacement_route_gate is not None
            blockage_recovery_provider = _OdomBlockageRecoveryAdapter(
                blockage_recovery_provider,
                odom_execution_context,
                odom_replacement_route_gate,
            )

    if committed_route is not None:
        assert committed_route is not None and args.route_manifest is not None
        route_source = DynamicRouteSource(
            args.route_manifest,
            stream_id=str(committed_route.manifest["stream_id"]),
            leg_index=args.leg_index,
            expected_writer_id=committed_route.writer_id,
            max_manifest_age_sec=args.max_route_manifest_age_sec,
            max_observation_age_sec=args.max_route_observation_age_sec,
            max_join_distance_m=args.max_route_join_distance_m,
            terminal_route_lock_distance_m=(
                args.dynamic_route_terminal_lock_distance_m
            ),
            # The dynamic planner already emitted a collision-checked,
            # shortcut route. Generic thinning could create an unchecked
            # chord, so authoritative dynamic revisions are never re-thinned.
            thinning_min_spacing_m=0.0,
        )

        def waypoint_provider(pose):
            return route_source.poll(pose)
    if mission_leg_motion_permit is not None:
        try:
            mission_leg_receipt_path = (
                default_mission_leg_motion_consumption_receipt_path(
                    args.mission_leg_motion_permit_json
                )
            )
            mission_leg_receipt = consume_mission_leg_motion_permit(
                permit_path=args.mission_leg_motion_permit_json,
                permit=mission_leg_motion_permit,
                session_id=args.mission_session_id,
                run_id=args.run_id,
                mission_leg_kind=(
                    mission_leg_motion_permit.mission_leg_kind
                ),
                mission_leg_index=(
                    mission_leg_motion_permit.mission_leg_index
                ),
                target_id=mission_leg_motion_permit.target_id,
            )
        except ValueError as exc:
            return _record_motion_authorization_rejection(
                args=args,
                resolved=resolved,
                leg=leg,
                event_logger=event_logger,
                failure=exc,
            )
        print(
            "Using the mission-level RUN for this exact routine child leg. "
            "All live gates passed and its one-use receipt was claimed; no "
            "additional operator input is requested."
        )
        emit_event(
            event_logger,
            "mission_leg_motion_permit_consumed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            mission_leg_kind=(
                mission_leg_motion_permit.mission_leg_kind.value
            ),
            mission_leg_index=(
                mission_leg_motion_permit.mission_leg_index
            ),
            target_id=mission_leg_motion_permit.target_id,
            coverage_leg_index=(
                mission_leg_motion_permit.mission_leg_index
                if mission_leg_motion_permit.mission_leg_kind
                is MissionLegKind.COVERAGE
                else None
            ),
            target_viewpoint_id=(
                mission_leg_motion_permit.target_id
                if mission_leg_motion_permit.mission_leg_kind
                is MissionLegKind.COVERAGE
                else ""
            ),
            mission_leg_motion_authorization_json=str(
                args.mission_leg_motion_authorization_json
            ),
            mission_leg_motion_permit_json=str(
                args.mission_leg_motion_permit_json
            ),
            mission_leg_motion_permit_sha256=(
                mission_leg_motion_permit_sha256(
                    mission_leg_motion_permit
                )
            ),
            mission_leg_motion_consumption_receipt_json=str(
                mission_leg_receipt_path
            ),
            mission_leg_motion_consumption_receipt_sha256=(
                mission_leg_motion_consumption_receipt_sha256(
                    mission_leg_receipt
                )
            ),
            covered_by_initial_mission_run=True,
            additional_typed_run_required=False,
        )
    if startup_reseal_motion_permit is not None:
        try:
            (
                startup_mission_leg_kind,
                startup_mission_leg_index,
                startup_target_id,
            ) = resolve_startup_reseal_permit_identity(
                startup_reseal_motion_permit
            )
            startup_receipt_path = (
                default_startup_reseal_motion_consumption_receipt_path(
                    args.startup_reseal_motion_permit_json
                )
            )
            startup_receipt = consume_startup_reseal_motion_permit(
                permit_path=args.startup_reseal_motion_permit_json,
                permit=startup_reseal_motion_permit,
                session_id=args.mission_session_id,
                run_id=args.run_id,
                leg_index=startup_reseal_motion_permit.leg_index,
                target_viewpoint_id=(
                    startup_reseal_motion_permit.target_viewpoint_id
                ),
                reseal_index=startup_reseal_motion_permit.reseal_index,
                mission_leg_kind=(
                    startup_mission_leg_kind
                ),
                mission_leg_index=(
                    startup_mission_leg_index
                ),
                target_id=startup_target_id,
            )
        except ValueError as exc:
            return _record_motion_authorization_rejection(
                args=args,
                resolved=resolved,
                leg=leg,
                event_logger=event_logger,
                failure=exc,
            )
        print(
            "Using the mission-level RUN for this exact bounded, same-target "
            "startup recovery. All live gates passed and the one-use receipt "
            "was claimed; no additional operator input is requested."
        )
        emit_event(
            event_logger,
            "startup_reseal_motion_permit_consumed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            mission_leg_kind=(
                startup_mission_leg_kind.value
            ),
            mission_leg_index=(
                startup_mission_leg_index
            ),
            target_id=startup_target_id,
            target_viewpoint_id=(
                startup_target_id
                if startup_mission_leg_kind is MissionLegKind.COVERAGE
                else ""
            ),
            coverage_leg_index=(
                startup_mission_leg_index
                if startup_mission_leg_kind is MissionLegKind.COVERAGE
                else None
            ),
            recovery_source_kind=(
                startup_reseal_motion_permit.recovery_source_kind
            ),
            reseal_index=startup_reseal_motion_permit.reseal_index,
            rejected_run_id=startup_reseal_motion_permit.rejected_run_id,
            startup_reseal_motion_authorization_json=str(
                args.startup_reseal_motion_authorization_json
            ),
            startup_reseal_motion_permit_json=str(
                args.startup_reseal_motion_permit_json
            ),
            startup_reseal_motion_permit_sha256=(
                startup_reseal_motion_permit_sha256(
                    startup_reseal_motion_permit
                )
            ),
            startup_reseal_motion_consumption_receipt_json=str(
                startup_receipt_path
            ),
            startup_reseal_motion_consumption_receipt_sha256=(
                startup_reseal_motion_consumption_receipt_sha256(
                    startup_receipt
                )
            ),
            covered_by_initial_mission_run=True,
            additional_typed_run_required=False,
        )
    if runtime_motion_permit is not None:
        try:
            receipt_path = default_runtime_motion_consumption_receipt_path(
                args.runtime_localization_motion_permit_json
            )
            runtime_motion_receipt = consume_runtime_motion_permit(
                permit_path=args.runtime_localization_motion_permit_json,
                permit=runtime_motion_permit,
                session_id=args.mission_session_id,
                run_id=args.run_id,
                leg_index=runtime_motion_permit.leg_index,
                target_viewpoint_id=(
                    runtime_motion_permit.target_viewpoint_id
                ),
                reseal_index=runtime_motion_permit.reseal_index,
            )
        except ValueError as exc:
            return _record_motion_authorization_rejection(
                args=args,
                resolved=resolved,
                leg=leg,
                event_logger=event_logger,
                failure=exc,
            )
        print(
            "Using the mission-level RUN for this exact bounded, same-target "
            "runtime-localization recovery. All live gates passed and the "
            "one-use receipt was claimed; no additional operator input is "
            "requested."
        )
        emit_event(
            event_logger,
            "runtime_localization_motion_permit_consumed",
            run_id=args.run_id,
            leg_index=leg.leg_index,
            target_viewpoint_id=(
                runtime_motion_permit.target_viewpoint_id
            ),
            coverage_leg_index=runtime_motion_permit.leg_index,
            reseal_index=runtime_motion_permit.reseal_index,
            rejected_run_id=runtime_motion_permit.rejected_run_id,
            mission_motion_authorization_json=str(
                args.mission_motion_authorization_json
            ),
            runtime_localization_motion_permit_json=str(
                args.runtime_localization_motion_permit_json
            ),
            runtime_localization_motion_permit_sha256=(
                runtime_localization_motion_permit_sha256(
                    runtime_motion_permit
                )
            ),
            runtime_motion_consumption_receipt_json=str(receipt_path),
            runtime_motion_consumption_receipt_sha256=(
                runtime_motion_consumption_receipt_sha256(
                    runtime_motion_receipt
                )
            ),
            covered_by_initial_mission_run=True,
            additional_typed_run_required=False,
        )
    emit_event(
        event_logger,
        "motion_started",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        **build_mission_leg_event_fields(args),
        # This is an execution-attempt boundary, emitted immediately before
        # entering the follower.  It is not evidence of a nonzero Twist.
        motion_published=False,
        event_semantics="child_execution_attempt_started_before_follower",
        resolved_cmd_vel_topic=resolved.cmd_vel_topic,
    )
    follower_kwargs = {}
    if blockage_recovery_provider is not None:
        follower_kwargs["blockage_recovery_provider"] = (
            blockage_recovery_provider
        )
    if args.controller_trace_jsonl is not None:
        follower_kwargs["controller_trace_path"] = (
            args.controller_trace_jsonl
        )
    if odom_execution_context is not None:
        follower_kwargs["odom_execution_context"] = odom_execution_context
    result = run_simple_waypoint_follower(
        resolved,
        execution_waypoints,
        follower_config,
        waypoint_provider,
        route_update_callback,
        **follower_kwargs,
    )
    _append_result(args, resolved, leg, preflight_ok=True, result=result)
    motion_event_fields = {
        "run_id": args.run_id,
        "leg_index": leg.leg_index,
        # ``leg_index`` above selects a row in this child route artifact and
        # is normally zero.  Keep the autonomous coverage identity explicit
        # so recovery permits never confuse route-local and mission indices.
        **build_mission_leg_event_fields(args),
        "status": result.status,
        "stop_reason": result.stop_reason,
        "duration_sec": result.duration_sec,
        "distance_estimate_m": result.distance_estimate_m,
        "motion_published": result.motion_published,
    }
    if result.status != "completed":
        motion_event_fields["stop_details"] = result.stop_details or {}
    emit_event(
        event_logger,
        "motion_completed" if result.status == "completed" else "safety_stop",
        **motion_event_fields,
    )
    emit_event(
        event_logger,
        "run_finished",
        run_id=args.run_id,
        final_status=result.status,
        stop_reason=result.stop_reason,
        results_csv=str(args.results_csv),
        semantic_log_path=str(args.semantic_log),
        preflight_json_path=str(args.preflight_json or ""),
    )
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
