"""Route identity, binding, and certificate admission before ROS use."""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.navigation.driving_behavior import (
    CATALOG_PHYSICAL_ROUTE_KINDS,
    DYNAMIC_VIEWPOINT_ROUTE_KINDS,
    STATIC_PHYSICAL_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.detected_stand_preapproach import (
    DETECTED_STAND_PREAPPROACH_ROUTE_KIND,
    validate_detected_stand_preapproach_binding,
)
from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    validate_arena_boundary_evidence,
)
from scripts.aufgabe04.navigation.mission_execution_gate import (
    MissionExecutionBinding,
    load_diagnostics_snapshot,
    validate_logistics_execution_bundle,
)
from scripts.aufgabe04.navigation.route_revision_store import RouteRevisionError
from scripts.aufgabe04.navigation.run_events import emit_event
from scripts.aufgabe04.navigation.safety_checks import (
    catalog_start_egress_certificate,
    validate_catalog_route_binding_json,
    validate_route_diagnostics_json,
    validate_speed_limits,
)
from scripts.aufgabe04.navigation.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
    validate_stand_discovery_route_binding,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    load_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.transient_overlay_resume_state import (
    TransientOverlayResumeState,
    transient_overlay_resume_state_sha256,
    validate_transient_overlay_resume_state_diagnostics_binding,
)

from .reporting import _append_status_result
from .route_bundle import (
    _authoritative_route_paths,
    _execution_certificate_failures,
    _load_execution_route_leg,
    _simulation_odom_fallback_admission_failure,
)

LEGACY_SIMULATION_ROUTE_KIND = "legacy_simulation_waypoint"

def _validated_coverage_replan_resume_state(
    args,
    *,
    diagnostics_path: Path,
) -> TransientOverlayResumeState | None:
    """Integrity-load one inherited overlay without authorizing motion."""

    state_path = args.coverage_transient_replan_resume_state_json
    if state_path is None:
        return None
    if not args.coverage_transient_replan_enabled:
        raise ValueError(
            "transient overlay resume state requires coverage replanning"
        )
    if args.coverage_plan is None:
        raise ValueError(
            "transient overlay resume state requires the coverage plan"
        )
    survey_plan_path = (
        Path(args.coverage_transient_replan_survey_root)
        / "coverage_plan.json"
    )
    try:
        supplied_plan_path = Path(args.coverage_plan).resolve(strict=True)
        expected_plan_path = survey_plan_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("coverage resume plan is unavailable") from exc
    if supplied_plan_path != expected_plan_path:
        raise ValueError(
            "coverage resume state and replanner use different plans"
        )
    plan = load_coverage_survey_plan(supplied_plan_path)
    state = validate_transient_overlay_resume_state_diagnostics_binding(
        diagnostics_path,
        resume_state_path=state_path,
        plan=plan,
        expected_coverage_leg_index=(
            args.coverage_transient_replan_leg_index
        ),
        expected_target_viewpoint_id=(
            args.coverage_transient_replan_target_viewpoint_id
        ),
        expected_max_replans=args.coverage_transient_replan_max_count,
    )
    if args.run_id in state.source_run_ids:
        raise ValueError(
            "coverage resume state cannot be replayed by a source child run"
        )
    return state


def admit_execution_route(
    *,
    parser,
    args,
    resolved,
    resolved_runtime_nomotion_update_service: str,
    odom_execution_enabled: bool,
    event_logger,
    require_motion: bool,
):
    try:
        route_csv_path, diagnostics_json_path, committed_route = _authoritative_route_paths(args)
    except (OSError, ValueError, RouteRevisionError) as exc:
        emit_event(
            event_logger,
            "route_manifest_rejected",
            run_id=args.run_id,
            status="failed",
            stop_reason=str(exc),
            route_manifest=str(args.route_manifest or ""),
        )
        parser.exit(2, f"error: authoritative route validation failed: {exc}\n")
    if committed_route is not None and not args.allow_sim_time:
        parser.exit(2, "error: authoritative dynamic route is simulation-only\n")
    if args.dynamic_route_refresh_sec > 0.0 and committed_route is None:
        parser.exit(2, "error: dynamic route refresh requires an authoritative route manifest\n")
    emit_event(
        event_logger,
        "run_started",
        run_id=args.run_id,
        robot_id=args.robot_id,
        route_csv=str(args.route_csv),
        diagnostics_json=str(args.diagnostics_json),
        authoritative_route_csv=str(route_csv_path),
        authoritative_diagnostics_json=str(diagnostics_json_path),
        route_manifest=str(args.route_manifest or ""),
        leg_index=args.leg_index,
        results_csv=str(args.results_csv),
        semantic_log_path=str(args.semantic_log),
        preflight_json_path=str(args.preflight_json or ""),
        controller_trace_jsonl=str(args.controller_trace_jsonl or ""),
    )
    emit_event(
        event_logger,
        "runtime_resolved",
        run_id=args.run_id,
        robot_id=args.robot_id,
        namespace=resolved.namespace,
        resolved_cmd_vel_topic=resolved.cmd_vel_topic,
        resolved_scan_topic=resolved.scan_topic,
        resolved_odom_topic=resolved.odom_topic,
        resolved_amcl_topic=resolved.amcl_topic,
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        localization_source=resolved.localization_source,
        ros_domain_id=resolved.ros_domain_id,
        allow_sim_time=args.allow_sim_time,
        allow_simulation_odom_after_stale_tf_requested=(
            args.allow_simulation_odom_after_stale_tf
        ),
        preflight_nomotion_update_service=args.nomotion_update_service,
        preflight_nomotion_update_timeout_sec=args.nomotion_update_timeout_sec,
        runtime_nomotion_update_service=(
            resolved_runtime_nomotion_update_service
        ),
        runtime_nomotion_update_service_configured=(
            args.runtime_nomotion_update_service
        ),
        runtime_nomotion_update_timeout_sec=(
            args.runtime_nomotion_update_timeout_sec
        ),
        amcl_edge_future_tolerance_sec=(
            args.max_localization_tf_future_sec
        ),
    )
    if committed_route is not None:
        manifest = committed_route.manifest
        emit_event(
            event_logger,
            "authoritative_route_resolved",
            run_id=args.run_id,
            leg_index=args.leg_index,
            route_manifest=str(committed_route.manifest_path),
            manifest_sha256=committed_route.manifest_sha256,
            stream_id=manifest["stream_id"],
            writer_id=committed_route.writer_id,
            writer_generation=committed_route.writer_generation,
            route_revision=committed_route.route_revision,
            target_revision=committed_route.target_revision,
            route_sha256=committed_route.route_hash,
            published_unix_sec=manifest["published_unix_sec"],
            observation_unix_sec=manifest["observation_unix_sec"],
            source_robot_pose=manifest.get("source_robot_pose", {}),
            target=manifest.get("target", {}),
            previous_route_length_m=manifest.get("previous_route_length_m"),
            new_route_length_m=manifest.get("new_route_length_m"),
        )
    try:
        leg = _load_execution_route_leg(
            route_csv_path,
            args.leg_index,
            require_motion=require_motion,
            requested_thinning_min_spacing_m=args.thinning_min_spacing_m,
            authoritative_dynamic_route=committed_route is not None,
        )
    except (OSError, ValueError) as exc:
        emit_event(
            event_logger,
            "route_validation_failed",
            run_id=args.run_id,
            leg_index=args.leg_index,
            status="failed",
            stop_reason=str(exc),
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="route_validation_failed",
            stop_reason=str(exc),
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        parser.exit(2, f"error: route validation failed: {exc}\n")

    try:
        diagnostics_snapshot = load_diagnostics_snapshot(
            diagnostics_json_path,
            require_metadata=leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS,
        )
    except ValueError as exc:
        emit_event(
            event_logger,
            "route_validation_failed",
            run_id=args.run_id,
            leg_index=args.leg_index,
            status="failed",
            stop_reason=str(exc),
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="route_validation_failed",
            stop_reason=str(exc),
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        parser.exit(2, f"error: route diagnostics validation failed: {exc}\n")

    diagnostics_metadata = diagnostics_snapshot.payload.get("metadata")
    route_purpose_value = (
        diagnostics_metadata.get("route_purpose")
        if isinstance(diagnostics_metadata, dict)
        else None
    )
    route_purpose = (
        route_purpose_value
        if isinstance(route_purpose_value, str)
        else ""
    )
    known_route_kinds = DYNAMIC_VIEWPOINT_ROUTE_KINDS | STATIC_PHYSICAL_ROUTE_KINDS
    if leg.route_kind == LEGACY_SIMULATION_ROUTE_KIND:
        if not args.allow_legacy_simulation_route:
            parser.exit(
                2,
                "error: legacy simulation route requires "
                "--allow-legacy-simulation-route\n",
            )
        if not leg.simulation_only or not args.allow_sim_time:
            parser.exit(
                2,
                "error: legacy route escape hatch is simulation-only and requires "
                "simulation_only=true plus --allow-sim-time\n",
            )
        if committed_route is not None:
            parser.exit(2, "error: legacy simulation route cannot use a route manifest\n")
    elif leg.route_kind not in known_route_kinds:
        parser.exit(
            2,
            f"error: missing or unknown Aufgabe04 route kind: {leg.route_kind!r}\n",
        )
    if odom_execution_enabled and leg.route_kind not in STATIC_PHYSICAL_ROUTE_KINDS:
        parser.exit(
            2,
            "error: odom execution currently requires a sealed static physical route\n",
        )
    if leg.route_kind in DYNAMIC_VIEWPOINT_ROUTE_KINDS and not leg.simulation_only:
        parser.exit(2, "error: dynamic viewpoint route is missing simulation_only provenance\n")
    if (
        args.coverage_transient_replan_enabled
        and leg.route_kind != STAND_DISCOVERY_ROUTE_KIND
    ):
        parser.exit(
            2,
            "error: physical transient replanning is restricted to "
            "stand_discovery_corridor\n",
        )
    if args.coverage_transient_replan_enabled and args.allow_sim_time:
        parser.exit(
            2,
            "error: physical transient replanning is not a simulation route mode\n",
        )
    try:
        coverage_replan_resume_state = (
            _validated_coverage_replan_resume_state(
                args,
                diagnostics_path=diagnostics_json_path,
            )
        )
    except (OSError, ValueError) as exc:
        emit_event(
            event_logger,
            "transient_overlay_resume_state_rejected",
            run_id=args.run_id,
            leg_index=args.leg_index,
            status="failed_closed",
            stop_reason=str(exc),
            motion_published=False,
        )
        parser.exit(
            2,
            f"error: transient overlay resume state rejected: {exc}\n",
        )
    if coverage_replan_resume_state is not None:
        emit_event(
            event_logger,
            "transient_overlay_resume_state_validated",
            run_id=args.run_id,
            leg_index=args.leg_index,
            resume_state_json=str(
                args.coverage_transient_replan_resume_state_json
            ),
            resume_state_sha256=(
                transient_overlay_resume_state_sha256(
                    coverage_replan_resume_state
                )
            ),
            completed_replan_count=(
                coverage_replan_resume_state.completed_replan_count
            ),
            max_replans=coverage_replan_resume_state.max_replans,
            remaining_replans=(
                coverage_replan_resume_state.remaining_replans
            ),
            motion_continues_authorized=False,
            automatic_motion_authorized=False,
        )
    if leg.route_kind in DYNAMIC_VIEWPOINT_ROUTE_KINDS and committed_route is None:
        parser.exit(2, "error: dynamic viewpoint route requires its authoritative manifest\n")
    if leg.simulation_only and not args.allow_sim_time:
        parser.exit(
            2,
            "error: simulation-only synchronized-viewpoint routes require --allow-sim-time\n",
        )
    if committed_route is not None and leg.route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        parser.exit(
            2,
            f"error: authoritative route has unknown dynamic route kind: {leg.route_kind!r}\n",
        )
    simulation_odom_fallback_admission_failure = (
        _simulation_odom_fallback_admission_failure(
            args,
            resolved,
            leg,
            route_purpose=route_purpose,
            authoritative_dynamic_route=committed_route is not None,
        )
    )
    if simulation_odom_fallback_admission_failure:
        parser.exit(
            2,
            "error: "
            + simulation_odom_fallback_admission_failure
            + "\n",
        )
    allow_simulation_odom_after_stale_tf = bool(
        args.allow_simulation_odom_after_stale_tf
    )

    diagnostics_status = validate_route_diagnostics_json(
        diagnostics_json_path,
        args.leg_index,
        csv_point_count=len(leg.raw_waypoints),
        require_motion=require_motion,
        diagnostics_payload=diagnostics_snapshot.payload,
    )
    catalog_binding_status = (
        validate_catalog_route_binding_json(
            diagnostics_json_path,
            leg,
            catalog_path_override=args.arrival_pose_catalog,
            diagnostics_payload=diagnostics_snapshot.payload,
        )
        if leg.route_kind in CATALOG_PHYSICAL_ROUTE_KINDS
        else None
    )
    detected_stand_binding_status = (
        validate_detected_stand_preapproach_binding(
            diagnostics_json_path,
            leg,
            candidate_snapshot_path=args.candidate_snapshot,
            diagnostics_payload=diagnostics_snapshot.payload,
        )
        if leg.route_kind == DETECTED_STAND_PREAPPROACH_ROUTE_KIND
        else None
    )
    stand_discovery_binding_status = (
        validate_stand_discovery_route_binding(
            diagnostics_json_path,
            leg,
            coverage_plan_path=args.coverage_plan,
            diagnostics_payload=diagnostics_snapshot.payload,
        )
        if leg.route_kind == STAND_DISCOVERY_ROUTE_KIND
        else None
    )
    catalog_egress_certificate = None
    catalog_egress_failures = []
    execution_certificate_failures = []
    mission_execution_failures = []
    mission_execution_binding: MissionExecutionBinding | None = None
    if leg.route_kind in CATALOG_PHYSICAL_ROUTE_KINDS:
        try:
            catalog_egress_certificate = catalog_start_egress_certificate(
                diagnostics_json_path,
                leg,
                diagnostics_payload=diagnostics_snapshot.payload,
            )
        except ValueError as exc:
            catalog_egress_failures.append(
                f"catalog start-egress certificate is invalid: {exc}"
            )
    if leg.route_kind in STATIC_PHYSICAL_ROUTE_KINDS:
        execution_certificate_failures = _execution_certificate_failures(
            route_leg=leg,
            diagnostics_snapshot=diagnostics_snapshot,
            explicit_certificate_path=args.route_certificate_json,
            route_kind=leg.route_kind,
            runtime_namespace=resolved.namespace,
            runtime_planning_frame=resolved.map_frame,
            tracking_tube_radius_m=args.certified_route_tube_radius_m,
        )
        try:
            validate_arena_boundary_evidence(diagnostics_snapshot.metadata)
            if leg.route_kind == DETECTED_STAND_PREAPPROACH_ROUTE_KIND:
                if route_purpose != "pre_approach":
                    raise ValueError(
                        "detected stand route requires route_purpose=pre_approach"
                    )
                if args.candidate_snapshot is None:
                    raise ValueError(
                        "detected stand pre-approach requires --candidate-snapshot"
                    )
            elif leg.route_kind == STAND_DISCOVERY_ROUTE_KIND:
                if route_purpose != "stand_discovery":
                    raise ValueError(
                        "stand discovery route requires "
                        "route_purpose=stand_discovery"
                    )
                if args.coverage_plan is None:
                    raise ValueError(
                        "stand discovery route requires --coverage-plan"
                    )
            elif route_purpose == "logistics":
                missing = [
                    option
                    for option, value in (
                        ("--mission-plan-manifest", args.mission_plan_manifest),
                        ("--survey-manifest", args.survey_manifest),
                        ("--route-bundle-json", args.route_bundle_json),
                        ("--planner-config-json", args.planner_config_json),
                        ("--runtime-map-bundle-json", args.runtime_map_bundle_json),
                        ("--runtime-environment", args.runtime_environment),
                        ("--candidate-snapshot", args.candidate_snapshot),
                        (
                            "--station-identity-registry",
                            args.station_identity_registry,
                        ),
                        ("--arrival-pose-catalog", args.arrival_pose_catalog),
                        ("--task-snapshot", args.task_snapshot),
                    )
                    if value is None
                ]
                if missing:
                    raise ValueError(
                        "logistics execution requires " + ", ".join(missing)
                    )
                mission_execution_binding = validate_logistics_execution_bundle(
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
                    station_identity_registry_path=(
                        args.station_identity_registry
                    ),
                    arrival_pose_catalog_path=args.arrival_pose_catalog,
                    task_snapshot_path=args.task_snapshot,
                    robot_id=args.robot_id,
                    runtime_planning_frame=resolved.map_frame,
                    diagnostics_snapshot=diagnostics_snapshot,
                )
            elif route_purpose == "survey":
                if not (
                    args.allow_unbound_survey_simulation_route
                    and args.allow_sim_time
                    and leg.simulation_only
                ):
                    raise ValueError(
                        "static survey route is unbound to a task mission; a "
                        "simulation demonstration requires "
                        "--allow-unbound-survey-simulation-route, "
                        "--allow-sim-time, and simulation_only=true"
                    )
            else:
                raise ValueError(
                    f"static route has missing or unknown route_purpose: {route_purpose!r}"
                )
        except (OSError, ValueError) as exc:
            mission_execution_failures.append(
                f"mission execution binding is invalid: {exc}"
            )
    speed_status = validate_speed_limits(args.max_linear_mps, args.max_angular_radps)
    pure_failures = (
        diagnostics_status.failures
        + ([] if catalog_binding_status is None else catalog_binding_status.failures)
        + (
            []
            if detected_stand_binding_status is None
            else detected_stand_binding_status.failures
        )
        + (
            []
            if stand_discovery_binding_status is None
            else stand_discovery_binding_status.failures
        )
        + catalog_egress_failures
        + execution_certificate_failures
        + mission_execution_failures
        + speed_status.failures
    )
    if pure_failures:
        stop_reason = "; ".join(pure_failures)
        emit_event(
            event_logger,
            "route_validation_failed",
            run_id=args.run_id,
            leg_index=args.leg_index,
            status="failed",
            failures=pure_failures,
        )
        _append_status_result(
            args,
            resolved,
            leg,
            preflight_ok=False,
            status="route_validation_failed",
            stop_reason=stop_reason,
        )
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="route_validation_failed",
            stop_reason=stop_reason,
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
        parser.exit(2, "error: validation failed:\n" + "\n".join(f"- {failure}" for failure in pure_failures) + "\n")

    emit_event(
        event_logger,
        "route_validated",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        raw_point_count=len(leg.raw_waypoints),
        executable_point_count=len(leg.executable_waypoints),
        route_length_m=leg.route_length_m,
        require_motion=require_motion,
        allow_noop=args.allow_noop,
    )
    return (
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
    )
