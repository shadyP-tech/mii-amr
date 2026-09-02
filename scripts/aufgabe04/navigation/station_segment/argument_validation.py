"""Fail-closed CLI normalization before route or ROS observation."""

from __future__ import annotations

import math
import os
from pathlib import Path
import uuid

from scripts.aufgabe04.navigation.execution.mission_leg_identity_args import (
    resolve_mission_leg_event_identity,
)
from scripts.aufgabe04.navigation.foundation.run_events import configure_event_logger
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
)

DEFAULT_EVENT_LOG_DIR = Path("results/aufgabe04/run_events")


def prepare_runtime_arguments(parser, args):
    recovery_fields = (
        args.coverage_transient_replan_survey_root,
        args.coverage_transient_replan_session_root,
        args.coverage_transient_replan_map,
        args.coverage_transient_replan_robot_radius_m,
        args.coverage_transient_replan_leg_index,
    )
    args.coverage_transient_replan_enabled = any(
        value is not None for value in recovery_fields
    ) or bool(
        args.coverage_transient_replan_semantic_map_id
        or args.coverage_transient_replan_target_viewpoint_id
        or args.coverage_transient_replan_max_count
        or args.coverage_transient_replan_resume_state_json is not None
    )
    if args.coverage_transient_replan_enabled and (
        any(value is None for value in recovery_fields)
        or not args.coverage_transient_replan_semantic_map_id
        or not args.coverage_transient_replan_target_viewpoint_id
        or args.coverage_transient_replan_max_count <= 0
    ):
        parser.error(
            "physical coverage transient replanning requires survey root, "
            "session root, map, semantic map ID, target viewpoint ID, positive "
            "robot radius, and positive max count"
        )
    try:
        resolve_mission_leg_event_identity(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.localization_branch_proof_id = str(
        args.localization_branch_proof_id
    ).strip()
    odom_execution_enabled = args.execution_pose_frame == "odom"
    if odom_execution_enabled:
        missing = [
            flag
            for flag, value in (
                (
                    "--odom-execution-certificate-json",
                    args.odom_execution_certificate_json,
                ),
                ("--uncertainty-budget-json", args.uncertainty_budget_json),
                ("--uncertainty-map-yaml", args.uncertainty_map_yaml),
                (
                    "--localization-branch-proof-id",
                    args.localization_branch_proof_id,
                ),
                (
                    "--uncertainty-robot-radius-m",
                    args.uncertainty_robot_radius_m,
                ),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(
                "odom execution requires " + ", ".join(missing)
            )
        if args.localization_source != "amcl":
            parser.error(
                "odom execution requires AMCL as the global consistency monitor"
            )
        if args.map_frame == args.odom_frame:
            parser.error("odom execution requires distinct map and odom frames")
        if args.dynamic_route_refresh_sec > 0.0:
            parser.error(
                "odom execution does not admit simulation route hot-reload"
            )
        if args.allow_simulation_odom_after_stale_tf:
            parser.error(
                "odom execution may not enable the simulation stale-TF fallback"
            )
        uncertainty_values = (
            (
                "--uncertainty-robot-radius-m",
                args.uncertainty_robot_radius_m,
                True,
            ),
            (
                "--uncertainty-collision-margin-m",
                args.uncertainty_collision_margin_m,
                False,
            ),
            (
                "--uncertainty-odom-drift-bound-m",
                args.uncertainty_odom_drift_bound_m,
                False,
            ),
            (
                "--uncertainty-braking-latency-distance-m",
                args.uncertainty_braking_latency_distance_m,
                False,
            ),
            (
                "--uncertainty-sigma-multiplier",
                args.uncertainty_sigma_multiplier,
                True,
            ),
            (
                "--uncertainty-clearance-sample-spacing-m",
                args.uncertainty_clearance_sample_spacing_m,
                True,
            ),
            (
                "--max-map-odom-yaw-drift-rad",
                args.max_map_odom_yaw_drift_rad,
                True,
            ),
            (
                "--max-map-odom-translation-drift-m",
                args.max_map_odom_translation_drift_m,
                True,
            ),
        )
        for flag, value, strictly_positive in uncertainty_values:
            if (
                value is None
                or not math.isfinite(value)
                or (value <= 0.0 if strictly_positive else value < 0.0)
            ):
                qualifier = "positive" if strictly_positive else "non-negative"
                parser.error(f"{flag} must be finite and {qualifier}")
        if args.uncertainty_heading_lever_arm_m is not None and (
            not math.isfinite(args.uncertainty_heading_lever_arm_m)
            or args.uncertainty_heading_lever_arm_m <= 0.0
        ):
            parser.error(
                "--uncertainty-heading-lever-arm-m must be finite and positive"
            )
    if args.dynamic_route_refresh_sec < 0.0:
        parser.error("--dynamic-route-refresh-sec must be non-negative")
    if args.dynamic_route_refresh_sec > 0.0 and not args.allow_sim_time:
        parser.error("dynamic route hot-reload is simulation-only and requires --allow-sim-time")
    if args.max_route_manifest_age_sec <= 0.0 or args.max_route_observation_age_sec <= 0.0:
        parser.error("dynamic route freshness limits must be positive")
    if args.max_route_join_distance_m <= 0.0:
        parser.error("--max-route-join-distance-m must be positive")
    if (
        not math.isfinite(args.terminal_heading_timeout_sec)
        or args.terminal_heading_timeout_sec <= 0.0
    ):
        parser.error("--terminal-heading-timeout-sec must be positive")
    if (
        not math.isfinite(args.axis_acquisition_wait_timeout_sec)
        or args.axis_acquisition_wait_timeout_sec <= 0.0
    ):
        parser.error("--axis-acquisition-wait-timeout-sec must be positive")
    if (
        not math.isfinite(args.viewpoint_sampling_timeout_sec)
        or args.viewpoint_sampling_timeout_sec <= 0.0
    ):
        parser.error("--viewpoint-sampling-timeout-sec must be positive")
    if (
        not math.isfinite(args.viewpoint_sampling_target_timeout_sec)
        or args.viewpoint_sampling_target_timeout_sec <= 0.0
    ):
        parser.error("--viewpoint-sampling-target-timeout-sec must be positive")
    if (
        not math.isfinite(args.physical_waypoint_tolerance_m)
        or args.physical_waypoint_tolerance_m <= 0.0
    ):
        parser.error("--physical-waypoint-tolerance-m must be positive")
    if (
        not math.isfinite(args.physical_goal_tolerance_m)
        or args.physical_goal_tolerance_m <= 0.0
    ):
        parser.error("--physical-goal-tolerance-m must be positive")
    if (
        not math.isfinite(args.max_future_timestamp_sec)
        or args.max_future_timestamp_sec < 0.0
    ):
        parser.error("--max-future-timestamp-sec must be non-negative")
    if (
        not math.isfinite(args.max_localization_tf_future_sec)
        or args.max_localization_tf_future_sec < 0.0
    ):
        parser.error("--max-localization-tf-future-sec must be non-negative")
    if (
        not math.isfinite(args.nomotion_update_timeout_sec)
        or args.nomotion_update_timeout_sec <= 0.0
    ):
        parser.error("--nomotion-update-timeout-sec must be positive")
    args.runtime_nomotion_update_service = str(
        args.runtime_nomotion_update_service
    ).strip()
    if not args.runtime_nomotion_update_service:
        parser.error("--runtime-nomotion-update-service must not be empty")
    if (
        not math.isfinite(args.runtime_nomotion_update_timeout_sec)
        or args.runtime_nomotion_update_timeout_sec <= 0.0
        or args.runtime_nomotion_update_timeout_sec > 2.0
    ):
        parser.error(
            "--runtime-nomotion-update-timeout-sec must be finite and in (0, 2.0]"
        )
    if args.stationary_amcl_sample_count < 2:
        parser.error("--stationary-amcl-sample-count must be at least 2")
    if (
        args.skip_nomotion_update_before_preflight
        and not args.allow_sim_time
        and args.localization_source == "amcl"
    ):
        parser.error(
            "real AMCL runs may not skip the stationary localization gate"
        )
    for flag, value in (
        (
            "--stationary-amcl-sample-interval-sec",
            args.stationary_amcl_sample_interval_sec,
        ),
        (
            "--max-stationary-amcl-position-spread-m",
            args.max_stationary_amcl_position_spread_m,
        ),
        (
            "--max-stationary-amcl-yaw-spread-rad",
            args.max_stationary_amcl_yaw_spread_rad,
        ),
        (
            "--max-stationary-amcl-position-std-m",
            args.max_stationary_amcl_position_std_m,
        ),
        (
            "--max-stationary-amcl-yaw-std-rad",
            args.max_stationary_amcl_yaw_std_rad,
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{flag} must be positive")
    if (
        not math.isfinite(args.certified_route_tube_radius_m)
        or args.certified_route_tube_radius_m <= 0.0
    ):
        parser.error("--certified-route-tube-radius-m must be positive")
    localization_position_limit_m = 0.5 * args.certified_route_tube_radius_m
    localization_tube_limits = [
        (
            "--max-stationary-amcl-position-spread-m",
            args.max_stationary_amcl_position_spread_m,
        )
    ]
    if not odom_execution_enabled:
        localization_tube_limits.append(
            (
                "--max-stationary-amcl-position-std-m",
                args.max_stationary_amcl_position_std_m,
            )
        )
    for flag, value in localization_tube_limits:
        if value > localization_position_limit_m:
            parser.error(
                f"{flag} must not exceed half the certified route tube "
                f"({localization_position_limit_m:.6f} m)"
            )
    if args.physical_goal_tolerance_m > args.certified_route_tube_radius_m:
        parser.error(
            "--physical-goal-tolerance-m must not exceed "
            "--certified-route-tube-radius-m"
        )
    if (
        args.physical_waypoint_tolerance_m
        > args.certified_route_tube_radius_m
    ):
        parser.error(
            "--physical-waypoint-tolerance-m must not exceed "
            "--certified-route-tube-radius-m"
        )
    if (
        not math.isfinite(args.certified_route_chord_sample_spacing_m)
        or args.certified_route_chord_sample_spacing_m <= 0.0
    ):
        parser.error("--certified-route-chord-sample-spacing-m must be positive")
    if args.certified_corner_max_reacquire_attempts < 0:
        parser.error(
            "--certified-corner-max-reacquire-attempts must be non-negative"
        )
    if (
        not math.isfinite(args.viewpoint_sampling_goal_tolerance_m)
        or args.viewpoint_sampling_goal_tolerance_m <= 0.0
    ):
        parser.error("--viewpoint-sampling-goal-tolerance-m must be positive")
    if (
        not math.isfinite(
            args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        )
        or args.viewpoint_sampling_terminal_heading_hold_tolerance_m <= 0.0
        or args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        > INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
        or args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        < min(
            args.viewpoint_sampling_goal_tolerance_m,
            INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
        )
    ):
        parser.error(
            "--viewpoint-sampling-terminal-heading-hold-tolerance-m must "
            "be no smaller than the effective entry tolerance and no "
            "greater than 0.020"
        )
    if (
        not math.isfinite(args.viewpoint_sampling_heading_tolerance_rad)
        or args.viewpoint_sampling_heading_tolerance_rad <= 0.0
    ):
        parser.error("--viewpoint-sampling-heading-tolerance-rad must be positive")
    if (
        not math.isfinite(args.viewpoint_sampling_target_distance_m)
        or args.viewpoint_sampling_target_distance_m
        <= args.viewpoint_sampling_terminal_heading_hold_tolerance_m
    ):
        parser.error(
            "--viewpoint-sampling-target-distance-m must be finite and "
            "greater than the radial hold tolerance"
        )
    if (
        not math.isfinite(
            args.viewpoint_sampling_terminal_heading_target_envelope_radius_m
        )
        or args.viewpoint_sampling_terminal_heading_target_envelope_radius_m
        < args.viewpoint_sampling_terminal_heading_hold_tolerance_m
        or args.viewpoint_sampling_terminal_heading_target_envelope_radius_m
        > INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    ):
        parser.error(
            "--viewpoint-sampling-terminal-heading-target-envelope-radius-m "
            "must be no smaller than the radial hold tolerance and no "
            "greater than 0.030"
        )
    if (
        not math.isfinite(args.dynamic_route_join_tolerance_m)
        or args.dynamic_route_join_tolerance_m <= 0.0
    ):
        parser.error("--dynamic-route-join-tolerance-m must be positive")
    if (
        not math.isfinite(args.start_egress_waypoint_tolerance_m)
        or args.start_egress_waypoint_tolerance_m <= 0.0
    ):
        parser.error("--start-egress-waypoint-tolerance-m must be positive")
    if (
        not math.isfinite(args.start_egress_alignment_tolerance_rad)
        or args.start_egress_alignment_tolerance_rad <= 0.0
        or args.start_egress_alignment_tolerance_rad > math.pi / 2.0
    ):
        parser.error(
            "--start-egress-alignment-tolerance-rad must be in (0, pi/2]"
        )
    if (
        not math.isfinite(args.start_egress_max_linear_mps)
        or args.start_egress_max_linear_mps <= 0.0
    ):
        parser.error("--start-egress-max-linear-mps must be positive")
    if (
        not math.isfinite(args.stuck_heading_progress_epsilon_rad)
        or args.stuck_heading_progress_epsilon_rad <= 0.0
    ):
        parser.error("--stuck-heading-progress-epsilon-rad must be positive")
    if (
        not math.isfinite(args.linear_motion_floor_mps)
        or args.linear_motion_floor_mps <= 0.0
        or args.linear_motion_floor_mps > args.max_linear_mps
    ):
        parser.error(
            "--linear-motion-floor-mps must be positive and no greater than "
            "--max-linear-mps"
        )
    smoothing_values = {
        "--max-linear-accel-mps2": args.max_linear_accel_mps2,
        "--max-angular-accel-radps2": args.max_angular_accel_radps2,
    }
    for name, value in smoothing_values.items():
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if (
        not args.disable_command_smoothing
        and args.max_linear_accel_mps2 / 10.0 + 1.0e-12
        < args.linear_motion_floor_mps
    ):
        parser.error(
            "--max-linear-accel-mps2 must reach "
            "--linear-motion-floor-mps within one 10 Hz control period"
        )
    if (
        not math.isfinite(args.blockage_confirmation_timeout_sec)
        or args.blockage_confirmation_timeout_sec < 0.5
    ):
        parser.error(
            "--blockage-confirmation-timeout-sec must be finite and at least 0.5"
        )
    if not 3 <= args.blockage_confirmation_min_samples <= 7:
        parser.error(
            "--blockage-confirmation-min-samples must be between 3 and 7"
        )
    if (
        not math.isfinite(args.dynamic_route_terminal_lock_distance_m)
        or args.dynamic_route_terminal_lock_distance_m <= 0.0
    ):
        parser.error("--dynamic-route-terminal-lock-distance-m must be positive")
    args.run_id = args.run_id or f"aufgabe04-segment-{uuid.uuid4().hex[:8]}"
    args.semantic_log = args.semantic_log or DEFAULT_EVENT_LOG_DIR / f"{args.run_id}.jsonl"
    args.preflight_json = args.preflight_json or args.semantic_log.with_name(
        f"{args.run_id}_preflight.json"
    )
    bundle_dir = os.environ.get("MII_AMR_RUN_BUNDLE_DIR", "").strip()
    if (
        args.controller_trace_jsonl is None
        and not args.dry_run
        and bundle_dir
    ):
        args.controller_trace_jsonl = Path(bundle_dir) / "controller_trace.jsonl"
    if (
        args.controller_trace_jsonl is not None
        and not args.dry_run
        and args.controller_trace_jsonl.exists()
    ):
        parser.error(
            "refusing to append controller evidence to an existing trace: "
            f"{args.controller_trace_jsonl}"
        )
    event_logger = configure_event_logger(args.semantic_log)
    return odom_execution_enabled, event_logger
