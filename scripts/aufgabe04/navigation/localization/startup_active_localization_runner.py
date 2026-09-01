"""Run one separately authorized, bounded startup-localization rotation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import (
    RuntimeConfig,
    resolve_runtime_config,
    resolve_topic,
)
from scripts.aufgabe04.navigation.foundation.run_events import (
    configure_event_logger,
    emit_event,
)
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ANGULAR_SPEED_RADPS,
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_ATTEMPTS,
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ROTATION_RAD,
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_TIMEOUT_SEC,
    STARTUP_ACTIVE_LOCALIZATION_AUTHORIZATION_HASH_FIELD,
    STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
    STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD,
    StartupActiveLocalizationConfig,
    startup_active_localization_result_payload,
    startup_active_localization_signed_turn,
    stored_content_hash,
    write_startup_active_localization_result,
)
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
    STARTUP_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import (
    run_ros_preflight,
)
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    run_startup_active_localization_motion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_ATTEMPTS,
    )
    parser.add_argument(
        "--rotation-rad",
        type=float,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ROTATION_RAD,
    )
    parser.add_argument(
        "--angular-speed-radps",
        type=float,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ANGULAR_SPEED_RADPS,
    )
    parser.add_argument(
        "--maximum-angular-speed-radps",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_TIMEOUT_SEC,
    )
    parser.add_argument("--source-route-selection-json", type=Path, required=True)
    parser.add_argument("--source-route-selection-sha256", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--controller-trace-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-log", type=Path, required=True)
    return parser


def _physical_checklist(resolved, args, config, signed_turn: float) -> None:
    print("\nStartup active localization will rotate the robot in place.")
    print("Safety requirements:")
    print("  - clear the full robot footprint and 0.20 m LiDAR envelope")
    print("  - keep an operator beside the robot")
    print("  - keep Ctrl+C and the physical stop ready")
    print(f"  - keep a zero Twist terminal ready for {resolved.cmd_vel_topic}")
    print("  - verify no Nav2 controller or other follower owns velocity")
    print("  - this phase does not authorize any mission route")
    print(f"Rejected route evidence: {args.source_route_selection_json}")
    print(
        "Bounded rotation: "
        f"{signed_turn:.3f} rad at {config.angular_speed_radps:.3f} rad/s"
    )


def _confirm_localize(resolved, args, config, signed_turn: float) -> None:
    _physical_checklist(resolved, args, config, signed_turn)
    response = input(
        "Type LOCALIZE to authorize only this bounded in-place rotation: "
    ).strip()
    if response != STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION:
        raise RuntimeError("operator did not authorize startup active localization")


def _validate_source_selection(args) -> None:
    payload = load_content_hashed_json(
        args.source_route_selection_json,
        hash_field=STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
    )
    actual = stored_content_hash(
        args.source_route_selection_json,
        hash_field=STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
    )
    if actual != args.source_route_selection_sha256:
        raise ValueError(
            "source startup route-selection evidence hash does not match"
        )
    if payload.get("schema_version") != (
        STARTUP_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION
    ) or payload.get("phase") != (
        "precheckpoint_initial_coverage_route_selection"
    ):
        raise ValueError("source route-selection evidence contract is invalid")
    selection = payload.get("selection")
    decision = (
        selection.get("decision")
        if isinstance(selection, dict)
        else None
    )
    if not isinstance(decision, dict) or decision.get("ready") is not False:
        raise ValueError(
            "source route-selection evidence is not a rejected decision"
        )
    if decision.get("selected_option_id") is not None:
        raise ValueError(
            "rejected route-selection evidence contains a selected option"
        )
    if payload.get("motion_authorized") is not False or payload.get(
        "motion_published"
    ) is not False:
        raise ValueError(
            "source route-selection evidence is not a no-motion rejection"
        )
    if payload.get("target_committed_before_selection") is not False or (
        payload.get("retargeting_allowed_after_selection") is not False
    ):
        raise ValueError(
            "source route-selection rejection has an invalid target boundary"
        )


def _require_new_attempt_outputs(args) -> None:
    """Reject replay or continuation of any partially written motion attempt."""

    output_paths = (
        args.result_json,
        args.controller_trace_jsonl,
        args.semantic_log,
        args.result_json.parent
        / "startup_active_localization_preflight.json",
        args.result_json.parent
        / "startup_active_localization_authorization.json",
    )
    existing = [
        path
        for path in output_paths
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise ValueError(
            "refusing to reuse startup active-localization artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def _run_preflight(resolved, args):
    preflight = run_ros_preflight(
        resolved,
        max_scan_age_sec=0.5,
        max_odom_age_sec=0.5,
        max_tf_age_sec=1.0,
        max_amcl_age_sec=2.0,
        max_localization_tf_future_sec=1.1,
        observation_window_sec=2.0,
        require_real_time=True,
        request_nomotion_update=True,
        nomotion_update_service=resolve_topic(
            "request_nomotion_update",
            resolved.namespace,
        ),
        nomotion_update_timeout_sec=15.0,
        max_stationary_amcl_position_spread_m=0.015,
        max_stationary_amcl_yaw_spread_rad=0.03,
        max_stationary_amcl_position_std_m=0.30,
        max_stationary_amcl_yaw_std_rad=0.35,
    )
    preflight_path = (
        args.result_json.parent / "startup_active_localization_preflight.json"
    )
    preflight_sha256 = write_content_hashed_json(
        preflight_path,
        preflight.to_json_dict(),
        hash_field=STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD,
    )
    if not preflight.ok:
        raise RuntimeError(
            "startup active-localization ROS preflight failed: "
            + "; ".join(preflight.failures)
        )
    return preflight_path, preflight_sha256


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = StartupActiveLocalizationConfig(
            enabled=True,
            max_attempts=args.max_attempts,
            rotation_rad=args.rotation_rad,
            angular_speed_radps=args.angular_speed_radps,
            timeout_sec=args.timeout_sec,
        )
        signed_turn = startup_active_localization_signed_turn(
            config,
            attempt_index=args.attempt_index,
        )
        if (
            not math.isfinite(args.maximum_angular_speed_radps)
            or args.maximum_angular_speed_radps <= 0.0
            or config.angular_speed_radps
            > args.maximum_angular_speed_radps + 1.0e-12
        ):
            raise ValueError(
                "active-localization angular speed exceeds robot profile maximum"
            )
        _validate_source_selection(args)
        _require_new_attempt_outputs(args)
    except ValueError as exc:
        parser.error(str(exc))

    resolved = resolve_runtime_config(
        RuntimeConfig(
            namespace=args.namespace,
            scan_topic=args.scan_topic,
            odom_topic=args.odom_topic,
            cmd_vel_topic=args.cmd_vel_topic,
            amcl_topic=args.amcl_topic,
            map_frame=args.map_frame,
            odom_frame=args.odom_frame,
            base_frame=args.base_frame,
            localization_source="amcl",
            use_sim_time=False,
        )
    )
    event_logger = configure_event_logger(args.semantic_log)
    try:
        preflight_path, preflight_sha256 = _run_preflight(resolved, args)
        emit_event(
            event_logger,
            "startup_active_localization_preflight_passed",
            run_id=args.run_id,
            attempt_index=args.attempt_index,
            preflight_json=str(preflight_path),
            preflight_sha256=preflight_sha256,
            route_authorized=False,
            mission_run_authorized=False,
        )
        _confirm_localize(resolved, args, config, signed_turn)
        authorization_path = (
            args.result_json.parent
            / "startup_active_localization_authorization.json"
        )
        authorization_payload = {
            "schema_version": 1,
            "phase": "startup_active_localization",
            "run_id": args.run_id,
            "attempt_index": args.attempt_index,
            "operator_confirmation": STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
            "scope": "one bounded in-place startup localization rotation",
            "config": config.to_evidence_dict(),
            "runtime_config": resolved.as_log_dict(),
            "source_route_selection_json": str(
                args.source_route_selection_json
            ),
            "source_route_selection_sha256": (
                args.source_route_selection_sha256
            ),
            "preflight_json": str(preflight_path),
            "preflight_sha256": preflight_sha256,
            "route_authorized": False,
            "mission_run_authorized": False,
        }
        authorization_sha256 = write_content_hashed_json(
            authorization_path,
            authorization_payload,
            hash_field=STARTUP_ACTIVE_LOCALIZATION_AUTHORIZATION_HASH_FIELD,
        )
        emit_event(
            event_logger,
            "startup_active_localization_authorized",
            run_id=args.run_id,
            attempt_index=args.attempt_index,
            authorization_json=str(authorization_path),
            authorization_sha256=authorization_sha256,
            signed_turn_radians=signed_turn,
            route_authorized=False,
            mission_run_authorized=False,
        )
        result = run_startup_active_localization_motion(
            resolved,
            FollowerConfig(
                controller=ControllerConfig(
                    max_linear_mps=0.055,
                    max_angular_radps=args.maximum_angular_speed_radps,
                ),
                min_obstacle_distance_m=config.minimum_clearance_m,
                omnidirectional_hard_stop_distance_m=0.12,
                max_scan_age_sec=0.5,
                max_odom_age_sec=0.5,
                initial_sensor_wait_sec=5.0,
                control_rate_hz=config.control_rate_hz,
            ),
            config,
            attempt_index=args.attempt_index,
            controller_trace_path=args.controller_trace_jsonl,
        )
        payload = startup_active_localization_result_payload(
            run_id=args.run_id,
            attempt_index=args.attempt_index,
            result=result,
            config=config,
            runtime_config=resolved.as_log_dict(),
            source_route_selection_json=args.source_route_selection_json,
            source_route_selection_sha256=(
                args.source_route_selection_sha256
            ),
            preflight_json=preflight_path,
            preflight_sha256=preflight_sha256,
            controller_trace_jsonl=args.controller_trace_jsonl,
        )
        result_sha256 = write_startup_active_localization_result(
            args.result_json,
            payload,
        )
        emit_event(
            event_logger,
            "startup_active_localization_finished",
            run_id=args.run_id,
            attempt_index=args.attempt_index,
            status=result.status,
            stop_reason=result.stop_reason,
            motion_published=result.motion_published,
            result_json=str(args.result_json),
            result_sha256=result_sha256,
            route_authorized=False,
            mission_run_authorized=False,
        )
        return 0 if result.completed else 1
    except Exception as exc:
        emit_event(
            event_logger,
            "startup_active_localization_finished",
            run_id=args.run_id,
            attempt_index=args.attempt_index,
            status="stopped",
            stop_reason=str(exc),
            motion_published="unknown_after_exception",
            fail_closed=True,
            route_authorized=False,
            mission_run_authorized=False,
        )
        print(f"ERROR: startup active localization failed closed: {exc}")
        return 1


__all__ = ["build_parser", "main"]
