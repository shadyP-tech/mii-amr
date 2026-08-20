"""Pure child-process command and semantic-outcome contracts.

This module deliberately does not launch a process or authorize motion.  It
only constructs argv lists and interprets the append-only semantic log emitted
by ``run_single_station_segment.py``.  Keeping this boundary ROS-free makes it
possible to validate the autonomous parent/child contract offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind,
)
from scripts.aufgabe04.navigation.transient_overlay_resume_state import (
    load_jsonl_event_objects,
)


# These values are the shared autonomous parent/child argv contract.  Exporting
# them keeps route admission and child construction on one source of truth.
DEFAULT_TRACKING_TUBE_RADIUS_M = 0.03
DEFAULT_COLLISION_MARGIN_M = 0.02
DEFAULT_LIDAR_STOP_DISTANCE_M = 0.20
# Charge a two-sigma AMCL envelope to static-route clearance.  The child still
# clamps continuity at its hard translation/yaw caps, and route admission
# fails closed whenever the map does not have enough clearance for this
# larger, explicitly persisted allowance.
DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER = 2.0


@dataclass(frozen=True)
class MotionLegOutcome:
    """One unambiguous terminal child-run outcome."""

    run_id: str
    status: str
    stop_reason: str
    stop_details: dict[str, object]
    motion_published: bool
    returncode: int
    semantic_log_path: Path
    semantic_log_start_offset: int = 0
    dry_preflight_path: Path | None = None
    odom_execution_certificate_path: Path | None = None
    dry_uncertainty_budget_path: Path | None = None
    motion_authorization_permit_path: Path | None = None
    motion_authorization_permit_sha256: str = ""
    mission_leg_motion_permit_path: Path | None = None
    mission_leg_motion_permit_sha256: str = ""
    startup_reseal_motion_permit_path: Path | None = None
    startup_reseal_motion_permit_sha256: str = ""


def _resolve_route_artifact_leg_index(
    *,
    route_artifact_leg_index: int | None,
    legacy_leg_index: int | None,
) -> int:
    """Resolve only the leg identity encoded inside the sealed route CSV.

    Autonomous coverage and candidate routes are sealed independently, so
    their local route-artifact index is normally zero even when the enclosing
    mission or coverage-replan index is nonzero.  ``legacy_leg_index`` remains
    a compatibility alias for callers that explicitly selected a route row;
    it must never be inferred from either enclosing identity.
    """

    if route_artifact_leg_index is not None and legacy_leg_index is not None:
        raise ValueError(
            "route_artifact_leg_index and legacy leg_index are mutually "
            "exclusive"
        )
    resolved = (
        route_artifact_leg_index
        if route_artifact_leg_index is not None
        else legacy_leg_index
    )
    if resolved is None:
        resolved = 0
    if type(resolved) is not int or resolved < 0:
        raise ValueError(
            "route_artifact_leg_index must be a non-negative integer"
        )
    return resolved


def build_child_runner_command(
    *,
    profile,
    route_csv: Path,
    diagnostics_json: Path,
    certificate_json: Path,
    run_id: str,
    session_root: Path,
    route_artifact_leg_index: int | None = None,
    leg_index: int | None = None,
    coverage_plan: Path | None = None,
    candidate_snapshot: Path | None = None,
    coverage_transient_replan: dict[str, object] | None = None,
    dry_run: bool,
    uncertainty_map_yaml: Path | None = None,
    uncertainty_sigma_multiplier: float = (
        DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER
    ),
    localization_branch_proof_id: str = "",
    odom_execution_certificate_json: Path | None = None,
    uncertainty_budget_json: Path | None = None,
    mission_motion_authorization_json: Path | None = None,
    runtime_localization_motion_permit_json: Path | None = None,
    mission_leg_motion_authorization_json: Path | None = None,
    mission_leg_motion_permit_json: Path | None = None,
    mission_leg_kind: MissionLegKind | str | None = None,
    mission_leg_index: int | None = None,
    mission_leg_target_id: str = "",
    mission_leg_semantic_map_id: str = "",
    mission_leg_dry_preflight_json: Path | None = None,
    mission_leg_dry_odom_certificate_json: Path | None = None,
    mission_leg_dry_uncertainty_budget_json: Path | None = None,
    startup_reseal_motion_authorization_json: Path | None = None,
    startup_reseal_motion_permit_json: Path | None = None,
    startup_reseal_target_viewpoint_id: str = "",
    startup_reseal_semantic_map_id: str = "",
    mission_session_id: str = "",
) -> list[str]:
    """Build the established single-segment child argv without executing it.

    ``route_artifact_leg_index`` selects a leg encoded in ``route_csv``.  It is
    deliberately independent of ``mission_leg_index`` and the transient
    coverage-replan ``leg_index``.  The older ``leg_index`` keyword remains a
    compatibility alias for an explicit route-artifact selection.
    """

    resolved_route_artifact_leg_index = _resolve_route_artifact_leg_index(
        route_artifact_leg_index=route_artifact_leg_index,
        legacy_leg_index=leg_index,
    )

    run_phase = "dry" if dry_run else "execute"
    odom_fields = (
        uncertainty_map_yaml,
        str(localization_branch_proof_id).strip(),
        odom_execution_certificate_json,
        uncertainty_budget_json,
    )
    odom_execution_requested = any(
        value is not None and value != "" for value in odom_fields
    )
    preflight_name = (
        f"{run_id}_{run_phase}.json"
        if odom_execution_requested
        else f"{run_id}.json"
    )
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/run_single_station_segment.py",
        "--route-csv",
        str(route_csv),
        "--diagnostics-json",
        str(diagnostics_json),
        "--route-certificate-json",
        str(certificate_json),
        "--leg-index",
        str(resolved_route_artifact_leg_index),
        "--run-id",
        run_id,
        "--robot-id",
        profile.robot_id,
        "--namespace",
        profile.namespace,
        "--scan-topic",
        profile.scan_topic,
        "--odom-topic",
        profile.odom_topic,
        "--cmd-vel-topic",
        profile.cmd_vel_topic,
        "--amcl-topic",
        profile.amcl_topic,
        "--map-frame",
        profile.map_frame,
        "--odom-frame",
        profile.odom_frame,
        "--base-frame",
        profile.base_frame,
        "--localization-source",
        profile.localization_source,
        "--max-linear-mps",
        str(profile.max_linear_speed_mps),
        "--max-angular-radps",
        str(profile.max_angular_speed_radps),
        "--min-obstacle-distance-m",
        str(DEFAULT_LIDAR_STOP_DISTANCE_M),
        "--certified-route-tube-radius-m",
        str(DEFAULT_TRACKING_TUBE_RADIUS_M),
        "--results-csv",
        str(session_root / "station_segment_runs.csv"),
        "--semantic-log",
        str(session_root / "run_events" / f"{run_id}.jsonl"),
        "--preflight-json",
        str(session_root / "preflight" / preflight_name),
        "--operator-note",
        "UNLOADED autonomous stand exploration",
    ]
    if odom_execution_requested:
        if any(value is None or value == "" for value in odom_fields):
            raise ValueError(
                "uncertainty-aware odom execution arguments must be complete"
            )
        command.extend(
            [
                "--execution-pose-frame",
                "odom",
                "--odom-execution-certificate-json",
                str(odom_execution_certificate_json),
                "--uncertainty-budget-json",
                str(uncertainty_budget_json),
                "--uncertainty-map-yaml",
                str(uncertainty_map_yaml),
                "--localization-branch-proof-id",
                str(localization_branch_proof_id).strip(),
                "--uncertainty-robot-radius-m",
                str(profile.robot_radius_m),
                "--uncertainty-sigma-multiplier",
                str(uncertainty_sigma_multiplier),
                # Mean stability remains strict. Reported covariance is
                # charged against the route-specific clearance budget.
                "--max-stationary-amcl-position-std-m",
                "0.30",
                "--max-stationary-amcl-yaw-std-rad",
                "0.35",
            ]
        )
    if coverage_plan is not None:
        command.extend(["--coverage-plan", str(coverage_plan)])
    if candidate_snapshot is not None:
        command.extend(["--candidate-snapshot", str(candidate_snapshot)])
    if coverage_transient_replan is not None:
        command.extend(
            [
                "--coverage-transient-replan-survey-root",
                str(coverage_transient_replan["survey_root"]),
                "--coverage-transient-replan-session-root",
                str(coverage_transient_replan["session_root"]),
                "--coverage-transient-replan-map",
                str(coverage_transient_replan["map_yaml"]),
                "--coverage-transient-replan-semantic-map-id",
                str(coverage_transient_replan["semantic_map_id"]),
                "--coverage-transient-replan-target-viewpoint-id",
                str(coverage_transient_replan["target_viewpoint_id"]),
                "--coverage-transient-replan-robot-radius-m",
                str(coverage_transient_replan["robot_radius_m"]),
                "--coverage-transient-replan-max-count",
                str(coverage_transient_replan["max_replans"]),
                "--coverage-transient-replan-leg-index",
                str(coverage_transient_replan["leg_index"]),
                "--omnidirectional-hard-stop-distance-m",
                str(
                    float(coverage_transient_replan["robot_radius_m"])
                    + DEFAULT_COLLISION_MARGIN_M
                ),
            ]
        )
        resume_state_json = coverage_transient_replan.get("resume_state_json")
        if resume_state_json is not None:
            command.extend(
                [
                    "--coverage-transient-replan-resume-state-json",
                    str(resume_state_json),
                ]
            )
    authorization_fields = (
        mission_motion_authorization_json,
        runtime_localization_motion_permit_json,
    )
    if any(value is not None for value in authorization_fields):
        if any(value is None for value in authorization_fields):
            raise ValueError(
                "mission motion authorization and runtime localization "
                "permit must be supplied together"
            )
        if dry_run:
            raise ValueError(
                "runtime localization motion permits are live-run only"
            )
        if not str(mission_session_id).strip():
            raise ValueError(
                "runtime localization motion permit requires mission_session_id"
            )
        if coverage_transient_replan is None:
            raise ValueError(
                "runtime localization motion permit requires a coverage leg"
            )
        command.extend(
            [
                "--mission-motion-authorization-json",
                str(mission_motion_authorization_json),
                "--runtime-localization-motion-permit-json",
                str(runtime_localization_motion_permit_json),
                "--mission-session-id",
                str(mission_session_id).strip(),
            ]
        )
    mission_leg_fields = (
        mission_leg_motion_authorization_json,
        mission_leg_motion_permit_json,
        mission_leg_kind,
        mission_leg_index,
        mission_leg_target_id or None,
        mission_leg_semantic_map_id or None,
        mission_leg_dry_preflight_json,
        mission_leg_dry_odom_certificate_json,
        mission_leg_dry_uncertainty_budget_json,
    )
    if any(value is not None for value in mission_leg_fields):
        if any(value is None for value in mission_leg_fields):
            raise ValueError(
                "mission-leg authorization arguments must be supplied together"
            )
        if any(value is not None for value in authorization_fields):
            raise ValueError(
                "routine mission-leg and runtime-localization permits are "
                "mutually exclusive"
            )
        if dry_run:
            raise ValueError("mission-leg motion permits are live-run only")
        if not str(mission_session_id).strip():
            raise ValueError(
                "mission-leg motion permit requires mission_session_id"
            )
        kind = MissionLegKind(mission_leg_kind)
        if kind not in ROUTINE_MISSION_LEG_KINDS:
            raise ValueError("mission-leg permit requires a routine leg kind")
        assert mission_leg_index is not None
        command.extend(
            [
                "--mission-leg-motion-authorization-json",
                str(mission_leg_motion_authorization_json),
                "--mission-leg-motion-permit-json",
                str(mission_leg_motion_permit_json),
                "--mission-leg-kind",
                kind.value,
                "--mission-leg-index",
                str(mission_leg_index),
                "--mission-leg-target-id",
                str(mission_leg_target_id).strip(),
                "--mission-leg-semantic-map-id",
                str(mission_leg_semantic_map_id).strip(),
                "--mission-leg-dry-preflight-json",
                str(mission_leg_dry_preflight_json),
                "--mission-leg-dry-odom-certificate-json",
                str(mission_leg_dry_odom_certificate_json),
                "--mission-leg-dry-uncertainty-budget-json",
                str(mission_leg_dry_uncertainty_budget_json),
                "--mission-session-id",
                str(mission_session_id).strip(),
            ]
        )
    startup_reseal_fields = (
        startup_reseal_motion_authorization_json,
        startup_reseal_motion_permit_json,
        startup_reseal_target_viewpoint_id or None,
        startup_reseal_semantic_map_id or None,
    )
    if any(value is not None for value in startup_reseal_fields):
        if any(value is None for value in startup_reseal_fields):
            raise ValueError(
                "startup-reseal authorization arguments must be supplied together"
            )
        if any(value is not None for value in authorization_fields) or any(
            value is not None for value in mission_leg_fields
        ):
            raise ValueError(
                "startup-reseal, routine-leg, and runtime-localization permits "
                "are mutually exclusive"
            )
        if dry_run:
            raise ValueError("startup-reseal motion permits are live-run only")
        if not str(mission_session_id).strip():
            raise ValueError(
                "startup-reseal motion permit requires mission_session_id"
            )
        if coverage_transient_replan is None:
            raise ValueError(
                "startup-reseal motion permit requires a coverage leg"
            )
        command.extend(
            [
                "--startup-reseal-motion-authorization-json",
                str(startup_reseal_motion_authorization_json),
                "--startup-reseal-motion-permit-json",
                str(startup_reseal_motion_permit_json),
                "--startup-reseal-target-viewpoint-id",
                str(startup_reseal_target_viewpoint_id).strip(),
                "--startup-reseal-semantic-map-id",
                str(startup_reseal_semantic_map_id).strip(),
                "--mission-session-id",
                str(mission_session_id).strip(),
            ]
        )
    if dry_run:
        command.append("--dry-run")
    return command


def build_bundle_command(profile, run_id: str, runner: list[str]) -> list[str]:
    """Build the evidence-bundle wrapper argv without launching it."""

    return [
        "scripts/common/run_with_bundle.sh",
        "--namespace",
        profile.namespace,
        "--cmd-vel-topic",
        profile.cmd_vel_topic,
        "--scan-topic",
        profile.scan_topic,
        "--odom-topic",
        profile.odom_topic,
        "--amcl-topic",
        profile.amcl_topic,
        "--map-frame",
        profile.map_frame,
        "--odom-frame",
        profile.odom_frame,
        "--base-frame",
        profile.base_frame,
        run_id,
        "--",
        *runner,
    ]


def parse_motion_leg_outcome(
    semantic_log_path: Path,
    *,
    run_id: str,
    returncode: int,
    start_offset: int = 0,
) -> MotionLegOutcome:
    """Parse exactly one terminal event for one child invocation."""

    try:
        events = load_jsonl_event_objects(
            Path(semantic_log_path),
            start_offset=start_offset,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"invalid motion semantic log for {run_id}: {exc}"
        ) from exc
    terminal_events = [
        event
        for event in events
        if event.get("run_id") == run_id
        and event.get("event")
        in {"motion_completed", "safety_stop", "preflight_failed"}
    ]
    if not terminal_events:
        raise RuntimeError(
            f"motion runner produced no terminal motion event for {run_id}"
        )
    if len(terminal_events) != 1:
        raise RuntimeError(
            f"motion runner produced ambiguous terminal events for {run_id}"
        )
    event = terminal_events[0]
    if event.get("event") == "preflight_failed":
        failures = event.get("failures", [])
        if not isinstance(failures, list) or not all(
            isinstance(failure, str) for failure in failures
        ):
            raise RuntimeError(f"preflight failures are invalid for {run_id}")
        status = "preflight_failed"
        stop_reason = "; ".join(failures) or "ROS preflight failed"
        details = {
            "failures": list(failures),
            "observations": event.get("observations", []),
            "runtime_config": event.get("runtime_config", {}),
            "fail_closed": True,
        }
        motion_published = event.get("motion_published", False)
        if motion_published is not False:
            raise RuntimeError(
                "preflight_failed event must carry false motion_published "
                f"evidence for {run_id}"
            )
    else:
        status = str(event.get("status", ""))
        stop_reason = str(event.get("stop_reason", ""))
        details = event.get("stop_details", {})
        motion_published = event.get("motion_published")
        if not isinstance(motion_published, bool):
            raise RuntimeError(
                "motion runner returned non-boolean motion_published "
                f"for {run_id}"
            )
    if status not in {"completed", "stopped", "preflight_failed"}:
        raise RuntimeError(
            f"motion runner returned invalid status {status!r} for {run_id}"
        )
    if (status == "completed") != (returncode == 0):
        raise RuntimeError(
            f"motion runner exit/status mismatch for {run_id}: "
            f"returncode={returncode} status={status}"
        )
    if not isinstance(details, dict):
        raise RuntimeError(
            f"motion runner stop details are invalid for {run_id}"
        )
    return MotionLegOutcome(
        run_id=run_id,
        status=status,
        stop_reason=stop_reason,
        stop_details=dict(details),
        motion_published=motion_published,
        returncode=returncode,
        semantic_log_path=Path(semantic_log_path),
        semantic_log_start_offset=start_offset,
    )


def parse_dry_run_outcome(
    semantic_log_path: Path,
    *,
    run_id: str,
    returncode: int,
    start_offset: int = 0,
) -> MotionLegOutcome:
    """Parse one successful dry child invocation from semantic evidence.

    A process exit code is not dry-run evidence.  The child must append exactly
    one matching ``dry_run_completed`` event which independently states the
    no-motion terminal contract.  Artifact-path validation remains the
    autonomous parent's responsibility because those paths are supplied by
    the parent rather than emitted by this event.
    """

    if type(returncode) is not int or returncode != 0:
        raise RuntimeError(
            f"dry motion runner exit mismatch for {run_id}: "
            f"returncode={returncode!r}"
        )
    try:
        events = load_jsonl_event_objects(
            Path(semantic_log_path),
            start_offset=start_offset,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"invalid dry motion semantic log for {run_id}: {exc}"
        ) from exc
    dry_terminal_events = [
        event
        for event in events
        if event.get("run_id") == run_id
        and event.get("event") == "dry_run_completed"
    ]
    conflicting_terminal_events = [
        event
        for event in events
        if event.get("run_id") == run_id
        and event.get("event")
        in {"motion_completed", "safety_stop", "preflight_failed"}
    ]
    if not dry_terminal_events:
        raise RuntimeError(
            f"dry motion runner produced no dry_run_completed event for {run_id}"
        )
    if len(dry_terminal_events) != 1 or conflicting_terminal_events:
        raise RuntimeError(
            "dry motion runner produced ambiguous terminal events "
            f"for {run_id}"
        )
    event = dry_terminal_events[0]
    if event.get("status") != "dry_run_ok":
        raise RuntimeError(
            "dry motion runner returned invalid dry status "
            f"{event.get('status')!r} for {run_id}"
        )
    if event.get("motion_published") is not False:
        raise RuntimeError(
            "dry_run_completed event must carry false motion_published "
            f"evidence for {run_id}"
        )
    return MotionLegOutcome(
        run_id=run_id,
        status="dry_run_ok",
        stop_reason="",
        stop_details={},
        motion_published=False,
        returncode=returncode,
        semantic_log_path=Path(semantic_log_path),
        semantic_log_start_offset=start_offset,
    )


def semantic_log_size(path: Path) -> int:
    """Return a trusted append boundary for one child invocation."""

    source = Path(path)
    if source.is_symlink():
        raise RuntimeError(f"motion semantic log must not be a symlink: {source}")
    if not source.exists():
        return 0
    if not source.is_file():
        raise RuntimeError(
            f"motion semantic log is not a normal file: {source}"
        )
    return source.stat().st_size


__all__ = [
    "DEFAULT_COLLISION_MARGIN_M",
    "DEFAULT_LIDAR_STOP_DISTANCE_M",
    "DEFAULT_TRACKING_TUBE_RADIUS_M",
    "DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER",
    "MotionLegOutcome",
    "build_bundle_command",
    "build_child_runner_command",
    "parse_dry_run_outcome",
    "parse_motion_leg_outcome",
    "semantic_log_size",
]
