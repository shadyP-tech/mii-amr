"""Parent-side adapter for one startup active-localization child process."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
    STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD,
    StartupActiveLocalizationConfig,
    load_startup_active_localization_authorization,
    load_startup_active_localization_result,
    startup_active_localization_attempt_dir,
    stored_content_hash,
)
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    StartupRouteUncertaintySelectionRejected,
)


@dataclass(frozen=True)
class StartupActiveLocalizationChildRequest:
    session_id: str
    session_root: Path
    profile: object
    config: StartupActiveLocalizationConfig
    attempt_index: int
    rejected_selection: StartupRouteUncertaintySelectionRejected


@dataclass(frozen=True)
class StartupActiveLocalizationChildOutcome:
    result: Mapping[str, object]
    result_path: Path
    semantic_log_path: Path
    controller_trace_path: Path
    preflight_path: Path
    returncode: int


def build_startup_active_localization_child_command(
    request: StartupActiveLocalizationChildRequest,
) -> tuple[list[str], Path, Path, Path]:
    if not isinstance(request.config, StartupActiveLocalizationConfig):
        raise ValueError("request config must be StartupActiveLocalizationConfig")
    request.config.direction_for_attempt(request.attempt_index)
    profile = request.profile
    attempt_dir = startup_active_localization_attempt_dir(
        request.session_root,
        attempt_index=request.attempt_index,
    )
    run_id = (
        f"{request.session_id}_startup_active_localization_"
        f"{request.attempt_index:03d}"
    )
    result_path = attempt_dir / "startup_active_localization_result.json"
    semantic_log_path = attempt_dir / "startup_active_localization_events.jsonl"
    controller_trace_path = attempt_dir / "controller_trace.jsonl"
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/entrypoints/"
        "run_startup_active_localization.py",
        "--run-id",
        run_id,
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
        "--attempt-index",
        str(request.attempt_index),
        "--max-attempts",
        str(request.config.max_attempts),
        "--rotation-rad",
        str(request.config.rotation_rad),
        "--angular-speed-radps",
        str(request.config.angular_speed_radps),
        "--maximum-angular-speed-radps",
        str(profile.max_angular_speed_radps),
        "--timeout-sec",
        str(request.config.timeout_sec),
        "--source-route-selection-json",
        str(request.rejected_selection.evidence_path),
        "--source-route-selection-sha256",
        request.rejected_selection.evidence_sha256,
        "--result-json",
        str(result_path),
        "--controller-trace-jsonl",
        str(controller_trace_path),
        "--semantic-log",
        str(semantic_log_path),
    ]
    return command, result_path, semantic_log_path, controller_trace_path


def run_startup_active_localization_child(
    request: StartupActiveLocalizationChildRequest,
    *,
    run_process: Callable[..., object] = subprocess.run,
) -> StartupActiveLocalizationChildOutcome:
    command, result_path, semantic_log_path, controller_trace_path = (
        build_startup_active_localization_child_command(request)
    )
    authorization_path = (
        result_path.parent / "startup_active_localization_authorization.json"
    )
    preflight_path = (
        result_path.parent / "startup_active_localization_preflight.json"
    )
    existing = [
        path
        for path in (
            result_path,
            semantic_log_path,
            controller_trace_path,
            authorization_path,
            preflight_path,
        )
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise RuntimeError(
            "refusing to reuse startup active-localization artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    completed = run_process(command, check=False)
    returncode = int(getattr(completed, "returncode", -1))
    try:
        result = load_startup_active_localization_result(result_path)
    except ValueError as exc:
        raise RuntimeError(
            "startup active localization produced invalid result evidence: "
            f"{exc}"
        ) from exc
    try:
        authorization = load_startup_active_localization_authorization(
            authorization_path
        )
    except ValueError as exc:
        raise RuntimeError(
            "startup active localization produced invalid authorization "
            f"evidence: {exc}"
        ) from exc
    try:
        preflight = load_content_hashed_json(
            preflight_path,
            hash_field=STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD,
        )
        preflight_sha256 = stored_content_hash(
            preflight_path,
            hash_field=STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD,
        )
    except ValueError as exc:
        raise RuntimeError(
            "startup active localization produced invalid preflight evidence: "
            f"{exc}"
        ) from exc
    if preflight.get("ok") is not True or preflight.get("failures") != []:
        raise RuntimeError(
            "startup active localization preflight evidence is not successful"
        )
    expected = {
        "run_id": (
            f"{request.session_id}_startup_active_localization_"
            f"{request.attempt_index:03d}"
        ),
        "attempt_index": request.attempt_index,
        "source_route_selection_json": str(
            request.rejected_selection.evidence_path
        ),
        "source_route_selection_sha256": (
            request.rejected_selection.evidence_sha256
        ),
        "mission_run_authorized": False,
        "route_authorized": False,
        "requires_fresh_stationary_localization": True,
        "requires_separate_mission_run": True,
        "operator_confirmation": STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
        "translation_commanded": False,
        "controller_trace_jsonl": str(controller_trace_path),
        "preflight_json": str(preflight_path),
        "preflight_sha256": preflight_sha256,
        "config": request.config.to_evidence_dict(),
    }
    mismatches = {
        name: {"expected": expected_value, "actual": result.get(name)}
        for name, expected_value in expected.items()
        if result.get(name) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "startup active-localization result binding mismatch: "
            f"{mismatches}"
        )
    authorization_expected = {
        "run_id": expected["run_id"],
        "attempt_index": request.attempt_index,
        "source_route_selection_json": expected[
            "source_route_selection_json"
        ],
        "source_route_selection_sha256": expected[
            "source_route_selection_sha256"
        ],
        "config": request.config.to_evidence_dict(),
        "operator_confirmation": STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
        "route_authorized": False,
        "mission_run_authorized": False,
        "preflight_json": str(preflight_path),
        "preflight_sha256": preflight_sha256,
    }
    authorization_mismatches = {
        name: {"expected": wanted, "actual": authorization.get(name)}
        for name, wanted in authorization_expected.items()
        if authorization.get(name) != wanted
    }
    if authorization_mismatches:
        raise RuntimeError(
            "startup active-localization authorization binding mismatch: "
            f"{authorization_mismatches}"
        )
    if returncode != 0 or result.get("status") != "completed":
        raise RuntimeError(
            "startup active localization failed before route retry: "
            + str(result.get("stop_reason", "unknown"))
        )
    if result.get("motion_published") is not True:
        raise RuntimeError(
            "startup active localization completed without motion evidence"
        )
    if not controller_trace_path.is_file() or (
        controller_trace_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            "startup active localization completed without controller trace"
        )
    if not semantic_log_path.is_file() or semantic_log_path.stat().st_size <= 0:
        raise RuntimeError(
            "startup active localization completed without semantic events"
        )
    zero_count = result.get("zero_command_count")
    if (
        type(zero_count) is not int
        or zero_count < request.config.stop_command_count
    ):
        raise RuntimeError(
            "startup active localization has insufficient zero-command evidence"
        )
    try:
        maximum_translation_m = float(result["maximum_translation_m"])
        requested_rotation_rad = float(result["requested_rotation_rad"])
        accumulated_progress_rad = float(result["accumulated_progress_rad"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "startup active localization has invalid motion measurements"
        ) from exc
    if not all(
        math.isfinite(value)
        for value in (
            maximum_translation_m,
            requested_rotation_rad,
            accumulated_progress_rad,
        )
    ):
        raise RuntimeError(
            "startup active localization has non-finite motion measurements"
        )
    if maximum_translation_m > request.config.maximum_translation_m + 1.0e-9:
        raise RuntimeError(
            "startup active localization exceeded its translation bound"
        )
    if abs(requested_rotation_rad - request.config.rotation_rad) > 1.0e-9:
        raise RuntimeError(
            "startup active localization result changed the requested rotation"
        )
    if accumulated_progress_rad + 1.0e-9 < request.config.target_progress_rad:
        raise RuntimeError(
            "startup active localization completed below its yaw target"
        )
    stop_details = result.get("stop_details")
    stationary_odom = (
        stop_details.get("stationary_odom")
        if isinstance(stop_details, Mapping)
        else None
    )
    if not isinstance(stationary_odom, Mapping) or (
        stationary_odom.get("accepted") is not True
    ):
        raise RuntimeError(
            "startup active localization lacks stopped odometry proof"
        )
    return StartupActiveLocalizationChildOutcome(
        result=result,
        result_path=result_path,
        semantic_log_path=semantic_log_path,
        controller_trace_path=controller_trace_path,
        preflight_path=preflight_path,
        returncode=returncode,
    )


__all__ = [
    "StartupActiveLocalizationChildOutcome",
    "StartupActiveLocalizationChildRequest",
    "build_startup_active_localization_child_command",
    "run_startup_active_localization_child",
]
