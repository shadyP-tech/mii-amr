"""Structured event and CSV reporting for station-segment runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Mapping

from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.foundation.run_events import emit_event
from scripts.aufgabe04.navigation.foundation.segment_run_logger import append_segment_run

def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """Append one post-adoption mission event or fail the zero-held handoff."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
            + "\n"
        )

def _record_motion_authorization_rejection(
    *,
    args,
    resolved,
    leg,
    event_logger,
    failure: object,
) -> int:
    """Persist one no-motion authorization failure with terminal evidence."""

    stop_reason = f"motion authorization rejected: {failure}"
    stop_details = {
        "reason": stop_reason,
        "fault_code": "motion_authorization_rejected",
        "source": "motion_authorization",
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
    _append_result(args, resolved, leg, preflight_ok=False, result=result)
    emit_event(
        event_logger,
        "motion_authorization_rejected",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        status=result.status,
        stop_reason=stop_reason,
        motion_published=False,
        stop_details=stop_details,
    )
    emit_event(
        event_logger,
        "preflight_failed",
        run_id=args.run_id,
        leg_index=leg.leg_index,
        status=result.status,
        failures=[stop_reason],
        observations=[],
        runtime_config=resolved.as_log_dict(),
        motion_published=False,
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

def _base_log_row(args, resolved, leg, preflight_ok: bool) -> Dict[str, object]:
    configured = resolved.configured
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "robot_id": args.robot_id,
        "namespace": resolved.namespace,
        "configured_cmd_vel_topic": configured.cmd_vel_topic,
        "resolved_cmd_vel_topic": resolved.cmd_vel_topic,
        "configured_scan_topic": configured.scan_topic,
        "resolved_scan_topic": resolved.scan_topic,
        "configured_odom_topic": configured.odom_topic,
        "resolved_odom_topic": resolved.odom_topic,
        "map_frame": resolved.map_frame,
        "odom_frame": resolved.odom_frame,
        "base_frame": resolved.base_frame,
        "leg_index": leg.leg_index,
        "raw_point_count": len(leg.raw_waypoints),
        "executable_point_count": len(leg.executable_waypoints),
        "route_length_m": f"{leg.route_length_m:.6f}",
        "preflight_ok": preflight_ok,
        "operator_note": args.operator_note,
    }

def _append_result(args, resolved, leg, preflight_ok: bool, result: FollowerResult) -> None:
    row = _base_log_row(args, resolved, leg, preflight_ok)
    row.update(
        {
            "status": result.status,
            "stop_reason": result.stop_reason,
            "duration_sec": f"{result.duration_sec:.3f}",
            "distance_estimate_m": f"{result.distance_estimate_m:.6f}",
            "motion_published": result.motion_published,
            "semantic_log_path": args.semantic_log,
            "preflight_json_path": args.preflight_json or "",
        }
    )
    append_segment_run(args.results_csv, row)


def record_unexpected_follower_exception(
    *,
    args,
    resolved,
    leg,
    event_logger,
    failure: Exception,
    mission_leg_event_fields: Mapping[str, object],
) -> None:
    """Best-effort terminal evidence for an exception inside the follower.

    The follower did not return its authoritative motion history, so this
    boundary must conservatively report that motion may have been published.
    Terminal events are attempted before the nonessential CSV write, and no
    reporting failure is allowed to mask the original follower exception.
    """

    exception_type = type(failure).__name__
    exception_message = str(failure)
    stop_reason = (
        f"unexpected follower exception: {exception_type}: {exception_message}"
    )
    stop_details = {
        "reason": stop_reason,
        "fault_code": "unexpected_follower_exception",
        "source": "station_segment_follower_boundary",
        "fail_closed": True,
        "motion_published": True,
        "motion_history_uncertain": True,
        "motion_history_uncertainty": (
            "the follower raised before returning its motion result; "
            "nonzero command publication cannot be ruled out"
        ),
        "exception_type": exception_type,
        "exception_message": exception_message,
        "recovery_attempted": False,
        "continuation_allowed": False,
    }
    safety_stop_fields = {
        "run_id": args.run_id,
        "leg_index": leg.leg_index,
        **dict(mission_leg_event_fields),
        "status": "stopped",
        "stop_reason": stop_reason,
        "motion_published": True,
        "stop_details": stop_details,
    }

    try:
        emit_event(event_logger, "safety_stop", **safety_stop_fields)
    except Exception:
        pass
    try:
        emit_event(
            event_logger,
            "run_finished",
            run_id=args.run_id,
            final_status="stopped",
            stop_reason=stop_reason,
            motion_published=True,
            motion_history_uncertain=True,
            fail_closed=True,
            fault_code="unexpected_follower_exception",
            results_csv=str(args.results_csv),
            semantic_log_path=str(args.semantic_log),
            preflight_json_path=str(args.preflight_json or ""),
        )
    except Exception:
        pass

    row = _base_log_row(args, resolved, leg, preflight_ok=True)
    row.update(
        {
            "status": "stopped",
            "stop_reason": stop_reason,
            "duration_sec": "",
            "distance_estimate_m": "",
            "motion_published": True,
            "semantic_log_path": args.semantic_log,
            "preflight_json_path": args.preflight_json or "",
        }
    )
    try:
        append_segment_run(args.results_csv, row)
    except Exception:
        pass


def _append_status_result(
    args,
    resolved,
    leg,
    *,
    preflight_ok: bool,
    status: str,
    stop_reason: str,
) -> None:
    _append_result(
        args,
        resolved,
        leg,
        preflight_ok,
        FollowerResult(status, stop_reason, 0.0, 0.0, False),
    )

def _observation_log_rows(observations) -> list[dict[str, object]]:
    return [
        {
            **observation.data,
            "name": observation.name,
            "ok": observation.ok,
            "detail": observation.detail,
        }
        for observation in observations
    ]
