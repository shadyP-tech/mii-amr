"""Server-driven unloaded navigation to previously admitted QR poses.

The server receives a stored-pose arrival report, never a claimed fresh camera
scan. Timed server actions are waited out and recorded; this navigation-only
runner performs no physical pickup, processing, dropoff or charging operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import time
from urllib.parse import quote
import uuid

from scripts.aufgabe04.artifacts.content_store import payload_sha256, write_content_hashed_json


class StationTourError(RuntimeError):
    """A tour stopped before issuing its next operation."""


def validate_available_qr_ids(available_qr_ids) -> int:
    """Return the server qr_count for Start plus a contiguous numbered set."""
    ids = tuple(available_qr_ids)
    if any(not isinstance(qr, str) for qr in ids) or len(set(ids)) != len(ids):
        raise ValueError("saved QR identities must be unique strings")
    count = len(ids) - 1
    expected = {"Start", *(f"QR_{index:03d}" for index in range(1, count + 1))}
    if not 4 <= count <= 10 or set(ids) != expected:
        raise ValueError("randomized tour requires Start plus contiguous QR_001 through QR_00N (4 to 10 numbered QRs)")
    return count


def _identifier(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise StationTourError(f"missing or invalid {label}")
    return value


def _string_list(value, label):
    if not isinstance(value, list):
        raise StationTourError(f"{label} must be a list")
    return [_identifier(item, label) for item in value]


def _mapping_records(value, *, robot_id, available):
    if not isinstance(value, list) or not value:
        raise StationTourError("server QR mappings must be a nonempty list")
    records, seen_qr, seen_station = [], set(), set()
    for item in value:
        if not isinstance(item, Mapping) or item.get("robot_id") != robot_id:
            raise StationTourError("server QR mapping belongs to another robot")
        qr = _identifier(item.get("qr_code_id"), "mapping QR")
        station = _identifier(item.get("station_id"), "mapping station")
        kind = _identifier(item.get("station_type"), "mapping station type")
        if qr not in available or qr in seen_qr or station in seen_station:
            raise StationTourError("server mapping has missing or ambiguous saved identity")
        if kind not in {"start", "depot", "charging", "processing"} or ((qr == "Start") != (kind == "start")):
            raise StationTourError("server mapping has inconsistent Start or station type")
        if (qr == "Start") != (station == "START"):
            raise StationTourError("server mapping conflicts with the built-in Start identity")
        seen_qr.add(qr)
        seen_station.add(station)
        records.append({"robot_id": robot_id, "qr_code_id": qr, "station_id": station, "station_type": kind})
    # The deployed task server reserves physical QR Start / station START;
    # RobotPlanView.qr_mappings contains only the numbered physical QR codes.
    if "Start" not in seen_qr:
        records.append({"robot_id": robot_id, "qr_code_id": "Start", "station_id": "START", "station_type": "start"})
        seen_qr.add("Start")
    if seen_qr != available:
        raise StationTourError("server mappings do not cover exactly the saved QR identities")
    return sorted(records, key=lambda item: item["qr_code_id"])


@dataclass(frozen=True)
class FrozenTourPlan:
    payload: dict
    qr_order: tuple[str, ...]

    @property
    def sha256(self):
        return payload_sha256(self.payload)

    @property
    def by_qr(self):
        return {item["qr_code_id"]: item for item in self.payload["qr_mappings"]}


def validate_tour_plan(plan, mappings, *, robot_id, available_qr_ids) -> FrozenTourPlan:
    """Freeze identity and exact expanded order, excluding only progress cursors."""
    available = set(available_qr_ids)
    if not isinstance(plan, Mapping) or plan.get("robot_id") != robot_id:
        raise StationTourError("server plan belongs to another robot")
    records = _mapping_records(mappings, robot_id=robot_id, available=available)
    embedded = _mapping_records(plan.get("qr_mappings"), robot_id=robot_id, available=available)
    if records != embedded:
        raise StationTourError("plan and QR mapping endpoints disagree")
    path = _string_list(plan.get("expanded_path"), "expanded_path")
    stations = {item["station_id"]: item["qr_code_id"] for item in records}
    if any(station not in stations for station in path):
        raise StationTourError("server plan references a station with no admitted pose")
    order = tuple(stations[station] for station in path)
    if len(order) < 3 or order[0] != "Start" or order[-1] != "Start" or "Start" in order[1:-1]:
        raise StationTourError("server expanded path must begin and finish at Start")
    for field in ("next_job_index", "next_step_index"):
        if type(plan.get(field)) is not int or plan[field] < 0:
            raise StationTourError(f"server plan {field} must be a nonnegative integer")
    generated = plan.get("generated_at")
    if generated is not None:
        _timestamp(generated)
    return FrozenTourPlan({
        "robot_id": robot_id, "mode": _identifier(plan.get("mode"), "plan mode"),
        "processing_sequence": _string_list(plan.get("processing_sequence"), "processing_sequence"),
        "plan_steps": _string_list(plan.get("plan_steps"), "plan_steps"),
        "expanded_path": path, "qr_mappings": records, "generated_at": generated,
    }, order)


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        return parsed.timestamp()
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise StationTourError("server timestamp must be a timezone-aware ISO date") from exc


def _now(clock):
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise StationTourError("clock must return finite nonnegative Unix seconds")
    return float(value)


def _scan_result(response, *, frozen, qr_id, visit_index, mission_id):
    if not isinstance(response, Mapping) or response.get("accepted") is not True:
        raise StationTourError("server rejected the stored-pose arrival report")
    if response.get("robot_id") != frozen.payload["robot_id"] or response.get("qr_code_id") != qr_id:
        raise StationTourError("scan response robot or QR identity changed")
    actual_mission = _identifier(response.get("mission_id"), "scan mission_id")
    if mission_id is not None and actual_mission != mission_id:
        raise StationTourError("server mission changed during tour")
    if response.get("scanned_station") != frozen.by_qr[qr_id]["station_id"]:
        raise StationTourError("scan response station differs from frozen QR mapping")
    if response.get("penalty") is not None:
        raise StationTourError("server reported a penalty; tour stopped")
    final = visit_index == len(frozen.qr_order) - 1
    state = response.get("state")
    next_target = response.get("next_target")
    if final:
        if state != "FINISHED" or next_target is not None or response.get("station_result") not in {"mission_finished", "correct_station"}:
            raise StationTourError("server terminal response contradicts the frozen path")
        next_qr = None
    else:
        if state not in {"GO_TO_DEPOT_PICKUP", "GO_TO_PROCESSING", "GO_TO_DEPOT_DROPOFF", "GO_TO_CHARGING", "GO_TO_START"}:
            raise StationTourError("server ended or rejected the tour before its final Start")
        if response.get("station_result") != "correct_station" or not isinstance(next_target, Mapping):
            raise StationTourError("server omitted a valid next target")
        next_qr = frozen.qr_order[visit_index + 1]
        expected = frozen.by_qr[next_qr]
        if any(next_target.get(key) != expected[key] for key in ("qr_code_id", "station_id", "station_type")):
            raise StationTourError("server next target differs from frozen plan or QR mapping")
        expected_type = {"GO_TO_DEPOT_PICKUP": "depot", "GO_TO_DEPOT_DROPOFF": "depot",
                         "GO_TO_PROCESSING": "processing", "GO_TO_CHARGING": "charging",
                         "GO_TO_START": "start"}[state]
        if expected["station_type"] != expected_type:
            raise StationTourError("server state contradicts its next target type")
    actions = response.get("actions", [])
    if not isinstance(actions, list):
        raise StationTourError("server actions must be a list")
    duration = 0
    for action in actions:
        if (not isinstance(action, Mapping) or action.get("type") not in {
                "pickup_material", "dropoff_item", "process_item", "charge_robot"}
                or type(action.get("duration_s", 0)) is not int or action.get("duration_s", 0) < 0):
            raise StationTourError("server action has invalid type or duration")
        duration += action.get("duration_s", 0)
    earliest = response.get("earliest_next_scan_at")
    return actual_mission, next_qr, actions, duration, None if earliest is None else _timestamp(earliest)


def run_station_tour(client, available_qr_ids, navigate, output_root, *, station_visits=3,
                     clock=time.time, wait=time.sleep, event_id_factory=None,
                     cover_all_stands=True, shuffle=None):
    """Run once; persist every request before I/O and stop on any ambiguous write.

    ``navigate(qr_id, visit_index)`` must return a JSON mapping containing the
    exact ``qr_id`` and ``arrival_verified: true`` after measured arrival.
    ``output_root`` must be new. Exceptions propagate after writing failure
    evidence; existing journals are never resumed or overwritten automatically.
    """
    available = tuple(available_qr_ids)
    qr_count = validate_available_qr_ids(available)
    if type(station_visits) is not int or not 3 <= station_visits <= 100:
        raise ValueError("station_visits must be an integer from 3 to 100")
    if type(cover_all_stands) is not bool:
        raise ValueError("cover_all_stands must be boolean")
    robot_id = _identifier(client.robot_id, "client robot_id")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    journal = root / "journal.jsonl"
    event_id_factory = event_id_factory or (lambda: uuid.uuid4().hex)
    shuffle = shuffle or random.SystemRandom().shuffle
    summary = {"schema_version": 1, "status": "running", "robot_id": robot_id,
               "server_base_url": getattr(client, "base_url", None), "qr_count": qr_count,
               "station_visits": station_visits, "report_source": "stored_pose_arrival",
               "fresh_camera_scan": False, "navigation_only": True, "physical_actions_performed": False,
               "completed_visits": 0, "qr_visit_order": [], "randomization_attempted": False,
               "automatic_write_retries": 0, "journal": str(journal), "visits": [],
               "cover_all_stands": cover_all_stands, "server_mission_finished": False,
               "supplemental_visits": [], "supplemental_qr_visit_order": []}

    def log(event, **data):
        entry = {"event": event, "timestamp_unix_sec": _now(clock), **data}
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def call(operation, invoke, **request):
        robot_path = "/api/v1/robots/" + quote(robot_id, safe="")
        if operation == "report_arrival":
            method, path = "POST", "/api/v1/qr/" + quote(request["qr_id"], safe="") + "/scan"
            body = {"robot_id": robot_id, "client_event_id": request["client_event_id"]}
        elif operation == "randomize_plan":
            method, path, body = "POST", robot_path + "/plan/randomize", dict(request)
        else:
            method, body = "GET", None
            path = robot_path + ("/plan" if operation == "get_plan" else "/qr-mappings")
        log("server_request", operation=operation, request={**request, "method": method, "path": path, "body": body})
        summary["write_outcome_unknown"] = method != "GET"
        try:
            response = invoke()
        except Exception as exc:
            summary["write_outcome_unknown"] = getattr(exc, "write_outcome_unknown", method != "GET")
            log("server_request_failed", operation=operation, error=str(exc),
                write_outcome_unknown=summary["write_outcome_unknown"],
                status_code=getattr(exc, "status_code", None), response=getattr(exc, "response", None))
            raise
        log("server_response", operation=operation, response=response)
        summary["write_outcome_unknown"] = False
        return response

    def arrive(qr, index):
        log("navigation_requested", qr_id=qr, visit_index=index)
        arrival = navigate(qr, index)
        log("navigation_result", qr_id=qr, visit_index=index, arrival=arrival)
        if not isinstance(arrival, Mapping) or arrival.get("arrival_verified") is not True:
            raise StationTourError("navigation did not verify arrival at the stored QR pose")
        if arrival.get("qr_id") != qr:
            raise StationTourError("navigation arrival belongs to another QR")
        return dict(arrival)

    try:
        log("tour_started", **{key: value for key, value in summary.items() if key != "visits"})
        arrival = arrive("Start", 0)
        summary["randomization_attempted"] = True
        randomized = call("randomize_plan", lambda: client.randomize_plan(qr_count=qr_count, stations=station_visits),
                          qr_count=qr_count, stations=station_visits)
        mappings = call("get_qr_mappings", client.get_qr_mappings)
        frozen = validate_tour_plan(randomized, mappings, robot_id=robot_id, available_qr_ids=available)
        if (frozen.payload["mode"] != "random" or len(frozen.payload["processing_sequence"]) != station_visits
                or randomized["next_job_index"] != 0 or randomized["next_step_index"] != 0):
            raise StationTourError("randomized plan does not match the requested fresh production sequence")
        write_content_hashed_json(root / "frozen_plan.json", frozen.payload, hash_field="station_tour_plan_sha256")
        summary.update(plan_sha256=frozen.sha256, planned_qr_order=list(frozen.qr_order),
                       frozen_plan=str(root / "frozen_plan.json"),
                       saved_qr_ids_not_in_server_plan=sorted(set(available) - set(frozen.qr_order)))
        previous_progress = (randomized["next_job_index"], randomized["next_step_index"])

        def refresh():
            nonlocal previous_progress
            plan = call("get_plan", client.get_plan)
            current_mappings = call("get_qr_mappings", client.get_qr_mappings)
            current = validate_tour_plan(plan, current_mappings, robot_id=robot_id, available_qr_ids=available)
            if current.sha256 != frozen.sha256:
                raise StationTourError("server plan or identity mapping changed during tour")
            progress = (plan["next_job_index"], plan["next_step_index"])
            if any(after < before for before, after in zip(previous_progress, progress)):
                raise StationTourError("server progress moved backwards during tour")
            previous_progress = progress

        used_events = set()
        mission_id = None
        next_qr = "Start"
        for visit_index, qr_id in enumerate(frozen.qr_order):
            if next_qr != qr_id:
                raise StationTourError("server target is not the next frozen visit")
            if visit_index:
                refresh()
                arrival = arrive(qr_id, visit_index)
            refresh()
            event_id = _identifier(event_id_factory(), "client_event_id")
            if event_id in used_events:
                raise StationTourError("client event ID was already used for another visit")
            used_events.add(event_id)
            log("stored_pose_arrival_reporting", qr_id=qr_id, visit_index=visit_index,
                client_event_id=event_id, arrival=arrival, report_source="stored_pose_arrival", fresh_camera_scan=False)
            response = call("report_arrival", lambda: client.report_arrival(qr_id, client_event_id=event_id),
                            qr_id=qr_id, robot_id=robot_id, client_event_id=event_id)
            mission_id, next_qr, actions, duration, earliest = _scan_result(
                response, frozen=frozen, qr_id=qr_id, visit_index=visit_index, mission_id=mission_id)
            summary["mission_id"] = mission_id
            visit = {"visit_index": visit_index, "qr_id": qr_id, "client_event_id": event_id,
                     "arrival": arrival, "server_state": response["state"], "actions": actions,
                     "physical_actions_performed": False, "action_wait_completed": False}
            summary["visits"].append(visit)
            deadline = max(_now(clock) + duration, 0.0 if earliest is None else earliest)
            log("server_action_wait_started", visit_index=visit_index, actions=actions,
                duration_s=duration, wait_until_unix_sec=deadline, physical_actions_performed=False)
            while (remaining := deadline - _now(clock)) > 1e-6:
                before = _now(clock)
                wait(min(remaining, 30.0))
                if _now(clock) <= before:
                    raise StationTourError("wait did not advance the clock; actions are unfinished")
            visit["action_wait_completed"] = True
            log("server_action_wait_completed", visit_index=visit_index, physical_actions_performed=False)
            summary["completed_visits"] += 1
            summary["qr_visit_order"].append(qr_id)
        refresh()
        summary["server_mission_finished"] = True
        missing = list(summary["saved_qr_ids_not_in_server_plan"])
        if cover_all_stands and missing:
            shuffle(missing)
            if len(set(missing)) != len(missing) or set(missing) != set(summary["saved_qr_ids_not_in_server_plan"]):
                raise StationTourError("supplemental shuffle changed the missing stand population")
            supplemental = [*missing, "Start"]
            log("supplemental_navigation_planned", qr_order=supplemental,
                server_mission_finished=True, server_reports_enabled=False)
            for offset, qr_id in enumerate(supplemental):
                visit_index = len(frozen.qr_order) + offset
                arrival = arrive(qr_id, visit_index)
                summary["supplemental_visits"].append({
                    "visit_index": visit_index, "qr_id": qr_id, "arrival": arrival,
                    "server_report_sent": False, "purpose": "visit_unrequested_saved_stands_then_return_start",
                })
                summary["supplemental_qr_visit_order"].append(qr_id)
                log("supplemental_arrival_verified", visit_index=visit_index, qr_id=qr_id,
                    arrival=arrival, server_report_sent=False)
        visited = set(summary["qr_visit_order"]) | set(summary["supplemental_qr_visit_order"])
        summary.update(visited_qr_ids=sorted(visited), all_saved_stands_visited=visited == set(available),
                       supplemental_visit_count=len(summary["supplemental_visits"]),
                       total_arrivals_verified=summary["completed_visits"] + len(summary["supplemental_visits"]))
        summary["status"] = "completed"
        log("tour_completed", mission_id=mission_id, completed_visits=summary["completed_visits"])
        write_content_hashed_json(root / "summary.json", summary, hash_field="station_tour_summary_sha256")
        return summary
    except BaseException as exc:
        summary.update(status="failed_closed", error=str(exc), error_type=type(exc).__name__,
                       write_outcome_unknown=getattr(exc, "write_outcome_unknown", summary.get("write_outcome_unknown", False)))
        log("tour_failed", error=str(exc), error_type=type(exc).__name__,
            completed_visits=summary["completed_visits"])
        write_content_hashed_json(root / "summary.json", summary, hash_field="station_tour_summary_sha256")
        raise
