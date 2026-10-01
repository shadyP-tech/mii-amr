from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.logistics.station_tour import (
    StationTourError, run_station_tour, validate_available_qr_ids, validate_tour_plan,
)
from scripts.aufgabe04.task_client.station_tour_client import StationTourHttpError


QR_IDS = ("Start", "QR_001", "QR_002", "QR_003", "QR_004")


def plan_fixture():
    mappings = [{"robot_id": "robot", "qr_code_id": qr, "station_id": station,
                 "station_type": kind, "display_name": None}
                for qr, station, kind in (
                    ("QR_001", "DEPOT_01", "depot"), ("QR_002", "CHARGE_01", "charging"),
                    ("QR_003", "PROC_03", "processing"), ("QR_004", "PROC_04", "processing"))]
    return {"robot_id": "robot", "mode": "random", "processing_sequence": ["PROC_03", "PROC_03", "PROC_03"],
            "plan_steps": ["PROC_03", "DEPOT_PICKUP", "PROC_03"],
            "expanded_path": ["START", "PROC_03", "DEPOT_01", "PROC_03", "START"],
            "qr_mappings": mappings, "next_job_index": 0, "next_step_index": 0,
            "generated_at": "2026-09-28T10:00:00Z"}


class FakeClock:
    def __init__(self):
        self.value = 1790590000.0
        self.waits = []

    def __call__(self):
        return self.value

    def wait(self, duration):
        self.waits.append(duration)
        self.value += duration


class FakeClient:
    robot_id = "robot"
    base_url = "http://fixture.invalid"

    def __init__(self, events):
        self.events = events
        self.plan = plan_fixture()
        self.reported = []
        self.responses = []
        order = validate_tour_plan(self.plan, self.plan["qr_mappings"], robot_id=self.robot_id,
                                   available_qr_ids=QR_IDS).qr_order
        by_qr = validate_tour_plan(self.plan, self.plan["qr_mappings"], robot_id=self.robot_id,
                                  available_qr_ids=QR_IDS).by_qr
        for index, qr in enumerate(order):
            target = None if index == len(order) - 1 else by_qr[order[index + 1]]
            state = ("FINISHED" if target is None else
                     {"start": "GO_TO_START", "depot": "GO_TO_DEPOT_PICKUP",
                      "processing": "GO_TO_PROCESSING", "charging": "GO_TO_CHARGING"}[target["station_type"]])
            self.responses.append({"accepted": True, "robot_id": "robot", "qr_code_id": qr,
                "mission_id": "mission-1", "scanned_station": by_qr[qr]["station_id"],
                "state": state, "station_result": "mission_finished" if target is None else "correct_station",
                "actions": [], "next_target": target, "earliest_next_scan_at": None, "penalty": None})

    def randomize_plan(self, *, qr_count, stations):
        self.events.append(("randomize", qr_count, stations))
        return deepcopy(self.plan)

    def get_plan(self):
        self.events.append(("get_plan",))
        return deepcopy(self.plan)

    def get_qr_mappings(self):
        self.events.append(("get_mappings",))
        return deepcopy(self.plan["qr_mappings"])

    def report_arrival(self, qr, *, client_event_id):
        self.events.append(("report", qr, client_event_id))
        index = len(self.reported)
        self.reported.append(qr)
        self.plan["next_step_index"] += 1
        return deepcopy(self.responses[index])


class StationTourTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "server"
        self.events = []
        self.client = FakeClient(self.events)
        self.clock = FakeClock()
        self.counter = 0

    def navigate(self, qr, index):
        self.events.append(("navigate", qr, index, self.clock()))
        return {"arrival_verified": True, "qr_id": qr, "position_error_m": 0.01}

    def event_id(self):
        self.counter += 1
        return f"event-{self.counter}"

    def run_tour(self, **overrides):
        args = dict(client=self.client, available_qr_ids=QR_IDS, navigate=self.navigate,
                    output_root=self.output, station_visits=3, clock=self.clock,
                    wait=self.clock.wait, event_id_factory=self.event_id,
                    shuffle=lambda values: values.reverse())
        args.update(overrides)
        return run_station_tour(**args)

    def read_summary(self):
        return json.loads((self.output / "summary.json").read_text())

    def test_validated_server_plan_precedes_start_and_every_report_requires_arrival(self):
        summary = self.run_tour(cover_all_stands=False)
        self.assertEqual(self.events[:2], [("randomize", 4, 3), ("get_mappings",)])
        self.assertEqual(self.events[2][:3], ("navigate", "Start", 0))
        self.assertEqual(sum(event[0] == "randomize" for event in self.events), 1)
        self.assertEqual(summary["qr_visit_order"], ["Start", "QR_003", "QR_001", "QR_003", "Start"])
        self.assertEqual(summary["completed_visits"], 5)
        self.assertFalse(summary["physical_actions_performed"])
        self.assertFalse(summary["fresh_camera_scan"])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len({event[2] for event in self.events if event[0] == "report"}), 5)
        for index, event in enumerate(self.events):
            if event[0] == "report":
                self.assertEqual(next(item for item in reversed(self.events[:index]) if item[0] == "navigate")[1], event[1])

    def test_missing_saved_stands_are_visited_after_finish_without_server_reports(self):
        summary = self.run_tour()
        self.assertEqual(summary["saved_qr_ids_not_in_server_plan"], ["QR_002", "QR_004"])
        self.assertEqual(summary["supplemental_qr_visit_order"], ["QR_004", "QR_002", "Start"])
        self.assertTrue(summary["all_saved_stands_visited"])
        self.assertEqual(summary["visited_qr_ids"], sorted(QR_IDS))
        self.assertEqual(len(self.client.reported), 5)
        self.assertEqual(summary["total_arrivals_verified"], 8)
        tail = [event for event in self.events if event[0] == "navigate"][-3:]
        self.assertEqual([event[2] for event in tail], [5, 6, 7])

    def test_failed_start_leaves_created_plan_but_never_reports_arrival(self):
        with self.assertRaises(StationTourError):
            self.run_tour(navigate=lambda qr, index: {"arrival_verified": False, "qr_id": qr})
        self.assertEqual(self.events, [("randomize", 4, 3), ("get_mappings",)])
        self.assertTrue(self.read_summary()["randomization_attempted"])
        self.assertTrue(self.read_summary()["randomization_completed"])
        self.assertEqual(self.client.reported, [])
        self.assertTrue((self.output / "frozen_plan.json").exists())

    def test_arrival_requires_explicit_matching_qr(self):
        for arrival in ({"arrival_verified": True}, {"arrival_verified": True, "qr_id": "wrong"}):
            with self.subTest(arrival=arrival), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(StationTourError):
                    self.run_tour(output_root=Path(temporary)/"run", navigate=lambda qr, index: arrival)
        self.assertFalse(any(event[0] == "report" for event in self.events))

    def test_server_randomization_failure_never_starts_navigation_or_retries(self):
        def failed(**request):
            self.events.append(("randomize", request["qr_count"], request["stations"]))
            raise StationTourHttpError("plan service unavailable", write_outcome_unknown=True)
        self.client.randomize_plan = failed
        messages = []
        with self.assertRaisesRegex(StationTourHttpError, "unavailable"):
            self.run_tour(progress=messages.append)
        self.assertEqual(self.events, [("randomize", 4, 3)])
        self.assertEqual(self.client.reported, [])
        self.assertTrue(self.read_summary()["write_outcome_unknown"])
        self.assertFalse(self.read_summary()["randomization_completed"])
        self.assertIn("POST /api/v1/robots/robot/plan/randomize", messages[0])
        self.assertIn("plan service unavailable", messages[-1])

    def test_malformed_initial_plan_or_mappings_never_starts_navigation(self):
        for failure in ("plan", "mappings"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                client = FakeClient(self.events)
                if failure == "plan":
                    client.plan["expanded_path"][1] = "UNKNOWN"
                else:
                    client.get_qr_mappings = lambda: []
                with self.assertRaises(StationTourError):
                    self.run_tour(client=client, output_root=Path(folder)/"server")
                self.assertFalse(any(event[0] == "navigate" for event in self.events))
                self.assertEqual(client.reported, [])

    def test_malformed_saved_identity_set_has_no_side_effect(self):
        for ids in (QR_IDS[:-1], (*QR_IDS, "QR_004"), ("Start", "QR001", "QR002", "QR003", "QR004")):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_tour(available_qr_ids=ids)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.events, [])

    def test_action_durations_and_server_deadline_are_waited_before_departure(self):
        start_time = self.clock()
        response = self.client.responses[0]
        response["actions"] = [{"type": "pickup_material", "duration_s": 4}, {"type": "process_item", "duration_s": 6}]
        response["earliest_next_scan_at"] = datetime.fromtimestamp(start_time+12, timezone.utc).isoformat()
        self.run_tour(cover_all_stands=False)
        next_arrival = next(event for event in self.events if event[:3] == ("navigate", "QR_003", 1))
        self.assertEqual(next_arrival[3], start_time+12)
        self.assertEqual(sum(self.clock.waits), 12)

    def test_progress_reports_requests_validated_target_and_dwell_in_order(self):
        messages = []
        self.client.responses[0]["actions"] = [{"type": "pickup_material", "duration_s": 4}]
        self.run_tour(cover_all_stands=False, progress=messages.append)
        plan = next(i for i, text in enumerate(messages) if "Server plan validated" in text)
        start = messages.index("Arrival verified: Start (visit 0).")
        target = messages.index("Server accepted Start; next target: QR_003.")
        hold = next(i for i, text in enumerate(messages) if "Waiting 4.0 s at Start" in text)
        finished = messages.index("Server action wait complete at Start.")
        onward = messages.index("Arrival verified: QR_003 (visit 1).")
        self.assertLess(plan, start)
        self.assertLess(start, target)
        self.assertLess(target, hold)
        self.assertLess(hold, finished)
        self.assertLess(finished, onward)
        self.assertEqual(messages[-1], "Station tour completed.")

    def test_invalid_action_or_wait_failure_never_departs(self):
        self.client.responses[0]["actions"] = [{"type": "pickup_material", "duration_s": 5}]
        with self.assertRaisesRegex(StationTourError, "clock"):
            self.run_tour(wait=lambda duration: None)
        self.assertEqual(len([event for event in self.events if event[0] == "navigate"]), 1)
        self.assertEqual(self.read_summary()["status"], "failed_closed")

    def test_lost_scan_response_is_recorded_once_with_original_event_id(self):
        def lost(qr, *, client_event_id):
            self.events.append(("report_lost", qr, client_event_id))
            raise StationTourHttpError("lost response", write_outcome_unknown=True)
        self.client.report_arrival = lost
        with self.assertRaises(StationTourHttpError):
            self.run_tour()
        self.assertEqual(len([event for event in self.events if event[0] == "report_lost"]), 1)
        journal = [json.loads(line) for line in (self.output/"journal.jsonl").read_text().splitlines()]
        request = next(event for event in journal if event["event"] == "server_request" and event["operation"] == "report_arrival")
        self.assertEqual(request["request"]["client_event_id"], "event-1")
        self.assertTrue(self.read_summary()["write_outcome_unknown"])

    def test_changed_plan_before_scan_stops_without_reporting(self):
        original = self.client.get_plan
        def changed():
            plan = original()
            plan["expanded_path"][1] = "PROC_04"
            return plan
        self.client.get_plan = changed
        with self.assertRaisesRegex(StationTourError, "changed"):
            self.run_tour()
        self.assertEqual(self.client.reported, [])

    def test_randomization_must_return_requested_production_count(self):
        self.client.plan["processing_sequence"].pop()
        with self.assertRaisesRegex(StationTourError, "requested fresh production"):
            self.run_tour()
        self.assertEqual(self.client.reported, [])
        self.assertFalse(any(event[0] == "navigate" for event in self.events))

    def test_mutated_mapping_stops_even_if_both_endpoints_agree(self):
        original = self.client.get_plan
        def changed():
            first, second = self.client.plan["qr_mappings"][-2:]
            first["station_id"], second["station_id"] = second["station_id"], first["station_id"]
            return original()
        self.client.get_plan = changed
        with self.assertRaisesRegex(StationTourError, "identity mapping changed"):
            self.run_tour()
        self.assertEqual(self.client.reported, [])

    def test_interrupt_during_post_keeps_unknown_write_outcome(self):
        def interrupted(qr, *, client_event_id):
            raise KeyboardInterrupt("operator interrupted request")
        self.client.report_arrival = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.run_tour()
        self.assertTrue(self.read_summary()["write_outcome_unknown"])

    def test_server_target_must_match_the_frozen_next_visit(self):
        self.client.responses[0]["next_target"]["qr_code_id"] = "QR_004"
        with self.assertRaisesRegex(StationTourError, "next target"):
            self.run_tour()
        self.assertEqual(self.client.reported, ["Start"])
        self.assertEqual(len([event for event in self.events if event[0] == "navigate"]), 1)

    def test_terminal_state_cannot_finish_early_or_continue_after_final_start(self):
        self.client.responses[0].update(state="FINISHED", station_result="mission_finished", next_target=None)
        with self.assertRaisesRegex(StationTourError, "before its final Start"):
            self.run_tour()
        self.assertEqual(self.client.reported, ["Start"])

    def test_mission_identity_cannot_change(self):
        self.client.responses[1]["mission_id"] = "other"
        with self.assertRaisesRegex(StationTourError, "mission changed"):
            self.run_tour()
        self.assertEqual(self.client.reported, ["Start", "QR_003"])

    def test_rejected_scan_does_not_issue_next_navigation(self):
        self.client.responses[0]["accepted"] = False
        with self.assertRaisesRegex(StationTourError, "rejected"):
            self.run_tour()
        self.assertEqual(len([event for event in self.events if event[0] == "navigate"]), 1)

    def test_unfinished_final_state_does_not_allow_supplemental_navigation(self):
        self.client.responses[-1]["state"] = "GO_TO_START"
        with self.assertRaisesRegex(StationTourError, "terminal"):
            self.run_tour()
        self.assertFalse(self.read_summary()["server_mission_finished"])
        self.assertEqual(len([event for event in self.events if event[0] == "navigate"]), 5)

    def test_finished_state_with_next_target_is_a_contradiction(self):
        self.client.responses[-1]["next_target"] = deepcopy(self.client.responses[0]["next_target"])
        with self.assertRaisesRegex(StationTourError, "terminal"):
            self.run_tour()
        self.assertFalse(self.read_summary()["server_mission_finished"])

    def test_plan_progress_cannot_reset_after_scan(self):
        original = self.client.get_plan
        reads = [0]
        def reset():
            plan = original()
            reads[0] += 1
            if reads[0] == 3:
                plan["next_step_index"] = 0
            return plan
        self.client.get_plan = reset
        with self.assertRaisesRegex(StationTourError, "backwards"):
            self.run_tour()
        self.assertEqual(self.client.reported, ["Start"])

    def test_failed_later_arrival_is_never_reported(self):
        def navigation(qr, index):
            result = self.navigate(qr, index)
            result["arrival_verified"] = index == 0
            return result
        with self.assertRaisesRegex(StationTourError, "verify arrival"):
            self.run_tour(navigate=navigation)
        self.assertEqual(self.client.reported, ["Start"])

    def test_timed_action_failure_stops_and_records_no_physical_action(self):
        self.client.responses[0]["actions"] = [{"type": "pickup_material", "duration_s": -1}]
        with self.assertRaisesRegex(StationTourError, "duration"):
            self.run_tour()
        self.assertFalse(self.read_summary()["physical_actions_performed"])
        self.assertEqual(self.client.reported, ["Start"])

    def test_supplemental_failure_preserves_finished_server_state(self):
        def navigation(qr, index):
            result = self.navigate(qr, index)
            if index == 5:
                result["arrival_verified"] = False
            return result
        with self.assertRaises(StationTourError):
            self.run_tour(navigate=navigation)
        summary = self.read_summary()
        self.assertTrue(summary["server_mission_finished"])
        self.assertEqual(summary["status"], "failed_closed")
        self.assertEqual(len(self.client.reported), 5)

    def test_reused_event_id_stops_before_second_report(self):
        with self.assertRaisesRegex(StationTourError, "already used"):
            self.run_tour(event_id_factory=lambda: "duplicate")
        self.assertEqual(self.client.reported, ["Start"])

    def test_randomization_request_is_persisted_before_mutating_call(self):
        original = self.client.randomize_plan
        def randomized(**request):
            last = json.loads((self.output/"journal.jsonl").read_text().splitlines()[-1])
            self.assertEqual(last["event"], "server_request")
            self.assertEqual(last["operation"], "randomize_plan")
            self.assertEqual(last["request"]["method"], "POST")
            self.assertEqual(last["request"]["body"], {"qr_count": 4, "stations": 3})
            return original(**request)
        self.client.randomize_plan = randomized
        self.run_tour(cover_all_stands=False)

    def test_existing_output_is_never_reused(self):
        self.output.mkdir()
        (self.output/"marker").write_text("original")
        with self.assertRaises(FileExistsError):
            self.run_tour()
        self.assertEqual((self.output/"marker").read_text(), "original")
        self.assertEqual(self.events, [])

    def test_supplemental_shuffle_cannot_add_or_drop_identities(self):
        with self.assertRaisesRegex(StationTourError, "shuffle"):
            self.run_tour(shuffle=lambda values: values.append("unknown"))
        self.assertTrue(self.read_summary()["server_mission_finished"])
        self.assertEqual(len(self.client.reported), 5)

    def test_actual_server_plan_fixture_allows_builtin_start_and_unused_mappings(self):
        plan = plan_fixture()
        frozen = validate_tour_plan(plan, plan["qr_mappings"], robot_id="robot", available_qr_ids=QR_IDS)
        self.assertEqual(frozen.by_qr["Start"]["station_id"], "START")
        self.assertEqual(frozen.qr_order.count("QR_003"), 2)
        self.assertNotIn("QR_002", frozen.qr_order)

    def test_plan_rejects_missing_duplicate_and_conflicting_identity(self):
        for change in ("missing", "duplicate", "foreign", "builtin_conflict"):
            with self.subTest(change=change):
                plan = plan_fixture()
                mappings = plan["qr_mappings"]
                if change == "missing":
                    mappings.pop()
                elif change == "duplicate":
                    mappings.append(deepcopy(mappings[0]))
                elif change == "foreign":
                    mappings[0]["robot_id"] = "other"
                else:
                    mappings[0]["station_id"] = "START"
                with self.assertRaises(StationTourError):
                    validate_tour_plan(plan, mappings, robot_id="robot", available_qr_ids=QR_IDS)


if __name__ == "__main__":
    unittest.main()
