"""A detour consumes a new bounded slot and preserves the real stopped child."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.execution.mission_leg_motion_consumption import (
    default_mission_leg_motion_consumption_receipt_path,
    mission_leg_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, file_sha256,
    load_mission_leg_motion_authorization, mission_leg_motion_authorization_sha256,
    mission_leg_motion_permit_sha256, write_mission_leg_motion_authorization,
    write_mission_leg_motion_permit,
)
from scripts.aufgabe04.navigation.execution.tour_replan_binding import (
    is_replannable_tour_stop, tour_mission_leg_index, validate_tour_navigation,
    validate_tour_obstacle_monitor_admission, write_tour_terminal_evidence,
)
from scripts.aufgabe04.navigation.execution.tour_terminal_evidence import load_tour_terminal_evidence
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import capture_payload
from tests.aufgabe04 import test_stored_pose_tour_authorization as fixtures


class TourReplanBindingTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StoredPoseTourAuthorizationTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.fixture.master = replace(self.fixture.master, scope_text=TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE)
        self.fixture.master_path = self.root / "dynamic-tour-master.json"
        self.fixture.master_hash = write_mission_leg_motion_authorization(self.fixture.master_path, self.fixture.master)

    def _permit(self, *, stage=0, replans=0, visit=0, final=True, previous=None, uid="candidate-work", qr="Werkbank"):
        permit, path = self.fixture._permit(visit=visit, stage=stage, final=final, uid=uid, qr=qr)
        nav = {"execution_index": stage + replans, "stage_index": stage, "replan_count": replans,
               "previous_terminal_json": str(previous[0]) if previous else "",
               "previous_terminal_sha256": previous[1] if previous else ""}
        diagnostics_path = Path(permit.diagnostics_path)
        payload = json.loads(diagnostics_path.read_text())
        metadata = payload["metadata"]
        target_path = Path(metadata["target_evidence_json"])
        target = json.loads(target_path.read_text())
        target["tour_navigation"] = nav
        target_path.write_text(json.dumps(target))
        overlay_path = self.root / f"{permit.run_id}-overlay.json"
        sources = [{"path": "fixture"}]
        if previous:
            previous_payload = json.loads(previous[0].read_text())
            if previous_payload["scan_capture_json"]:
                sources = [{"path": previous_payload["scan_capture_json"], "sha256": previous_payload["scan_capture_sha256"]}]
        write_content_hashed_json(overlay_path, {
            "schema_version": 1, "artifact_kind": "stored_pose_tour_temporary_obstacle_overlay",
            "tour_id": permit.session_id, "planning_frame_admission": None,
            "scan_frame": "base_scan", "capture_sources": sources,
        }, hash_field="temporary_obstacle_overlay_sha256")
        metadata.update(tour_navigation=nav, target_evidence_sha256=file_sha256(target_path),
            temporary_obstacle_overlay_json=str(overlay_path), temporary_obstacle_overlay_sha256=file_sha256(overlay_path))
        diagnostics_path.write_text(json.dumps(payload))
        return replace(permit, mission_leg_index=tour_mission_leg_index(visit, stage + replans),
                       diagnostics_sha256=file_sha256(diagnostics_path)), path

    def _terminal(self, permit, path, *, stopped=True, stage=0, replans=0, final=True):
        receipt = self.fixture._consume(permit, path)
        status = "stopped" if stopped else "completed"
        details = {"source": "global_scan", "nearest_valid_range_m": .16, "threshold_m": .2, "valid_sample_count": 7} if stopped else {}
        reason = "obstacle too close" if stopped else "route completed"
        identity = {"run_id": permit.run_id, "mission_leg_kind": "stored_pose_tour", "mission_leg_index": permit.mission_leg_index, "target_id": permit.target_id}
        events = [
            {**identity, "event": "mission_leg_motion_permit_consumed", "mission_leg_motion_permit_sha256": mission_leg_motion_permit_sha256(permit),
             "mission_leg_motion_consumption_receipt_sha256": mission_leg_motion_consumption_receipt_sha256(receipt)},
            {**identity, "event": "motion_started", "motion_published": False, "event_semantics": "child_execution_attempt_started_before_follower"},
            {**identity, "event": "safety_stop" if stopped else "motion_completed", "status": status,
             "stop_reason": reason, "stop_details": details, "motion_published": True},
            {"run_id": permit.run_id, "event": "run_finished", "final_status": status,
             "timestamp": datetime.fromtimestamp(100, timezone.utc).isoformat()},
        ]
        log = self.root / f"{permit.run_id}-events.jsonl"
        log.write_text("".join(json.dumps(event) + "\n" for event in events))
        outcome = SimpleNamespace(run_id=permit.run_id, status=status, stop_reason=reason, stop_details=details,
                                  motion_published=True, returncode=1 if stopped else 0, semantic_log_path=log, semantic_log_start_offset=0)
        arrival_path = None
        capture_path = None
        if stopped:
            pose = {"x_m": .1, "y_m": .2, "yaw_rad": 0.}
            scans = [{"stamp_sec": stamp, "received_at_unix_sec": stamp + .01,
                "scan_pose_stamp_sec": stamp, "base_pose_stamp_sec": stamp,
                "scan_pose_odom": pose, "base_pose_odom": pose,
                "angle_min": -3.14, "angle_increment": 1.57, "range_min": .1, "range_max": 3., "ranges": [.5] * 5,
            } for stamp in (100.1, 100.2, 100.3)]
            capture_path = self.root / f"{permit.run_id}-scan.json"
            write_content_hashed_json(capture_path, capture_payload(scans, tour_id=permit.session_id,
                odom_frame="odom", base_frame="base_link", scan_frame="base_scan", captured_at_unix_sec=100.32),
                hash_field="tour_scan_capture_sha256")
        else:
            arrival_path = self.root / f"{permit.run_id}-arrival.json"
            write_content_hashed_json(arrival_path, {"artifact_kind": "stored_pose_tour_leg_arrival", "tour_id": permit.session_id,
                "visit_index": permit.mission_leg_index // 6, "stage_index": stage, "run_id": permit.run_id, "final_stage": final,
                "status": status, "returncode": 0, "motion_published": True, "arrival_verified": True,
                "candidate_uid": permit.target_id, "position_error_m": .01, "heading_error_rad": .02, "odom_progress_m": .4,
            }, hash_field="stored_pose_tour_leg_arrival_sha256")
        request = SimpleNamespace(**{name: getattr(permit, name) for name in ("run_id", "session_id", "mission_leg_kind", "mission_leg_index", "target_id")}, permit_json_path=path)
        result = write_tour_terminal_evidence(self.root / f"{permit.run_id}-terminal.json", outcome=outcome, request=request,
            stage_index=stage, replan_count=replans, final_stage=final, arrival_path=arrival_path, scan_capture_path=capture_path)
        return result, outcome

    def test_blocked_planned_final_can_use_two_new_permits_without_reusing_spent_slots(self):
        first, outcome = self._terminal(*self._permit())
        self.assertEqual(load_tour_terminal_evidence(first[0])["status"], "stopped")
        second, _ = self._terminal(*self._permit(replans=1, previous=first), replans=1)
        permit, path = self._permit(replans=2, previous=second)
        self.fixture._consume(permit, path)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.fixture._consume(*self._permit(replans=2, previous=second))
        with self.assertRaisesRegex(ValueError, "bounded budget"):
            write_mission_leg_motion_permit(*reversed(self._permit(replans=3, previous=second)))
        self.assertTrue(is_replannable_tour_stop(outcome))

    def test_completed_intermediate_requires_arrival_and_advances_stage_only(self):
        prior, _ = self._terminal(*self._permit(final=False), stopped=False, final=False)
        self.fixture._consume(*self._permit(stage=1, previous=prior))
        with self.assertRaisesRegex(ValueError, "counters"):
            write_mission_leg_motion_permit(*reversed(self._permit(replans=1, previous=prior)))

    def test_final_arrival_never_authorizes_continuation(self):
        prior, _ = self._terminal(*self._permit(), stopped=False)
        with self.assertRaisesRegex(ValueError, "genuine final arrival"):
            write_mission_leg_motion_permit(*reversed(self._permit(stage=1, previous=prior)))

    def test_terminal_log_can_grow_but_original_bytes_cannot_change(self):
        prior, outcome = self._terminal(*self._permit())
        with outcome.semantic_log_path.open("a") as stream:
            stream.write(json.dumps({"run_id": "other", "event": "run_started"}) + "\n")
        load_tour_terminal_evidence(prior[0])
        outcome.semantic_log_path.write_text(outcome.semantic_log_path.read_text().replace("global_scan", "global_scam"))
        with self.assertRaisesRegex(ValueError, "source log bytes changed"):
            load_tour_terminal_evidence(prior[0])

    def test_distinct_target_or_unconsumed_prior_cannot_continue(self):
        prior, _ = self._terminal(*self._permit())
        with self.assertRaisesRegex(ValueError, "same-target execution"):
            write_mission_leg_motion_permit(*reversed(self._permit(replans=1, previous=prior, uid="other", qr="Other")))
        prior_payload = load_tour_terminal_evidence(prior[0])
        Path(prior_payload["receipt_json"]).unlink()
        with self.assertRaises(ValueError):
            write_mission_leg_motion_permit(*reversed(self._permit(replans=1, previous=prior)))

    def test_monitor_dry_and_live_require_exact_flag_scope_overlay_and_frame(self):
        permit, _ = self._permit()
        metadata = json.loads(Path(permit.diagnostics_path).read_text())["metadata"]
        args = SimpleNamespace(stored_pose_tour_obstacle_monitor=True, stored_pose_tour_scan_frame="base_scan",
                               allow_sim_time=False, dry_run=True, mission_leg_motion_authorization_json=None)
        validate_tour_obstacle_monitor_admission(args, metadata)
        args.dry_run = False
        with self.assertRaisesRegex(ValueError, "explicit tour authorization"):
            validate_tour_obstacle_monitor_admission(args, metadata)
        args.mission_leg_motion_authorization_json = self.fixture.master_path
        validate_tour_obstacle_monitor_admission(args, metadata)
        args.stored_pose_tour_scan_frame = "wrong_scan"
        with self.assertRaisesRegex(ValueError, "scan frame"):
            validate_tour_obstacle_monitor_admission(args, metadata)
        args.stored_pose_tour_scan_frame = "base_scan"
        args.stored_pose_tour_obstacle_monitor = False
        with self.assertRaisesRegex(ValueError, "sealed dynamic tour route"):
            validate_tour_obstacle_monitor_admission(args, metadata)

    def test_generic_failures_never_become_detour_authority(self):
        for reason in ("stale scan", "odom stuck", "unexpected velocity publisher", "stored pose tour obstacle monitor unavailable", "localization invalid"):
            self.assertFalse(is_replannable_tour_stop(SimpleNamespace(status="stopped", returncode=1, stop_reason=reason, stop_details={})))
        with self.assertRaisesRegex(ValueError, "stage_index"):
            validate_tour_navigation({"execution_index": 4, "stage_index": 4, "replan_count": 0,
                                      "previous_terminal_json": "x", "previous_terminal_sha256": "a"*64})
        with self.assertRaisesRegex(ValueError, "six-execution"):
            tour_mission_leg_index(0, 6)

    def test_new_scope_requires_navigation_and_old_scope_cannot_gain_detours(self):
        permit, path = self.fixture._permit()
        with self.assertRaisesRegex(ValueError, "tour_navigation fields"):
            write_mission_leg_motion_permit(path, permit)
        permit, path = self._permit()
        legacy_path = self.root / "tour-master.json"
        legacy_master = load_mission_leg_motion_authorization(legacy_path)
        legacy_permit = replace(permit, master_authorization_path=str(legacy_path),
            master_authorization_sha256=mission_leg_motion_authorization_sha256(legacy_master))
        with self.assertRaisesRegex(ValueError, "legacy tour scope"):
            write_mission_leg_motion_permit(path, legacy_permit)
        args = SimpleNamespace(stored_pose_tour_obstacle_monitor=True, stored_pose_tour_scan_frame="base_scan",
            allow_sim_time=False, dry_run=False, mission_leg_motion_authorization_json=legacy_path)
        with self.assertRaisesRegex(ValueError, "legacy tour scope"):
            validate_tour_obstacle_monitor_admission(args, json.loads(Path(permit.diagnostics_path).read_text())["metadata"])

    def test_rehashing_terminal_as_completed_does_not_change_original_stopped_event(self):
        prior, _ = self._terminal(*self._permit())
        payload = json.loads(prior[0].read_text())
        payload.pop("tour_terminal_evidence_sha256")
        payload.update(status="completed", returncode=0, stop_details={})
        path = self.root / "forged-completed.json"
        write_content_hashed_json(path, payload, hash_field="tour_terminal_evidence_sha256")
        with self.assertRaisesRegex(ValueError, "original child event"):
            load_tour_terminal_evidence(path)

    def test_detour_must_plan_with_exact_post_stop_capture(self):
        prior, _ = self._terminal(*self._permit())
        permit, path = self._permit(replans=1, previous=prior)
        diagnostics_path = Path(permit.diagnostics_path)
        diagnostics = json.loads(diagnostics_path.read_text())
        original_overlay = Path(diagnostics["metadata"]["temporary_obstacle_overlay_json"])
        overlay = json.loads(original_overlay.read_text())
        overlay.pop("temporary_obstacle_overlay_sha256")
        overlay["capture_sources"] = [{"path": "different-capture", "sha256": "a" * 64}]
        altered_overlay = self.root / "different-overlay.json"
        write_content_hashed_json(altered_overlay, overlay, hash_field="temporary_obstacle_overlay_sha256")
        diagnostics["metadata"].update(temporary_obstacle_overlay_json=str(altered_overlay), temporary_obstacle_overlay_sha256=file_sha256(altered_overlay))
        diagnostics_path.write_text(json.dumps(diagnostics))
        permit = replace(permit, diagnostics_sha256=file_sha256(diagnostics_path))
        with self.assertRaisesRegex(ValueError, "authenticated post-stop scan"):
            write_mission_leg_motion_permit(path, permit)


if __name__ == "__main__":
    unittest.main()
