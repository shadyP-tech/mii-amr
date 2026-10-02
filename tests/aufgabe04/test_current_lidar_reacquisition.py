"""Passive retries require fresh, source-bound cohorts and unchanged admission."""

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.real_robot.candidate.current_lidar_reacquisition import (
    require_current_lidar_target,
)
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import (
    HASH_FIELD, assess_current_lidar_targets, load_current_lidar_target,
)
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
    capture_candidate_lidar_view,
)
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import (
    HASH_FIELD as CAPTURE_HASH, head_capture_payload,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from tests.aufgabe04.test_coverage_visibility_reporting import _plan
from tests.aufgabe04.test_current_lidar_targets import fixture, raw_scans


ONE_AMBIGUOUS = ["stand"] * 7 + ["ambiguous"]


class Captures:
    """Replace only sensor acquisition and time; run every artifact validator."""

    def __init__(self, tmp_path, cohorts, *, snapshot=None, wait_advance=2.0):
        source = fixture()
        self.config = SimpleNamespace(
            snapshot=snapshot or source["snapshot"],
            plan=replace(_plan(), survey_id="survey"),
            camera_calibration=SimpleNamespace(base_frame="base_footprint"),
            lidar_scan_frame="base_scan", lidar_scan_topic="/scan",
            measured_stand_model=source["stand_model"],
        )
        self.planning = source["planning_frame"]
        self.root = tmp_path / "support"
        self.cohorts = iter(cohorts)
        self.now = 100.0
        self.wait_advance = wait_advance
        self.requests, self.waits, self.events = [], [], []
        self.effects = SimpleNamespace(
            clock=lambda: self.now,
            capture_lidar_view=self.capture,
            wait_for_lidar_reacquisition=self.wait,
            event_sink=self.event,
            observe_candidate=Mock(side_effect=AssertionError("camera not admitted")),
        )

    def event(self, path, event):
        self.events.append(event)
        with path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += self.wait_advance

    def capture(self, request):
        self.requests.append(request)
        spec = next(self.cohorts)

        def acquire(_):
            scans = raw_scans(spec.get("kinds"), spec.get("shifts"))
            delta = request.observation_not_before_sec - 100.0
            for scan in scans:
                for field in ("stamp_sec", "received_at_unix_sec", "scan_pose_stamp_sec",
                              "base_pose_stamp_sec"):
                    scan[field] += delta
                scan["head_plane_mount"]["exact_transform_stamp_sec"] += delta
                for field in ("scan_pose_odom", "base_pose_odom"):
                    scan[field]["x_m"] += spec.get("base_shift", 0.0)
            raw = head_capture_payload(
                scans, tour_id=request.viewpoint_id, odom_frame="odom",
                base_frame="base_footprint", scan_frame="base_scan",
                captured_at_unix_sec=request.observation_not_before_sec + .71,
            )
            path = request.output_dir / "scan_cohort.json"
            write_content_hashed_json(path, raw, hash_field=CAPTURE_HASH)
            if spec.get("corrupt"):
                raw = json.loads(path.read_text())
                raw["scans"][0]["ranges"][50] = 1.5
                path.write_text(json.dumps(raw))
            self.now = request.observation_not_before_sec + spec.get("completion_age", .72)
            return path

        return capture_candidate_lidar_view(request, capture_cohort=acquire,
                                           clock=self.effects.clock)

    def require(self):
        return require_current_lidar_target(
            config=self.config, effects=self.effects, planning_frame=self.planning,
            candidate_uid="candidate_1", output_dir=self.root, attempt_index=4,
        )


def decision(evidence):
    return evidence["candidate_decisions"]["candidate_1"]


def test_seven_supported_scans_do_not_override_single_ambiguous_scan():
    estimates, evidence = assess_current_lidar_targets(**fixture(kinds=ONE_AMBIGUOUS))
    assert estimates == {}
    assert decision(evidence)["supported_scan_count"] == 7
    assert decision(evidence)["scan_count"] == 8
    assert decision(evidence)["reasons"] == ["ambiguous_current_target_correspondence"]


def test_fresh_second_cohort_is_replayable_without_combining_or_rewriting_first(tmp_path):
    captures = Captures(tmp_path, [dict(kinds=ONE_AMBIGUOUS), dict(shifts=[.06] * 8)])
    snapshot_before = copy.deepcopy(captures.config.snapshot)
    estimate, evidence = captures.require()

    assert captures.waits == [2.0]
    assert len(captures.requests) == 2
    assert evidence["reacquisition"]["attempt_count"] == 2
    first, second = evidence["reacquisition"]["attempts"]
    first_payload = load_content_hashed_json(Path(first["evidence_path"]), hash_field=HASH_FIELD)
    second_payload = load_content_hashed_json(Path(second["evidence_path"]), hash_field=HASH_FIELD)
    assert decision(first_payload)["supported_scan_count"] == 7
    assert decision(first_payload)["accepted"] is False
    assert decision(second_payload)["supported_scan_count"] == 8
    assert estimate["x_m"] == pytest.approx(.06)
    assert load_current_lidar_target(evidence["evidence_path"], candidate_uid="candidate_1",
                                    snapshot=captures.config.snapshot) == estimate
    assert payload_sha256(first_payload) == first["evidence_sha256"]
    assert payload_sha256(second_payload) == second["evidence_sha256"]
    assert captures.config.snapshot == snapshot_before

    assert Path(first["evidence_path"]) == captures.root / "current_lidar_targets.json"
    assert Path(second["evidence_path"]) == captures.root / "reacquire_001/current_lidar_targets.json"
    assert captures.requests[0].viewpoint_id != captures.requests[1].viewpoint_id
    assert captures.requests[1].observation_not_before_sec > max(
        scan["scan_stamp_sec"] for scan in decision(first_payload)["scans"])
    for key in ("capture_path", "capture_sha256", "candidate_lidar_view_path",
                "candidate_lidar_view_sha256", "receipt_set_sha256"):
        assert first_payload[key] != second_payload[key]
    assert first_payload["planning_frame"] == second_payload["planning_frame"]
    assert [event["retry_scheduled"] for event in captures.events] == [True, False]
    assert [event["accepted"] for event in captures.events] == [False, True]
    assert all(event["observation_attempt_index"] == 4 for event in captures.events)
    for event in captures.events:
        assert event["motion_authorized"] is False
        assert event["stand_axis_authorized"] is False
        assert event["keepouts_changed"] is False
    captures.effects.observe_candidate.assert_not_called()


def test_persistent_ambiguity_stops_after_three_independent_cohorts(tmp_path):
    captures = Captures(tmp_path, [dict(kinds=ONE_AMBIGUOUS)] * 4)
    with pytest.raises(CandidateObservationUnavailableError) as raised:
        captures.require()
    error = raised.value
    assert error.candidate_uid == "candidate_1"
    assert error.observation_attempt_index == 4
    assert error.reason == "candidate_target_ineligible"
    assert error.process_evidence == {"observer_started": False, "motion_authorized": False}
    evidence = error.status_evidence["current_lidar_support"]
    assert captures.waits == [2.0, 2.0]
    assert len(captures.requests) == 3
    assert evidence["reacquisition"]["attempt_count"] == 3
    attempts = evidence["reacquisition"]["attempts"]
    assert [a["retry_scheduled"] for a in attempts] == [True, True, False]
    assert [Path(a["evidence_path"]).parent for a in attempts] == [
        captures.root, captures.root / "reacquire_001", captures.root / "reacquire_002"]
    last_stamp = None
    for attempt in attempts:
        payload = load_content_hashed_json(Path(attempt["evidence_path"]), hash_field=HASH_FIELD)
        assert payload_sha256(payload) == attempt["evidence_sha256"]
        assert decision(payload)["accepted"] is False
        if last_stamp is not None:
            assert payload["observation_not_before_sec"] > last_stamp
        last_stamp = max(scan["scan_stamp_sec"] for scan in decision(payload)["scans"])
    captures.effects.observe_candidate.assert_not_called()


def test_successful_first_capture_retains_original_evidence_and_does_not_wait(tmp_path):
    captures = Captures(tmp_path, [{}])
    estimate, evidence = captures.require()
    assert len(captures.requests) == 1
    assert captures.waits == captures.events == []
    assert "reacquisition" not in evidence
    assert Path(evidence["evidence_path"]) == captures.root / "current_lidar_targets.json"
    assert load_current_lidar_target(evidence["evidence_path"], candidate_uid="candidate_1",
                                    snapshot=captures.config.snapshot) == estimate


@pytest.mark.parametrize("spec,reason", [
    (dict(kinds=["absent"] * 8), "insufficient_current_lidar_support"),
    (dict(kinds=["wall"] * 8), "insufficient_current_lidar_support"),
    (dict(kinds=["ambiguous"] * 8), "insufficient_current_lidar_support"),
    (dict(kinds=["stand"] * 5 + ["absent"] * 2 + ["ambiguous"]),
     "insufficient_current_lidar_support"),
    (dict(kinds=ONE_AMBIGUOUS, shifts=[-.04, .04] * 4), "current_cluster_centers_unstable"),
])
def test_other_target_rejection_reasons_are_not_retried(tmp_path, spec, reason):
    captures = Captures(tmp_path, [spec, {}])
    with pytest.raises(CandidateObservationUnavailableError) as raised:
        captures.require()
    assert reason in decision(raised.value.status_evidence["current_lidar_support"])["reasons"]
    assert len(captures.requests) == 1
    assert captures.waits == []


def test_competing_candidate_blocks_retry_even_with_supported_and_split_clusters(tmp_path):
    snapshot = fixture()["snapshot"]
    candidate = snapshot.candidates[0]
    other = replace(candidate, candidate_uid="candidate_2",
                    geometry=replace(candidate.geometry, x_m=.30),
                    source=replace(candidate.source, observation_ids=("other",)))
    snapshot = replace(snapshot, candidates=(candidate, other))
    spec = dict(kinds=["stand"] * 6 + ["ambiguous", "stand"], shifts=[0.] * 7 + [.15])
    captures = Captures(tmp_path, [spec, {}], snapshot=snapshot)
    with pytest.raises(CandidateObservationUnavailableError) as raised:
        captures.require()
    rejected = decision(raised.value.status_evidence["current_lidar_support"])
    assert rejected["supported_scan_count"] == 6
    assert rejected["reasons"] == ["ambiguous_current_target_correspondence"]
    assert {scan["reason"] for scan in rejected["scans"]} == {
        "supported", "ambiguous_clusters", "competing_candidate"}
    assert len(captures.requests) == 1
    assert captures.waits == []


@pytest.mark.parametrize("spec", [dict(completion_age=2.0), dict(corrupt=True)])
@pytest.mark.parametrize("after_ambiguity", [False, True])
def test_sensor_or_artifact_failures_remain_systemic_without_further_retry(
        tmp_path, spec, after_ambiguity):
    cohorts = ([dict(kinds=ONE_AMBIGUOUS)] if after_ambiguity else []) + [spec, {}]
    captures = Captures(tmp_path, cohorts)
    with pytest.raises(ValueError):
        captures.require()
    assert len(captures.requests) == 1 + int(after_ambiguity)
    assert len(captures.waits) == int(after_ambiguity)
    captures.effects.observe_candidate.assert_not_called()


def test_reacquisition_with_overlapping_epoch_is_rejected(tmp_path):
    # A backwards clock makes the second cohort overlap the first despite
    # independently valid timestamps and newly bound artifact/viewpoint IDs.
    captures = Captures(tmp_path, [dict(kinds=ONE_AMBIGUOUS), {}], wait_advance=-.03)
    with pytest.raises(ValueError, match="reused the prior scan epoch"):
        captures.require()
    assert len(captures.requests) == 2
    assert captures.waits == [2.0]


def test_base_movement_between_cohorts_is_not_hidden_by_new_stationary_capture(tmp_path):
    captures = Captures(tmp_path, [dict(kinds=ONE_AMBIGUOUS), dict(base_shift=.02)])
    with pytest.raises(ValueError, match="base pose differs from planning start"):
        captures.require()
    assert len(captures.requests) == 2
    assert captures.waits == [2.0]
    assert not (captures.root / "reacquire_001/current_lidar_targets.json").exists()
    captures.effects.observe_candidate.assert_not_called()
