"""Recorded LDS endpoint fragments require independent, recomputable witnesses."""

import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import pytest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.perception.scan_topology import ScanTopology
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext, StoppedScanTargetPersistence, registered_target_is_unique,
    registered_target_metadata_is_unique, validated_witnessed_fragmentation,
)
from scripts.aufgabe04.real_robot.observer.scan_endpoint_fragments import ENDPOINT_WITNESS_KIND
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose, PassiveObserverEvidence
from scripts.aufgabe04.real_robot.observer.head_observation_window import CurrentHeadWindowInput, review_current_head_window
from scripts.aufgabe04.real_robot.observer.head_temporal_consistency import StationaryHeadConsistency
from scripts.aufgabe04.real_robot.observer.head_observation_confidence import HeadObservationConfidence, HeadConfidenceInput
from scripts.aufgabe04.perception.stand_axis.head_backside_appearance import HeadBacksideAppearance
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.artifacts.backside_axis_observation import REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE


ROWS = {r["index"]: r for r in json.loads((Path(__file__).parent /
    "fixtures/scan_endpoint_20260918.json").read_text())["rows"]}


def inputs(index):
    row = ROWS[index]
    scan = PlainLaserScan(**{**row["scan"], "ranges": tuple(
        math.nan if v is None else v for v in row["scan"]["ranges"])})
    context = ScanPersistenceContext(**{**row["context"], **{
        k: Pose2D(**row["context"][k]) for k in ("robot_pose", "scan_pose_map", "scan_pose_robot")}})
    return scan, context, dict(row["parameters"])


def resolve(state, index, *, scan=None, context=None, parameters=None, preview=False):
    original_scan, original_context, original_parameters = inputs(index)
    scan, context = scan or original_scan, context or original_context
    parameters = parameters or original_parameters
    raw = associate_camera_registered_candidate_lidar_target(scan, **parameters)
    result = (state.preview if preview else state.resolve)(raw, scan, context=context,
        now_sec=parameters["now_sec"], max_scan_age_sec=parameters["max_scan_age_sec"])
    return result, raw


def seed():
    state = StoppedScanTargetPersistence()
    # Ambiguous frames between unique witnesses must not count as witnesses,
    # nor erase compatible evidence solely because the third scan is pending.
    for index in range(52, 57):
        resolve(state, index)
    return state


def test_recorded_sequence_resolves_only_after_three_valid_real_witnesses():
    state = StoppedScanTargetPersistence()
    accepted = []
    for index in range(50, 63):
        result, raw = resolve(state, index)
        if result.associated:
            accepted.append(index)
        if index in (50, 51, 53, 55):
            assert not result.associated
        if index == 60:
            assert not raw.associated
            assert registered_target_is_unique(result)
            assert result.search_association.eligible_cluster_count == 2
            assert result.search_association.selected_cluster_wraps_scan_seam
            assert not result.search_association.scan_topology["circular_adjacency_enabled"]
            proof = json.loads(json.dumps(result.witnessed_fragmentation, allow_nan=False))
            assert proof["kind"] == ENDPOINT_WITNESS_KIND
            assert len(proof["witnesses"]) == 3
            assert [w["scan"]["scan_stamp_sec"] for w in proof["witnesses"]] == [
                ROWS[i]["scan"]["scan_stamp_sec"] for i in (52, 54, 56)]
            assert validated_witnessed_fragmentation(proof) == result
            assert registered_target_metadata_is_unique(json.loads(json.dumps(asdict(result))))
    assert accepted == [52, 54, 56, 57, 58, 59, 60, 61]
    # The three old witnesses expire; resolved fragments never renew them.
    assert not resolve(state, 62)[0].associated


def test_original_topology_and_valid_single_cluster_are_unchanged():
    scan, _, _ = inputs(50)
    assert not ScanTopology(len(scan.ranges), scan.angle_min, scan.angle_increment,
                            scan.angle_max, "full_rotation").joins_endpoints(len(scan.ranges)-1, 0)
    result, raw = resolve(StoppedScanTargetPersistence(), 93)
    assert result == raw and result.associated and result.witnessed_fragmentation is None


@pytest.mark.parametrize("corrected", [False, True])
def test_recorded_angles_borders_and_backside_confidence_reach_seven_only_with_fix(corrected):
    state = StoppedScanTargetPersistence()
    _, initial, _ = inputs(50)
    evidence = PassiveObserverEvidence(target_key=initial.target_key,
        anchor_pose=EvidencePose(**asdict(initial.robot_pose)), required_axis_samples=7,
        max_axis_deviation_rad=math.radians(8.), axis_ttl_sec=5.)
    window = StationaryHeadConsistency(required_samples=7, max_axis_span_rad=math.radians(8.), window_ttl_sec=5.)
    confidence = HeadObservationConfidence(required_samples=7, ttl_sec=5.)
    for index in range(50, 62):
        row = ROWS[index]
        scan, context, parameters = inputs(index)
        result, raw = resolve(state, index)
        associated = registered_target_is_unique(result if corrected else raw)
        assert row["quality"]["accepted"]
        corners = tuple((p["u_px"], p["v_px"]) for p in row["appearance"]["corners"])
        current = CurrentHeadWindowInput(context.image_stamp_sec, row["profile_sha256"],
            tuple(row["camera_signature"]), corners, tuple(row["projected_center_px"]),
            row["expected_head_height_px"], row["camera_yaw_rad"], (row["camera_yaw_rad"],))
        update = evidence.record_frame(target_key=context.target_key,
            pose=EvidencePose(**asdict(context.robot_pose)), frame_stamp_sec=context.image_stamp_sec,
            lidar_stamp_sec=scan.scan_stamp_sec, observed_at_sec=parameters["now_sec"],
            lidar_associated=associated, axis_yaw_rad=row["camera_yaw_rad"],
            axis_source=(REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE if row["appearance"]["accepted"]
                         else "model_current_measured_head"),
            axis_window_review=lambda snapshot: review_current_head_window(window, current, snapshot=snapshot)[0])
        appearance = HeadBacksideAppearance(**{**row["appearance"],
            "corners": tuple(ImagePoint(*p) for p in corners)})
        conf = confidence.observe(HeadConfidenceInput(context.image_stamp_sec, appearance,
            associated, False, False, True, "recorded_quality_accepted"),
            update=update, observed_at_sec=parameters["now_sec"])
    assert (update.axis_consensus is not None) == corrected
    assert (conf["backside"]["state"] == "backside_supported") == corrected
    assert update.snapshot.current_axis_sample_count == (7 if corrected else 3)


def test_preview_isolated_and_proof_cannot_be_promoted_or_tampered():
    state = seed()
    history = copy.deepcopy(state._history)
    result, _ = resolve(state, 57, preview=True)
    assert result.associated
    result.witnessed_fragmentation["witnesses"][0]["scan"]["ranges"][0] = 99.
    assert state._history == history
    assert resolve(state, 57)[0].associated


@pytest.mark.parametrize("mutation", [
    lambda p: p["witnesses"].pop(),
    lambda p: p["witnesses"].__setitem__(1, copy.deepcopy(p["witnesses"][0])),
    lambda p: p["witnesses"][0]["context"].update(epoch_key="other"),
    lambda p: p["witnesses"][0]["context"].update(target_key="other"),
    lambda p: p["witnesses"][0]["context"]["robot_pose"].update(x_m=9.),
    lambda p: p["witnesses"][0]["scan"].update(scan_topology_profile="linear"),
    lambda p: p["witnesses"][0]["scan"].update(angle_max=5.),
    lambda p: p["witnesses"][0]["parameters"].update(observed_camera_bearing_rad=.8),
    lambda p: p["current"].update(now_sec=p["current"]["now_sec"]+1.),
    lambda p: p["current"]["context"].update(image_stamp_sec=1.),
    lambda p: p["current"]["scan"].update(receipt_sec=1.),
    lambda p: p["current"]["scan"].update(angle_max=5.),
    lambda p: p["current"]["scan"].update(scan_topology_profile="linear"),
    lambda p: p.update(kind="one_internal_missing_beam_witnessed"),
    lambda p: p.update(kind=[]),
])
def test_invalid_proof_rejected(mutation):
    result, _ = resolve(seed(), 57)
    proof = copy.deepcopy(result.witnessed_fragmentation)
    mutation(proof)
    with pytest.raises(ValueError):
        validated_witnessed_fragmentation(proof)


@pytest.mark.parametrize("change", ["range_jump", "third_cluster", "missing_endpoint", "partial_scan", "wrong_epoch", "motion", "stale"])
def test_contradiction_rejects_and_consumes_history(change):
    state = seed()
    scan, context, parameters = inputs(57)
    ranges = list(scan.ranges)
    if change == "range_jump":
        ranges[0] += .08
    elif change == "third_cluster":
        ranges[1] = .46  # In the same cone/range gate, but a separate object.
    elif change == "missing_endpoint":
        ranges[0] = math.nan
    elif change == "partial_scan":
        scan = replace(scan, angle_max=5.)
    elif change == "wrong_epoch":
        context = replace(context, epoch_key="other")
    elif change == "motion":
        context = replace(context, robot_pose=replace(context.robot_pose, x_m=context.robot_pose.x_m+.02))
    else:
        parameters["now_sec"] += 1.
    scan = replace(scan, ranges=tuple(ranges))
    result, _ = resolve(state, 57, scan=scan, context=context, parameters=parameters)
    assert not result.associated
    assert not resolve(state, 58)[0].associated


def test_real_endpoint_scan_witnesses_can_arrive_without_camera_success():
    state = StoppedScanTargetPersistence()
    for index in (52, 54, 56):
        scan, context, parameters = inputs(index)
        # Bind independent scan witnesses to their exact scan-time pose.
        ext = context.scan_pose_robot
        yaw = context.scan_pose_map.yaw_rad-ext.yaw_rad
        pose = Pose2D(context.scan_pose_map.x_m-math.cos(yaw)*ext.x_m+math.sin(yaw)*ext.y_m,
                      context.scan_pose_map.y_m-math.sin(yaw)*ext.x_m-math.cos(yaw)*ext.y_m, yaw)
        context = replace(context, robot_pose=pose, image_stamp_sec=scan.scan_stamp_sec)
        assert state.ingest_scan(scan, context=context, now_sec=parameters["now_sec"], max_scan_age_sec=.5)
    assert registered_target_is_unique(resolve(state, 57)[0])
