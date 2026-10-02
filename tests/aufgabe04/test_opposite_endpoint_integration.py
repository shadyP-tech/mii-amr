"""Head-only runtime admission and historical endpoint receipt compatibility.

Runtime cases detect the head from original pixels; only the decoded payload is
stubbed to remove desktop backend differences. Historical QR-corner receipts
are constructed explicitly through legacy producers, never the live pipeline.
"""
from dataclasses import asdict, replace
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
    HASH_FIELD, validate_qr_verified_observation_pose,
)
from scripts.aufgabe04.artifacts.retained_facing import build_retained_facing
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    load_recommendation, recommendation_to_dict,
)
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.camera_calibration import CameraCalibration
from scripts.aufgabe04.real_robot.configuration.profile import camera_calibration_sha256
from scripts.aufgabe04.real_robot.observer import opposite_identity
from scripts.aufgabe04.real_robot.observer import opposite_raw_qr
from scripts.aufgabe04.real_robot.observer.opposite_endpoint_confirmation import (
    build_opposite_endpoint_hint, confirm_opposite_endpoint, validate_opposite_endpoint,
)
from scripts.aufgabe04.real_robot.observer.qr_observation_binding import load_bound_qr_observation_pose
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import commit_qr_observation_pose, prepare_qr_observation_pose
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop, bind_crop_text
from scripts.aufgabe04.real_robot.observer.opposite_target_support import (
    OppositeQrOutline, support_opposite_qr_outline, validate_target_support,
)
from scripts.aufgabe04.real_robot.observer.opposite_head_support import HEAD_REGION_POLICY, HEAD_POLICY
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.target_reconciliation import (
    StoppedTargetReconciliation, validate_reconciliation,
)
from tests.aufgabe04 import test_camera_observer_processing as observer_fixtures
from tests.aufgabe04 import test_projected_retained_facing as projected_fixtures
from tests.aufgabe04.opposite_endpoint_fixture import recorded_endpoint, rectified_endpoint_image


IMAGE = Path(__file__).with_name("fixtures") / "opposite_endpoint_20261001.jpg"


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    case = recorded_endpoint(tmp_path)
    clock = observer_fixtures.CameraObserverProcessingTest()
    adapter = clock.make_adapter()
    row, orientation = case["row"], case["orientation"]
    clock.clock_sec = case["now_sec"]
    adapter.cv2 = cv2
    adapter.calibration = case["calibration_profile"]
    assert camera_calibration_sha256(adapter.calibration) == orientation["calibration_profile_sha256"]
    adapter.stand_model_profile = case["model"]
    adapter.args.stand_id = row["candidate_uid"]
    adapter.args.stream_id = row["target_key"].split(":" + row["candidate_uid"] + ":", 1)[0]
    adapter.args.stand_x, adapter.args.stand_y = row["stand_center"]
    adapter.args.candidate_crop_snapshot = case["snapshot_path"]
    adapter.args.stand_model_profile = case["model_path"]
    adapter.args.qr_observation_pose_json = tmp_path / "qr_observation_pose.json"
    adapter.args.recommended_pose_json = tmp_path / "recommendation.json"
    adapter.args.status_json = tmp_path / "observer_status.json"
    adapter.args.candidate_centering_json = None
    adapter.profile.map_frame = "map"
    adapter.profile.base_frame = "base_footprint"
    adapter.profile.scan_frame = "base_scan"
    adapter.profile.camera_optical_frame = "camera"
    assert adapter._target_evidence_key() == row["target_key"]
    adapter._capture_pending = {}
    tracker = adapter._target_reconciliation = StoppedTargetReconciliation()
    calls = []

    def reconcile(fragmentation):
        calls.append(fragmentation)
        return tracker.observe(**{**row, "robot_pose": tuple(asdict(case["robot_pose"]).values()),
            "now_sec": clock.clock_sec}, fragmentation=fragmentation)

    base_camera = next(t for t in case["metadata"]["tf_samples"]
        if t["target_frame"] == "base_footprint" and t["source_frame"] == "camera")
    intrinsics = case["intrinsics"]
    kwargs = dict(context=SimpleNamespace(orientation=orientation, snapshot=case["snapshot"]),
        frame=rectified_endpoint_image(case, IMAGE), intrinsics=intrinsics,
        raw_frame=cv2.imread(str(IMAGE)), camera_calibration=case["calibration"],
        robot_pose=case["robot_pose"], image_stamp_sec=row["image_stamp_sec"], scan=case["scan"],
        camera_signature=(intrinsics.fx_px, intrinsics.fy_px, intrinsics.cx_px, intrinsics.cy_px),
        scan_from_map=case["scan_from_map"], camera_from_map=case["camera_from_map"],
        map_bearing_rad=row["options"]["map_bearing_rad"], accepted_range_m=row["options"]["accepted_range_m"],
        scan_from_camera=case["scan_from_camera"],
        base_from_camera=RigidTransform("base_footprint", "camera", tuple(base_camera["translation_xyz_m"]),
            tuple(base_camera["rotation_xyzw"])), image_stamp=None,
        target_reconciliation=None, require_target_reconciliation=True,
        persistence_context=case["persistence_context"], reconcile_target=reconcile)
    # Real current head detection uses original pixels. These are structural
    # receipt tests, not latency tests: allow its cooperative budget enough
    # room for a cold desktop backend and stub only the cornerless payload.
    detect = opposite_identity.detect_opposite_head_region
    def detect_head(*args, **options):
        return detect(*args, **{**options, 'max_elapsed_sec': .5})
    monkeypatch.setattr(opposite_identity, 'detect_opposite_head_region', detect_head)
    adapter._test_decoder_modes = []
    def decode_identity(*args, **options):
        adapter._test_decoder_modes.append(options['identity_only'])
        return (DecodedQrObservation('Start', None, 'payload_only_fixture', 4.),)
    monkeypatch.setattr(opposite_identity, 'detect_qr_observations_bgr', decode_identity)
    # The lightweight adapter supplies the recorded robot digest. Camera
    # calibration is the authentic, sealed profile and its real hash is checked.
    monkeypatch.setattr("scripts.aufgabe04.real_robot.observer.qr_observation_pose.real_robot_profile_sha256",
        lambda _: orientation["robot_profile_sha256"])
    return case, adapter, clock, kwargs, calls


def _legacy_crop(endpoint):
    """Construct old QR-based proof/crop directly for reader compatibility."""
    case, adapter, _, kwargs, _ = endpoint
    hint, seed = build_opposite_endpoint_hint(**case['hint_kwargs'])
    value = case['outline']
    outline = OppositeQrOutline(tuple(tuple(p) for p in value['corners_px']),
        tuple(value['center_px']), value['image_stamp_sec'], tuple(value['image_shape']),
        value['expected_symbol_height_px'])
    fragment = confirm_opposite_endpoint(seed, outline, now_sec=case['now_sec'])
    proof = StoppedTargetReconciliation().observe(**case['row'], fragmentation=fragment)
    options = dict(case['row']['options'], scan=case['scan'], scan_from_map=case['scan_from_map'],
        camera_from_map=case['camera_from_map'], intrinsics=case['intrinsics'], model_profile=case['model'],
        image_stamp_sec=case['row']['image_stamp_sec'], sync_tolerance_sec=.1,
        target_reconciliation=proof, fragmentation=fragment, now_sec=case['now_sec'], max_scan_age_sec=.5)
    search = current_scan_qr_search(**options)
    support_options = {k: v for k, v in options.items()
        if k not in ('scan_from_map', 'camera_from_map', 'sync_tolerance_sec')}
    support = support_opposite_qr_outline(outline, attempt=search[0],
        image_shape=kwargs['frame'].shape[:2], scan_from_camera=case['scan_from_camera'], **support_options)
    assert support is not None
    attempt, crop = exclusive_identity_crop(candidate_uid=adapter.args.stand_id,
        snapshot=case['snapshot'], support=support, search_result=search, **options)
    assert attempt is not None and crop['sampling'] == 'isolated_current_qr_quad'
    return attempt, crop, support, proof


def _legacy_decode(endpoint, crop, support, **overrides):
    case, adapter, _, kwargs, _ = endpoint
    options = dict(calibration=case['calibration'], calibration_profile=adapter.calibration,
        intrinsics=case['intrinsics'], support=support, crop=crop,
        image_stamp_sec=case['row']['image_stamp_sec'], max_elapsed_sec=.12)
    options.update(overrides)
    return opposite_raw_qr.decode_opposite_raw_qr(kwargs['raw_frame'], cv2, **options)


def _legacy_payload(endpoint):
    """Seal an old-style receipt without invoking current camera processing."""
    case, adapter, _, kwargs, _ = endpoint
    attempt, crop, support, proof = _legacy_crop(endpoint)
    observations = _legacy_decode(endpoint, crop, support)
    assert len(observations) == 1 and observations[0].text == 'Start'
    binding = bind_crop_text(observations, crop)
    adapter._pending_qr_observation_pose = prepare_qr_observation_pose(qr_binding=binding,
        qr_observations=observations, observed_qr_texts=('Start',),
        image_stamp_sec=case['row']['image_stamp_sec'], scan_stamp_sec=case['scan'].scan_stamp_sec,
        robot_pose=case['robot_pose'], target_key=adapter._target_evidence_key(),
        camera_signature=kwargs['camera_signature'], image_shape=kwargs['frame'].shape,
        roi=attempt.roi, model_profile_sha256=case['model'].sha256, metadata={},
        retained_backside_orientation=case['orientation'], arrival_target_reconciliation=proof)
    adapter._record_observation_frame(robot_pose=case['robot_pose'],
        image_stamp_sec=case['row']['image_stamp_sec'], scan_stamp_sec=case['scan'].scan_stamp_sec,
        observed_at_sec=case['now_sec'], lidar_associated=True, axis_yaw_rad=None, axis_source=None,
        qr_texts=binding.qr_texts_for_evidence, qr_symbol_count=1)
    assert commit_qr_observation_pose(adapter) is not None
    return deepcopy(_bound_load(case, adapter))


def _bound_load(case, adapter):
    return load_bound_qr_observation_pose(adapter.args.qr_observation_pose_json,
        candidate_uid=adapter.args.stand_id, stream_id=adapter.args.stream_id,
        planning_frame="map", stand_x_m=adapter.args.stand_x, stand_y_m=adapter.args.stand_y,
        stand_model_profile_sha256=case["model"].sha256,
        robot_profile_sha256=case["orientation"]["robot_profile_sha256"],
        calibration_profile_sha256=case["orientation"]["calibration_profile_sha256"],
        base_frame="base_footprint", scan_frame="base_scan", camera_frame="camera")


def test_original_opposite_frame_commits_current_qr_and_roundtrips_retained_geometry(endpoint, tmp_path):
    case, adapter, _, kwargs, calls = endpoint
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert len(calls) == 1
    assert adapter._test_decoder_modes == [True]
    assert calls[0]["kind"] == "retained_opposite_current_qr_endpoint_confirmation"
    assert calls[0]['outline']['policy'] == HEAD_REGION_POLICY
    proof = adapter._current_position_epoch_proof
    validate_reconciliation(proof, candidate_uid=adapter.args.stand_id,
        image_stamp_sec=case["row"]["image_stamp_sec"], scan_stamp_sec=case["scan"].scan_stamp_sec)
    assert len(proof["entries"]) == 1
    assert adapter._pending_qr_observation_pose.qr_binding.accepted
    assert adapter._pending_qr_observation_pose.qr_binding.qr_texts_for_evidence == ("Start",)
    # One current identity sample reuses the certified seven-sample angle;
    # no new angle consensus is fabricated or requested.
    state = adapter.observation_evidence.snapshot()
    assert state.current_axis_sample_count == 0 and state.current_qr_sample_count == 1
    assert adapter._capture_pending["detector_metadata"]["current_angle_refit"] is False
    committed = commit_qr_observation_pose(adapter)
    assert committed is not None and adapter.completed
    qr = _bound_load(case, adapter)
    assert qr["qr_id"] == "Start" and qr["motion_authorized"] is False and qr["facing_ready"] is False
    assert qr["retained_backside_orientation"] == case["orientation"]
    crop = qr['qr_binding']['current_head_binding']
    assert crop['sampling'] == 'masked_current_head_region'
    assert crop['target_support']['policy'] == HEAD_POLICY
    assert 'raw_pixel_binding' not in crop and qr['qr_corners_px'] is None
    assert qr["arrival_target_reconciliation"]["entries"][0]["scan"]["scan_stamp_sec"] == case["scan"].scan_stamp_sec
    rec = build_retained_facing(adapter.args.qr_observation_pose_json,
        stand_radius_m=case["snapshot"].candidate_for(adapter.args.stand_id).geometry.radius_m, target_distance_m=.4)
    assert rec.axis_sample_count == case["orientation"]["axis_sample_count"] == 7
    assert rec.bounded_orientation == case["orientation"]["bounded_orientation"]
    assert rec.stand.uncertainty_m == case["orientation"]["validated_target_center"]["uncertainty_m"]
    adapter.args.recommended_pose_json.write_text(json.dumps(recommendation_to_dict(rec)))
    assert load_recommendation(adapter.args.recommended_pose_json) == rec
    projected, original, _, _ = projected_fixtures.projected.__wrapped__(
        (adapter.args.qr_observation_pose_json, qr, case["snapshot"]), tmp_path)
    assert projected.schema_version == 5 and projected.axis_sample_count == 7
    assert projected.bounded_orientation["half_width_rad"] == original.bounded_orientation["half_width_rad"]
    path = tmp_path / "projected.json"
    path.write_text(json.dumps(recommendation_to_dict(projected)))
    assert load_recommendation(path) == projected


def test_missing_current_head_allows_search_but_cannot_bind_cornerless_identity(endpoint, monkeypatch):
    _, adapter, _, kwargs, calls = endpoint
    monkeypatch.setattr(opposite_identity, "detect_opposite_head_region", lambda *a, **k: None)
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert calls == []
    assert adapter._test_decoder_modes == [False]
    assert getattr(adapter, "_current_position_epoch_proof", None) is None
    assert getattr(adapter, "_pending_qr_observation_pose", None) is None
    assert not adapter._last_observation_update.frame_accepted
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


def test_current_head_endpoint_path_never_constructs_native_qr_outline_detector(endpoint, monkeypatch):
    _, adapter, _, kwargs, calls = endpoint
    def forbidden(*args, **options):
        raise AssertionError('camera exploration requested QR outline detection')
    monkeypatch.setattr(cv2, 'QRCodeDetector', forbidden)
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert len(calls) == 1 and calls[0]['outline']['policy'] == HEAD_REGION_POLICY
    assert adapter._pending_qr_observation_pose.qr_binding.qr_texts_for_evidence == ('Start',)
    assert commit_qr_observation_pose(adapter) is not None


@pytest.mark.parametrize('texts', [(), ('Start', 'QR_003')])
def test_endpoint_support_cannot_replace_missing_or_conflicting_payload(endpoint, monkeypatch, texts):
    _, adapter, _, kwargs, calls = endpoint
    monkeypatch.setattr(opposite_identity, 'detect_qr_observations_bgr',
        lambda *a, **kw: tuple(DecodedQrObservation(t, None, 'payload_only_fixture', 4.) for t in texts))
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert len(calls) == 1
    assert not adapter._pending_qr_observation_pose.qr_binding.accepted
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


@pytest.mark.parametrize('tamper', ['image_stamp', 'gap_coverage'])
def test_current_head_endpoint_requires_same_tuple_and_visual_gap_coverage(endpoint, monkeypatch, tamper):
    _, adapter, _, kwargs, calls = endpoint
    detect = opposite_identity.detect_opposite_head_region
    def changed_region(*args, **options):
        region = detect(*args, **options)
        assert region is not None
        if tamper == 'image_stamp':
            return replace(region, image_stamp_sec=region.image_stamp_sec-.01)
        return replace(region, corners_px=tuple((x+25., y) for x, y in region.corners_px),
            full_image_center_px=(region.full_image_center_px[0]+25., region.full_image_center_px[1]))
    monkeypatch.setattr(opposite_identity, 'detect_opposite_head_region', changed_region)
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert calls == []
    reason = adapter._capture_pending['detector_metadata']['endpoint_confirmation']['reason']
    assert ('differs from its current calibrated hint' if tamper == 'image_stamp'
            else 'does not span both scan gap endpoints') in reason
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


def test_head_work_that_outlives_current_tuple_cannot_admit_endpoint(endpoint, monkeypatch):
    case, adapter, clock, kwargs, calls = endpoint
    detect = opposite_identity.detect_opposite_head_region

    def delayed(*args, **options):
        region = detect(*args, **options)
        assert region is not None
        clock.clock_sec = case["row"]["image_stamp_sec"] + .6
        return region

    monkeypatch.setattr(opposite_identity, "detect_opposite_head_region", delayed)
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert calls == []
    assert getattr(adapter, "_current_position_epoch_proof", None) is None
    assert not adapter._last_observation_update.frame_accepted
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


def test_hint_alone_cannot_supply_reconciliation_or_target_uniqueness(endpoint):
    case, _, _, _, _ = endpoint
    hint, seed = build_opposite_endpoint_hint(**case["hint_kwargs"])
    assert hint is not None
    with pytest.raises(ValueError):
        validate_opposite_endpoint(seed)
    row = {**case["row"], "robot_pose": tuple(asdict(case["robot_pose"]).values())}
    assert StoppedTargetReconciliation().observe(**row, fragmentation=seed) is None


def test_ready_current_qr_still_cannot_publish_after_source_expiry(endpoint):
    case, adapter, clock, kwargs, _ = endpoint
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert adapter._qr_observation_pose_ready is not None
    clock.clock_sec = case["row"]["image_stamp_sec"] + .6
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.completed and not adapter.args.qr_observation_pose_json.exists()
    assert adapter._last_camera_publication_freshness["accepted"] is False


def test_endpoint_proof_cannot_produce_rectangular_identity_crop_without_its_head(endpoint):
    case, adapter, _, kwargs, calls = endpoint
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert len(calls) == 1
    options = dict(case["row"]["options"], scan=case["scan"],
        scan_from_map=case["scan_from_map"], camera_from_map=case["camera_from_map"],
        intrinsics=case["intrinsics"], model_profile=case["model"],
        image_stamp_sec=case["row"]["image_stamp_sec"],
        sync_tolerance_sec=adapter.args.sync_tolerance_sec,
        target_reconciliation=adapter._current_position_epoch_proof,
        fragmentation=calls[0], now_sec=case["now_sec"], max_scan_age_sec=.5)
    search = current_scan_qr_search(**options)
    assert search[0] is not None
    attempt, evidence = exclusive_identity_crop(candidate_uid=adapter.args.stand_id,
        snapshot=case["snapshot"], support=None, search_result=search, **options)
    assert attempt is None and evidence["accepted"] is False
    assert evidence["reason"] == "endpoint confirmation requires its masked current head support"


@pytest.mark.parametrize("tamper", ("remove_support", "different_valid_quad"))
def test_persisted_endpoint_receipt_rejects_crop_rebinding_even_with_new_hash(endpoint, tamper):
    case, adapter, _, kwargs, _ = endpoint
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert commit_qr_observation_pose(adapter) is not None
    payload = deepcopy(_bound_load(case, adapter))
    crop = payload["qr_binding"]["current_head_binding"]
    if tamper == "remove_support":
        crop.pop("target_support")
        crop["sampling"] = "rectangular_crop"
        reason = "endpoint confirmation requires its masked current head support"
    else:
        support = crop["target_support"]
        # A tiny, symmetric change preserves the same calibrated center and
        # valid target ray. It still is not the exact head region that proved the
        # endpoints, so a producer cannot substitute it at the file boundary.
        delta = 1 / 1024
        support["corners_px"] = [
            [x + dx * delta, y + dy * delta]
            for (x, y), (dx, dy) in zip(support["corners_px"],
                ((1, 1), (-1, 1), (-1, -1), (1, -1)))
        ]
        validate_target_support(support)
        reason = "endpoint confirmation differs from the current head region"
    payload.pop(HASH_FIELD)
    forged = content_hashed_payload(payload, hash_field=HASH_FIELD)
    with pytest.raises(ValueError, match=reason):
        validate_qr_verified_observation_pose(forged)


@pytest.mark.parametrize("tamper", ("projection", "stamp", "other_quad", "missing_corners"))
def test_persisted_raw_payload_binding_replays_calibration_stamp_and_own_corners(endpoint, tamper):
    payload = _legacy_payload(endpoint)
    crop = payload["qr_binding"]["current_head_binding"]
    raw = crop["raw_pixel_binding"]
    assert raw["qr_id"] == "Start"
    # Replay the source calibration and the decoder's own corners. Native
    # corner locations vary slightly by backend/version; the production
    # geometric tolerance, followed by these tamper controls, is the contract.
    mapped = opposite_raw_qr._rectified_quad(raw['decoded_raw_corners_px'],
        endpoint[0]['calibration'], cv2)
    error = opposite_raw_qr._corner_error(mapped, raw['confirmed_corners_px'])
    assert raw['corner_error_px'] == pytest.approx(error)
    assert 0 <= error <= raw['maximum_corner_error_px']
    assert raw["motion_authorized"] is False and raw["supplies_angle"] is False
    if tamper == "projection":
        raw["calibration"]["projection_matrix"][0] += 1.
    elif tamper == "stamp":
        raw["image_stamp_sec"] += .001
    elif tamper == "other_quad":
        # The alternate valid quad stays in the source crop, but is outside
        # the confirmed-outline tolerance. It cannot borrow this proof.
        raw["decoded_raw_corners_px"] = [[x + 5., y] for x, y in raw["decoded_raw_corners_px"]]
        assert all(crop["raw_pixel_binding"]["source_bounds_xyxy"][0] <= x
            < crop["raw_pixel_binding"]["source_bounds_xyxy"][2]
            for x, _ in raw["decoded_raw_corners_px"])
    else:
        raw["decoded_raw_corners_px"] = None
    payload.pop(HASH_FIELD)
    with pytest.raises(ValueError):
        validate_qr_verified_observation_pose(content_hashed_payload(payload, hash_field=HASH_FIELD))


@pytest.mark.parametrize("tamper", ("missing_corners", "other_quad", "backend_error"))
def test_legacy_raw_decoder_rejects_unbound_payload_or_native_failure(endpoint, monkeypatch, tamper):
    _, crop, support, _ = _legacy_crop(endpoint)
    decode = opposite_raw_qr.detect_qr_observations_bgr

    def invalid_native_result(*args, **options):
        if tamper == "backend_error":
            raise cv2.error("injected native QR failure")
        observations = decode(*args, **options)
        assert len(observations) == 1 and observations[0].text == "Start"
        changed = (None if tamper == "missing_corners" else
            tuple((x + 5., y) for x, y in observations[0].corners))
        return (replace(observations[0], corners=changed),)

    monkeypatch.setattr(opposite_raw_qr, "detect_qr_observations_bgr", invalid_native_result)
    assert _legacy_decode(endpoint, crop, support) == ()
    assert 'raw_pixel_binding' not in crop


def _changed_calibration(calibration, field):
    values = list(getattr(calibration, field))
    if field == "distortion":
        values = [0.] * len(values)
    elif field == "rectification_matrix":
        # A valid small rotation, not malformed matrix data.
        theta = .001
        values = [math.cos(theta), -math.sin(theta), 0.,
            math.sin(theta), math.cos(theta), 0., 0., 0., 1.]
    else:
        values[0] += 1.
    changed = replace(calibration, **{field: tuple(values)})
    assert changed != calibration
    return changed


@pytest.mark.parametrize("field", ("projection_matrix", "camera_matrix", "distortion",
    "rectification_matrix", "sealed_profile"))
def test_legacy_raw_decoder_rejects_changed_calibration_before_decoding(endpoint, monkeypatch, field):
    case, adapter, _, _, _ = endpoint
    _, crop, support, _ = _legacy_crop(endpoint)
    options = {}
    if field == "sealed_profile":
        adapter.calibration = replace(adapter.calibration, source="untrusted replacement calibration")
    else:
        options['calibration'] = _changed_calibration(case["calibration"], field)
    def unexpected_decode(*args, **options):
        raise AssertionError("mismatched calibration reached QR decoding")
    monkeypatch.setattr(opposite_raw_qr, "detect_qr_observations_bgr", unexpected_decode)
    assert _legacy_decode(endpoint, crop, support, **options) == ()
    assert 'raw_pixel_binding' not in crop


@pytest.mark.parametrize("field", ("camera_matrix", "distortion", "rectification_matrix"))
@pytest.mark.parametrize("replace_sealed_profile", (False, True))
def test_rehashed_coherent_raw_calibration_substitution_cannot_change_sealed_source(
        endpoint, field, replace_sealed_profile):
    payload = _legacy_payload(endpoint)
    crop = payload["qr_binding"]["current_head_binding"]
    raw = crop["raw_pixel_binding"]
    calibration = _changed_calibration(CameraCalibration(**raw["calibration"]), field)
    raw["calibration"] = asdict(calibration)
    if replace_sealed_profile:
        profile_field = "distortion_coefficients" if field == "distortion" else field
        raw["sealed_calibration_profile"][profile_field] = getattr(calibration, field)
    _, proof = opposite_raw_qr._endpoint_proof(crop)
    confirmed = tuple(tuple(p) for p in raw["confirmed_corners_px"])
    raw_quad = opposite_raw_qr._raw_quad(confirmed, calibration, cv2)
    bounds, margin = opposite_raw_qr._bounds(raw_quad, payload["image_shape"])
    mapped = opposite_raw_qr._rectified_quad(raw_quad, calibration, cv2)
    raw.update(raw_source_quad_px=raw_quad, source_bounds_xyxy=bounds,
        source_margin_px=margin, decoded_raw_corners_px=raw_quad,
        mapped_rectified_corners_px=mapped,
        corner_error_px=opposite_raw_qr._corner_error(mapped, confirmed))
    # This intentionally forged evidence is internally consistent: all raw
    # geometry was recomputed under its substituted calibration. The sealed
    # source binding, rather than a malformed-corner check, must reject it.
    assert opposite_raw_qr._validate_binding_pixels(raw, proof=proof,
        calibration=calibration, image_shape=payload["image_shape"], cv2=cv2) is raw
    payload.pop(HASH_FIELD)
    reason = ("raw QR sealed calibration digest differs from the certified target"
        if replace_sealed_profile else "raw QR current calibration differs from the sealed calibration")
    with pytest.raises(ValueError, match=reason):
        validate_qr_verified_observation_pose(content_hashed_payload(payload, hash_field=HASH_FIELD))
