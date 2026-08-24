from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from scripts.aufgabe04.navigation.approach.exact_two_camera_admission import (
    SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
    build_exact_two_camera_candidate_snapshot,
    exact_two_camera_handoff_sha256,
    new_exact_two_camera_handoff,
    require_handoff_candidate_support,
    write_exact_two_camera_admission,
    write_exact_two_camera_handoff,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.record_stand_candidate_decision import (
    main as record_candidate_decision,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_CONFIRMED,
    STATUS_PROVISIONAL,
    load_stand_survey_registry,
    write_coverage_survey_plan,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    recommendation_to_payload,
)
from scripts.aufgabe04.real_robot.autonomous_candidate_approach import (
    CandidateApproachConfig,
    CandidateApproachEffects,
    build_camera_candidate_decision_receipt,
    execute_candidate_approach_phase,
    validate_candidate_approach_handoff,
)
from scripts.aufgabe04.real_robot.recommendation_builder import (
    build_real_viewpoint_recommendation,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    write_candidate_snapshot,
)
from tests.aufgabe04.test_exact_two_camera_admission import _ready_inputs


TERMINAL_HASH = "c" * 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_recommendation(
    path: Path,
    candidate,
    *,
    stand_id: str | None = None,
) -> None:
    geometry = candidate.geometry
    recommendation = build_real_viewpoint_recommendation(
        stream_id=f"camera_{candidate.candidate_uid}",
        stand_id=(
            candidate.candidate_uid if stand_id is None else stand_id
        ),
        planning_frame="map",
        stand_center=Pose2D(geometry.x_m, geometry.y_m, 0.0),
        stand_radius_m=geometry.radius_m,
        stand_uncertainty_m=geometry.uncertainty_m,
        robot_pose=Pose2D(geometry.x_m - 0.70, geometry.y_m, 0.0),
        stand_axis_rad=0.0,
        axis_confidence=0.90,
        axis_sample_count=7,
        sensor_stamp_sec=20.0,
        expected_qr_id="QR_01",
        observed_qr_ids=("QR_01",),
        target_distance_m=0.35,
        observation_unix_sec=20.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            recommendation_to_payload(recommendation),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _fixture(root: Path):
    plan, progress, registry, _, admission = _ready_inputs()
    survey_root = root / "survey"
    survey_root.mkdir()
    write_coverage_survey_plan(survey_root / "coverage_plan.json", plan)
    write_survey_progress(
        survey_root / "coverage_progress.json", progress, plan
    )
    write_stand_survey_registry(
        survey_root / "stand_registry.json", registry, plan
    )

    camera_admission_path = root / "camera_admission.json"
    camera_admission_sha256 = write_exact_two_camera_admission(
        camera_admission_path,
        admission,
    )
    snapshot = build_exact_two_camera_candidate_snapshot(
        plan,
        registry,
        admission,
        snapshot_id="exact_two_camera_candidates",
    )
    snapshot_path = root / "candidate_snapshot.json"
    write_candidate_snapshot(snapshot_path, snapshot)
    handoff = new_exact_two_camera_handoff(
        handoff_id="exact_two_camera_handoff",
        created_unix_sec=30.0,
        admission=admission,
        terminal_checkpoint_path=root / "mission_summary.json",
        terminal_checkpoint_sha256=TERMINAL_HASH,
        lidar_admission_path=root / "lidar_admission.json",
        lidar_admission_sha256=admission.lidar_checkpoint_sha256,
        camera_admission_path=camera_admission_path,
        camera_admission_sha256=camera_admission_sha256,
        candidate_snapshot_path=snapshot_path,
        candidate_snapshot=snapshot,
    )
    handoff_path = root / "camera_handoff.json"
    handoff_sha256 = write_exact_two_camera_handoff(
        handoff_path,
        handoff,
    )
    config = CandidateApproachConfig(
        session_root=root / "session",
        survey_root=survey_root,
        session_id="mission",
        semantic_map_id="arena",
        planning_frame="map",
        map_yaml=root / "map.yaml",
        plan=plan,
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        approach_offset_m=0.70,
        inflation_radius_m=0.25,
        candidate_transit_radius_m=0.31,
        physical_clearance={
            "minimum_active_standoff_m": 0.32,
            "minimum_candidate_transit_radius_m": 0.31,
            "minimum_static_inflation_m": 0.25,
        },
        uncertainty_sigma_multiplier=2.0,
        localization_branch_proof_id="known_start",
        mission_leg_motion_authorization_json=(
            root / "mission_authorization.json"
        ),
        exact_two_camera_handoff_path=handoff_path,
        exact_two_camera_handoff_sha256=handoff_sha256,
    )
    return SimpleNamespace(
        plan=plan,
        progress=progress,
        registry=registry,
        survey_root=survey_root,
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        handoff=handoff,
        handoff_path=handoff_path,
        handoff_sha256=handoff_sha256,
        config=config,
    )


def _write_exact_two_receipt(
    fixture,
    root: Path,
    candidate_uid: str,
    *,
    recommendation_stand_id: str | None = None,
) -> Path:
    candidate = fixture.snapshot.candidate_for(candidate_uid)
    assert candidate is not None
    recommendation_path = root / f"{candidate_uid}_recommendation.json"
    _write_recommendation(
        recommendation_path,
        candidate,
        stand_id=recommendation_stand_id,
    )
    evidence = require_handoff_candidate_support(
        fixture.handoff,
        candidate_uid,
    )
    assert evidence.support_class is not None
    payload = build_camera_candidate_decision_receipt(
        config=fixture.config,
        candidate=candidate,
        recommendation_path=recommendation_path,
        exact_two_support_by_uid={candidate_uid: evidence.support_class},
    )
    receipt_path = root / f"{candidate_uid}_decision.json"
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return receipt_path


def _record(fixture, receipt_path: Path) -> int:
    return record_candidate_decision(
        [
            "--survey-root",
            str(fixture.survey_root),
            "--decision-receipt-json",
            str(receipt_path),
            "--exact-two-camera-handoff-json",
            str(fixture.handoff_path),
            "--candidate-snapshot-json",
            str(fixture.snapshot_path),
        ]
    )


class ExactTwoCameraDecisionTest(unittest.TestCase):
    def test_valid_handoff_allows_listed_provisional_camera_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            uid = "survey_candidate_0003"
            receipt_path = _write_exact_two_receipt(fixture, root, uid)

            with redirect_stdout(StringIO()):
                result = _record(fixture, receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_CONFIRMED,
            )
            stored = json.loads(
                (fixture.survey_root / "decisions" / f"{uid}.json").read_text()
            )
            self.assertEqual(stored["schema_version"], 2)
            self.assertEqual(
                stored["candidate_support_class"],
                SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
            )
            self.assertEqual(
                stored["exact_two_camera_handoff_sha256"],
                exact_two_camera_handoff_sha256(fixture.handoff),
            )
            self.assertEqual(
                stored["camera_recommendation_sha256"],
                _sha256(Path(stored["camera_evidence_path"])),
            )

    def test_missing_or_tampered_handoff_fails_before_registry_mutation(self):
        for failure in ("missing_arguments", "tampered_artifact"):
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    fixture = _fixture(root)
                    uid = "survey_candidate_0003"
                    receipt_path = _write_exact_two_receipt(
                        fixture, root, uid
                    )
                    argv = [
                        "--survey-root",
                        str(fixture.survey_root),
                        "--decision-receipt-json",
                        str(receipt_path),
                    ]
                    if failure == "tampered_artifact":
                        payload = json.loads(fixture.handoff_path.read_text())
                        payload["camera_population_ready"] = False
                        fixture.handoff_path.write_text(json.dumps(payload))
                        argv.extend(
                            [
                                "--exact-two-camera-handoff-json",
                                str(fixture.handoff_path),
                                "--candidate-snapshot-json",
                                str(fixture.snapshot_path),
                            ]
                        )
                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            record_candidate_decision(argv)
                    registry = load_stand_survey_registry(
                        fixture.survey_root / "stand_registry.json",
                        fixture.plan,
                    )
                    self.assertEqual(
                        registry.candidate_for(uid).status,
                        STATUS_PROVISIONAL,
                    )

    def test_unlisted_provisional_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            extra = replace(
                fixture.registry.candidates[-1],
                candidate_uid="survey_candidate_0006",
                x_m=1.50,
                y_m=0.60,
                source_observation_ids=("observation_0006",),
            )
            changed_registry = replace(
                fixture.registry,
                candidates=(*fixture.registry.candidates, extra),
            )
            write_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                changed_registry,
                fixture.plan,
            )
            recommendation_path = root / "unlisted_recommendation.json"
            frozen_template = fixture.snapshot.candidates[-1]
            unlisted_frozen = replace(
                frozen_template,
                candidate_uid=extra.candidate_uid,
                geometry=replace(
                    frozen_template.geometry,
                    x_m=extra.x_m,
                    y_m=extra.y_m,
                ),
            )
            _write_recommendation(recommendation_path, unlisted_frozen)
            receipt = {
                "schema_version": 2,
                "survey_id": fixture.plan.survey_id,
                "candidate_uid": extra.candidate_uid,
                "decision": "confirmed",
                "decision_source": "camera_evidence",
                "camera_evidence_path": str(recommendation_path),
                "camera_recommendation_sha256": _sha256(recommendation_path),
                "exact_two_camera_handoff_path": str(fixture.handoff_path),
                "exact_two_camera_handoff_sha256": fixture.handoff_sha256,
                "candidate_snapshot_path": str(fixture.snapshot_path),
                "candidate_snapshot_sha256": (
                    fixture.handoff.candidate_snapshot_sha256
                ),
                "candidate_support_class": (
                    SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION
                ),
            }
            receipt_path = root / "unlisted_decision.json"
            receipt_path.write_text(json.dumps(receipt))

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    _record(fixture, receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json", fixture.plan
            )
            self.assertEqual(
                registry.candidate_for(extra.candidate_uid).status,
                STATUS_PROVISIONAL,
            )

    def test_cross_candidate_recommendation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            uid = "survey_candidate_0003"
            receipt_path = _write_exact_two_receipt(
                fixture,
                root,
                uid,
                recommendation_stand_id="survey_candidate_0004",
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    _record(fixture, receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json", fixture.plan
            )
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_PROVISIONAL,
            )

    def test_live_registry_geometry_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            uid = "survey_candidate_0003"
            receipt_path = _write_exact_two_receipt(fixture, root, uid)
            original = fixture.registry.candidate_for(uid)
            assert original is not None
            changed = replace(
                original,
                x_m=original.x_m + 0.01,
            )
            changed_registry = replace(
                fixture.registry,
                candidates=tuple(
                    changed if item.candidate_uid == uid else item
                    for item in fixture.registry.candidates
                ),
            )
            write_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                changed_registry,
                fixture.plan,
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    _record(fixture, receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json", fixture.plan
            )
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_PROVISIONAL,
            )

    def test_schema_v1_standard_mode_keeps_provisional_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            for uid, expected_success in (
                ("survey_candidate_0001", True),
                ("survey_candidate_0003", False),
            ):
                receipt_path = root / f"v1_{uid}.json"
                receipt_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "survey_id": fixture.plan.survey_id,
                            "candidate_uid": uid,
                            "decision": "confirmed",
                            "decision_source": "operator",
                            "operator_confirmed": True,
                        }
                    )
                )
                argv = [
                    "--survey-root",
                    str(fixture.survey_root),
                    "--decision-receipt-json",
                    str(receipt_path),
                ]
                if expected_success:
                    with redirect_stdout(StringIO()):
                        self.assertEqual(record_candidate_decision(argv), 0)
                else:
                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            record_candidate_decision(argv)

    def test_invalid_handoff_blocks_before_first_live_robot_effect(self):
        for failure in ("missing_hash", "wrong_hash", "tampered_file"):
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    fixture = _fixture(root)
                    if failure == "missing_hash":
                        config = replace(
                            fixture.config,
                            exact_two_camera_handoff_sha256=None,
                        )
                    elif failure == "wrong_hash":
                        config = replace(
                            fixture.config,
                            exact_two_camera_handoff_sha256="f" * 64,
                        )
                    else:
                        config = fixture.config
                        payload = json.loads(fixture.handoff_path.read_text())
                        payload["camera_population_ready"] = False
                        fixture.handoff_path.write_text(json.dumps(payload))
                    read_pose = Mock()
                    effects = CandidateApproachEffects(
                        read_current_pose=read_pose,
                        run_motion_leg=Mock(),
                        capture_observation=Mock(),
                    )

                    with self.assertRaises(ValueError):
                        execute_candidate_approach_phase(config, effects)

                    read_pose.assert_not_called()
                    effects.run_motion_leg.assert_not_called()

    def test_valid_handoff_support_partition_is_available_pre_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))

            support = validate_candidate_approach_handoff(fixture.config)

            self.assertIsNotNone(support)
            self.assertEqual(len(support), 5)
            self.assertEqual(
                support["survey_candidate_0003"],
                SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
            )

    def test_valid_handoff_support_reaches_route_selector_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp))
            read_pose = Mock(return_value=Pose2D(0.0, 0.0, 0.0))
            motion = Mock()
            capture = Mock()
            observed_support = {}

            def inspect_selection(request):
                observed_support.update(request.support_class_by_uid or {})
                raise RuntimeError("stop after route-selection admission")

            with self.assertRaisesRegex(
                RuntimeError,
                "stop after route-selection admission",
            ):
                execute_candidate_approach_phase(
                    fixture.config,
                    CandidateApproachEffects(
                        select_initial_preapproach=inspect_selection,
                        read_current_pose=read_pose,
                        run_motion_leg=motion,
                        capture_observation=capture,
                    ),
                )

            self.assertEqual(len(observed_support), 5)
            self.assertEqual(
                observed_support["survey_candidate_0003"],
                SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
            )
            motion.assert_not_called()
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
