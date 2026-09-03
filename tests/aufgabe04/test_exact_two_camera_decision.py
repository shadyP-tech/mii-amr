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

from scripts.aufgabe04.artifacts.content_store import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_admission import (
    SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
    ExactTwoCameraAdmissionError,
    build_exact_two_camera_candidate_snapshot,
    evaluate_exact_two_camera_admission,
    exact_two_camera_handoff_sha256,
    new_exact_two_camera_handoff,
    require_handoff_candidate_support,
    write_exact_two_camera_admission,
    write_exact_two_camera_handoff,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_population_binding import (
    validate_live_exact_two_camera_population_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
    project_candidate_snapshot_to_planning_frame,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    CandidatePoint2D,
)
from scripts.aufgabe04.navigation.approach.camera_decision_geometry_binding import (
    CameraCandidateFrameBinding,
    require_projected_camera_candidate_binding,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_lifecycle import (
    evaluate_exact_two_lidar_checkpoint,
)
from scripts.aufgabe04.navigation.coverage.stand_candidate_population_retention import (
    STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.record_stand_candidate_decision import (
    main as record_candidate_decision,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    REJECTION_BASIS_CAMERA,
    REJECTION_BASIS_NEGATIVE_VISIBILITY,
    STATUS_CONFIRMED,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    load_stand_survey_registry,
    stand_survey_registry_sha256,
    write_coverage_survey_plan,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    recommendation_to_payload,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachConfig,
    CandidateApproachEffects,
    CandidateObservation,
    CameraCandidateInitialSelection,
    build_camera_candidate_decision_receipt,
    execute_candidate_approach_phase,
    validate_candidate_approach_handoff,
)
from scripts.aufgabe04.real_robot.execution.child_runner import (
    MotionLegOutcome,
)
from scripts.aufgabe04.real_robot.configuration.recommendation import (
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


def _fixture(
    root: Path,
    *,
    boundary_uid: str | None = None,
    with_boundary_audit_only_extra: bool = False,
    with_retained_negative_visibility_history: bool = False,
    with_frame_provenance: bool = False,
):
    plan, progress, registry, _, admission = _ready_inputs()
    if with_boundary_audit_only_extra:
        template = registry.candidates[-1]
        registry = replace(
            registry,
            candidates=(
                *registry.candidates,
                replace(
                    template,
                    candidate_uid="survey_candidate_0006",
                    x_m=1.50,
                    y_m=0.60,
                    first_seen_sec=16.0,
                    last_seen_sec=17.0,
                    source_observation_ids=("observation_0006",),
                    static_map_disposition=(
                        STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
                    ),
                ),
            ),
        )
        lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)
        admission = evaluate_exact_two_camera_admission(
            plan,
            progress,
            registry,
            lidar,
        )
    if with_retained_negative_visibility_history:
        template = registry.candidates[-1]
        registry = replace(
            registry,
            candidates=(
                *registry.candidates,
                replace(
                    template,
                    candidate_uid="survey_candidate_0006",
                    x_m=1.50,
                    y_m=0.60,
                    first_seen_sec=16.0,
                    last_seen_sec=17.0,
                    source_observation_ids=("observation_0006",),
                    status=STATUS_REJECTED,
                    rejection_basis=REJECTION_BASIS_NEGATIVE_VISIBILITY,
                ),
            ),
        )
        lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)
        admission = evaluate_exact_two_camera_admission(
            plan,
            progress,
            registry,
            lidar,
        )
    if with_frame_provenance:
        registry = replace(
            registry,
            candidates=tuple(
                replace(
                    candidate,
                    frame_provenance=CandidateFrameProvenance(
                        map_frame="map",
                        odom_frame="odom",
                        canonical_odom_point=CandidatePoint2D(
                            candidate.x_m,
                            candidate.y_m,
                        ),
                        source_evidence_id=(
                            f"frame_evidence_{candidate.candidate_uid}"
                        ),
                    ),
                )
                for candidate in registry.candidates
            ),
        )
        lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)
        admission = evaluate_exact_two_camera_admission(
            plan,
            progress,
            registry,
            lidar,
        )
    if boundary_uid is not None:
        if registry.candidate_for(boundary_uid) is None:
            raise ValueError(f"unknown boundary fixture UID: {boundary_uid}")
        registry = replace(
            registry,
            candidates=tuple(
                replace(
                    candidate,
                    static_map_disposition=(
                        STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
                    ),
                )
                if candidate.candidate_uid == boundary_uid
                else candidate
                for candidate in registry.candidates
            ),
        )
        lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)
        admission = evaluate_exact_two_camera_admission(
            plan,
            progress,
            registry,
            lidar,
        )
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


def _camera_frame_projection(
    fixture,
    root: Path,
    *,
    translation_x_m: float = 0.012902,
    translation_y_m: float = -0.047650,
):
    projection = project_candidate_snapshot_to_planning_frame(
        fixture.snapshot,
        fixture.registry,
        CandidatePlanningFrame(
            current_pose=Pose2D(-0.5, -0.7, 0.0),
            map_from_odom=PlanarTransform2D(
                translation_x_m,
                translation_y_m,
                0.0,
            ),
        ),
    )
    projection_root = root / "camera_frame_projection"
    projected_snapshot_path = projection_root / "candidate_snapshot.json"
    projected_snapshot_sha256 = write_candidate_snapshot(
        projected_snapshot_path,
        projection.projected_snapshot,
    )
    evidence_path = projection_root / "candidate_frame_projection.json"
    evidence_sha256 = write_content_hashed_json(
        evidence_path,
        {
            **projection.to_evidence(),
            "source_candidate_snapshot_path": str(fixture.snapshot_path),
            "projected_candidate_snapshot_path": str(
                projected_snapshot_path
            ),
        },
        hash_field="candidate_frame_projection_sha256",
    )
    return SimpleNamespace(
        projection=projection,
        projected_snapshot_path=projected_snapshot_path,
        projected_snapshot_sha256=projected_snapshot_sha256,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
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


def _write_projected_exact_two_receipt(
    fixture,
    root: Path,
    candidate_uid: str,
    *,
    recommendation_uses_canonical_geometry: bool = False,
):
    frame_binding = _camera_frame_projection(fixture, root)
    camera_candidate = frame_binding.projection.projected_snapshot.candidate_for(
        candidate_uid
    )
    canonical_candidate = fixture.snapshot.candidate_for(candidate_uid)
    assert camera_candidate is not None
    assert canonical_candidate is not None
    recommendation_candidate = (
        canonical_candidate
        if recommendation_uses_canonical_geometry
        else camera_candidate
    )
    recommendation_path = root / f"{candidate_uid}_recommendation.json"
    _write_recommendation(recommendation_path, recommendation_candidate)
    evidence = require_handoff_candidate_support(
        fixture.handoff,
        candidate_uid,
    )
    assert evidence.support_class is not None
    payload = build_camera_candidate_decision_receipt(
        config=fixture.config,
        candidate=camera_candidate,
        recommendation_path=recommendation_path,
        exact_two_support_by_uid={candidate_uid: evidence.support_class},
        camera_frame_binding=CameraCandidateFrameBinding(
            camera_snapshot_path=frame_binding.projected_snapshot_path,
            camera_snapshot_sha256=(
                frame_binding.projected_snapshot_sha256
            ),
            projection_path=frame_binding.evidence_path,
            projection_sha256=frame_binding.evidence_sha256,
        ),
    )
    receipt_path = root / f"{candidate_uid}_projected_decision.json"
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return SimpleNamespace(
        receipt_path=receipt_path,
        recommendation_path=recommendation_path,
        frame_binding=frame_binding,
        canonical_candidate=canonical_candidate,
        camera_candidate=camera_candidate,
    )


def _rewrite_projection_evidence(
    binding,
    receipt_path: Path,
    mutate,
) -> None:
    hash_field = "candidate_frame_projection_sha256"
    evidence = json.loads(binding.evidence_path.read_text())
    evidence.pop(hash_field)
    mutate(evidence)
    evidence_sha256 = payload_sha256(evidence)
    evidence[hash_field] = evidence_sha256
    binding.evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt[hash_field] = evidence_sha256
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def _record(fixture, receipt_path: Path) -> int:
    receipt = json.loads(receipt_path.read_text())
    argv = [
        "--survey-root",
        str(fixture.survey_root),
        "--decision-receipt-json",
        str(receipt_path),
        "--exact-two-camera-handoff-json",
        str(fixture.handoff_path),
        "--candidate-snapshot-json",
        str(fixture.snapshot_path),
    ]
    if receipt["schema_version"] == 3:
        argv.extend(
            [
                "--camera-candidate-snapshot-json",
                str(receipt["camera_candidate_snapshot_path"]),
                "--candidate-frame-projection-json",
                str(receipt["candidate_frame_projection_path"]),
            ]
        )
    return record_candidate_decision(argv)


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

    def test_schema_v2_commits_sequential_selected_candidates_with_sealed_boundary_extra(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(
                root,
                with_boundary_audit_only_extra=True,
            )
            first_uid = "survey_candidate_0001"
            second_uid = "survey_candidate_0003"

            self.assertEqual(len(fixture.registry.candidates), 6)
            self.assertEqual(len(fixture.snapshot.candidates), 5)
            self.assertEqual(
                fixture.handoff.admission_decision.selected_candidate_uids,
                tuple(
                    f"survey_candidate_{index:04d}"
                    for index in range(1, 6)
                ),
            )
            self.assertEqual(
                fixture.handoff.admission_decision.boundary_audit_only_candidate_uids,
                ("survey_candidate_0006",),
            )
            self.assertEqual(
                fixture.handoff.admission_decision.excluded_candidate_uids,
                ("survey_candidate_0006",),
            )
            self.assertEqual(
                fixture.handoff.source_registry_sha256,
                stand_survey_registry_sha256(fixture.registry),
            )

            first_receipt = _write_exact_two_receipt(
                fixture,
                root,
                first_uid,
            )
            second_receipt = _write_exact_two_receipt(
                fixture,
                root,
                second_uid,
            )
            first_payload = json.loads(first_receipt.read_text())
            first_payload["decision"] = STATUS_REJECTED
            first_receipt.write_text(
                json.dumps(first_payload, indent=2, sort_keys=True) + "\n"
            )
            with redirect_stdout(StringIO()):
                self.assertEqual(_record(fixture, first_receipt), 0)
                self.assertEqual(_record(fixture, second_receipt), 0)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertEqual(
                registry.candidate_for(first_uid).status,
                STATUS_REJECTED,
            )
            self.assertEqual(
                registry.candidate_for(second_uid).status,
                STATUS_CONFIRMED,
            )
            self.assertEqual(
                registry.candidate_for("survey_candidate_0006").status,
                STATUS_PROVISIONAL,
            )
            self.assertTrue(
                (
                    fixture.survey_root
                    / "decisions"
                    / f"{first_uid}.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    fixture.survey_root
                    / "decisions"
                    / f"{second_uid}.json"
                ).is_file()
            )

    def test_schema_v3_accepts_authenticated_arrival_projected_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root, with_frame_provenance=True)
            uid = "survey_candidate_0003"
            decision = _write_projected_exact_two_receipt(
                fixture,
                root,
                uid,
            )

            displacement_m = (
                (
                    decision.camera_candidate.geometry.x_m
                    - decision.canonical_candidate.geometry.x_m
                )
                ** 2
                + (
                    decision.camera_candidate.geometry.y_m
                    - decision.canonical_candidate.geometry.y_m
                )
                ** 2
            ) ** 0.5
            self.assertGreater(displacement_m, 0.049)
            with redirect_stdout(StringIO()):
                result = _record(fixture, decision.receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            stored = json.loads(
                (fixture.survey_root / "decisions" / f"{uid}.json").read_text()
            )
            self.assertEqual(result, 0)
            self.assertEqual(registry.candidate_for(uid).status, STATUS_CONFIRMED)
            self.assertEqual(stored["schema_version"], 3)
            self.assertEqual(
                stored["candidate_snapshot_sha256"],
                fixture.handoff.candidate_snapshot_sha256,
            )
            self.assertEqual(
                stored["camera_candidate_snapshot_sha256"],
                decision.frame_binding.projected_snapshot_sha256,
            )
            self.assertEqual(
                stored["candidate_frame_projection_sha256"],
                decision.frame_binding.evidence_sha256,
            )
            projection_evidence = json.loads(
                decision.frame_binding.evidence_path.read_text()
            )
            self.assertEqual(
                projection_evidence["source_registry_sha256"],
                stand_survey_registry_sha256(fixture.registry),
            )

    def test_schema_v3_accepts_projected_selected_candidate_with_sealed_boundary_extra(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(
                root,
                with_boundary_audit_only_extra=True,
                with_frame_provenance=True,
            )
            uid = "survey_candidate_0003"
            decision = _write_projected_exact_two_receipt(
                fixture,
                root,
                uid,
            )
            receipt = json.loads(decision.receipt_path.read_text())

            bound_candidate = require_projected_camera_candidate_binding(
                receipt,
                canonical_snapshot_path=fixture.snapshot_path,
                canonical_snapshot=fixture.snapshot,
                handoff=fixture.handoff,
                registry=fixture.registry,
                camera_snapshot_path=(
                    decision.frame_binding.projected_snapshot_path
                ),
                projection_path=decision.frame_binding.evidence_path,
                candidate_uid=uid,
            )
            self.assertEqual(bound_candidate, decision.camera_candidate)

            with redirect_stdout(StringIO()):
                result = _record(fixture, decision.receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_CONFIRMED,
            )
            self.assertEqual(
                registry.candidate_for("survey_candidate_0006").status,
                STATUS_PROVISIONAL,
            )

    def test_population_binding_accepts_sealed_negative_visibility_history(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(
                Path(tmp),
                with_retained_negative_visibility_history=True,
            )
            retained_uid = "survey_candidate_0006"
            evidence_uids = tuple(
                evidence.candidate_uid
                for evidence in fixture.handoff.admission_decision.candidate_evidence
            )

            self.assertEqual(len(fixture.registry.candidates), 6)
            self.assertEqual(len(fixture.snapshot.candidates), 5)
            self.assertNotIn(retained_uid, evidence_uids)
            self.assertNotIn(retained_uid, fixture.snapshot.candidate_uids)
            self.assertEqual(
                fixture.handoff.source_registry_sha256,
                stand_survey_registry_sha256(fixture.registry),
            )

            bound = validate_live_exact_two_camera_population_binding(
                fixture.handoff,
                fixture.snapshot,
                fixture.registry,
                candidate_snapshot_path=fixture.snapshot_path,
            )

            self.assertEqual(
                tuple(bound),
                fixture.handoff.admission_decision.selected_candidate_uids,
            )
            retained = fixture.registry.candidate_for(retained_uid)
            assert retained is not None
            self.assertEqual(retained.status, STATUS_REJECTED)
            self.assertEqual(
                retained.rejection_basis,
                REJECTION_BASIS_NEGATIVE_VISIBILITY,
            )

    def test_schema_v3_commits_with_sealed_negative_visibility_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(
                root,
                with_retained_negative_visibility_history=True,
                with_frame_provenance=True,
            )
            uid = "survey_candidate_0003"
            retained_uid = "survey_candidate_0006"
            decision = _write_projected_exact_two_receipt(
                fixture,
                root,
                uid,
            )

            with redirect_stdout(StringIO()):
                result = _record(fixture, decision.receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            retained = registry.candidate_for(retained_uid)
            assert retained is not None
            self.assertEqual(result, 0)
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_CONFIRMED,
            )
            self.assertEqual(retained.status, STATUS_REJECTED)
            self.assertEqual(
                retained.rejection_basis,
                REJECTION_BASIS_NEGATIVE_VISIBILITY,
            )
            self.assertTrue(
                (
                    fixture.survey_root
                    / "decisions"
                    / f"{uid}.json"
                ).is_file()
            )

    def test_population_binding_rejects_changed_negative_visibility_history(
        self,
    ):
        for failure in ("geometry_mutation", "unsealed_addition"):
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    fixture = _fixture(
                        Path(tmp),
                        with_retained_negative_visibility_history=True,
                    )
                    retained_uid = "survey_candidate_0006"
                    retained = fixture.registry.candidate_for(retained_uid)
                    assert retained is not None
                    if failure == "geometry_mutation":
                        changed_candidates = tuple(
                            replace(candidate, x_m=candidate.x_m + 0.01)
                            if candidate.candidate_uid == retained_uid
                            else candidate
                            for candidate in fixture.registry.candidates
                        )
                    else:
                        changed_candidates = (
                            *fixture.registry.candidates,
                            replace(
                                retained,
                                candidate_uid="survey_candidate_0007",
                                x_m=retained.x_m + 0.25,
                                y_m=retained.y_m + 0.10,
                                first_seen_sec=18.0,
                                last_seen_sec=19.0,
                                source_observation_ids=("observation_0007",),
                            ),
                        )
                    changed_registry = replace(
                        fixture.registry,
                        candidates=changed_candidates,
                    )
                    self.assertNotEqual(
                        stand_survey_registry_sha256(changed_registry),
                        fixture.handoff.source_registry_sha256,
                    )

                    with self.assertRaises(
                        ExactTwoCameraAdmissionError
                    ) as raised:
                        validate_live_exact_two_camera_population_binding(
                            fixture.handoff,
                            fixture.snapshot,
                            changed_registry,
                            candidate_snapshot_path=fixture.snapshot_path,
                        )

                    self.assertEqual(
                        raised.exception.code,
                        "live_registry_mismatch",
                    )
                    self.assertIn(
                        "no longer matches the sealed camera handoff",
                        str(raised.exception),
                    )

    def test_schema_v3_accepts_after_another_canonical_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(
                root,
                with_boundary_audit_only_extra=True,
                with_frame_provenance=True,
            )
            projected_uid = "survey_candidate_0003"
            previously_decided_uid = "survey_candidate_0001"
            decision = _write_projected_exact_two_receipt(
                fixture,
                root,
                projected_uid,
            )
            proof = json.loads(
                decision.frame_binding.evidence_path.read_text()
            )
            prior_receipt = _write_exact_two_receipt(
                fixture,
                root,
                previously_decided_uid,
            )
            with redirect_stdout(StringIO()):
                self.assertEqual(_record(fixture, prior_receipt), 0)

            changed_registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertNotEqual(
                stand_survey_registry_sha256(changed_registry),
                proof["source_registry_sha256"],
            )
            self.assertEqual(
                proof["source_registry_sha256"],
                fixture.handoff.source_registry_sha256,
            )

            with redirect_stdout(StringIO()):
                result = _record(fixture, decision.receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                registry.candidate_for(previously_decided_uid).status,
                STATUS_CONFIRMED,
            )
            self.assertEqual(
                registry.candidate_for(projected_uid).status,
                STATUS_CONFIRMED,
            )

    def test_exact_two_runtime_emits_arrival_projection_bound_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root, with_frame_provenance=True)
            uid = "survey_candidate_0003"
            translation_x_m = 0.012902
            translation_y_m = -0.047650
            canonical = fixture.snapshot.candidate_for(uid)
            assert canonical is not None
            arrived_x_m = canonical.geometry.x_m + translation_x_m
            arrived_y_m = canonical.geometry.y_m + translation_y_m
            robot_pose = Pose2D(arrived_x_m - 0.70, arrived_y_m, 0.0)
            planning_frames = iter(
                (
                    CandidatePlanningFrame(
                        current_pose=robot_pose,
                        map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    CandidatePlanningFrame(
                        current_pose=robot_pose,
                        map_from_odom=PlanarTransform2D(
                            translation_x_m,
                            translation_y_m,
                            0.0,
                        ),
                    ),
                )
            )
            committed = []

            def select(_request):
                return CameraCandidateInitialSelection(
                    candidate_uid=uid,
                    prepared_plan=None,
                    evidence={
                        "schema_version": 1,
                        "selected_candidate_uid": uid,
                        "motion_authorized": False,
                    },
                )

            def completed(request):
                return MotionLegOutcome(
                    run_id=request.run_id,
                    status="completed",
                    stop_reason="",
                    stop_details={},
                    motion_published=True,
                    returncode=0,
                    semantic_log_path=(
                        request.session_root / f"{request.run_id}.jsonl"
                    ),
                )

            def capture(request):
                recommendation_path = (
                    request.output_dir / "recommendation.json"
                )
                _write_recommendation(
                    recommendation_path,
                    request.candidate,
                )
                return CandidateObservation(
                    recommendation_path=recommendation_path,
                    qr_id="QR_01",
                    axis_observation_path=None,
                )

            def commit(request):
                committed.append(request)
                raise RuntimeError("stop after projection-bound commit")

            with self.assertRaisesRegex(
                RuntimeError,
                "stop after projection-bound commit",
            ):
                execute_candidate_approach_phase(
                    fixture.config,
                    CandidateApproachEffects(
                        read_current_pose=lambda: robot_pose,
                        admit_planning_frame=lambda _path: next(
                            planning_frames
                        ),
                        select_initial_preapproach=select,
                        plan_preapproach=lambda _request: {
                            "route_csv": "route.csv"
                        },
                        run_motion_leg=completed,
                        capture_observation=capture,
                        validate_facing=lambda request: {
                            "candidate_uid": request.candidate.candidate_uid
                        },
                        commit_decision=commit,
                        clock=lambda: 40.0,
                    ),
                )

            self.assertEqual(len(committed), 1)
            request = committed[0]
            receipt = json.loads(request.receipt_path.read_text())
            self.assertEqual(receipt["schema_version"], 3)
            self.assertEqual(
                Path(receipt["candidate_snapshot_path"]),
                fixture.snapshot_path,
            )
            self.assertEqual(
                Path(receipt["camera_candidate_snapshot_path"]),
                request.camera_candidate_snapshot_path,
            )
            self.assertEqual(
                Path(receipt["candidate_frame_projection_path"]),
                request.candidate_frame_projection_path,
            )
            self.assertIn(
                "arrival_frame_projection",
                str(request.camera_candidate_snapshot_path),
            )
            self.assertNotEqual(
                receipt["candidate_snapshot_sha256"],
                receipt["camera_candidate_snapshot_sha256"],
            )

    def test_schema_v3_rejects_recommendation_using_canonical_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root, with_frame_provenance=True)
            uid = "survey_candidate_0003"
            decision = _write_projected_exact_two_receipt(
                fixture,
                root,
                uid,
                recommendation_uses_canonical_geometry=True,
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    _record(fixture, decision.receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_PROVISIONAL,
            )

    def test_schema_v3_projection_proof_tampering_fails_closed(self):
        def replace_candidate_point(evidence):
            point = evidence["candidate_reprojections"][
                "survey_candidate_0003"
            ]["current_map_point"]
            point["x_m"] += 0.01

        def substitute_cross_candidate_reprojection(evidence):
            evidence["candidate_reprojections"][
                "survey_candidate_0003"
            ] = evidence["candidate_reprojections"][
                "survey_candidate_0004"
            ]

        mutations = {
            "canonical_snapshot_hash": lambda evidence: evidence.__setitem__(
                "source_candidate_snapshot_sha256", "d" * 64
            ),
            "projected_snapshot_hash": lambda evidence: evidence.__setitem__(
                "projected_candidate_snapshot_sha256", "e" * 64
            ),
            "registry_hash": lambda evidence: evidence.__setitem__(
                "source_registry_sha256", "f" * 64
            ),
            "motion_authorized": lambda evidence: evidence.__setitem__(
                "motion_authorized", True
            ),
            "candidate_reprojection": replace_candidate_point,
            "cross_candidate_reprojection": (
                substitute_cross_candidate_reprojection
            ),
        }
        for failure, mutate in mutations.items():
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    fixture = _fixture(root, with_frame_provenance=True)
                    uid = "survey_candidate_0003"
                    decision = _write_projected_exact_two_receipt(
                        fixture,
                        root,
                        uid,
                    )
                    _rewrite_projection_evidence(
                        decision.frame_binding,
                        decision.receipt_path,
                        mutate,
                    )

                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            _record(fixture, decision.receipt_path)

                    registry = load_stand_survey_registry(
                        fixture.survey_root / "stand_registry.json",
                        fixture.plan,
                    )
                    self.assertEqual(
                        registry.candidate_for(uid).status,
                        STATUS_PROVISIONAL,
                    )

    def test_schema_v3_receipt_path_and_hash_mismatches_fail_closed(self):
        mutations = {
            "camera_snapshot_path": lambda receipt, root: receipt.__setitem__(
                "camera_candidate_snapshot_path",
                str(root / "unbound_candidate_snapshot.json"),
            ),
            "camera_snapshot_hash": lambda receipt, _root: receipt.__setitem__(
                "camera_candidate_snapshot_sha256",
                "d" * 64,
            ),
            "projection_path": lambda receipt, root: receipt.__setitem__(
                "candidate_frame_projection_path",
                str(root / "unbound_candidate_frame_projection.json"),
            ),
            "projection_hash": lambda receipt, _root: receipt.__setitem__(
                "candidate_frame_projection_sha256",
                "e" * 64,
            ),
        }
        for failure, mutate in mutations.items():
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    fixture = _fixture(root, with_frame_provenance=True)
                    uid = "survey_candidate_0003"
                    decision = _write_projected_exact_two_receipt(
                        fixture,
                        root,
                        uid,
                    )
                    receipt = json.loads(decision.receipt_path.read_text())
                    mutate(receipt, root)
                    decision.receipt_path.write_text(
                        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
                    )

                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            _record(fixture, decision.receipt_path)

                    registry = load_stand_survey_registry(
                        fixture.survey_root / "stand_registry.json",
                        fixture.plan,
                    )
                    self.assertEqual(
                        registry.candidate_for(uid).status,
                        STATUS_PROVISIONAL,
                    )

    def test_schema_v3_rejects_camera_snapshot_not_bound_by_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root, with_frame_provenance=True)
            uid = "survey_candidate_0003"
            decision = _write_projected_exact_two_receipt(
                fixture,
                root,
                uid,
            )
            original = decision.camera_candidate
            substituted = replace(
                decision.frame_binding.projection.projected_snapshot,
                candidates=tuple(
                    replace(
                        candidate,
                        geometry=replace(
                            candidate.geometry,
                            x_m=candidate.geometry.x_m + 0.01,
                        ),
                    )
                    if candidate.candidate_uid == uid
                    else candidate
                    for candidate in (
                        decision.frame_binding.projection.projected_snapshot.candidates
                    )
                ),
            )
            unbound_path = root / "unbound_camera_candidate_snapshot.json"
            unbound_sha256 = write_candidate_snapshot(unbound_path, substituted)
            receipt = json.loads(decision.receipt_path.read_text())
            receipt["camera_candidate_snapshot_path"] = str(unbound_path)
            receipt["camera_candidate_snapshot_sha256"] = unbound_sha256
            decision.receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            self.assertNotEqual(
                substituted.candidate_for(uid).geometry,
                original.geometry,
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    _record(fixture, decision.receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_PROVISIONAL,
            )

    def test_schema_v2_projected_recommendation_without_proof_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root, with_frame_provenance=True)
            uid = "survey_candidate_0003"
            frame_binding = _camera_frame_projection(fixture, root)
            camera_candidate = (
                frame_binding.projection.projected_snapshot.candidate_for(uid)
            )
            assert camera_candidate is not None
            recommendation_path = root / "unbound_projected_recommendation.json"
            _write_recommendation(recommendation_path, camera_candidate)
            evidence = require_handoff_candidate_support(fixture.handoff, uid)
            assert evidence.support_class is not None
            payload = build_camera_candidate_decision_receipt(
                config=fixture.config,
                candidate=camera_candidate,
                recommendation_path=recommendation_path,
                exact_two_support_by_uid={uid: evidence.support_class},
            )
            receipt_path = root / "unbound_projected_decision.json"
            receipt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    _record(fixture, receipt_path)

            registry = load_stand_survey_registry(
                fixture.survey_root / "stand_registry.json",
                fixture.plan,
            )
            self.assertEqual(
                registry.candidate_for(uid).status,
                STATUS_PROVISIONAL,
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

    def test_selected_decision_rejects_unsealed_or_mutated_registry_extra(self):
        for failure in ("unsealed_extra", "mutated_sealed_extra"):
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    fixture = _fixture(
                        root,
                        with_boundary_audit_only_extra=(
                            failure == "mutated_sealed_extra"
                        ),
                    )
                    uid = "survey_candidate_0003"
                    receipt_path = _write_exact_two_receipt(
                        fixture,
                        root,
                        uid,
                    )
                    extra = (
                        fixture.registry.candidate_for(
                            "survey_candidate_0006"
                        )
                        if failure == "mutated_sealed_extra"
                        else replace(
                            fixture.registry.candidates[-1],
                            candidate_uid="survey_candidate_0006",
                            x_m=1.50,
                            y_m=0.60,
                            first_seen_sec=16.0,
                            last_seen_sec=17.0,
                            source_observation_ids=("observation_0006",),
                            static_map_disposition=(
                                STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
                            ),
                        )
                    )
                    assert extra is not None
                    if failure == "mutated_sealed_extra":
                        extra = replace(extra, x_m=extra.x_m + 0.01)
                        changed_candidates = tuple(
                            extra
                            if candidate.candidate_uid
                            == extra.candidate_uid
                            else candidate
                            for candidate in fixture.registry.candidates
                        )
                    else:
                        changed_candidates = (
                            *fixture.registry.candidates,
                            extra,
                        )
                    changed_registry = replace(
                        fixture.registry,
                        candidates=changed_candidates,
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
                        fixture.survey_root / "stand_registry.json",
                        fixture.plan,
                    )
                    self.assertEqual(
                        registry.candidate_for(uid).status,
                        STATUS_PROVISIONAL,
                    )
                    self.assertFalse(
                        (
                            fixture.survey_root
                            / "decisions"
                            / f"{uid}.json"
                        ).exists()
                    )

    def test_projected_binding_rejects_mutated_sealed_boundary_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(
                root,
                with_boundary_audit_only_extra=True,
                with_frame_provenance=True,
            )
            uid = "survey_candidate_0003"
            decision = _write_projected_exact_two_receipt(
                fixture,
                root,
                uid,
            )
            receipt = json.loads(decision.receipt_path.read_text())
            excluded = fixture.registry.candidate_for(
                "survey_candidate_0006"
            )
            assert excluded is not None
            changed_registry = replace(
                fixture.registry,
                candidates=tuple(
                    replace(candidate, x_m=candidate.x_m + 0.01)
                    if candidate.candidate_uid == excluded.candidate_uid
                    else candidate
                    for candidate in fixture.registry.candidates
                ),
            )

            with self.assertRaises(ValueError):
                require_projected_camera_candidate_binding(
                    receipt,
                    canonical_snapshot_path=fixture.snapshot_path,
                    canonical_snapshot=fixture.snapshot,
                    handoff=fixture.handoff,
                    registry=changed_registry,
                    camera_snapshot_path=(
                        decision.frame_binding.projected_snapshot_path
                    ),
                    projection_path=decision.frame_binding.evidence_path,
                    candidate_uid=uid,
                )

    def test_population_binding_rejects_unsealed_lifecycle_transitions(self):
        mutations = (
            (
                "selected_confirmed_without_receipt",
                "survey_candidate_0001",
                STATUS_CONFIRMED,
                None,
            ),
            (
                "selected_camera_rejected_without_receipt",
                "survey_candidate_0001",
                STATUS_REJECTED,
                REJECTION_BASIS_CAMERA,
            ),
            (
                "selected_rejected_without_basis",
                "survey_candidate_0001",
                STATUS_REJECTED,
                None,
            ),
            (
                "selected_rejected_for_non_camera_reason",
                "survey_candidate_0001",
                STATUS_REJECTED,
                REJECTION_BASIS_NEGATIVE_VISIBILITY,
            ),
            (
                "excluded_candidate_confirmed",
                "survey_candidate_0006",
                STATUS_CONFIRMED,
                None,
            ),
        )
        for failure, uid, status, rejection_basis in mutations:
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    fixture = _fixture(
                        Path(tmp),
                        with_boundary_audit_only_extra=True,
                    )
                    changed_registry = replace(
                        fixture.registry,
                        candidates=tuple(
                            replace(
                                candidate,
                                status=status,
                                rejection_basis=rejection_basis,
                            )
                            if candidate.candidate_uid == uid
                            else candidate
                            for candidate in fixture.registry.candidates
                        ),
                    )

                    with self.assertRaises(ValueError):
                        validate_live_exact_two_camera_population_binding(
                            fixture.handoff,
                            fixture.snapshot,
                            changed_registry,
                            candidate_snapshot_path=fixture.snapshot_path,
                        )

    def test_recorder_rejects_selected_lifecycle_edit_without_canonical_receipt(
        self,
    ):
        transitions = (
            (STATUS_CONFIRMED, None),
            (STATUS_REJECTED, REJECTION_BASIS_CAMERA),
        )
        for status, rejection_basis in transitions:
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    fixture = _fixture(
                        root,
                        with_boundary_audit_only_extra=True,
                    )
                    changed_uid = "survey_candidate_0001"
                    current_uid = "survey_candidate_0003"
                    current_receipt = _write_exact_two_receipt(
                        fixture,
                        root,
                        current_uid,
                    )
                    changed_registry = replace(
                        fixture.registry,
                        candidates=tuple(
                            replace(
                                candidate,
                                status=status,
                                rejection_basis=rejection_basis,
                            )
                            if candidate.candidate_uid == changed_uid
                            else candidate
                            for candidate in fixture.registry.candidates
                        ),
                    )
                    write_stand_survey_registry(
                        fixture.survey_root / "stand_registry.json",
                        changed_registry,
                        fixture.plan,
                    )

                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            _record(fixture, current_receipt)

                    registry = load_stand_survey_registry(
                        fixture.survey_root / "stand_registry.json",
                        fixture.plan,
                    )
                    self.assertEqual(
                        registry.candidate_for(changed_uid).status,
                        status,
                    )
                    self.assertEqual(
                        registry.candidate_for(current_uid).status,
                        STATUS_PROVISIONAL,
                    )
                    self.assertFalse(
                        (fixture.survey_root / "decisions").exists()
                    )

    def test_recorder_rejects_mismatched_or_unauthenticated_prior_receipt(
        self,
    ):
        mutations = {
            "lifecycle_status_mismatch": lambda receipt: receipt.__setitem__(
                "decision", STATUS_REJECTED
            ),
            "camera_evidence_hash_mismatch": lambda receipt: receipt.__setitem__(
                "camera_recommendation_sha256", "d" * 64
            ),
        }
        for failure, mutate in mutations.items():
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    fixture = _fixture(
                        root,
                        with_boundary_audit_only_extra=True,
                    )
                    prior_uid = "survey_candidate_0001"
                    current_uid = "survey_candidate_0003"
                    prior_receipt = _write_exact_two_receipt(
                        fixture,
                        root,
                        prior_uid,
                    )
                    current_receipt = _write_exact_two_receipt(
                        fixture,
                        root,
                        current_uid,
                    )
                    with redirect_stdout(StringIO()):
                        self.assertEqual(_record(fixture, prior_receipt), 0)

                    canonical_path = (
                        fixture.survey_root
                        / "decisions"
                        / f"{prior_uid}.json"
                    )
                    canonical = json.loads(canonical_path.read_text())
                    mutate(canonical)
                    canonical_path.write_text(
                        json.dumps(canonical, indent=2, sort_keys=True) + "\n"
                    )

                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            _record(fixture, current_receipt)

                    registry = load_stand_survey_registry(
                        fixture.survey_root / "stand_registry.json",
                        fixture.plan,
                    )
                    self.assertEqual(
                        registry.candidate_for(prior_uid).status,
                        STATUS_CONFIRMED,
                    )
                    self.assertEqual(
                        registry.candidate_for(current_uid).status,
                        STATUS_PROVISIONAL,
                    )
                    self.assertFalse(
                        (
                            fixture.survey_root
                            / "decisions"
                            / f"{current_uid}.json"
                        ).exists()
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

    def test_boundary_candidate_reaches_route_selector_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            uid = "survey_candidate_0005"
            fixture = _fixture(Path(tmp), boundary_uid=uid)
            motion = Mock()
            capture = Mock()
            observed_support = {}

            def inspect_selection(request):
                observed_support.update(request.support_class_by_uid or {})
                raise RuntimeError("stop after boundary route selection")

            with self.assertRaisesRegex(
                RuntimeError,
                "stop after boundary route selection",
            ):
                execute_candidate_approach_phase(
                    fixture.config,
                    CandidateApproachEffects(
                        select_initial_preapproach=inspect_selection,
                        read_current_pose=Mock(
                            return_value=Pose2D(0.0, 0.0, 0.0)
                        ),
                        run_motion_leg=motion,
                        capture_observation=capture,
                    ),
                )

            evidence = fixture.handoff.admission_decision.candidate_for(uid)
            self.assertIsNotNone(evidence)
            self.assertEqual(
                evidence.static_map_disposition,
                STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
            )
            self.assertEqual(len(observed_support), 5)
            self.assertEqual(
                observed_support[uid],
                SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
            )
            self.assertFalse(fixture.handoff.motion_authorized)
            motion.assert_not_called()
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
